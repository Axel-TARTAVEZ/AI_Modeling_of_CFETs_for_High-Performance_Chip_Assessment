import os
import warnings
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import itertools
import time
import sys
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("torch.onnx").setLevel(logging.ERROR)

script_dir = os.path.dirname(os.path.abspath(__file__))
figure_dir = os.path.join(script_dir, "..", "figure")
model_dir = os.path.join(script_dir, "..", "model")

# ==========================================
# 1. ARCHITECTURES (Renommées pour 2DFET)
# ==========================================
class FET2D_MLP(nn.Module):
    def __init__(self, in_d, out_d, h, act):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_d, h), act, nn.Linear(h, h*2), act, nn.Linear(h*2, h), act, nn.Linear(h, out_d))
    def forward(self, x): return self.net(x)

class FET2D_OriginalMLP(nn.Module):
    def __init__(self, in_d, out_d, h, act):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_d, h), act, nn.Linear(h, h*4), act, nn.Linear(h*4, h), act, nn.Linear(h, out_d))
    def forward(self, x): return self.net(x)

class FET2D_DeepMLP(nn.Module):
    def __init__(self, in_d, out_d, h, act):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_d, h), act, nn.Linear(h, h*2), act, nn.Linear(h*2, h*2), act, nn.Linear(h*2, h), act, nn.Linear(h, h), act, nn.Linear(h, out_d))
    def forward(self, x): return self.net(x)

class FET2D_ResNet(nn.Module):
    def __init__(self, in_d, out_d, h, act):
        super().__init__()
        self.act = act
        self.in_l = nn.Linear(in_d, h)
        self.fc1, self.fc2 = nn.Linear(h, h), nn.Linear(h, h)
        self.fc3, self.fc4 = nn.Linear(h, h), nn.Linear(h, h)
        self.out_l = nn.Linear(h, out_d)
    def forward(self, x):
        x = self.act(self.in_l(x))
        x = self.act(self.fc2(self.act(self.fc1(x))) + x)
        x = self.act(self.fc4(self.act(self.fc3(x))) + x)
        return self.out_l(x)

class FET2D_DeepResNet(nn.Module):
    def __init__(self, in_d, out_d, h, act):
        super().__init__()
        self.act = act
        self.in_l = nn.Linear(in_d, h)
        self.fc1, self.fc2 = nn.Linear(h, h), nn.Linear(h, h)
        self.fc3, self.fc4 = nn.Linear(h, h), nn.Linear(h, h)
        self.fc5, self.fc6 = nn.Linear(h, h), nn.Linear(h, h)
        self.fc7, self.fc8 = nn.Linear(h, h), nn.Linear(h, h)
        self.out_l = nn.Linear(h, out_d)
    def forward(self, x):
        x = self.act(self.in_l(x))
        x = self.act(self.fc2(self.act(self.fc1(x))) + x)
        x = self.act(self.fc4(self.act(self.fc3(x))) + x)
        x = self.act(self.fc6(self.act(self.fc5(x))) + x)
        x = self.act(self.fc8(self.act(self.fc7(x))) + x)
        return self.out_l(x)

def get_activation(act_name):
    if act_name == 'SiLU': return nn.SiLU()
    if act_name == 'GELU': return nn.GELU()
    if act_name == 'Mish': return nn.Mish()
    if act_name == 'LReLU': return nn.LeakyReLU()
    if act_name == 'ELU': return nn.ELU()
    if act_name == 'ReLU': return nn.ReLU()
    return nn.ReLU()

# ==========================================
# 2. WORKER FUNCTION
# ==========================================
def nas_worker(task_info, data_arrays, y_scales):

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    a_name = task_info['Archi']
    act_name = task_info['Act']
    h_s = int(task_info['Size'])
    
    f_l = task_info['f_l']
    f_w = task_info['f_w']
    f_r = task_info['f_r']
    eps = task_info['eps']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_train_t = torch.tensor(data_arrays['X_train'], dtype=torch.float32).to(device)
    y_train_t = torch.tensor(data_arrays['y_train'], dtype=torch.float32).to(device)
    X_test_t = torch.tensor(data_arrays['X_test'], dtype=torch.float32).to(device)
    y_test_t = torch.tensor(data_arrays['y_test'], dtype=torch.float32).to(device)
    
    y_mean = torch.tensor(y_scales['mean'], dtype=torch.float32).to(device)
    y_std = torch.tensor(y_scales['std'], dtype=torch.float32).to(device)
    
    criterion = nn.MSELoss()
    act_f = get_activation(act_name)
    
    if a_name == 'MLP': m = FET2D_MLP(X_train_t.shape[1], y_train_t.shape[1], h_s, act_f).to(device)
    elif a_name == 'OriginalMLP': m = FET2D_OriginalMLP(X_train_t.shape[1], y_train_t.shape[1], h_s, act_f).to(device)
    elif a_name == 'DeepMLP': m = FET2D_DeepMLP(X_train_t.shape[1], y_train_t.shape[1], h_s, act_f).to(device)
    elif a_name == 'ResNet': m = FET2D_ResNet(X_train_t.shape[1], y_train_t.shape[1], h_s, act_f).to(device)
    else: m = FET2D_DeepResNet(X_train_t.shape[1], y_train_t.shape[1], h_s, act_f).to(device)
        
    opt = optim.Adam(m.parameters(), lr=0.001, weight_decay=1e-4)
    
    history_epochs, history_test_loss = [], []
    b_l, b_e = float('inf'), 0
    patience = 100
    epochs_no_improve = 0
    
    for ep in range(eps):
        m.train()
        opt.zero_grad()
        prog = max(0.0, (ep - f_w) / f_r) if ep >= f_w else 0.0
        c_l = f_l * min(1.0, prog)
        
        preds = m(X_train_t)
        
        l_d = criterion(preds, y_train_t)
        p_p = preds * y_std + y_mean
        
        i_n, io_n, vth_n, s_n = p_p[:,0], torch.pow(10.0, p_p[:,1]), p_p[:,2], p_p[:,3]
        i_p, io_p, vth_p, s_p = p_p[:,4], torch.pow(10.0, p_p[:,5]), p_p[:,6], p_p[:,7]
        
        pen_c = torch.mean(torch.relu(io_n - i_n + 1e-9)/1e-5) + torch.mean(torch.relu(io_p - i_p + 1e-9)/1e-5)
        
        pen_s = torch.mean(torch.relu(59.0 - s_n)/60.0) + torch.mean(torch.relu(59.0 - s_p)/60.0)
        
        pen_vth = torch.mean(torch.relu(-vth_n)) + torch.mean(torch.relu(vth_p))

        ratio = 10000.0
        pen_ratio = torch.mean(torch.relu((ratio * io_n) - i_n)) + torch.mean(torch.relu((ratio * io_p) - i_p))
        
        loss = l_d + (c_l * (pen_c + pen_s + pen_vth + pen_ratio))
        
        loss.backward()
        opt.step()
        
        if (ep + 1) % 10 == 0:
            m.eval()
            with torch.no_grad():
                t_l = criterion(m(X_test_t), y_test_t).item()
                history_epochs.append(ep + 1)
                history_test_loss.append(t_l)
                
                if t_l < b_l:
                    b_l, b_e = t_l, ep + 1
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 10
            
            if epochs_no_improve >= patience:
                break

    return {
        'Archi': a_name, 'Act': act_name, 'Size': h_s, 'Loss': b_l, 'Epoch': b_e,
        'history_epochs': history_epochs, 'history_test_loss': history_test_loss
    }

# ==========================================
# 3. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == '__main__':
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    device_display = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_path = os.path.join(script_dir, "../../TCAD/data", "TCAD_train.csv")
    test_path = os.path.join(script_dir, "../../TCAD/data", "TCAD_test.csv")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    input_cols = ['Lch', 'Wch', 'Tox1', 'Tox2']
    output_cols = ['Ion_N', 'Ioff_N', 'Vth_N', 'SS_N', 'Ion_P', 'Ioff_P', 'Vth_P', 'SS_P']

    X_train = df_train[input_cols].values
    y_train = df_train[output_cols].values

    X_test = df_test[input_cols].values
    y_test = df_test[output_cols].values

    y_train[:, 1] = np.log10(np.abs(y_train[:, 1]) + 1e-20)
    y_train[:, 5] = np.log10(np.abs(y_train[:, 5]) + 1e-20)

    y_test[:, 1] = np.log10(np.abs(y_test[:, 1]) + 1e-20)
    y_test[:, 5] = np.log10(np.abs(y_test[:, 5]) + 1e-20)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    data_arrays = {
        'X_train': X_train_scaled, 'X_test': X_test_scaled,
        'y_train': y_train_scaled, 'y_test': y_test_scaled
    }
    y_scales = {'mean': scaler_y.mean_, 'std': scaler_y.scale_}

    archis = ['MLP', 'OriginalMLP', 'DeepMLP', 'ResNet', 'DeepResNet']
    acts = [
    ('SiLU', nn.SiLU()), 
    ('GELU', nn.GELU()), 
    ('Mish', nn.Mish()), 
    ('ELU', nn.ELU()), 
    ('SELU', nn.SELU()),            
    ('Softplus', nn.Softplus())     
    ]
    sizes = [16, 32, 64, 128]

    grid = list(itertools.product(archis, acts, sizes))
    tot = len(grid)

    f_l, f_w, f_r, eps = 0.5, 100, 800, 3500
    time_per_run_mins = 3.2 / 60.0

    print(f"\n[INFO] Initiating Neural Architecture Search (NAS) - {tot} configurations")
    print(f"[INFO] Hardware: {device_display} | Estimated duration: ~{(tot * time_per_run_mins):.2f} min\n")

    tasks = []
    for a_name, (act_n, _), h_s in grid:
        tasks.append({
            'Archi': a_name, 'Act': act_n, 'Size': h_s,
            'f_l': f_l, 'f_w': f_w, 'f_r': f_r, 'eps': eps
        })

    res = []
    st = time.time()
    max_workers = 4

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(nas_worker, task, data_arrays, y_scales): task for task in tasks}
        
        for idx, future in enumerate(as_completed(futures)):
            res.append(future.result())
            
            pct = (idx + 1) / tot
            bar = '█' * int(30 * pct) + '-' * (30 - int(30 * pct))
            sys.stdout.write(f'\r[PROGRESS] |{bar}| {(pct*100):.1f}% (Run {idx+1}/{tot}) - Global Min Loss: {min([r["Loss"] for r in res]):.4f}')
            sys.stdout.flush()

    res.sort(key=lambda x: x['Loss'])

    res_csv = [{'Archi': r['Archi'], 'Act': r['Act'], 'Size': r['Size'], 'Loss': r['Loss'], 'Epoch': r['Epoch']} for r in res]
    df_results = pd.DataFrame(res_csv)
    df_results.to_csv(os.path.join(model_dir, "All_Architectures_Log.csv"), index=False)

    best_diverse_df = df_results.loc[df_results.groupby('Archi')['Loss'].idxmin()].sort_values(by='Loss')
    best_diverse_df.to_csv(os.path.join(model_dir, "Top_Diverse_Architectures.csv"), index=False)

    print(f"\n\n[SUCCESS] NAS completed in {(time.time() - st)/60:.2f} minutes.")
    print("[INFO] Full log saved to 'All_Architectures_Log.csv'")
    print("\n[RESULTS] Top architectures exported for Hyperparameter Tuning:")
    for index, row in best_diverse_df.iterrows():
        print(f" -> {row['Archi']:<12} | {row['Act']:<4} | Size: {row['Size']:<3} | Test Loss: {row['Loss']:.5f} (Epoch {row['Epoch']})")

    # ==========================================
    # GRAPH 1 : TOP 5 OVERALL ARCHITECTURES
    # ==========================================
    plt.figure(figsize=(12, 7))
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    for i in range(min(5, len(res))):
        r = res[i]
        label = f"{r['Archi']} ({r['Act']}/{r['Size']}) [Min: {r['Loss']:.4f}]"
        plt.plot(r['history_epochs'], r['history_test_loss'], label=label, color=colors[i], linewidth=2, alpha=0.8)
        plt.scatter([r['Epoch']], [r['Loss']], color=colors[i], s=50, zorder=5)

    plt.title('NAS Comparison (Top 5 Overall Architectures)')
    plt.xlabel('Epochs')
    plt.ylabel('Test Error (MSE)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "Top_5_Overall_NAS_Curves.png"), dpi=300, bbox_inches='tight')

    # ==========================================
    # GRAPH 2 : TOP 5 DIVERSE (Best of each family)
    # ==========================================
    diverse_best = []
    seen_archis = set()
    for r in res:
        if r['Archi'] not in seen_archis:
            diverse_best.append(r)
            seen_archis.add(r['Archi'])
            if len(diverse_best) == 5:
                break

    plt.figure(figsize=(12, 7))
    for i in range(len(diverse_best)):
        r = diverse_best[i]
        label = f"Best {r['Archi']}: ({r['Act']}/{r['Size']}) [Min: {r['Loss']:.4f}]"
        plt.plot(r['history_epochs'], r['history_test_loss'], label=label, color=colors[i], linewidth=2, alpha=0.8)
        plt.scatter([r['Epoch']], [r['Loss']], color=colors[i], s=50, zorder=5)

    plt.title('NAS Diverse Architectures (Champion of each family)')
    plt.xlabel('Epochs')
    plt.ylabel('Test Error (MSE)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "Top_5_Diverse_NAS_Curves.png"), dpi=300, bbox_inches='tight')

    print("[INFO] Graphs saved: 'Top_5_Overall_NAS_Curves.png' & 'Top_5_Diverse_NAS_Curves.png'")
    
    plt.show()