import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import itertools
import time
import sys
import os
import matplotlib.pyplot as plt
import warnings
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

script_dir = os.path.dirname(os.path.abspath(__file__))
figure_dir = os.path.join(script_dir, "..", "figure")
model_dir = os.path.join(script_dir, "..", "model")

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("torch.onnx").setLevel(logging.ERROR)

# ==========================================
# 1. ARCHITECTURES & LOSS
# ==========================================
class CFET_MLP(nn.Module):
    def __init__(self, in_d, out_d, h, act):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_d, h), act, nn.Linear(h, h*2), act, nn.Linear(h*2, h), act, nn.Linear(h, out_d))
    def forward(self, x): return self.net(x)

class CFET_DeepMLP(nn.Module):
    def __init__(self, in_d, out_d, h, act):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_d, h), act, nn.Linear(h, h*2), act, nn.Linear(h*2, h*2), act, nn.Linear(h*2, h), act, nn.Linear(h, h), act, nn.Linear(h, out_d))
    def forward(self, x): return self.net(x)

class CFET_ResNet(nn.Module):
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

class CFET_DeepResNet(nn.Module):
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
    return nn.ReLU()

# ==========================================
# 2. WORKER FUNCTION (PARALLEL EXECUTION)
# ==========================================
def train_worker(task_info, data_arrays, y_scales):

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
    max_lambda, warmup_epochs, ramp_epochs, lr, wd = task_info['params']
    epochs = task_info['epochs']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_train_t = torch.tensor(data_arrays['X_train'], dtype=torch.float32).to(device)
    y_train_t = torch.tensor(data_arrays['y_train'], dtype=torch.float32).to(device)
    X_test_t = torch.tensor(data_arrays['X_test'], dtype=torch.float32).to(device)
    y_test_t = torch.tensor(data_arrays['y_test'], dtype=torch.float32).to(device)
    
    y_mean = torch.tensor(y_scales['mean'], dtype=torch.float32).to(device)
    y_std = torch.tensor(y_scales['std'], dtype=torch.float32).to(device)
    
    criterion = nn.MSELoss()
    act_f = get_activation(act_name)
    
    if a_name == 'MLP': m = CFET_MLP(X_train_t.shape[1], y_train_t.shape[1], h_s, act_f).to(device)
    elif a_name == 'DeepMLP': m = CFET_DeepMLP(X_train_t.shape[1], y_train_t.shape[1], h_s, act_f).to(device)
    elif a_name == 'ResNet': m = CFET_ResNet(X_train_t.shape[1], y_train_t.shape[1], h_s, act_f).to(device)
    else: m = CFET_DeepResNet(X_train_t.shape[1], y_train_t.shape[1], h_s, act_f).to(device)
        
    opt = optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    
    history_epochs, history_test_loss = [], []
    b_l, b_e = float('inf'), 0
    patience = 200
    epochs_no_improve = 0
    
    for ep in range(epochs):
        m.train()
        opt.zero_grad()
        prog = max(0.0, (ep - warmup_epochs) / ramp_epochs) if ep >= warmup_epochs else 0.0
        c_l = max_lambda * min(1.0, prog)
        
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
                    b_l = t_l
                    b_e = ep + 1
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 10
            
            if epochs_no_improve >= patience:
                break
    
    return {
        'Archi_Name': f"{a_name} ({act_name}/{h_s})",
        'Lambda': max_lambda, 'Warmup': warmup_epochs, 'Ramp': ramp_epochs, 'LR': lr, 'WD': wd,
        'best_loss': b_l, 'best_epoch': b_e,
        'history_epochs': history_epochs, 'history_test_loss': history_test_loss
    }

# ==========================================
# 3. MAIN EXECUTION & TIME ESTIMATION
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

    epochs = 5000
    lambdas_to_test = [0, 0.2, 0.5, 2.0]
    warmups_to_test = [50, 100, 150]
    ramps_to_test = [200, 600, 1000]
    learning_rates = [1e-3, 5e-4]
    weight_decays = [1e-4, 1e-5]

    param_grid = list(itertools.product(lambdas_to_test, warmups_to_test, ramps_to_test, learning_rates, weight_decays))
    num_params = len(param_grid)
    
    time_per_run_mins = 5.1861 / 60.0

    count_all = len(pd.read_csv(os.path.join(model_dir, "All_Architectures_Log.csv"))) if os.path.exists(os.path.join(model_dir, "All_Architectures_Log.csv")) else 0
    count_div = len(pd.read_csv(os.path.join(model_dir, "Top_Diverse_Architectures.csv"))) if os.path.exists(os.path.join(model_dir, "Top_Diverse_Architectures.csv")) else 0

    est_time_1 = (min(3, count_all) * num_params) * time_per_run_mins if count_all > 0 else 0.0
    est_time_2 = (count_div * num_params) * time_per_run_mins if count_div > 0 else 0.0
    est_time_3 = (count_all * num_params) * time_per_run_mins if count_all > 0 else 0.0

    print("\n[INFO] Select Architecture Testing Mode:")
    print(f" 1. Top 3 Best Architectures       (Est. duration: ~{est_time_1:.1f} min)")
    print(f" 2. Top Diverse Architectures      (Est. duration: ~{est_time_2:.1f} min)")
    print(f" 3. ALL Architectures              (Est. duration: ~{est_time_3:.1f} min)")
    choice = input("Enter your choice (1, 2, or 3): ").strip()

    if choice == '1':
        if count_all == 0: sys.exit("[ERROR] 'All_Architectures_Log.csv' not found. Run NAS script first.")
        df_archis = pd.read_csv(os.path.join(model_dir, "All_Architectures_Log.csv")).sort_values(by='Loss').head(3)
    elif choice == '2':
        if count_div == 0: sys.exit("[ERROR] 'Top_Diverse_Architectures.csv' not found. Run NAS script first.")
        df_archis = pd.read_csv(os.path.join(model_dir, "Top_Diverse_Architectures.csv"))
    elif choice == '3':
        if count_all == 0: sys.exit("[ERROR] 'All_Architectures_Log.csv' not found. Run NAS script first.")
        df_archis = pd.read_csv(os.path.join(model_dir, "All_Architectures_Log.csv"))
    else:
        sys.exit("[ERROR] Invalid choice. Execution aborted.")

    architectures_to_test = df_archis.to_dict('records')
    
    tasks = []
    for archi in architectures_to_test:
        for params in param_grid:
            tasks.append({
                'Archi': archi['Archi'], 'Act': archi['Act'], 'Size': archi['Size'],
                'params': params, 'epochs': epochs
            })
            
    total_runs = len(tasks)
    max_workers = 4 
    
    print(f"\n[INFO] Launching Parallel Tuning with {max_workers} concurrent workers.")
    print(f"[INFO] Total configurations to process: {total_runs}")
    print(f"[INFO] Hardware: {device_display} | Estimated duration: ~{(total_runs * time_per_run_mins):.2f} min\n")

    results = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(train_worker, task, data_arrays, y_scales): task for task in tasks}
        
        for idx, future in enumerate(as_completed(futures)):
            results.append(future.result())
            
            pct = (idx + 1) / total_runs
            bar = '█' * int(30 * pct) + '-' * (30 - int(30 * pct))
            sys.stdout.write(f'\r[PROGRESS] |{bar}| {(pct*100):.1f}% - Min Loss Global: {min([r["best_loss"] for r in results]):.4f}')
            sys.stdout.flush()

    results.sort(key=lambda x: x['best_loss'])

    print(f"\n\n[SUCCESS] Tuning completed in {(time.time() - start_time)/60:.2f} minutes.")

    with open(os.path.join(model_dir, "Top_Tuned_Configurations.txt"), "w") as f:
        for r in results:
            line = f"{r['Archi_Name']} | L:{r['Lambda']} W:{r['Warmup']} R:{r['Ramp']} LR:{r['LR']} WD:{r['WD']} --> Test Loss: {r['best_loss']:.5f} (Epoch {r['best_epoch']})\n"
            f.write(line)
            
    print(f"[INFO] All {len(results)} results saved successfully to 'Top_Tuned_Configurations.txt'")
    
    print("\n[RESULTS] TOP 3 CONFIGURATIONS:")
    for i in range(3):
        r = results[i]
        print(f" {i+1}. {r['Archi_Name']} | L:{r['Lambda']} W:{r['Warmup']} R:{r['Ramp']} LR:{r['LR']} WD:{r['WD']} --> Test Loss: {r['best_loss']:.5f} (Epoch {r['best_epoch']})")

    plt.figure(figsize=(12, 7))
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    for i in range(min(5, len(results))):
        r = results[i]
        label = f"{r['Archi_Name']} [L:{r['Lambda']} LR:{r['LR']} WD:{r['WD']}] | Err: {r['best_loss']:.5f}"
        plt.plot(r['history_epochs'], r['history_test_loss'], label=label, color=colors[i], linewidth=2, alpha=0.8)
        plt.scatter([r['best_epoch']], [r['best_loss']], color=colors[i], s=50, zorder=5)

    plt.title('Global Model Comparison (Hyperparameter Tuning)', fontsize=16)
    plt.xlabel('Epochs')
    plt.ylabel('Test Error (MSE)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "Learning_Curves_Hyperparameter_Tuning.png"), dpi=300, bbox_inches='tight')
    plt.show()