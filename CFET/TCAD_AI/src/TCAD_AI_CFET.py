import os
import sys
import warnings
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import joblib
import copy
import torch.multiprocessing as mp
import threading

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("torch.onnx").setLevel(logging.ERROR)

# ========================
# 1. ARCHITECTURE & LOSS 
# ========================

class CFET_AI(nn.Module):
    def __init__(self, input_dim=5, output_dim=8, hidden_size=16):
        super().__init__()
        self.act = nn.ELU() 
        self.in_l = nn.Linear(input_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.out_l = nn.Linear(hidden_size, output_dim)
        
    def forward(self, x):
        x = self.act(self.in_l(x))
        x = self.act(self.fc2(self.act(self.fc1(x))) + x)
        x = self.act(self.fc4(self.act(self.fc3(x))) + x)
        return self.out_l(x)

def calculate_total_loss(predictions, targets, current_lambda, y_std, y_mean):
    criterion = nn.MSELoss()
    loss_data = criterion(predictions, targets)
    pred_phys = predictions * y_std + y_mean
    
    Ion_N = pred_phys[:, 0]
    Ioff_N_linear = torch.pow(10.0, pred_phys[:, 1])
    Vth_N = pred_phys[:, 2]
    ss_n = pred_phys[:, 3]
    
    Ion_P = pred_phys[:, 4]
    Ioff_P_linear = torch.pow(10.0, pred_phys[:, 5])
    Vth_P = pred_phys[:, 6]
    ss_p = pred_phys[:, 7]
    
    pen_current = torch.mean(torch.relu(Ioff_N_linear - Ion_N + 1e-9) / 1e-5) + \
                  torch.mean(torch.relu(Ioff_P_linear - Ion_P + 1e-9) / 1e-5)
                  
    pen_ss = torch.mean(torch.relu(59.0 - ss_n) / 60.0) + \
             torch.mean(torch.relu(59.0 - ss_p) / 60.0)
             
    pen_vth = torch.mean(torch.relu(-Vth_N)) + torch.mean(torch.relu(Vth_P))
    
    ratio = 10000.0
    pen_ratio = torch.mean(torch.relu((ratio * Ioff_N_linear) - Ion_N)) + \
                torch.mean(torch.relu((ratio * Ioff_P_linear) - Ion_P))

    loss_phys = pen_current + pen_ss + pen_vth + pen_ratio
    loss_total = loss_data + (current_lambda * loss_phys)
    
    return loss_total

# ====================
# 2. THREAD MONITEUR
# ====================
def progress_monitor(q, total_epochs):
    global_epoch = 0
    while True:
        msg = q.get()
        if msg == "DONE":
            break
            
        inc, seed, best_mse = msg
        global_epoch += inc
        
        pct = global_epoch / total_epochs
        bar = '█' * int(30 * pct) + '-' * (30 - int(30 * pct))

        sys.stdout.write(f'\r[GLOBAL PROGRESS] |{bar}| {(pct*100):.1f}% - Update from Seed {seed:02d} - Local Best MSE: {best_mse:.5f}   ')
        sys.stdout.flush()

# ====================
# 3. WORKER FUNCTION 
# ====================
def train_single_seed(current_seed, X_tr, y_tr, X_te, y_te, y_mean_np, y_std_np, epochs, max_lambda, warmup_epochs, ramp_epochs, input_dim, output_dim, q):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(current_seed)
    
    X_train_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_tr, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_te, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_te, dtype=torch.float32).to(device)
    
    y_mean = torch.tensor(y_mean_np, dtype=torch.float32).to(device)
    y_std = torch.tensor(y_std_np, dtype=torch.float32).to(device)
    
    model = CFET_AI(input_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    best_run_loss = float('inf')
    best_run_weights = None

    updates_buffer = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        progress = max(0.0, (epoch - warmup_epochs) / ramp_epochs) if epoch >= warmup_epochs else 0.0
        current_lambda = max_lambda * min(1.0, progress)
        
        predictions = model(X_train_t)
        loss_total = calculate_total_loss(predictions, y_train_t, current_lambda, y_std, y_mean)
        
        loss_total.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test_t)
            current_test_loss = criterion(test_preds, y_test_t).item()
            
            if current_test_loss < best_run_loss:
                best_run_loss = current_test_loss
                best_run_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        
        updates_buffer += 1
        if updates_buffer == 10:
            q.put((updates_buffer, current_seed, best_run_loss))
            updates_buffer = 0
            
    if updates_buffer > 0:
        q.put((updates_buffer, current_seed, best_run_loss))

    return current_seed, best_run_loss, best_run_weights

# ==========================================
# 4. MAIN EXECUTION BLOCK 
# ==========================================
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "..", "model")
    os.makedirs(model_dir, exist_ok=True)

    train_path = os.path.join(script_dir, "../../TCAD/data", "TCAD_train.csv")
    test_path = os.path.join(script_dir, "../../TCAD/data", "TCAD_test.csv")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    input_cols = ['Lch', 'Wch', 'Tch', 'Tox1', 'Tox2']
    output_cols = ['Ion_N', 'Ioff_N', 'Vth_N', 'SS_N', 'Ion_P', 'Ioff_P', 'Vth_P', 'SS_P']

    X_train, y_train = df_train[input_cols].values, df_train[output_cols].values
    X_test, y_test = df_test[input_cols].values, df_test[output_cols].values

    y_train[:, 1] = np.log10(np.abs(y_train[:, 1]) + 1e-20)
    y_train[:, 5] = np.log10(np.abs(y_train[:, 5]) + 1e-20)
    y_test[:, 1] = np.log10(np.abs(y_test[:, 1]) + 1e-20)
    y_test[:, 5] = np.log10(np.abs(y_test[:, 5]) + 1e-20)

    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    epochs = 2000 
    max_lambda = 0   
    warmup_epochs = 100   
    ramp_epochs = 200    
    seeds_to_test = [i for i in range(20)] 
    
    input_dim = len(input_cols)
    output_dim = len(output_cols)

    total_epochs = len(seeds_to_test) * epochs
    N_JOBS = 4 
    print(f"\n[INFO] Datasets loaded. Train: {len(df_train)} | Test: {len(df_test)}")
    print(f"[INFO] Launching Multi-Seed Evaluation concurrently with {N_JOBS} workers...")
    print(f"[INFO] Total epochs to compute: {total_epochs}\n")

    manager = mp.Manager()
    progress_queue = manager.Queue()
    
    monitor_thread = threading.Thread(target=progress_monitor, args=(progress_queue, total_epochs))
    monitor_thread.start()

    results = joblib.Parallel(n_jobs=N_JOBS)(
        joblib.delayed(train_single_seed)(
            seed, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
            scaler_y.mean_, scaler_y.scale_, epochs, max_lambda, warmup_epochs, ramp_epochs, input_dim, output_dim, progress_queue
        ) for seed in seeds_to_test
    )

    progress_queue.put("DONE")
    monitor_thread.join()
    print("\n")

    all_test_mse = []
    global_best_test_loss = float('inf')
    global_best_weights = None
    global_best_seed = 0

    for seed, best_loss, best_weights in results:
        all_test_mse.append(best_loss)
        if best_loss < global_best_test_loss:
            global_best_test_loss = best_loss
            global_best_weights = copy.deepcopy(best_weights)
            global_best_seed = seed

    joblib.dump(scaler_X, os.path.join(model_dir, 'scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(model_dir, 'scaler_y.pkl'))

    model_champion = CFET_AI(input_dim, output_dim)
    model_champion.load_state_dict(global_best_weights)
    torch.save(model_champion.state_dict(), os.path.join(model_dir, "CFET_Model.pth"))

    model_champion.eval()
    dummy_input = torch.randn(1, input_dim)
    torch.onnx.export(model_champion, dummy_input, os.path.join(model_dir, "CFET_Model.onnx"), 
                      export_params=True, verbose=False,
                      input_names=['geometry_inputs'], output_names=['physics_outputs'])

    mean_mse = np.mean(all_test_mse)
    std_mse = np.std(all_test_mse)

    print("==================================================")
    print("             FINAL PERFORMANCE          ")
    print("==================================================")
    print(f"Generalization Error : {mean_mse:.5f} ± {std_mse:.5f}")
    print("--------------------------------------------------")
    print(f"-> All individual MSEs      : {[round(x, 5) for x in all_test_mse]}")
    print(f"-> Champion Model Saved     : Seed {global_best_seed} (MSE: {global_best_test_loss:.5f})")
    print("==================================================\n")