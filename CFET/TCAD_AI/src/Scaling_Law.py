import os
import warnings
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

script_dir = os.path.dirname(os.path.abspath(__file__))
figure_dir = os.path.join(script_dir, "..", "figure")
os.makedirs(figure_dir, exist_ok=True)

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("torch.onnx").setLevel(logging.ERROR)

import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, dim, act_layer):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            act_layer(),
            nn.Linear(dim, dim)
        )
        self.act = act_layer()
        
    def forward(self, x):
        return self.act(self.block(x) + x)

class CFET_ResNet(nn.Module):
    def __init__(self, input_dim=5, output_dim=8, hidden_size=16, act_layer=nn.ELU):
        super().__init__()
        
        self.proj_in = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            act_layer()
        )
        
        self.res1 = ResBlock(hidden_size, act_layer)
        self.res2 = ResBlock(hidden_size, act_layer)
        
        self.proj_out = nn.Linear(hidden_size, output_dim)
        
    def forward(self, x):
        x = self.proj_in(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.proj_out(x)

def calculate_total_loss(predictions, targets, current_lambda, criterion, y_mean, y_std):
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
    return loss_data + (current_lambda * loss_phys)

def scaling_worker(task, data_arrays):
    size = task['size']
    epochs = task['epochs']
    max_lambda = task['max_lambda']
    warmup_epochs = task['warmup_epochs']
    ramp_epochs = task['ramp_epochs']
    seed = task['seed']
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    
    X_train_full = data_arrays['X_train_full']
    y_train_full = data_arrays['y_train_full']
    X_test = data_arrays['X_test']
    y_test = data_arrays['y_test']
    
    indices = np.random.choice(len(X_train_full), size, replace=False)
    X_train_sub = X_train_full[indices]
    y_train_sub = y_train_full[indices]
    
    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_sub)
    y_train_scaled = scaler_y.fit_transform(y_train_sub)
    
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)
    
    y_mean_t = torch.tensor(scaler_y.mean_, dtype=torch.float32).to(device)
    y_std_t = torch.tensor(scaler_y.scale_, dtype=torch.float32).to(device)
    
    model = CFET_ResNet(X_train_t.shape[1], y_train_t.shape[1], hidden_size=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    best_test_mse = float('inf')
    epochs_no_improve = 0
    patience = 200
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        progress = max(0.0, (epoch - warmup_epochs) / ramp_epochs) if epoch >= warmup_epochs else 0.0
        current_lambda = max_lambda * min(1.0, progress)
        
        preds = model(X_train_t)
        loss = calculate_total_loss(preds, y_train_t, current_lambda, criterion, y_mean_t, y_std_t)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_preds = model(X_test_t)
                current_test_mse = criterion(test_preds, y_test_t).item()
                
                if current_test_mse < best_test_mse:
                    best_test_mse = current_test_mse
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 10
            
            if epochs_no_improve >= patience:
                break
                
    return {'size': size, 'test_mse': best_test_mse, 'seed': seed}

def create_task(size, run_id):
    return {
        'size': int(size),
        'epochs': 2000,
        'max_lambda': 0,
        'warmup_epochs': 50,
        'ramp_epochs': 200,
        'seed': 42 + (int(size) * 100) + run_id
    }

if __name__ == '__main__':
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    RUNS_PER_POINT = 10
    
    train_path = os.path.join(script_dir, "../../TCAD/data", "TCAD_train.csv")
    test_path = os.path.join(script_dir, "../../TCAD/data", "TCAD_test.csv")
    results_csv = os.path.join(figure_dir, "scaling_results.csv")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    input_cols = ['Lch', 'Wch', 'Tch', 'Tox1', 'Tox2']
    output_cols = ['Ion_N', 'Ioff_N', 'Vth_N', 'SS_N', 'Ion_P', 'Ioff_P', 'Vth_P', 'SS_P']

    X_train_full = df_train[input_cols].values
    y_train_full = df_train[output_cols].values
    X_test = df_test[input_cols].values
    y_test = df_test[output_cols].values

    y_train_full[:, 1] = np.log10(np.abs(y_train_full[:, 1]) + 1e-20)
    y_train_full[:, 5] = np.log10(np.abs(y_train_full[:, 5]) + 1e-20)
    y_test[:, 1] = np.log10(np.abs(y_test[:, 1]) + 1e-20)
    y_test[:, 5] = np.log10(np.abs(y_test[:, 5]) + 1e-20)

    total_train_samples = len(X_train_full)
    data_arrays = {
        'X_train_full': X_train_full, 'y_train_full': y_train_full,
        'X_test': X_test, 'y_test': y_test
    }

    existing_counts = {}
    if os.path.exists(results_csv):
        df_exist = pd.read_csv(results_csv)
        existing_counts = df_exist['size'].value_counts().to_dict()

    print("\n==================================================")
    print("       SCALING LAW DATASET STATUS                 ")
    print("==================================================")
    if existing_counts:
        num_existing_points = len(existing_counts)
        runs_per_point = list(existing_counts.values())
        print(f"-> Total simulations saved : {sum(runs_per_point)}")
        print(f"-> Unique data points      : {num_existing_points}")
        print(f"-> Runs configuration      : Target is {RUNS_PER_POINT} runs/point")
        incompletes = {k: v for k, v in existing_counts.items() if v < RUNS_PER_POINT}
        if incompletes:
            print(f"-> Incomplete points found : {len(incompletes)} (Will be repaired automatically)")
        else:
            print("-> Dataset integrity       : PERFECT (No missing runs)")
    else:
        print("-> No existing data found. Starting fresh.")
    print("==================================================\n")

    try:
        user_input = int(input("[INPUT] How many NEW points do you want to add? (Enter 0 to just plot): "))
    except ValueError:
        user_input = 0

    tasks = []
    
    for sz, count in existing_counts.items():
        if count < RUNS_PER_POINT:
            for i in range(RUNS_PER_POINT - count):
                tasks.append(create_task(sz, count + i))

    current_sizes = list(existing_counts.keys())
    
    if user_input > 0:
        if not current_sizes:
            new_sizes = np.unique(np.geomspace(50, total_train_samples, num=user_input)).tolist()
            for sz in new_sizes:
                for i in range(RUNS_PER_POINT):
                    tasks.append(create_task(sz, i))
            current_sizes = new_sizes
        else:
            if 50 not in current_sizes: current_sizes.append(50)
            if total_train_samples not in current_sizes: current_sizes.append(total_train_samples)
            
            added_sizes = []
            for _ in range(user_input):
                current_sizes.sort()
                max_gap = 0
                best_new_sz = None
                
                for i in range(len(current_sizes) - 1):
                    sz1, sz2 = current_sizes[i], current_sizes[i+1]
                    if sz2 - sz1 > 1: 
                        ratio = sz2 / sz1
                        candidate = int(np.sqrt(sz1 * sz2))

                        if candidate == sz1 or candidate == sz2:
                            candidate = (sz1 + sz2) // 2
                            
                        if candidate not in current_sizes and ratio > max_gap:
                            max_gap = ratio
                            best_new_sz = candidate
                            
                if best_new_sz is not None:
                    current_sizes.append(best_new_sz)
                    added_sizes.append(best_new_sz)
                    
            for sz in added_sizes:
                for i in range(RUNS_PER_POINT):
                    tasks.append(create_task(sz, i))

    total_tasks = len(tasks)
    max_workers = 3

    if total_tasks > 0:
        print(f"\n[INFO] Initializing {total_tasks} simulations (Repairs + New Points)...")
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(scaling_worker, task, data_arrays): task for task in tasks}
            
            for idx, future in enumerate(as_completed(futures)):
                res = future.result()
                file_exists = os.path.exists(results_csv)
                pd.DataFrame([res]).to_csv(results_csv, mode='a', header=not file_exists, index=False)
                
                pct = (idx + 1) / total_tasks
                bar = '█' * int(30 * pct) + '-' * (30 - int(30 * pct))
                sys.stdout.write(f'\r[PROGRESS] |{bar}| {(pct*100):.1f}% (Completed: {idx+1}/{total_tasks})')
                sys.stdout.flush()

        print(f"\n\n[SUCCESS] Execution completed in {(time.time() - start_time)/60:.2f} minutes.")
    else:
        print("\n[INFO] No tasks to run. Loading plot...")

    if os.path.exists(results_csv):
        df_results = pd.read_csv(results_csv)
        grouped_stats = df_results.groupby('size')['test_mse'].agg(['mean', 'std']).reset_index()

        sizes = grouped_stats['size'].values
        mse_mean = grouped_stats['mean'].values
        mse_std = grouped_stats['std'].values

        plt.figure(figsize=(10, 6))
        plt.plot(sizes, mse_mean, linestyle='-', color='b', linewidth=2, label='Mean Generalization Error')
        plt.fill_between(sizes, mse_mean - mse_std, mse_mean + mse_std, color='b', alpha=0.2, label='Variance (±1 Std Dev)')

        plt.title('Performance vs Dataset Size (Scaling Law)')
        plt.xlabel('Number of Training Samples')
        plt.ylabel('Generalization Error (Test MSE)')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.xscale('linear')
        plt.yscale('linear') 

        plt.tight_layout()
        save_path = os.path.join(figure_dir, "Dataset_Learning_Curve.png")
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Graph saved as '{save_path}'")

        plt.show()
    else:
        print("[ERROR] No data to plot.")