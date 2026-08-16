import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib

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

script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "..", "model")

# 1. Load Data
df_test = pd.read_csv(os.path.join(script_dir, "../../TCAD/data", "TCAD_test.csv"))
input_cols = ['Lch', 'Wch', 'Tch', 'Tox1', 'Tox2']
output_cols = ['Ion_N', 'Ioff_N', 'Vth_N', 'SS_N', 'Ion_P', 'Ioff_P', 'Vth_P', 'SS_P']

# 2. TCAD Data Sanitization
Ion_N_true = np.abs(df_test['Ion_N'].values)
Ion_P_true = np.abs(df_test['Ion_P'].values)
beta_ratio = Ion_P_true / (Ion_N_true + 1e-20)

mask = (beta_ratio >= 0.35) & (beta_ratio <= 1.5)
df_filtered = df_test[mask].reset_index(drop=True)

print("\n==========================================================================================")
print("                              TCAD DATASET SANITIZATION                                   ")
print("==========================================================================================")
print(f"Initial samples     : {len(df_test)}")
print(f"Valid samples kept  : {len(df_filtered)}")
print(f"Outliers discarded  : {len(df_test) - len(df_filtered)}")
print("==========================================================================================\n")

X_test = df_filtered[input_cols].values
y_test_phys = df_filtered[output_cols].values

# 3. Model & Scalers
scaler_X = joblib.load(os.path.join(model_dir, "scaler_X.pkl"))
scaler_y = joblib.load(os.path.join(model_dir, "scaler_y.pkl"))

X_test_scaled = scaler_X.transform(X_test)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

model_path = os.path.join(model_dir, "CFET_Model.pth") 
model = CFET_AI()
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

# 4. Inference & Denormalization
with torch.no_grad():
    preds_scaled = model(X_test_t).numpy()

preds_log = scaler_y.inverse_transform(preds_scaled)
preds_phys = np.copy(preds_log)
preds_phys[:, 1] = 10 ** preds_log[:, 1]
preds_phys[:, 5] = 10 ** preds_log[:, 5]

# 5. Error Calculation
absolute_errors = np.abs(preds_phys - y_test_phys)
relative_errors = (absolute_errors / (np.abs(y_test_phys) + 1e-15)) * 100

print("==========================================================================================")
print("         PHYSICAL UNCERTAINTY (CLEANED DATASET) - MEAN vs MEDIAN                          ")
print("==========================================================================================")
print(f"{'Variable':<10} | {'RMSE':<12} | {'Mean Rel. Err':<16} | {'Median Rel. Err (Typical)':<20}")
print("------------------------------------------------------------------------------------------")

for i, col in enumerate(output_cols):
    rmse_col = np.sqrt(np.mean(absolute_errors[:, i]**2))
    mean_rel = np.mean(relative_errors[:, i])
    median_rel = np.median(relative_errors[:, i])
    
    print(f"{col:<10} | {rmse_col:<12.3e} | {mean_rel:>10.2f} %     | {median_rel:>15.2f} %")
print("==========================================================================================\n")

# 6. Worst-Case Investigation
df_results = df_filtered[input_cols].copy()
targets = [('Ion_N', 0), ('Ioff_N', 1), ('Ion_P', 4), ('Ioff_P', 5)]

for name, idx in targets:
    df_results[f'True_{name}'] = y_test_phys[:, idx]
    df_results[f'Pred_{name}'] = preds_phys[:, idx]
    df_results[f'ErrRel_{name}(%)'] = relative_errors[:, idx]

print("==========================================================================================")
print("                     WORST-CASE GEOMETRY INVESTIGATION (POST-FILTER)                      ")
print("==========================================================================================\n")

for name, idx in targets:
    err_col = f'ErrRel_{name}(%)'
    worst_cases = df_results.sort_values(by=err_col, ascending=False).head(3)
    
    print(f"--- Top 3 Worst Predictions for {name} ---")
    for _, row in worst_cases.iterrows():
        print(f"Geometry : Lch={row['Lch']}nm, Wch={row['Wch']}nm, Tch={row['Tch']}nm, Tox1={row['Tox1']}nm, Tox2={row['Tox2']}nm")
        
        true_val = row[f'True_{name}']
        pred_val = row[f'Pred_{name}']
        
        if 'Ioff' in name:
            print(f"   => TRUE: {true_val:.3e} A | PRED: {pred_val:.3e} A | Error: {row[err_col]:.1f} %")
        else:
            print(f"   => TRUE: {true_val*1e6:.2f} µA | PRED: {pred_val*1e6:.2f} µA | Error: {row[err_col]:.1f} %")
    print("-" * 90)