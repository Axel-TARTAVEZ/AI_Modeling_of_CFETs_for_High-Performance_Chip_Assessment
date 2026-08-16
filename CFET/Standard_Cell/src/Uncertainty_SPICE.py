import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "model"))

DATA_PATH = os.path.join(DATA_DIR, "Spice_output.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "SPICE_AI.pth")
SCALER_X_PATH = os.path.join(MODEL_DIR, "SPICE_AI_scaler_x.pkl")
SCALER_Y_PATH = os.path.join(MODEL_DIR, "SPICE_AI_scaler_y.pkl")

class CustomScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class SpiceNet(nn.Module):
    def __init__(self, input_dim=11, output_dim=2, hidden_size=64):
        super(SpiceNet, self).__init__()
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

def get_test_set_only():
    df = pd.read_csv(DATA_PATH).dropna()
    
    df = df[(df['Ion_N'] > 1e-6) & (df['Ion_P'] > 1e-6)]
    df = df[(df['Ioff_N'] < 1e-6) & (df['Ioff_P'] < 1e-6)]
    df = df[(df['Vth_N'].abs() > 0.05) & (df['Vth_N'].abs() < 0.5)]
    df = df[(df['Vth_P'].abs() > 0.05) & (df['Vth_P'].abs() < 0.5)]
    df = df[(df['t_delay'] > 0.1e-12) & (df['t_delay'] < 200e-12)]
    df = df[(df['i_leak'].abs() < 10e-6)]

    df['is_INV'] = (df['Gate_Type'] == 'INV').astype(float)
    df['is_NAND2'] = (df['Gate_Type'] == 'NAND2').astype(float)
    df['is_NOR2'] = (df['Gate_Type'] == 'NOR2').astype(float)
    
    features = [
        'is_INV', 'is_NAND2', 'is_NOR2', 
        'Ion_N', 'Ioff_N', 'Vth_N', 'SS_N',
        'Ion_P', 'Ioff_P', 'Vth_P', 'SS_P'
    ]
    
    targets = ['t_delay', 'i_leak']
    
    X = df[features].values
    y = df[targets].values
    
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    test_count = int(len(X) * 0.20)
    test_idx = indices[:test_count]  
    
    X_test_unscaled = X[test_idx]
    y_test_phys = y[test_idx]
    df_test = df.iloc[test_idx].copy()
    
    return X_test_unscaled, y_test_phys, df_test, targets

def run_uncertainty_analysis():
    X_test_unscaled, y_test_phys, df_test, targets = get_test_set_only()
    
    print("\n==========================================================================================")
    print("                              SPICE DATASET SANITIZATION                                  ")
    print("==========================================================================================")
    print(f"Total valid dataset samples : {int(len(df_test) * 5)}") 
    print(f"Test samples isolated       : {len(df_test)} (Strictly unseen by the model)")
    print("==========================================================================================\n")

    with open(SCALER_X_PATH, 'rb') as f:
        scaler_x = pickle.load(f)
    with open(SCALER_Y_PATH, 'rb') as f:
        scaler_y = pickle.load(f)
        
    X_test_scaled = scaler_x.transform(X_test_unscaled)
    X_test_t = torch.FloatTensor(X_test_scaled)
    
    model = SpiceNet(11, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    
    with torch.no_grad():
        preds_scaled = model(X_test_t).numpy()
        
    preds_phys = scaler_y.inverse_transform(preds_scaled)
    
    absolute_errors = np.abs(preds_phys - y_test_phys)
    relative_errors = (absolute_errors / (np.abs(y_test_phys) + 1e-15)) * 100

    print("==========================================================================================")
    print("         PHYSICAL UNCERTAINTY (UNSEEN TEST SET) - MEAN vs MEDIAN                          ")
    print("==========================================================================================")
    print(f"{'Variable':<12} | {'RMSE (Absolute)':<18} | {'Mean Rel. Err':<16} | {'Median Rel. Err (Typical)':<20}")
    print("------------------------------------------------------------------------------------------")

    units = [1e12, 1e9]  # ps, nA
    unit_labels = ["ps", "nA"]

    for i, col in enumerate(targets):
        rmse_col = np.sqrt(np.mean(absolute_errors[:, i]**2)) * units[i]
        mean_rel = np.mean(relative_errors[:, i])
        median_rel = np.median(relative_errors[:, i])
        
        print(f"{col:<12} | {rmse_col:>8.3f} {unit_labels[i]:<7} | {mean_rel:>10.2f} %     | {median_rel:>15.2f} %")
    print("==========================================================================================\n")

    print("==========================================================================================")
    print("                     WORST-CASE TOPOLOGY INVESTIGATION (TEST SET)                         ")
    print("==========================================================================================\n")

    for i, col in enumerate(targets):
        df_test[f'True_{col}'] = y_test_phys[:, i]
        df_test[f'Pred_{col}'] = preds_phys[:, i]
        df_test[f'ErrRel_{col}(%)'] = relative_errors[:, i]
        
        worst_cases = df_test.sort_values(by=f'ErrRel_{col}(%)', ascending=False).head(3)
        
        print(f"--- Top 3 Worst Predictions for {col} ---")
        for _, row in worst_cases.iterrows():
            gate = row['Gate_Type']
            true_val = row[f'True_{col}']
            pred_val = row[f'Pred_{col}']
            err = row[f'ErrRel_{col}(%)']
            
            print(f"Topology : Gate={gate} (1.0 fF Load)")
            
            if 'leak' in col:
                print(f"   => TRUE: {true_val*1e9:.3f} nA | PRED: {pred_val*1e9:.3f} nA | Error: {err:.1f} %")
            else:
                print(f"   => TRUE: {true_val*1e12:.3f} ps | PRED: {pred_val*1e12:.3f} ps | Error: {err:.1f} %")
        print("-" * 90)

if __name__ == "__main__":
    run_uncertainty_analysis()