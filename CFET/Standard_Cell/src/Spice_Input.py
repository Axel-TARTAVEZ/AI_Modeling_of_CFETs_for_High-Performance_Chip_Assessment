import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from scipy.stats import qmc

script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "../../TCAD_AI", "model")
data_dir = os.path.join(script_dir, "..", "data")
os.makedirs(data_dir, exist_ok=True)

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

NUM_POINTS = 10000
device = torch.device("cpu")

bounds = {
    'Lch':  [15.0, 35.0],
    'Wch':  [20.0, 80.0],
    'Tch':  [4.0, 8.0],
    'Tox1': [0.2, 0.4],
    'Tox2': [1.5, 2.0]
}

print(f"[INFO] Generating {NUM_POINTS} LHS samples...")
sampler = qmc.LatinHypercube(d=len(bounds))
sample = sampler.random(n=NUM_POINTS)

l_bounds = np.array([bounds[k][0] for k in bounds])
u_bounds = np.array([bounds[k][1] for k in bounds])
X_phys_lhs = qmc.scale(sample, l_bounds, u_bounds)

print("[INFO] Loading model and scalers...")
model = CFET_AI(input_dim=5).to(device)
model.load_state_dict(torch.load(os.path.join(model_dir, "CFET_Model.pth"), map_location=device, weights_only=True))
model.eval()

scaler_X = joblib.load(os.path.join(model_dir, 'scaler_X.pkl'))
scaler_y = joblib.load(os.path.join(model_dir, 'scaler_y.pkl'))

X_scaled = scaler_X.transform(X_phys_lhs)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

print("[INFO] Running inference...")
with torch.no_grad():
    y_pred_scaled = model(X_tensor).numpy()

y_pred_phys = scaler_y.inverse_transform(y_pred_scaled)

y_pred_phys[:, 1] = 10 ** y_pred_phys[:, 1]
y_pred_phys[:, 5] = 10 ** y_pred_phys[:, 5]

input_cols = list(bounds.keys())
output_cols = ['Ion_N', 'Ioff_N', 'Vth_N', 'SS_N', 'Ion_P', 'Ioff_P', 'Vth_P', 'SS_P']

df_final = pd.DataFrame(np.hstack((X_phys_lhs, y_pred_phys)), columns=input_cols + output_cols)

output_csv = os.path.join(data_dir, "Spice_input.csv")
df_final.to_csv(output_csv, index=False)

print(f"[SUCCESS] Dataset saved to: {output_csv}")