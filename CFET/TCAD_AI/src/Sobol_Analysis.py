import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt

from SALib.sample import saltelli
from SALib.analyze import sobol

script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "..", "model")
fig_dir = os.path.join(script_dir, "..", "figure")
os.makedirs(fig_dir, exist_ok=True)

# AI Model Architecture
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

print("[INFO] Loading AI model and scalers...")
model = CFET_AI()
model.load_state_dict(torch.load(os.path.join(model_dir, "CFET_Model.pth"), weights_only=True))
model.eval()

scaler_X = joblib.load(os.path.join(model_dir, 'scaler_X.pkl'))
scaler_y = joblib.load(os.path.join(model_dir, 'scaler_y.pkl'))

input_cols = ['Lch', 'Wch', 'Tch', 'Tox1', 'Tox2']
output_cols = ['Ion_N', 'Ioff_N', 'Vth_N', 'SS_N', 'Ion_P', 'Ioff_P', 'Vth_P', 'SS_P']

# Define the SALib Problem based on CFET bounds
problem = {
    'num_vars': 5,
    'names': input_cols,
    'bounds': [
        [15.0, 35.0],  # Lch (nm)
        [15.0, 80.0],  # Wch (nm)
        [4.0,  8.0],   # Tch (nm)
        [0.2,  1.2],   # Tox1 (nm)
        [1.5,  2.2]    # Tox2 (nm)
    ]
}

print("[INFO] Generating Saltelli sequence (DoE) for Sobol analysis...")
# N=2048 generates N * (2D + 2) = 24,576 samples for 5 variables
X_saltelli = saltelli.sample(problem, 2048)

print(f"[INFO] Running AI inference on {X_saltelli.shape[0]} synthetic samples...")
X_scaled = scaler_X.transform(X_saltelli)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

with torch.no_grad():
    y_pred_scaled = model(X_tensor).numpy()

y_pred_phys = scaler_y.inverse_transform(y_pred_scaled)

print("\n==================================================")
print("          COMPUTING & PLOTTING SOBOL INDICES      ")
print("==================================================")

x_pos = np.arange(len(input_cols))
width = 0.35  

for i, target_name in enumerate(output_cols):
    print(f"Processing Sobol Indices for {target_name}...")
    
    Y = y_pred_phys[:, i]
    
    # Calculate Sobol indices
    Si = sobol.analyze(problem, Y, print_to_console=False)
    
    S1 = Si['S1']
    ST = Si['ST']
    
    # Clean up minor numerical artifacts (e.g., -0.001 becomes 0)
    S1 = np.maximum(S1, 0)
    ST = np.maximum(ST, 0)

    # Generate grouped bar chart
    fig, ax = plt.subplots(figsize=(9, 6))
    
    rects1 = ax.bar(x_pos - width/2, S1, width, label='First-Order (S1) - Direct Impact', color='#1f77b4')
    rects2 = ax.bar(x_pos + width/2, ST, width, label='Total-Order (ST) - Includes Interactions', color='#ff7f0e')

    ax.set_ylabel('Sobol Sensitivity Index', fontsize=12, fontweight='bold')
    ax.set_title(f'Variance-Based Sensitivity Analysis for {target_name}', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(input_cols, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set y-limit dynamically but max at 1.05 to leave room for labels
    max_val = max(np.max(S1), np.max(ST))
    ax.set_ylim(0, min(1.1, max_val + 0.1))

    # Add numeric labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0.01:
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"Sobol_Indices_{target_name}.png"), dpi=300)
    plt.close()

print("\n[SUCCESS] All Sobol bar charts generated successfully in the 'figure' directory.")