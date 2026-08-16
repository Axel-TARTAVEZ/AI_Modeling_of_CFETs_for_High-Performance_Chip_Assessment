import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib

script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "..", "model")
fig_dir = os.path.join(script_dir, "..", "figure")
os.makedirs(fig_dir, exist_ok=True)

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

class SingleOutputWrapper(nn.Module):
    def __init__(self, model, output_index):
        super().__init__()
        self.model = model
        self.output_index = output_index
        
    def forward(self, x):
        return self.model(x)[:, self.output_index].unsqueeze(1)

df_train = pd.read_csv(os.path.join(script_dir, "../../TCAD/data", "TCAD_train.csv"))
df_test = pd.read_csv(os.path.join(script_dir, "../../TCAD/data", "TCAD_test.csv"))
input_cols = ['Lch', 'Wch', 'Tch', 'Tox1', 'Tox2']
output_cols = ['Ion_N', 'Ioff_N', 'Vth_N', 'SS_N', 'Ion_P', 'Ioff_P', 'Vth_P', 'SS_P']

model = CFET_AI()
model.load_state_dict(torch.load(os.path.join(model_dir, "CFET_Model.pth"), weights_only=True))
model.eval()

scaler_X = joblib.load(os.path.join(model_dir, 'scaler_X.pkl'))
X_train_scaled = scaler_X.transform(df_train[input_cols].values)
X_test_scaled = scaler_X.transform(df_test[input_cols].values)

background = torch.tensor(shap.utils.sample(X_train_scaled, 200), dtype=torch.float32)
test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
X_test_phys = df_test[input_cols].values

def compute_shap_for_target(output_idx):
    wrapper = SingleOutputWrapper(model, output_index=output_idx)
    explainer = shap.GradientExplainer(wrapper, background)
    shap_values = explainer.shap_values(test_tensor)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
        
    return np.array(shap_values).reshape(X_test_phys.shape)

shap_values_dict = {}

print("==================================================")
print("             GENERATING SUMMARY PLOTS             ")
print("==================================================")

for idx, target_name in enumerate(output_cols):
    print(f"Processing SHAP values for {target_name}...")
    shap_values_dict[target_name] = compute_shap_for_target(idx)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_dict[target_name], X_test_phys, feature_names=input_cols, show=False)
    plt.title(f"Geometric Impact on {target_name}", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"SHAP_Beeswarm_{target_name}.png"), dpi=300)
    plt.close()

print("\n==================================================")
print("            GENERATING DEPENDENCE PLOTS           ")
print("==================================================")

dependence_plots = [
    ("Lch", "Vth_N", "Short Channel Effect (Vth roll-off NMOS)"),
    ("Lch", "Vth_P", "Short Channel Effect (Vth roll-off PMOS)"),
    ("Tox1", "SS_N", "Gate Electrostatic Control (NMOS)"),
    ("Tox2", "SS_P", "Gate Electrostatic Control (PMOS)"),
    ("Wch", "Ion_N", "Drive Current Scaling (NMOS)")
]

for feature, target, title in dependence_plots:
    print(f"Generating Dependence Plot: {feature} vs {target}...")
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(feature, shap_values_dict[target], X_test_phys, feature_names=input_cols, show=False)
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"SHAP_Dependence_{feature}_{target}.png"), dpi=300)
    plt.close()

print("\n[SUCCESS] All plots generated successfully in the 'figure' directory.")