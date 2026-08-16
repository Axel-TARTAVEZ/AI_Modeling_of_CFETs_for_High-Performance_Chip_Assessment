import os
import warnings
import webbrowser
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import pickle
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --- 1. SETUP PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TCAD_MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "..", "TCAD_AI", "model")
SPICE_MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "model")
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figure")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --- 2. AI ARCHITECTURES ---
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

class CustomScaler:
    def __init__(self):
        self.mean, self.std = None, None
    def inverse_transform(self, data): return (data * self.std) + self.mean
    def transform(self, data): return (data - self.mean) / self.std

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

print("[Step 1/4] Loading Models...")
tcad_model = CFET_AI()
tcad_model.load_state_dict(torch.load(os.path.join(TCAD_MODEL_DIR, "CFET_Model.pth"), weights_only=True))
tcad_model.eval()
tcad_scaler_X = joblib.load(os.path.join(TCAD_MODEL_DIR, 'scaler_X.pkl'))
tcad_scaler_y = joblib.load(os.path.join(TCAD_MODEL_DIR, 'scaler_y.pkl'))

spice_model = SpiceNet(11, 2)
spice_model.load_state_dict(torch.load(os.path.join(SPICE_MODEL_DIR, "SPICE_AI.pth"), map_location='cpu'))
spice_model.eval()

with open(os.path.join(SPICE_MODEL_DIR, 'SPICE_AI_scaler_x.pkl'), 'rb') as f: spice_scaler_X = pickle.load(f)
with open(os.path.join(SPICE_MODEL_DIR, 'SPICE_AI_scaler_y.pkl'), 'rb') as f: spice_scaler_y = pickle.load(f)

# Load limits to avoid out-of-distribution AI extrapolation
with open(os.path.join(SPICE_MODEL_DIR, 'SPICE_AI_bounds.json'), 'r') as f:
    spice_bounds = json.load(f)

DEVICE_X_MIN, DEVICE_X_MAX = np.array(spice_bounds['X_min'][3:]), np.array(spice_bounds['X_max'][3:])
Y_MIN, Y_MAX = np.array(spice_bounds['y_min']), np.array(spice_bounds['y_max'])
X_MARGIN, Y_MARGIN_HIGH = 0.05, 2.0

# --- 3. TCAD INFERENCE ---
N_SAMPLES = 100000
np.random.seed(42) 
X_synth = np.column_stack((
    np.random.uniform(15.0, 35.0, N_SAMPLES), np.random.uniform(15.0, 80.0, N_SAMPLES),
    np.random.uniform(4.0, 8.0, N_SAMPLES), np.random.uniform(0.2, 1.2, N_SAMPLES),
    np.random.uniform(1.5, 2.2, N_SAMPLES)
))

with torch.no_grad():
    y_pred_tcad = tcad_scaler_y.inverse_transform(tcad_model(torch.tensor(tcad_scaler_X.transform(X_synth), dtype=torch.float32)).numpy())

print(f"[Step 2/4] TCAD Inference: {N_SAMPLES} profiles generated.")

Ion_N, Ioff_N, Vth_N, SS_N = np.abs(y_pred_tcad[:,0]), 10**y_pred_tcad[:,1], y_pred_tcad[:,2], y_pred_tcad[:,3]
Ion_P, Ioff_P, Vth_P, SS_P = np.abs(y_pred_tcad[:,4]), 10**y_pred_tcad[:,5], y_pred_tcad[:,6], y_pred_tcad[:,7]

beta = Ion_P / (Ion_N + 1e-20)
sym_mask = (beta >= 0) & (beta <= 1.20)

X_valid, beta_valid = X_synth[sym_mask], beta[sym_mask]
device_params = np.column_stack((
    Ion_N[sym_mask], Ioff_N[sym_mask], Vth_N[sym_mask], SS_N[sym_mask],
    Ion_P[sym_mask], Ioff_P[sym_mask], Vth_P[sym_mask], SS_P[sym_mask]
))

x_range = DEVICE_X_MAX - DEVICE_X_MIN
indist_mask = np.all((device_params >= DEVICE_X_MIN - X_MARGIN * x_range) & (device_params <= DEVICE_X_MAX + X_MARGIN * x_range), axis=1)
device_params_valid, X_valid, beta_valid = device_params[indist_mask], X_valid[indist_mask], beta_valid[indist_mask]

print(f"[Step 3/4] Physical Filters: {len(device_params_valid)} valid candidates.")

# --- 4. SPICE INFERENCE & PARETO ---
print(f"[Step 4/4] SPICE Inference & Pareto Extraction...")
gates, VDD = ["INV", "NAND2", "NOR2"], 0.7
plot_data, all_pareto_dfs = {}, []

for gate in gates:
    flags = [1.0 if gate == 'INV' else 0.0, 1.0 if gate == 'NAND2' else 0.0, 1.0 if gate == 'NOR2' else 0.0]
    spice_inputs = np.hstack((np.tile(flags, (len(device_params_valid), 1)), device_params_valid))
    
    with torch.no_grad():
        spice_preds_scaled = spice_model(torch.FloatTensor(spice_scaler_X.transform(spice_inputs))).numpy()
    spice_preds = spice_scaler_y.inverse_transform(spice_preds_scaled)
    
    delay_ps, leakage_nA = np.clip(np.abs(spice_preds[:, 0]) * 1e12, 1e-3, None), np.clip(np.abs(spice_preds[:, 1]) * 1e9, 1e-6, None)
    power_nW = leakage_nA * VDD 
    
    y_ok_mask = (
        (np.abs(spice_preds[:, 0]) >= Y_MIN[0]) & (np.abs(spice_preds[:, 0]) <= Y_MAX[0] * Y_MARGIN_HIGH) &
        (np.abs(spice_preds[:, 1]) >= Y_MIN[1]) & (np.abs(spice_preds[:, 1]) <= Y_MAX[1] * Y_MARGIN_HIGH) &
        (power_nW <= 15000.0)
    )
    
    d_filt, p_filt = delay_ps[y_ok_mask], power_nW[y_ok_mask]
    x_filt, b_filt = X_valid[y_ok_mask], beta_valid[y_ok_mask]
    
    sort_idx = np.argsort(d_filt)
    d_sorted, p_sorted, x_sorted, b_sorted = d_filt[sort_idx], p_filt[sort_idx], x_filt[sort_idx], b_filt[sort_idx]
    
    pareto_idx, min_power = [], float('inf')
    for i in range(len(d_sorted)):
        if p_sorted[i] < min_power:
            pareto_idx.append(i)
            min_power = p_sorted[i]
            
    d_par, p_par, x_par, b_par = d_sorted[pareto_idx], p_sorted[pareto_idx], x_sorted[pareto_idx], b_sorted[pareto_idx]
    fmax_par = 1000.0 / d_par 

    if len(d_par) >= 3:
        labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(StandardScaler().fit_transform(np.column_stack((d_par, np.log10(p_par)))))
        ord_clust = np.argsort([np.mean(d_par[labels == i]) for i in range(3)])
        hp, rvt, lp = ord_clust[0], ord_clust[1], ord_clust[2]
    else:
        labels, hp, rvt, lp = np.zeros(len(d_par), dtype=int), 0, 0, 0
    
    colors, flavors = np.empty(len(labels), dtype=object), np.empty(len(labels), dtype=object)
    colors[labels == hp], flavors[labels == hp] = '#d62728', 'HP'
    colors[labels == rvt], flavors[labels == rvt] = '#ff7f0e', 'RVT'
    colors[labels == lp], flavors[labels == lp] = '#1f77b4', 'LP'
    
    plot_data[gate] = {'all_d': d_filt, 'all_p': p_filt, 'p_d': d_par, 'p_p': p_par, 'c': colors, 'f': flavors, 'x': x_par, 'fmax': fmax_par, 'b': b_par}
    
    all_pareto_dfs.append(pd.DataFrame({
        'Gate': [gate]*len(d_par), 'Flavor': flavors,
        'Lch_nm': np.round(x_par[:, 0], 2), 'Wch_nm': np.round(x_par[:, 1], 2), 'Tch_nm': np.round(x_par[:, 2], 2),
        'Tox1_nm': np.round(x_par[:, 3], 2), 'Tox2_nm': np.round(x_par[:, 4], 2),
        'Delay_ps': np.round(d_par, 3), 'Power_nW': np.round(p_par, 3), 'Beta_Ratio': np.round(b_par, 3)
    }))

pd.concat(all_pareto_dfs, ignore_index=True).to_csv(os.path.join(DATA_DIR, "Pareto_Frontier_CFET.csv"), index=False)

# --- 5. DASHBOARD ---
fig = go.Figure()
for i, gate in enumerate(gates):
    d = plot_data[gate]
    is_vis = (i == 0)
    fig.add_trace(go.Scatter(x=d['all_d'], y=d['all_p'], mode='markers', marker=dict(color='lightgrey', size=4, opacity=0.3), name=f'{gate} Valid', hoverinfo='skip', visible=is_vis))
    fig.add_trace(go.Scatter(
        x=d['p_d'], y=d['p_p'], mode='markers+lines', marker=dict(color=d['c'], size=8, line=dict(color='black', width=1)),
        line=dict(color='black', width=1, dash='dash'), name=f'{gate} Pareto', visible=is_vis,
        customdata=np.column_stack((d['x'], d['f'], d['fmax'], d['b'])),
        hovertemplate="<b>Flavor: %{customdata[5]}</b><br><br><b>Delay (tp):</b> %{x:.2f} ps<br><b>Fmax:</b> %{customdata[6]:.1f} GHz<br><b>Power:</b> %{y:.2f} nW<br><b>Beta:</b> %{customdata[7]:.2f}<br>---<br>Lch: %{customdata[0]:.1f} nm<br>Wch: %{customdata[1]:.1f} nm<br>Tch: %{customdata[2]:.2f} nm<br>Tox1: %{customdata[3]:.2f} nm<br>Tox2: %{customdata[4]:.2f} nm<extra></extra>"
    ))

buttons = [{"label": g, "method": "update", "args": [{"visible": [j//2 == idx for j in range(len(gates)*2)]}, {"title": f"Cascaded AI: {g} Delay vs Static Power"}]} for idx, g in enumerate(gates)]
fig.update_layout(title=f"Cascaded AI: {gates[0]} Delay vs Static Power", xaxis_title="Delay tp [ps]", yaxis_title="Static Power [nW]", yaxis_type="log", plot_bgcolor='white', xaxis=dict(range=[0, 80], gridcolor="lightgrey"), yaxis=dict(gridcolor="lightgrey"), updatemenus=[dict(active=0, buttons=buttons, x=0.01, xanchor="left", y=0.99, yanchor="top", bgcolor="white")])
html_path = os.path.join(FIG_DIR, "AI_Design_Space.html")
fig.write_html(html_path)
webbrowser.open(f"file:///{os.path.abspath(html_path)}")
print("[SUCCESS] Dashboard opened.")