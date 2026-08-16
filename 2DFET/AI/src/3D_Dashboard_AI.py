import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import joblib
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

warnings.filterwarnings("ignore")

script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "..", "model")

class CFET_AI(nn.Module):
    def __init__(self, input_dim=4, output_dim=8, hidden_size=16):
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
model = CFET_AI(input_dim=4, output_dim=8)
model.load_state_dict(torch.load(os.path.join(model_dir, "2DFET_AI_Model.pth"), weights_only=True, map_location='cpu'))
model.eval()

scaler_X = joblib.load(os.path.join(model_dir, '2DFET_AI_scaler_x.pkl'))
scaler_y = joblib.load(os.path.join(model_dir, '2DFET_AI_scaler_y.pkl'))

TARGET_INFO = {
    'Ion_N': (0, 'Ion_N (A/µm)'),
    'Ioff_N': (1, 'Log(Ioff_N) (A/µm)'),
    'Vth_N': (2, 'Vth_N (V)'),
    'SS_N': (3, 'SS_N (mV/dec)'),
    'Ion_P': (4, 'Ion_P (A/µm)'),
    'Ioff_P': (5, 'Log(Ioff_P) (A/µm)'),
    'Vth_P': (6, 'Vth_P (V)'),
    'SS_P': (7, 'SS_P (mV/dec)')
}

app = dash.Dash(__name__)

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'padding': '20px'}, children=[
    html.H2("AI Physical Validation: Continuous 3D Extrapolation (2DFET)"),
    
    html.Div(style={'display': 'flex', 'gap': '40px'}, children=[

        html.Div(style={'flex': '1', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}, children=[
            
            html.H4("1. Electrical Parameter (Z-axis)", style={'color': '#005b96'}),
            dcc.Dropdown(
                id='dropdown-target',
                options=[{'label': k, 'value': k} for k in TARGET_INFO.keys()],
                value='Ioff_N',
                clearable=False,
                style={'marginBottom': '30px'}
            ),

            html.H4("2. Fixed Geometric Parameters", style={'color': '#005b96'}),
            
            html.Label("Channel Width (Wch) [15 - 80 nm]", style={'fontWeight': 'bold', 'marginTop': '10px'}),
            dcc.Slider(id='slider-wch', min=15, max=80, step=1, value=47.5, 
                       marks={15: '15', 30: '30', 50: '50', 80: '80'}),
            
            html.Label("High-k Thickness (Tox2) [1.5 - 2.2 nm]", style={'fontWeight': 'bold', 'marginTop': '30px'}),
            dcc.Slider(id='slider-tox2', min=1.5, max=2.2, step=0.05, value=1.85, 
                       marks={1.5: '1.5', 1.85: '1.85', 2.2: '2.2'})
        ]),
        
        html.Div(style={'flex': '3'}, children=[
            dcc.Graph(id='3d-surface', style={'height': '80vh'})
        ])
    ])
])


@app.callback(
    Output('3d-surface', 'figure'),
    [Input('slider-wch', 'value'),
     Input('slider-tox2', 'value'),
     Input('dropdown-target', 'value')]
)
def update_graph(wch_val, tox2_val, target_name):

    lch_vals = np.linspace(10.0, 45.0, 100) 
    tox1_vals = np.linspace(0.1, 1.5, 100) 
    L_grid, T1_grid = np.meshgrid(lch_vals, tox1_vals)

    L_flat = L_grid.flatten()
    T1_flat = T1_grid.flatten()
    

    W_flat = np.full_like(L_flat, wch_val)
    Tox2_flat = np.full_like(L_flat, tox2_val)


    X_synthetic = np.column_stack((L_flat, W_flat, T1_flat, Tox2_flat))

    X_scaled = scaler_X.transform(X_synthetic)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        y_pred_scaled = model(X_tensor).numpy()

    y_pred_phys = scaler_y.inverse_transform(y_pred_scaled)

    idx, z_label = TARGET_INFO[target_name]
    Z_vals = y_pred_phys[:, idx]
    Z_grid = Z_vals.reshape(100, 100)

    fig = go.Figure(data=[go.Surface(
        z=Z_grid, 
        x=L_grid, 
        y=T1_grid, 
        colorscale='Viridis',
        hovertemplate=(
            "<b>Lch:</b> %{x:.1f} nm<br>" +
            "<b>Tox1:</b> %{y:.2f} nm<br>" +
            f"<b>{target_name}:</b> %{{z:.4f}}<br>" +
            "<extra></extra>" 
        ),
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))
    )])

    fig.update_layout(
        title=f"AI-Modeled Response Surface: {target_name}",
        scene=dict(
            xaxis_title='Lch (nm)',
            yaxis_title='Tox1 (nm)',
            zaxis_title=z_label,
            xaxis=dict(backgroundcolor="white", gridcolor="lightgrey"),
            yaxis=dict(backgroundcolor="white", gridcolor="lightgrey"),
            zaxis=dict(backgroundcolor="white", gridcolor="lightgrey")
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        paper_bgcolor="white"
    )
    
    return fig


if __name__ == '__main__':
    print("[SUCCESS] Local server started!")
    print("Open your web browser and go to: http://127.0.0.1:8050/")
    app.run(debug=True, use_reloader=False)