import os
import warnings
import webbrowser
import pandas as pd
import numpy as np
import plotly.express as px

warnings.filterwarnings("ignore")

print("[INFO] Loading and cleaning dataset...")
script_dir = os.path.dirname(os.path.abspath(__file__))
figure_dir = os.path.join(script_dir, "..", "figure")
file_path = os.path.join(script_dir, "../data", "TCAD_output.txt")
df = pd.read_csv(file_path)
df_clean = df.dropna()
df_clean = df_clean[(df_clean != -999.0).all(axis=1)]

print("[INFO] Generating 3D Plot: Short-Channel Effect...")
df_clean['Log_Ioff_N'] = np.log10(np.abs(df_clean['Ioff_N']) + 1e-20)

fig = px.scatter_3d(
    df_clean, 
    x='Lch', y='Tox1', z='Log_Ioff_N',
    color='Log_Ioff_N',
    color_continuous_scale='Viridis',
    title="Physical Validation: Short-Channel Effect on NMOS Leakage Current",
    labels={
        'Log_Ioff_N': 'Log(I_off) (A/µm)', 
        'Lch': 'Channel Length (nm)', 
        'Tox1': 'Oxide Thickness (nm)'
    },
    opacity=0.8
)

fig.update_traces(marker=dict(size=5, line=dict(width=1, color='DarkSlateGrey')))
fig.update_layout(scene=dict(bgcolor="white"))

html_filename = "3D_Short_Channel_Effect.html"
fig.write_html(os.path.join(figure_dir, html_filename))

html_path = f"file:///{os.path.abspath(os.path.join(figure_dir, html_filename))}"
webbrowser.open(html_path)

print(f"[SUCCESS] 3D plot saved and opened: {html_filename}")