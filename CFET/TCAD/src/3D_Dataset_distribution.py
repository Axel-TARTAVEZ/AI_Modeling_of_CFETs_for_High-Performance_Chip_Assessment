import os
import pandas as pd
import plotly.express as px
import webbrowser

script_dir = os.path.dirname(os.path.abspath(__file__))
figure_dir = os.path.join(script_dir, "..", "figure")

train_path = os.path.join(script_dir, "../data", "TCAD_train.csv")
test_path = os.path.join(script_dir, "../data", "TCAD_test.csv")

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

train_count = len(df_train)
test_count = len(df_test)

print(f"[INFO] Train samples: {train_count}")
print(f"[INFO] Test samples: {test_count}")

train_label = f"Train ({train_count})"
test_label = f"Test ({test_count})"

df_train['Dataset'] = train_label
df_test['Dataset'] = test_label

df_all = pd.concat([df_train, df_test], ignore_index=True)

fig = px.scatter_3d(
    df_all, 
    x='Lch', 
    y='Wch', 
    z='Tch',
    color='Dataset',
    color_discrete_map={
        train_label: '#1f77b4',
        test_label: '#d62728'
    },
    title="CFET Dataset Space Distribution",
    labels={
        'Lch': 'Channel Length (Lch) [nm]', 
        'Wch': 'Channel Width (Wch) [nm]', 
        'Tch': 'Channel Thickness (Tch) [nm]'
    },
    opacity=0.8
)

fig.update_traces(marker=dict(size=4, line=dict(width=0)))

html_filename = os.path.join(figure_dir, "3D_Train_Test_Distribution.html")
fig.write_html(html_filename)

print(f"[SUCCESS] HTML plot exported to: {html_filename}")
webbrowser.open(f"file:///{os.path.abspath(html_filename)}")