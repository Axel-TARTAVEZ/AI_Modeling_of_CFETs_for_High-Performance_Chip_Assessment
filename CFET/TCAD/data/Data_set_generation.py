import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min

np.random.seed(42)

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "TCAD_output.txt")
train_path = os.path.join(script_dir, "TCAD_train.csv")
test_path = os.path.join(script_dir, "TCAD_test.csv")

nb_test_target = 300

input_cols = ['Lch', 'Wch', 'Tch', 'Tox1', 'Tox2']

print("[INFO] Loading raw TCAD file...")
df = pd.read_csv(file_path)
total_lines = len(df)

df_nan = df[df.isna().any(axis=1)]
df_999 = df[(df == -999.0).any(axis=1)]

df_clean = df.dropna()
df_clean = df_clean[(df_clean != -999.0).all(axis=1)].reset_index(drop=True)
total_simulated = len(df_clean)


if os.path.exists(test_path) and os.path.exists(train_path):
    print(f"[INFO] Existing datasets detected. Loading TRAIN and TEST directly.")
    df_test = pd.read_csv(test_path)
    df_train = pd.read_csv(train_path)

else:
    print(f"[INFO] No complete dataset found. Creating {nb_test_target} reference points via K-Means...")
    X_geom = df_clean[input_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_geom)


    kmeans = KMeans(n_clusters=nb_test_target, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_scaled)
    unique_indices = np.unique(closest_indices)

    if len(unique_indices) < nb_test_target:
        missing = nb_test_target - len(unique_indices)
        remaining_indices = np.setdiff1d(np.arange(total_simulated), unique_indices)
        pad_indices = np.random.choice(remaining_indices, missing, replace=False)
        test_indices = np.concatenate([unique_indices, pad_indices])

    else:
        test_indices = unique_indices

    df_test = df_clean.iloc[test_indices].copy()
    df_train = df_clean.drop(index=test_indices).copy()

    df_test.to_csv(test_path, index=False)
    df_train.to_csv(train_path, index=False)

    print("[SUCCESS] 'TCAD_train.csv' and 'TCAD_test.csv' generated and locked.")

print("\n==================================================")
print("            DATASET GENERATION REPORT             ")
print("==================================================")
print(f"-> Total rows read (TCAD_output) : {total_lines}")
print(f"-> Unexecuted Simulations (NaN)  : {len(df_nan)}")
print(f"-> TCAD Crashes (-999.0)         : {len(df_999)}")
print("--------------------------------------------------")
print(f"-> TOTAL VALID SIMULATIONS       : {total_simulated}")
print(f"   => Transistors for TRAIN      : {len(df_train)}")
print(f"   => Locked transistors for TEST: {len(df_test)}")
print("==================================================\n")