import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))

train_path = os.path.join(script_dir, "../../TCAD/data", "TCAD_train.csv")

output_cols = ['Ion_N', 'Ioff_N', 'Vth_N', 'SS_N', 'Ion_P', 'Ioff_P', 'Vth_P', 'SS_P']

print("[INFO] Chargement des données brutes (échelle linéaire)...")
df_train = pd.read_csv(train_path)

print("\n==================================================")
print("              PHYSICAL STATISTICS                  ")
print("==================================================")

for col in output_cols:
    std_val = df_train[col].std()
    mean_val = df_train[col].mean()
    
    if 'I' in col:
        print(f"{col:<7} : STD = {std_val:.4e} A \t(Moyenne = {mean_val:.4e} A)")
    elif 'Vth' in col:
        print(f"{col:<7} : STD = {std_val:.4f} V \t(Moyenne = {mean_val:.4f} V)")
    else:
        print(f"{col:<7} : STD = {std_val:.2f} mV/dec \t(Moyenne = {mean_val:.2f} mV/dec)")
        
print("==================================================\n")