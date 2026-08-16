import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import torch
import torch.nn as nn
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

script_dir = os.path.dirname(os.path.abspath(__file__))
figure_dir = os.path.join(script_dir, "..", "figure")
model_dir = os.path.join(script_dir, "..", "model")

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

input_dim = 5
output_dim = 8 

device = torch.device('cpu')

model_pinn = CFET_AI(input_dim, output_dim)
model_pinn.load_state_dict(torch.load(os.path.join(model_dir, "CFET_Model.pth"), map_location=device))
model_pinn.eval()

scaler_X = joblib.load(os.path.join(model_dir, 'scaler_X.pkl'))
scaler_y = joblib.load(os.path.join(model_dir, 'scaler_y.pkl'))

output_cols = ['Ion_N', 'Ioff_N', 'Vth_N', 'SS_N', 'Ion_P', 'Ioff_P', 'Vth_P', 'SS_P']

print("\n===========")
print("  TCAD AI  ")
print("===========\n")

while True:
    try:
        Lch = float(input("Lch (nm)    : "))
        Wch = float(input("Wch (nm)    : "))
        Tch = float(input("Tch (nm)    : "))
        Tox1 = float(input("Tox1 (nm)   : "))
        Tox2 = float(input("Tox2 (nm)   : "))
    except ValueError:
        print("Error: Please enter numbers only.")
        continue

    new_cfet = np.array([[Lch, Wch, Tch, Tox1, Tox2]])

    with torch.no_grad():
        cfet_scaled = scaler_X.transform(new_cfet)
        cfet_tensor = torch.tensor(cfet_scaled, dtype=torch.float32)
        
        prediction_scaled = model_pinn(cfet_tensor)
        
        prediction_real = scaler_y.inverse_transform(prediction_scaled.numpy())
        
        prediction_real[0][1] = 10 ** prediction_real[0][1]
        prediction_real[0][5] = 10 ** prediction_real[0][5]

        print("\n--- Model Predictions ---")
        for name, value in zip(output_cols, prediction_real[0]):
            print(f"{name:<6} : {value:.4e}")
        print("-" * 40)
        
    choix = input("\nTest another geometry? (y/n) : ")
    if choix.lower() != 'y':
        print("Closing simulator.")
        break
    print("\n")