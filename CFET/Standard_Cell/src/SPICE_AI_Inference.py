import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import warnings
import torch
import torch.nn as nn
import numpy as np
import pickle
import joblib

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPICE_MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "model"))
SPICE_MODEL_PATH = os.path.join(SPICE_MODEL_DIR, "SPICE_AI.pth")
SPICE_SCALER_X_PATH = os.path.join(SPICE_MODEL_DIR, "SPICE_AI_scaler_x.pkl")
SPICE_SCALER_Y_PATH = os.path.join(SPICE_MODEL_DIR, "SPICE_AI_scaler_y.pkl")

TCAD_MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "TCAD_AI", "model"))
TCAD_MODEL_PATH = os.path.join(TCAD_MODEL_DIR, "CFET_Model.pth")
TCAD_SCALER_X_PATH = os.path.join(TCAD_MODEL_DIR, "scaler_X.pkl")
TCAD_SCALER_Y_PATH = os.path.join(TCAD_MODEL_DIR, "scaler_y.pkl")

class CustomScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def inverse_transform(self, data):
        return (data * self.std) + self.mean
        
    def transform(self, data):
        return (data - self.mean) / self.std

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

class DTCO_Predictor:
    def __init__(self):
        if not os.path.exists(SPICE_MODEL_PATH):
            raise FileNotFoundError(f"SPICE model not found: {SPICE_MODEL_PATH}")
        with open(SPICE_SCALER_X_PATH, 'rb') as f: 
            self.spice_scaler_x = pickle.load(f)
        with open(SPICE_SCALER_Y_PATH, 'rb') as f: 
            self.spice_scaler_y = pickle.load(f)
        
        self.spice_model = SpiceNet(11, 2)
        self.spice_model.load_state_dict(torch.load(SPICE_MODEL_PATH, map_location='cpu'))
        self.spice_model.eval()

        if not os.path.exists(TCAD_MODEL_PATH):
            print(f"[Warning] TCAD model not found at {TCAD_MODEL_PATH}. 'End-to-End' mode disabled.")
            self.tcad_available = False
        else:
            self.tcad_available = True
            self.tcad_scaler_x = joblib.load(TCAD_SCALER_X_PATH)
            self.tcad_scaler_y = joblib.load(TCAD_SCALER_Y_PATH)
            
            self.tcad_model = CFET_AI(5, 8)
            self.tcad_model.load_state_dict(torch.load(TCAD_MODEL_PATH, map_location='cpu'))
            self.tcad_model.eval()

    def predict_spice(self, gate_type, n_params, p_params):
        is_inv = 1.0 if gate_type == 'INV' else 0.0
        is_nand = 1.0 if gate_type == 'NAND2' else 0.0
        is_nor = 1.0 if gate_type == 'NOR2' else 0.0
        
        features = np.array([[is_inv, is_nand, is_nor, *n_params, *p_params]])
        scaled_features = self.spice_scaler_x.transform(features)
        
        with torch.no_grad():
            scaled_preds = self.spice_model(torch.FloatTensor(scaled_features)).numpy()
            
        preds = self.spice_scaler_y.inverse_transform(scaled_preds)[0]
        
        return {
            't_delay_ps': abs(preds[0]) * 1e12,
            'i_leak_nA': abs(preds[1]) * 1e9
        }

    def predict_e2e(self, gate_type, geom_params):
        if not self.tcad_available: 
            return None, None, None
        
        scaled_geom = self.tcad_scaler_x.transform(np.array([geom_params]))
        with torch.no_grad():
            scaled_tcad = self.tcad_model(torch.FloatTensor(scaled_geom)).numpy()
        tcad_out = self.tcad_scaler_y.inverse_transform(scaled_tcad)[0]
        
        n_params = (abs(tcad_out[0]), 10**tcad_out[1], tcad_out[2], tcad_out[3])
        p_params = (abs(tcad_out[4]), 10**tcad_out[5], tcad_out[6], tcad_out[7])
        
        spice_res = self.predict_spice(gate_type, n_params, p_params)
        
        return spice_res, n_params, p_params

def get_input(prompt, expected_count=None):
    while True:
        user_input = input(prompt).strip()
        if user_input.lower() in ['q', 'quit', 'exit']: 
            return None
        if not expected_count: 
            return user_input.upper()
        
        try:
            values = tuple(float(x.strip()) for x in user_input.split(','))
            if len(values) != expected_count:
                print(f"Error: Expected exactly {expected_count} comma-separated values.")
                continue
            return values
        except ValueError:
            print("Format error. Valid example: 20.0, 10.5, 5.0, 0.3, 1.5")

if __name__ == "__main__":
    print("\n" + "="*27)
    print("CASCADED AI (TCAD -> SPICE)")
    print("="*27)
    
    try:
        engine = DTCO_Predictor()
        
        while True:
            print("\nMain Menu (Type 'q' to quit):")
            print("  [1] End-to-End Mode (Enter Geometry: Lch, Wch...)")
            print("  [2] SPICE Only Mode (Enter Electrical Characteristics)")
            
            mode = input("\nSelection (1 or 2): ").strip()
            if mode.lower() == 'q': break
            if mode not in ['1', '2']: continue
            
            gate = get_input("\nGate topology (INV, NAND2, NOR2): ")
            if not gate or gate not in ['INV', 'NAND2', 'NOR2']: 
                if gate: print("Invalid topology.")
                continue

            if mode == '1':
                if not engine.tcad_available: continue
                
                print("\nEnter the 5 geometric parameters separated by commas:")
                geom = get_input("Lch (nm), Wch (nm), Tch (nm), Tox1 (nm), Tox2 (nm): ", expected_count=5)
                if not geom: break
                
                res, n_p, p_p = engine.predict_e2e(gate, geom)
                
                print("\n" + "-"*50)
                print("1. TCAD PREDICTIONS (Physical Device)")
                print("-" * 50)
                print(f"NMOS -> Ion: {n_p[0]:.2e} A | Ioff: {n_p[1]:.2e} A | Vth: {n_p[2]:.3f} V")
                print(f"PMOS -> Ion: {p_p[0]:.2e} A | Ioff: {p_p[1]:.2e} A | Vth: {p_p[2]:.3f} V")
                
            else:
                print("\nEnter the electrical parameters separated by commas:")
                n_p = get_input("NMOS (Ion, Ioff, Vth, SS): ", expected_count=4)
                if not n_p: break
                
                p_p = get_input("PMOS (Ion, Ioff, Vth, SS): ", expected_count=4)
                if not p_p: break
                
                res = engine.predict_spice(gate, n_p, p_p)

            print("\n" + "-"*50)
            print("2. SPICE PREDICTIONS (N3 Circuit Performance at 1.0 fF Load)")
            print("-" * 50)
            print(f"▶ Propagation Delay : {res['t_delay_ps']:>8.3f} ps")
            print(f"▶ Leakage Current   : {res['i_leak_nA']:>8.3f} nA")
            print("="*27)
            
    except KeyboardInterrupt:
        print("\nProgram terminated.")