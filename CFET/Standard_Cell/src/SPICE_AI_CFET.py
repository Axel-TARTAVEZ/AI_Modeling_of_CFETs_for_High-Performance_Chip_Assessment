import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "model"))
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "Spice_output.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "SPICE_AI.pth")
SCALER_X_PATH = os.path.join(MODEL_DIR, "SPICE_AI_scaler_x.pkl")
SCALER_Y_PATH = os.path.join(MODEL_DIR, "SPICE_AI_scaler_y.pkl")
BOUNDS_PATH = os.path.join(MODEL_DIR, "SPICE_AI_bounds.json")

class CustomScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit_transform(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.std[self.std == 0.0] = 1.0
        return (data - self.mean) / self.std

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

def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH).dropna()
    initial_len = len(df)
    
    # Physics filters
    df = df[(df['Ion_N'] > 1e-6) & (df['Ion_P'] > 1e-6)]
    df = df[(df['Ioff_N'] < 1e-6) & (df['Ioff_P'] < 1e-6)]
    df = df[(df['Vth_N'].abs() > 0.05) & (df['Vth_N'].abs() < 0.5)]
    df = df[(df['Vth_P'].abs() > 0.05) & (df['Vth_P'].abs() < 0.5)]
    df = df[(df['t_delay'] > 0.1e-12) & (df['t_delay'] < 200e-12)]
    df = df[(df['i_leak'].abs() < 10e-6)]
    
    print(f"--- DATA CLEANING ---\nInitial: {initial_len}\nRetained: {len(df)}")

    # Encoding
    df['is_INV'] = (df['Gate_Type'] == 'INV').astype(float)
    df['is_NAND2'] = (df['Gate_Type'] == 'NAND2').astype(float)
    df['is_NOR2'] = (df['Gate_Type'] == 'NOR2').astype(float)
    
    features = [
        'is_INV', 'is_NAND2', 'is_NOR2',
        'Ion_N', 'Ioff_N', 'Vth_N', 'SS_N',
        'Ion_P', 'Ioff_P', 'Vth_P', 'SS_P'
    ]
    targets = ['t_delay', 'i_leak']
    
    X = df[features].values
    y = df[targets].values
    
    scaler_x, scaler_y = CustomScaler(), CustomScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    with open(SCALER_X_PATH, 'wb') as f: pickle.dump(scaler_x, f)
    with open(SCALER_Y_PATH, 'wb') as f: pickle.dump(scaler_y, f)

    # Export boundary envelope for Design Space Explorer
    bounds = {
        'features': features,
        'targets': targets,
        'X_min': X.min(axis=0).tolist(),
        'X_max': X.max(axis=0).tolist(),
        'y_min': y.min(axis=0).tolist(),
        'y_max': y.max(axis=0).tolist(),
    }
    with open(BOUNDS_PATH, 'w') as f: json.dump(bounds, f, indent=2)
    
    # 80/20 Split
    np.random.seed(42)
    indices = np.random.permutation(len(X_scaled))
    test_count = int(len(X_scaled) * 0.20)
    return X_scaled[test_count:], X_scaled[:test_count], y_scaled[test_count:], y_scaled[:test_count]

def train_model():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    dataset = TensorDataset(torch.FloatTensor(X_train).to(device), torch.FloatTensor(y_train).to(device))
    X_test_t, y_test_t = torch.FloatTensor(X_test).to(device), torch.FloatTensor(y_test).to(device)
    
    batch_size = min(4096, len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=(len(dataset) >= 2*batch_size))
    
    model = SpiceNet(input_dim=11, output_dim=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    epochs, patience, best_val_loss, patience_counter = 2000, 200, float('inf'), 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_test_t), y_test_t).item()
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Train: {epoch_loss/len(dataloader):.5f} | Val: {val_loss:.5f} | Best: {best_val_loss:.5f}")

        if patience_counter >= patience:
            print(f"\n[Early Stopping] Epoch {epoch+1}.")
            break

if __name__ == "__main__":
    train_model()