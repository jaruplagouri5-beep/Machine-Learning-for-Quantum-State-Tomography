import torch
import torch.optim as optim
from dataset import QuantumDataset
from model import DensityMatrixReconstructor
from torch.utils.data import DataLoader
import numpy as np
import os

def train():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Data
    # 2 qubits -> dim=4. Input features = 16 (4x4 Pauli measurements)
    dataset = QuantumDataset(num_samples=2000, num_qubits=2)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 3. Model
    model = DensityMatrixReconstructor(input_dim=16, dim=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Loss function: Mean Squared Error
    criterion = torch.nn.MSELoss()

    # 4. Loop
    epochs = 20
    for epoch in range(epochs):
        total_loss = 0
        for measurements, rho_true in loader:
            measurements = measurements.to(device)
            rho_true = rho_true.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            rho_pred = model(measurements)
            
            # Calculate Loss (Split complex into real/imag for MSE)
            loss = criterion(rho_pred.real, rho_true.real) + \
                   criterion(rho_pred.imag, rho_true.imag)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.6f}")

    # 5. Save Model
    # --- FIX IS HERE ---
    # We changed "../outputs/..." to "outputs/..."
    save_path = "outputs/model_weights.pt"
    
    # Just in case the folder doesn't exist, we create it
    os.makedirs("outputs", exist_ok=True)
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()