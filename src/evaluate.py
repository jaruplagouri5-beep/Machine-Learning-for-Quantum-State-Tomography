import torch
import numpy as np
from dataset import QuantumDataset
from model import DensityMatrixReconstructor
from torch.utils.data import DataLoader
import qutip as qt
import time
import os

def calculate_metrics(rho_pred, rho_true):
    # Convert PyTorch tensors to Numpy complex matrices
    rho_pred_np = rho_pred.detach().cpu().numpy()
    rho_true_np = rho_true.detach().cpu().numpy()
    
    fidelities = []
    trace_dists = []
    
    for i in range(len(rho_pred_np)):
        # Convert to Qutip objects for easy math
        # Dimensions for 2 qubits are [2, 2]
        dims = [[2, 2], [2, 2]]
        
        dm_pred = qt.Qobj(rho_pred_np[i], dims=dims)
        dm_true = qt.Qobj(rho_true_np[i], dims=dims)
        
        # 1. Fidelity: F = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2
        fid = qt.fidelity(dm_pred, dm_true)**2
        fidelities.append(fid)
        
        # 2. Trace Distance: T = 0.5 * Tr|rho - sigma|
        tr_dist = qt.tracedist(dm_pred, dm_true)
        trace_dists.append(tr_dist)
        
    return np.mean(fidelities), np.mean(trace_dists)

def evaluate():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on: {device}")
    
    # 2. Load Data (Use a fresh batch for testing)
    print("Generating Test Data...")
    test_dataset = QuantumDataset(num_samples=500, num_qubits=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 3. Load Model
    model = DensityMatrixReconstructor(input_dim=16, dim=4).to(device)
    
    # --- PATH FIX: Check for weights in parent directory first, then local ---
    weights_path = "../outputs/model_weights.pt"
    if not os.path.exists(weights_path):
        weights_path = "outputs/model_weights.pt" # Fallback if not in src folder
        
    print(f"Loading weights from: {weights_path}")
    model.load_state_dict(torch.load(weights_path))
    model.eval() # Set to evaluation mode
    
    print("Starting Evaluation...")
    total_fidelity = 0
    total_tr_dist = 0
    total_batches = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for measurements, rho_true in test_loader:
            measurements = measurements.to(device)
            rho_true = rho_true.to(device)
            
            # Predict
            rho_pred = model(measurements)
            
            # Calculate Metrics
            fid, tr_dist = calculate_metrics(rho_pred, rho_true)
            
            total_fidelity += fid
            total_tr_dist += tr_dist
            total_batches += 1
            
    end_time = time.time()
    
    # 4. Final Report
    avg_fidelity = total_fidelity / total_batches
    avg_tr_dist = total_tr_dist / total_batches
    
    # --- LATENCY FIX: Use actual dataset length ---
    total_samples = len(test_dataset)
    avg_time_per_sample = (end_time - start_time) * 1000 / total_samples # in ms
    
    print("-" * 30)
    print("FINAL RESULTS")
    print("-" * 30)
    print(f"Mean Fidelity:       {avg_fidelity:.4f} (Target: > 0.90)")
    print(f"Mean Trace Distance: {avg_tr_dist:.4f} (Target: < 0.10)")
    print(f"Inference Latency:   {avg_time_per_sample:.2f} ms per state")
    print("-" * 30)

if __name__ == "__main__":
    evaluate()