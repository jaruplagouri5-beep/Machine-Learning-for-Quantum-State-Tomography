import torch
import torch.nn as nn
import os
import time
import numpy as np
from torch.utils.data import DataLoader
from dataset import QuantumDataset
from model import DensityMatrixReconstructor
from evaluate import calculate_metrics

def print_model_size(model, name):
    """Helper to check how much memory the model uses"""
    torch.save(model.state_dict(), "temp.p")
    size_kb = os.path.getsize("temp.p") / 1024
    os.remove("temp.p")
    print(f"📦 {name} Size: {size_kb:.2f} KB")

def run_quantization():
    print("⚙️ STARTING QUANTIZATION (TRACK 2)...")
    
    # --- CRITICAL FIX FOR MAC (M1/M2/M3) ---
    # This forces PyTorch to use the ARM-optimized backend (QNNPACK)
    # instead of the Intel backend (FBGEMM) which was causing your crash.
    torch.backends.quantized.engine = 'qnnpack'
    
    # 1. Load the Trained Float Model
    model_fp32 = DensityMatrixReconstructor()
    
    # Load weights (ensure mapping to CPU)
    weights_path = "outputs/model_weights.pt"
    
    if not os.path.exists(weights_path):
        print(f"❌ Error: Could not find {weights_path}. Please run 'python3 src/train.py' first!")
        return

    # Load the trained weights
    model_fp32.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model_fp32.eval()
    print("✅ Loaded Float32 Model")

    # 2. Configure Quantization
    # We use 'qnnpack' here to match the engine we set above
    model_fp32.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    
    # 3. Prepare (Fuse Layers for Speed)
    # Fusing Linear -> ReLU saves memory and is standard for hardware deployment
    torch.quantization.fuse_modules(model_fp32.net, [['0', '1'], ['2', '3']], inplace=True)
    model_int8 = torch.quantization.prepare(model_fp32)
    
    # 4. Calibrate
    # Run sample data through so the model learns the range of values (min/max)
    print("🔄 Calibrating with representative data...")
    dataset = QuantumDataset(num_samples=100, num_qubits=2)
    loader = DataLoader(dataset, batch_size=32)
    
    with torch.no_grad():
        for measurements, _ in loader:
            model_int8(measurements) 

    # 5. Convert to Int8
    model_int8 = torch.quantization.convert(model_int8)
    print("✅ QUANTIZATION COMPLETE (Float32 -> Int8)")

    # 6. Save Quantized Model
    # We use TorchScript because quantized models need strict structure
    save_path = "outputs/model_quantized.pt"
    traced_model = torch.jit.trace(model_int8, torch.randn(1, 16))
    torch.jit.save(traced_model, save_path)
    print(f"💾 Saved to {save_path}")

    # 7. Compare Performance (Size & Accuracy)
    print("\n⚖️ COMPARISON REPORT:")
    print_model_size(model_fp32, "Original (Float32)")
    print_model_size(model_int8, "Quantized (Int8)")
    
    # Evaluate Fidelity
    print("\n🔍 Evaluating Quantized Accuracy...")
    test_dataset = QuantumDataset(num_samples=200, num_qubits=2)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    fidelities = []
    
    with torch.no_grad():
        for measurements, rho_true in test_loader:
            # Quantized model runs on CPU
            rho_pred = model_int8(measurements) 
            fid, _ = calculate_metrics(rho_pred, rho_true)
            fidelities.append(fid)
            
    avg_fid = np.mean(fidelities)
    print(f"🎯 Mean Fidelity (Quantized): {avg_fid:.4f}")
    
    if avg_fid > 0.85:
        print("✅ SUCCESS: Model works on simulated low-precision hardware!")
    else:
        print("⚠️ WARNING: Accuracy dropped. You might need to retrain.")

if __name__ == "__main__":
    run_quantization()