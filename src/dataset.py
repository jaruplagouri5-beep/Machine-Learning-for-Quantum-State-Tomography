import numpy as np
import torch
from torch.utils.data import Dataset
import qutip as qt

class QuantumDataset(Dataset):
    def __init__(self, num_samples=1000, num_qubits=2):
        self.num_samples = num_samples
        self.num_qubits = num_qubits
        self.dim = 2**num_qubits
        
        # 1. Generate Data
        self.measurements, self.densities = self.generate_data()

    def generate_data(self):
        measurements_list = []
        densities_list = []

        print(f"Generating {self.num_samples} quantum states...")
        
        # Define Pauli matrices
        paulis = [qt.qeye(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]
        observables = []
        
        # Create 2-qubit observables
        for p1 in paulis:
            for p2 in paulis:
                observables.append(qt.tensor(p1, p2))
        
        # Correct dimensions for 2 qubits: [[2, 2], [2, 2]]
        dims = [[2] * self.num_qubits, [2] * self.num_qubits]

        for _ in range(self.num_samples):
            # --- FIX IS HERE ---
            # 1. Create the matrix first (without passing dims argument)
            rho = qt.rand_dm(self.dim)
            
            # 2. Manually force the dimensions to be [2, 2] instead of [4]
            # This fixes the "Incompatible dimensions" error
            rho.dims = dims
            
            rho_np = rho.full()
            
            # B. Simulate Measurements
            meas_vector = []
            for obs in observables:
                val = qt.expect(obs, rho)
                meas_vector.append(val)
            
            measurements_list.append(np.array(meas_vector, dtype=np.float32))
            
            # C. Store true Density Matrix
            densities_list.append(rho_np.astype(np.complex64))

        return np.array(measurements_list), np.array(densities_list)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.measurements[idx], self.densities[idx]