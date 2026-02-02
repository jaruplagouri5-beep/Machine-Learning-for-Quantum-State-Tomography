import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub

class DensityMatrixReconstructor(nn.Module):
    def __init__(self, input_dim=16, dim=4):
        super().__init__()
        self.dim = dim 
        
        # Quantization Stubs (Required for Track 2)
        self.quant = QuantStub()   # Converts Float input -> Int8
        self.dequant = DeQuantStub() # Converts Int8 output -> Float
        
        # The Neural Network (Same as before)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, dim * dim * 2) 
        )

    def forward(self, x):
        # 1. Quantize Input (Simulating Hardware Entry)
        x = self.quant(x)
        
        # 2. Run Network (Integer Operations)
        out = self.net(x)
        
        # 3. De-Quantize Output (Simulating Hardware Exit)
        out = self.dequant(out)
        
        # 4. Physical Constraints (Post-Processing)
        # Note: Cholesky reconstruction usually happens on the host CPU, 
        # so we do this part in Float32 after dequantizing.
        batch_size = x.size(0)
        out = out.view(batch_size, self.dim, self.dim, 2)
        L_real = out[..., 0]
        L_imag = out[..., 1]
        L = torch.complex(L_real, L_imag)
        L = torch.tril(L)
        L_dagger = torch.transpose(L.conj(), 1, 2)
        rho_unnorm = torch.matmul(L, L_dagger)
        tr = torch.diagonal(rho_unnorm, dim1=-2, dim2=-1).sum(-1)
        tr = tr.view(batch_size, 1, 1)
        rho = rho_unnorm / (tr + 1e-9)
        
        return rho