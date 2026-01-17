 Quantum State Reconstruction  
Track 2: Hardware-Centric



📌 Project Overview
This project implements a Neural Network-based Quantum State Tomography (QST) system.  
The objective is to reconstruct a physically valid 2-qubit density matrix ($\rho$) from Pauli measurement data.

The project follows Track 2 (Hardware-Centric), emphasizing:
- Deterministic neural architectures
- Low-precision inference
- Hardware feasibility using Verilog simulation

The model is optimized for edge deployment (FPGA/ASIC) using Int8 quantization and validated through cycle-accurate hardware simulation.



 Model Working (Architecture & Mathematics)

 1. Quantum Constraint Enforcement
A valid density matrix must satisfy:
- Hermitian property
- Positive Semi-Definiteness (PSD)
- Unit trace

To guarantee these constraints, the model predicts a lower-triangular matrix $L$, and reconstructs the density matrix using Cholesky Decomposition:

\[
\rho = \frac{L L^\dagger}{\text{Tr}(L L^\dagger)}
\]

This formulation ensures that the output is always a physically valid quantum state.

---

 2. Neural Architecture
- Model Type: Multi-Layer Perceptron (MLP)
- Reasoning:
  - Fixed computational graph
  - Deterministic latency
  - Hardware-friendly matrix–vector operations
- Output: Parameters of matrix $L$ for density matrix reconstruction



 3. Hardware-Centric Optimization
To simulate realistic hardware deployment:
- Post-Training Static Quantization was applied
- Model converted from Float32 → Int8
- Achieved significant memory reduction with negligible accuracy loss

This makes the model suitable for low-power and resource-constrained platforms.



 🛠️ Replication Guide (How to Run)

 1. Environment Setup
Ensure Python 3.8+ is installed.

Install required dependencies:
```bash
pip install torch numpy qutip

Install Verilog simulator (Mac/Linux):
brew install icarus-verilog

2. Train the Model (Software)

Generates 2,000 synthetic quantum states and trains the MLP.

python3 src/train.py

 Output:
outputs/model_weights.pt

3. Quantize the Model (Hardware Optimization)

Converts the trained model to Int8 precision.

python3 src/quantize.py

Output:

outputs/model_quantized.pt

4. Hardware Simulation (Verilog)

A hardware-oriented neuron implementing Int8 Multiply–Accumulate (MAC) logic is simulated.

Compile the hardware design:

iverilog -o outputs/hardware_sim src/mlp_hardware.v src/testbench.v

Run the simulation:

vvp outputs/hardware_sim


Generated File:

outputs/simulation.vcd

This waveform validates:

-->Fixed-point arithmetic correctness

-->Deterministic signal transitions

-->Hardware feasibility of the quantized model

5. Evaluate Performance

Evaluates the model on unseen quantum states.

python3 src/evaluate.py

📊 Performance Results
🔹 Reconstruction Accuracy
Metric	         Value	          Target	Status
Mean Fidelity	  0.9774	       > 0.90	Passed
Trace Distance	   0.0799         	< 0.10	Passed
Inference Latency	0.52 ms / state   	—	Fast

🔹 Hardware Optimization Summary
Model Version	File Size	Precision
Original Model	55.61 KB	Float32
Quantized Model	18.81 KB	Int8
Size Reduction	~66%	     ✔

⚙️ Hardware Simulation Details

-->To validate hardware feasibility beyond software quantization:

-->Hardware Module: src/mlp_hardware.v
Implements Int8 MAC operations

-->Testbench: src/testbench.v
Drives deterministic test vectors

-->Waveform Output: outputs/simulation.vcd
Confirms correct digital signal switching

🤖 AI Usage Disclosure

AI tools were used strictly for assistance.

Tools Used:  Gemini

Use Cases:

Debugging quantization backend issues

Verilog boilerplate generation

Documentation refinement

Verification:

All AI-generated outputs were manually reviewed

Verified against PyTorch documentation

Validated using evaluation scripts

Full disclosure available in:

AI_USAGE.md

 Conclusion

This project presents a physics-aware, hardware-efficient approach to quantum state tomography.
By combining:

   -->Cholesky-based physical constraints

    --> Deterministic MLP architecture

   --> Int8 quantization

   -->Verilog-level simulation

the system demonstrates strong suitability for real-time, low-resource quantum hardware platforms.

structure

assignment_2/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── quantize.py
│   ├── mlp_hardware.v
│   └── testbench.v
├── outputs/
│   ├── model_weights.pt
│   ├── model_quantized.pt
│   └── simulation.vcd
├── docs/
│   └── model_explanation.md
├── AI_USAGE.md
├── requirements.txt
└── README.md
