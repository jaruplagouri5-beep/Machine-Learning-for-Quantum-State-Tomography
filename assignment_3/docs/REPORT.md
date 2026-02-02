# Scalable Neural Quantum Tomography

## Overview
This project implements a scalable Neural Quantum Tomography pipeline using a Deep Neural Network (DNN) surrogate.
The model reconstructs quantum density matrices from Pauli measurement statistics while enforcing physical validity.
Scalability is benchmarked from 2 to 6 qubits, highlighting both architectural efficiency and fundamental limits.

---

## 1. Executive Summary
We implemented and evaluated a neural-network–based quantum state tomography framework that:
- Reconstructs valid density matrices from Pauli expectation values
- Scales from N = 2 to N = 6 qubits
- Uses a Cholesky-decomposition–based architecture to guarantee physicality

The study reveals severe exponential scaling bottlenecks, making full tomography infeasible for larger quantum systems.

---

## 2. Methodology & Serialization

### Architecture
- Model: Deep Neural Network (DNN)
- Constraint: Physical validity of the density matrix  
  ρ ≥ 0
- Technique: Cholesky decomposition–based parametrization

This guarantees:
- Positive semi-definiteness
- Hermiticity
- Valid quantum states after normalization

---

### Serialization Strategy
A hybrid serialization strategy was used:

- pickle: Lightweight metadata and configuration dictionaries
- torch.save: Neural network weights and large tensors

Checkpoint structure:

 models/

├── model_<track><nqubits>.pkl

└── model<track>_<nqubits>_weights.pt


This approach balances reproducibility with performance and follows standard PyTorch practices.

---

## 3. Findings & Scalability Limits

### Curse of Dimensionality

#### Parameter Explosion
- Input dimension scales as 4^N (Pauli basis)
- For N = 6:
  4^6 = 4096 input features
- Parameter count and memory usage quadruple with every added qubit

---

#### Runtime Wall
- Inference time grows exponentially with qubit count
- Negligible latency for N = 2
- Significant slowdown observed at N = 6

---

#### Physical Measurement Bottleneck
- Full Pauli tomography requires 4^N measurement settings
- For N > 10, this exceeds 1 million circuit executions
- This makes the approach impractical for near-term quantum hardware

---

## 4. Ablation Study: Network Depth

We analyzed the impact of network depth from 1 to 16 layers.

Observations:
- Inference speed increases linearly with depth
- Deeper networks (16 layers) show negligible fidelity improvement
- 16-layer models are approximately 3× slower than shallow (2-layer) networks

Conclusion:
Shallow networks are optimal for density matrix reconstruction in this task.

---

## 5. Future Directions

### Classical Shadows
- Replace full tomography with Classical Shadows
- Reduce sample complexity from exponential O(4^N) to polynomial
- Neural networks would process measurement snapshots instead of full expectation vectors

---

### Hardware Validation
- Current simulations assume Gaussian noise
- Future validation on real quantum hardware is required
- Robustness must be tested against:
  - Readout errors
  - Coherent drift
  - Systematic noise
- Candidate platforms include Qiskit and Rigetti

---

### Mutually Unbiased Bases (MUBs)
- Replace Pauli measurements with MUBs
- For prime-power dimensions, input dimensionality can be reduced
- This partially alleviates the exponential input bottleneck

---

## Conclusion
Neural networks provide a powerful surrogate for quantum state reconstruction.
However, full neural quantum tomography is fundamentally limited by exponential scaling.
Scalable alternatives such as Classical Shadows are essential for extending these methods to larger quantum systems.
