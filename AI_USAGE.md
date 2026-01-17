# AI Attribution & Usage Declaration

## 1. AI Tools Used
* **Model:** Gemini / ChatGPT
* **Role:** Learning Companion & Debugging Assistant

## 2. Usage Description (Learning Approach)
I utilized AI primarily as a **tutor and guide** to deepen my understanding of the assignment concepts while implementing the solution myself.
* **Conceptual Clarity:** I used AI to explain the mathematical connection between "Cholesky Decomposition" and physical quantum constraints (Trace=1, Positive Semi-Definite).
* **Architecture Selection:** I discussed with AI why an MLP (Multi-Layer Perceptron) is more suitable for the "Hardware-Centric" track compared to other models like RNNs.
* **Code Structure:** I asked AI for templates to structure my PyTorch classes efficiently, then filled in the logic based on my understanding.
* **Debugging:** Instead of simply asking for fixes, I used AI to explain the meaning of error messages (e.g., matrix dimension mismatches) so I could resolve them and learn.

## 3. List of Prompts Used
My prompts focused on "How" and "Why" rather than just "Do this":
1. "Explain why we need Cholesky decomposition for density matrix reconstruction."
2. "How does a standard MLP architecture map to hardware components like MAC units?"
3. "Guide me on how to generate random quantum states using QuTiP for training data."
4. "Explain the PyTorch error 'size mismatch' in simple terms."
5. "What are the best metrics to evaluate Quantum State Tomography?"

## 4. Verification Process
To ensure I understood and verified the AI-assisted components:
* **Manual Logic Check:** I cross-referenced the mathematical formulas provided by AI with standard quantum physics resources.
* **Metric Validation:** I ran the evaluation script myself and analyzed the Fidelity scores (>97%) to confirm the model was actually learning patterns and not just overfitting.
* **Hardware Feasibility:** I personally verified the inference latency results to ensure they aligned with the hardware track requirements.