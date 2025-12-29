# Evaluation of Quantum Machine Learning on Binarized MNIST

This project benchmarks **Quantum Machine Learning (QML)** against classical neural networks using the **binarized MNIST dataset**. It explores the trade-offs between model complexity and parameter efficiency in hybrid architectures.

## Key Results

## Key Results

* **Hybrid Efficiency (HQNN):**  
  The final hybrid model reached a test accuracy of **0.9765**, closely trailing the high-accuracy classical CNN benchmark of **0.9877**. Performance was stabilized by adding a **0.15 Dropout** layer before the VQC to prevent overfitting.

* **Standalone QNN Optimization:**  
  Standalone quantum performance was pushed from **0.7075** to **0.9351**. This was achieved by expanding the circuit to **10 qubits and 4 layers** and implementing a non-linear **BetterDownscale MLP** for optimized feature mapping.

* **Parameter Economy:**  
  The core **Variational Quantum Circuit (VQC)** performs high-level classification using only **298 quantum parameters**. While the total HQNN uses **55,434 parameters**, the majority are utilized by the classical front-end for initial data compression from **784 pixels to 8 features**.


## Benchmarks

| Model | Architecture | Accuracy | Params |
| :--- | :--- | :--- | :--- |
| **CNN** | Convolutional Baseline | **0.9877** | 220,234 |
| **HQNN** | Hybrid CNN + VQC | **0.9765** | 55,434 |
| **FC MLP** | 3-Layer Dense | **0.9691** | 109,386 |
| **QNN** | Optimized Standalone | **0.9351** | 298 |
| **QCNN** | Quantum Convolutional | **0.9387** | 196 |

## Future Work

* **Scalability:** Implementation of **quantum kernels** for direct high-dimensional data encoding.
* **Expressivity:** Developing "smart" local entanglement to mitigate **Barren Plateaus**.
* **Robustness:** Validating models against **Shot Noise** on quantum simulators.

## Files

* **`baseline_cnn_mnist.ipynb`**: High-accuracy classical CNN benchmark implementation.
* **`baseline_dense_mnist.ipynb`**: Implementation of the fully-connected MLP baseline.
* **`qml_mnist_exp_1.ipynb`**: Documentation of initial baseline QML models before optimization.
* **`qml_mnist_exp_2.ipynb`**: Optimized QML experiments featuring HQNN and QCNN improvements.
* **`combined_accuracy_exp1_adjusted.png`**: Plot visualizing initial benchmarking phase and information bottlenecks.
* **`combined_accuracy_exp2.png`**: Plot illustrating the impact of the architectural overhaul on model accuracy.