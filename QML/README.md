# Evaluation of Quantum Machine Learning on Binarized MNIST

This project benchmarks **Quantum Machine Learning (QML)** against classical neural networks using the **binarized MNIST dataset**. It explores the trade-offs between model complexity and parameter efficiency in hybrid architectures.

## Key Results

* **Hybrid Efficiency:** The **HQNN** model achieved a test accuracy of **0.9765**, closely trailing the classical CNN baseline of **0.9877**.
* **Parameter Economy:** The variational quantum circuit (VQC) classified digits using only **298 quantum parameters**, whereas the classical CNN required **220,234 parameters**.
* **Performance Gain:** Architectural optimizations, including non-linear feature mapping (BetterDownscale MLP) and dropout (0.15), raised standalone QML accuracy from **0.7075 to 0.9351**.

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