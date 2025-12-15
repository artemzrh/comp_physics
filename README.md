This repository contains research-oriented implementations spanning 
**Physics-Informed Machine Learning, Strongly Correlated Electron Systems, and Quantum Machine Learning**.

The work connects classical numerical methods, physics-informed neural networks (PINNs),
many-body techniques (NRG, Slave-Boson mean-field), and modern quantum machine learning (QML),
with a focus on **variational principles, optimization challenges, and physical inductive biases**.

In particular, the repository includes work on:

- **Physics-Informed Neural Networks (PINNs)**
  - Solving ODEs and PDEs by embedding physical laws into the loss function
  - Analysis of spectral bias, convergence issues, and loss landscape complexity
  - Robustness to noise and mesh-free formulations

- **Numerical Renormalization Group (NRG)**
  - Treatment of quantum impurity models (e.g. Kondo, Anderson models)
  - Low-energy fixed points and scale separation
  - Benchmark systems relevant for quantum simulation

- **Slave-Boson Mean-Field Theory**
  - Variational treatment of strongly correlated electron systems
  - Saddle-point optimization with auxiliary particles
  - Conceptual parallels to variational quantum eigensolvers (VQE)

- **Quantum Machine Learning (QML)**
  - Variational quantum circuits and hybrid quantum–classical models
  - Quantum feature maps and supervised learning tasks
  - Study of optimization behavior and expressivity in the NISQ regime

A central theme of this repository is the connection between classical and quantum variational methods:

- Slave-boson mean-field theory and VQE share a variational ansatz structure
- PINNs and QML both suffer from complex loss landscapes and optimization pathologies
- Strongly correlated Hamiltonians (Hubbard, Anderson, Kondo) are key targets for near-term quantum simulations

These connections motivate a unified perspective on learning and optimization in physics-based models, both classical and quantum.
