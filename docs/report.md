# Phase 1: Environment Setup & Codebase Familiarization Report

## 1. Environment Setup Overview
- **Repository Setup**: The original `prospect` repository (Ronakdm/prospect) was cloned successfully into the workspace.
- **Environment Management**: Transitioned from the originally mandated `conda` setup to a native Python virtual environment (`venv`) inside the `.venv` directory, as requested. 
- **Dependencies**: Established the installation pipeline for essential packages including standard data science libraries (`pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `jupyterlab`, `numba`) and PyTorch for CUDA 11.8. 

## 2. Codebase Architecture & Familiarization
Our codebase review focused on understanding how the authors maintain stability during training and structure their experimentation, which will be essential when integrating our non-convex Wasserstein framework.

### `tutorial.ipynb`
- **Purpose**: Acts as an entry point for initializing and dispatching experiments.
- **Mechanism**: Demonstrates standard baseline configurations and mapping objective strings to actual optimizer classes. It provides a clear blueprint for how to feed synthetic and benchmark dataset structures into the models.

### `src/utils/training.py`
- **Optimizer Integration**: Serves as the factory layer for baselines and proposed methods. It maps specific configurations to target classes:
  - Baselines: `StochasticSubgradientMethod` (SGD), `StochasticRegularizedDualAveraging`, `SmoothedLSVRG`, `SaddleSAGA`
  - Core Methods: `Prospect` and `ProspectMoreau`
- **Objective Matching**: Matches target spectral risk measures (e.g., ERM, Extremile, Superquantile, ESRM) via the `get_objective` binding functions.
- **Training Structure**: The core loop (`train_model()`) controls step progression and epoch management. Crucially, calling `optimizer.step()` triggers the underlying loss table tracking and variance reduction logic established in the `Prospect` and `ProspectMoreau` cores. 

## 3. Implications for Next Phases
With the codebase structure now clear, several key integration points have been identified for Phase 2 & 3:
1. **Inner Solver Replacement**: The core optimizer (`Prospect`) handles the adversarial distribution (originally computed exactly via Numba-accelerated PAV for $f$-divergences). This module will be targeted and replaced with the new Wasserstein inner solver.
2. **End-to-End Backprop Integration**: The `train_model` step and underlying PyTorch tensor state management in `src/utils/training.py` will serve as the foundation to reconfigure how backpropagation propagates through the full depth of non-frozen neural networks, shifting away from their linear probe paradigm.

## 4. Phase 3: Algorithmic Modifications (Inner Solver)
- Overrided the `Prospect` and `ProspectMoreau` optimizer classes in `src/optim/prospect.py` to optionally calculate a static sequence cost matrix $C$ depending on the specified penalty parameter. This computes the $L_2$ Euclidean distance across the raw `objective.X` features.
- Implemented `get_wasserstein_weights` in `src/optim/smoothing.py` as a closed-form differentiable replacement for the Numba PAV solver. By recognizing that the target distribution relies entirely on the uniform prior's marginal constraint, the new adversarial weights act essentially as a Softmax mapping scaled by Sinkhorn's entropy parameter $\epsilon$. 
- Integrated and linked the conditional flag `penalty="wasserstein"` to enforce the usage of our Wasserstein functionality across the core step update functions.

The foundation is now set to commence **Phase 3: Part 2** and deep end-to-end backpropagation modifications.