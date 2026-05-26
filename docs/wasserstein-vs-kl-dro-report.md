# Wasserstein-DRO vs. KL-DRO Implementation Comparison

This document explains the current entropy-regularized Wasserstein-DRO implementation and compares it directly with the original $f$-divergence (e.g., L2, KL) DRO implementation previously used in the Prospect framework.

---

## 1. High-Level Mathematical Comparison

In Distributionally Robust Optimization (DRO), we seek to minimize the worst-case risk over an uncertainty set of distributions $q$ centered around the empirical (uniform) distribution $p = \frac{1}{n}$.

### The Original Approach ($f$-divergence / KL / L2)
The original Prospect implementation penalized the adversary using an $f$-divergence:
$$ \max_{q \in \Delta_n} \sum_{i=1}^n q_i \ell_i - \nu D_f(q \| p) $$

Because $f$-divergences (like KL or $\chi^2$) are **point-wise independent**, the penalty only depends on how much mass is assigned to point $i$ relative to the uniform distribution $1/n$. The adversary cannot move mass between data points based on their features; it simply upweights high-loss points and downweights low-loss points. This allows the inner problem to be solved efficiently using 1D sorting and the Pool Adjacent Violators (PAV) algorithm (Isotonic Regression).

### The Current Approach (Wasserstein-DRO)
The current implementation replaces the $f$-divergence with the Wasserstein distance:
$$ \max_{q \in \Delta_n} \sum_{i=1}^n q_i \ell_i - \nu W_C(q, p) $$

The Wasserstein distance is **geometry-aware**. It requires a pairwise cost matrix $C$, where $C_{ij}$ is the distance between data point $i$ and $j$. The adversary can "transport" probability mass from a low-loss data point to a nearby high-loss data point, constrained by the geometric distance between them. 

Because this couples all data points together, 1D sorting is no longer sufficient. Instead, we add entropy regularization ($-\nu \epsilon H(P)$) to smooth the problem, allowing us to find the optimal weights $q$ using Optimal Transport techniques (like Sinkhorn iterations or kernel formulations).

---

## 2. Codebase Swaps and Correspondences

Here is exactly what parts of the codebase correspond to the original implementation and what was swapped out to support Wasserstein-DRO.

### A. Initialization & Distance Geometry
**Original (KL/L2-DRO)**
- No geometric information about the data is needed. The `Objective` class (`src/optim/objectives.py`) only requires the input features `X` to compute the loss $\ell$.

**Swapped to (Wasserstein-DRO)**
- The model must know the distances between all data points.
- **Modification**: In `Objective.__init__`, we now construct an $n \times n$ cost matrix `self.C` using either Euclidean distance or Cosine distance. We also precompute the transport kernel `self.K = exp(-self.C / epsilon)` to dramatically speed up future computations.

### B. The Inner Solver (Computing the Adversarial Weights $q$)
This is the core of the Prospect algorithm: computing the dual weights $q$ at each step.

**Original (KL/L2-DRO)**
- **Function**: `get_smooth_weights` and `get_smooth_weights_sorted` in `src/optim/smoothing.py`.
- **Mechanism**: Sorts the losses, then applies Numba-accelerated 1D Isotonic Regression (PAV) to find the optimal $q$.

**Swapped to (Wasserstein-DRO)**
- **Function**: `get_wasserstein_weights` in `src/optim/smoothing.py`.
- **Mechanism**: Replaces isotonic regression with an $N \times N$ kernel-based Optimal Transport formulation: $q = \frac{1}{n} v \odot (K z)$. Instead of sorting, we exponentiate the losses $v = \exp(\ell / \nu \epsilon)$ and perform two dense matrix-vector multiplications with the precomputed kernel $K$.

### C. Calculating the Total Risk
When evaluating the model (e.g., computing validation loss or logging training loss), we compute the worst-case risk.

**Original (KL/L2-DRO)**
- **Implementation**: In `Objective.get_batch_loss`, the risk is computed by finding $q$ via isotonic regression and calculating the dot product $\sum q_i \ell_i$, minus the specific $f$-divergence penalty term.

**Swapped to (Wasserstein-DRO)**
- **Implementation**: The dual of the entropy-regularized Wasserstein problem has a closed-form reduction using the Log-Sum-Exp function over the kernel matrix. 
- **Code Swap**: In `Objective.get_batch_loss`, instead of calling an inner solver, we directly compute the risk using the stabilized formula: 
  `risk = (nu * epsilon / n) * torch.sum(log_sum_exp + math.log(n))`
  This exactly matches the theoretical dual objective of entropy-regularized Wasserstein-DRO.

### D. Optimizer Updates (`Prospect` class)
At every step of coordinate/stochastic gradient descent, the optimizer updates the adversarial weights.

**Original (KL/L2-DRO)**
- In `prospect.py`, the `step()` method calls `self.lam = get_smooth_weights(...)` to update the dual weights (where `lam` represents $q$).

**Swapped to (Wasserstein-DRO)**
- In `step()`, the exact same variables are updated, but the call is routed to `self.lam = get_wasserstein_weights(..., K=self.K)`. 
- **Correspondence**: `self.lam` serves the exact same role in both algorithms—it acts as the worst-case sampling weights $q$ used to weight the gradients $\sum q_i \nabla \ell_i$. The rest of the `Prospect` optimizer (maintaining the running subgradient, proximal updates) remains completely mathematically identical.

---

## 3. Summary

| Component | Original KL/L2-DRO | Current Wasserstein-DRO |
| :--- | :--- | :--- |
| **Geometry** | Point-wise independent | Geometry-aware (Euclidean/Cosine) |
| **Matrix Memory** | None | Requires $n \times n$ Cost ($C$) and Kernel ($K$) matrices |
| **Inner Solver** | 1D Isotonic Regression (PAV algorithm) | Sinkhorn-like Kernel Matrix-Vector Product |
| **Risk Evaluation** | $q^T \ell - \text{penalty}$ | Log-Sum-Exp over the cost matrix |
| **Gradients/Optimizer** | $\sum q_i \nabla \ell_i$ | $\sum q_i \nabla \ell_i$ (Identical framework, different $q$) |
