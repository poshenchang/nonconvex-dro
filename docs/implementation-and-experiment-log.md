# Implementation and Experiment Log

This document provides a detailed breakdown of the modifications made to the codebase for implementing the Wasserstein Sinkhorn solver and outlines the experimental evaluations performed.

---

## 1. File Modifications and Purposes

| File Path | Description of Changes | Purpose |
| :--- | :--- | :--- |
| **`prospect/src/optim/smoothing.py`** | <ul><li>Added the `sinkhorn_knopp` function to compute the optimal transport plan and distance iteratively.</li><li>Rewrote `get_wasserstein_weights` to compute adversarial weights using a mathematically equivalent kernel-based formula: $q = \frac{1}{n} v \odot (K z)$ with log-sum-exp numerical stabilization.</li></ul> | Resolve a severe performance bottleneck ($30\times$ speedup) by avoiding $O(n^2)$ exponentiations at every coordinate update. |
| **`prospect/src/optim/objectives.py`** | <ul><li>Constructed the pairwise cost matrix $C$ using `euclidean` or `cosine` metrics.</li><li>Precomputed the kernel matrix $K = \exp(-C / \epsilon)$ during objective initialization.</li><li>Rewrote `get_batch_loss` using the optimized kernel formulation.</li><li>Updated `get_batch_subgrad` methods to slice the precomputed kernel matrix and pass it to the weight solver.</li></ul> | Support selectable distance metrics and reuse the precomputed kernel matrix to completely avoid redundant operations. |
| **`prospect/src/optim/prospect.py`** | <ul><li>Updated constructors for `Prospect` and `ProspectMoreau` to extract the cost matrix `C` and precompute the kernel `K` (or reuse them from the objective).</li><li>Passed the precomputed kernel `K` to `get_wasserstein_weights` in both initialization and update steps.</li></ul> | Prevent $O(n^2)$ kernel reconstruction overhead at each step of coordinate descent optimization. |
| **`prospect/src/utils/training.py`** | <ul><li>Updated `get_objective` to parse the `distance_metric` from configuration dictionaries and pass it to the `Objective` constructor.</li></ul> | Integrate the new distance metric options into the training initialization pipeline. |
| **`prospect/scripts/train.py`** | <ul><li>Added `--distance_metric` and `--penalty` command-line arguments.</li><li>Defined `n_class = None` under the `diabetes` dataset branch to avoid undefined variable exceptions.</li></ul> | Expose regularization choices via the command-line interface. |
| **`prospect/scripts/download_diabetes.py`** | <ul><li>Corrected the dataset download target path from `../data/` to `data/`.</li><li>Fixed a syntax typo where a set `{"?", "Unknown"}` was used instead of a dictionary `{"?": "Unknown"}`.</li><li>Cast integer columns (`admission_source_id`, `discharge_disposition_id`) to `object` to avoid pandas assignment errors.</li></ul> | Allow the dataset downloading and preprocessing pipeline to execute successfully. |

---

## 2. Experimental Setup and Results

### Benchmarks
We evaluated the framework on the **diabetes** dataset ($N = 4000$ training samples, $31$ features, binary classification, ESRM objective) across three regularization penalties over $16$ epochs with `--epoch_len 100`.

- **L2 Regularization Baseline**
- **Wasserstein-Regularized (Euclidean Distance Metric)**
- **Wasserstein-Regularized (Cosine Distance Metric)**

### Results Format & Storage Location
The results for each configuration are saved under the `prospect/results/diabetes/` directory inside folders named after the specific hyperparameter values:

- **L2 Baseline Path**:
  `results/diabetes/distance_metric_euclidean_l2_reg_1.00e+00_loss_binary_cross_entropy_objective_esrm_penalty_l2_shift_cost_1.00e+00/prospect/`
- **Wasserstein-Euclidean Path**:
  `results/diabetes/distance_metric_euclidean_l2_reg_1.00e+00_loss_binary_cross_entropy_objective_esrm_penalty_wasserstein_shift_cost_1.00e+00/prospect/`
- **Wasserstein-Cosine Path**:
  `results/diabetes/distance_metric_cosine_l2_reg_1.00e+00_loss_binary_cross_entropy_objective_esrm_penalty_wasserstein_shift_cost_1.00e+00/prospect/`

Each results directory contains the following pickled files:
1. `best_cfg.p`: A Python dictionary representing the selected optimal hyperparameters (e.g. learning rate).
2. `best_weights.p`: A PyTorch Tensor containing the optimal model parameters.
3. `best_traj.p`: A pandas DataFrame containing metrics across all epochs, containing the columns:
   - `epoch`: Epoch number.
   - `average_train_loss`: Worst-case training loss across all training seeds.
   - `seed_1_train` / `seed_2_train`: Individual training split loss trajectories.
   - `seed_1_val` / `seed_2_val`: Individual validation split loss trajectories.
