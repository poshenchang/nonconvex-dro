## Phase 1: Environment Setup & Code Familiarization
Before making any algorithmic changes, you need to establish the baseline provided by the authors. 
*   [x] **Set up the environment:** Clone the repository and build the `dro` environment (using `venv` instead of conda). Manually install PyTorch matching the CUDA distribution, along with data science dependencies (pandas, tqdm, numba, etc.). Ensure your hardware meets the recommended minimums: 32GB of CPU RAM and a GPU with at least 12GB of VRAM.
*   [x] **Run the tutorials:** Walk through `tutorial.ipynb` to understand the codebase structure and how the experiments are loaded.
*   [x] **Analyze the core implementation:** Review the code in the `src` folder (specifically `src/utils/training.py`) to understand exactly how the authors maintain the running loss tables and gradient control variates for Prospect's bias and variance reduction.

## Phase 2: Theoretical Formulation (Skipped for now)
*   **Define the objective:** Formalize your objective using a deep neural network (a non-convex loss function) and replace the paper's spectral risk measures with a **Wasserstein uncertainty set**.
*   **Adapt the convergence proofs:** Use the "regular subdifferential" to handle the non-convexity of the neural network's loss landscape, directly answering the authors' call for future work. You will need to show that the bias and variance reduction guarantees still hold without the strict convex, $G$-Lipschitz, and $L$-smooth assumptions of the original paper.

## Phase 3: Algorithmic Modification
This is where you will heavily modify the existing codebase.
*   [x] **Replace the inner solver:** In the original codebase, the inner maximization problem (finding the most adversarial distribution) is solved exactly in $O(n)$ time using the Pool Adjacent Violators (PAV) algorithm, which is accelerated using the Numba library. Because PAV is specific to $f$-divergences and spectral risk measures, you will need to implement a new inner solver specifically designed to compute the Wasserstein penalty.
*   **Enable end-to-end backpropagation:** The original repository bypasses non-convexity on complex tasks by using *frozen* deep representations (like a fine-tuned BERT for text or a pretrained ResNet50 for images) and only training a linear probe classifier on top. You will need to modify the training loop to utilize PyTorch's full `autograd` capabilities to update the entire neural network's weights end-to-end.

## Phase 4: Empirical Evaluation
*   **Start with sanity checks:** Use the automatically downloaded tabular datasets (like `yacht`, `energy`, or `concrete`) to ensure your new Wasserstein solver is functioning correctly and reducing the training loss.
*   **Run distribution shift benchmarks:** Move to the WILDS datasets included in the repo. You can prepare the `iwildcam` data or fine-tune the BERT model using the provided `download_amazon.ipynb` notebook. 
*   **Compare against baselines:** Evaluate your non-convex Prospect variant against the provided baselines (SGD, SRDA, LSVRG, and SaddleSAGA) by adapting the `amazon.ipynb` and `iwildcam.ipynb` experiment notebooks.