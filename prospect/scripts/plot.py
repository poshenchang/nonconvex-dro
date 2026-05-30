import os
import argparse
import pickle
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Plot logarithmic training loss curves for different learning rates.")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True, 
        help="Path to the results directory (e.g., prospect/results/diabetes/...)"
    )
    return parser.parse_args()

def robust_load(filepath):
    """Attempt to load using standard pickle first to avoid PyTorch weights_only warnings."""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception:
        try:
            import torch
            return torch.load(filepath, map_location='cpu', weights_only=False)
        except Exception:
            return None

def extract_metadata(filepath, result_dict):
    """Extract learning rate, seed, and training loss from the loaded dictionary and filepath."""
    if not isinstance(result_dict, dict) or "metrics" not in result_dict:
        return None, None, None

    metrics_data = result_dict["metrics"]
    train_loss = None

    # Handle both pandas DataFrame and raw dictionary formats safely
    if isinstance(metrics_data, pd.DataFrame) and "train_loss" in metrics_data.columns:
        train_loss = metrics_data["train_loss"].tolist()
    elif isinstance(metrics_data, dict) and "train_loss" in metrics_data:
        train_loss = metrics_data["train_loss"]

    if train_loss is None:
        return None, None, None

    # Ensure all loss elements are standard floats
    try:
        train_loss = [float(v.item() if hasattr(v, 'item') else v) for v in train_loss]
    except Exception:
        return None, None, None

    # Extract seed from filename (e.g., seed_1.p -> 1)
    filename = filepath.name
    seed_match = re.search(r'seed_?(\d+)', filename)
    seed = int(seed_match.group(1)) if seed_match else None

    # Extract learning rate (LR) from the full path string (parent directory name)
    path_str = str(filepath)
    lr_match = re.search(r'lr_([0-9eE\.\-\+]+)', path_str)
    lr = None
    if lr_match:
        try:
            lr = float(lr_match.group(1))
        except ValueError:
            pass

    return lr, seed, train_loss

def main():
    args = parse_args()
    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"Error: Directory '{input_dir}' does not exist.")
        return

    # Dictionary to hold data dynamically: data[seed][lr] = [loss_values...]
    data = {}
    
    # Locate all checkpoint files, ignoring top-level summary checkpoints
    file_extensions = ["*.p", "*.pt", "*.pkl"]
    files = []
    for ext in file_extensions:
        files.extend(list(input_dir.rglob(ext)))
    files = [f for f in files if "nb_checkpoints" not in f.name and "best_" not in f.name]

    print(f"Scanning {len(files)} files in directory...")

    # Process and extract metrics from files
    for filepath in files:
        result_dict = robust_load(filepath)
        if result_dict is None:
            continue
            
        lr, seed, train_loss = extract_metadata(filepath, result_dict)
        
        if lr is not None and seed is not None and train_loss is not None:
            if seed not in data:
                data[seed] = {}
            data[seed][lr] = np.array(train_loss)

    # Validate if any metrics were recovered
    all_lrs = set()
    for seed in data:
        all_lrs.update(data[seed].keys())

    if not all_lrs:
        print("Warning: No valid training data found in the specified directory.")
        print("Please verify that the directory contains 'seed_X.p' files inside hyperparameter subfolders.")
        return

    best_lr = None
    best_loss = float('inf')
    convergent_lrs = set()
    divergent_lrs = set()

    for lr in all_lrs:
        final_losses = []
        is_divergent = False

        for seed in data:
            if lr in data[seed]:
                loss_curve = data[seed][lr]
                # Identify divergence via NaNs, Infs, or massive final loss thresholds
                if np.isnan(loss_curve).any() or np.isinf(loss_curve).any() or (len(loss_curve) > 0 and loss_curve[-1] > 1e4):
                    is_divergent = True
                elif len(loss_curve) > 0:
                    final_losses.append(loss_curve[-1])

        if is_divergent or len(final_losses) == 0:
            divergent_lrs.add(lr)
        else:
            convergent_lrs.add(lr)
            avg_final_loss = np.mean(final_losses)
            if avg_final_loss < best_loss:
                best_loss = avg_final_loss
                best_lr = lr

    # Terminal Summary Output
    print("\n" + "=" * 60)
    print("TRAINING CURVE ANALYSIS SUMMARY")
    print("=" * 60)
    if best_lr is not None:
        print(f"[+] Best Learning Rate : {best_lr} (Final Avg Loss: {best_loss:.6f})")
    else:
        print("[-] Best Learning Rate : None (All runs diverged or missing)")
        
    print(f"[+] Convergent LRs     : {sorted(list(convergent_lrs))}")
    print(f"[-] Divergent LRs      : {sorted(list(divergent_lrs))}")
    print("=" * 60 + "\n")

    # Setup plotting configuration
    plot_path = input_dir / "plot.png"

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle(f"Training Loss Curves", fontsize=16)

    # Assign distinct colors to each unique learning rate
    sorted_lrs = sorted(list(all_lrs))
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, len(sorted_lrs))]
    lr_to_color = dict(zip(sorted_lrs, colors))

    # Plot both requested seed subplots (Seed 1 and Seed 2)
    for idx, seed in enumerate([1, 2]):
        ax = axes[idx]
        ax.set_title(f"Seed {seed}", fontsize=14)
        ax.set_xlabel("Epoch", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Log Average Train Loss", fontsize=12)

        ax.set_yscale('log')
        ax.grid(True, which="both", ls="--", alpha=0.4)

        if seed_data := data.get(seed):
            for lr in sorted_lrs:
                if lr in seed_data:
                    loss_curve = seed_data[lr]
                    linestyle = '-' if lr in convergent_lrs else ':'
                    alpha_val = 0.9 if lr in convergent_lrs else 0.4
                    
                    ax.plot(
                        loss_curve, 
                        label=f"lr={lr}", 
                        color=lr_to_color[lr], 
                        linewidth=2,
                        linestyle=linestyle,
                        alpha=alpha_val
                    )
            ax.legend(loc='upper right', fontsize=9, ncol=2)
        else:
            ax.text(0.5, 0.5, f"No Data for Seed {seed}", ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[*] Success! Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()