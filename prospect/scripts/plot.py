"""
plot.py  —  Visualize training curves from results produced by train.py.

Usage:
    python plot.py --input_path results/diabetes/distance_metric_euclidean_...

Outputs (written to input_path/):
    <optimizer>.png   — per-LR curves for each optimizer, two subplots (seed 1 / seed 2)
    all.png           — best-config curve for every optimizer on one plot
"""

import os
import re
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_path",
    type=str,
    required=True,
    help="Path to the model-config result folder (contains per-LR and best-config subdirs).",
)
parser.add_argument(
    "--loss_key",
    type=str,
    default="train_loss",
    choices=["train_loss", "train_loss_unreg", "val_loss"],
    help="Which loss column to plot (default: train_loss).",
)
parser.add_argument(
    "--seeds",
    type=int,
    nargs="+",
    default=[1, 2],
    help="Seeds to plot as subplots (default: 1 2).",
)
args = parser.parse_args()

INPUT_PATH = args.input_path.rstrip("/")
LOSS_KEY   = args.loss_key
SEEDS      = args.seeds


# ---------------------------------------------------------------------------
# Folder parsing
# ---------------------------------------------------------------------------

# Long-form folder: "distance_metric_euclidean_lr_3.00e-06_optimizer_srda_penalty_l2_shift_cost_1.00e+00"
_LR_RE   = re.compile(r"_lr_([0-9.e+\-]+)_")
_OPT_RE  = re.compile(r"_optimizer_([^_]+)_")

def parse_lr_folder(name):
    """Extract (optimizer, lr_float) from a long-form folder name. Returns None if no match."""
    m_opt = _OPT_RE.search(name)
    m_lr  = _LR_RE.search(name)
    if m_opt and m_lr:
        return m_opt.group(1), float(m_lr.group(1))
    return None

def is_best_cfg_folder(name):
    """Short-form folders (e.g. 'lsvrg', 'prospect') hold best-config results."""
    return _LR_RE.search(name) is None and _OPT_RE.search(name) is None


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Scan input_path and group results
# ---------------------------------------------------------------------------

# lr_results[optimizer][lr][seed] = metrics DataFrame
lr_results = {}

# best_results[optimizer] = best_traj DataFrame
best_results = {}

for entry in sorted(os.scandir(INPUT_PATH), key=lambda e: e.name):
    if not entry.is_dir():
        continue
    name = entry.name

    parsed = parse_lr_folder(name)
    if parsed is not None:
        # Per-LR result folder.
        optimizer, lr = parsed
        lr_results.setdefault(optimizer, {}).setdefault(lr, {})
        for seed in SEEDS:
            seed_file = os.path.join(entry.path, f"seed_{seed}.p")
            if os.path.exists(seed_file):
                result = load_pickle(seed_file)
                # result may be FAIL_CODE (-1) if training diverged
                if isinstance(result, dict) and "metrics" in result:
                    lr_results[optimizer][lr][seed] = result["metrics"]

    elif is_best_cfg_folder(name):
        # Best-config summary folder — optimizer name is the folder name itself.
        optimizer = name
        traj_file = os.path.join(entry.path, "best_traj.p")
        if os.path.exists(traj_file):
            best_results[optimizer] = load_pickle(traj_file)


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def make_lr_colormap(lrs):
    """Assign a distinct color to each lr value, sorted ascending."""
    sorted_lrs = sorted(lrs)
    colors = cm.viridis(np.linspace(0.1, 0.9, len(sorted_lrs)))
    return {lr: colors[i] for i, lr in enumerate(sorted_lrs)}

# Fixed color per optimizer for the combined plot.
_OPTIMIZER_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
]

def optimizer_color(optimizers):
    return {opt: _OPTIMIZER_COLORS[i % len(_OPTIMIZER_COLORS)]
            for i, opt in enumerate(sorted(optimizers))}


# ---------------------------------------------------------------------------
# Plot 1: per-optimizer, per-LR curves (one figure per optimizer)
# ---------------------------------------------------------------------------

def safe_clip(values):
    """Clip to a small positive floor so log-scale axes don't break on zero."""
    return np.clip(np.asarray(values, dtype=float), 1e-12, None)


def plot_optimizer(optimizer, lr_seed_metrics):
    """
    Draw one figure for a single optimizer.
    Subplots = one per seed. Lines = one per LR, colored by magnitude.
    Y-axis uses log scale so raw loss values are displayed with log spacing.
    """
    lrs       = sorted(lr_seed_metrics.keys())
    color_map = make_lr_colormap(lrs)
    n_seeds   = len(SEEDS)

    fig, axes = plt.subplots(
        1, n_seeds,
        figsize=(6 * n_seeds, 4),
        sharey=True,
    )
    if n_seeds == 1:
        axes = [axes]

    fig.suptitle(f"{optimizer}  —  {LOSS_KEY}", fontsize=13)

    for ax, seed in zip(axes, SEEDS):
        ax.set_title(f"seed {seed}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(LOSS_KEY if ax == axes[0] else "")
        ax.set_yscale("log")  # log scale on y-axis; tick labels remain raw loss values

        for lr in lrs:
            metrics = lr_seed_metrics[lr].get(seed)
            if metrics is None:
                continue  # result missing or diverged
            epochs = metrics["epoch"].values
            loss   = safe_clip(metrics[LOSS_KEY].values)
            ax.plot(
                epochs, loss,
                color=color_map[lr],
                linewidth=1.2,
                label=f"lr={lr:.1e}",
            )

        ax.grid(True, linestyle="--", alpha=0.4)

    # Shared legend on the right of the last subplot.
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(
            handles, labels,
            fontsize=7,
            loc="upper right",
            framealpha=0.7,
            ncol=max(1, len(lrs) // 8),
        )

    plt.tight_layout()
    out = os.path.join(INPUT_PATH, f"{optimizer}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


for optimizer, lr_seed_metrics in lr_results.items():
    if not lr_seed_metrics:
        continue
    print(f"Plotting {optimizer} ({len(lr_seed_metrics)} LR configs)...")
    plot_optimizer(optimizer, lr_seed_metrics)


# ---------------------------------------------------------------------------
# Plot 2: all optimizers, best config, averaged over seeds
# ---------------------------------------------------------------------------

def plot_all_best(best_results):
    """
    One curve per optimizer using its best-config average training loss.
    Falls back to seed-level columns when average_train_loss is unavailable.
    Y-axis uses log scale; tick labels remain raw loss values.
    """
    if not best_results:
        print("No best_traj.p files found — skipping all.png.")
        return

    opt_colors = optimizer_color(best_results.keys())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"Best config — {LOSS_KEY}", fontsize=13)
    ax.set_xlabel("epoch")
    ax.set_ylabel(LOSS_KEY)
    ax.set_yscale("log")  # log scale on y-axis

    for optimizer, df in sorted(best_results.items()):
        color = opt_colors[optimizer]

        # Prefer the pre-averaged column; fall back to mean of per-seed columns.
        if "average_train_loss" in df.columns and LOSS_KEY == "train_loss":
            loss_vals = df["average_train_loss"].values
        else:
            # Build average from whichever seed columns exist.
            key_suffix = LOSS_KEY.replace("train_loss", "train").replace("val_loss", "val")
            seed_cols = [c for c in df.columns if c.endswith(f"_{key_suffix}")]
            if not seed_cols:
                print(f"  [{optimizer}] no usable columns for '{LOSS_KEY}', skipping.")
                continue
            loss_vals = df[seed_cols].mean(axis=1).values

        epochs = df["epoch"].values
        loss   = safe_clip(loss_vals)

        ax.plot(epochs, loss, color=color, linewidth=2.0, label=optimizer)

    ax.legend(fontsize=9, loc="upper right", framealpha=0.8)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = os.path.join(INPUT_PATH, "all.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


print("Plotting best-config comparison (all optimizers)...")
plot_all_best(best_results)