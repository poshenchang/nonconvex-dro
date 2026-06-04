"""
plot.py  —  Visualize training curves from results produced by train.py.

Usage:
    python plot.py --input_path results/diabetes/distance_metric_euclidean_...

Outputs (written to input_path/):
    <optimizer>.png   — per-LR curves for each optimizer, two subplots (seed 1 / seed 2)
    all.png           — best-config suboptimality for every optimizer on one plot

Suboptimality is defined as:
    (F(w) - F(w*)) / (F(w^0) - F(w*))
where w* is the L-BFGS optimum loaded from input_path/lbfgs_min_loss.p
and w^0 is the initial iterate (epoch -1).
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
    default="val_loss",
    choices=["train_loss", "train_loss_unreg", "val_loss"],
    help="Which loss column to plot (default: val_loss).",
)
parser.add_argument(
    "--seeds",
    type=int,
    nargs="+",
    default=[1, 2],
    help="Seeds to plot as subplots (default: 1 2).",
)
parser.add_argument(
    "--lbfgs_path",
    type=str,
    default=None,
    help=(
        "Path to lbfgs_min_loss.p. "
        "Defaults to input_path/lbfgs_min_loss.p. "
        "Pass 'none' to skip suboptimality and plot raw loss instead."
    ),
)
args = parser.parse_args()

INPUT_PATH = args.input_path.rstrip("/")
LOSS_KEY   = args.loss_key
SEEDS      = args.seeds

# ---------------------------------------------------------------------------
# Load L-BFGS optimal value for suboptimality normalization
# ---------------------------------------------------------------------------

def _load_lbfgs_min(input_path, lbfgs_path_arg):
    """
    Load the L-BFGS minimum loss F(w*) from a pickle file.
    Returns None if the file is absent or the user passed --lbfgs_path none.
    """
    if lbfgs_path_arg is not None and lbfgs_path_arg.lower() == "none":
        return None
    path = lbfgs_path_arg if lbfgs_path_arg else os.path.join(input_path, "lbfgs_min_loss.p")
    if not os.path.exists(path):
        print(f"[warning] lbfgs_min_loss.p not found at '{path}' — plotting raw loss.")
        return None
    with open(path, "rb") as fh:
        val = pickle.load(fh)
    # Stored value may be a 0-d array or a plain float.
    return float(val)

LBFGS_MIN = _load_lbfgs_min(INPUT_PATH, args.lbfgs_path)
USE_SUBOPT = LBFGS_MIN is not None
if USE_SUBOPT:
    print(f"L-BFGS minimum: {LBFGS_MIN:.6g}  (suboptimality mode)")
else:
    print("L-BFGS minimum not available — plotting raw loss.")


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
    "#4daf4a", "#e41a1c", "#ffcc00", "#377eb8",
    "#984ea3", "#a65628", "#f781bf", "#999999",
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


def to_suboptimality(loss_vals, init_loss):
    """
    Convert raw loss values to normalised suboptimality (Eq. 8 in the paper):
        subopt(w) = (F(w) - F(w*)) / (F(w^0) - F(w*))
    where F(w*) = LBFGS_MIN and F(w^0) = init_loss (epoch -1 value).
    Values are clipped to [1e-12, inf] so log-scale rendering is safe.
    """
    arr  = np.asarray(loss_vals, dtype=float)
    denom = init_loss - LBFGS_MIN
    if abs(denom) < 1e-15:
        return safe_clip(arr - LBFGS_MIN)
    return safe_clip((arr - LBFGS_MIN) / denom)


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

    y_label = "suboptimality" if USE_SUBOPT else LOSS_KEY
    fig.suptitle(f"{optimizer}  —  {y_label}", fontsize=13)

    for ax, seed in zip(axes, SEEDS):
        ax.set_title(f"seed {seed}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(y_label if ax == axes[0] else "")
        ax.set_yscale("log")

        for lr in lrs:
            metrics = lr_seed_metrics[lr].get(seed)
            if metrics is None:
                continue  # result missing or diverged
            epochs    = metrics["epoch"].values
            raw_loss  = np.asarray(metrics[LOSS_KEY].values, dtype=float)
            # epoch -1 is the pre-training checkpoint; use it as F(w^0)
            init_loss = raw_loss[0] if len(raw_loss) > 0 else raw_loss[0]
            loss = to_suboptimality(raw_loss, init_loss) if USE_SUBOPT else safe_clip(raw_loss)
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

    y_label = "suboptimality" if USE_SUBOPT else LOSS_KEY
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"Best config — {y_label}", fontsize=13)
    ax.set_xlabel("epoch")
    ax.set_ylabel(y_label)
    ax.set_yscale("log")

    for optimizer, df in sorted(best_results.items()):
        color = opt_colors[optimizer]

        # Prefer the pre-averaged column; fall back to mean of per-seed columns.
        if "average_train_loss" in df.columns and LOSS_KEY == "train_loss":
            loss_vals = df["average_train_loss"].values
        else:
            key_suffix = LOSS_KEY.replace("train_loss", "train").replace("val_loss", "val")
            seed_cols = [c for c in df.columns if c.endswith(f"_{key_suffix}")]
            if not seed_cols:
                print(f"  [{optimizer}] no usable columns for '{LOSS_KEY}', skipping.")
                continue
            loss_vals = df[seed_cols].mean(axis=1).values

        epochs    = df["epoch"].values
        raw_loss  = np.asarray(loss_vals, dtype=float)
        init_loss = raw_loss[0] if len(raw_loss) > 0 else raw_loss[0]
        loss = to_suboptimality(raw_loss, init_loss) if USE_SUBOPT else safe_clip(raw_loss)

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