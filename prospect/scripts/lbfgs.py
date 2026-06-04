"""
Run L-BFGS optimizer to get optimal value of spectral risk for a given dataset and regularizer.
Used to compute suboptimality of the optimizers assessed.
"""

import os
import sys
import numpy as np
import torch
from scipy.optimize import minimize
import pickle
import argparse

sys.path.append(".")
from src.utils.data import load_dataset
from src.utils.training import get_objective, OptimizationError
from src.utils.io import var_to_str, get_path

L2_REG = 1.0
SHIFT_COST = 1.0

# Create parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=[
        "yacht",
        "energy",
        "simulated",
        "concrete",
        "iwildcam",
        "kin8nm",
        "power",
        "acsincome",
        "diabetes",
        "amazon",
    ],
)
parser.add_argument(
    "--objective",
    type=str,
    required=True,
    choices=[
        "extremile",
        "superquantile",
        "esrm",
        "erm",
        "extremile_lite",
        "superquantile_lite",
        "esrm_lite",
        "extremile_hard",
        "superquantile_hard",
        "esrm_hard",
    ],
)
parser.add_argument(
    "--penalty",
    type=str,
    default="l2",
    choices=["l2", "neg_entropy", "wasserstein"],
)
parser.add_argument(
    "--distance_metric",
    type=str,
    default="euclidean",
    choices=["euclidean", "cosine"],
)
parser.add_argument(
    "--mlp_loss",
    action="store_true",
    default=False,
    help="Use MLP version of the loss function (mlp_* prefix).",
)
parser.add_argument(
    "--hidden_dim",
    type=int,
    default=16,
    help="Hidden layer width for MLP losses (default: 16).",
)
args = parser.parse_args()

dataset = args.dataset
if dataset in ["yacht", "energy", "concrete", "kin8nm", "power", "acsincome"]:
    loss = "mlp_squared_error" if args.mlp_loss else "squared_error"
    n_class = None
elif dataset == "iwildcam":
    loss = "mlp_multinomial_cross_entropy" if args.mlp_loss else "multinomial_cross_entropy"
    n_class = 60
elif dataset == "amazon":
    loss = "mlp_multinomial_cross_entropy" if args.mlp_loss else "multinomial_cross_entropy"
    n_class = 5
elif dataset == "diabetes":
    loss = "mlp_binary_cross_entropy" if args.mlp_loss else "binary_cross_entropy"
    n_class = None

model_cfg = {
    "objective": args.objective,
    "l2_reg": L2_REG,
    "shift_cost": SHIFT_COST,
    "loss": loss,
    "n_class": n_class,
    "penalty": args.penalty,
    "distance_metric": args.distance_metric,
    "hidden_dim": args.hidden_dim,
}

X_train, y_train, X_val, y_val = load_dataset(dataset)
objective = get_objective(model_cfg, X_train, y_train)


# Define function and Jacobian oracles.
def fun(w):
    return objective.get_batch_loss(torch.tensor(w, dtype=torch.float64)).item()


def jac(w):
    return (
        objective.get_batch_subgrad(
            torch.tensor(w, dtype=torch.float64, requires_grad=True)
        )
        .detach()
        .numpy()
    )


# Run optimizer.
# For MLP losses use Kaiming init (zeros cause dead ReLU); linear models use zeros.
if args.mlp_loss:
    from src.optim.baselines import _kaiming_init
    init = _kaiming_init(objective.num_parameters, objective).numpy()
else:
    num_params = (model_cfg["n_class"] * objective.d
                  if model_cfg["n_class"] else objective.d)
    init = np.zeros(num_params, dtype=np.float64)
output = minimize(fun, init, method="L-BFGS-B", jac=jac)
if output.success:
    path = get_path([dataset, var_to_str(model_cfg)])
    f = os.path.join(path, "lbfgs_min_loss.p")
    pickle.dump(output.fun, open(f, "wb"))
else:
    raise OptimizationError(output.message)