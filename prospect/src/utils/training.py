import pandas as pd
import time
from tqdm import tqdm
import torch
import pickle
import os
import datetime

from src.optim.baselines import (
    StochasticSubgradientMethod,
    StochasticRegularizedDualAveraging,
    SmoothedLSVRG,
    SaddleSAGA,
)
from src.optim.prospect import Prospect, ProspectMoreau
from src.optim.objectives import (
    Objective,
    get_extremile_weights,
    get_superquantile_weights,
    get_esrm_weights,
    get_erm_weights,
)
from src.utils.io import save_results, load_results, var_to_str, get_path
from src.utils.data import load_dataset

SUCCESS_CODE = 0
FAIL_CODE = -1

# Losses that require gradient-mapping oracle (no closed-form prox)
_GRAD_ORACLE_LOSSES = {"mlp_binary_cross_entropy", "mlp_squared_error", "mlp_multinomial_cross_entropy"}


class OptimizationError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(optimizer, val_objective, n_epochs, save_iters=False):
    epoch_len = optimizer.get_epoch_len()
    metrics = [compute_metrics(-1, optimizer, val_objective, 0.0)]
    init_loss = metrics[0]["train_loss"]

    iterates = []
    for epoch in tqdm(range(n_epochs)):
        tic = time.time()
        optimizer.start_epoch()
        for _ in range(epoch_len):
            optimizer.step()
        optimizer.end_epoch()
        toc = time.time()
        if save_iters:
            iterates.append(optimizer.weights.clone())

        metrics.append(compute_metrics(epoch, optimizer, val_objective, toc - tic))
        if metrics[-1]["train_loss"] >= 1.5 * init_loss:
            raise OptimizationError(
                f"Train loss 50 % greater than initial loss! (epoch {epoch})"
            )

    result = {
        "weights": optimizer.weights,
        "metrics": pd.DataFrame(metrics),
    }
    if save_iters:
        return result, iterates
    return result


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def get_optimizer(optim_cfg, objective, seed, device="cpu"):
    name       = optim_cfg["optimizer"]
    lr         = optim_cfg["lr"]
    epoch_len  = optim_cfg["epoch_len"]
    shift_cost = optim_cfg["shift_cost"]
    lrd        = optim_cfg.get("lrd", 0.5)
    penalty    = optim_cfg.get("penalty", "l2")

    if name == "sgd":
        return StochasticSubgradientMethod(
            objective, lr=lr, seed=seed, epoch_len=epoch_len
        )
    elif name == "srda":
        return StochasticRegularizedDualAveraging(
            objective, lr=lr, seed=seed, epoch_len=epoch_len
        )
    elif name == "lsvrg":
        return SmoothedLSVRG(
            objective,
            lr=lr,
            smooth_coef=shift_cost,
            smoothing=penalty,
            seed=seed,
            length_epoch=epoch_len,
        )
    elif name == "saddlesaga":
        return SaddleSAGA(
            objective,
            lrp=lr,
            lrd=lr / 10,
            smoothing=penalty,
            sm_coef=shift_cost,
            seed_grad=seed,
            seed_table=3 * seed,
            epoch_len=epoch_len,
        )
    elif name == "prospect":
        # Auto-select oracle: gradient mapping for non-linear losses, prox otherwise
        oracle = "grad" if objective.loss_name in _GRAD_ORACLE_LOSSES else "prox"
        return Prospect(
            objective,
            lrp=lr,
            epoch_len=epoch_len,
            shift_cost=shift_cost,
            penalty=penalty,
            seed_grad=seed,
            seed_table=3 * seed,
            oracle_reg=oracle,
        )
    elif name == "moreau":
        return ProspectMoreau(
            objective,
            lr=lr,
            penalty=penalty,
            shift_cost=shift_cost,
            seed_grad=seed,
            seed_table=3 * seed,
            epoch_len=epoch_len,
            device=device,
        )
    else:
        raise ValueError(f"Unrecognized optimizer: '{name}'")


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------

def get_objective(model_cfg, X, y, dataset=None, autodiff=True):
    name            = model_cfg["objective"]
    l2_reg          = model_cfg["l2_reg"]
    loss            = model_cfg["loss"]
    n_class         = model_cfg["n_class"]
    shift_cost      = model_cfg["shift_cost"]
    penalty         = model_cfg.get("penalty", "l2")
    distance_metric = model_cfg.get("distance_metric", "euclidean")

    _weight_fns = {
        "erm":               lambda n: get_erm_weights(n),
        "extremile":         lambda n: get_extremile_weights(n, 2.0),
        "superquantile":     lambda n: get_superquantile_weights(n, 0.5),
        "esrm":              lambda n: get_esrm_weights(n, 1.0),
        "extremile_lite":    lambda n: get_extremile_weights(n, 1.5),
        "superquantile_lite":lambda n: get_superquantile_weights(n, 0.25),
        "esrm_lite":         lambda n: get_esrm_weights(n, 0.5),
        "extremile_hard":    lambda n: get_extremile_weights(n, 2.5),
        "superquantile_hard":lambda n: get_superquantile_weights(n, 0.75),
        "esrm_hard":         lambda n: get_esrm_weights(n, 2.0),
    }
    if name not in _weight_fns:
        raise ValueError(f"Unrecognized objective: '{name}'")
    weight_function = _weight_fns[name]

    return Objective(
        X,
        y,
        weight_function,
        l2_reg=l2_reg,
        loss=loss,
        n_class=n_class,
        risk_name=name,
        dataset=dataset,
        shift_cost=shift_cost,
        penalty=penalty,
        autodiff=autodiff,
        distance_metric=distance_metric,
        hidden_dim=model_cfg.get("hidden_dim", 16),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(epoch, optimizer, val_objective, elapsed):
    return {
        "epoch": epoch,
        "train_loss": optimizer.objective.get_batch_loss(optimizer.weights).item(),
        "train_loss_unreg": optimizer.objective.get_batch_loss(
            optimizer.weights, include_reg=False
        ).item(),
        "val_loss": val_objective.get_batch_loss(optimizer.weights).item(),
        "elapsed": elapsed,
    }


# ---------------------------------------------------------------------------
# Main entry point: compute and cache training curves
# ---------------------------------------------------------------------------

def compute_training_curve(
    dataset,
    model_cfg,
    optim_cfg,
    seed,
    n_epochs,
    out_path="results/",
    data_path="data/",
):
    X_train, y_train, X_val, y_val = load_dataset(dataset, data_path=data_path)

    if model_cfg["loss"] == "multinomial_cross_entropy":
        model_cfg["n_class"] = len(torch.unique(y_train))

    if result_exists(dataset, model_cfg, optim_cfg, seed, out_path=out_path):
        print("*** Result exists ***")
        print(f"  dataset:   {dataset}")
        print(f"  model_cfg: {model_cfg}")
        print(f"  optim_cfg: {optim_cfg}")
        print(f"  seed:      {seed}")
        return SUCCESS_CODE

    train_objective = get_objective(model_cfg, X_train, y_train, dataset=dataset)
    val_objective   = get_objective(model_cfg, X_val,   y_val,   dataset=dataset)
    optimizer = get_optimizer(optim_cfg, train_objective, seed)

    try:
        result = train_model(optimizer, val_objective, n_epochs)
        exit_code = SUCCESS_CODE
    except OptimizationError as e:
        print(f"[OptimizationError] {e}")
        result = FAIL_CODE
        exit_code = FAIL_CODE

    save_results(result, dataset, model_cfg, optim_cfg, seed, out_path=out_path)
    return exit_code


# ---------------------------------------------------------------------------
# Result caching helpers
# ---------------------------------------------------------------------------

def result_exists(dataset, model_cfg, optim_cfg, seed, out_path="results"):
    path = "/".join([out_path, dataset, var_to_str(model_cfg), var_to_str(optim_cfg)])
    f = os.path.join(path, f"seed_{seed}.p")
    return os.path.exists(f)


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))


# ---------------------------------------------------------------------------
# Hyperparameter selection helpers
# ---------------------------------------------------------------------------

def compute_average_train_loss(
    dataset, model_cfg, optim_cfg, seeds, out_path="results/"
):
    total = 0.0
    for seed in seeds:
        results = load_results(dataset, model_cfg, optim_cfg, seed, out_path=out_path)
        if isinstance(results, int) and results == FAIL_CODE:
            return [torch.inf]
        total += torch.tensor(results["metrics"]["train_loss"])
    return total / len(seeds)


def find_best_optim_cfg(dataset, model_cfg, optim_cfgs, seeds, out_path="results/"):
    """Select best config by lowest average train loss over last 10 epochs."""
    best_loss = torch.inf
    best_traj = None
    best_cfg  = None

    for optim_cfg in optim_cfgs:
        avg_train_loss = compute_average_train_loss(
            dataset, model_cfg, optim_cfg, seeds, out_path=out_path
        )
        if len(avg_train_loss) > 1 and torch.mean(avg_train_loss[-10:]) < best_loss:
            best_loss = torch.mean(avg_train_loss[-10:])
            best_traj = avg_train_loss
            best_cfg  = optim_cfg

    df = pd.DataFrame(
        {
            "epoch": list(range(len(best_traj))),
            "average_train_loss": [v.item() for v in best_traj],
        }
    )

    path = get_path(
        [dataset, var_to_str(model_cfg), optim_cfgs[0]["optimizer"]], out_path=out_path
    )

    for seed in seeds:
        results = load_results(dataset, model_cfg, best_cfg, seed, out_path=out_path)
        df[f"seed_{seed}_train"] = results["metrics"]["train_loss"]
        df[f"seed_{seed}_val"]   = results["metrics"]["val_loss"]
        if "nb_checkpoints" in results:
            pickle.dump(
                results["nb_checkpoints"],
                open(os.path.join(path, "nb_checkpoints.p"), "wb"),
            )
        if seed in (0, 1):
            weights = results["weights"]

    print(f"Saving results to: {path}")
    pickle.dump(best_cfg,   open(os.path.join(path, "best_cfg.p"),     "wb"))
    pickle.dump(weights,    open(os.path.join(path, "best_weights.p"), "wb"))
    pickle.dump(df,         open(os.path.join(path, "best_traj.p"),    "wb"))