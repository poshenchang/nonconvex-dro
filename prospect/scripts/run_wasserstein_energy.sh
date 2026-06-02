#!/bin/bash
# Limit threads to prevent thread thrashing and memory exhaustion on large servers
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

tasks="20-29"
dataset="energy"
penalty="wasserstein"

# Run for all three robust objectives (superquantile = CVaR)
for objective in esrm superquantile extremile
do
    # Run LBFGS to compute exact optimum for suboptimality benchmarks
    ../.venv/bin/python scripts/lbfgs.py --dataset $dataset --objective $objective --penalty $penalty

    # Run baseline methods (sgd, srda, lsvrg) along with the wasserstein-dro method (prospect)
    for optim in sgd srda lsvrg prospect
    do
        # Reduced n_jobs to 2 to strictly avoid memory and thread exhaustion
        taskset -c $tasks ../.venv/bin/python scripts/train.py --dataset $dataset --objective $objective --optimizer $optim --penalty $penalty --n_jobs 2 --n_epochs 128
    done
done
