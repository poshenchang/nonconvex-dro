#!/bin/bash
dataset="diabetes"

echo "Reproducing original experiments for dataset: $dataset"

for penalty in l2 neg_entropy
do
    for objective in erm extremile superquantile esrm
    do
        echo "Running L-BFGS for $objective with $penalty"
        ../.venv/bin/python scripts/lbfgs.py --dataset $dataset --objective $objective --penalty $penalty
        
        for optim in sgd srda lsvrg saddlesaga prospect
        do
            echo "Running $optim for $objective with $penalty"
            ../.venv/bin/python scripts/train.py --dataset $dataset --objective $objective --optimizer $optim --n_jobs 8 --n_epochs 64 --penalty $penalty
        done
    done
done
