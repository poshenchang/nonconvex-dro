#!/bin/bash

# Define hyperparameter arrays
datasets=("diabetes") # datasets=("yacht" "energy" "concrete" "kin8nm" "power" "acsincome" "iwildcam" "amazon" "diabetes")
objectives=("erm" "extremile" "superquantile" "esrm")
optimizers=("sgd" "srda" "lsvrg" "saddlesaga" "prospect")

# Define combinations to skip using an associative array
declare -A skip_combinations
skip_combinations["diabetes,erm,sgd"]=1
skip_combinations["diabetes,erm,srda"]=1
skip_combinations["diabetes,erm,lsvrg"]=1
skip_combinations["diabetes,erm,prospect"]=1
# You can add more skip combinations here

# Grid search loop over all combinations
for dataset in "${datasets[@]}"; do
    for objective in "${objectives[@]}"; do
        for optimizer in "${optimizers[@]}"; do
            
            # Check if current combination should be skipped
            key="${dataset},${objective},${optimizer}"
            if [[ -n "${skip_combinations[$key]}" ]]; then
                echo "Skipping: ${key}"
                continue
            fi
            
            # Execute the training script
            echo "Running: --dataset ${dataset} --objective ${objective} --optimizer ${optimizer}"
            python scripts/train.py \
                --dataset "${dataset}" \
                --objective "${objective}" \
                --optimizer "${optimizer}" \
                --parallel 0 # Safer for mlp tasks
                
        done
    done
done