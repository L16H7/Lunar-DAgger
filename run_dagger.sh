#!/bin/bash

# List of seeds to use for the runs
seeds=(42 123 456 789 101)

# Loop over each seed and run the command
for seed in "${seeds[@]}"; do
    echo "Running with seed: $seed"
    python dagger.py --total-timesteps 100_000 --beta 0 --beta-decay 0 --group "full_learner" --seed $seed
done

echo "All runs completed."