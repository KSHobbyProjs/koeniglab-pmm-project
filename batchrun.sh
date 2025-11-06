#!/bin/bash

SEEDS=(0 1 2 3 4 5)

for SEED in "${SEEDS[@]}"; do
    echo "working through seed $SEED"
    python3 main.py --seed $SEED
done
