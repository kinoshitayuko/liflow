#!/usr/bin/env bash

export LD_LIBRARY_PATH=/home/user/anaconda3/envs/liflow/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

mkdir -p logs

seeds=(47 48 49 50)
gpus=(0 1 2 3)

for i in "${!seeds[@]}"; do
    seed="${seeds[$i]}"
    gpu="${gpus[$i]}"

    echo "Starting seed=${seed} on GPU=${gpu}"

    CUDA_VISIBLE_DEVICES="$gpu" python3 -u test_TiO2SiO2_H2_20261105_cameraready.py \
        --seed "$seed" \
        > "logs/seed_${seed}_gpu_${gpu}_1ns_after.log" 2>&1 &
done

wait