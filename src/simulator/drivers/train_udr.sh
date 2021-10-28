#!/bin/bash

set -e

SAVE_DIR=../../results_0928
# SAVE_DIR=tmp

for seed in 10 20 30 40 50 60 70 80 90 100; do
    # exp_name=udr1
    # CUDA_VISIBLE_DEVICES="" mpiexec -np 2 python train_rl.py \
    #     --save-dir ${SAVE_DIR}/${exp_name}/seed_${seed} \
    #     --exp-name ${exp_name} \
    #     --tensorboard-log aurora_tensorboard \
    #     --total-timesteps 2000000 \
    #     --randomization-range-file ../../config/train/udr_7_dims_1023/udr_small_seed_${seed}.json \
    #     --seed ${seed} \
    #     --validation \
    #     --pretrained-model-path /tank/zxxia/PCC-RL/results_0826/udr_6/udr_start/seed_20/model_step_151200.ckpt

    # exp_name=udr2
    # CUDA_VISIBLE_DEVICES="" mpiexec -np 2 python train_rl.py \
    #     --save-dir ${SAVE_DIR}/${exp_name}/seed_${seed} \
    #     --exp-name ${exp_name} \
    #     --tensorboard-log aurora_tensorboard \
    #     --total-timesteps 2000000 \
    #     --randomization-range-file ../../config/train/udr_7_dims_1023/udr_mid_seed_${seed}.json \
    #     --seed ${seed} \
    #     --validation \
    #     --pretrained-model-path /tank/zxxia/PCC-RL/results_0826/udr_6/udr_start/seed_20/model_step_151200.ckpt  &
    #
    # exp_name=udr3
    # CUDA_VISIBLE_DEVICES="" mpiexec -np 2 python train_rl.py \
    #     --save-dir ${SAVE_DIR}/${exp_name}/seed_${seed} \
    #     --exp-name ${exp_name} \
    #     --tensorboard-log aurora_tensorboard \
    #     --total-timesteps 2000000 \
    #     --randomization-range-file ../../config/train/udr_7_dims_0826/udr_large.json \
    #     --seed ${seed} \
    #     --validation \
    #     --pretrained-model-path /tank/zxxia/PCC-RL/results_0826/udr_6/udr_start/seed_20/model_step_151200.ckpt  &
    # exp_name=udr3_lossless
    # CUDA_VISIBLE_DEVICES="" mpiexec -np 2 python train_rl.py \
    #     --save-dir ${SAVE_DIR}/${exp_name}/seed_${seed} \
    #     --exp-name ${exp_name} \
    #     --tensorboard-log aurora_tensorboard \
    #     --total-timesteps 2000000 \
    #     --randomization-range-file ../../config/train/udr_7_dims_0826/udr_large_lossless.json \
    #     --seed ${seed} \
    #     --validation \
    #     --pretrained-model-path /tank/zxxia/PCC-RL/results_0826/udr_6/udr_start/seed_20/model_step_151200.ckpt  &

    exp_name=udr3_lossless_no_reward_scale
    CUDA_VISIBLE_DEVICES="" mpiexec -np 2 python train_rl.py \
        --save-dir ${SAVE_DIR}/${exp_name}/seed_${seed} \
        --exp-name ${exp_name} \
        --tensorboard-log aurora_tensorboard \
        --total-timesteps 2000000 \
        --randomization-range-file ../../config/train/udr_7_dims_0826/udr_large_lossless.json \
        --seed ${seed} \
        --validation \
        --pretrained-model-path /tank/zxxia/PCC-RL/results_0826/udr_6/udr_start/seed_20/model_step_151200.ckpt  &
done
        # --pretrained-model-path /tank/zxxia/PCC-RL/results_0826/udr_6/udr_start/seed_20/model_step_21600.ckpt &
