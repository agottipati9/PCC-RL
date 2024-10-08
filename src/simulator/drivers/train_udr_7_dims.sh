#!/bin/bash

set -e

# SAVE_DIR=../../results_0415/udr_7_dims_fix_val_reward
# SAVE_DIR=../../results_0430/udr_7_dims
# SAVE_DIR=../../results_0515/udr_7_dims_2_feats_fix_lat_ratio
# SAVE_DIR=../../results_0515/udr_7_dims_fix_lat_ratio_reward
# SAVE_DIR=../../results_0515/udr_mid_simple_stateless_fix_max_tput
SAVE_DIR=../../results_0515/udr_large_lossless_recv_ratio
SAVE_DIR=../../results_0820/udr_recv_ratio
SAVE_DIR=../../results_0826/udr_7
SAVE_DIR=../../results_0904/udr
SAVE_DIR=../../results_0905/udr
SAVE_DIR=../../results_0909/manual
SAVE_DIR=../../results_0909/manual_start_from_real_world_model
# SAVE_DIR=../../results_0430/udr_7_dims
# SAVE_DIR=tmp
# /../../results_0415/udr_7_dims
#rand_duration rand_bw rand_delay rand_loss rand_queue rand_bw_freq rand_delay_freq
# for exp_name in range0 range1 range2; do
# for exp_name in range2_no_vary_bw; do
# range0_queue10
# for exp_name in 12mbps_queue50; do
#     CUDA_VISIBLE_DEVICES="" python train_rl.py \
#         --save-dir ${SAVE_DIR}/${exp_name} \
#         --exp-name ${exp_name}_seed_${seed} \
#         --tensorboard-log aurora_tensorboard \
#         --total-timesteps 5000000 \
#         --delta-scale 1 \
#         --randomization-range-file ../../config/train/udr_7_dims_0415/${exp_name}.json \
#         --seed ${seed} \
#         --time-variant-bw
# done
        # --pretrained-model-path ../../results_0415/udr_7_dims/range0_vary_bw_old/model_step_1576800.ckpt
# for exp_name in udr_mid_simple; do
# for exp_name in udr_mid_simple; do
# for exp_name in udr_mid_lossless udr_small_lossless; do
# for exp_name in udr_mid_lossless udr_small_lossless; do
# for exp_name in udr_large udr_mid udr_small; do
#     # for seed in 10 20 30 40 50; do
#     for seed in 20; do
#         CUDA_VISIBLE_DEVICES="" mpiexec -np 1 python train_rl.py \
#             --save-dir ${SAVE_DIR}/${exp_name}/seed_${seed} \
#             --exp-name ${exp_name}_seed_${seed}_recv_ratio \
#             --tensorboard-log aurora_tensorboard \
#             --total-timesteps 5000000 \
#             --delta-scale 1 \
#             --randomization-range-file ../../config/train/udr_7_dims_0820/${exp_name}.json \
#             --seed ${seed} \
#             --time-variant-bw &
#     done
# done


# exp_name=udr_start
# for seed in 10; do # 60 70 80 90 100
#     exp_name=short_queue
#     n_proc=8
#     CUDA_VISIBLE_DEVICES="" mpiexec -np ${n_proc} python train_rl.py \
#         --save-dir ${SAVE_DIR}/${exp_name}_new_para_val/seed_${seed} \
#         --exp-name ${exp_name}_seed_${seed}_recv_ratio \
#         --tensorboard-log aurora_tensorboard \
#         --total-timesteps 1000000 \
#         --delta-scale 1 \
#         --randomization-range-file ../../config/train/udr_7_dims_0905/short_queue.json \
#         --seed ${seed} \
#         --n-proc ${n_proc} \
#         --pretrained-model-path ../../results_0826/genet_cubic_exp_3/bo_11/model_step_28800.ckpt
#         # --pretrained-model-path ../../results_0909/manual1/short_queue/seed_10/model_step_28800.ckpt
#         # --pretrained-model-path ../../results_0909/manual1/short_queue/seed_10/model_step_28800.ckpt
#             # ../../results_0826/udr_6/udr_start/seed_20/model_step_64800.ckpt
#         # --pretrained-model-path /tank/zxxia/PCC-RL/results_0826/udr_6/udr_start/seed_20/model_step_151200.ckpt &
# done
exp_name=udr_large
for seed in 10 30 40 50 ; do # 60 70 80 90 100
    CUDA_VISIBLE_DEVICES="" mpiexec -np 2 python train_rl.py \
        --save-dir ${SAVE_DIR}/${exp_name}/seed_${seed} \
        --exp-name ${exp_name}_seed_${seed}_recv_ratio \
        --tensorboard-log aurora_tensorboard \
        --total-timesteps 1000000 \
        --randomization-range-file ../../config/train/udr_7_dims_0826/${exp_name}.json \
        --seed ${seed} \
        --pretrained-model-path /tank/zxxia/PCC-RL/results_0826/udr_6/udr_start/seed_20/model_step_21600.ckpt &
        # --pretrained-model-path /tank/zxxia/PCC-RL/results_0826/udr_6/udr_start/seed_20/model_step_151200.ckpt &
done

for seed in 10 20 30 40 50; do #60 70 80 90 100
    exp_name=udr_large
    CUDA_VISIBLE_DEVICES="" mpiexec -np 2 python train_rl.py \
        --save-dir ${SAVE_DIR}/${exp_name}/seed_${seed} \
        --exp-name ${exp_name}_seed_${seed}_recv_ratio \
        --tensorboard-log aurora_tensorboard \
        --total-timesteps 1000000 \
        --randomization-range-file ../../config/train/udr_7_dims_0827/${exp_name}_seed_${seed}.json \
        --seed ${seed} \
        --pretrained-model-path /tank/zxxia/PCC-RL/results_0826/udr_6/udr_start/seed_20/model_step_21600.ckpt &
        # --pretrained-model-path /tank/zxxia/PCC-RL/results_0826/udr_6/udr_start/seed_20/model_step_151200.ckpt \
done
for seed in 10 20 30 40 50;do #60 70 80 90 100
    exp_name=udr_small
    CUDA_VISIBLE_DEVICES="" mpiexec -np 2 python train_rl.py \
        --save-dir ${SAVE_DIR}/${exp_name}/seed_${seed} \
        --exp-name ${exp_name}_seed_${seed}_recv_ratio \
        --tensorboard-log aurora_tensorboard \
        --total-timesteps 1000000 \
        --randomization-range-file ../../config/train/udr_7_dims_0827/${exp_name}_seed_${seed}.json \
        --seed ${seed} \
        --pretrained-model-path /tank/zxxia/PCC-RL/results_0826/udr_6/udr_start/seed_20/model_step_21600.ckpt &
        # --pretrained-model-path /tank/zxxia/PCC-RL/results_0826/udr_6/udr_start/seed_20/model_step_151200.ckpt \
done

# for seed in 10 20 30 40 50; do #60 70 80 90 100
#     exp_name=udr_mid
#     CUDA_VISIBLE_DEVICES="" mpiexec -np 2 python train_rl.py \
#         --save-dir ${SAVE_DIR}/${exp_name}/seed_${seed} \
#         --exp-name ${exp_name}_seed_${seed}_recv_ratio \
#         --tensorboard-log aurora_tensorboard \
#         --total-timesteps 1000000 \
#         --delta-scale 1 \
#         --randomization-range-file ../../config/train/udr_7_dims_0905/${exp_name}_seed_${seed}.json \
#         --seed ${seed} \
#         --pretrained-model-path ../../results_0826/udr_6/udr_start/seed_20/model_step_64800.ckpt \
#         --time-variant-bw &
#         # --pretrained-model-path /tank/zxxia/PCC-RL/results_0826/udr_6/udr_start/seed_20/model_step_151200.ckpt \
# done
#
# for seed in 10 20 30 40 50;do #60 70 80 90 100
#     exp_name=udr_small
#     CUDA_VISIBLE_DEVICES="" mpiexec -np 2 python train_rl.py \
#         --save-dir ${SAVE_DIR}/${exp_name}/seed_${seed} \
#         --exp-name ${exp_name}_seed_${seed}_recv_ratio \
#         --tensorboard-log aurora_tensorboard \
#         --total-timesteps 1000000 \
#         --delta-scale 1 \
#         --randomization-range-file ../../config/train/udr_7_dims_0905/${exp_name}_seed_${seed}.json \
#         --seed ${seed} \
#         --pretrained-model-path ../..//results_0826/udr_6/udr_start/seed_20/model_step_64800.ckpt \
#         --time-variant-bw &
#         # --pretrained-model-path /tank/zxxia/PCC-RL/results_0826/udr_6/udr_start/seed_20/model_step_151200.ckpt \
# done

# exp_name=real_fail
# seed=40
# for noise in 0 ; do
# CUDA_VISIBLE_DEVICES="" mpiexec -np 2 python train_rl.py \
#     --save-dir ${SAVE_DIR}/20mbps/${exp_name}${noise} \
#     --exp-name ${exp_name}_noise${noise} \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 1000000 \
#     --delta-scale 1 \
#     --randomization-range-file ../../config/train/udr_7_dims_0905/20mbps/${exp_name}${noise}.json \
#     --seed ${seed} \
#     --pretrained-model-path ../../results_0826/udr_6/udr_start/seed_20/model_step_64800.ckpt
#     # --pretrained-model-path ../../results_0905/udr/real_fail20/model_step_259200.ckpt \
# done
    # --pretrained-model-path /tank/zxxia/PCC-RL/results_0826/udr_6/udr_start/seed_20/model_step_151200.ckpt \



# seed=40
# for noise in 0 2 5 10 20; do # 60 70 80 90 100
#     exp_name=udr_large_short_queue_${noise}
#     CUDA_VISIBLE_DEVICES="" mpiexec -np 2 python train_rl.py \
#         --save-dir ${SAVE_DIR}/${exp_name}/seed_${seed} \
#         --exp-name ${exp_name}_seed_${seed}_recv_ratio \
#         --tensorboard-log aurora_tensorboard \
#         --total-timesteps 1000000 \
#         --delta-scale 1 \
#         --randomization-range-file ../../config/train/udr_7_dims_0905/${exp_name}.json \
#         --seed ${seed} \
#         --pretrained-model-path ../../results_0826/udr_6/udr_start/seed_20/model_step_64800.ckpt &
#         # --pretrained-model-path /tank/zxxia/PCC-RL/results_0826/udr_6/udr_start/seed_20/model_step_151200.ckpt &
# done
