#!/bin/bash
# Minimal runner for the flow-matching experiment

# set -euo pipefail

# DATA_PATH=${DATA_PATH:-../../../cite_all_D-100_d-3_pca.npz}
DATA_PATH=${DATA_PATH:-/home/azweig/projects/finfm/benchmark/flow_matching_minimal/data/cite_all_D-100_d-3_pca.npz}
SEED=${SEED:-1111}
# W&B logging is off by default; export WANDB_ARGS to enable (e.g., '--wandb --wandb_mode offline --wandb_project gaga-flow-matching').
WANDB_ARGS=${WANDB_ARGS:-}

python train_clean.py \
    --seed "${SEED}" \
    --deterministic \
    --max_epochs 1 \
    --neg_method add \
    --num_samples 128 \
    --batch_size 32 \
    --noise_levels 0.5 1.0 \
    --sampling_rejection \
    --sampling_rejection_method sugar \
    --sampling_rejection_threshold 0.2 \
    --disc_batch_size 128 \
    --disc_layer_widths 256 128 64 \
    --disc_factor 10 \
    --disc_max_epochs 1 \
    --alpha 8.0 \
    --embed_t \
    --start_group 0 \
    --end_group 2 \
    --test_group 1 \
    --range_size 0.3 \
    --data_path "${DATA_PATH}" \
    --train_autoencoder \
    --ae_max_epochs 1 \
    --ae_early_stop_patience 50 \
    --ae_latent_dim 3 \
    --ae_batch_norm \
    --ae_use_spectral_norm \
    --ae_dropout 0.2 \
    --ae_dist_mse_decay 0.0 \
    --ae_weights_dist 77.4 \
    --ae_weights_reconstr 0 \
    --ae_weights_cycle 0 \
    --ae_weights_cycle_dist 0 \
    --ae_lr 1e-3 \
    --ae_weight_decay 1e-4 \
    --ae_activation relu \
    --hidden_dim 64 \
    --n_tsteps 20 \
    --flow_weight 1.0 \
    --length_weight 1.0 \
    ${WANDB_ARGS}
