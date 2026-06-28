#!/bin/bash
# Single-GPU batch-size probe for pi0 multi-task fine-tuning on LIBERO-Mem.
# Use on the A100 debug node to find the largest batch_size that fits before
# launching the 2x H100 production sbatch.
#
# Usage:
#   bash scripts/llp/run_pi0_debug.sh [BATCH_SIZE]
# e.g.:
#   bash scripts/llp/run_pi0_debug.sh 16
#
# Watch nvidia-smi in another pane; if no OOM after step ~5, that batch fits.
# It runs only 50 steps and exits — purely for memory probing, not training.

set -e

BATCH_SIZE="${1:-16}"

PI0_CKPT=/rlwrld2/home/gyeonghun_kim/ckpt/pi0_base
DATA_ROOT=/rlwrld2/home/gyeonghun_kim/data/LIBERO-Mem/LIBERO-Mem-LeRobot
PYBIN=/rlwrld2/home/gyeonghun_kim/miniconda3/envs/HeLM/bin/python

export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

cd /rlwrld2/home/gyeonghun_kim/codes/HeLM
export PYTHONPATH=$(pwd):${PYTHONPATH}

CUDA_VISIBLE_DEVICES=0 \
${PYBIN} train/train_llp/train_pi0.py \
    --policy.path=${PI0_CKPT} \
    --train_dataset.repo_id=local \
    --train_dataset.root=${DATA_ROOT} \
    --test_dataset.repo_id=local \
    --test_dataset.root=${DATA_ROOT} \
    --batch_size=${BATCH_SIZE} \
    --steps=50 \
    --log_freq=10 \
    --save_checkpoint=false \
    --num_workers=4 \
    --output_dir=/rlwrld2/home/gyeonghun_kim/result/debug_pi0_bs${BATCH_SIZE} \
    --job_name=debug_pi0_bs${BATCH_SIZE} \
    --wandb.enable=false \
    --method.core=vanilla \
    --dist_mode=none
