#!/bin/bash
#SBATCH --job-name=61_mouse_full_groot_normalized
#SBATCH --partition=rlwrld
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=48:00:00
#SBATCH --output=logs/round1_pi0_%j.out
#SBATCH --error=logs/round1_pi0_%j.err

# Round 1: multi-task pi0 full fine-tuning on LIBERO-Mem (10 tasks, 961 episodes).
# Launches torchrun with 2 ranks (one per H100) -> DDP across both GPUs.

set -e

PI0_CKPT=/rlwrld2/home/gyeonghun_kim/ckpt/pi0_base
DATA_ROOT=/rlwrld2/home/gyeonghun_kim/data/LIBERO-Mem/LIBERO-Mem-LeRobot
PYBIN=/rlwrld2/home/gyeonghun_kim/miniconda3/envs/HeLM/bin/python
RUN_NAME=round1_pi0_libero_mem_normalized_acc_0502
OUTPUT_DIR=/rlwrld2/home/gyeonghun_kim/result/${RUN_NAME}
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export WANDB_PROJECT=RefMe

cd /rlwrld2/home/gyeonghun_kim/codes/HeLM
export PYTHONPATH=$(pwd):${PYTHONPATH}
mkdir -p logs

# IMPORTANT: tune --batch_size to whatever the H100 80GB can hold per GPU
# (effective global batch = batch_size * num_gpus). Confirm via run_pi0_debug.sh
# on the A100 first; H100 typically fits ~1.5x what an A100 80GB does.
BATCH_PER_GPU=24

${PYBIN} -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29501 \
    train/train_llp/train_pi0.py \
    --policy.path=${PI0_CKPT} \
    --train_dataset.repo_id=LIBERO-Mem-LeRobot \
    --train_dataset.root=${DATA_ROOT} \
    --test_dataset.repo_id=LIBERO-Mem-LeRobot \
    --test_dataset.root=${DATA_ROOT} \
    --batch_size=${BATCH_PER_GPU} \
    --steps=30000 \
    --save_freq=5000 \
    --log_freq=100 \
    --test_freq=500 \
    --num_workers=16 \
    --dataloader_type=default \
    --output_dir=${OUTPUT_DIR} \
    --job_name=${RUN_NAME} \
    --wandb.enable=true \
    --wandb.project=RefMe \
    --wandb.entity=kgh011016-seoul-national-university \
    --method.core=vanilla \
    --dist_mode=ddp \
    --use_ddp=true \
    --seed=42
