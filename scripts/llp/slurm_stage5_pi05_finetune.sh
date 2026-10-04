#!/bin/bash
#SBATCH --job-name=train_66_mousebox_packing_groot_n15_absolute_seed42
#SBATCH --partition=rlwrld
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH --output=logs/stage5_pi05_%j.out
#SBATCH --error=logs/stage5_pi05_%j.err

# Stage 5 — fine-tune lerobot's pi0.5 on a merged sub-task dataset.
#
# pi05 is NOT integrated into HeLM's pi0 fork (transformers 5.x conflict).
# We use lerobot's native train CLI from the HeLM_pi05 conda env.
#
# Warm-start: lerobot/pi05_libero (already LIBERO-pretrained).
#
# Override defaults via `sbatch --export=ALL,DATA_ROOT=...,DATA_REPO_ID=...,RUN_NAME=...`
# (this cluster's slurm strips inline `VAR=val sbatch` vars — must use --export).
#
# Available datasets (delta = native pi05_libero prior, abs = HeLM v2 abs_xyz target):
#   bowl_bottle_sub                (delta, 4 sub-task)
#   bowl_bottle_sub_abs_xyz        (abs,   4 sub-task)
#   bowl_bottle_creamcheese_sub    (delta, 8 sub-task, +cream cheese)

set -e

PI05_CKPT="${PI05_CKPT:-${HOME}/ckpt/pi05_libero}"

# Default: delta. Use --export to override for abs or creamcheese runs.
DATA_ROOT="${DATA_ROOT:-${HOME}/data/LIBERO-Mem/LIBERO-Mem-LeRobot-v2-100ep/bowl_bottle_sub}"
DATA_REPO_ID="${DATA_REPO_ID:-bowl_bottle_sub}"
PYBIN=${HOME}/miniconda3/envs/HeLM_pi05/bin/python
LEROBOT_TRAIN=${HOME}/miniconda3/envs/HeLM_pi05/bin/lerobot-train
RUN_NAME="${RUN_NAME:-stage5_pi05_${DATA_REPO_ID}_$(date +%m%d)}"
OUTPUT_DIR=${HOME}/result/${RUN_NAME}

BATCH_PER_GPU="${BATCH_PER_GPU:-12}"
STEPS="${STEPS:-30000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"

export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export WANDB_PROJECT=RefMe

cd /rlwrld2/home/gyeonghun_kim/codes/HeLM
mkdir -p logs

if [ ! -d "${DATA_ROOT}" ]; then
    echo "ERROR: dataset missing at ${DATA_ROOT}" >&2
    exit 1
fi
if [ ! -f "${PI05_CKPT}/config.json" ]; then
    echo "ERROR: pi05 ckpt missing at ${PI05_CKPT}" >&2
    exit 1
fi

echo "[stage5] ckpt        = ${PI05_CKPT}"
echo "[stage5] data        = ${DATA_ROOT}"
echo "[stage5] output      = ${OUTPUT_DIR}"
echo "[stage5] bs/gpu      = ${BATCH_PER_GPU}  steps=${STEPS}"

# lerobot-train uses accelerate under the hood for DDP. The CLI itself reads
# `accelerate` config; launching via `accelerate launch` gives explicit DDP.
ACCEL_BIN=${HOME}/miniconda3/envs/HeLM_pi05/bin/accelerate

if [ -x "${ACCEL_BIN}" ]; then
    LAUNCHER="${ACCEL_BIN} launch --num_processes=2 --num_machines=1 --mixed_precision=bf16"
    TRAIN_ENTRY="-m lerobot.scripts.lerobot_train"
else
    # Fallback: single-GPU via plain CLI.
    LAUNCHER="${LEROBOT_TRAIN}"
    TRAIN_ENTRY=""
fi

${LAUNCHER} ${TRAIN_ENTRY} \
    --policy.path="${PI05_CKPT}" \
    --policy.push_to_hub=false \
    --dataset.repo_id="${DATA_REPO_ID}" \
    --dataset.root="${DATA_ROOT}" \
    --dataset.use_imagenet_stats=false \
    --rename_map='{"observation.images.base_0_rgb":"observation.images.image","observation.images.left_wrist_0_rgb":"observation.images.image2"}' \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${RUN_NAME}" \
    --batch_size="${BATCH_PER_GPU}" \
    --steps="${STEPS}" \
    --save_freq="${SAVE_FREQ}" \
    --log_freq="${LOG_FREQ}" \
    --num_workers="${NUM_WORKERS}" \
    --wandb.enable=true \
    --wandb.project=RefMe \
    --wandb.entity=kgh011016-seoul-national-university \
    --seed=42
