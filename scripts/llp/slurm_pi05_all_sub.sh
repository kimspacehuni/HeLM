#!/bin/bash
#SBATCH --job-name=train_helm_llp_pi05_all_sub_0603_bs12x2_steps40k_seed42
#SBATCH --partition=rlwrld
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH --output=logs/pi05_all_sub_%j.out
#SBATCH --error=logs/pi05_all_sub_%j.err

# Single-stage pi05 LLP fine-tune on the unified 20-vocab sub-task dataset
# (bowl_bottle_creamcheese_sub + swap_rotate_sub merged via tools/merge_lerobot_v3.py).
#
# Warm-start: ~/ckpt/pi05_libero. Single-stage avoids the multi-stage drift
# / loss-spike issue from the previous bowl_bottle_creamcheese_sub → swap_rotate_sub
# 2-stage approach.
#
# Override via --export=ALL,...:
#   sbatch --export=ALL,STEPS=40000,BATCH_PER_GPU=16 scripts/llp/slurm_pi05_all_sub.sh

set -e

PI05_CKPT="${PI05_CKPT:-${HOME}/ckpt/pi05_libero}"
DATA_REPO_ID="${DATA_REPO_ID:-all_sub_0603}"
DATA_ROOT="${DATA_ROOT:-${HOME}/data/LIBERO-Mem/LIBERO-Mem-LeRobot-v2-100ep/${DATA_REPO_ID}}"

RUN_NAME="${RUN_NAME:-stage5_pi05_${DATA_REPO_ID}_$(date +%m%d_%H%M)}"
OUTPUT_DIR=${HOME}/result/${RUN_NAME}

BATCH_PER_GPU="${BATCH_PER_GPU:-12}"
STEPS="${STEPS:-40000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_GPUS="${NUM_GPUS:-2}"

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

echo "[llp]  ckpt          = ${PI05_CKPT}"
echo "[llp]  dataset       = ${DATA_ROOT}"
echo "[llp]  output        = ${OUTPUT_DIR}"
echo "[llp]  num_gpus      = ${NUM_GPUS}  batch/gpu = ${BATCH_PER_GPU}  effective = $((BATCH_PER_GPU * NUM_GPUS))"
echo "[llp]  steps         = ${STEPS}  save_freq = ${SAVE_FREQ}"

ACCEL_BIN=${HOME}/miniconda3/envs/HeLM_pi05/bin/accelerate
LEROBOT_TRAIN=${HOME}/miniconda3/envs/HeLM_pi05/bin/lerobot-train

if [ -x "${ACCEL_BIN}" ] && [ "${NUM_GPUS}" -gt 1 ]; then
    LAUNCHER="${ACCEL_BIN} launch --num_processes=${NUM_GPUS} --num_machines=1 --mixed_precision=bf16"
    TRAIN_ENTRY="-m lerobot.scripts.lerobot_train"
else
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
