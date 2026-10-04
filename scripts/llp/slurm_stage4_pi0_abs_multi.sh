#!/bin/bash
#SBATCH --job-name=train_40_coffeepot_pour_cup_groot_n1__delta_seed42
#SBATCH --partition=rlwrld
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH --output=logs/stage4_pi0_abs_multi_%j.out
#SBATCH --error=logs/stage4_pi0_abs_multi_%j.err

# Stage 4 — pi0 LLP multi-task abs-xyz training on bowl + bottle sub-task data.
#
# Data: merged dataset (bowl + bottle sub-task split, abs-xyz target)
#   = task_01_sub_abs_xyz + task_02_sub_abs_xyz, unified tasks.jsonl with
#     {lift_bowl, place_bowl, lift_bottle, place_bottle}
#
# Start: pi0_base (cold). Override with PI0_CKPT=... env var for warm-start.
#
# Eval still uses ORIENT=vflip.

set -e

PI0_CKPT="${PI0_CKPT:-/rlwrld2/home/gyeonghun_kim/ckpt/pi0_libero}"

DATA_ROOT=/rlwrld2/home/gyeonghun_kim/data/LIBERO-Mem/LIBERO-Mem-LeRobot-v2-100ep/bowl_bottle_sub_abs_xyz
DATA_REPO_ID=bowl_bottle_sub_abs_xyz
PYBIN=/rlwrld2/home/gyeonghun_kim/miniconda3/envs/HeLM/bin/python
RUN_NAME="${RUN_NAME:-stage4_pi0_bowl_bottle_sub_abs_xyz_libero_$(date +%m%d)}"
OUTPUT_DIR=/rlwrld2/home/gyeonghun_kim/result/${RUN_NAME}

export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export WANDB_PROJECT=RefMe

cd /rlwrld2/home/gyeonghun_kim/codes/HeLM
export PYTHONPATH=$(pwd):${PYTHONPATH}
mkdir -p logs

if [ ! -d "${DATA_ROOT}" ]; then
    echo "ERROR: merged dataset missing at ${DATA_ROOT}" >&2
    echo "Generate it first via:" >&2
    echo "  ${PYBIN} tools/merge_lerobot_datasets.py \\" >&2
    echo "    --srcs /rlwrld2/.../task_01_sub_abs_xyz /rlwrld2/.../task_02_sub_abs_xyz \\" >&2
    echo "    --dst ${DATA_ROOT}" >&2
    exit 1
fi

BATCH_PER_GPU=24

N_TOTAL=$(wc -l < "${DATA_ROOT}/meta/episodes.jsonl")
N_VAL=$(( N_TOTAL / 10 ))
N_TRAIN=$(( N_TOTAL - N_VAL ))
TRAIN_EPS="[$(seq -s, 0 $((N_TRAIN - 1)))]"
VAL_EPS="[$(seq -s, ${N_TRAIN} $((N_TOTAL - 1)))]"
echo "[split] dataset n=${N_TOTAL}  train=${N_TRAIN}  val=${N_VAL}"
echo "[ckpt]  start from ${PI0_CKPT}"

# Cold start from pi0_base. 30k steps × DDP 2 GPU × batch 48 ≈ 12-15h.
${PYBIN} -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29305 \
    train/train_llp/train_pi0.py \
    --policy.path=${PI0_CKPT} \
    --policy.action_abs_xyz=true \
    --train_dataset.repo_id=${DATA_REPO_ID} \
    --train_dataset.root=${DATA_ROOT} \
    --train_dataset.episodes="${TRAIN_EPS}" \
    --test_dataset.repo_id=${DATA_REPO_ID} \
    --test_dataset.root=${DATA_ROOT} \
    --test_dataset.episodes="${VAL_EPS}" \
    --batch_size=${BATCH_PER_GPU} \
    --steps=30000 \
    --save_freq=5000 \
    --log_freq=10 \
    --test_freq=500 \
    --num_workers=8 \
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
