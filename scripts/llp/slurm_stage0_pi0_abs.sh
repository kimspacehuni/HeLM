#!/bin/bash
#SBATCH --job-name=train_40_coffeepot_pour_cup_groot_n1__absolute_seed42
#SBATCH --partition=rlwrld
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH --output=logs/stage2_helm_pi0_abs_%j.out
#SBATCH --error=logs/stage2_helm_pi0_abs_%j.err

# Stage 2 (HeLM-base) — abs-xyz target variant.
#
# Dataset action[:, 0:3] has been pre-transformed offline to absolute EEF xyz
# (next state's xyz). action[:, 3:7] (rotation delta + gripper) unchanged.
# pi0 trains as vanilla on the new dataset; at inference, select_action
# converts predicted abs xyz back to per-step delta using current obs anchor.
#
# Why: LIBERO bang-bang ±0.4 delta is multimodal & ill-suited for flow matching.
# Absolute xyz at 20Hz is a smooth continuous trajectory, friendlier target.
#
# Pre-req: run `tools/transform_dataset_abs_xyz.py` once on task_01_sub to
# produce task_01_sub_abs_xyz before submitting this.
#
# Eval: ORIENT=vflip same as Round 2.

set -e

# Always start from pi0_base (user convention) — Round 2 warmstart was tried
# but action head weights baked for delta-action stats may conflict with the
# new abs-xyz stats. Override with PI0_CKPT=... if needed.
PI0_BASE=/rlwrld2/home/gyeonghun_kim/ckpt/pi0_base
PI0_CKPT=${PI0_CKPT:-${PI0_BASE}}

DATA_ROOT=/rlwrld2/home/gyeonghun_kim/data/LIBERO-Mem/LIBERO-Mem-LeRobot-v2_abs_xyz
DATA_REPO_ID=task_01_sub_abs_xyz
PYBIN=/rlwrld2/home/gyeonghun_kim/miniconda3/envs/HeLM/bin/python
RUN_NAME=stage2_helm_pi0_abs_xyz_0510
OUTPUT_DIR=/rlwrld2/home/gyeonghun_kim/result/${RUN_NAME}

export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export WANDB_PROJECT=RefMe

cd /rlwrld2/home/gyeonghun_kim/codes/HeLM
export PYTHONPATH=$(pwd):${PYTHONPATH}
mkdir -p logs

if [ ! -d "${DATA_ROOT}" ]; then
    echo "ERROR: abs-xyz dataset missing at ${DATA_ROOT}" >&2
    echo "Generate it first:" >&2
    echo "  ${PYBIN} tools/transform_dataset_abs_xyz.py \\" >&2
    echo "    --src /rlwrld2/home/gyeonghun_kim/data/LIBERO-Mem/LIBERO-Mem-LeRobot-v2-100ep/task_01_sub \\" >&2
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
echo "[ckpt]  warm-start from ${PI0_CKPT}"

# Warm-start from Round 2 → fewer steps needed (action head only needs to
# re-fit dim 0-2 to abs xyz scale; rest is already adapted to LIBERO).
${PYBIN} -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29304 \
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
