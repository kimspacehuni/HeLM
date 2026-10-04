#!/bin/bash
#SBATCH --job-name=train_40_coffeepot_pour_cup_groot_n1__delta_seed42
#SBATCH --partition=rlwrld
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH --output=logs/stage2_helm_pi0_%j.out
#SBATCH --error=logs/stage2_helm_pi0_%j.err

# Stage 2 (HeLM-base): pi0 LLP fine-tune on subtask-split LIBERO-Mem data.
#
# Differs from Round 2 (round2_pi0_libero_mem_v2_0503):
#   - Trains on the SUBTASK-SPLIT dataset (each lift / place is its own
#     episode) so the model learns to follow short subtask commands like
#     "lift_bowl" / "place_bowl" issued by the HLP.
#   - Smaller corpus (≈58 subtask episodes from 30 originals) → fewer steps.
#
# Eval still uses ORIENT=vflip (canonical orientation, same as Round 2).

set -e

PI0_CKPT=/rlwrld2/home/gyeonghun_kim/result/round2_pi0_libero_mem_v2_0503/checkpoints/020000/pretrained_model
DATA_ROOT=/rlwrld2/home/gyeonghun_kim/data/LIBERO-Mem/LIBERO-Mem-LeRobot-v2-100ep/task_01_sub
# repo_id is also used as a wandb tag (cfg_to_group in common/utils/wandb_utils.py),
# which is capped at 64 chars. Keep it short — the on-disk assert in
# lerobot_dataset.py passes (chunks_size patch) so no HF Hub fallback is triggered.
DATA_REPO_ID=task_01_sub
PYBIN=/rlwrld2/home/gyeonghun_kim/miniconda3/envs/HeLM/bin/python
RUN_NAME=stage2_helm_pi0_task01_sub_ep100_delta_0508
OUTPUT_DIR=/rlwrld2/home/gyeonghun_kim/result/${RUN_NAME}

export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export WANDB_PROJECT=RefMe

cd /rlwrld2/home/gyeonghun_kim/codes/HeLM
export PYTHONPATH=$(pwd):${PYTHONPATH}
mkdir -p logs

# Smaller dataset (~58 ep), so fewer steps. ~96 steps/epoch at batch=48 → 10k
# steps ≈ 100 epochs.
BATCH_PER_GPU=24

# 90/10 train/val split. Sub-episodes are sorted (source_ep, start_frame), so
# contiguous index ranges keep each source episode in a single split (no leak).
N_TOTAL=$(wc -l < "${DATA_ROOT}/meta/episodes.jsonl")
N_VAL=$(( N_TOTAL / 10 ))
N_TRAIN=$(( N_TOTAL - N_VAL ))
TRAIN_EPS="[$(seq -s, 0 $((N_TRAIN - 1)))]"
VAL_EPS="[$(seq -s, ${N_TRAIN} $((N_TOTAL - 1)))]"
echo "[split] dataset n=${N_TOTAL}  train=${N_TRAIN}  val=${N_VAL}"

${PYBIN} -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29303 \
    train/train_llp/train_pi0.py \
    --policy.path=${PI0_CKPT} \
    --policy.action_chunk_relative=false \
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
