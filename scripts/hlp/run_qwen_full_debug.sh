#!/bin/bash
# Single-A100 debug-node Qwen2.5-VL HLP fine-tune.
# Targets ~3-hour convergence with larger batch + fewer epochs than the
# full multi-GPU sbatch run (slurm_helm_qwen.sh).
#
# Default STAGE=7 = full taskspec set:
#   bowl 1/3/5/7 + bottle 1/3 + creamcheese (4 variants).
#
# Prereqs:
#   1) Per-taskspec qwen_data must exist (bash scripts/data/05_build_qwen.sh ...)
#   2) Stage merge: STAGE=7 bash scripts/data/06_make_stage.sh
#
# Usage:
#   bash scripts/hlp/run_qwen_full_debug.sh
#   STAGE=4 BATCH=16 N_EPOCHS=4 bash scripts/hlp/run_qwen_full_debug.sh
#
# Env vars:
#   STAGE       stage id matching helm_dataset_<N>  (default 7)
#   BATCH       total batch (must be multiple of 4; split 4-way across labels)
#   N_EPOCHS    training epochs (default 5)
#   LR          learning rate (default 2e-5; bump to 5e-5 for faster convergence)
#   RUN_NAME    run dir suffix (default stage<N>_qwen_debug_<date>)
#   MODEL_NAME  Qwen ckpt dir name under ~/ckpt (default Qwen2.5-VL-7B-Instruct)

set -e

STAGE="${STAGE:-7}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-VL-7B-Instruct}"
MODEL_PATH=${HOME}/ckpt/${MODEL_NAME}

STAGE_ROOT=${HOME}/data/helm_data/qwen_data/helm_dataset_${STAGE}/merged
TRAIN_JSONL=${STAGE_ROOT}/all_train.jsonl
VAL_JSONL=${STAGE_ROOT}/all_val.jsonl

PYBIN=${HOME}/miniconda3/envs/HeLM/bin/python
RUN_NAME="${RUN_NAME:-stage${STAGE}_qwen_debug_$(date +%m%d_%H%M)}"
OUTPUT_DIR=${HOME}/result/stage${STAGE}/qwen/${RUN_NAME}

if [ ! -f "${TRAIN_JSONL}" ]; then
    echo "ERROR: ${TRAIN_JSONL} not found." >&2
    echo "Run: STAGE=${STAGE} bash scripts/data/06_make_stage.sh" >&2
    exit 1
fi
if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: model not at ${MODEL_PATH}" >&2
    exit 1
fi

export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export WANDB_PROJECT=RefMe

cd /rlwrld2/home/gyeonghun_kim/codes/HeLM
export PYTHONPATH=$(pwd):${PYTHONPATH}
mkdir -p logs "$(dirname ${OUTPUT_DIR})"

# A100 80GB + QLoRA can fit batch 16 (vs 8 in slurm full). Mixed batch must sum
# to BATCH, split equally across the 4 label pools.
BATCH="${BATCH:-16}"
if (( BATCH % 4 != 0 )); then
    echo "ERROR: BATCH must be divisible by 4 (got ${BATCH})" >&2
    exit 1
fi
PER_LABEL=$((BATCH / 4))

N_EPOCHS="${N_EPOCHS:-2}"
LR="${LR:-2e-5}"
# checkpoint / eval cadence — scale to total step count.
# stage 7 ~25k rows  → batch 16 → ~1.5k step/epoch → save_steps=500 ≈ 3 ckpt/epoch
# stage 9 ~94k rows  → batch 16 → ~5.9k step/epoch → save_steps=1000 ≈ 6 ckpt/epoch
SAVE_STEPS="${SAVE_STEPS:-500}"
EVAL_STEPS="${EVAL_STEPS:-200}"
# Bigger limit so we can pick the best ckpt afterwards rather than just last.
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-8}"

# Approximate train rows per stage (informational; check 06_make_stage.sh output):
#   stage 2 (bowl_1):           ~1k
#   stage 3 (+bowl 3/5/7):     ~18k
#   stage 4 (+bottle 1/3):     ~30k
#   stage 7 (full):            ~45-50k
#   stage 8 (swap+rotate only): ~70k
#   stage 9 (full incl swap+rotate, 34 taskspec): ~94k
echo "[qwen-debug] stage=${STAGE}  batch=${BATCH} (per-label=${PER_LABEL})  epochs=${N_EPOCHS}  lr=${LR}"
echo "[qwen-debug] train: ${TRAIN_JSONL}"
echo "[qwen-debug] val:   ${VAL_JSONL}"
echo "[qwen-debug] out:   ${OUTPUT_DIR}"

${PYBIN} train/train_helm/train_helm.py \
    --model_name_or_path ${MODEL_PATH} \
    --train_jsonl ${TRAIN_JSONL} \
    --val_jsonl ${VAL_JSONL} \
    --num_images 1 \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH} \
    --n_detect_pos ${PER_LABEL} \
    --n_detect_neg ${PER_LABEL} \
    --n_update_intra ${PER_LABEL} \
    --n_update_transition ${PER_LABEL} \
    --num_train_epochs ${N_EPOCHS} \
    --with_replacement True \
    --learning_rate ${LR} \
    --warmup_ratio 0.05 \
    --logging_steps 10 \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --eval_steps ${EVAL_STEPS} \
    --eval_max_samples 40 \
    --use_qlora True \
    --bf16 True \
    --attn_impl sdpa \
    --dataloader_num_workers 4 \
    --wandb_project RefMe \
    --wandb_run_name ${RUN_NAME}
