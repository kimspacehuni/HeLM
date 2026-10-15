#!/bin/bash
# Single-A100 debug-node Qwen2.5-VL HLP fine-tune.
#
# Default STAGE=7 = full taskspec set:
#   bowl 1/3/5/7 + bottle 1/3 + creamcheese (4 variants).
#
# Prerequisite: all_train.jsonl and all_val.jsonl must exist under STAGE_ROOT.
#
# Usage:
#   bash scripts/hlp/run_qwen_full_debug.sh
#   STAGE=4 BATCH=16 N_EPOCHS=4 bash scripts/hlp/run_qwen_full_debug.sh
#
# Env vars:
#   STAGE       stage id matching helm_dataset_<N>  (default 7)
#   BATCH       total batch (must be multiple of 4; split 4-way across labels)
#   N_EPOCHS    training epochs (default 2)
#   LR          learning rate (default 2e-5)
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
    echo "Prepare all_train.jsonl and all_val.jsonl under ${STAGE_ROOT}." >&2
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

# Mixed batches split equally across the four label pools.
BATCH="${BATCH:-16}"
if (( BATCH % 4 != 0 )); then
    echo "ERROR: BATCH must be divisible by 4 (got ${BATCH})" >&2
    exit 1
fi
PER_LABEL=$((BATCH / 4))

N_EPOCHS="${N_EPOCHS:-2}"
LR="${LR:-2e-5}"
# checkpoint / eval cadence — scale to total step count.
SAVE_STEPS="${SAVE_STEPS:-500}"
EVAL_STEPS="${EVAL_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-8}"

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
