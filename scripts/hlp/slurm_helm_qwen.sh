#!/bin/bash
#SBATCH --job-name=train_mousebox_packing_groot_n15_absolute_action_seed42
#SBATCH --partition=rlwrld
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH --output=logs/helm_qwen_%j.out
#SBATCH --error=logs/helm_qwen_%j.err

# Qwen2.5-VL HLP fine-tune on slurm with DDP (torchrun, multi-GPU).
#
# Stage-agnostic. Set STAGE to the dataset stage (matches helm_dataset_<N> built
# by scripts/data/06_make_stage.sh):
#   STAGE=2  → bowl_1 only (~1k rows)
#   STAGE=3  → +bowl 3/5/7 (~18k)
#   STAGE=4  → +bottle 1/3 (~30k)
#   STAGE=7  → bowl/bottle + cream cheese (~50k)
#   STAGE=8  → swap + rotate only (~70k)
#   STAGE=9  → all 10 tasks (~94k)        ← current default
#
# Prereq:
#   STAGE=<N> bash scripts/data/06_make_stage.sh
#
# Usage:
#   sbatch scripts/hlp/slurm_helm_qwen.sh
#   sbatch --export=ALL,STAGE=9,N_EPOCHS=2,BATCH=16 scripts/hlp/slurm_helm_qwen.sh
#
# Env vars (all optional):
#   STAGE            dataset stage id (default 9)
#   N_EPOCHS         training epochs (default 2)
#   BATCH            per-rank mixed-batch size, must be divisible by 4 (default 16)
#   LR               learning rate (default 3e-5; auto-scale from default 2e-5
#                    accounts for effective batch = BATCH × num_gpus)
#   SAVE_STEPS       checkpoint save cadence (default 500)
#   SAVE_TOTAL_LIMIT max ckpts to keep (default 8)
#   RUN_NAME         output run name (default stage<N>_qwen_<date>)
#   MODEL_NAME       Qwen ckpt dir name under ~/ckpt (default Qwen2.5-VL-7B-Instruct)
#   NUM_GPUS         GPUs per node — must match #SBATCH --gpus-per-node (default 2)

set -e

STAGE="${STAGE:-9}"
N_EPOCHS="${N_EPOCHS:-2}"
BATCH="${BATCH:-16}"
LR="${LR:-3e-5}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-8}"
NUM_GPUS="${NUM_GPUS:-2}"

if (( BATCH % 4 != 0 )); then
    echo "ERROR: BATCH must be divisible by 4 (got ${BATCH})" >&2
    exit 1
fi
PER_LABEL=$((BATCH / 4))

MODEL_NAME="${MODEL_NAME:-Qwen2.5-VL-7B-Instruct}"
MODEL_PATH=${HOME}/ckpt/${MODEL_NAME}

STAGE_ROOT=${HOME}/data/helm_data/qwen_data/helm_dataset_${STAGE}/merged
TRAIN_JSONL=${STAGE_ROOT}/all_train.jsonl
VAL_JSONL=${STAGE_ROOT}/all_val.jsonl

PYBIN=${HOME}/miniconda3/envs/HeLM/bin/python
TORCHRUN=${HOME}/miniconda3/envs/HeLM/bin/torchrun

RUN_NAME="${RUN_NAME:-stage${STAGE}_qwen_$(date +%m%d_%H%M)}"
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
if [ ! -x "${TORCHRUN}" ]; then
    echo "ERROR: torchrun not found at ${TORCHRUN}" >&2
    exit 1
fi

export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export WANDB_PROJECT=RefMe

cd /rlwrld2/home/gyeonghun_kim/codes/HeLM
export PYTHONPATH=$(pwd):${PYTHONPATH}
mkdir -p logs "$(dirname ${OUTPUT_DIR})"

echo "[helm-qwen] stage=${STAGE}  num_gpus=${NUM_GPUS}  per-rank batch=${BATCH} (per-label=${PER_LABEL})"
echo "[helm-qwen] epochs=${N_EPOCHS}  lr=${LR}  save_steps=${SAVE_STEPS}  save_total_limit=${SAVE_TOTAL_LIMIT}"
echo "[helm-qwen] effective batch = ${BATCH} × ${NUM_GPUS} = $((BATCH * NUM_GPUS))"
echo "[helm-qwen] train: ${TRAIN_JSONL}"
echo "[helm-qwen] val:   ${VAL_JSONL}"
echo "[helm-qwen] out:   ${OUTPUT_DIR}"
echo

${TORCHRUN} --nproc_per_node=${NUM_GPUS} --master-port=29545 \
    train/train_helm/train_helm.py \
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
    --warmup_ratio 0.03 \
    --logging_steps 10 \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --eval_steps 200 \
    --eval_max_samples 40 \
    --use_qlora True \
    --bf16 True \
    --attn_impl sdpa \
    --dataloader_num_workers 4 \
    --wandb_project RefMe \
    --wandb_run_name ${RUN_NAME}
