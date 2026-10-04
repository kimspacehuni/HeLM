#!/bin/bash
# Spin up the pi0 inference server pointing at a Round 2 fine-tuned checkpoint.
# Pi0-only (no HLP/Qwen). Round 2 = trained on the v2 (canonical) LIBERO-Mem
# dataset (floor-bottom + L/R-correct), so eval client should use ORIENT=vflip
# (mujoco_raw → vflip → canonical) to match the training distribution.

set -e

STEP="${PI0_STEP:-${1:-030000}}"
PORT="${PORT:-${2:-8000}}"

PYBIN=/rlwrld2/home/gyeonghun_kim/miniconda3/envs/HeLM/bin/python
RUN_DIR=/rlwrld2/home/gyeonghun_kim/result/stage4_pi0_bowl_bottle_sub_abs_xyz_0510
CKPT=${RUN_DIR}/checkpoints/${STEP}/pretrained_model

if [ ! -f "${CKPT}/config.json" ]; then
    echo "ERROR: pi0 ckpt not found at ${CKPT}" >&2
    echo "Available checkpoints:" >&2
    ls ${RUN_DIR}/checkpoints/ 2>&1 >&2 || true
    exit 1
fi

export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

cd /rlwrld2/home/gyeonghun_kim/codes/HeLM
export PYTHONPATH=$(pwd):${PYTHONPATH}

echo "[server] ckpt = ${CKPT}"
echo "[server] port = ${PORT}"

CUDA_VISIBLE_DEVICES=0 \
${PYBIN} tools/helm_pi0_server.py \
    --ckpt "${CKPT}" \
    --device cuda:0 \
    --host 0.0.0.0 \
    --port "${PORT}"
