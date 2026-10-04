#!/bin/bash
# Spin up the pi0 inference server pointing at a Round 1 fine-tuned checkpoint.
# Pi0-only (no HLP/Qwen) — Round 1 was vanilla full-FT, no Qwen yet.
#
# Usage:
#   bash scripts/eval/run_round1_server.sh [STEP] [PORT]
# e.g.:
#   bash scripts/eval/run_round1_server.sh 030000 8000
#
# Defaults to step 030000 of the round1_pi0_libero_mem_normalized_acc_0502 run.

set -e

STEP="${1:-030000}"
PORT="${2:-8000}"

PYBIN=/rlwrld2/home/gyeonghun_kim/miniconda3/envs/HeLM/bin/python
CKPT=/rlwrld2/home/gyeonghun_kim/result/stage2_helm_pi0_abs_xyz_0510/checkpoints/${STEP}/pretrained_model

if [ ! -f "${CKPT}/config.json" ]; then
    echo "ERROR: pi0 ckpt not found at ${CKPT}" >&2
    echo "Available checkpoints:" >&2
    ls /rlwrld2/home/gyeongh un_kim/result/round1_pi0_libero_mem_normalized_acc_0502/checkpoints/ 2>&1 >&2 || true
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
