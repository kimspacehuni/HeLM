#!/bin/bash
# Spin up the pi0.5 inference server (lerobot PI05Policy) for LIBERO-Mem eval.
#
# Runs in the HeLM_pi05 conda env (transformers 5.x + lerobot[pi]). The default
# ckpt is the LIBERO-pretrained pi05 (~/ckpt/pi05_libero). For zero-shot ceiling
# check on LIBERO-Mem, no fine-tune is needed.
#
# Usage:
#   bash scripts/eval/run_pi05_server.sh
#   CKPT=~/ckpt/pi05_libero PORT=8000 bash scripts/eval/run_pi05_server.sh
# CKPT=~/result/stage5_pi05_bowl_bottle_0519/checkpoints/030000/pretrained_model ABS_XYZ_INVERSE=1  bash scripts/eval/run_pi05_server.sh

# CKPT=~/result/stage5_pi05_bowl_bottle_creamcheese_sub_0520/checkpoints/030000/pretrained_model \
# PORT=8001 \
#   bash scripts/eval/run_pi05_server.sh

set -e

CKPT="${CKPT:-${HOME}/ckpt/pi05_libero}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-cuda:0}"
# For ckpts fine-tuned on absolute EEF target xyz (e.g. bowl_bottle_sub_abs_xyz),
# set ABS_XYZ_INVERSE=1 so the server converts target xyz back to delta before
# returning to the env client.
ABS_XYZ_INVERSE="${ABS_XYZ_INVERSE:-0}"
ABS_XYZ_GAIN="${ABS_XYZ_GAIN:-80.0}"

PYBIN=/rlwrld2/home/gyeonghun_kim/miniconda3/envs/HeLM_pi05/bin/python

if [ ! -f "${CKPT}/config.json" ]; then
    echo "ERROR: pi05 ckpt not found at ${CKPT}" >&2
    exit 1
fi

export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

cd /rlwrld2/home/gyeonghun_kim/codes/HeLM

echo "[server] ckpt = ${CKPT}"
echo "[server] port = ${PORT}"
echo "[server] env  = HeLM_pi05 (lerobot pi05)"

ABS_FLAG=""
if [ "${ABS_XYZ_INVERSE}" = "1" ]; then
    ABS_FLAG="--abs-xyz-inverse --abs-xyz-gain ${ABS_XYZ_GAIN}"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
${PYBIN} tools/helm_pi05_server.py \
    --ckpt "${CKPT}" \
    --device "${DEVICE}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    ${ABS_FLAG}
