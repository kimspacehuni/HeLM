#!/bin/bash
# Full HeLM server with Qwen HLP + pi05 LLP (via HTTP forward).
#
# 2-process setup (Qwen and pi05 need incompatible transformers versions):
#   Process A:  pi05 server (HeLM_pi05 env)  → PI_PORT (default 8001)
#       bash scripts/eval/run_pi05_server.sh PORT=8001 \
#           CKPT=~/result/stage5_pi05_bowl_bottle_sub_0520/checkpoints/030000/pretrained_model
#   Process B:  THIS script (HeLM env)       → PORT (default 8000)
#       Loads Qwen for /detect /update; HTTP-forwards /predict /reset to pi05.
#
# Client (mem_9) connects only to PORT (8000). /predict appears transparent.
#
# Env vars:
#   PI_FORWARD_URL  — pi05 server URL (default http://localhost:8001)
#   QWEN_BASE       — Qwen2.5-VL ckpt dir
#   QWEN_RUN        — Qwen adapter run dir under result/stage<N>/qwen/<run>
#   QWEN_STAGE      — stage number (default 7)
#   QWEN_STEP       — checkpoint step number (e.g. 1500)
#   PORT            — this server's port (default 8000)

# QWEN_RUN=stage7_qwen_debug_0522_0026 QWEN_STAGE=7 QWEN_STEP=2000 bash scripts/eval/run_helm_pi05_server.sh

set -e

PI_FORWARD_URL="${PI_FORWARD_URL:-http://localhost:8001}"
PORT="${PORT:-8000}"

PYBIN=${HOME}/miniconda3/envs/HeLM/bin/python
QWEN_BASE="${QWEN_BASE:-${HOME}/ckpt/Qwen2.5-VL-7B-Instruct}"
QWEN_STAGE="${QWEN_STAGE:-7}"
QWEN_RUN="${QWEN_RUN:-stage3_qwen_bowl_multicycle_0508}"
QWEN_STEP="${QWEN_STEP:-2000}"
QWEN_ADAPTER=${HOME}/result/stage${QWEN_STAGE}/qwen/${QWEN_RUN}/checkpoint-${QWEN_STEP}

if [ ! -d "${QWEN_ADAPTER}" ]; then
    echo "ERROR: qwen adapter not found at ${QWEN_ADAPTER}" >&2
    echo "Available qwen runs:" >&2
    ls ${HOME}/result/stage${QWEN_STAGE}/qwen/ 2>&1 >&2 || true
    exit 1
fi

export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

cd /rlwrld2/home/gyeonghun_kim/codes/HeLM
export PYTHONPATH=$(pwd):${PYTHONPATH}

echo "[helm-server] pi-forward url: ${PI_FORWARD_URL}"
echo "[helm-server] qwen base:      ${QWEN_BASE}"
echo "[helm-server] qwen adapter:   ${QWEN_ADAPTER}"
echo "[helm-server] port:           ${PORT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
${PYBIN} tools/helm_full_server.py \
    --pi-forward-url "${PI_FORWARD_URL}" \
    --qwen-base "${QWEN_BASE}" \
    --qwen-adapter "${QWEN_ADAPTER}" \
    --device cuda:0 \
    --host 0.0.0.0 \
    --port "${PORT}"
