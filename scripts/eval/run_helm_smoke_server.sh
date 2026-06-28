#!/bin/bash
# Start the combined pi0 + Qwen2.5-VL FastAPI server for the HeLM × LIBERO smoke test.
# Run on a GPU worker node inside the HeLM env.
#
# Usage:
#   bash scripts/llp/run_helm_smoke_server.sh [PORT]
# e.g.:
#   bash scripts/llp/run_helm_smoke_server.sh 8000
#
# After this is up, run scripts/llp/run_helm_smoke_client.sh in another shell
# (in helm_libero env) on the same node.

set -e

PORT="${1:-8000}"

PYBIN=/rlwrld2/home/gyeonghun_kim/miniconda3/envs/HeLM/bin/python
PI0_CKPT=/rlwrld2/home/gyeonghun_kim/ckpt/pi0_base
QWEN_BASE=/rlwrld2/home/gyeonghun_kim/ckpt/Qwen2.5-VL-7B-Instruct

export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

cd /rlwrld2/home/gyeonghun_kim/codes/HeLM
export PYTHONPATH=$(pwd):${PYTHONPATH}

CUDA_VISIBLE_DEVICES=0 \
${PYBIN} tools/helm_full_server.py \
    --pi0-ckpt ${PI0_CKPT} \
    --qwen-base ${QWEN_BASE} \
    --device cuda:0 \
    --host 0.0.0.0 \
    --port ${PORT}
