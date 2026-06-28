#!/bin/bash
# Stage 2 HeLM evaluation server — pi0 LLP + Qwen2.5-VL HLP (with QLoRA adapter).
#
# Loads:
#   pi0:           result/stage2_helm_pi0_task01_sub_ep30_0505/checkpoints/<PI0_STEP>/pretrained_model
#   qwen base:     ~/ckpt/Qwen2.5-VL-7B-Instruct
#   qwen adapter:  result/stage2/qwen/stage2_qwen_task01_0505/checkpoint-<QWEN_STEP>
#
# Usage:
#   bash scripts/eval/run_stage2_helm_server.sh
#   PI0_STEP=030000 QWEN_STEP=500 PORT=8000 bash scripts/eval/run_stage2_helm_server.sh
#
# Env vars:
#   PI0_STEP    — pi0 ckpt step name (default 030000)
#   QWEN_STEP   — qwen adapter checkpoint number (default 500)
#   PORT        — server port (default 8000)

set -e

PI0_STEP="${PI0_STEP:-030000}"
QWEN_STEP="${QWEN_STEP:-500}"
QWEN_STAGE="${QWEN_STAGE:-2}"   # qwen ckpt under result/stage<N>/qwen/<RUN>
PORT="${PORT:-8000}"

PYBIN=/rlwrld2/home/gyeonghun_kim/miniconda3/envs/HeLM/bin/python
PI0_RUN="${PI0_RUN:-stage4_pi0_bowl_bottle_sub_abs_xyz_libero_0517}"
QWEN_RUN="${QWEN_RUN:-stage2_qwen_task01_0505}"

PI0_CKPT=/rlwrld2/home/gyeonghun_kim/result/${PI0_RUN}/checkpoints/${PI0_STEP}/pretrained_model
QWEN_BASE=/rlwrld2/home/gyeonghun_kim/ckpt/Qwen2.5-VL-7B-Instruct
QWEN_ADAPTER=/rlwrld2/home/gyeonghun_kim/result/stage${QWEN_STAGE}/qwen/${QWEN_RUN}/checkpoint-${QWEN_STEP}

if [ ! -f "${PI0_CKPT}/config.json" ]; then
    echo "ERROR: pi0 ckpt not found at ${PI0_CKPT}" >&2
    echo "Available pi0 steps:" >&2
    ls /rlwrld2/home/gyeonghun_kim/result/${PI0_RUN}/checkpoints/ 2>&1 >&2 || true
    exit 1
fi
if [ ! -d "${QWEN_ADAPTER}" ]; then
    echo "ERROR: qwen adapter not found at ${QWEN_ADAPTER}" >&2
    echo "Available qwen checkpoints:" >&2
    ls /rlwrld2/home/gyeonghun_kim/result/stage${QWEN_STAGE}/qwen/${QWEN_RUN}/ 2>&1 >&2 || true
    exit 1
fi

export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

cd /rlwrld2/home/gyeonghun_kim/codes/HeLM
export PYTHONPATH=$(pwd):${PYTHONPATH}

echo "[server] pi0 ckpt:      ${PI0_CKPT}"
echo "[server] qwen base:     ${QWEN_BASE}"
echo "[server] qwen adapter:  ${QWEN_ADAPTER}"
echo "[server] port:          ${PORT}"

CUDA_VISIBLE_DEVICES=0 \
${PYBIN} tools/helm_full_server.py \
    --pi0-ckpt "${PI0_CKPT}" \
    --qwen-base "${QWEN_BASE}" \
    --qwen-adapter "${QWEN_ADAPTER}" \
    --device cuda:0 \
    --host 0.0.0.0 \
    --port "${PORT}"
