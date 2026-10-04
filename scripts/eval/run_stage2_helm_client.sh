#!/bin/bash
# Stage 2 HeLM evaluation client. Drives mem_7_helm_eval.py against the
# server spun up by run_stage2_helm_server.sh.
#
# Single task / single episode (smoke test). For multi-episode benchmark,
# we'll need a HeLM version of mem_8 (TODO).
#
# Usage:
#   bash scripts/eval/run_stage2_helm_client.sh [TASK_ID] [MAX_STEPS] [SEED]
#   TASK_ID=0 MAX_STEPS=200 SEED=0 bash scripts/eval/run_stage2_helm_client.sh
#
# Stage 2 is task_01 only → TASK_ID=0 (LIBERO benchmark id 0 = "1 pick up the bowl ...").

set -e

TASK_ID="${1:-${TASK_ID:-0}}"
MAX_STEPS="${2:-${MAX_STEPS:-200}}"
SEED="${3:-${SEED:-0}}"
SERVER="${SERVER:-http://localhost:8000}"
ORIENT="${ORIENT:-vflip}"            # v2-trained pi0 → vflip

PYBIN=/rlwrld2/home/gyeonghun_kim/miniconda3/envs/helm_libero/bin/python
LIBERO_SRC=/rlwrld2/home/gyeonghun_kim/codes/HeLM/libero-mem

export MUJOCO_GL=egl
export PYTHONPATH=${LIBERO_SRC}:${PYTHONPATH}

cd /rlwrld2/home/gyeonghun_kim/codes/HeLM/libero-mem/scripts

echo "[client] server=${SERVER}  task=${TASK_ID}  max_steps=${MAX_STEPS}  seed=${SEED}  orient=${ORIENT}"

TASKSPEC="${TASKSPEC:-/rlwrld2/home/gyeonghun_kim/codes/HeLM/helm_datasets/libero_taskspecs/liftput_bowl_1.json}"

${PYBIN} mem_7_helm_eval.py \
    --server "${SERVER}" \
    --task-suite libero_mem \
    --task-id "${TASK_ID}" \
    --max-steps "${MAX_STEPS}" \
    --seed "${SEED}" \
    --detect-every 10 \
    --orient "${ORIENT}" \
    --taskspec "${TASKSPEC}" \
    --save-dir ~/codes/HeLM/outputs/eval/stage2_helm/absolute_30k_${TASK_ID}_seed${SEED}
