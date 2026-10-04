#!/bin/bash
# Multi-task multi-episode benchmark for a pi0 ckpt on LIBERO-Mem.
# Drives mem_8_pi0_benchmark.py through all 10 tasks with N episodes each.
#
# Prereqs:
#   - pi0 server already running (see scripts/eval/run_round1_server.sh /
#     run_round2_server.sh) on the same node
#   - helm_libero env on this node has libero + h5py + imageio + requests
#
# Usage:
#   c
#
# Env vars (all optional):
#   ORIENT       — vflip (Round 2 / v2-trained ckpt) | hflip (Round 1 / v1)
#   N_EPISODES   — episodes per task (default 50, paper-aligned)
#   TASKS        — space-separated task ids (default "0 1 2 3 4 5 6 7 8 9")
#   MAX_STEPS    — per-episode rollout cap (default 300)
#   SAVE_DIR     — output dir (default repo/outputs/eval/benchmark_<orient>_<ts>)
#   SERVER       — server URL (default http://localhost:8000)
#   NO_VIDEO     — set to 1 to skip per-episode mp4 saving (logs only; saves disk)
#
# Examples:
#   ORIENT=vflip bash scripts/eval/run_benchmark.sh                     # full benchmark
#   ORIENT=hflip N_EPISODES=10 TASKS="0 2" bash scripts/eval/run_benchmark.sh
#   NO_VIDEO=1 bash scripts/eval/run_benchmark.sh                       # logs only

set -e

ORIENT="${ORIENT:-vflip}"
N_EPISODES="${N_EPISODES:-20}"            # paper protocol: N=20 validation demos
TASKS="${TASKS:-0 1 2 3 4 5 6 7 8 9}"
ACTION_HORIZON="${ACTION_HORIZON:-10}"    # pi0 generates 50; openpi uses first 10
DEMO_START="${DEMO_START:-100}"           # eval on held-out demos 100..119 by default
MAX_STEPS="${MAX_STEPS:-}"                # default: per-task TASK_LENGTHS lookup
SERVER="${SERVER:-http://localhost:8000}"

PYBIN=/rlwrld2/home/gyeonghun_kim/miniconda3/envs/helm_libero/bin/python
LIBERO_SRC=/rlwrld2/home/gyeonghun_kim/codes/HeLM/libero-mem

export MUJOCO_GL=egl
export PYTHONPATH=${LIBERO_SRC}:${PYTHONPATH}

NO_VIDEO_FLAG=""
if [ "${NO_VIDEO:-0}" = "1" ]; then
    NO_VIDEO_FLAG="--no-video"
fi

SAVE_DIR_FLAG=""
if [ -n "${SAVE_DIR}" ]; then
    SAVE_DIR_FLAG="--save-dir ${SAVE_DIR}"
fi

cd ${LIBERO_SRC}/scripts

MAX_STEPS_FLAG=""
if [ -n "${MAX_STEPS}" ]; then
    MAX_STEPS_FLAG="--max-steps ${MAX_STEPS}"
fi

echo "[bench] server=${SERVER}  orient=${ORIENT}  n_episodes=${N_EPISODES}"
echo "[bench] action_horizon=${ACTION_HORIZON}  demo_start=${DEMO_START}  tasks=[${TASKS}]"

${PYBIN} mem_8_pi0_benchmark.py \
    --server "${SERVER}" \
    --task-suite libero_mem \
    --task-ids ${TASKS} \
    --n-episodes ${N_EPISODES} \
    --demo-start ${DEMO_START} \
    --action-horizon ${ACTION_HORIZON} \
    ${MAX_STEPS_FLAG} \
    --orient "${ORIENT}" \
    ${NO_VIDEO_FLAG} \
    ${SAVE_DIR_FLAG}
