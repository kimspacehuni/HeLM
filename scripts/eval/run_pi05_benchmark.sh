#!/bin/bash
# pi0/pi05-only benchmark on LIBERO-Mem (no HLP/memory).
# Drives mem_8_pi0_benchmark.py — sends the full LIBERO task.language to the
# server's /predict (no /detect, no /update).
#
# For pi05 zero-shot ceiling check, this is the correct driver: pi05_libero
# was trained on full LIBERO tasks, not on HeLM sub-task commands.
#
# Prereqs:
#   - pi05 server already running (run_pi05_server.sh) OR pi0 server
#
# Env vars (all optional, mirror run_helm_benchmark.sh):
#   ORIENT          — vflip | hflip | rotate180 | none (default vflip)
#   N_EPISODES      — episodes per task (default 10)
#   TASKS           — space-separated LIBERO suite task ids (default "0")
#   DEMO_START      — first demo index (default 100 = val split)
#   MAX_STEPS       — per-episode rollout cap
#   ACTION_HORIZON  — closed-loop replan window (default 10)
#   SAVE_DIR        — output dir
#   SERVER          — server URL (default http://localhost:8000)
#   NO_VIDEO        — set to 1 to skip mp4
#
# LIBERO suite id mapping:
#   suite[0]  '1  pick up the bowl ...'
#   suite[1]  '2  lift the bottle ...'
#   suite[2]  '3  lift the bowl ... 3 times'
#   ...

set -e

ORIENT="${ORIENT:-vflip}"
N_EPISODES="${N_EPISODES:-10}"
TASKS="${TASKS:-0}"
DEMO_START="${DEMO_START:-100}"
ACTION_HORIZON="${ACTION_HORIZON:-10}"
MAX_STEPS="${MAX_STEPS:-}"
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

MAX_STEPS_FLAG=""
if [ -n "${MAX_STEPS}" ]; then
    MAX_STEPS_FLAG="--max-steps ${MAX_STEPS}"
fi

cd ${LIBERO_SRC}/scripts

echo "[pi05-bench] server=${SERVER}  orient=${ORIENT}  n_episodes=${N_EPISODES}"
echo "[pi05-bench] tasks=[${TASKS}]  demo_start=${DEMO_START}  action_horizon=${ACTION_HORIZON}"

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
