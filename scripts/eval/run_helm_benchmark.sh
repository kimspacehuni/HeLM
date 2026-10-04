#!/bin/bash
# Multi-task multi-episode HeLM benchmark (HLP/LLP).
# Drives mem_9_helm_benchmark.py against a HeLM full server (pi0 + Qwen).
#
# Prereqs:
#   - HeLM full server already running (run_stage2_helm_server.sh)
#   - helm_libero env on this node has libero + h5py + imageio + requests
#
# Env vars (all optional):
#   ORIENT          — vflip (Round 2 / v2-trained ckpt) | hflip | rotate180 | none
#   N_EPISODES      — episodes per task (default 20)
#   TASKS           — space-separated LIBERO suite task ids (NOT dataset task_index)
#                     Default "0" = "1 pick up the bowl ..."
#   DEMO_START      — first demo index (default 100; 0..99 = train, 100..119 = val)
#   MAX_STEPS       — per-episode rollout cap (default lookup TASK_LENGTHS)
#   DETECT_EVERY    — run /detect every N env steps (default 10)
#   TASKSPECS_DIR   — directory of v4 taskspec JSONs
#   TASKSPEC_MAP    — overrides "0=liftput_bowl_1.json,2=liftput_bowl_3.json"
#   SAVE_DIR        — output dir (default repo/outputs/eval/helm_benchmark_<orient>_<ts>)
#   SERVER          — server URL (default http://localhost:8000)
#   NO_VIDEO        — set to 1 to skip per-episode mp4 (logs only)
#
# Examples:
#   bash scripts/eval/run_helm_benchmark.sh
#   N_EPISODES=10 DEMO_START=0 TASKS="0" bash scripts/eval/run_helm_benchmark.sh
#   TASKS="0 2 4 5" bash scripts/eval/run_helm_benchmark.sh   # bowl 1/3/5/7
#
# LIBERO suite id ↔ dataset task_index ↔ taskspec mapping (see project notes):
#   suite[0]  '1  pick up the bowl ...'           → liftput_bowl_1
#   suite[1]  '2  lift the bottle ...'            → liftput_bottle_1
#   suite[2]  '3  lift the bowl ... 3 times'      → liftput_bowl_3
#   suite[3]  '4  pick up the bottle ... 3 times' → liftput_bottle_3
#   suite[4]  '5  lift the bowl ... 5 times'      → liftput_bowl_5
#   suite[5]  '6  pick up the bowl ... 7 times'   → liftput_bowl_7

set -e

ORIENT="${ORIENT:-vflip}"
N_EPISODES="${N_EPISODES:-20}"
TASKS="${TASKS:-0}"
DEMO_START="${DEMO_START:-100}"
DETECT_EVERY="${DETECT_EVERY:-10}"
ACTION_HORIZON="${ACTION_HORIZON:-10}"
DETECT_STREAK_REQUIRED="${DETECT_STREAK_REQUIRED:-1}"
POST_UPDATE_COOLDOWN="${POST_UPDATE_COOLDOWN:-30}"
MAX_STEPS="${MAX_STEPS:-}"
SERVER="${SERVER:-http://localhost:8000}"
TASKSPECS_DIR="${TASKSPECS_DIR:-/rlwrld2/home/gyeonghun_kim/codes/HeLM/helm_datasets/libero_taskspecs}"
TASKSPEC_MAP="${TASKSPEC_MAP:-}"

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

TASKSPEC_MAP_FLAG=""
if [ -n "${TASKSPEC_MAP}" ]; then
    TASKSPEC_MAP_FLAG="--taskspec-map ${TASKSPEC_MAP}"
fi

cd ${LIBERO_SRC}/scripts

echo "[helm-bench] server=${SERVER}  orient=${ORIENT}  n_episodes=${N_EPISODES}"
echo "[helm-bench] tasks=[${TASKS}]  demo_start=${DEMO_START}  detect_every=${DETECT_EVERY}"
echo "[helm-bench] taskspecs_dir=${TASKSPECS_DIR}"

${PYBIN} mem_9_helm_benchmark.py \
    --server "${SERVER}" \
    --task-suite libero_mem \
    --task-ids ${TASKS} \
    --n-episodes ${N_EPISODES} \
    --demo-start ${DEMO_START} \
    --detect-every ${DETECT_EVERY} \
    --detect-streak-required ${DETECT_STREAK_REQUIRED} \
    --post-update-cooldown ${POST_UPDATE_COOLDOWN} \
    --action-horizon ${ACTION_HORIZON} \
    ${MAX_STEPS_FLAG} \
    --orient "${ORIENT}" \
    --taskspecs-dir "${TASKSPECS_DIR}" \
    ${TASKSPEC_MAP_FLAG} \
    ${NO_VIDEO_FLAG} \
    ${SAVE_DIR_FLAG}
