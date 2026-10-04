#!/bin/bash
# LIBERO-side pi0-only eval client for Round 1 fine-tuned checkpoints.
# Run on the SAME node as run_round1_server.sh, inside helm_libero env.
#
# Usage:
#   bash scripts/eval/run_round1_client.sh [TASK_ID] [MAX_STEPS] [N_EPISODES]
# e.g.:
#   bash scripts/eval/run_round1_client.sh 0 300 10
#
# Round 1 was trained on the v1 (rotated) dataset, so this client KEEPS the
# default 180° rotation when sending obs to the server. For v2 evaluation
# later, set ROTATE=0.

set -e

TASK_ID="${1:-0}"
MAX_STEPS="${2:-300}"
N_EPISODES="${3:-1}"
SERVER="${SERVER:-http://localhost:8000}"

PYBIN=/rlwrld2/home/gyeonghun_kim/miniconda3/envs/helm_libero/bin/python
LIBERO_SRC=/rlwrld2/home/gyeonghun_kim/codes/HeLM/libero-mem

export MUJOCO_GL=egl
export PYTHONPATH=${LIBERO_SRC}:${PYTHONPATH}

# Pick image-orientation transform applied to env.obs before sending to pi0.
#
# Final replay-based diagnosis (LEFT=sim_raw vs RIGHT=v2 mp4 differ only in V):
#   - v2 = canonical world view (right-side up + L/R-correct)
#   - v1 = 180°(v2) = upside-down + L/R-mirrored ← Round 1 was trained on this
#   - env.obs at runtime = mujoco_raw = upside-down + L/R-correct (vflip(world))
#
# Round 1 ckpt eval: pi0 saw v1 (upside-down + L/R-mirrored). To match at eval,
# apply hflip to mujoco_raw → upside-down + L/R-mirrored = v1's orientation.
#
#   ORIENT=hflip      → Round 1 (v1-trained ckpt)                  ← default
#   ORIENT=vflip      → Round 2 (v2-trained ckpt; canonical)
#   ORIENT=rotate180  → original buggy default (V-mismatch from v1)
#   ORIENT=none       → mujoco_raw directly (no datasets use this)
ORIENT="${ORIENT:-hflip}"

cd /rlwrld2/home/gyeonghun_kim/codes/HeLM/libero-mem/scripts

echo "[client] task=${TASK_ID}  max_steps=${MAX_STEPS}  n_episodes=${N_EPISODES}"
echo "[client] server=${SERVER}  orient=${ORIENT}"

# Run N_EPISODES rollouts back-to-back (different seeds).
for i in $(seq 0 $((N_EPISODES - 1))); do
    echo ""
    echo "=== episode ${i} (seed=${i}) ==="
    ${PYBIN} mem_6_pi0_eval.py \
        --server "${SERVER}" \
        --save-dir ~/codes/HeLM/outputs/eval/pi0_vanilla_seed0_2_hflip/${TASK_ID} \
        --task-suite libero_mem \
        --task-id "${TASK_ID}" \
        --max-steps "${MAX_STEPS}" \
        --seed "${i}" \
        --orient "${ORIENT}"
done
