#!/bin/bash
# LIBERO-side client for the HeLM × LIBERO smoke test.
# Run on the SAME node as run_helm_smoke_server.sh, inside helm_libero env.
#
# Usage:
#   bash scripts/llp/run_helm_smoke_client.sh [TASK_ID] [MAX_STEPS]
# e.g.:
#   bash scripts/llp/run_helm_smoke_client.sh 0 200
#
# If `import libero` fails (NFS .pth staleness across worker nodes), the
# PYTHONPATH below adds the libero-mem source dir as a workaround. Permanent
# fix: re-run `pip install -e libero-mem/` in the helm_libero env on this node.

set -e

TASK_ID="${1:-0}"
MAX_STEPS="${2:-200}"
SERVER="${SERVER:-http://localhost:8000}"

PYBIN=/rlwrld2/home/gyeonghun_kim/miniconda3/envs/helm_libero/bin/python
LIBERO_SRC=/rlwrld2/home/gyeonghun_kim/codes/HeLM/libero-mem

# Note: PYTHONNOUSERSITE intentionally NOT set for helm_libero env. Many of its
# deps (torch, requests, ...) live in ~/.local user-site rather than the conda
# env site-packages. Unlike HeLM env (where transformers version war required
# user-site isolation), helm_libero has no version-sensitive deps that benefit
# from blocking user-site.
export MUJOCO_GL=egl
# Workaround for NFS .pth staleness; if libero is properly installed on this
# node, this PYTHONPATH addition is harmless.
export PYTHONPATH=${LIBERO_SRC}:${PYTHONPATH}

cd /rlwrld2/home/gyeonghun_kim/codes/HeLM/libero-mem/scripts

# Set ROTATE=0 to use the v2 (non-rotated) dataset path:
#   ROTATE=0 bash scripts/llp/run_helm_smoke_client.sh 0 200
ROTATE_FLAG=""
if [ "${ROTATE:-1}" = "0" ]; then
    ROTATE_FLAG="--no-rotate"
fi

${PYBIN} mem_7_helm_eval.py \
    --server ${SERVER} \
    --task-suite libero_mem \
    --task-id ${TASK_ID} \
    --max-steps ${MAX_STEPS} \
    --detect-every 10 \
    ${ROTATE_FLAG}
