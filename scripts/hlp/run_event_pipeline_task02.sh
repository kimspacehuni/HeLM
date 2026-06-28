#!/bin/bash
# Pilot pipeline for task 02 (liftput_bowl, single-subtask):
#   1) extract_frames     : v2-PerTask 비디오에서 frame 추출 (annotation 입력)
#   2) annotate_app       : streamlit web UI에서 event(frame_idx) 마킹 (수동)
#   3) build_helm         : taskspec + events → HLP 학습 jsonl
#   4) validate_helm_data : 검증
#
# Step 1, 3, 4 는 자동. Step 2 는 streamlit으로 따로 띄움 (이 script가 실행 명령만 안내).

set -e

PYBIN=/rlwrld2/home/gyeonghun_kim/miniconda3/envs/HeLM/bin/python
LEROBOT_ROOT=/rlwrld2/home/gyeonghun_kim/data/LIBERO-Mem/LIBERO-Mem-LeRobot-v2-PerTask/task_01_pick_up_the_bowl_and_place_it_back_on_the_plate
OUT_ROOT=/rlwrld2/home/gyeonghun_kim/data/helm_data/liftput_bowl_1
TASKSPECS_DIR=/rlwrld2/home/gyeonghun_kim/codes/HeLM/helm_datasets/libero_taskspecs
FPS_FRAMES=5

export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

cd /rlwrld2/home/gyeonghun_kim/codes/HeLM
export PYTHONPATH=$(pwd):${PYTHONPATH}

STAGE="${1:-help}"

case "${STAGE}" in
  extract)
    echo "=== [1/4] extract_frames (lerobot mp4 → jpeg frames) ==="
    ${PYBIN} -m helm_datasets.extract_frames \
        --lerobot_root "${LEROBOT_ROOT}" \
        --out_root "${OUT_ROOT}" \
        --exterior_key observation.images.base_0_rgb \
        --wrist_key observation.images.left_wrist_0_rgb \
        --fps_frames ${FPS_FRAMES}
    echo ""
    echo "✓ frames at: ${OUT_ROOT}"
    echo "  next: bash $0 annotate"
    ;;

  annotate)
    echo "=== [2/4] annotate_app (streamlit web UI) ==="
    echo "→ Open the printed local/network URL in a browser and mark events."
    echo "  Output is written back to per-episode JSON under ${OUT_ROOT}."
    echo ""
    cd /rlwrld2/home/gyeonghun_kim/codes/HeLM
    ${PYBIN} -m streamlit run helm_datasets/annotate_app.py -- \
        --out_root "${OUT_ROOT}" \
        --fps_frames ${FPS_FRAMES}
    ;;

  build)
    echo "=== [3/4] build_helm (taskspec + events → jsonl) ==="
    ${PYBIN} -m helm_datasets.build_helm \
        --data_root "${OUT_ROOT}" \
        --out_root "${OUT_ROOT}" \
        --taskspecs_dir "${TASKSPECS_DIR}" \
        --fps_out ${FPS_FRAMES} \
        --n_images 1 \
        --val_ratio 0.1 \
        --shard_size 5000
    echo ""
    echo "✓ jsonl at: ${OUT_ROOT}/jsonl_v4 (or similar)"
    echo "  next: bash $0 validate"
    ;;

  validate)
    echo "=== [4/4] validate_helm_data ==="
    ${PYBIN} -m helm_datasets.validate_helm_data \
        --jsonl_root "${OUT_ROOT}/jsonl_v4"
    ;;

  help|*)
    cat <<USAGE
Pilot HeLM event-annotation pipeline for task 02 (liftput_bowl).

Usage:
  bash $0 <stage>

Stages:
  extract   1) Decode mp4 → jpeg frames at ${FPS_FRAMES} fps  (auto)
  annotate  2) Streamlit UI to mark event frames               (manual; opens browser)
  build     3) taskspec + events → HLP training jsonl          (auto)
  validate  4) sanity-check generated jsonl                    (auto)

Paths:
  LEROBOT_ROOT  = ${LEROBOT_ROOT}
  OUT_ROOT      = ${OUT_ROOT}
  TASKSPECS_DIR = ${TASKSPECS_DIR}
  FPS_FRAMES    = ${FPS_FRAMES}
USAGE
    ;;
esac
