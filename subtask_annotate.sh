# subtask_annotate.sh — annotation UI 실행 (수동 클릭으로 boundary 마킹)
source /rlwrld2/home/gyeonghun_kim/miniconda3/envs/HeLM/bin/activate  # 또는 별도 envs/subtask-ann
cd /rlwrld2/home/gyeonghun_kim/codes/HeLM/subtask_annotator
python -m backend.main --data-root /rlwrld2/home/gyeonghun_kim/data/LIBERO-Mem/LIBERO-Mem-LeRobot-v2-30ep/task_01 --port 8000