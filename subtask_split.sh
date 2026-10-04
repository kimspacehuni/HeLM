# subtask_split.sh — annotation 후 split
cd /rlwrld2/home/gyeonghun_kim/codes/HeLM/subtask_annotator
python -m scripts.split_by_subtask \
  --source /rlwrld2/home/gyeonghun_kim/data/LIBERO-Mem/LIBERO-Mem-LeRobot-v2-30ep/task_01 \
  --dest /rlwrld2/home/gyeonghun_kim/data/LIBERO-Mem/LIBERO-Mem-LeRobot-v2-30ep/task_01_sub \
  --force --workers 4 --enc-threads 8