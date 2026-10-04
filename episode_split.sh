cd /rlwrld2/home/gyeonghun_kim/codes/HeLM/subtask_annotator
source /rlwrld2/home/gyeonghun_kim/miniconda3/envs/HeLM/bin/activate

python -m scripts.subset_episodes \
  --source /rlwrld2/home/gyeonghun_kim/data/LIBERO-Mem/LIBERO-Mem-LeRobot-v2-PerTask/task_01_pick_up_the_bowl_and_place_it_back_on_the_plate \
  --dest /rlwrld2/home/gyeonghun_kim/data/LIBERO-Mem/LIBERO-Mem-LeRobot-v2-30ep/task_01 \
  --num-episodes 30
