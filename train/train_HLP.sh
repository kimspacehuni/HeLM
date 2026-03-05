echo "--- 🚀 Starting HLP Training ---"

MODEL_NAME="Qwen2.5-VL-7B-Instruct"
PROJECT_NAME="HLP_qwen_2.5_7b_QLoRA_r16_press_the_blue_button_ep60_1208"
DATASET_NAME="ghkim/piper_press_the_blue_button_ep60/gpt-5-mini/eval_final"

export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master-port=29935 train/train_hlp/train_vlm.py \
  --model_name_or_path /ckpt/${MODEL_NAME} \
  --dataset_dir /data/${DATASET_NAME} \
  --output_dir /result/ghkim/${PROJECT_NAME} \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --save_strategy "steps" \
  --max_steps 3000 \
  --save_steps 1000 \
  --warmup_steps 50 \
  --lr_scheduler_type "cosine" \
  --learning_rate 5e-4 \
  --bf16 True \
  --gradient_checkpointing False \
  --dataloader_num_workers 16 \
  --use_qlora True \
  --lora_r 16 \
  --lora_alpha 32 \
  --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --logging_steps 10 \
  --report_to "wandb" \
  --wandb_project RefMe \
  --run_name ${PROJECT_NAME}