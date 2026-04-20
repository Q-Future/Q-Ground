INIT_MODEL="./qg_model_multiscale_stage1_epoch_9"
SAVE_MODEL='./qg_model_multiscale_stage2'
SR='0,0,6,4,0,0'

CKPT="pixllm-7b"

deepspeed --master_port=27020 train_ds.py --precision='bf16' \
  --version=$INIT_MODEL \
  --dataset_dir='./dataset' \
  --dataset="sem_seg||multi_reason_seg||vqa||qinstruct||qground||qgroundgpt" \
  --sample_rates=$SR \
  --sem_seg_data="ade20k||cocostuff" --vqa_data="llava_v1_5_mix665k" \
  --exp_name=$CKPT \
  --vision-tower='openai/clip-vit-large-patch14-336' \
  --seg_token_num=3 \
  --num_classes_per_question=3 \
  --lr 0.0002 --beta2 0.95 --grad_accumulation_steps 3 \
  --batch_size 8 --epochs 10 --steps_per_epoch 500 --warmup_steps 100 \
  --lora_r 64 --lora_alpha 128 \
  --bce_loss_weight 2 --dice_loss_weight 0.5 \
  --model_max_length=2048 \
  --pad_train_clip_images \
  --preprocessor_config='./configs/preprocessor_448.json' \
  --resize_vision_tower \
  --resize_vision_tower_size=448 \
  --vision_tower_for_mask \
  --use_expand_question_list \
  --image_feature_scale_num=2 \
  --separate_mm_projector  \
  --mm_vision_select_layer 6,-11,-2 \
  --save_hf_model_path $SAVE_MODEL \
  --use_multi_scale_mm_projector # --debug --auto_resume
