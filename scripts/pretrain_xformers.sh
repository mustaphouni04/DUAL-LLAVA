#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

MODEL_VERSION=gemma-3-1b-it
# MODEL_VERSION=llama-2-7b-chat

########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=plain
########### DO NOT CHANGE ###########

deepspeed ../llava/train/train_xformers.py \
    --deepspeed zero2.json \
    --model_name_or_path google/gemma-3-1b-it \
    --version $PROMPT_VERSION \
    --data_path ../../cc3m_filtered/blip_laion_cc_sbu_558k.json \
    --image_folder ../../cc3m_filtered/images \
    --vision_tower google/siglip2-large-patch16-512 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 False \
    --output_dir ./checkpoints/llava-$MODEL_VERSION-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 6000 \
    --save_total_limit 1 \
    --learning_rate 0.00048 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 1000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
