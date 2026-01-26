#!/bin/bash
# LoRA Fine-tuning Script for GlobalHealthAtlas Scorer

# Model settings
MODEL_PATH="/root/Qwen3-8B"
OUTPUT_DIR="/root/autodl-tmp/Lora-LLM/saves/Qwen3-8B/lora/sft2"

# Dataset settings
DATASET_NAME="distill_psychology-10k-r1"
DATASET_DIR="/root/LLaMA-Factory/data"

# Training parameters
FINETUNING_TYPE="lora"
LORA_RANK=16
LORA_TARGET="self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj"

CUTOFF_LEN=2048
MAX_SAMPLES=5400
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=5e-5
NUM_TRAIN_EPOCHS=1.0
LR_SCHEDULER_TYPE="cosine"
WARMUP_RATIO=0.1

# Optimization
BF16=true
OVERWRITE_CACHE=true
LOGGING_STEPS=20
SAVE_STEPS=100
PREPROCESSING_WORKERS=16
DATALOADER_WORKERS=4
DDP_TIMEOUT=180000000

# Run training (using LLaMA-Factory)
CUDA_VISIBLE_DEVICES=0 python src/train.py \
    --stage sft \
    --model_name_or_path $MODEL_PATH \
    --dataset $DATASET_NAME \
    --template qwen \
    --finetuning_type $FINETUNING_TYPE \
    --lora_rank $LORA_RANK \
    --lora_target $LORA_TARGET \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --save_total_limit 3 \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --plot_loss true \
    --ddp_timeout $DDP_TIMEOUT \
    --preprocessing_num_workers $PREPROCESSING_WORKERS \
    --dataloader_num_workers $DATALOADER_WORKERS \
    --bf16 $BF16 \
    --overwrite_cache $OVERWRITE_CACHE \
    --cutoff_len $CUTOFF_LEN \
    --max_samples $MAX_SAMPLES \
    --warmup_ratio $WARMUP_RATIO
