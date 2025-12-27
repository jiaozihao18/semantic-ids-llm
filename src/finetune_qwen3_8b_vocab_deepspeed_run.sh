#!/bin/bash
# Run script for Qwen3-8B vocabulary extension training with DeepSpeed

export WANDB_MODE=disabled  # Set to "online" if you want to use W&B
export CUDA_LAUNCH_BLOCKING=1

# Configuration
CATEGORY=Video_Games
BASE_MODEL=Qwen/Qwen3-8B
DATA_DIR=./data
OUTPUT_DIR=./models/qwen3_8b_vocab_extended

# DeepSpeed config (adjust path as needed)
DEEPSPEED_CONFIG=./config/ds_z3_bf16.json  # Or create your own config

# Training parameters (can be overridden via command line)
BATCH_SIZE=32
LEARNING_RATE=1e-3
MAX_STEPS=1000
GRADIENT_ACCUMULATION_STEPS=1

# Run training with torchrun for multi-GPU support
# For single GPU, you can use: python finetune_qwen3_8b_vocab_deepspeed.py
torchrun --nproc_per_node=8 --master_port=23324 src/finetune_qwen3_8b_vocab_deepspeed.py \
    --model_name $BASE_MODEL \
    --category $CATEGORY \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_steps $MAX_STEPS \
    --deepspeed $DEEPSPEED_CONFIG

