#!/bin/bash

pkill -f llama-server

MODEL_PATH="./models/Qwen2.5-VL-32B-Instruct-q4_k_m.gguf"
MMPROJ_PATH="./models/Qwen2.5-VL-32B-Instruct-mmproj-f16.gguf"

./build/bin/llama-server \
    --model $MODEL_PATH \
    --mmproj $MMPROJ_PATH \
    --alias "qwen_optimize_A" \
    --host 0.0.0.0 \
    --port 10000 \
    --n-gpu-layers 25 \
    --ctx-size 8192 \
    --batch-size 2048 \
    --ubatch-size 512 \
    --cont-batching \
    --flash-attn \
    --threads 16 \
    --n-predict 2048 \
    --verbose