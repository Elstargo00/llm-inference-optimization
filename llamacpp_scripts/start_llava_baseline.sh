#!/bin/bash

pkill -9 -f llama-server
sleep 3

MODEL_PATH="./models/llava-v1.5-13b-f16.gguf"
MMPROJ_PATH="./models/mmproj-model-f16.gguf"

./build/bin/llama-server \
    --model $MODEL_PATH \
    --mmproj $MMPROJ_PATH \
    --alias "llava_baseline" \
    --host 0.0.0.0 \
    --port 10000 \
    --n-gpu-layers 10 \
    --ctx-size 8192 \
    --batch-size 2048 \
    --ubatch-size 512 \
    --threads 16 \
    --n-predict 512 \
    --temp 0 \
    --seed 77 \
    --verbose