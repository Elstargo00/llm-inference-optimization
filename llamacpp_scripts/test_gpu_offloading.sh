#!/bin/bash

pkill -f llama-server
sleep 3

MODEL_PATH="/home/tinyg/Projects/llama.cpp/models/llava-v1.5-13b-f16.gguf"


echo "Testing offloading values for RTX 4090 (24GB VARM)..."
echo ""

for LAYERS in 10 20 30 35 40; do

    echo -n "Testing $LAYERS layers: "

    OUTPUT=$(./build/bin/llama-cli \
                --model $MODEL_PATH \
                --n-gpu-layers $LAYERS \
                --prompt "Test" \
                -n 5 2>&1 | grep "offloaded")

    echo "$OUTPUT"
done