#!/bin/bash

# Set Fireworks API
export OPENAI_API_KEY="fw_3ZJ"
export OPENAI_API_BASE="https://api.fireworks.ai/inference/v1"

cd /home/jovyan/SafeDecoding/mt_bench

# Step 1: Generate answers
echo "Generating model answers..."
python gen_model_answer.py \
    --model-path meta-llama/Llama-2-7b-chat-hf \
    --model-id llama2-safedecoding \
    --bench-name mt_bench \
    --defense SafeDecoding

# Step 2: Generate judgments
echo "Running GPT-4 judgment..."
python gen_judgment.py \
    --model-list llama2-safedecoding \
    --judge-model gpt-4 \
    --mode single

# Step 3: Show results
echo "Showing results..."
python show_result.py --model-list llama2-safedecoding