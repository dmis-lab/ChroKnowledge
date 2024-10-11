#!/bin/bash

# Set CUDA devices (adjust as necessary)
export CUDA_VISIBLE_DEVICES=0

# Automatically count the number of GPUs
if [[ $CUDA_VISIBLE_DEVICES == *","* ]]; then
    multi_gpu=($(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l))
else
    multi_gpu=(1)
fi

# Open source LM: Llama3.1_8B, Llama3_8B, Llama2_7B, Phi3.5_Mini, Mistral7B, SOLAR_10.7B, Gemma2_9B, Gemma_7B ...
# Priprietary LM: gpt-4o-mini, gpt-4o, gpt-3.5-turbo-0125 ...
model_name="Llama3.1_8B"
dtype="bfloat16"
device_num="auto"
gpu_util=0.90
max_tokens=50

# Domains of ChroKnowBench: General, Biomedical, Legal
domain="General"

# In our setting, we only use 0.0 for temperatrue.
temperature=0.0

# Year span for the experiments
prev_span=3
next_span=3

# Optional Huggingface token and cache directory (uncomment and set if needed)
# token="Your Huggingface token if exists"
# cache_dir="Your model directory if exists"

save_results=True

echo "Running with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, model_name=$model_name, dtype=$dtype, device_num=$device_num, gpu_util=$gpu_util, max_tokens=$max_tokens, multi_gpu=$multi_gpu, domain=$domain, temperature=$temperature, prev_span=$prev_span, next_span=$next_span, save_results=$save_results"

# Run the python script with the arguments
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ChroKnowPrompt.py \
    --model_name $model_name \
    --dtype $dtype \
    --device_num $device_num \
    --gpu_util $gpu_util \
    --multi_gpu $multi_gpu \
    --max_tokens $max_tokens \
    --domain $domain \
    --temperature $temperature \
    --prev_span $prev_span \
    --next_span $next_span \
    --save_results $save_results

# Uncomment the following lines if token or cache_dir is needed
#    --token $token \
#    --cache_dir $cache_dir \

echo "All executions completed for $model_name."
