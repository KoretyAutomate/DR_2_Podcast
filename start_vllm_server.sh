#!/bin/bash
# vLLM Server Startup Script for DR_2_Podcast
# Model: DeepSeek-R1-Distill-Qwen-32B
# Port: 8000

# Activate conda environment
source /home/korety/miniconda3/bin/activate podcast_flow

# Set CUDA library path for CUDA 13 (vLLM expects CUDA 12)
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

# Configuration
MODEL_NAME="Qwen/Qwen2.5-32B-Instruct-AWQ"
PORT=8000
MAX_MODEL_LEN=32768  # 32k context window
GPU_MEMORY_UTIL=0.9  # Use 90% of GPU memory

echo "=========================================="
echo "Starting vLLM Server for DR_2_Podcast"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Port: $PORT"
echo "Max Context: $MAX_MODEL_LEN tokens"
echo "GPU Memory: ${GPU_MEMORY_UTIL}%"
echo "=========================================="

# Kill any existing vLLM server on port 8000
lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
sleep 2

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port $PORT \
  --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization $GPU_MEMORY_UTIL \
  --dtype auto \
  --trust-remote-code \
  --served-model-name "$MODEL_NAME" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  2>&1 | tee vllm_server.log

# If server exits unexpectedly
echo ""
echo "vLLM server stopped."
echo "Check vllm_server.log for details."
