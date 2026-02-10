#!/bin/bash
# vLLM Docker Server Startup Script for DR_2_Podcast
# Model: DeepSeek-R1-Distill-Qwen-32B
# Port: 8000

# Configuration
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
PORT=8000
MAX_MODEL_LEN=32768  # 32k context window
GPU_MEMORY_UTIL=0.9  # Use 90% of GPU memory

echo "=========================================="
echo "Starting vLLM Server (Docker) for DR_2_Podcast"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Port: $PORT"
echo "Max Context: $MAX_MODEL_LEN tokens"
echo "GPU Memory: ${GPU_MEMORY_UTIL}%"
echo "=========================================="

# Kill any existing container
docker ps -a --filter "name=vllm-server" -q | xargs docker rm -f 2>/dev/null || true

# Start vLLM server in Docker
docker run --runtime nvidia --gpus all \
  --name vllm-server \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization $GPU_MEMORY_UTIL \
  --dtype auto \
  --trust-remote-code

# If server exits unexpectedly
echo ""
echo "vLLM Docker server stopped."
echo "Check docker logs vllm-server for details."
