#!/bin/bash
# vLLM Docker Server Startup Script for DR_2_Podcast
# Model: Qwen3-32B-AWQ (thinking mode disabled via /no_think in prompts)
# Port: 8000
# Context: 64K via YaRN rope scaling (config.json patched)

# Configuration
MODEL_NAME="Qwen/Qwen3-32B-AWQ"
PORT=8000
MAX_MODEL_LEN=65536  # 64k context window (YaRN-extended from 40k base)
GPU_MEMORY_UTIL=0.8  # Use 80% of GPU memory (~102GB), leaves ~26GB for system/Ollama

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
# Note: Uses v0.13.0 image (pinned) due to CUDA driver compat with GB10
# Note: VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 required for YaRN-extended context
docker run --runtime nvidia --gpus all \
  --name vllm-server \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --ipc=host \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  vllm/vllm-openai:v0.13.0 \
  --model "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization $GPU_MEMORY_UTIL \
  --dtype auto \
  --trust-remote-code \
  --enforce-eager

# If server exits unexpectedly
echo ""
echo "vLLM Docker server stopped."
echo "Check docker logs vllm-server for details."
