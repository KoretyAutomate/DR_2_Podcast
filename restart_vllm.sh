#!/bin/bash
docker rm -f vllm-final 2>/dev/null
docker run --runtime nvidia --gpus all \
  --name vllm-final \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --ipc=host \
  nvcr.io/nvidia/vllm:26.01-py3 \
  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct-AWQ \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --dtype auto \
    --trust-remote-code \
    --served-model-name Qwen/Qwen2.5-32B-Instruct-AWQ \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
