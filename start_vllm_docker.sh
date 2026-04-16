#!/bin/bash
# vLLM Docker Server Startup Script
# Model:  RedHatAI/Qwen3.5-122B-A10B-NVFP4 (NVFP4 pre-quantized, ~75.6 GB)
# Image:  ghcr.io/bjk110/vllm-spark:v019-ngc2603
#         (vLLM 0.19.1 · FlashInfer 0.6.7 · SM121 source build · NGC 26.03)
# Source: https://github.com/JungkwanBan/spark_vllm_docker
# Port:   8000

# Mount the full model directory (not just snapshot) so that symlinks to blobs/ resolve inside the container
MODEL_HOST_PATH="$HOME/.cache/huggingface/hub/models--RedHatAI--Qwen3.5-122B-A10B-NVFP4"
MODEL_CONTAINER_BASE="/models/RedHatAI_Qwen3.5-122B-A10B-NVFP4"
MODEL_CONTAINER_PATH="${MODEL_CONTAINER_BASE}/snapshots/49d19c108259a21450c40b8af38828b0a97390d8"
SERVED_MODEL_NAME="RedHatAI/Qwen3.5-122B-A10B-NVFP4"
ENTRYPOINT_DIR="/tmp/spark-vllm-docker2"   # cloned from spark_vllm_docker

PORT=8000
MAX_MODEL_LEN=32768       # 32k context; model supports up to 262k
MAX_NUM_SEQS=4
MAX_NUM_BATCHED_TOKENS=32768
GPU_MEMORY_UTIL=0.82      # 82% of 119.7GiB (~98.1GiB): 72GiB weights + ~26GiB KV cache
                          # IMPORTANT: Stop Qwen3-TTS BEFORE running this script.
                          # GB10 has no GDS support (nogds_force.patch), so weight loading
                          # uses CPU-intermediate copies: peak = 72GB weights + CPU buffer.
                          # With Qwen3-TTS running (~6GiB), peak hits ~115GiB > 119GiB → OOM.
                          # Restart Qwen3-TTS after vLLM is serving (steady-state is ~102GiB OK).
                          # First-run compile: ~30 min total. Subsequent starts: ~5 min (cached).

echo "=========================================="
echo "Starting vLLM Server (Docker)"
echo "=========================================="
echo "Model: $SERVED_MODEL_NAME"
echo "Port:  $PORT"
echo "Context: $MAX_MODEL_LEN tokens"
echo "GPU Memory: ${GPU_MEMORY_UTIL} (~112 GB of 128 GB)"
echo "=========================================="

# Kill any existing container
docker ps -a --filter "name=vllm-server" -q | xargs docker rm -f 2>/dev/null || true
sleep 2

# Persistent vLLM cache (avoids kernel recompilation on each restart)
mkdir -p /home/korety/.vllm-cache/vllm

# Start vLLM via the unified spark entrypoint
# ROLE=head + TP_SIZE=1 → standalone serve (no Ray, no multi-node overhead)
# Key SM121/NVFP4 notes (from redhatai-122b-nvfp4.env preset):
#   - No --quantization flag: RedHatAI model is compressed-tensors format, auto-detected
#   - VLLM_USE_FLASHINFER_MOE_FP4=0: use native cutlass_moe_fp4 (SM12x compatible)
#   - VLLM_NVFP4_MOE_FORCE_MARLIN=0: stable CUTLASS path
#   - Thinking mode disabled at call sites via /no_think prefix
docker run --runtime nvidia --gpus all \
  --name vllm-server \
  -v "${MODEL_HOST_PATH}:${MODEL_CONTAINER_BASE}:ro" \
  -v "/home/korety/.vllm-cache/vllm:/root/.cache/vllm" \
  -v "${ENTRYPOINT_DIR}/entrypoint.sh:/entrypoint.sh:ro" \
  -v "${ENTRYPOINT_DIR}/patches:/patches:ro" \
  -p "${PORT}:8000" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e ROLE=head \
  -e TP_SIZE=1 \
  -e MODEL_CONTAINER_PATH="${MODEL_CONTAINER_PATH}" \
  -e SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
  -e HOST_PORT=8000 \
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTIL}" \
  -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
  -e VLLM_EXTRA_ARGS="--enable-chunked-prefill --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes" \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=0 \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_NVFP4_MOE_FORCE_MARLIN=0 \
  -e TORCH_MATMUL_PRECISION=high \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ghcr.io/bjk110/vllm-spark:v019-ngc2603 \
  /entrypoint.sh

echo ""
echo "vLLM Docker server stopped."
echo "Check docker logs vllm-server for details."
