#!/bin/bash
# vLLM Docker Server Startup Script
# Model:  Intel/Qwen3.5-122B-A10B-int4-AutoRound (INT4 AutoRound + FP8 KV cache, ~65 GB)
# Image:  ghcr.io/bjk110/vllm-spark:v019-ngc2603
#         (vLLM 0.19.1 · FlashInfer 0.6.7 · SM121 source build · NGC 26.03)
# Source: https://github.com/JungkwanBan/SPARK_Qwen3.5-122B-A10B-NVFP4 (multi-quant)
# Port:   8000

# Mount the full model directory (not just snapshot) so that symlinks to blobs/ resolve inside the container.
# Snapshot path resolved dynamically from refs/main so future model updates don't require manual hash bumps.
MODEL_HOST_PATH="$HOME/.cache/huggingface/hub/models--Intel--Qwen3.5-122B-A10B-int4-AutoRound"
MODEL_CONTAINER_BASE="/models/Intel_Qwen3.5-122B-A10B-int4-AutoRound"
MODEL_SNAPSHOT_HASH="$(cat ${MODEL_HOST_PATH}/refs/main 2>/dev/null)"
MODEL_CONTAINER_PATH="${MODEL_CONTAINER_BASE}/snapshots/${MODEL_SNAPSHOT_HASH}"
SERVED_MODEL_NAME="Intel/Qwen3.5-122B-A10B-int4-AutoRound"
ENTRYPOINT_DIR="/home/korety/opt/spark_vllm_docker"   # cloned from spark_vllm_docker (persistent; /tmp is wiped on reboot)

PORT=8000
MAX_MODEL_LEN=65536       # 64k context; model supports up to 262k. Phase 4 CrewAI ReAct +
                          # tool observations hit the 32k ceiling on Japanese runs, so 65k.
MAX_NUM_SEQS=4
MAX_NUM_BATCHED_TOKENS=65536
GPU_MEMORY_UTIL=0.82      # 82% of ~121GiB unified RAM (~99.8GiB). Reduced from Nemotron's
                          # 0.88 (2026-04-30) — INT4 weights (~65GB) + FP8 KV cache halve the
                          # KV memory vs Nemotron NVFP4 + BF16 KV. Estimated headroom at 65k×4
                          # is ~14GB. GATE: if vLLM startup logs "Available KV cache memory:
                          # -X GiB", bump to 0.85 and retry.
                          # GB10 has no GDS support (nogds_force.patch), so weight loading
                          # uses CPU-intermediate copies. First-run compile: ~30 min total.
                          # Subsequent starts: ~5 min (cached).
                          # VOICEVOX runs on CPU (Docker) — no GPU memory conflict.

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
# Key SM121/INT4 AutoRound notes (from intel-122b-int4.env preset):
#   - No --quantization flag: AutoRound checkpoint is auto-detected by vLLM Marlin
#   - VLLM_MARLIN_USE_ATOMIC_ADD=1: required for INT4 AutoRound numerical stability
#   - --kv-cache-dtype fp8: halves KV memory vs BF16, enables 65k context at 0.82 GPU util
#   - VLLM_USE_FLASHINFER_MOE_FP4=0 / VLLM_NVFP4_MOE_FORCE_MARLIN=0: NVFP4 flags inert for INT4
#     but kept to avoid unintended fallback if image defaults change
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
  -e VLLM_EXTRA_ARGS="--enable-chunked-prefill --kv-cache-dtype fp8 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes" \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_NVFP4_MOE_FORCE_MARLIN=0 \
  -e TORCH_MATMUL_PRECISION=high \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ghcr.io/bjk110/vllm-spark:v019-ngc2603 \
  /entrypoint.sh

echo ""
echo "vLLM Docker server stopped."
echo "Check docker logs vllm-server for details."
