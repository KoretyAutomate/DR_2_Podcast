#!/usr/bin/env bash
# Start (or restart) the Qwen3-TTS API server using the qwen3_tts conda env.
# Run init_and_start.sh first to download the model and set up the env.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV_BIN="/home/korety/miniconda3/envs/qwen3_tts/bin"
API_PORT=8082

echo "=== Qwen3-TTS Server ==="
[ -d "$SCRIPT_DIR/checkpoints" ] && [ -n "$(ls -A "$SCRIPT_DIR/checkpoints" 2>/dev/null)" ] \
    || { echo "ERROR: No checkpoints — run init_and_start.sh first"; exit 1; }

pkill -f "uvicorn tts_server:app" 2>/dev/null && echo "  Stopped previous instance." || true
sleep 1

mkdir -p logs
CHECKPOINTS_PATH="$SCRIPT_DIR/checkpoints" \
    "$CONDA_ENV_BIN/uvicorn" tts_server:app \
        --host 0.0.0.0 --port "$API_PORT" --workers 1 \
        --log-level info \
    > "$SCRIPT_DIR/logs/tts_server.log" 2>&1 &

echo "  API starting on http://localhost:$API_PORT — model load takes ~30s"
echo "  Logs: $SCRIPT_DIR/logs/tts_server.log"
echo "  Health: curl http://localhost:$API_PORT/health"
