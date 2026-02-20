#!/usr/bin/env bash
set -euo pipefail
CHECKPOINTS_PATH="${CHECKPOINTS_PATH:-/app/checkpoints}"
MODE="${1:-api}"
echo "=== Qwen3-TTS | Mode: $MODE | Checkpoints: $CHECKPOINTS_PATH ==="
if [ ! -d "$CHECKPOINTS_PATH" ] || [ -z "$(ls -A "$CHECKPOINTS_PATH" 2>/dev/null)" ]; then
    echo "ERROR: No model checkpoints at $CHECKPOINTS_PATH — run init_and_start.sh first"
    exit 1
fi
if [ "$MODE" = "api" ]; then
    exec uvicorn tts_server:app --host 0.0.0.0 --port 8080 --workers 1
elif [ "$MODE" = "shell" ]; then
    exec /bin/bash
else
    echo "Unknown mode: $MODE (use 'api' or 'shell')"; exit 1
fi
