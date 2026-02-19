#!/usr/bin/env bash
# ============================================================
# Fish Speech V1.5 — Container Entrypoint
#
# MODE (env var or first argument):
#   api    → REST API server on port 8080 (default, use for DR_2_Podcast)
#   webui  → Gradio UI on port 7860
#   both   → API server + WebUI simultaneously
#   shell  → Drop to bash (for debugging)
# ============================================================
set -euo pipefail

MODE="${1:-${MODE:-api}}"
CHECKPOINT="${CHECKPOINTS_PATH:-/app/checkpoints/fish-speech-1.5}"
API_PORT="${API_PORT:-8080}"
WEBUI_PORT="${WEBUI_PORT:-7860}"
API_WORKERS="${API_WORKERS:-1}"

# Verify checkpoints are present before starting
check_checkpoints() {
    local required_files=(
        "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
        "model.pth"
    )
    local missing=0
    for f in "${required_files[@]}"; do
        if [[ ! -f "${CHECKPOINT}/${f}" ]]; then
            echo "✗ Missing checkpoint: ${CHECKPOINT}/${f}"
            missing=1
        fi
    done
    if [[ "$missing" -eq 1 ]]; then
        echo ""
        echo "ERROR: Model weights not found at ${CHECKPOINT}"
        echo "Run the weight download step first:"
        echo "  docker exec fish-speech /app/download_weights.sh"
        echo "  — or — run init_and_start.sh from the host"
        exit 1
    fi
    echo "✓ Checkpoints verified at ${CHECKPOINT}"
}

start_api() {
    check_checkpoints
    echo "Starting Fish Speech API server on 0.0.0.0:${API_PORT} ..."
    exec python tools/api_server.py \
        --listen "0.0.0.0:${API_PORT}" \
        --checkpoint-path "${CHECKPOINT}" \
        --workers "${API_WORKERS}"
}

start_webui() {
    check_checkpoints
    echo "Starting Fish Speech WebUI on 0.0.0.0:${WEBUI_PORT} ..."
    exec python -m tools.webui \
        --listen \
        --server-name "0.0.0.0" \
        --server-port "${WEBUI_PORT}" \
        --checkpoint-path "${CHECKPOINT}"
}

case "$MODE" in
    api)
        start_api
        ;;
    webui)
        start_webui
        ;;
    both)
        check_checkpoints
        echo "Starting API server (background) + WebUI (foreground)..."
        python tools/api_server.py \
            --listen "0.0.0.0:${API_PORT}" \
            --checkpoint-path "${CHECKPOINT}" \
            --workers "${API_WORKERS}" &
        API_PID=$!
        echo "API server PID: ${API_PID}"
        # Give API a moment to bind
        sleep 3
        exec python -m tools.webui \
            --listen \
            --server-name "0.0.0.0" \
            --server-port "${WEBUI_PORT}" \
            --checkpoint-path "${CHECKPOINT}"
        ;;
    shell|bash|zsh)
        echo "Dropping to shell for debugging..."
        exec /bin/bash
        ;;
    *)
        echo "Unknown MODE: $MODE"
        echo "Usage: MODE=api|webui|both|shell"
        exit 1
        ;;
esac
