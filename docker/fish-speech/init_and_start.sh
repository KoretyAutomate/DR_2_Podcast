#!/usr/bin/env bash
# ============================================================
# Fish Speech V1.5 — Full Initialization Script
#
# What this does, in order:
#   1. Pre-flight GPU verification
#   2. Create host-side directory structure
#   3. Build the Docker image (ARM64 / sm_121 Blackwell)
#   4. Download model weights from HuggingFace (host-side)
#   5. Start the API server container
#   6. Smoke-test the API endpoint
#
# Usage:
#   chmod +x init_and_start.sh
#   ./init_and_start.sh
#
#   Optional env vars:
#     HF_TOKEN=hf_xxx ./init_and_start.sh   # for gated HF repos
#     SKIP_BUILD=1 ./init_and_start.sh       # skip image rebuild
#     SKIP_DOWNLOAD=1 ./init_and_start.sh    # skip weight download
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
log()  { echo -e "${GREEN}▶${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC}  $*"; }
die()  { echo -e "${RED}✗${NC}  $*" >&2; exit 1; }
step() { echo -e "\n${BOLD}══ $* ${NC}"; }

IMAGE_NAME="fish-speech:v1.5-arm64"
CONTAINER_NAME="fish-speech-api"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints/fish-speech-1.5"
HF_REPO="fishaudio/fish-speech-1.5"
API_PORT="${API_PORT:-8082}"   # 8080 is SearXNG — Fish Speech uses 8082
HF_TOKEN="${HF_TOKEN:-}"

# ── Step 1: Pre-flight verification ──────────────────────────
step "1/6  Pre-flight GPU Verification"
if [[ -x "${SCRIPT_DIR}/verify_gpu_setup.sh" ]]; then
    bash "${SCRIPT_DIR}/verify_gpu_setup.sh" || {
        warn "Some verification checks failed. Continuing anyway..."
        warn "Fix any [FAIL] items before reporting issues."
    }
else
    warn "verify_gpu_setup.sh not found — skipping pre-flight checks"
fi

# ── Step 2: Directory structure ───────────────────────────────
step "2/6  Creating Host Directories"
mkdir -p \
    "${CHECKPOINT_DIR}" \
    "${SCRIPT_DIR}/reference_audio" \
    "${SCRIPT_DIR}/outputs"
log "Checkpoint dir:     ${CHECKPOINT_DIR}"
log "Reference audio:    ${SCRIPT_DIR}/reference_audio"
log "Output dir:         ${SCRIPT_DIR}/outputs"

# ── Step 3: Build Docker image ────────────────────────────────
step "3/6  Building Docker Image (fish-speech:v1.5-arm64)"
if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
    warn "SKIP_BUILD=1 — skipping image build (using existing image)"
    docker image inspect "$IMAGE_NAME" &>/dev/null \
        || die "Image $IMAGE_NAME not found. Remove SKIP_BUILD=1 to build it."
else
    log "Building $IMAGE_NAME ..."
    log "(First build: ~20-40 min — pulls 18GB NGC base + compiles deps)"
    log "Subsequent builds: ~2-5 min (Docker layer cache)"
    echo ""
    docker build \
        --platform linux/arm64 \
        --build-arg FISH_SPEECH_TAG=v1.5.0 \
        --tag "$IMAGE_NAME" \
        "${SCRIPT_DIR}" \
        2>&1 | tee "${SCRIPT_DIR}/build.log"
    log "Image built successfully: $IMAGE_NAME"
fi

# ── Step 4: Download model weights ───────────────────────────
step "4/6  Downloading Fish Speech V1.5 Weights from HuggingFace"

# Required weight files for Fish Speech 1.5
REQUIRED_FILES=(
    "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
    "model.pth"
    "config.json"
)

# Check if weights already exist
ALL_PRESENT=1
for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "${CHECKPOINT_DIR}/${f}" ]]; then
        ALL_PRESENT=0
        break
    fi
done

if [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
    warn "SKIP_DOWNLOAD=1 — skipping weight download"
elif [[ "$ALL_PRESENT" -eq 1 ]]; then
    log "All required weight files already present in ${CHECKPOINT_DIR}"
    log "Skipping download (use SKIP_DOWNLOAD=0 to force re-download)"
else
    log "Downloading weights from HuggingFace: ${HF_REPO}"
    log "Destination: ${CHECKPOINT_DIR}"
    log "(~4-6 GB download — may take several minutes)"
    echo ""

    # Run download inside the container using the built image.
    # This guarantees the correct huggingface_hub version and avoids
    # host Python environment conflicts.
    HF_TOKEN_ARG=""
    if [[ -n "$HF_TOKEN" ]]; then
        HF_TOKEN_ARG="-e HF_TOKEN=${HF_TOKEN}"
        log "Using HF_TOKEN from environment"
    fi

    docker run --rm \
        --platform linux/arm64 \
        -v "${CHECKPOINT_DIR}:/download_target" \
        ${HF_TOKEN_ARG} \
        "$IMAGE_NAME" /bin/bash -c "
            set -e
            pip install -q huggingface_hub[cli] 2>/dev/null || true
            echo 'Downloading Fish Speech 1.5 weights...'
            python3 -c \"
import os, sys
from huggingface_hub import snapshot_download

token = os.getenv('HF_TOKEN') or None
print(f'HF token: {\\\"set\\\" if token else \\\"not set (public repo, OK)\\\"}')

snapshot_download(
    repo_id='${HF_REPO}',
    local_dir='/download_target',
    token=token,
    ignore_patterns=['*.md', 'README*', '.gitattributes'],
)
print('Download complete.')
\"
        "
    log "Weights downloaded to ${CHECKPOINT_DIR}"
fi

# Verify all required files are present after download
echo ""
log "Verifying checkpoint files:"
MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    if [[ -f "${CHECKPOINT_DIR}/${f}" ]]; then
        SIZE=$(du -sh "${CHECKPOINT_DIR}/${f}" | cut -f1)
        echo "  ✓  ${f}  (${SIZE})"
    else
        echo "  ✗  ${f}  — MISSING"
        MISSING=1
    fi
done
if [[ "$MISSING" -eq 1 ]]; then
    die "Some weight files are missing. Check HuggingFace access and re-run."
fi

# ── Step 5: Launch API server ─────────────────────────────────
step "5/6  Starting Fish Speech API Server"

# Stop and remove any existing container with the same name
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log "Removing existing container: ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
fi

log "Launching container: ${CONTAINER_NAME} on port ${API_PORT}"
docker run -d \
    --name "${CONTAINER_NAME}" \
    --platform linux/arm64 \
    --restart unless-stopped \
    --gpus all \
    -p "${API_PORT}:8080" \
    -v "${CHECKPOINT_DIR}:/app/checkpoints/fish-speech-1.5:ro" \
    -v "${SCRIPT_DIR}/reference_audio:/app/reference_audio" \
    -v "${SCRIPT_DIR}/outputs:/app/outputs" \
    -e CHECKPOINTS_PATH=/app/checkpoints/fish-speech-1.5 \
    -e API_PORT=8080 \
    -e API_WORKERS=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    -e TOKENIZERS_PARALLELISM=false \
    -e MODE=api \
    "$IMAGE_NAME" api

log "Container started (logs: docker logs -f ${CONTAINER_NAME})"

# ── Step 6: Smoke test the API ────────────────────────────────
step "6/6  Smoke-Testing the API Endpoint"

log "Waiting for server to initialize (up to 90s)..."
MAX_WAIT=90
INTERVAL=5
ELAPSED=0
HTTP_OK=0

while [[ "$ELAPSED" -lt "$MAX_WAIT" ]]; do
    if curl -sf "http://localhost:${API_PORT}/v1/health" &>/dev/null; then
        HTTP_OK=1
        break
    fi
    # Some versions use /health instead of /v1/health
    if curl -sf "http://localhost:${API_PORT}/health" &>/dev/null; then
        HTTP_OK=1
        break
    fi
    printf "  Waiting... (%ds elapsed)\r" "$ELAPSED"
    sleep "$INTERVAL"
    ELAPSED=$((ELAPSED + INTERVAL))
done
echo ""

if [[ "$HTTP_OK" -eq 1 ]]; then
    log "API server is healthy at http://localhost:${API_PORT}"

    # Send a short test TTS request
    log "Sending test TTS request (Japanese)..."
    TEST_OUTPUT="${SCRIPT_DIR}/outputs/smoke_test.wav"
    HTTP_STATUS=$(curl -s -o "$TEST_OUTPUT" -w "%{http_code}" \
        -X POST "http://localhost:${API_PORT}/v1/tts" \
        -H "Content-Type: application/json" \
        -d '{
            "text": "こんにちは、これはフィッシュスピーチのテストです。",
            "reference_id": null,
            "format": "wav",
            "mp3_bitrate": 64,
            "normalize": true,
            "latency": "normal"
        }' 2>/dev/null || echo "000")

    if [[ "$HTTP_STATUS" == "200" ]]; then
        FILESIZE=$(du -sh "$TEST_OUTPUT" 2>/dev/null | cut -f1 || echo "?")
        log "TTS smoke test PASSED — output: ${TEST_OUTPUT} (${FILESIZE})"
    else
        warn "TTS request returned HTTP ${HTTP_STATUS}"
        warn "Check 'docker logs ${CONTAINER_NAME}' for details"
        warn "Server may still be loading — retry in 30s"
    fi
else
    warn "API server did not respond within ${MAX_WAIT}s"
    warn "Check logs: docker logs -f ${CONTAINER_NAME}"
fi

# ── Done ──────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Fish Speech V1.5 Setup Complete${NC}"
echo -e "${BOLD}══════════════════════════════════════════════════════${NC}"
echo ""
echo "  API server:  http://localhost:${API_PORT}"
echo "  Health:      http://localhost:${API_PORT}/v1/health"
echo "  TTS endpoint: POST http://localhost:${API_PORT}/v1/tts"
echo ""
echo "  Container logs:  docker logs -f ${CONTAINER_NAME}"
echo "  Stop:            docker stop ${CONTAINER_NAME}"
echo "  Restart:         docker start ${CONTAINER_NAME}"
echo ""
echo "  To launch WebUI (port 7860):"
echo "    docker compose --profile webui up fish-speech-webui"
echo ""
echo "  Next step → integrate into DR_2_Podcast:"
echo "    Update audio_engine.py to call FishSpeech API at localhost:${API_PORT}"
echo ""
