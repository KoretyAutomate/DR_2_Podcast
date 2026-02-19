#!/usr/bin/env bash
# ============================================================
# Fish Speech Docker — GPU & Container Toolkit Verification
# Target: NVIDIA DGX Spark (GB10 / sm_121, ARM64/aarch64)
# ============================================================
set -euo pipefail

PASS=0; WARN=0; FAIL=0
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

ok()   { echo -e "${GREEN}[PASS]${NC} $*"; ((PASS++)); }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; ((WARN++)); }
fail() { echo -e "${RED}[FAIL]${NC} $*"; ((FAIL++)); }

echo "============================================================"
echo "  Fish Speech Docker — Pre-flight GPU Verification"
echo "  $(date)"
echo "============================================================"

# ── 1. Architecture ─────────────────────────────────────────
echo ""
echo "── 1. Host Architecture ──"
ARCH=$(uname -m)
if [[ "$ARCH" == "aarch64" ]]; then
    ok "Architecture: aarch64 (ARM64) ✓"
else
    fail "Unexpected architecture: $ARCH  (expected aarch64 for DGX Spark)"
fi

# ── 2. NVIDIA Driver ─────────────────────────────────────────
echo ""
echo "── 2. NVIDIA Driver ──"
if command -v nvidia-smi &>/dev/null; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    ok "GPU detected: $GPU_NAME"
    ok "Driver version: $DRIVER_VER"
    ok "Compute capability: sm_$(echo $COMPUTE_CAP | tr -d '.')"
    ok "GPU memory: $GPU_MEM"
    # Warn if driver < 570 (required for sm_121 / Blackwell + CUDA 12.8)
    DRIVER_MAJOR=$(echo "$DRIVER_VER" | cut -d. -f1)
    if [[ "$DRIVER_MAJOR" -ge 570 ]]; then
        ok "Driver >= 570 — sm_121 Blackwell support confirmed"
    elif [[ "$DRIVER_MAJOR" -ge 550 ]]; then
        warn "Driver $DRIVER_VER — sm_121 may work but ≥570 recommended for GB10"
    else
        fail "Driver $DRIVER_VER is too old for GB10 (sm_121). Upgrade to ≥570."
    fi
    # Check for sm_121 specifically
    if [[ "$COMPUTE_CAP" == "12.1" ]]; then
        ok "Compute capability 12.1 (sm_121) confirmed — DGX Spark GB10"
    else
        warn "Compute capability is $COMPUTE_CAP — expected 12.1 for DGX Spark GB10"
    fi
else
    fail "nvidia-smi not found — NVIDIA driver is not installed or not on PATH"
fi

# ── 3. NVIDIA Container Toolkit ──────────────────────────────
echo ""
echo "── 3. NVIDIA Container Toolkit ──"
if command -v nvidia-ctk &>/dev/null; then
    CTK_VER=$(nvidia-ctk --version 2>&1 | grep -oP '\d+\.\d+\.\d+' | head -1 || echo "unknown")
    ok "nvidia-ctk found (version: $CTK_VER)"
    # Check libnvidia-container
    if dpkg -l libnvidia-container1 &>/dev/null 2>&1 || rpm -q libnvidia-container 2>/dev/null; then
        ok "libnvidia-container is installed"
    elif ldconfig -p | grep -q libnvidia-container; then
        ok "libnvidia-container found in ldconfig"
    else
        warn "libnvidia-container package not confirmed — toolkit may still work"
    fi
else
    fail "nvidia-ctk not found. Install with:"
    echo "       curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    echo "       curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \\"
    echo "         sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \\"
    echo "         sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
    echo "       sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
    echo "       sudo nvidia-ctk runtime configure --runtime=docker"
    echo "       sudo systemctl restart docker"
fi

# ── 4. Docker daemon GPU runtime configuration ────────────────
echo ""
echo "── 4. Docker GPU Runtime Configuration ──"
if command -v docker &>/dev/null; then
    DOCKER_VER=$(docker --version | grep -oP '\d+\.\d+\.\d+' | head -1)
    ok "Docker found: $DOCKER_VER"

    # Check /etc/docker/daemon.json for nvidia runtime
    DAEMON_JSON="/etc/docker/daemon.json"
    if [[ -f "$DAEMON_JSON" ]]; then
        if grep -q '"nvidia"' "$DAEMON_JSON" 2>/dev/null; then
            ok "NVIDIA runtime registered in $DAEMON_JSON"
        elif grep -q 'nvidia' "$DAEMON_JSON" 2>/dev/null; then
            ok "NVIDIA configuration found in $DAEMON_JSON"
        else
            warn "$DAEMON_JSON exists but no nvidia runtime entry found"
            echo "       Run: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
        fi
    else
        warn "$DAEMON_JSON not found — runtime may not be configured"
        echo "       Run: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
    fi

    # Check docker info for nvidia runtime
    if docker info 2>/dev/null | grep -qi "nvidia"; then
        ok "Docker reports NVIDIA runtime available"
    else
        warn "NVIDIA runtime not visible in 'docker info' (may still work if daemon.json is correct)"
    fi
else
    fail "Docker not found — install Docker Engine first"
fi

# ── 5. Docker GPU smoke test ──────────────────────────────────
echo ""
echo "── 5. Docker GPU Smoke Test ──"
echo "   Pulling minimal CUDA image and running nvidia-smi inside container..."
echo "   (This may take a moment on first run)"

CUDA_IMAGE="nvcr.io/nvidia/cuda:12.8.0-base-ubuntu22.04"
if docker run --rm --gpus all "$CUDA_IMAGE" nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null; then
    ok "GPU passthrough to Docker container confirmed"
else
    # Try without nvcr image first — use a simpler test
    if docker run --rm --gpus all ubuntu:22.04 ls /dev/nvidia* 2>/dev/null | grep -q nvidia; then
        ok "/dev/nvidia* devices visible in container"
    else
        fail "GPU passthrough failed. Ensure nvidia-ctk runtime configure has been run and Docker restarted."
        echo "       Diagnosis: run 'sudo docker run --rm --gpus all ubuntu nvidia-smi'"
    fi
fi

# ── 6. Required ports availability ───────────────────────────
echo ""
echo "── 6. Port Availability ──"
for PORT in 8082 7860; do
    if ss -tlnp "sport = :$PORT" 2>/dev/null | grep -q "$PORT"; then
        warn "Port $PORT is already in use — update docker-compose.yml to use a different host port"
    else
        ok "Port $PORT is available"
    fi
done

# ── 7. Disk space ────────────────────────────────────────────
echo ""
echo "── 7. Disk Space ──"
WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
AVAIL_GB=$(df -BG "$WORKSPACE_DIR" | awk 'NR==2 {gsub("G",""); print $4}')
if [[ "$AVAIL_GB" -ge 40 ]]; then
    ok "Available disk: ${AVAIL_GB}GB (need ~30GB for image + weights)"
elif [[ "$AVAIL_GB" -ge 20 ]]; then
    warn "Available disk: ${AVAIL_GB}GB — tight; Fish Speech weights alone are ~5GB, Docker image ~15GB"
else
    fail "Available disk: ${AVAIL_GB}GB — insufficient; need at least 30GB"
fi

# ── 8. HuggingFace CLI ───────────────────────────────────────
echo ""
echo "── 8. HuggingFace CLI ──"
if command -v huggingface-cli &>/dev/null; then
    ok "huggingface-cli found — can use for weight download"
elif python3 -c "import huggingface_hub" 2>/dev/null; then
    ok "huggingface_hub Python package found (use: python3 -m huggingface_hub.commands.huggingface_cli)"
else
    warn "huggingface-cli not found — init_weights.sh will install it inside the container"
fi

# ── Summary ──────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Summary: ${PASS} passed, ${WARN} warnings, ${FAIL} failed"
echo "============================================================"
if [[ "$FAIL" -gt 0 ]]; then
    echo -e "${RED}✗ Fix failures above before proceeding.${NC}"
    exit 1
elif [[ "$WARN" -gt 0 ]]; then
    echo -e "${YELLOW}⚠ Warnings present — review before proceeding.${NC}"
    exit 0
else
    echo -e "${GREEN}✓ All checks passed — ready to build and launch Fish Speech.${NC}"
    exit 0
fi
