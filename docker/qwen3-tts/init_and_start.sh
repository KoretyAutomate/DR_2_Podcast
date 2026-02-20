#!/usr/bin/env bash
# =============================================================================
# Qwen3-TTS Init & Start — Conda-based (no Docker)
#
# NOTE: Docker was abandoned because the NGC ARM64 custom torch binary is ABI-
# incompatible with any standard PyPI torchaudio build. The conda approach uses
# the host's GPU-enabled torch 2.9.0+cu130 + torchaudio 2.9.0.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_REPO="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
CONDA_ENV="qwen3_tts"
API_PORT=8082

echo "=== Qwen3-TTS Init & Start ==="
echo "[1/5] Pre-flight..."
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found"; exit 1; }
nvidia-smi -L 2>/dev/null || echo "WARNING: nvidia-smi not available"

echo "[2/5] Setting up conda env '$CONDA_ENV'..."
if conda env list | grep -q "^${CONDA_ENV} "; then
    echo "  Env already exists — checking deps..."
    conda run -n "$CONDA_ENV" python3 -c "from qwen_tts import Qwen3TTSModel" 2>/dev/null \
        && echo "  ✓ qwen-tts importable" \
        || echo "  WARNING: qwen-tts import failed, re-installing deps..."
else
    echo "  Creating env with Python 3.12..."
    conda create -n "$CONDA_ENV" python=3.12 -y
    echo "  Installing PyTorch 2.9.0+cu130 + torchaudio..."
    conda run -n "$CONDA_ENV" pip install torch==2.9.0 torchaudio==2.9.0 \
        --index-url https://download.pytorch.org/whl/cu130 -q
    echo "  Installing qwen-tts and FastAPI deps..."
    conda run -n "$CONDA_ENV" pip install --no-cache-dir \
        qwen-tts sentencepiece accelerate einops librosa sox \
        soundfile numpy "fastapi>=0.111.0" "uvicorn[standard]>=0.30.0" -q
fi

echo "[3/5] Downloading model (~3.4 GB)..."
mkdir -p checkpoints
if [ -z "$(ls -A checkpoints 2>/dev/null)" ]; then
    conda run -n "$CONDA_ENV" python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$MODEL_REPO', local_dir='checkpoints', local_dir_use_symlinks=False)
print('Download complete.')
"
else
    echo "  Already present, skipping."
fi

echo "[4/5] Starting Qwen3-TTS API server..."
mkdir -p outputs logs
# Kill any previous instance on this port
pkill -f "uvicorn tts_server:app" 2>/dev/null || true
sleep 1

CONDA_ENV_BIN="/home/korety/miniconda3/envs/${CONDA_ENV}/bin"
CHECKPOINTS_PATH="$SCRIPT_DIR/checkpoints" \
    "$CONDA_ENV_BIN/uvicorn" tts_server:app \
        --host 0.0.0.0 --port "$API_PORT" --workers 1 \
        --log-level info \
    > "$SCRIPT_DIR/logs/tts_server.log" 2>&1 &

echo "  Waiting for health (up to 3 min)..."
for i in $(seq 1 36); do
    if curl -sf "http://localhost:$API_PORT/health" >/dev/null 2>&1; then
        echo "  ✓ API healthy at http://localhost:$API_PORT"; break
    fi
    if [ "$i" -eq 36 ]; then
        echo "  ✗ Timed out — check logs/tts_server.log"; exit 1
    fi
    sleep 5
done

echo "[5/5] Smoke testing both voices..."
python3 - <<'PYEOF'
import requests, pathlib, sys
API = "http://localhost:8082"
OUT = pathlib.Path("outputs")
OUT.mkdir(exist_ok=True)
tests = [
    ("Erika", "こんにちは、エリカです。今日もよろしくお願いします。", "test_erika.wav"),
    ("Kaz",   "こんにちは、カズです。今日は最新のAIニュースをお届けします。", "test_kaz.wav"),
]
for speaker, text, fname in tests:
    print(f"  Testing {speaker}...")
    r = requests.post(f"{API}/tts", json={"text": text, "speaker": speaker}, timeout=120)
    if r.status_code == 200:
        (OUT / fname).write_bytes(r.content)
        print(f"  ✓ {fname} saved ({len(r.content):,} bytes)")
    else:
        print(f"  ✗ Failed: {r.status_code} {r.text}"); sys.exit(1)
print("\nSmoke test PASSED — listen to outputs/test_kaz.wav and outputs/test_erika.wav")
PYEOF

echo "=== Done! API at http://localhost:$API_PORT ==="
echo "    Logs: $SCRIPT_DIR/logs/tts_server.log"
echo "    To restart: bash run_server.sh"
