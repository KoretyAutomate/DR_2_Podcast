# Fish Speech V1.5 — Docker Setup for DGX Spark (ARM64 / sm_121)

## Why a custom Docker image?

The DGX Spark GB10 uses compute capability **sm_121** (Blackwell).
Standard pip-installed PyTorch wheels (e.g. the `podcast_flow` conda env that runs Kokoro) are compiled without sm_121 PTX — they crash on GPU ops. The NGC base image `nvcr.io/nvidia/pytorch:25.01-py3` explicitly compiles for sm_121, resolving this.

## Directory Layout

```
fish-speech-docker/
├── Dockerfile              # ARM64 image (NGC PyTorch 25.01 base)
├── docker-compose.yml      # API server + WebUI services
├── entrypoint.sh           # Container startup logic
├── init_and_start.sh       # One-shot setup: build → download → start
├── verify_gpu_setup.sh     # Pre-flight GPU checks
├── .env.example            # Environment variable template
├── checkpoints/
│   └── fish-speech-1.5/   # Model weights (downloaded here, mounted ro)
├── reference_audio/        # Put .wav reference files here for voice cloning
└── outputs/                # TTS output files land here
```

## Step-by-Step Commands

### 1 — Verify GPU + Docker setup

```bash
chmod +x verify_gpu_setup.sh
./verify_gpu_setup.sh
```

Fix any `[FAIL]` items. `[WARN]` items are informational.

**If NVIDIA Container Toolkit is not installed (fail on step 3):**
```bash
# Install on Ubuntu/Debian ARM64
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2 — One-shot initialization (recommended)

```bash
chmod +x init_and_start.sh entrypoint.sh
./init_and_start.sh
```

This runs all 6 steps in sequence:
1. Pre-flight verification
2. Create directories
3. Build Docker image (~20-40 min on first run)
4. Download weights from HuggingFace (~5 GB)
5. Start API server container
6. Smoke-test TTS endpoint

---

### Manual step-by-step (alternative)

#### Build the image
```bash
docker build \
  --platform linux/arm64 \
  --build-arg FISH_SPEECH_TAG=v1.5.0 \
  --tag fish-speech:v1.5-arm64 \
  .
```

#### Download weights (host-side, using container to avoid host Python issues)
```bash
docker run --rm \
  --platform linux/arm64 \
  -v "$(pwd)/checkpoints/fish-speech-1.5:/download_target" \
  fish-speech:v1.5-arm64 /bin/bash -c "
    python3 -c \"
from huggingface_hub import snapshot_download
snapshot_download(
    'fishaudio/fish-speech-1.5',
    local_dir='/download_target',
    ignore_patterns=['*.md','README*','.gitattributes'],
)
print('Done.')
\"
  "
```

#### Start the API server (docker run)
```bash
docker run -d \
  --name fish-speech-api \
  --platform linux/arm64 \
  --restart unless-stopped \
  --gpus all \
  -p 8082:8080 \
  -v "$(pwd)/checkpoints/fish-speech-1.5:/app/checkpoints/fish-speech-1.5:ro" \
  -v "$(pwd)/reference_audio:/app/reference_audio" \
  -v "$(pwd)/outputs:/app/outputs" \
  -e MODE=api \
  -e CHECKPOINTS_PATH=/app/checkpoints/fish-speech-1.5 \
  fish-speech:v1.5-arm64 api
```

#### Start WebUI for testing
```bash
docker run -d \
  --name fish-speech-webui \
  --platform linux/arm64 \
  --gpus all \
  -p 7860:7860 \
  -v "$(pwd)/checkpoints/fish-speech-1.5:/app/checkpoints/fish-speech-1.5:ro" \
  -v "$(pwd)/reference_audio:/app/reference_audio" \
  -e MODE=webui \
  -e CHECKPOINTS_PATH=/app/checkpoints/fish-speech-1.5 \
  fish-speech:v1.5-arm64 webui
```

Or using Compose (starts WebUI profile):
```bash
docker compose --profile webui up fish-speech-webui
```

---

## Useful Commands

```bash
# Follow API server logs
docker logs -f fish-speech-api

# Exec into container for debugging
docker exec -it fish-speech-api /bin/bash

# Check GPU usage inside container
docker exec fish-speech-api nvidia-smi

# Verify PyTorch sees sm_121 GPU
docker exec fish-speech-api python3 -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('Capability:', torch.cuda.get_device_capability(0))
t = torch.randn(4, 4).cuda()
print('GPU tensor OK:', t.device)
"

# Reload container after a config change (no rebuild needed)
docker restart fish-speech-api

# Rebuild image only (after Dockerfile changes)
docker build --platform linux/arm64 --tag fish-speech:v1.5-arm64 .
docker restart fish-speech-api
```

---

## API Reference (for DR_2_Podcast integration)

```bash
# Health check
curl http://localhost:8082/v1/health

# Basic TTS (Japanese)
curl -X POST http://localhost:8082/v1/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "こんにちは、テストです。",
    "format": "wav",
    "normalize": true,
    "latency": "normal"
  }' \
  --output test.wav

# Voice cloning (reference audio + text)
curl -X POST http://localhost:8082/v1/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "合成したいテキストをここに入力します。",
    "references": [
      {
        "audio": "<base64-encoded-reference-wav>",
        "text": "参照音声のトランスクリプト"
      }
    ],
    "format": "wav",
    "normalize": true
  }' \
  --output cloned.wav
```

---

## Port Layout (all services)

SearXNG already occupies port 8080 on this host, so Fish Speech API is mapped to **8082**.

| Service | Host Port |
|---|---|
| vLLM (Qwen2.5) | 8000 |
| SearXNG | 8080 |
| **Fish Speech API** | **8082** |
| **Fish Speech WebUI** | **7860** |
| Ollama (DR_2_Podcast fast model) | 11434 |
| Ollama (coding-agent member) | 11435 |

The container-internal port is always 8080; only the host-side binding changes (`8082:8080` in docker-compose.yml).

---

## Known Limitations on ARM64

| Feature | Status |
|---|---|
| Flash Attention 2 | No pre-built wheel — Dockerfile attempts source build; falls back to PyTorch SDPA |
| GPU inference (sm_121) | ✓ Works with NGC 25.01 base |
| Japanese TTS | ✓ Fish Speech 1.5 natively supports Japanese |
| Voice cloning | ✓ Via reference audio in API request |
| Batch inference | ✓ Single GPU, set `API_WORKERS=1` |
