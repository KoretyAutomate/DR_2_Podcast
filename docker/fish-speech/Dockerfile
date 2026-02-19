# ============================================================
# Fish Speech V1.5 — ARM64 / aarch64 Docker Image
# Target: NVIDIA DGX Spark (GB10, sm_121, Blackwell)
#
# Base: nvcr.io/nvidia/pytorch:25.01-py3
#   • Compiled for aarch64
#   • PyTorch 2.6 with sm_121 (Blackwell GB10) kernel support
#   • CUDA 12.7.1 — compatible with GB10 driver 570+
#
# NOTE on sm_121: The DGX Spark GB10 is compute capability 12.1.
# PyTorch ≤2.5 (e.g. the conda podcast_flow env) has no sm_121
# PTX fallback compiled in, which is why Kokoro crashes on GPU.
# This NGC image resolves that — sm_121 is explicitly compiled.
# ============================================================

FROM nvcr.io/nvidia/pytorch:25.01-py3

# ── Build args ───────────────────────────────────────────────
ARG FISH_SPEECH_TAG=v1.5.0
ARG DEBIAN_FRONTEND=noninteractive

# ── System dependencies ──────────────────────────────────────
# libsndfile + ffmpeg: audio I/O
# libopenmpi: needed by some torch extensions
# git-lfs: for pulling large model files if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        git-lfs \
        ffmpeg \
        libsndfile1 \
        libsndfile1-dev \
        libopus0 \
        libopus-dev \
        sox \
        wget \
        curl \
        unzip \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# ── Create non-root user ─────────────────────────────────────
RUN useradd -m -u 1000 -s /bin/bash fishspeech
WORKDIR /app

# ── Clone Fish Speech V1.5 ───────────────────────────────────
# We pin to the release tag; fall back to main if tag not yet published.
RUN git clone --depth 1 \
        --branch "${FISH_SPEECH_TAG}" \
        https://github.com/fishaudio/fish-speech.git . \
    || (echo "Tag ${FISH_SPEECH_TAG} not found, cloning main" \
        && git clone --depth 1 https://github.com/fishaudio/fish-speech.git .)

# ── Python dependencies ──────────────────────────────────────
# The NGC base already provides:
#   torch 2.6, torchaudio, torchvision, CUDA 12.7 dev headers
#
# Install Fish Speech's own requirements first (they may pin torch
# versions we don't want to override), then install the package itself.
#
# --no-deps on fish-speech install: torch is already correct in the
# base image; we do NOT want pip to downgrade it to a cu121 wheel
# that lacks sm_121 PTX.

# Install heavy audio/ML dependencies separately for better caching
RUN pip install --no-cache-dir \
        "transformers>=4.41.0" \
        "huggingface_hub>=0.23.0" \
        "vector-quantize-pytorch>=1.14.4" \
        "einops>=0.8.0" \
        "encodec" \
        "loralib" \
        "lightning>=2.3.0" \
        "loguru>=0.7.2" \
        "natsort" \
        "pyrootutils" \
        "soundfile>=0.12.1" \
        "librosa>=0.10.2" \
        "pydub>=0.25.1" \
        "av>=12.0.0" \
        "silero-vad>=4.0.0"

# Install API server dependencies
RUN pip install --no-cache-dir \
        "fastapi>=0.111.0" \
        "uvicorn[standard]>=0.30.0" \
        "ormsgpack>=1.4.2" \
        "pydantic>=2.7.0" \
        "httpx>=0.27.0"

# Install WebUI dependencies
RUN pip install --no-cache-dir \
        "gradio>=4.36.0" \
        "gradio-i18n" \
        "hydra-core>=1.3.2"

# ── Flash Attention (optional, best-effort on ARM64) ─────────
# Flash Attention 2 does not have pre-built aarch64 wheels.
# We attempt a source build; if it fails we skip — Fish Speech
# degrades gracefully to standard SDPA attention (slower, same output).
RUN pip install --no-cache-dir packaging ninja && \
    pip install --no-cache-dir \
        --no-build-isolation \
        "flash-attn>=2.6.0" \
    || echo "⚠ Flash Attention build failed on ARM64 — using PyTorch SDPA fallback (this is expected and OK)"

# ── Install Fish Speech itself ───────────────────────────────
# Use --no-deps to prevent pip from replacing the NGC torch with a
# cu121/cu124 wheel that has no sm_121 kernels.
RUN pip install --no-cache-dir --no-deps -e .

# Fix up any remaining deps that --no-deps skipped (everything except torch family)
RUN pip install --no-cache-dir \
        $(python -c "
import tomllib, sys
with open('pyproject.toml','rb') as f:
    d=tomllib.load(f)
deps=d.get('project',{}).get('dependencies',[])
skip={'torch','torchaudio','torchvision'}
filtered=[x for x in deps if not any(s in x.lower() for s in skip)]
print(' '.join(repr(x) for x in filtered))
" 2>/dev/null || echo "") || true

# ── Checkpoints directory ────────────────────────────────────
# Model weights are mounted from the host at runtime.
# We create the directory here so the volume mount has a target.
RUN mkdir -p /app/checkpoints/fish-speech-1.5

# ── HuggingFace cache directory ──────────────────────────────
ENV HF_HOME=/app/hf_cache
RUN mkdir -p /app/hf_cache

# ── Ownership ────────────────────────────────────────────────
RUN chown -R fishspeech:fishspeech /app

# ── Environment ─────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Tell PyTorch/CUDA to be ready for sm_121 (Blackwell GB10)
ENV TORCH_CUDA_ARCH_LIST="9.0;10.0;12.1"
# Disable tokenizer parallelism warnings in forked processes
ENV TOKENIZERS_PARALLELISM=false
# Default checkpoint path (overridable via env in docker-compose)
ENV CHECKPOINTS_PATH=/app/checkpoints/fish-speech-1.5

# ── Ports ────────────────────────────────────────────────────
EXPOSE 8080   # REST / OpenAI-compat API server
EXPOSE 7860   # Gradio WebUI

# ── Entrypoint ───────────────────────────────────────────────
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER fishspeech
ENTRYPOINT ["/entrypoint.sh"]
CMD ["api"]
