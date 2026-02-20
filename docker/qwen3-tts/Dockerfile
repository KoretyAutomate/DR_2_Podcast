FROM nvcr.io/nvidia/pytorch:25.01-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libsndfile1 libsndfile1-dev sox wget curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1001 -s /bin/bash qwen3tts
WORKDIR /app

# Install qwen-tts WITHOUT overwriting the base image's CUDA torch/torchvision.
# --no-deps prevents pip from pulling a CPU-only torch that breaks torchvision.
RUN pip install --no-cache-dir --no-deps qwen-tts

# Install qwen-tts runtime deps.
# Omit torch/torchvision/torchaudio — the NGC base image already provides CUDA builds.
# Installing them here would pull CPU-only versions from PyPI and break everything.
# torchaudio: install --no-deps to match the NGC torch 2.6.x without overwriting it
RUN pip install --no-cache-dir --no-deps torchaudio

RUN pip install --no-cache-dir \
        accelerate einops gradio librosa \
        onnxruntime \
        sox \
        transformers sentencepiece \
        "fastapi>=0.111.0" \
        "uvicorn[standard]>=0.30.0" \
        "soundfile>=0.12.1" \
        "numpy"

COPY tts_server.py /app/tts_server.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN mkdir -p /app/checkpoints /app/outputs
RUN chown -R qwen3tts:qwen3tts /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TORCH_CUDA_ARCH_LIST="9.0;10.0;12.1"
ENV TOKENIZERS_PARALLELISM=false
ENV CHECKPOINTS_PATH=/app/checkpoints

EXPOSE 8080

USER qwen3tts
ENTRYPOINT ["/entrypoint.sh"]
CMD ["api"]
