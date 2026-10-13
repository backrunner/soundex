# SoundEx Training Dockerfile
#
# Target stack: Python 3.12 + PyTorch 2.12.1 + CUDA 12.6/13.0 family (image-embedded)
#
# Build (from this directory):
#   docker build -t soundex-train .
#   docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.12.1-cuda13.0-cudnn9-runtime -t soundex-train:cu130 .
#
# Run with prepared pairs matching the selected profile:
#   docker run --gpus all -v /data:/data \
#     -e MUSIC_LIBRARY_PATH=/data/music-library soundex-train
# Prepare reviewed originals explicitly with data/preprocess_library.py first.
#
# Env check:
#   docker run --rm --gpus all -e CHECK_ENV_ONLY=1 soundex-train
#
# Titan Xp / older drivers: prefer cuda12.6 base (default), not cuda13.0.
# See scripts/environment.md

ARG BASE_IMAGE=pytorch/pytorch:2.12.1-cuda12.6-cudnn9-runtime
FROM ${BASE_IMAGE}

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_ROOT=/data \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg/libsndfile: codec synthesis and soundfile I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ ./configs/
COPY data/ ./data/
COPY models/ ./models/
COPY scripts/ ./scripts/
COPY evaluation/ ./evaluation/
COPY *.py ./
COPY parity_policy.json ./

RUN chmod +x /workspace/scripts/entrypoint.sh /workspace/scripts/check_env.py

VOLUME ["/data"]

# Default: train on prepared audio-library pairs
ENTRYPOINT ["/workspace/scripts/entrypoint.sh"]
CMD []

# Metadata
LABEL org.opencontainers.image.title="soundex-train" \
      org.opencontainers.image.description="SoundEx audio restoration training with reviewed lossless libraries" \
      org.opencontainers.image.source="https://github.com/backrunner/soundex"
