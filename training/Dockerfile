# SoundEx Training Dockerfile
#
# Target stack: Python 3.12 + PyTorch 2.12.1 + CUDA 12.6/13.0 family (image-embedded)
#
# Build (from this directory):
#   docker build -t soundex-train .
#   docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.12.1-cuda13.0-cudnn9-runtime -t soundex-train:cu130 .
#
# Run (download MUSDB mix-only + preprocess + train):
#   docker run --gpus all -v /data:/data \
#     -e DOWNLOAD_DATASETS=musdb18-hq \
#     soundex-train
#
# Run (MUSDB + Slakh mix-only extract — needs large free disk at peak):
#   docker run --gpus all -v /data:/data \
#     -e DOWNLOAD_DATASETS=musdb18-hq,slakh2100 \
#     soundex-train
#
# Data prep only:
#   docker run --gpus all -v /data:/data \
#     -e DOWNLOAD_DATASETS=musdb18-hq -e RUN_TRAIN=0 \
#     soundex-train
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

# curl: large Zenodo resumes; ffmpeg/libsndfile: preprocess + soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ ./configs/
COPY data/ ./data/
COPY models/ ./models/
COPY scripts/ ./scripts/
COPY artifact_contract.py checkpoint.py checkpoint_state.py export_validation.py ./
COPY train.py export_onnx.py evaluate.py ./

# Ensure entrypoint is executable even on volume-unfriendly FS
RUN chmod +x /workspace/scripts/entrypoint.sh /workspace/scripts/check_env.py \
    /workspace/scripts/download_datasets.py /workspace/scripts/filter_mix_only.py

VOLUME ["/data"]

# Default: train with config; set DOWNLOAD_DATASETS to auto-fetch first
ENTRYPOINT ["/workspace/scripts/entrypoint.sh"]
CMD []

# Metadata
LABEL org.opencontainers.image.title="soundex-train" \
      org.opencontainers.image.description="SoundEx BWE training (PyTorch 2.12 + mix-only dataset download)" \
      org.opencontainers.image.source="https://github.com/soundex"
