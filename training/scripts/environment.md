# Training environment compatibility

## Target stack

| Component | Target | Status vs SoundEx code |
|-----------|--------|-------------------------|
| **Python** | **3.12.x** | Compatible (uses 3.10+ typing: `X \| Y`, `list[T]`) |
| **PyTorch** | **2.12.1** | Compatible (standard `nn`, `stft`, AdamW, DataLoader, `weights_only` load) |
| **CUDA (PyTorch wheel)** | **≤ 13.0** family (`cu126` / `cu130`) | Code has **no** custom CUDA extensions; any PyTorch GPU wheel works if the **GPU arch + driver** match that wheel |
| **torchaudio** | optional 2.12.1 | **Not imported** by current training scripts |

## Code audit summary

SoundEx training is pure Python + stock PyTorch:

- No `torch.utils.cpp_extension` / custom CUDA kernels  
- No hardcoded CUDA toolkit path  
- Device selection: `torch.cuda.is_available()` → `"cuda"`  
- Checkpoint I/O: `torch.load(..., weights_only=True)` (supported since 2.0, correct on 2.12)  
- ONNX export: forces `dynamo=False` when the parameter exists (2.x safe path)  
- Audio I/O: `soundfile` + `ffmpeg` subprocess (not torchaudio)

Therefore **application architecture is compatible** with Python 3.12 + PyTorch 2.12.1 + CUDA-≤13.0 **wheels**, subject to the host/GPU caveats below.

## CUDA “≤ 13.0” clarified

Two different “CUDA versions” matter:

1. **Host driver** (`nvidia-smi` → Driver Version / CUDA Version)  
2. **PyTorch bundled CUDA runtime** (e.g. `torch==2.12.1+cu130` or `+cu126`)

The project does **not** require a full local CUDA Toolkit install for training.  
`nvcc` is only needed if you compile custom extensions (we do not).

| PyTorch wheel | Typical use |
|---------------|-------------|
| `cu130` | Drivers new enough for CUDA 13.0 userspace; modern GPUs (Turing+) |
| `cu126` | Older drivers / broader legacy GPU support within the 12.x line |

**Requirement “CUDA ≤ 13.0”** is satisfied by using official wheels built for **cu126 or cu130**, not by requiring toolkit 13 on the host.

## GPU architecture caveats (important)

| GPU | Compute | Notes for torch 2.12.1 |
|-----|---------|-------------------------|
| **Titan Xp** | **sm_61 Pascal** | CUDA **13.x toolkit dropped Pascal**. Prefer **`cu126` (or older)** PyTorch builds and verify `torch.cuda.is_available()`. Do **not** assume `cu130` works on Titan Xp. |
| RTX 20/30/40… | Turing+ | `cu126` or `cu130` generally fine if driver is new enough |

Always run after install:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda compiled", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY
```

If you see *“CUDA capability 6.1 is not compatible”*, switch to a wheel that still ships `sm_61` (typically older CUDA index, e.g. cu126 if still includes Pascal, or an older torch with cu118).

## Recommended install (generic server, CUDA 13.0-class driver)

```bash
conda create -n soundex python=3.12 -y
conda activate soundex
pip install torch==2.12.1 torchaudio==2.12.1 \
  --index-url https://download.pytorch.org/whl/cu130
pip install -r training/requirements.txt
```

## Reproducible CPU export gate

The checkpoint-to-ONNX release path is tested on Python 3.12 with exact versions in
`training/constraints-cpu-py312.txt`:

```bash
python -m pip install torch==2.12.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r training/requirements.txt \
  -c training/constraints-cpu-py312.txt
cd training
python -m pytest -q tests/test_checkpoint.py tests/test_export.py
```

ONNX Runtime is mandatory for export. The exporter stages its output and publishes it only after
checker, metadata, fixed-shape, size, and deterministic numerical parity gates succeed.

## Recommended install (Titan Xp 12GB)

```bash
conda create -n soundex python=3.12 -y
conda activate soundex
# Prefer 12.6 wheels first; fall back if sm_61 missing
pip install torch==2.12.1 torchaudio==2.12.1 \
  --index-url https://download.pytorch.org/whl/cu126
pip install -r training/requirements.txt
python training/scripts/check_env.py
```

## Docker

Verified base tags on Docker Hub:

| Tag | Role |
|-----|------|
| `pytorch/pytorch:2.12.1-cuda12.6-cudnn9-runtime` | **Default** (Dockerfile `BASE_IMAGE`) |
| `pytorch/pytorch:2.12.1-cuda13.0-cudnn9-runtime` | CUDA 13.0 hosts (`--build-arg BASE_IMAGE=…`) |

```bash
cd training
docker build -t soundex-train .
# or: docker compose build

# Download Slakh rendered mixes + preprocess + train
docker run --gpus all -v /data:/data \
  -e DOWNLOAD_DATASETS=slakh2100 \
  soundex-train

# Custom storage + cache paths (bind-mount each host dir)
docker run --gpus all \
  -v /mnt/datasets:/datasets \
  -v /mnt/scratch:/scratch \
  -e DATA_ROOT=/datasets \
  -e CACHE_DIR=/scratch/soundex-cache \
  -e SLAKH2100_PATH=/datasets/slakh \
  -e DOWNLOAD_DATASETS=slakh2100 \
  soundex-train

# Data only
docker run --gpus all -v /data:/data \
  -e DOWNLOAD_DATASETS=slakh2100 -e RUN_TRAIN=0 \
  soundex-train
```

### Path controls

| Variable / flag | Meaning |
|-----------------|--------|
| `DATA_ROOT` / `--data-root` | Default parent for dataset folders |
| `CACHE_DIR` / `--cache-dir` | Where zip/tar archives are downloaded |
| `SLAKH2100_PATH` / `--slakh2100-path` | Slakh extract + `processed/` root |
| `MUSIC_LIBRARY_PATH` | Generic audio-library processed-data root |
| `--path NAME=DIR` | Same as per-dataset path (repeatable) |
| `PATHS_MANIFEST` | YAML written by downloader; read by `train.py` |

Priority for dataset storage: **CLI `--path` / dedicated flags → env `*_PATH` → `$DATA_ROOT/<name>`**.  
`train.py` priority: **env `*_PATH` → paths manifest → config YAML**.

Entrypoint: `scripts/entrypoint.sh`  
Downloader: `scripts/download_datasets.py` (Slakh mix-only; MUSDB removed)
Host needs a compatible NVIDIA driver + `nvidia-container-toolkit`.

The default recipe requires FFmpeg encoders `libmp3lame`, `aac` and `libvorbis`.
Check `ffmpeg -hide_banner -encoders`; some macOS builds omit `libvorbis`.
Use a complete FFmpeg build on `PATH`. Preprocessing reports missing encoders
before generating pairs; do not silently remove codec strata from a recipe.

## Dependency notes

| Package | Role | 3.12 / torch 2.12 risk |
|---------|------|-------------------------|
| numpy | arrays | Use ≥1.26; 2.x OK with torch 2.12 |
| soundfile / ffmpeg | I/O + MP3 degrade | Independent of CUDA |
| onnx / onnxruntime | export + verify | CPU onnxruntime fine for checks |
| tensorboard | logs | Fine on 3.12 |

## Verification script

```bash
python training/scripts/check_env.py
```
