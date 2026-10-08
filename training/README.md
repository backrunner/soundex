# Training SoundEx

Run commands from the repository root unless a block says `cd training`.
Use Python 3.12 and FFmpeg with the encoders required by your recipe. Install
PyTorch for your hardware first; see [environment setup](scripts/environment.md)
for CPU, Apple MPS and CUDA constraints. Do not install a CUDA wheel on macOS.

For a reproducible CPU export/test environment:

```bash
python3.12 -m venv training/.venv
source training/.venv/bin/activate
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r training/requirements.txt -c training/constraints-cpu-py312.txt
python -m pip check
```

Default paths and source quotas assume all three datasets are available. For a
single-dataset experiment, copy the config and set paths, ratios and validation
quotas to the datasets actually available; missing datasets must not silently
change the validation membership. Keep that resolved config with the checkpoint.


Install dependencies from `requirements.txt`, then preprocess **full-track mixtures
only** (stems are never used):

| Dataset | Clean source used | Script |
|---------|-------------------|--------|
| MUSDB18-HQ | `mixture.wav` | `data/preprocess_musdb.py` |
| [Slakh2100](http://www.slakh.com/) | `mix.flac` / `mix.wav` | `data/preprocess_slakh.py` |
| [MedleyDB](https://medleydb.weebly.com/) | `*_MIX.wav` | `data/preprocess_medleydb.py` |

```bash
cd training

python data/preprocess_musdb.py \
  --data-root /data/musdb18-hq \
  --output-dir /data/musdb18-hq/processed \
  --config configs/default.yaml

python data/preprocess_slakh.py \
  --data-root /data/slakh2100 \
  --output-dir /data/slakh2100/processed \
  --config configs/default.yaml

python data/preprocess_medleydb.py \
  --data-root /data/medleydb \
  --output-dir /data/medleydb/processed \
  --config configs/default.yaml

python train.py --config configs/default.yaml
python export_onnx.py \
  --checkpoint checkpoints/best-validation.pth \
  --output ../models/soundex-v1.onnx
```

Training checkpoints use schema 1.2 and contain the resolved model/audio/data configuration,
feature contract, dataset recipe and manifest hashes, provenance, and all RNG states needed to
resume sampling. Older 1.x checkpoints lack the crossover-bound feature contract and fail closed;
schema-0 checkpoints are inspection-only and cannot be exported. ONNX export
derives every model and feature setting from the checkpoint; `--config` is optional comparison
input and never overrides checkpoint semantics. A successful export is atomically published only
after ONNX checker and mandatory PyTorch/CPU-ORT [unit-aware parity](../docs/parity.md) pass, and prints the artifact SHA-256.

Runtime artifacts use artifact schema 1.3, opset 17, dynamic batch only, and fixed float32
input/output `[batch, 2, 1, 129]` for the default 256/128 contract. Rust validates all
`soundex.*` metadata, feature semantics, FFT/hop, supported sample rate, crossover width, tensor
names, and shapes before processing the first frame. Schema-1.1 artifacts remain loadable only
under explicit legacy 2048/512 configuration; schema-1.2 artifacts also support
1024/512 with an explicit matching configuration. The 512/256 fallback is
also supported by schema 1.3. Small-window training uses steady-state waveform loss,
excluding cropped Hann overlap boundaries; older profiles retain the original full-region loss.

Validation membership, crop positions, and source quotas are fixed. Checkpoint
selection minimizes the missing-high-band validation loss, using low-band identity
as the tie-breaker. `checkpoints/best-validation.pth` is the export candidate;
`checkpoints/final-resume.pth` is atomically updated every epoch for exact resume.
Each schema-1.2 checkpoint stores the full validation vector, selected-row manifest
hash, best-candidate state, and absolute validation schedule. The configured
linear-warmup/cosine learning-rate schedules advance per optimizer update; the
discriminator schedule starts only when adversarial training begins, and both
scheduler states are restored on resume.

Each preprocessor stages a new immutable `<corpus>-<recipe-hash>` directory, validates its
checksummed JSONL manifest, and atomically publishes it. Existing versions are never merged or
deleted. The manifest holds disjoint track-level `train`, `validation`, and untouched `test`
roles. `train.py` loads only the exact recipe version from the selected profile; missing roots or
versions are skipped. **Mix ratios and per-source caps** live under
`data.sampling` in `configs/default.yaml` (default ~80% real / 20% synthetic:
MUSDB full @ 50%, MedleyDB 30%, Slakh 20% with equal pool caps so MedleyDB ≥ Slakh).
Paths are set in the same config.

`data.recipe` is the sole preprocessing contract. It covers MP3 CBR/VBR, AAC-LC, Vorbis,
44.1/48 kHz, codec-delay alignment, and mono or post-codec left/right/mid/side examples. One AAC
setting is held out from training and retained in validation/test for generalization evaluation.

Release evaluation is manifest-driven and runs the ONNX artifact through the actual
Rust causal stream in offline and two arbitrary-chunk modes. It compares against the
degraded baseline, emits deterministic bootstrap confidence intervals and
per-stratum metrics, and fails closed when ViSQOLAudio, parity evidence, or another
required gate is missing. The differentiable training waveform loss models crossover,
phase blending, and causal overlap-add, but not Rust's stateful loudness matcher,
limiter, or sample-domain gate ramp; the Rust evaluation path is the authority for
deployed quality. See
[`evaluation/README.md`](evaluation/README.md) for commands and the
required blinded listening protocol.

**Disk:** only full-track mixes are needed. Prefer automatic download (mix-only extract/prune):

```bash
cd training
# Host: choose data-root, cache, and per-dataset storage
python scripts/download_datasets.py --datasets musdb18-hq,slakh2100 \
  --data-root /mnt/datasets \
  --cache-dir /mnt/scratch/soundex-cache \
  --path musdb18-hq=/mnt/datasets/musdb \
  --path slakh2100=/mnt/datasets/slakh \
  --preprocess

# Docker (build + download MUSDB + train), custom mounts:
docker compose build
docker run --gpus all \
  -v /mnt/datasets:/datasets -v /mnt/scratch:/scratch \
  -e DATA_ROOT=/datasets \
  -e CACHE_DIR=/scratch/soundex-cache \
  -e MUSDB18_HQ_PATH=/datasets/musdb \
  -e DOWNLOAD_DATASETS=musdb18-hq \
  soundex-train
```

Prune existing trees: `python training/scripts/filter_mix_only.py --root /data/<set> --delete`.
Size plan / CUDA notes: [`scripts/download_mix_only.md`](scripts/download_mix_only.md),
[`scripts/environment.md`](scripts/environment.md).

Weights and datasets use **separate** legal terms from the source code. Before publishing a
checkpoint or ONNX file, complete a [Model Card](../models/MODEL_CARD.template.md) and follow
[`../legal/MODEL_RELEASE_CHECKLIST.md`](../legal/MODEL_RELEASE_CHECKLIST.md).
