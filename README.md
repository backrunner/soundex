# SoundEx

SoundEx is a Rust audio bandwidth-extension library and CLI. It detects high-frequency loss in
PCM audio, runs a lightweight ONNX magnitude/phase model when needed, and applies crossover,
loudness, phase, and limiting post-processing.

The repository contains the inference library, DSP crate, file-based CLI, and PyTorch training
pipeline. A pretrained model is not included yet; train and export one before normal CLI use.

## Build and test

```bash
cargo build --workspace
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
```

The default `soundex-core` feature downloads and links ONNX Runtime. Build without an inference
backend for model-free analysis only with `--no-default-features`.

## CLI

Analyze a file without loading a model:

```bash
cargo run -p soundex-cli -- input.mp3 --dry-run
```

Enhance a file with an exported model:

```bash
cargo run -p soundex-cli -- input.mp3 \
  --model models/soundex-v1.onnx \
  --output output.wav \
  --bits 24
```

Supported inputs are MP3, AAC, FLAC, WAV, and OGG Vorbis. Output is 16-bit PCM, 24-bit PCM, or
32-bit float WAV.

## Library

`process_frame` accepts one interleaved hop (`hop_size * channels` samples). With the default
1024-point FFT and 512-sample hop, `latency_samples_per_channel()` reports 512 samples (one hop)
of causal STFT delay: about 11.6 ms at 44.1 kHz or 10.7 ms at 48 kHz, before inference and device
buffers. Fixed-hop callers receive startup padding directly and must call `finalize` once to
retrieve that delayed tail.

`process_chunk` accepts any number of complete interleaved sample frames and reports interleaved
samples consumed and produced. It withholds startup padding; concatenating every chunk result and
the single `finalize` result is sample-aligned and the same length as `process_buffer`. A finalized
stream rejects more input until `reset`. A processing failure leaves caller output unchanged and
poisons internal state, which also requires `reset`. Do not mix `process_frame` and `process_chunk`
within one stream.

```rust,no_run
use soundex_core::{SoundExConfig, SoundExProcessor};

let config = SoundExConfig::with_model("models/soundex-v1.onnx")
    .sample_rate(44_100)
    .channels(2);
let mut processor = SoundExProcessor::new(config)?;
let input = vec![0.0_f32; 44_100 * 2];
let output = processor.process_buffer(&input)?;
# Ok::<(), soundex_core::SoundExError>(())
```

Arbitrary streaming chunks append ready samples to an output vector:

```rust,no_run
use soundex_core::{SoundExConfig, SoundExProcessor};

let mut processor = SoundExProcessor::new(
    SoundExConfig::with_model("models/soundex-v1.onnx").channels(2),
)?;
let input = vec![0.0_f32; 4097 * 2];
let mut output = Vec::new();
for chunk in input.chunks(74) {
    processor.process_chunk(chunk, &mut output)?;
}
processor.finalize(&mut output)?;
assert_eq!(output.len(), input.len());
# Ok::<(), soundex_core::SoundExError>(())
```

## Performance evidence

SoundEx treats algorithmic delay, steady-state callback processing, cold model
construction, audio-device buffering, and offline throughput as separate values.
The default 1024/512 contract reports 512 samples per channel of algorithmic
delay and meets that isolated release threshold. This does not establish total
product latency or quality: a trained artifact must still pass the quality,
stereo p99, 30-minute deadline, memory, and cross-hardware gates below.

The production benchmark requires a non-trivial model through
`SOUNDEX_BENCH_MODEL`; it rejects `tests/fixtures/identity.onnx`. It records raw
latency samples and summary percentiles for mono/stereo at 44.1/48 kHz, using one
ORT session run per active stereo hop. Schema-2 reports bind the benchmark to
the artifact's 1024/512 frame protocol. Release runs default to 30 minutes per
case and require a declared CPU power mode:

```bash
SOUNDEX_BENCH_MODEL=models/soundex-v1.onnx \
SOUNDEX_BENCH_REPORT=models/soundex-v1.performance-apple.json \
SOUNDEX_BENCH_HARDWARE_LABEL='Apple M4 reference' \
SOUNDEX_BENCH_POWER_MODE='AC power, Low Power Mode off' \
cargo bench -p soundex-core --bench core_bench
```

See [`docs/performance.md`](docs/performance.md) for diagnostic runs, report
fields, cross-hardware validation, and model-card binding. No production latency
claim is made until the same artifact passes on named Apple Silicon and x86_64
reference machines and retains all release quality gates.

## Training

Install dependencies from `training/requirements.txt`, then preprocess **full-track mixtures
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
after ONNX checker and mandatory PyTorch/CPU-ORT parity pass, and prints the artifact SHA-256.

Runtime artifacts use artifact schema 1.2, opset 17, dynamic batch only, and fixed float32
input/output `[batch, 2, 1, 513]` for the default 1024/512 contract. Rust validates all
`soundex.*` metadata, feature semantics, FFT/hop, supported sample rate, crossover width, tensor
names, and shapes before processing the first frame. Schema-1.1 artifacts remain loadable only
under the explicit legacy 2048/512 configuration.

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
`data.sampling` in `training/configs/default.yaml` (default ~80% real / 20% synthetic:
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
[`training/evaluation/README.md`](training/evaluation/README.md) for commands and the
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
Size plan / CUDA notes: [`training/scripts/download_mix_only.md`](training/scripts/download_mix_only.md),
[`training/scripts/environment.md`](training/scripts/environment.md).

Weights and datasets use **separate** legal terms from the source code. Before publishing a
checkpoint or ONNX file, complete a [Model Card](models/MODEL_CARD.template.md) and follow
[`legal/MODEL_RELEASE_CHECKLIST.md`](legal/MODEL_RELEASE_CHECKLIST.md).

## License

SoundEx uses a **dual licensing** structure:

| What | License | Document |
|------|---------|----------|
| **Source code** (Rust, CLI, training scripts) | Apache License 2.0 | [`LICENSE`](LICENSE) |
| **Model weights** (`.onnx`, `.pth`, …) | SoundEx Model Weights License 1.0 + **Weight Tier** | [`MODEL_WEIGHTS_LICENSE.md`](MODEL_WEIGHTS_LICENSE.md) |
| **Training datasets** (audio you download) | Upstream only (not Apache) | [`NOTICE`](NOTICE) |

**Weight tiers** (declared per release in the Model Card):

- **Tier A** — Attribution; commercial use allowed (e.g. Slakh-only training).
- **Tier B** — Non-commercial + share-alike (e.g. includes MedleyDB, not MUSDB).
- **Tier C** — Research / evaluation only (default if undeclared; required if MUSDB is used).

Full overview: [`legal/LICENSING.md`](legal/LICENSING.md).  
Dataset → tier matrix: [`legal/TRAINING_DATA_AND_WEIGHT_TIERS.md`](legal/TRAINING_DATA_AND_WEIGHT_TIERS.md).  
Disclaimer: [`legal/DISCLAIMER.md`](legal/DISCLAIMER.md).

Default multi-dataset training (MUSDB + Slakh + MedleyDB) produces **Tier C** weights.
