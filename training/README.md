# Training SoundEx

Run commands from the repository root unless a block says `cd training`.
Use Python 3.12 and FFmpeg with the encoders required by your recipe. Install
PyTorch for your hardware first; see [environment setup](scripts/environment.md)
for CPU, Apple MPS and CUDA constraints. Do not install a CUDA wheel on macOS.

See [workspace layout](../docs/workspace.md) for local raw audio, prepared pairs,
reports and frozen runs. Source-quality audits use `scripts/audit_library.py`;
reviewed publisher original-file plans can be acquired with
`scripts/fetch_archive_masters.py`. Acquisition receipts and source catalogs are
separate from the final audited training selection.

For a reproducible CPU export/test environment:

```bash
python3.12 -m venv training/.venv
source training/.venv/bin/activate
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r training/requirements.txt -c training/constraints-cpu-py312.txt
python -m pip check
```

## Audio sources and codec synthesis

Default and Titan Xp profiles now use real music libraries; `music_library.yaml`
accepts generic audio libraries. The current regional run excludes Slakh. MUSDB has been removed from download/preprocessing/
training selection. Historical checkpoint identifiers remain readable for audits.

Import folders directly or add an optional source catalog. Training has no dataset
or license whitelist and requires no rights declaration to start. See
[dataset survey and optional catalog](../docs/datasets.md) and
[training data and publication](../legal/TRAINING_DATA.md). A clean recording is
sufficient: FFmpeg compression/decode plus alignment produces supervised pairs;
stems and separation labels are unnecessary.

```bash
cd training
# Audio library, directory route (or --catalog /data/catalog.jsonl):
python data/preprocess_library.py --data-root /data/audio-library \
  --output-dir /data/music-library/processed --config configs/music_library.yaml
MUSIC_LIBRARY_PATH=/data/music-library python train.py --config configs/music_library.yaml

python export_onnx.py --checkpoint checkpoints/best-validation.pth \
  --output ../target/candidate.onnx
```

Source credits and licenses are optional ingestion metadata; they are reviewed
when publishing official weights. Training keeps necessary technical checks for
format, finite samples, hashes, alignment, immutable recipes and disjoint splits.
Use positive source ratios and explicit validation quotas for enabled sources.

Clean targets must be lossless PCM WAV/FLAC/AIFF. FMA MP3 packages and lossy
encodings disguised by file extensions are excluded. Verify original-master
provenance too: transcoding MP3 into FLAC does not restore the original signal.
`configs/diverse_lossless.yaml` combines real music and speech; prepare
speech with `--corpus speech_library` using the same profile. Derive speech mass
from the smallest actual training genre with `data.sampling_weights.speech_mix_ratios`.
See [genre coverage and sampling](../docs/datasets.md).

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
`data.sampling` in the selected profile (default real music;
`music_library.yaml` uses a generic audio directory or optional catalog).
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

## Acquisition and release

The legacy dataset downloader requires an explicit `--datasets` choice; its Slakh
importer remains available for reproducing historical experiments, and is not
part of the current training route. Acquire music libraries from their publisher and
retain source information when available. Publication requires actual source grants,
not a blanket inference from a “free” site.

```bash
cd training
python scripts/download_datasets.py --datasets slakh2100 --data-root /mnt/datasets \
  --cache-dir /mnt/scratch/soundex-cache --preprocess
```

[Storage guide](scripts/download_mix_only.md) · [environment](scripts/environment.md)

Official source and released weights use Apache-2.0. Training audio retains its
upstream terms. An export is a local candidate until the completed model card,
artifact-bound rights review and all [release checks](../legal/MODEL_RELEASE_CHECKLIST.md)
pass. No application-ready learned weights have been published yet.

See [regional music acquisition](../docs/dataset-curation.md) for the current free-first, native-lossless regional run and its optional source/coverage audit.
