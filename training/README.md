# Training SoundEx

Run commands from the repository root unless a block says `cd training`.
Use Python 3.12 and FFmpeg with the encoders required by your recipe. Install
PyTorch for your hardware first; see [environment setup](scripts/environment.md)
for CPU, Apple MPS and CUDA constraints. Do not install a CUDA wheel on macOS.

The current maintainer run starts fresh with 300 reviewed native-lossless music
works plus VCTK speech. MUSDB and Slakh are disabled. See
[the current run](../docs/native-training-20261009.md) for source totals, recipe,
model profile, live status paths and local checkpoint locations.

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

Frame-local runtime artifacts use artifact schema 1.3, opset 17, dynamic batch only, and fixed float32
input/output `[batch, 2, 1, 129]` for the default 256/128 contract. Rust validates all
`soundex.*` metadata, feature semantics, FFT/hop, supported sample rate, crossover width, tensor
names, and shapes before processing the first frame. Schema-1.1 artifacts remain loadable only
under explicit legacy 2048/512 configuration; schema-1.2 artifacts also support
1024/512 with an explicit matching configuration. The 512/256 fallback is
also supported by schema 1.3. Small-window training uses steady-state waveform loss,
excluding cropped Hann overlap boundaries; older profiles retain the original full-region loss.

Validation membership, crop positions, and source quotas are fixed. Checkpoint
selection defaults to missing-high-band validation loss, using low-band identity
as the tie-breaker. The controlled `balanced_continuation.yaml` objective-v2 profile
uses normalized total loss and high-band loss as the tie-breaker, with raw metrics
retained for comparison. See [the loss audit](../docs/loss-audit-20261010.md).

`inpainting_continuation.yaml` introduces objective v3: missing-band spectral
shape, frame energy, reference-relative transient changes and high-band error
after differentiable crossover/phase/OLA reconstruction. Waveform and valid-phase
terms constrain fidelity alongside retained-band and seam preservation. It uses
8 causal training frames, independent-frame inference, frozen BatchNorm statistics
and an explicitly fresh optimizer trajectory. Initial weights were gradient-audited
on 64 training draws; they are experimental. V1/V2 totals are not comparable to V3.
See [the inpainting design and proxy limits](../docs/inpainting-objective.md).

`consistent_continuation.yaml` and `interacting_continuation.yaml` provide a paired
objective-v4 experiment: complex reconstruction, retained-band corruption and
high-band STFT consistency. Only the second enables zero-initialized amplitude/phase
encoder interactions (architecture 1.1). Both keep independent T=1 inference and
six-epoch budgets. A legacy-parent architecture migration explicitly requires
`--initialize-zero-interactions` with `--initialize-generator-from`; changed objectives
or structure cannot directly resume old optimizer state. The data and sample ratios
must be resolved from the existing native run, rather than using example paths/ratios.
See [the objective definitions](../docs/inpainting-objective.md#objective-v4复数重建与频谱一致性)
and [measured follow-up results](../docs/optimization-20261011.md).

`calibrated_continuation.yaml` and `circular_continuation.yaml` compare objective v5
with and without circular phase input features. V5 supervises reconstructed high-band
and subband energy plus reference-relative phase changes across frequency/time.
Both profiles use eight epochs, unchanged independent T=1 deployment, and frozen BN.
The architecture-1.2 variant expands the first phase convolution to raw/sin/cos
channels, adding 144 parameters; migration explicitly requires
`--initialize-phase-features --initialize-generator-from /path/to/parent.pth`.
It preserves the parent raw channel and initializes the added channels to zero.
Do not combine this migration with `--initialize-zero-interactions`.
Resolve real source paths and audited sampling ratios before running either profile.
See [the V5 methods and deployment evidence](../docs/goal-refinement-20261011.md).

`deployment_continuation.yaml` adds objective v6: default MatchEdge gain before
reconstruction and a reference-relative reconstructed temporal loss. It retains
raw V5 terms, the legacy architecture, eight epochs and frozen BN. The cropped
sequence gain uses a detached first-frame steady-state bootstrap; online cutoff,
limiting and gating still need actual Rust validation. Pass the canonical parent
data/ratios with `--initialize-generator-from`, not `--resume` with a changed loss.
See [the V6 experiment and limitations](../docs/deployment-refinement-20261011.md).

`scripts/prepare_architecture_study.py` derives matched control / polar-small /
shape-small / shape-large configurations from a reviewed parent without changing
its data paths or sampling. The optional `spectral_refiner` (architecture 1.3)
uses 129-bin frame-local global encoding, a zero-initialized bounded correction
head and widths 128/512. Explicit migration requires
`--initialize-spectral-refiner --initialize-generator-from /path/to/parent.pth`.
It adds no temporal context or runtime buffering. Keep the V6 objective and compare
actual Rust audio, model size and runtime costs before promoting a candidate.
See [the research and experiment protocol](../docs/architecture-research-20261011.md).
The [completed eight-epoch comparison](../docs/architecture-study-20261011.md)
separates development loss, Rust output quality and deployment cost; lower loss
alone does not promote any of the new experimental weights.

`scripts/prepare_temporal_study.py` derives matched frame-control / causal
continuations from a frame-refiner parent. The causal arm adds a depthwise temporal
filter and zero-initialized mixing on nine encoded observations; the U-Net processes
only the current frame. Explicit migration requires `--initialize-temporal-refiner`
with `--initialize-generator-from`. Artifact/architecture 1.4 declares input
`[B,2,9,129]`, output `[B,2,1,129]`; the ONNX graph has no hidden recurrent state.
The Rust host retains independent oldest-to-current spectral histories per physical
channel, including bypass/silence, and resets them on gaps/reloads. First eight hops
stay aligned dry while history fills; this enhancement warmup adds no output delay.
Both experiment arms use `sequence_warmup_frames: 8` followed by eight supervised
frames, so CPU validation must pass `warmup_frames=8` too. Keep loss and crop protocol
matched; totals from the earlier eight-frame-only study are not direct comparisons.
See [the causal experiment contract](../docs/causal-context-20261011.md).
`training.trainable_scope: temporal_only` freezes all parent parameters and trains
only the four new temporal tensors. It requires nine-frame context, frozen BN,
no GAN and a warm-started nonzero refiner head; a zero frozen head cannot propagate
learning into history. The default `all` retains normal training. Changing scope
requires a fresh optimizer trajectory; it is bound into the resolved checkpoint
configuration. This supports isolated learning-rate diagnostics, not automatic
quality promotion.

`checkpoints/best-validation.pth` is the export candidate;
`checkpoints/final-resume.pth` is atomically updated every epoch for exact resume.
Each schema-1.2 checkpoint stores the full validation vector, selected-row manifest
hash, best-candidate state, and absolute validation schedule. The configured
linear-warmup/cosine learning-rate schedules advance per optimizer update; the
discriminator schedule starts only when adversarial training begins, and both
scheduler states are restored on resume.

Changing the objective starts a new run: use `--initialize-generator-from /path/to/parent.pth`
with a compatible generator, feature contract, data manifests and validation membership.
This imports only generator weights, resets optimizer/discriminator state and records
the parent hash and initialization chain. `--resume` requires the original resolved config.
Bind real dataset paths and the audited sampling ratios in your run config; template
paths and ratios are examples.

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
# Prepare the enabled speech supplement with the same recipe as music:
PATH="$PWD/.tools/bin:$PATH" python data/preprocess_library.py \
  --data-root /path/to/native-speech --corpus speech_library \
  --output-dir ../data/processed/speech_library/processed \
  --config configs/diverse_lossless.yaml

# Use your own reviewed catalog; local source files are not included in Git:
PATH="$PWD/.tools/bin:$PATH" python scripts/run_native_training.py \
  --catalog ../data/catalogs/training-music.jsonl \
  --run-dir ../data/runs/native-300-YYYYMMDD-HHMMSS \
  --processed-root ../data/processed \
  --config configs/diverse_lossless.yaml --minimum-tracks 300 \
  --curation-policy configs/curation_300.yaml --evidence-root ../data/catalogs
```

[Storage guide](scripts/download_mix_only.md) · [environment](scripts/environment.md)

Official source and released weights use Apache-2.0. Training audio retains its
upstream terms. An export is a local candidate until the completed model card,
artifact-bound rights review and all [release checks](../legal/MODEL_RELEASE_CHECKLIST.md)
pass. No application-ready learned weights have been published yet.

See [regional music acquisition](../docs/dataset-curation.md) for the current free-first, native-lossless regional run and its optional source/coverage audit.


### Long reconstruction objective (v7, experimental)

Objective v7 adds an explicit `reconstruction_warmup_frames` prefix after real model
history. `causal_sequence_frames` still counts supervised frames only. The runner
infers the DSP prefix without gradients and frozen BN, initializes MatchEdge gain
at the Rust reset value of 1, advances it through the prefix, and excludes that
prefix from every supervised term. Gain history is finite and uses manifest cutoff;
this does not reproduce the online detector, gate ramp or limiter.

V7 requires `waveform_region: steady_state`, frozen BN and zero GAN weights.
The new observable terms reanalyse the fully supported OLA region at 2x/4x/8x the
model FFT size: `multires_high_spectral`, `multires_high_energy`,
`multires_high_temporal`, and `multires_low_complex`. They compare against the clean
reference (retained-band complex error compares against input), use detached
reference scales/floors, and penalize deviations from true temporal changes.
At FFT256/hop128 at least 21 supervised frames are needed for two longest-window
frames; the experiment uses 32 supervised + 32 DSP warmup + 8 model warmup frames.
No extra window is added to deployment inference or playback buffering.

`scripts/prepare_reconstruction_study.py` prepares two fixed four-epoch arms
with `--parent <frame-refiner.pth> --output-dir <new-study>`. Both use identical
long sequences and DSP warmup; only multiresolution loss weights differ. Final
checkpoints are rescored with one common CPU objective, so their differently
weighted training totals are not compared directly. See the
[protocol and results](../docs/multires-reconstruction-20261011.md).

### Input-conditioned gain calibration (experimental)

`scripts/fit_gain_calibration.py` fits ten bounded DSP coefficients using the
production Rust channel analyzer via `soundex-gain-data`. It accepts reviewed
music/speech manifests, excludes non-train recordings, chooses regularization on
a recording-disjoint inner split, and freezes an artifact before Rust development
scoring. The neural ONNX stays unchanged. The current regression target improves
while actual high-band recovery worsens; do not enable the fitted JSON by default.
See the [protocol and measured limits](../docs/gain-calibration-20261011.md).

### Phase reconstruction ablation (experimental)

`evaluation.compare_phase` fixes the model, detector, magnitude crossover and
MatchEdge recurrence while comparing `model`, `observed` and `surviving-power`
phase policies through the complete Rust stream. The last policy reduces the
model phase weight by the squared ratio of observed magnitude to the post-gain
candidate, capped at one. It is a heuristic and does not change model magnitude.

From `training/`, after rebuilding the release evaluator:

```bash
.venv/bin/python -m evaluation.compare_phase \
  --model /path/to/model.onnx \
  --manifest /path/to/music/manifest.jsonl \
  --manifest /path/to/speech/manifest.jsonl \
  --selection /path/to/frozen-validation-selection.json \
  --binary ../target/release/soundex-stream-eval \
  --output ../data/reports/phase-comparison.json
```

The bridge explicitly sets and verifies `SOUNDEX_EVAL_PHASE_POLICY`; benchmark
executables accept `SOUNDEX_PHASE_POLICY` and record the loaded policy. Both
experimental policies require Neural mode. The training loss proxy has not been
changed to use this policy; this is a DSP-only study with fixed ONNX weights.
See [results and limitations](../docs/phase-reconstruction-20261011.md).

## Original causal backbone probes

The [P0/P1 study](../docs/causal-prototypes-20261011.md) implements original GRU/TCN
complex-residual models with an [explicit-state 2.0 contract](../docs/causal-state-contract.md).
`export_causal.py` checks sequential ORT parity before publishing local artifacts;
`scripts/probe_causal_core.py` also checks Rust parity and irregular chunking.
`scripts/fit_causal_tiny.py` fits identity, ideal lowpass and real MP3 training clips,
using 64 burn-in hops and the production input detector trajectory. These are
learnability probes, not production weights or full-library training. The six runs
finished; matched-budget fresh training, independent testing and listening remain pending.

Build `soundex-state-parity`, `soundex-stream-eval` and `soundex-gain-data` first.
Run either script with `PYTHONPATH=. .venv/bin/python scripts/<name>.py --help`.
New artifacts require Model gain/phase and 64 observed warmup hops with aligned dry
audio. Benchmarks accept `SOUNDEX_HIGH_BAND_GAIN=model` and record this setting.
State warmup counters are separate from deadline availability; a timely dry packet
does not prove that enhancement was available. No raw audio/checkpoints are published.


## P2 matched training

`run_causal_study.py` trains fresh legacy/GRU/TCN models using the same frozen
paired draws, native-rate crops, 64-hop burn-in and `causal-reconstruction-v1`.
The registered comparison is 4096 updates × batch 4 × two seeds per backbone.
See [methods, numerical validation and limits](../docs/causal-training-20261012.md).

Build `soundex-gain-data`, `soundex-prediction-replay` and `soundex-stream-eval`
before running. From this directory, the active study can be resumed with:

```sh
PYTHONPATH=. .venv/bin/python scripts/run_causal_study.py \
  --manifest ../data/processed/music_library/processed/music_library-7fda9e58ab583617/manifest.jsonl \
  --manifest ../data/processed/speech_library/processed/speech_library-7fda9e58ab583617/manifest.jsonl \
  --selection ../data/reports/optimization-followup-20261010/inpainting/stream-selection.json \
  --detector ../target/release/soundex-gain-data \
  --stream-binary ../target/release/soundex-stream-eval \
  --output ../data/runs/causal-p2-20261012
```

The file lock rejects duplicate processes. Resume requires identical configuration,
training source hash and binaries. Checkpoints retain optimizer and RNG state;
`export_study.py` understands the separate `soundex-p2-v1` checkpoint format.
Every completed arm exports its validation-selected checkpoint and records actual
Rust output metrics. No checkpoint is automatically promoted or uploaded.
