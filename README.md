# SoundEx

[中文](README.zh-CN.md) · [CI](https://github.com/backrunner/soundex/actions/workflows/ci.yml) · [Apache-2.0](LICENSE)

SoundEx is an experimental Rust library and CLI for restoring compressed audio.
Its goal is high-frequency spectral inpainting followed by DSP fusion and artifact
control, bringing the final audio closer to its original lossless reference.
It combines a lightweight magnitude/phase ONNX model with causal spectral processing.
The repository includes the DSP, inference engine, real-time adapter and PyTorch training pipeline.

**Status:** the software and runnable integration demos are available. Qualified
restoration weights are not yet published. Bundled synthetic identity graphs
exercise inference but do not improve audio quality. See [model status](models/README.md).

The [current native-lossless run](docs/native-training-20261009.md) uses 300
reviewed music works plus VCTK speech, from fresh initialization with MUSDB and
Slakh disabled. Raw audio, prepared pairs and training checkpoints stay local;
the repository publishes their source documentation and reproducible workflow.
The original 200 epochs are complete; [loss auditing and controlled continuation](docs/loss-audit-20261010.md)
address the remaining fidelity issues. Deployment validation still determines release qualification.
The [inpainting objective and neural/DSP paths](docs/inpainting-objective.md) make
high-band reconstruction the main training target; DSP-only and hybrid paths remain experimental.
The [P2 matched training study](docs/causal-training-20261012.md) has completed all six local runs:
fresh legacy, GRU and TCN backbones, 4096 updates each across two seeds, with a
shared reconstruction objective and production DSP validation.
Experimental weights have not replaced the default model. The
[first-seed review](docs/causal-p2-review-20261012.md) finds gains over the matched
fresh control, but no overall win over the previous checkpoint and excess
generation on naturally weak high bands. The [P3 controlled continuation](docs/weak-band-study-20261012.md)
has completed both seeds with excess-energy supervision and identity/generalization controls.
It reduces excess generation but retains restoration tradeoffs. The optional
[smooth limiter](docs/limiter-study-20261012.md) reduces DSP distortion and avoids
recompressing worker output in the callback; the default remains unchanged.
The [P4 matched continuation](docs/balanced-restoration-study-20261012.md) completed six runs against
the smooth DSP. Bidirectional energy supervision improves energy recovery but regresses
other fidelity measures; no candidate is promoted.
The [P5 amplitude representation study](docs/envelope-study-20261012.md) completed six
fresh short/tiled/past-context runs. Past-only features improve one transient measure
but do not establish a consistent amplitude-quality win. Band-envelope/shape diagnostics
guide the next structural experiment; the 3.10 MiB amplitude probe is not a deployable audio model.
The [P6 band-envelope/shape study](docs/factorized-envelope-study-20261012.md) completed six
matched fresh runs and native DSP comparisons. Separating energy and shape reduces some
raw spectral errors but sacrifices energy recovery, without a consistent final-output win.
The 3.13 MiB probe meets the diagnostic compute budget; default weights remain unchanged.
The [P7 energy-coordinate experiment](docs/energy-coordinate-study-20261012.md) completed four
matched runs and 890 native DSP replays. Additive energy prediction improves average energy
recovery but regresses spectral and transient errors; no candidate is promoted.
The [objective audit](docs/energy-objective-audit.md) locates the main regression in excess
energy on naturally weak bands and documents loss-ranking and aggregate-score tradeoffs.
The [P8 reconstruction-objective study](docs/linear-reconstruction-study.md) completed four runs,
but every selected checkpoint is step zero: neither objective beats the initial common score.
The [optimization audit](docs/linear-optimization-audit.md) isolates first-step energy overshoot
and tests input-relative updates without changing loss or model capacity. These are diagnostics;
no new weights are promoted.
The [P9 shape-coordinate study](docs/shape-coordinate-study.md) separates band-relative shape
updates from exact identity initialization, with matched training and clean-identity native DSP
controls. All six runs and 2,136 native replays are complete; small raw-loss gains
still cause identity damage. The [P10 rendered-output study](docs/rendered-restoration-study.md)
now supervises DSP synthesis directly and balances identity preservation. Its
pilot and six 2048-step runs reduce damage but select step zero in every arm.
The completed diagnosis separates a generalization gap from preservation pressure;
no candidate has demonstrated a balanced restoration win. The
[P11 fixed-set fitting diagnostic](docs/fixed-fit-diagnostic.md) separates direct
amplitude/complex fitting from the shared network. All eight runs and 90 native
replays completed: direct fitting has substantial headroom and the existing network
can fit the small set, but identity damage and generalization remain unresolved.
The [P12 subset-scaling study](docs/subset-scaling-study.md) now compares 5/10/40
recordings at equal update and equal per-recording exposure budgets, with new-region
and reused-development checks fixed before training.
The [baseline audit](docs/baseline-generalization-20261012.md) identifies gain-policy tradeoffs
and limiter distortion even with an identity model; baseline overfitting is not established.

The [V4/architecture comparison](docs/optimization-20261011.md) evaluates complex
reconstruction, amplitude/phase interactions and prepared real-time startup.
The [V5 goal review](docs/goal-refinement-20261011.md) tests reconstructed energy
calibration and circular phase features, with recording-held-out rechecks and
explicit data-coverage limits. Experimental weights remain local.
The [V6 deployment-aware continuation](docs/deployment-refinement-20261011.md)
adds edge-gain matching and reconstructed transient supervision to the training proxy.
The [architecture research and capacity study](docs/architecture-research-20261011.md)
reviews recent restoration methods against the streaming budget and tests bounded
global spectral refinement with matched representation and capacity comparisons.
All three new arms completed eight epochs; the [measured results](docs/architecture-study-20261011.md)
show a lower validation loss, but no balanced quality win that warrants promotion.
The [causal follow-up](docs/causal-context-20261011.md) implements eight past frames
with unchanged output delay and two matched eight-epoch runs; it remains experimental.
The [V7 reconstruction study](docs/multires-reconstruction-20261011.md) adds real DSP
warmup and multiresolution waveform supervision. Two four-epoch runs are complete;
measured improvements remain mixed, so neither candidate replaces the baseline.
A [bounded DSP gain calibration study](docs/gain-calibration-20261011.md) fits ten
coefficients from production analyzer observations. Its regression improves but
actual high-band recovery regresses, so the default gain policy remains unchanged.
The [phase reconstruction study](docs/phase-reconstruction-20261011.md) retains
model magnitudes while anchoring phase to surviving input power. Waveform fidelity
and some high-band metrics improve, but music remains short of balanced restoration;
the new DSP policy is opt-in and no weights are promoted.
The [architecture reset review](docs/architecture-reset-20261011.md) broadens the
research to complex spectral generation, causal temporal models, differentiable
DSP and generative training. It proposes fresh backbone experiments with explicit
quality and streaming gates. The [first implementation stage](docs/causal-prototypes-20261011.md)
now includes original GRU/TCN prototypes, an explicit-state Rust/ONNX contract,
production-DSP oracle diagnostics and six completed tiny-set learning probes.
Full-library matched training and qualification remain pending.

## Quick start

Requires **Rust 1.88+** and a C/C++ build toolchain. The default build downloads
ONNX Runtime binaries; the first build needs network access. Run from the repo root:

```bash
git clone https://github.com/backrunner/soundex.git
cd soundex
cargo run --locked -p soundex-core --example offline_demo
cargo run --locked -p soundex-core --example realtime_demo
```

The offline demo generates two seconds of stereo audio, runs the included ONNX
fixture and writes `target/demo/input.wav` and `target/demo/output.wav`. The
real-time demo simulates 128-frame callbacks and prints worker/fallback counters;
it does not open an audio device. Both demos accept your own qualified **256/128**
model as the first argument:

```bash
cargo run --locked --release -p soundex-core --example offline_demo -- /path/to/model.onnx target/demo
cargo run --locked --release -p soundex-core --example realtime_demo -- /path/to/model.onnx
```

## File CLI

Analyze without a model:

```bash
cargo run --locked -p soundex-cli -- target/demo/input.wav --dry-run
```

Exercise the full file pipeline with the bundled test model:

```bash
cargo run --locked -p soundex-cli -- target/demo/input.wav \
  --model tests/fixtures/low-latency-identity.onnx \
  --output target/demo/cli-output.wav --bits 24
```

For actual restoration, replace the fixture path with a qualified model. Inputs:
MP3, AAC, FLAC, WAV and OGG Vorbis. Outputs: 16/24-bit PCM or 32-bit float WAV.
Use `cargo run -p soundex-cli -- --help` for options.

## Real-time integration

Create `RealtimeProcessor` on a control thread before starting audio. Its default
worker preparation runs inference before playback and waits on this control
thread only (up to 10 seconds). Pass
matching interleaved input/output slices from the audio callback:

```rust,no_run
use soundex_core::{RealtimeProcessor, SoundExConfig};

let mut stream = RealtimeProcessor::new(
    SoundExConfig::with_model("/path/to/model.onnx")
        .sample_rate(48_000)
        .channels(2),
)?;
let input = [0.0_f32; 128 * 2];
let mut output = [0.0_f32; 128 * 2];
stream.process(&input, &mut output)?; // audio callback
let health = stream.stats(); // observe outside the callback
stream.shutdown(); // control thread
# Ok::<(), soundex_core::SoundExError>(())
```

The callback performs no inference, heap allocation or mutex locking. A bounded
worker queue handles ONNX inference. Late/invalid results fall back to aligned
dry audio; channel-linked fades, input sanitization and final limiting keep
output continuous. Monitor deadline misses and worker failures during integration.

The live adapter supports mono/stereo **44.1/48 kHz**, **FFT 256 / hop 128**.
Its fixed added delay is **256 frames**: **5.80 ms at 44.1 kHz**, **5.33 ms at 48 kHz**.
Use hop-aligned **128-frame callbacks**. The software target is **<8 ms**, with a
**<10 ms redline**, including callback time and extra integration buffering.
Device/driver buffering must be measured separately; these fixed-delay figures
are not physical-device or end-to-end latency measurements.

[Library APIs](docs/library.md) · [Real-time acceptance standard](docs/realtime-standard.md) ·
[Latency/resource benchmarks](docs/performance.md)

Use synchronous `SoundExProcessor` for files and offline evaluation. It invokes
ORT directly and should not run in an audio callback. Build core without the
inference backend for model-free analysis with `--no-default-features`.

## Training and model contract

See [training setup](training/README.md) for dependencies, mix-only preprocessing,
deterministic data splits, CPU/MPS/CUDA setup and export commands. Datasets are
obtained separately and retain their upstream terms.

The default model uses FP32 `[batch, 2, 1, 129]` input/output, dynamic batch only,
checkpoint schema 1.2 and artifact schema 1.3. Export binds preprocessing and provenance to
the checkpoint and publishes atomically only after mandatory ONNX/ORT validation.
Rust validates tensor and `soundex.*` metadata before processing.
The optional [finite causal context](docs/causal-context-20261011.md) uses artifact
1.4: nine observed input frames and one output frame, with no lookahead. The host
retains per-channel history; eight-hop enhancement warmup keeps aligned dry output.

[Parity budgets](docs/parity.md) · [Held-out evaluation](training/evaluation/README.md) ·
[Model card](models/MODEL_CARD.template.md) · [Weight release checklist](legal/MODEL_RELEASE_CHECKLIST.md)

## Development

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace
cargo check --locked -p soundex-core --no-default-features --all-targets
cd training
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

Benchmarks are separate executable targets and need explicit model/report
configuration; see the benchmark guide rather than running them as ordinary tests.
See [contribution guide](CONTRIBUTING.md) for conventions.

## License

Source, documentation, demos, synthetic test graphs and **official released weights**
use **[Apache License 2.0](LICENSE)**. Released weights permit application integration,
redistribution, modification and commercial use under that license. Each weight
release includes its model card, checksum and applicable credits.

Training audio and dependencies retain their own terms. The existing MUSDB-trained
candidate is an unpublished research experiment, excluded from the official
application-ready weight route. Training accepts diverse audio folders without a
license whitelist; maintainers review source grants and lineage before publishing
official weights.

[Weight license scope](MODEL_WEIGHTS_LICENSE.md) · [NOTICE](NOTICE) ·
[Licensing overview](legal/LICENSING.md) · [Training data policy](legal/TRAINING_DATA.md) ·
[Dataset survey](docs/datasets.md)

Copyright 2026 BackRunner and SoundEx Contributors.
