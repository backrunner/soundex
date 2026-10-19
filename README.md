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

The [P28 small-model mechanism study](docs/small-mechanism-study.md) is complete:
431K–456K parameters, four controlled configurations, two seeds and 2,048 updates
each. It tests decoder normalization, budget-gradient isolation and observed-phase
inputs before any further scaling. None passed the two-seed improvement screen;
outputs remain near identity DSP even without decoder saturation in the diagnostic crops.
All 560,000 timing hops stayed below 5 ms, with five service overruns and 32 paced
completion misses. No production weights are changed.

The [P27 audit](docs/head-collapse-audit.md) retired the 11.92M-parameter route
from further scaling after finding widespread residual-decoder inactivity.
The compact DSP-excitation route remains a later experiment, not a trained model.

[P26](docs/broad-budget-study.md) completed eight full trajectories, native reviews
and 560,000 timing hops without establishing overall recovery. Compute stayed
below 5 ms in that run, but hop deadlines still missed. No weights were promoted.
Earlier studies and continuation commands are indexed in [research history](docs/research.md).

The reviewed library contains 300 native-lossless music works plus VCTK speech;
raw audio, prepared pairs and experimental checkpoints stay local. See the
[training guide](training/README.md), [source documentation](docs/datasets.md)
and [research history](docs/research.md) for protocols, evidence and limitations.

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
