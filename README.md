# SoundEx

[CI](https://github.com/backrunner/soundex/actions/workflows/ci.yml) · [Apache-2.0](LICENSE)

SoundEx is an experimental Rust library and CLI for restoring compressed audio.
Its goal is high-frequency spectral inpainting followed by DSP fusion and artifact
control, bringing the final audio closer to its original lossless reference.
It combines a lightweight magnitude/phase ONNX model with causal spectral processing.
The repository includes the DSP, inference engine, real-time adapter and PyTorch training pipeline.

**Status:** the software and runnable integration demos are available. Qualified
restoration weights are not yet published. Bundled synthetic identity graphs
exercise inference but do not improve audio quality. See [model status](models/README.md).

[P44 calibration](docs/pcm-fidelity-calibration.md) shows that existing identity
damage is much larger than numerical reconstruction noise. The completed
[P45 small-model pilot](docs/anchored-spectrum-pilot.md) stalls under its absolute
displacement guards: the loosest arm accepts only 2/3 updates across its two seeds,
and stricter arms accept none. Some rejected low-band corrections reduce error
against the clean reference; the preservation policy needs revision. No candidate
is qualified or promoted.

[P46](docs/reference-preservation-pilot.md) completed all twelve trajectories and
22 frozen endpoint reviews. Correcting preservation improves music MR by about
0.06-0.07% at the loosest ruler, but only 5-6 of 128 attempts take effect per arm.
Additional reference gradients do not increase acceptance, and all changed
candidates retain preservation failures on separate reused-TRAIN review clips.
No new weights are qualified; further work must address per-example preservation
and output coupling before repeating longer training.

[P47 attribution](docs/output-coupling-audit.md) isolates direct low-band updates
as the main source of current low-band movement. Reverting those updates retains
almost all of P46's small spectral gains, but does not fix identity or phase
recovery. Input-band bypass also removes some useful low-band correction. A
small adapter with isolated output support is tested in the completed
[P48 pilot](docs/spectral-adapter-pilot.md). All four 128-step runs and raw/native
reviews finish. High-band isolation preserves low frequencies more effectively,
but small spectral gains retain identity/phase/dynamics regressions. The first
Adam step overshoots even on its own fit data; update-scale calibration is needed
before another run. Final compute stays below 5 ms, but nine paced delays remain.
No new restoration weight is qualified.

[P43](docs/spectral-priority-update-audit.md) gives a positive spectrum-priority
diagnostic: at fixed radius 0.001, amplitude improves on 28/32 reused-training
transfer batches and MR on 29/32, versus 13/32 and 14/32 in the P42 control.
Phase/dynamics/identity tradeoffs remain. These reversible probes are not new
trained weights or quality qualification. The [CPU attribution run](docs/cpu-time-attribution.md) separates calling-
thread CPU time, enclosing wall time and wakeup delays. Historical real-time failures
remain open. Consult the [failure registry](docs/failed-experiments.md) before repeating
a route. All maintained project documentation is in English.

[P38](docs/detector-condition-study.md) completed all four 16,192-step CPU trajectories
and native reviews. Fixed detector conditions worsened music MR by 0.19–0.56% versus
matched controls; removing magnitude fusion did not reverse this. Neither path passed
the research screen. Across 280,000 timing hops, max compute was 5.411 ms, with two
≥5 ms calls, 11 service overruns and 27 paced delays. No promotion or new training.

Earlier protocols, full results and counterexamples are indexed in
[research history](docs/research.md) and the [failure registry](docs/failed-experiments.md).

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
