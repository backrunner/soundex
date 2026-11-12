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

[P75 research](docs/realized-supervision-p75.md) audits the objective and relevant
official implementations. Thirty-two frozen gradient probes find conflicts
between realized magnitude and complex fidelity; adding a global complex term
does not resolve them. The next diagnostic targets phase-head gradient routing,
with native/finite-step verification before training. No weights or inference
behavior changed; the initial numerical failure and corrected audit are retained.

[P74](docs/phase-temporal-p74.md) adds phase relations inside the small temporal
backbone, with a matched 216,802-parameter magnitude-only control. All four
16,384-update trajectories and fixed reviews are complete. Native spectral MR
changes -0.040630% / +0.165277% versus the matched controls; independent HF SC does
not improve in either seed. The added observation fails its registered criterion,
despite learned phase-history use and 3.24–3.45% MR gains versus historical legacy.
Trained compute peaks at 2.449333 ms over 52,000 hops, but four paced completions
are late. Both quality and continuous-deadline qualification remain open; no
default promotion, independent test or listening acceptance is claimed.

[P71](docs/evaluation-capacity-p71.md) adds a calibrated spectral/fidelity panel
and recording-paired uncertainty. A same-depth twofold parameter fixture passes
native parity and local compute/deadline checks (double maximum 2.243 ms).
[P72](docs/band-capacity-training-p72.md) completes all four 16,384-update trajectories.
Double models reduce long-form native MR 0.69–0.92% versus matched small models
and 4.06–4.10% versus legacy, with mixed shape/source-panel and HF PCM tradeoffs.
Trained compute stays below 5 ms but one paced completion is late. These are
development rechecks; no overall quality qualification or default promotion.
[P73 attribution](docs/capacity-tradeoff-p73.md) confirms that magnitude recovery
amplifies remaining phase mismatch, already before DSP. All frozen cases and the
paced failure remain recorded; it proposes a small temporal phase-information
experiment, not another unchanged capacity increase or a qualified new weight.

[P63](docs/band-training-p63.md) completed all six fresh 16,384-update endpoints
and a 160-case long-form recheck. Band-TCN native music MR improves 2.84–2.86%
versus parameter-matched controls and 2.67–2.93% versus legacy across both seeds,
with HF PCM/dynamics and codec-specific regressions. These are reused sources,
not independent tests. The original timing grid includes a 12.73 ms paced band
call and late completions; listening ratings and device evidence remain pending.

[P64 attribution](docs/band-attribution-p64.md) locates the remaining tradeoff:
MP3 magnitude gains increase phase-interaction cost, while AAC behaves differently.
DSP reduces some PCM regression. [P65](docs/band-phase-p65.md) tested a small
phase-only head with frozen magnitudes and actual Rust correspondence. Both
512-update calibrations finish: small raw fit gains do not improve native music
output. Its 52,000-hop compute screen stays below 5 ms, but one call exceeds the
hop period. Worker wake observations are extended; overall quality and real-device
qualification remain open.

[P66](docs/phase-transport-p66.md) locates a concrete mismatch: about 96–97% of
P65's pooled HF phase-correction power lies in bins where DSP retains input phase.
An experimental output-support constraint matches native support and preserves all
40 frozen-endpoint outputs exactly. It has not been retrained and is not a quality
gain. Stage/identity controls also separate fusion effects from limiter distortion.

[P67](docs/fused-phase-p67.md) completes all eight 512-update phase calibrations
and 240 native-role reviews. Native-support/fused-loss heads reduce mean native
music spectral error by 0.002671% / 0.001439% versus their zero-phase parents,
with source, dynamics and HF PCM exceptions. These are tiny TRAIN-fit gains,
not overall quality or convergence. The initial launch rejection remains documented.

[P68](docs/phase-gradients-p68.md) completes 64 frozen phase-gradient probes and
adds an optional cross-frequency phase observation with an equal-width control.
Native support improves gradient agreement; saturation does not explain the tiny
P67 gains. Four untrained nonzero fixtures pass native parity and a 52,000-hop
compute screen (max 1.465 ms, no measured service overruns or paced lateness).
This enables a matched training comparison; it is not improved restoration weights
or real-device qualification. Earlier timing failures remain recorded.

[P69](docs/cross-phase-training-p69.md) completes all four 16,384-update trajectories,
source-isolated reviews and the P55 long-form recheck. Phase training slightly
improves spectra over zero-phase parents, but adjacent-frequency observations trail
the matched repeated-input control in both seeds. Trained compute peaks at 1.419 ms
over 52,000 measured hops, with no service overrun or paced lateness in this grid.
Source/HF PCM/dynamics tradeoffs remain; no default weights are promoted.

[P70](docs/baseline-metric-audit-p70.md) separates the historical legacy model from
matched controls: P69 already reduces long-form native spectral MR by 2.82–3.06%
versus legacy. Independent rulers confirm the spectral progress and the small
matched-control failure. Analytical controls expose energy/log-distance blind
spots; frozen-head probes find weak use of the added observation and remaining
magnitude/phase headroom. No new training, perceptual qualification or promotion.

See the [research history](docs/research.md) and
[failed/unqualified experiment registry](docs/failed-experiments.md) for retained
controls, counterexamples and earlier work.

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
