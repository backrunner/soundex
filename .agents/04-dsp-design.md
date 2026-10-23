# SoundEx DSP design

## Role in restoration

DSP turns a causal model prediction into stable audio through band fusion, gain,
phase handling, overlap-add and safety control. It is not the original lossless
supervision target. Evaluate model-only, DSP-control and combined paths on the same
material; preserve original unclipped references when reporting limiter tradeoffs.

## Analysis and synthesis

The current low-latency geometry uses FFT 256 / hop 128. Windowing, overlap support,
normalization, startup and flush must agree between training analysis and Rust.
Validate known spectra, round trips, silence, low energy and arbitrary block boundaries.
Do not hide finite startup/tail regions in aggregate metrics or silently shift pairs.
See the implementation in `crates/soundex-dsp/src/stft.rs` and `window.rs` for exact
normalization; the [parity guide](../docs/parity.md) defines measured correspondence.

## Bandwidth, gain and phase

Bandwidth detection conditions processing and gating. It must behave smoothly across
frames and preserve a fixed timeline. Model conditioning and actual DSP gating are
separate interventions; a constant model condition does not disable the detector.

Gain matching should not replace spectral restoration with indiscriminate treble
boosting. Measure high-band shape, envelope and transients against the reference.
Phase is circular: interpolation across the ±π boundary must respect complex/unit-
circle geometry. Surviving input phase may be useful, but heuristic fusion is a
control to evaluate, not assumed ground truth.

Crossovers must avoid seams and preserve low-band content. Temporal consistency follows
reference changes rather than smoothing transients. Each physical channel maintains
its own model history; shared stereo policies must be verified with real stereo,
channel-isolation and mid/side measurements.

## Safety and real-time behavior

Keep output finite, bounded and continuous. Limiting can move output away from an
unclipped reference; report that explicitly with identity-through-DSP controls.
Do not alter the reference to disguise the tradeoff. Callback fallback/recovery is
smooth and timeline-preserving, with no callback allocation, inference or blocking.
See [real-time requirements](../docs/realtime-standard.md).

## Current findings

[P37](../docs/fusion-boundary-audit.md) isolates magnitude remixing, phase fusion and
gating. Removing magnitude remixing slightly improves spectral means, but identity
and local-band regressions remain; production defaults are unchanged. [P38](../docs/detector-condition-study.md)
shows that fixed-condition retraining does not reproduce the frozen-inference gain.
[P39 timing](../docs/interleaved-timing-audit.md) retains long-tail deadline failures.
No small p99 value or synthetic stereo probe certifies glitch-free devices.

Do not duplicate hard-coded DSP equations or thresholds here: use implementation,
registered configurations and the detailed study records so examples cannot silently
drift from the deployed path. Consult the [failure registry](../docs/failed-experiments.md)
before repeating a rejected fusion or calibration intervention.
