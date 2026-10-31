# P67: verified fused spectral supervision and matched phase calibration

Date: 2026-11-01. Status: numerical/native correspondence and nonzero native
preflights pass; the first registered trajectory is training. Endpoint reviews
are pending. Default weights are unchanged. No new restoration or runtime qualification
is claimed.

## Changed hypothesis

[P66](phase-transport-p66.md) found that most P65 phase-correction power lands in
bins whose phase native DSP retains from the input. Its output-support mask removes
those corrections without changing native audio. Magnitude fusion also changes the
MR tradeoff, so support alone is insufficient. P67 isolates output availability and
an original-reference spectral objective evaluated after deployment operations.

This is not a repeat of [P10](rendered-restoration-study.md). P10 jointly trained a
different magnitude model, used observed phase for primary rendering, and used an
identity-through-DSP preservation target. P67 freezes the fully trained P63
magnitude backbone, optimizes only the small P65 phase head, retains direct
original-reference supervision, and uses original lossless PCM for every
reconstruction term, including identity roles. P65's missing static cross-bin
phase information remains a separate representation limitation.

## Verified differentiable path

`models/fused_spectrum.py` reuses existing unit-phasor fusion, periodic-Hann OLA,
smooth-knee limiting and gate semantics. It preserves the native retained-bin copy
instead of a dB/linear/dB round trip. Enhancement is suppressed for the native
80-hop warmup, separately from the inherited 64-hop detector-condition mask.
History still advances during warmup, silence and bypass. Supervision starts at
hop 89 and excludes the final 128 samples.

Only input-derived decisions and model predictions enter the DSP path. Operators
are fixed, not learned labels. The target remains original aligned, unclipped
lossless PCM. Correct identity predictions can therefore retain nonzero loss when
the limiter changes a loud original; P66's limiter/reference conflict is not hidden.

The proxy covers independent mono training views and the diagnostic Model-gain /
Model-phase policy with a 1 kHz crossover. It is not a generic renderer for other
policies or a shared physical-stereo gate. Context/channel inference parity is
checked separately. Loss analysis windows introduce no inference lookahead or
buffering. Actual Rust output remains the final evaluation path.

### Frozen native correspondence

`data/runs/fused-spectrum-audit-p67-20261101/registration.json` freezes all 100 P66
stage traces, consumed inputs and executable Python before scoring: four frozen
P65 graphs and identity DSP, eight music/two speech TRAIN recordings, paired and
identity roles. Same native observations isolate DSP; this is not an end-to-end
frontend/model quality test or independent material.

Registered bounds reuse feature parity v2 and causal-renderer limits: complex
relative RMS `2e-4`, PCM peak `0.00006103515625`, PCM RMS `2e-5` and reconstructed
crossover-weight peak `2e-5`. All fully supported emitted samples from sample 128
onward are checked, including warmup; startup is reported separately. No threshold
or recording selection changes after scoring. All **100 cases pass**:

| Maximum over cases | Measured error |
| --- | --- |
| Fused complex-spectrum relative RMS | `8.07268e-8` |
| Reconstructed crossover weights, absolute | `1.60933e-6` |
| Final PCM maximum absolute difference | `3.57628e-7` |
| Final PCM error RMS | `2.42332e-8` |
| Pre-limit PCM error RMS | `2.81550e-8` |

The artificial single-window startup tail reaches `6.18445e-5` before the gate;
final-output startup difference is exactly zero. This separately reported tail
was outside the preregistered full-overlap gate and never enters supervision.
It is not a dropped counterexample or grounds for changing an audio threshold.
All records remain in `assessment.json` and the verified completion receipt.

Six new checks cover correct-reference behavior, silence/`1e-8` amplitude, finite
gradients, magnitude/phase finite differences, warmup exclusion, exact retained
bins, the unchanged target through limiting, and auxiliary gradients reaching the
phase head. The common objective exactly matches the existing P63/P65 path on a
model-backed check. All 33 targeted phase/direct/fused tests pass locally; Ruff and
formatting pass. All 16 native stress replays also pass before optimizer updates.

## Registered 2 x 2 experiment

Every arm uses the same 196,162-parameter causal model, with only 25,025 phase-head
parameters trainable. Backbone weights, per-frame predicted dB and base state
tensors stay frozen. State remains 12,354 floats per physical channel. The
production ABI and bundled weights are unchanged.

| Arm | Phase output support | Per-example optimized risk |
| --- | --- | --- |
| `broad-common` | Existing broad support | `log1p(direct) + log1p(raw MR)` |
| `native-common` | P66 native availability | Same common objective |
| `broad-fused` | Existing broad support | Common + `log1p(final-output MR)` |
| `native-fused` | P66 native availability | Same augmented objective |

Both MR terms retain fixed multiresolution amplitude, log-shape, energy-envelope
and reference-relative dynamics terms. Direct supervision is unchanged. The new
auxiliary coefficient is fixed at 1 before training, without track-specific
tuning, waveform-distance terms or weight search. Differently defined training
totals are not compared: all endpoints are rescored under both fixed objectives
and their individual components.

Seeds 20261012 and 20261013 receive all four arms: **eight trajectories, 512 Adam
updates each**, learning rate `1e-4`, default betas/epsilon, gradient clip 5 and
four paired/identity batches per update. Initialization, data order, optimizer and
exposure match within each origin. Both declared factors are isolated, without
changing capacity. All endpoints must finish and freeze before comparison; no
intermediate winner or shortened control. This diagnoses early optimization,
not convergence, full training or an architecture ceiling.

Each trajectory reuses P65/P63's first 2,048 draws: 10,240 paired crops plus
identities, 80% music/20% speech, 182 recordings, nine codecs and mono/left/right/
mid/side views. Unique supervised MR duration is 3,561.220 seconds; direct-analysis
support is 3,599.405 seconds. Repeating the schedule does not create new independent
material. Genre/region distributions and unknown labels remain in registration;
coverage is unequal. Before launch, source/work aliases are checked against every
current local evaluation catalog and the approved source catalog.

## Launch failure and retained correction

The first attempt, `data/runs/fused-phase-p67-20261101/`, stopped at input-hash
verification with **zero optimizer updates and no native preflight renders**.
`failure-1793472205298678000.json`, source, registration and log are retained.
The new preparation code had copied P65's historical registration without applying
its existing unused-runtime-alias canonicalization helper. This repeated a known
integration failure; it is not evidence of failed optimization or model quality.

The inherited alias `target/release/soundex-context-replay` changed during later
builds. Actual calls use the frozen P63 asset, whose bytes still match the original
alias hash `5750fdb17984f74f3958e925b6e831e3ec81033ac798f1466b670a6fb5a29eba`.
Preparation now invokes `canonical_runtime_alias`, which removes only that proven
byte-identical unused alias and records the binding; non-equivalent aliases fail
closed. The original attempt is unchanged. The corrected registration is
`data/runs/fused-phase-p67-20261101-v2/`, with the same schedule, objective, budget,
initialization and frozen runtime. No checkpoint or quality score was discarded.

## Review and operational state

The version-2 `launch.json`, `run.log` and `progress.json` identify the live process
and stage. Local training runs in the background with idle-sleep prevention tied
to its process. Check live state before claiming completion. Input/source hashes,
package versions and evaluation reservations are checked before/after execution.

Before optimization, deterministic nonzero exports passed sequential ONNX and
16 Rust checks at 44.1/48 kHz, mono/silent-left stereo, both support modes and origins.
The pipeline then finishes all eight endpoints and automatically reviews the first
eight inherited TRAIN draws (32 music/eight speech recordings) under raw MR, direct
loss and fused MR. Native review covers the same two-batch subset (eight music/two
speech), with both zero heads, all eight endpoints, legacy and identity DSP.
Every per-recording regression remains in the assessment.

Any fitted gains still need broader source-isolated review, listening and timing
evidence. No compute/deadline qualification runs concurrently with training.
Previous late-hop failures and the limiter/reference conflict remain valid.
