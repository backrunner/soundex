# P67: verified fused spectral supervision and matched phase calibration

Date: 2026-11-01. Status: all eight 512-update trajectories and fixed endpoint
reviews complete. Native-support arms slightly reduce mean native music spectral
error in both seeds, with source, HF PCM and dynamics exceptions. Default weights
are unchanged; no overall restoration or runtime qualification is claimed.

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

The version-2 process has exited successfully and `progress.json` reports complete.
All eight endpoint hashes, 512-update histories, finite tensors and frozen backbone
tensors verify. Both `broad-common` controls reproduce every P65 endpoint tensor
exactly. The completion receipt and all 240 native-role artifact hashes verify;
the corrected run has no failure record. Input/source hashes, package versions
and evaluation reservations were checked before/after execution.

Before optimization, deterministic nonzero exports passed sequential ONNX and
16 Rust checks at 44.1/48 kHz, mono/silent-left stereo, both support modes and origins.
The pipeline finished all eight endpoints and reviewed the first
eight inherited TRAIN draws (32 music/eight speech recordings) under raw MR, direct
loss and fused MR. Native review covers the same two-batch subset (eight music/two
speech), with both zero heads, all eight endpoints, legacy and identity DSP.
Every per-recording regression remains in the assessment. The supplemental
`endpoint-summary-v2.json` derives the matched raw subset and fixed-objective
means from saved reports without rerendering or training. Its receipt freezes the
summary script. The initial summary/receipt and executed source remain retained;
version 2 fixes loop-binding lint and a mutable source reference, and its summary
payload is byte-identical. No quality criterion or result changed.

## Completed endpoint results

Source-equal native error changes versus each frozen P63 backbone with zero phase
correction, on the same eight paired music recordings. Lower is better:

| Arm | MR, seed 20261012 | MR, seed 20261013 | HF PCM RMS, seed 20261012 | HF PCM RMS, seed 20261013 |
| --- | --- | --- | --- | --- |
| `broad-common` | +0.000661% | +0.000735% | +0.001220% | +0.001460% |
| `native-common` | -0.001512% | -0.001302% | +0.001891% | +0.001440% |
| `broad-fused` | -0.000247% | +0.000758% | +0.001643% | +0.001513% |
| `native-fused` | -0.002671% | -0.001439% | +0.002583% | +0.001611% |

Native support improves mean native MR relative to broad support under either
fixed objective in both seeds. Adding the fused term to native support improves
native MR by a further 0.001159% / 0.000137%. Adding it to broad support does not
give a consistent two-seed native MR improvement. This isolates a small support
benefit; it is not evidence that fusion-aware training alone solves the problem.

For `native-fused`, native music MR changes `0.3006278826 -> 0.3006198532` and
`0.2978579337 -> 0.2978536482`. The improvements are very small. MR still worsens
on **5/8 and 2/8** sources; a lower mean is not a uniform win. Its worst MR cases
are case 0 / role 3, `music_library:kcc-original-13300245-feff81af48fff34d`, MP3
VBR q2 (delta `+2.48849e-6`), and case 0 / role 0, Chris Zabriskie's
*What Does Anybody Know About Anything*, AAC 192 kbps (delta `+7.74860e-7`).

Native dynamics error rises 0.002672% / 0.003076%, with 7/8 and 4/8 adverse sources.
The largest dynamics exception in both origins remains *Crunch Me Beautiful*,
MP3 64 kbps, case 4 / role 1: deltas `+2.52724e-5` / `+3.18289e-5`.
Native HF PCM RMS increases by `1.11533e-7` / `6.96624e-8` absolute PCM units;
these tiny increments do not establish audible harm. Nor does a tiny mean spectral
gain establish audible improvement. Preserved-band and every other recorded
metric remain in the assessment; no exception is dropped.

On the wider 32-recording raw music panel, `native-fused` MR improves 0.002035% /
0.002031%. Raw clean-music HF RMS rises `3.17679e-6 -> 3.24549e-6` and
`3.73504e-6 -> 3.76328e-6`: +2.1625% / +0.7561%, but very small absolute changes.
Native clean-music MR instead improves 0.004357% / 0.004058%. Speech native MR
improves 0.001820% / 0.001325% on only two recordings; its second-origin dynamics
and HF PCM regressions remain. Native speech identity is unchanged.

Both common and augmented risks are rescored for every endpoint in `loss-review/`
and the supplemental summary. They must not be ranked by unlike training totals.
For example, the second origin's `native-fused` augmented risk is
`0.3666905165`, slightly above `native-common` at `0.3666904569`, despite its
lower native music MR. These mixed, tiny changes do not establish convergence.

## Decision

Retain native output availability as a supported design correction, while keeping
both fixed controls. The fused auxiliary yields only marginal additional benefit
and does not remove source/dynamics tradeoffs. Do not promote these weights or
repeat a larger unchanged budget on the strength of the mean alone. Before another
training design, inspect gradient allocation in the passed band and the phase
head's missing cross-bin information, separately from a convergence/exposure audit.
The completed 512-update calibration cannot reject the phase architecture family.

Any fitted gains still need broader source-isolated review, listening and timing
evidence. No compute/deadline qualification runs concurrently with training.
Previous late-hop failures and the limiter/reference conflict remain valid.
