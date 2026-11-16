# P82: phase relations after reconstruction and mixed-objective attribution

Completed 2026-11-16. Phase-relation supervision changes materially when measured
after overlap-add. It is connected, numerically checked and consistent with saved
native PCM within the declared correspondence controls. This is a mechanism audit,
not a new weight, optimizer trajectory or demonstrated restoration improvement.

## Question and fixed comparison

[P80](phase-relations-p80.md) tested direct FFT256 relations; [P75](realized-supervision-p75.md)
tested marginal complex distance. Neither isolates direct versus reconstructed
**relational** gradients. [P81](evaluation-audit-p81.md) also establishes that tiny
two-update effects and a scalar FP32 epsilon heuristic cannot decide an architecture's
ceiling. The [failure registry](failed-experiments.md) was consulted before this audit.

P82 uses all four fully trained P74 small endpoints (216,802 parameters), all eight
P80 fit draws, both circular/wrapped kernels and direct/raw/fused observation
domains. No origin, component, source or kernel is dropped after scoring. These
are 32 frozen FP64 endpoint/draw probes, covering the same 40 reused fit recordings
as P80. That is a finite diagnostic bank, not new full-library exposure.

The primary direct/raw/fused spectral objective, coefficient 0.1, original PCM,
input-only native availability and reference phase support are fixed. Circular
distance is `(1-cos(delta))/2`; wrapped distance is the shortest absolute angular
distance divided by pi. The three instantaneous/frequency/time terms receive
equal weight before `0.1*log1p`. The existing phase floor remains relative
amplitude `1e-3` and absolute unnormalized FFT amplitude `1e-6` (-120 dB feature
units, not PCM dBFS). No prediction-dependent support or inferred audible threshold
is introduced. Original magnitude objectives still penalize missing/invented audio.

## Common target and synthesis geometry

All three domains now share one FP64 Hann256/hop128 analysis of the **original
lossless PCM**, identical detached weights and exactly the same fully supported
frames. These are frames 89--253 inclusive (165 frames) in the 256-frame bank,
inside the historical supervised PCM region `[89*128, N-128)`. The previous direct
loss included two more end frames and used cached dB/phase targets. It is retained
as a separately named legacy-target control; its numbers are not silently replaced.

Direct phase is read from predicted complex features. Raw phase is reanalyzed
after causal overlap-add; fused phase is reanalyzed after the existing native DSP
proxy. Offline analysis windows do not add inference lookahead. Synthesis allows
neighboring frames and bins to contribute: frame 88 is a synthesis guard, not a
supervised reference frame or model-history warmup. Tests verify this boundary,
independent batch/channel gradients and zero influence from the unused final frame.
The reference is never passed through the limiter or reconstructed from features.

Compared with the original P80 direct definition, common-target head gradients
have median cosine **0.999820 circular / 0.999731 wrapped**. All are positive,
although wrapped reaches 0.847815 on one probe. Thus target/crop normalization is
not silently equated with exact historical arithmetic. The matched domain
comparison uses the new common target for every arm.

## What changes with measurement location?

These are canonical phase-head Euclidean gradient cosines across all 32 probes.
They are not Adam steps, independence-adjusted statistics or output-quality scores.

| Kernel | Direct vs raw, median | Direct vs fused, median | Raw vs fused, median |
| --- | ---: | ---: | ---: |
| Circular | 0.591131 | 0.261427 | 0.934865 |
| Wrapped | 0.612573 | 0.600255 | 0.931480 |

Circular direct/raw vectors oppose on three probes and direct/fused on four;
wrapped counts are one and four respectively. This is a materially different
derivative, unlike a simple scalar rescaling of the direct auxiliary.

| Auxiliary | Median auxiliary/head-old norm ratio | Negative auxiliary/fused-MR cosines | Negative auxiliary/complex cosines | Negative `(head-old + auxiliary)`/complex cosines |
| --- | ---: | ---: | ---: | ---: |
| Direct circular | 0.425548 | 7/32 | 16/32 | 13/32 |
| Raw circular | 0.244138 | 7/32 | 13/32 | 9/32 |
| Fused circular | 0.223457 | 12/32 | 14/32 | 10/32 |
| Direct wrapped | 1.069807 | 5/32 | 9/32 | 6/32 |
| Raw wrapped | 0.529886 | 5/32 | 2/32 | 10/32 |
| Fused wrapped | 0.483992 | 5/32 | 4/32 | 9/32 |

All six summed **balanced head** directions align positively with fused MR and
HF magnitude in all 32 probes. That does not mean every isolated auxiliary helps
spectra, nor that the entire routed model field or every source stratum improves.
Moving wrapped supervision reduces some auxiliary-only complex conflicts, yet its
sum with the old objective has more such conflicts than matched direct wrapped.
Coefficient/gradient scales differ by measurement domain; they are disclosed,
not norm-matched retrospectively. No universal winning kernel follows.

Component derivatives reproduce each total auxiliary head derivative with maximum
absolute error **4.562331e-19**, within the existing registered decomposition
tolerances. All frozen weights remain bitwise unchanged. Near-zero predictions
still need care: supported bins below the reference absolute phase floor total
9 direct, 92 raw and 95 fused across the 320 role views. These counts include
repeated endpoint views, not independent events. They do not remove support or
justify suppressing desired high-frequency reconstruction.

## Native correspondence and numerical sensitivity

All 40 existing P80 first-role traces are reused: eight endpoints, five draws,
five previously exposed music recordings. Each is an independent mono channel view,
not a physical stereo music test or the complete twenty-recording transfer panel.
Speech native correspondence is not established by this first-role trace subset.
No native binary is rerun and no new output/timing measurement is selected.

All original native metric dictionaries reproduce exactly. Protocol-v2 within-band
dynamics and complete PCM support are added, keeping the old scores intact.
Proxy/native PCM maxima remain **2.980232e-7 peak / 2.404784e-8 RMS**. Both relation
kernels satisfy a per-case analytic correspondence bound at every component.

| Kernel | Median absolute risk discrepancy | Maximum absolute risk discrepancy | Maximum analytic bound |
| --- | ---: | ---: | ---: |
| Circular | 1.066353e-7 | 8.393487e-6 | 0.001601039 |
| Wrapped | 1.852582e-7 | 1.082787e-5 | 0.001019253 |

For these pairs of complex spectra, phase separation is bounded by
`pi * min(1, abs(a-b)/min(abs(a),abs(b)))`; zero-magnitude cases use pi.
The circular and wrapped kernels have geodesic Lipschitz constants 1/2 and 1/pi;
frequency/time edge bounds add the two endpoints, then use the shared weights.
The `1e-12` allowance covers FP64 arithmetic in this check, not audible severity.
Eight cases have exactly zero discrepancy and zero support; no supported native
prediction is exactly zero. The bound can be loose for weak predictions. Small
PCM differences therefore do not imply equally tiny angular-risk differences.
This is same-case risk correspondence, not gradient equivalence, renderer
repeatability or a general noise floor for native candidate comparisons.

## Retained control failure and validation

The first test run passed 59 checks and failed wrapped-direct central finite
differences. Its constant-across-frequency perturbation placed frequency-relation
errors near the wrapped-L1 zero cusp (affected values approximately
`-2.686e-6` to `+2.371e-6`), inside the `1e-5` central-difference step. A derivative
across a nondifferentiable cusp is not a valid smooth-region check.

The exact failed source, assertion, log and diagnostic values are retained under
`data/runs/realized-relations-p82-controls-20261116/`. The revised fixture includes
an explicit frequency-varying error, leaving the kernel, floor and tolerances
unchanged. All **60 tests pass with warnings treated as errors**, including the
retained P80 cusp/periodicity controls, original-reference/silence/weak tests,
raw/fused magnitude and phase finite differences, actual phase-head routing,
parameter derivatives and analytic correspondence bounds. The passing snapshot
is `data/runs/realized-relations-p82-controls-smooth-20261116/`.

Main evidence is `data/runs/realized-relations-p82-20261116/`; the whole-grid
descriptive summary is `data/runs/realized-relations-p82-summary-20261116/`.
The balance follow-up below keeps the initial strata intact rather than presenting
hypothetical single-stratum updates as mixed-training behavior.

## Actual mixed-objective directions

The initial per-stratum `old + auxiliary` vectors describe hypothetical training
on that stratum alone. They cannot show how the actual paired/identity-balanced,
music/speech-mixed update affects it. A separately registered follow-up replays
all 32 probes, reproducing **every original field exactly**, and adds the actual
balanced head direction and full routed direction against each stratum's losses.
The latter retains all old base gradients and adds auxiliary gradients only to
the phase head. A new test independently compares it with the real training
`backward` implementation, including the old-only control. Final validation is
**61 passing tests**, with warnings treated as errors.

| Balanced head field | Paired music MR conflicts | Clean music MR conflicts | Paired speech MR conflicts | Clean speech MR conflicts |
| --- | ---: | ---: | ---: | ---: |
| Old control | 3/32 | 6/32 | 6/28 | 10/24 |
| Direct circular | 1/32 | 8/32 | 5/28 | 12/24 |
| Raw circular | 1/32 | 8/32 | 4/28 | 8/24 |
| Fused circular | 2/32 | 9/32 | 4/28 | 6/24 |
| Direct wrapped | 3/32 | 6/32 | 15/28 | 7/24 |
| Raw wrapped | 3/32 | 8/32 | 8/28 | 2/24 |
| Fused wrapped | 3/32 | 7/32 | 9/28 | 3/24 |

Four paired-speech and eight clean-speech vectors have zero target-gradient norm;
their cosines are undefined and are excluded from those denominators, not counted
as success. Positive gradient cosine predicts descent under the negative
Euclidean direction; negative predicts a first-order conflict. Counts are repeated
endpoint/draw diagnostics, not independent recording probabilities.

Raw circular introduces no new paired-music or paired-speech head MR conflict
relative to old, but introduces **three** clean-music conflicts while resolving
one. Examples of the **mixed direction against a four-recording stratum** are:

- Seed-20261012 phase, draw 3: cosine `+0.046351` to `-0.051992`.
- Seed-20261013 magnitude, draw 11: `+0.007505` to `-0.061419`.
- Seed-20261013 phase, draw 5: `+0.317843` to `-0.481168`.

These cannot be assigned to one track as its output error. Full source-hash lists,
including draw-5 music sources `8dc96920c00b4b21b2a1e1d375e09576816f5fdd7c9bcb7fe493564cbc052740`,
`e3a9769903f5f66e5bb220a4cd2bb3926e13c8dbb4b6bcab3e28307c88f5e771`,
`d5a00ba67556a11790de981af61d132f1e95308e90b60a7f6bbdeeff6dae2cf8`
and `71f04f184b1777e29bc13c19ff7ff12cce803cae96da54510421fde17c30acc7`,
are retained in the balance assessment. Every other adverse/resolved case remains.

For the **full routed model field**, all six recipes retain the old MR conflict
counts: paired music 1/32, clean music 7/32, paired speech 7/28 and clean speech
1/24. No sign transition occurs on these strata; most full-field cosines barely
change. The unchanged base direction dominates this Euclidean view. It would be
incorrect to report the more favorable head-only counts as whole-model progress.
Persistent Adam and its parameter-group scaling can change these directions;
this audit does not establish the size or sign of actual optimizer movements.

Evidence: `data/runs/realized-relations-p82-balance-20261116/`, including registration,
source snapshot, all exact replays and completion receipt. The independent mixed
backward control is retained under
`data/runs/realized-relations-p82-final-controls-20261116/`. No additional native
render, transfer scoring, source selection or timing retry was made in this review.

## Decision and next experiment

Reconstructed phase relations are a valid, distinct mechanism to test. Merely
moving the same auxiliary after DSP does not solve all objective conflicts, and
wrapped supervision's paired-speech conflicts make it an unsuitable automatic
replacement. Raw circular is the leading **hypothesis** for a matched finite-update
comparison: it changes the head direction, avoids new paired spectral conflicts
in this panel, and leaves the model's supervision target independent of DSP.
Its clean-music counterexamples remain a required stratum, not an exemption.

The next comparison must retain old-only and common-target direct-circular
controls, include raw/fused circular arms, copy the same persistent Adam state
and keep the original magnitude objectives. Measure the actual finite routed
update and native spectral/absolute preservation effects on all origins before
committing an adequate full-training budget. Use protocol v2, not a phase-sign
veto or the retired epsilon ruler. A short calibration is an optimization check,
not a family ceiling or evidence that a small phase-only change will automatically
improve the whole model. Do not select a seed, reset momentum, increase coefficients
or repeat an unchanged two-update recipe as a substitute for that question.

No optimizer update or new weight is produced in P82. The inference graph, causal
state, DSP and fixed output buffering are unchanged. There is no new compute,
CPU/RSS, listening, independent-test or device qualification. All failed and
completed evidence is preserved; no training input or checkpoint is deleted.
