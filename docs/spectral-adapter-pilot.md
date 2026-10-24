# P48: isolated spectral adapter pilot

## Hypothesis and matched controls

[P47](output-coupling-audit.md) attributes most current low-band movement to
direct low-band coefficient changes. Reverting those changes preserves almost
all of P46's small magnitude-spectrum gain, but simply passing through the input
also loses useful low-band reconstruction. P48 therefore preserves the original
trained path and tests a separate learnable residual on frozen causal features.
This tests structural update isolation, rather than repeating whole-model
gradient projection or increasing backbone capacity.

Both P38 native small-model starts are retained. Each gets two zero-initialized
adapters: unrestricted output and high-band output. All four trajectories use
identical data order, objective, fresh Adam settings and 128 actual optimizer
steps. They finish before any final endpoint review. This is a bounded mechanism
pilot, not full convergence training or an architecture-capacity comparison.

The causal encoder, temporal layers, original decoder and generation budget are
frozen. Their feature computation runs once per physical channel and frame. The
new linear adapter maps the same 192 latent features to 258 real/imaginary
coordinates, using input-relative phase coordinates, the existing 1/64 tanh slope
and fixed input-only residual units. It does not multiply its output by a learned
budget. The original path and its useful corrections remain present.

Both variants have 480,809 total parameters, of which 49,794 belong to the adapter.
Their nominal parameter counts match; masked low-frequency output rows have zero
effective gradient in the high-band variant, so active capacities are not equal.
No new temporal state, second backbone inference, lookahead or buffering is added.
That design claim does not replace measured runtime behavior.

The high variant uses fixed physical-frequency support: zero at/below 4 kHz,
smoothstep `x*x*(3-2*x)` across 4-8 kHz, and one at/above 8 kHz. Boundaries come
from the existing analysis bands, not track-dependent thresholds. The first
derivative is zero at both transition endpoints. Raw OLA can still transfer some
energy across bands, so preserved coefficients alone do not certify preserved
PCM or crossover quality. The full variant uses support one everywhere.

## Objective and exposure

The objective is unchanged from the P38/P45 scoring implementation: equal paired
and identity means of `log1p(direct_spectrum) + log1p(multires_spectrum)`, plus the
existing 0.1 budget auxiliary. The direct loss includes magnitude, complex,
envelope and reference-following temporal terms; MR uses 256/1024/2048 analysis
windows on original aligned PCM. These training analysis windows do not add
inference lookahead. Targets remain original and unclipped.

Because the base and budget are frozen, the auxiliary is constant with respect
to adapter parameters and has no adapter gradient. It remains logged for common
score interpretation, and is not an update acceptance gate. This does not claim
that removing regularization generally improves quality. Each actual Adam step
uses learning rate 1e-4, betas 0.9/0.999, epsilon 1e-8 and gradient norm cap 5.
Fresh optimizer state differs from P45's historical Adam state; those old runs
are contextual evidence, not a clean optimizer or architecture comparison.

Reuse P45's 128 registered groups: four equally weighted fit batches per update,
plus a separate recording/source-disjoint TRAIN guard batch used only for
diagnostics. Each batch has four music clips, one speech clip and their identity
counterparts. Fit exposure per arm is 2,560 paired crops / 436 recordings /
1,122.537 seconds of unique MR support; guard exposure is 640 crops / 318
recordings / 290.503 seconds. The durations are not a joint union. Unlike P46,
there is no proposal veto; nonzero updates are still counted explicitly rather
than assuming every optimizer call changes parameters.

Historical source-time exclusions across codecs, rates and channel views remain.
This is reused TRAIN exposure, and the P44 review has already been used for
diagnosis. It cannot prove independent generalization. Genre/region gaps and
independent channel views retain their existing limitations. Warmup and guard
context stay outside supervision; the frozen history cannot receive gradients.

## Registered checks and endpoint barrier

Preflight checks all four seed/support branches with one real-data update:

- Exact zero-adapter output and budget parity against each frozen start.
- A nonzero adapter update with every original parameter unchanged.
- Nonzero ONNX parity over 320 sequential hops, both rates, mono/stereo, silence
  and reset; no export of a zero adapter as evidence of learned compatibility.
- Actual Rust frontend, ORT and DSP timing on 3,000 measured 48-kHz stereo hops
  after 100 warmup hops, with CPU and process RSS recorded.

The feasibility screen requires p99 service time below the 2.667-ms hop period.
All >=5-ms compute calls and service overruns remain reported and prevent runtime
qualification; a feasibility screen is not a waiver of the user's maximum.
No timing retry is selected to obtain a pass. Training, tests and builds do not
run concurrently with these timing measurements.

All four completed endpoint hashes are frozen before reviewing 32 common P44
batches. Compare each against its original start, the matched full adapter and
untreated input, reporting spectra, complex error, dynamics and calibrated PCM.
Apply all three existing 16/20/24-bit reference-preservation rulers descriptively,
without choosing a favorable ruler. No intermediate score selects checkpoints.

Each final adapter is exported and timed at both rates and mono/stereo, 3,000
measured hops per condition, plus 1,000 paced 48-kHz stereo hops. This measures
the actual diagnostic native compute chain, not an integrated audio callback or
physical device. The fixed 256-frame adapter buffer remains 5.333 ms at 48 kHz
and 5.805 ms at 44.1 kHz, separate from the <5-ms compute requirement. Historical
worker/scheduling failures remain open regardless of a passing new compute run.

Native quality uses eight evenly spaced review batches (0,4,...,28): 32 music
and eight speech crops, each paired and identity. Compare both starts, all four
trained endpoints and an identity graph through the real unchanged Rust DSP.
Keep original clean PCM, untreated input and all native output files; no limited
reference substitution. These mono channel views are not simultaneous stereo
music qualification, and this subset is not the full raw-review pool.

## Evidence and status

Source entry point: `training/scripts/run_adapter_pilot.py`; model:
`training/models/spectral_adapter.py`. Completed run:
`data/runs/spectral-adapter-p48-20261024-v2/`. Registration binds sources, data,
schedule, starts, prior evidence, timing inputs and a frozen native binary.
Atomic checkpoints retain adapter optimizer state and full histories.

The first run, `data/runs/spectral-adapter-p48-20261024/`, stopped during
preflight before formal training. Python export parity passed, but Rust rejected
the graph because the study name had replaced its exact experimental-ABI marker.
The corrected exporter preserves `p20-context-head-diagnostic` and records P48
in a separate `soundex.study` field. Native stderr is now retained before raising
an execution error. The first registration, smoke checkpoint, graph, failure log
and capability probe are preserved with a supersession receipt. This execution
fix changes no model arithmetic, objective, schedule, budget or acceptance rule;
the new source snapshot uses a separate v2 directory.

Eight focused tests pass: exact zero-start behavior, frozen low bins/base/state,
physical smooth support, causal block/channel/reset/silence behavior, nonzero ONNX
parity, an original-reference loss directional finite difference, and exact
optimizer-resume equivalence. The initial derivative-test fixture was too short
for two complete 2048-point analysis windows; it was corrected from 100 to 120
frames before training, without changing the loss or deployment frame size.

After the metadata correction, the complete Python suite passes 714 tests
(228 existing/export warnings); Ruff and the English documentation/link checker
pass. Remote [CI 40062490776](https://github.com/backrunner/soundex/actions/runs/40062490776)
also passes both Rust and export-contract jobs for code commit `8b3d9f5`.

All four v2 preflights pass. Across 12,000 measured 48-kHz stereo hops, per-run
p99 model-plus-DSP service time is 0.510-0.647 ms; the overall maximum is
1.161 ms, with zero service overruns and zero >=5-ms calls. Graphs are
1,971,777 bytes (full) or 1,972,748 bytes (high); native process peak RSS is
30.86-31.88 MiB. These are one-update preflight results, not final endpoint
or continuous-device qualification.

## Completed endpoint results

All four trajectories finish 128 nonzero updates, with every original tensor
exactly unchanged. The six endpoint hashes (two starts, four adapters) are fixed
before review. All 365 bound input/source hashes and completion receipts verify.
The full 32-batch raw recheck and all 560 native role replays finish. No production
weight, DSP path or artifact contract changes; none of these candidates qualifies.

The following percentages are error changes against the corresponding starting
model under the same processing path; negative means improvement. Raw music
contains 128 clips; native music uses the fixed 32-clip subset. These columns
must not be subtracted as if they described identical pools.

| Seed / support | Raw music MR | Raw music complex | Raw music MR dynamics | Native music MR | Native music HF PCM RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 12 / full | -0.21185% | +0.64055% | +0.15307% | -0.23767% | +0.06678% |
| 12 / high | -0.21411% | +0.67505% | +0.16287% | -0.24228% | +0.06791% |
| 13 / full | -0.12188% | +0.36132% | +0.08081% | -0.13632% | +0.03876% |
| 13 / high | -0.11974% | +0.35169% | +0.08036% | -0.13695% | +0.03859% |

High support improves raw MR on 99/128 music clips for each seed, but complex
error worsens on 117/128 and 121/128; MR dynamics worsens on 93/128 and 91/128.
Its direct amplitude error falls 0.46125% / 0.24013%. On speech, raw MR falls
0.28336% / 0.12688% while complex and dynamics errors still rise. This is a small
magnitude-spectrum improvement, not evidence that the missing complex detail
has been reconstructed or that the overall audio sounds better.

On the native music subset, high-support MR is 2.88324% / 2.71413% below untreated
input, but the starts already achieve 2.64737% / 2.58071%. Relative to untreated
input, native high-band PCM error remains 3.88885% / 2.49201% higher. The old
model is not treated as ground truth, and inherited gains are not attributed to
128 adapter updates. The identity-through-DSP control itself lowers music MR
1.20203% while increasing full PCM error 0.33004%; it does not restore content.

### Preservation and absolute errors

The structural isolation works locally. Across 160 paired review clips, the
full adapters fail the low4k reference ruler on 41/37 examples at 16 bits,
85/85 at 20 bits and 101/103 at 24 bits. High support reduces those counts to
0/0, 0/0 and 9/9. The remaining 24-bit violations concern low-band peaks, showing
why preserved coefficients cannot imply exact OLA-band preservation.

Identity preservation remains unresolved. At 16/20/24 bits, full-support
identity violations are 155/159/159 and 150/159/159; high-support violations
are 134/159/159 and 123/158/159. Each denominator is 160 examples. These are
the existing engineering rulers, not fitted audibility thresholds.

Mean raw music identity RMS changes from 1.0277e-5 / 1.6169e-5 at the starts to
4.1197e-5 / 4.2469e-5 with full support, or 1.7825e-5 / 2.3554e-5 with high
support. High support reduces the full adapter's error by about 57% / 45%, but
still increases the starting error 73.45% / 45.68%. Raw high-support speech
identity RMS is 4.1970e-6 / 4.7950e-6. In terms of `20*log10(mean per-clip RMS)`,
these high-support music means are about -95.0 / -92.6 dBFS and speech means
-107.5 / -106.4 dBFS; these are not pooled power levels or listening judgments.
For the high band alone, raw high-support identity means are 1.362e-5 / 1.346e-5
PCM RMS for music (about -97.3 / -97.4 dBFS) and 3.793e-6 / 4.393e-6 for speech
(about -108.4 / -107.1 dBFS). Large relative weak-band increases must be read
together with these absolute errors, not interpreted as audibility estimates.

Native high-support music identity RMS is about 6.7960e-4 / 6.7971e-4, and speech
about 1.0425e-6 / 1.1539e-6. The large music absolute error is mostly inherited
from DSP protection: identity DSP alone has mean music RMS 6.7308e-4. A retained
post-hoc calculation reproduces this with the unchanged smooth limiter's
0.855 knee / 0.95 ceiling: eight of 32 music references enter the knee; mean
limiter-only RMS is 6.7306e-4, and maximum per-clip residual between native
identity DSP and that calculation is 4.53e-8 RMS. All eight speech references
stay below the knee. This attributes the protection distortion; it does not
replace original-reference scores with a limited target or waive adapter errors.

### Common objective and first-step diagnosis

Reconstructing the exact common direct/MR objective from all stored per-example
components shows that all four endpoints worsen overall reconstruction risk.
For full support the increases are 0.001347 / 0.001356; for high support they
are 0.001182 / 0.001190. The frozen budget auxiliary is identical, so total-risk
deltas are identical. High-support paired improvements of 0.000521 / 0.000266
are outweighed by identity-risk increases of 0.001703 / 0.001456. This failure is
visible to the existing loss; it is not solely an external preservation veto.

A post-hoc replay of the first registered group uses its original four fit
batches and disjoint-source TRAIN guard, with retained preflight step-one
checkpoints. All four first updates increase the very same fit objective by
0.02723-0.02971 and the guard objective by 0.03064-0.03649. At step 128, the
same fit objective remains 0.00083-0.00102 above its starting value. These
are fixed-group diagnostics, not a full training-distribution convergence result.

A second reversible probe scales the saved first Adam displacement by fixed
factors 0, 0.001, 0.01, 0.1 and 1. No optimizer step, export or replacement
checkpoint is produced. For all four arms the original gradient dotted with the
full displacement is negative (-0.00244 to -0.00254), yet the finite full step
increases loss. At factor 0.001, fit loss falls 1.92e-6 to 2.02e-6; already at
0.01 it rises 2.40e-5 to 2.50e-5. This is direct evidence of first-step overshoot,
consistent with the tested local gradient, rather than evidence that no useful
adapter direction exists. The source-disjoint guard still rises 1.49e-7 to
2.09e-7 for seed 12 at factor 0.001, while seed 13 falls about 8e-7. Shrinking a
step therefore does not by itself establish transfer or a usable training recipe.
These factors are diagnostics, not selected learning rates; subsequent Adam
trajectories cannot be inferred by scaling the first displacement.

## Final runtime and limits

The host is an Apple M5 Max with 18 CPU cores and 64 GiB RAM, running macOS
27.0.1. The diagnostic native process uses single-thread ORT; it is not a
physical-device callback test. No training, builds or tests overlap final timing.

| Seed / support | Measured hops | Worst compute | Compute >=5 ms | Service over hop | Paced late hops | Worst paced lateness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 / full | 13,000 | 2.480 ms | 0 | 0 | 3 | 1.313 ms |
| 12 / high | 13,000 | 1.656 ms | 0 | 0 | 4 | 5.386 ms |
| 13 / full | 13,000 | 1.797 ms | 0 | 0 | 2 | 3.514 ms |
| 13 / high | 13,000 | 0.870 ms | 0 | 0 | 0 | 0 |

Across the 52,000 final hops, unpaced per-run p99 is 0.118-0.320 ms; paced stereo
p99 is 0.726-1.132 ms. Final native process peak RSS is 29.64-30.06 MiB.
Including the 2,000 unscored startup/warmup hops leaves the same 2.480-ms maximum
and zero >=5-ms calls across 54,000 final-run hops.
Process user+system CPU time divided by processed audio duration is roughly
3.33-6.05% of one core unpaced and 12.27-22.16% paced, including process startup;
these are approximate process accounting ratios, not isolated inference CPU
cycles or application-wide utilization. All timing artifacts are retained.

Compute fits this measured budget, but nine paced completion delays remain.
No fastest-seed selection or rerun erases them. The fixed 256-frame buffer still
adds 5.333 ms at 48 kHz / 5.805 ms at 44.1 kHz separately from compute; reliable
continuous output and real device behavior remain unqualified.

## Decision and evidence

Retain the small shared-feature architecture as an isolation control, not a
qualified replacement. Do not extend the unchanged Adam recipe or scale capacity
based on this pilot. Calibrate the adapter's parameter-to-output update scale
and finite-step descent using training draws before another matched run; use
fixed disjoint diagnostics, rather than tuning on this review pool. Retain
spectrum, complex/dynamics, original PCM
and identity checks rather than removing phase or relaxing a known counterexample.
Earlier phase-only, risk-reweighting and whole-model guard failures remain in
the [failure registry](failed-experiments.md). This 128-step pilot neither tests
full convergence nor establishes a frozen-feature or architecture capacity ceiling.

Local evidence includes `registration.json`, `complete.json`,
`endpoint-integrity.json`, `review-registration.json`, `assessment.json`,
`descriptive-analysis.json`, `common-objective.json`, `first-step-diagnosis.json`,
`update-scale-diagnosis.json`,
`identity-dsp-attribution.json`, `cost-summary.json`, and all raw/native role
reports and PCM. `analysis-integrity.json` binds supplementary reports/scripts;
all 2,240 native artifacts and raw-review hashes verify. Descriptive post-hoc
analyses do not change the frozen endpoint
review or constitute independent testing. Listening, simultaneous stereo music,
genre/region coverage and unseen-recording generalization remain open.
