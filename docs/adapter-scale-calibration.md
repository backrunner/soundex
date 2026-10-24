# P49: adapter update-scale calibration

## Question and prior evidence

[P48](spectral-adapter-pilot.md) finds that all four first Adam steps increase
their own fit loss despite negative directional derivatives. Reducing the saved
displacement to 0.001 makes that one fit group descend, but does not consistently
help its separate guard. P49 asks whether finite-step calibration works across
different draws and whether fit descent transfers to other recordings.

This differs from [P17](consistency-learning-rate-study.md), which changed the
learning rate of a historical whole-model consistency continuation. P49 uses
the frozen-feature adapter, unchanged original-reference direct/MR objective,
fresh optimizer state and explicit same-direction finite-step controls.
It also differs from P45/P46: no PCM or guard-score veto selects the step, and
the budget auxiliary is frozen and constant. All preservation rulers remain
visible in review. This is not a new loss, larger architecture or DSP change.

## Fixed protocol

- Two original P38 starts, each with full/high support as in P48. Reset the
  adapter to zero and Adam to fresh state for every group; there is no cumulative
  training trajectory or claim about convergence.
- Use P45 groups 1,9,...121: sixteen score-independent choices excluding the
  already examined group zero. Each has four fit batches and one guard, with four
  music clips and one speech clip plus their clean identity counterparts per batch.
- Preserve original source-time exclusions. Verify fit/guard disjointness by
  recording and source checksum within each group. Sources can recur between
  groups and were used in historical training: these are reused-TRAIN checks,
  not independent test data. Record actual codec/channel/source-time coverage.
- Produce the same Adam displacement as P48: learning rate 1e-4, betas
  0.9/0.999, epsilon 1e-8, four equally weighted gradients and norm clipping at 5.
  Compute the directional derivative using the original unclipped gradient.
- Compare the full displacement, fixed factor 0.001, and fit-only backtracking.
  The fixed factor is a historical diagnostic control, not a selected global
  learning rate. Later Adam trajectories do not follow from scaling a first step.
- Backtracking tries ten factors `0.25**i`, starting at one. Require a negative
  derivative and actual fit decrease at least
  `max(1e-4 * factor * -derivative, 8 * float32_epsilon * abs(fit_loss))`.
  The latter is a numerical floor inherited from P42, not an audibility tolerance.
  Choose the first passing factor, or retain an explicit no-op. Never select
  using guard data, phase/PCM counters or the P44 review pool.
- Verify group-zero execution against all four retained P48 preflight
  displacements and fit objectives exactly. This execution control is excluded
  from the sixteen-group results.
- Freeze all 64 displacement hashes and selected factors before guard scoring.
  Score start/full/fixed/backtracked paths against the same original references;
  report the unchanged total, magnitude/MR, complex, temporal and identity risks,
  per-example raw spectra/absolute PCM, and all 16/20/24-bit reference rulers.
  Retain strict and beyond-numerical changes rather than calling roundoff a gain.

The first-stage fit reports, proposal tensors and all attempted backtracking
points remain local. Every proposal is restored exactly after probing. Source
snapshots, input hashes, registration, per-proposal receipts, the proposal barrier
and final review receipts are retained. Failed or incomplete runs are preserved.

## Interpretation and scope

Fit descent is necessary evidence about optimization, not sufficient evidence
about restoration. A useful continuation needs spectral progress on separated
recordings together with phase/temporal/identity and original-PCM evidence.
No-ops, improved identity alone, or a lower scalar objective do not qualify.
This audit does not choose a model from guard scores or automatically extend
the unchanged recipe merely because some first steps descend.

All audio targets remain original and unclipped. Model supervision remains
before DSP. No production weights, deployment ABI, state, buffering or DSP are
changed. P48's actual Rust timing and native quality evidence remains historical;
this calibration produces no newly qualified native model. Continuous scheduling,
simultaneous stereo music, listening, genre/region gaps and unseen-recording
generalization remain unresolved.

## Actual exposure

The shared sixteen-group schedule contains 320 paired fit crops from 214
recordings (256 music / 64 speech), with 146.890 seconds of unique MR-supervised
source support. Guard exposure is 80 paired crops from 72 recordings (64 music /
16 speech), with 37.018 seconds of unique MR support. Clean identity counterparts
are also evaluated. These are separate within-role unions, not additive global
durations; repeating them across seeds/supports does not multiply unique audio.

All nine registered codec recipes occur in both roles. Fit channel views are
left/mid/mono/right/side = 52/114/69/41/44; guard views are 14/27/17/13/9.
These are separate mono views, not simultaneous stereo music. All within-group
fit/guard recording and source-checksum intersections are empty, but 37
recordings recur across the global role pools. Every probe resets its model and
optimizer; this separation must not be reused as a globally held-out split for
cumulative training.

Fit music metadata includes 10 named genre labels plus 17 unlabeled crops;
guard has eight named labels plus six unlabeled crops, with no country or hiphop.
For China/Japan/Korea/Europe/North America/unknown metadata labels, fit crop
counts are 13/40/50/35/88/30 and guard counts 6/6/13/6/27/6. These are crop
counts and existing source labels, not verified national-style or balanced
genre coverage. The complete per-codec/genre/channel counts and source intervals
are retained in the registration.

## Completed results

All 64 independent first-step probes and frozen guard reviews are complete.
Every derivative is negative, but every full Adam step increases both fit and
guard total. Both scaled controls decrease all 64 fit totals beyond the registered
numerical floor. Backtracking selects 1/256 in 60 probes and 1/1024 in four;
there are no no-ops. A descending infinitesimal direction therefore does not
justify the original finite step.

Counts below use the existing eight-float32-epsilon numerical floor. "Near"
means neither an increase nor a decrease beyond that floor. Changes are in the
unchanged common objective, with negative values indicating improvement.

| Update | Fit better / worse / near | Guard better / worse / near | Mean fit change | Mean guard change |
| --- | --- | --- | --- | --- |
| Full Adam | 0 / 64 / 0 | 0 / 64 / 0 | +0.036901 | +0.034706 |
| Fixed 0.001 displacement | 64 / 0 / 0 | 35 / 16 / 13 | -2.7752e-6 | -7.9221e-7 |
| Fit-only backtracking | 64 / 0 / 0 | 18 / 42 / 4 | -3.6628e-6 | +3.3552e-6 |

Fit-only backtracking improves fit more, but its mean guard total worsens in
all four seed/support arms. Choosing the largest fit-admissible step is not a
reliable transfer rule. Fixed scaling is more conservative and improves mean
guard total, but does not improve every component or recording.

| Guard component | Fixed: better / worse / near | Backtracking: better / worse / near |
| --- | --- | --- |
| Paired amplitude | 58 / 6 / 0 | 58 / 6 / 0 |
| Paired multi-resolution magnitude | 40 / 7 / 17 | 49 / 8 / 7 |
| Paired complex spectrum | 19 / 38 / 7 | 21 / 42 / 1 |
| Paired MR dynamics | 21 / 34 / 9 | 23 / 40 / 1 |
| Clean identity MR | 5 / 59 / 0 | 4 / 60 / 0 |

Using strict signs rather than the numerical floor, amplitude and MR improve
together in 48/64 fixed probes and 46/64 backtracked probes; all five components
improve together in only 2/64 and 1/64. These are descriptive counts, not a new
all-components veto. Small waveform or complex-error increases alone cannot
reject a spectrum-first candidate, and the absolute errors below matter.

### Original-reference preservation and absolute error

Every fixed and backtracked probe passes the 16-bit identity and paired-low4k
rulers. At 20/24 bits, violation counts are as follows; each cell shows
seed 12 / seed 13, with 80 examples per seed and role. Empty counters are zero.

| Support and update | 20-bit identity | 20-bit paired low4k | 24-bit identity | 24-bit paired low4k |
| --- | --- | --- | --- | --- |
| Full, fixed | 2 / 2 | 0 / 0 | 37 / 45 | 24 / 18 |
| High, fixed | 0 / 0 | 0 / 0 | 22 / 21 | 0 / 0 |
| Full, backtracked | 25 / 31 | 4 / 0 | 63 / 72 | 33 / 29 |
| High, backtracked | 17 / 15 | 0 / 0 | 53 / 60 | 0 / 0 |

Thus fixed small updates with high support preserve the 16/20-bit rulers on
this review pool. All 24-bit identity counterexamples remain recorded. These
existing engineering rulers are not listening thresholds or release approval.

For fixed high-support probes, mean music identity full-PCM RMS error is
8.9968e-6 / 1.3267e-5, changing by -1.8555e-9 / +2.4088e-9 from the starts.
The corresponding levels, calculated as `20*log10(mean clip RMS)`, are
-100.918 / -97.545 dBFS. Mean speech identity full-PCM RMS is
5.0253e-7 / 2.6757e-7 (-125.977 / -131.451 dBFS).

The same music identity HF >=8 kHz RMS errors are 3.9083e-6 / 4.8518e-6
(-108.160 / -106.282 dBFS), and speech errors are 1.6040e-7 / 6.4227e-8
(-135.896 / -143.846 dBFS). Speech identity MR rises 20.42% / 69.00%, despite
small absolute PCM changes. Relative weak-band errors must not be interpreted
as evidence of comparable audible damage. These levels describe calibrated
PCM error, not unnormalized FFT power or the RMS pooled across recordings.

Fixed high-support mean paired music MR decreases by only 0.000458% / 0.000210%;
speech MR decreases by 0.000191% / 0.000099%. Music complex error increases
0.000351% / 0.000202%. These tiny first-step changes are neither meaningful
whole-track quality evidence nor a reason to discard the direction solely on
waveform grounds. They cannot be multiplied by a planned number of steps to
predict training gain. Each group has its own reset model: these aggregates
describe local transfer across different one-step models, not one trained weight.

## Decision and remaining experiment

Update overshoot is reproducible across the registered draws, and scaling fixes
same-batch descent. It does not by itself fix transfer, clean-input drift or
temporal/complex tradeoffs. The largest fit-admissible step is not adopted.
The fixed small high-support control warrants a bounded continuous-update
pilot to determine whether spectral gains accumulate while Adam moments and
identity errors evolve. That pilot is not launched by this calibration.

Before such a pilot, register equal budgets and fixed endpoints, separate its
fit and review pools globally by recording/source, and preserve the original
objective and all rulers. Freeze all endpoints before a common review. Report
actual supervised duration, accumulated raw/reference errors and any subsequent
Rust output separately. Neither long training, increased capacity nor a new
loss is justified solely by these first-step results. Generalization, listening
quality and P48's paced scheduling failures remain open; no weight is promoted.

## Evidence, execution recovery and checks

Entry point: `training/scripts/audit_adapter_scale.py`; finite-step helper:
`training/evaluation/adapter_scale.py`; summary:
`training/scripts/summarize_adapter_scale.py`.

Calibration artifacts: `data/runs/adapter-scale-p49-20261024/`.
All four group-zero P48 displacements and fit objectives reproduce exactly.
After all 64 proposals were frozen, review failed before any guard score because
the receipt path used `with_suffix("receipt.json")` without its required dot.
The original source snapshot, proposals, log and execution-failure record remain.

The correction adds the valid suffix and a real guard-review/resume regression
test. `training/scripts/resume_adapter_scale_review.py` performs review-only
recovery from the frozen proposals: no gradients, reselection or changed model
states. It verifies the original registration against retained source snapshots,
then binds the corrected review implementation and frozen inputs in a separate
registration. Completed evidence is in
`data/runs/adapter-scale-p49-review-20261024-v2/summary.json`.
All 759 bound inputs, 64 proposal hashes and 64 review receipts pass integrity
checks; rerunning the completed review verifies those receipts without scoring
again. The initial failure is not overwritten or presented as a completed run.

Seven focused calibration tests and the full 721-test Python suite pass after
the repair. Ruff check/format checks pass, and GitHub CI passes for both the
calibration implementation and recovery fix. Raw audio, proposals and reports
remain ignored local artifacts; only code and English documentation are published.
