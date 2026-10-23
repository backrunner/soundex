# P46: original-reference preservation pilot

## Changed hypothesis

[P45](anchored-spectrum-pilot.md) completed all registered attempts but mostly
stalled. Its below-4-kHz displacement cap rejected many corrections that reduced
error against the original PCM. Increasing that policy's budget or repeating more
steps would not test the intended distinction between restoration and damage.

P46 uses the same two frozen 431,015-parameter P38 starts, P45 schedule and
128-attempt budget. It has **two methods at all three 16/20/24-bit sensitivity
rulers**, giving twelve new trajectories. No ruler is chosen after seeing which
one passes. These are engineering experiments, not perceptual thresholds or full
convergence/capacity tests. P45 Adam and displacement-policy endpoints are reused
as matched controls, with endpoint/review hashes verified and each seed's first
Adam update reproduced exactly before training.

This is a post-hoc mechanism study on reused TRAIN draws and P44 review clips.
It does not create an independent test set or establish generalization.

## Original-reference constraints

Let `e = current_output - original_clean_PCM` and `e0 = frozen_start_output -
original_clean_PCM`. For every example in the four fit batches and separate TRAIN
guard batch, compare the following statistics of `e` with the same statistic of
`e0`:

| Scope | Statistic | Maximum increase |
| --- | --- | ---: |
| Clean identity, full band | RMS and absolute peak error | `q = 2^-bits` |
| Paired input, below 4 kHz | RMS and absolute peak error | `q` |
| Both scopes | Error first-difference RMS and hop-boundary difference RMS | `2q` |

The initial model stays fixed throughout the trajectory. These are cumulative
original-reference error allowances on sampled examples, not rolling per-step
allowances or population guarantees. Output displacement remains recorded but
does not decide acceptance. A large correction that approaches the original
reference can pass; a new peak error cannot hide behind a low average RMS. Error
differences subtract the reference's real temporal changes, so matching an actual
transient gives zero temporal error rather than a smoothing incentive.

The `2q` ruler follows the triangle bound on differences of two sample errors
bounded by `q`. It does not assert that arbitrary model corrections have bounded
pointwise perturbations or that this ruler is perceptually adequate. Existing
baseline error is still reported and is not inherited as release qualification.

Below-4-kHz Fourier projection is offline measurement. It is not a deployed
filter or additional lookahead, and not every codec necessarily preserves that
entire band. Original references remain unmodified and unclipped.

## Policy and direction controls

- **Reference policy:** retain P43's amplitude/MR primary direction and six
  first-order guards; use the new finite-step reference policy.
- **Reference projected:** same policy and primary direction, adding two
  first-order gradients: mean original-reference full-band identity MSE and mean
  original-reference paired below-4-kHz MSE. The existing certified active-set
  solver handles all eight guards together.

The second method changes direction construction, not a scalar spectrum-loss
weight. Group-mean MSE gradients constrain mean squared errors locally; they do
not certify individual peaks, temporal checks or finite-step effects. Every
candidate still undergoes independent per-example finite-step checks with actual
recomputed warmup. No-gradient history remains detached in gradient estimates.

Both methods retain P45's radii 0.001 then 0.0003, its requirement that both fit
amplitude and MR improve, and its fixed-initial 0.1% relative diagnostic checks
for complex error, MR dynamics, budget auxiliary and guard spectra. Rejecting all
radii restores parameters exactly and records a no-op. P46 versus P45 changes
reference-based acceptance **and** adds reference temporal checks; only the two
P46 methods isolate the added MSE-gradient constraints.

## Exposure, preflight and endpoint review

Each arm attempts the same four fit batches and one source/recording-disjoint
guard batch per update. Each batch contains four music clips, one speech clip and
their identity counterparts. Across attempts recordings may recur. P45's full
source-time exclusion of P44 and prior diagnostic/development clips is retained
across channel views, rates and codecs. Recorded attempted coverage is 2,560 fit
crops / 436 recordings / 1,122.537 seconds of unique MR support and 640 guard
crops / 318 recordings / 290.503 seconds. The role durations are not a joint union;
no-op attempts are not effective parameter updates.

Preflight runs the exact first Adam update and all six candidate branches for
each seed, without saving updated models or choosing a ruler. Known-answer tests
cover beneficial corrections, harmful spikes, real transients, fixed cumulative
anchors, zero/weak-energy gradients, finite differences, original-target detach
and column-shaped sample rates. These tests are numerical/functional evidence,
not listening or quality qualification.

All twelve trajectories must finish before the frozen endpoint barrier. Reports
retain all rejected proposals, all precision arms and both seeds. The common P44
raw review compares spectrum/complex/dynamics and calibrated PCM by music/speech
and paired/identity. It includes start, matched Adam, old P45, method-to-method
and untreated-compressed controls. An additional per-example review measures
full-band identity and below-4-kHz original-reference/temporal errors against the
initial anchor at all three rulers. A passing training batch does not certify
this separate reused-TRAIN review.

Actual Rust/DSP, simultaneous stereo, listening and real-time qualification remain
separate gates. No default weight replacement follows from this pilot. No model
architecture or runtime processing change is included.

Entry point: `training/scripts/run_reference_pilot.py`; local evidence:
`data/runs/reference-spectrum-p46-20261024/`. The run is hash-bound, resumable and
retains executed source snapshots. `complete.json` is written only after the
raw, method-contrast, untreated and reference-error reviews all finish.

## Status

All twelve candidate preflight branches completed, and both original Adam first
updates reproduced exactly. Both seeds' 16-bit policy-only and projected first
steps passed; tighter first-step policies rejected their proposals. This is
execution/acceptance evidence, not convergence or quality, and no arm or budget
was changed based on it.

The fixed training run is now active with checkpointed updates. Completed local
verification: **700 tests passed**, 223 existing warnings; Ruff checks and all
124 English documentation/link/index checks pass. The run records all first-step
preflight results, source hashes and logs. All twelve complete endpoints and their
frozen reviews are still required before interpreting model quality.
