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

## Completed results

All twelve trajectories completed 128 registered attempts each. All 22 endpoints
(two starts, eight reused P45 controls, twelve new candidates) finished the frozen
raw and original-reference reviews. Integrity verification checked 243 bound
inputs, all endpoint and raw-review hashes, final report receipts and all twelve
checkpoint histories. JSON-normalized checkpoint histories exactly match their
update logs; all four zero-update endpoints exactly match their initial parameters.
There were **30 accepted updates across 1,536 attempts**. This is completion of a
bounded method pilot, not 128 effective updates per arm or convergence.

All changes below are relative to the corresponding frozen start. Negative error
changes are better. Music means cover 128 paired or identity examples; reference
violation counts cover 160 examples per role (128 music, 32 speech), at each arm's
registered ruler. Categories can overlap, and these are reused TRAIN clips.

| Seed | Method / ruler | Accepted / 128 | Music amplitude change | Music MR change | Music complex change | Music identity RMS change | Review violations: identity / low4k |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261012 | policy / 16 | 6 | -0.10491% | -0.06319% | +0.06374% | +4.9739% | 12 / 11 |
| 20261012 | policy / 20 | 2 | -0.01178% | -0.00661% | +0.00576% | +0.4068% | 14 / 11 |
| 20261012 | policy / 24 | 0 | +0.00000% | +0.00000% | +0.00000% | +0.0000% | 0 / 0 |
| 20261012 | projected / 16 | 6 | -0.10073% | -0.06161% | +0.06302% | +3.8370% | 12 / 11 |
| 20261012 | projected / 20 | 2 | -0.01539% | -0.00883% | +0.00957% | +0.4562% | 16 / 15 |
| 20261012 | projected / 24 | 0 | +0.00000% | +0.00000% | +0.00000% | +0.0000% | 0 / 0 |
| 20261013 | policy / 16 | 5 | -0.10688% | -0.06801% | +0.10265% | +8.9085% | 16 / 13 |
| 20261013 | policy / 20 | 2 | -0.00948% | -0.00591% | +0.00781% | +0.4449% | 14 / 14 |
| 20261013 | policy / 24 | 0 | +0.00000% | +0.00000% | +0.00000% | +0.0000% | 0 / 0 |
| 20261013 | projected / 16 | 5 | -0.11364% | -0.07304% | +0.11489% | +7.9039% | 15 / 13 |
| 20261013 | projected / 20 | 2 | -0.01609% | -0.01098% | +0.00360% | +0.4809% | 16 / 13 |
| 20261013 | projected / 24 | 0 | +0.00000% | +0.00000% | +0.00000% | +0.0000% | 0 / 0 |

The 16-bit policy-only arms reduce music amplitude error by 0.105% / 0.107% and
MR by 0.063% / 0.068%, compared with P45's 0.004% / 0.006% MR reduction. Correcting
the displacement policy therefore permitted more useful movement, but adding the
two reference MSE gradients did not increase acceptance: both methods accept
6 / 5 updates at 16 bits, 2 / 2 at 20 bits and none at 24 bits. Projection has mixed
spectral effects across seeds and rulers; it is not a demonstrated overall gain.
Against untreated compressed input, the 16-bit candidates reduce music MR by
1.797-1.916%; their frozen starts already reduce it by 1.736% / 1.845%. Most of
that total benefit is inherited and must not be credited to P46.
The stronger matched Adam music MR reductions of 0.776% / 0.463% retain their
previous fidelity tradeoffs and are not a qualified alternative.

At 16 bits, speech paired MR falls 0.046% / 0.060% for policy and 0.044% / 0.067%
for projection. At 20 bits, projected seed 20261013 instead raises speech MR by
0.0064%; this counterexample is retained. Music complex error increases on every
nonzero candidate. These small increases do not alone veto spectrum-first work,
but neither phase nor reconstructed quality is established by magnitude gains.

Music identity RMS error increases 4.974% / 8.908% for the 16-bit policy arms and
3.837% / 7.904% for projection. Corresponding absolute PCM RMS increases are
5.111e-7 / 1.440e-6 and 3.943e-7 / 1.278e-6. Relative percentages are not audible
severity estimates. All eight changed candidates violate their own registered
reference ruler on some separate review examples; unchanged 24-bit arms pass
without restoring anything new.

## Failure attribution and next decision

Post-hoc attribution uses the last tested radius of every attempt; overlapping
violations must not be summed as disjoint causes. Fit amplitude and MR improve
on all 128 final proposals in every arm. The dominant obstruction is preservation
and transfer, not failure to find any spectrum-descending direction:

- At 16 bits, policy low4k peak checks fail on 90 / 103 attempts; identity peak
  checks fail on 76 / 92. Projection changes these to 88 / 92 and 77 / 82,
  respectively, without increasing accepted updates.
- Guard auxiliary-budget checks also fail on 63 / 102 policy attempts and
  63 / 101 projected attempts. This retained surrogate needs attribution against
  original-reference quality before another long run; it is not itself a PCM
  restoration measure, and its failures do not justify deleting the guard.
- Group-mean MSE gradients constrain local averages, not individual peak errors,
  finite recomputed-history responses or different recordings. On the separate
  review, the 16-bit paired-low4k mean RMS error decreases for all four candidates,
  yet 11 / 13 examples violate the corresponding peak allowance in each method.
  An improving mean cannot certify individual preservation.

Retained examples include seed 20261012, review draw 8731, paired example 3:
projected low4k peak error rises from 0.150603934 to 0.150862412 PCM (increase
2.58478e-4). Seed 20261013, draw 6198, identity example 1: projected peak error
rises from 0.001702636 to 0.001932651 (increase 2.30014e-4). These absolute error
statistics refer to the original clean target; no clipping or post-hoc ruler
relaxation is applied. Neither peak statistics nor this review establish audibility.

Do not simply extend this trajectory, enlarge the model, or loosen rulers until
it passes. The next bounded diagnostic should isolate output-band coupling and
the auxiliary-budget surrogate, then test whether update directions aimed at
per-example preservation failures transfer better than pooled MSE. Any altered
output support must preserve crossover continuity and be tested against original
clean spectra before longer training. These are proposed diagnostics, not new
implemented or launched experiments, and do not establish an architecture limit.

The default weights remain unchanged. Rust DSP, simultaneous stereo, listening,
independent generalization and real-time scheduling failures remain unqualified.
Completed code verification: 700 tests passed (223 existing warnings), Ruff and
124 English documentation/link/index checks passed; code CI 40011938045 passed.
Local evidence adds `gate-attribution.json` and `completion-integrity.json` to the
frozen run without modifying its registrations, checkpoints or review reports.
