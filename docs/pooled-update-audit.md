# P42: pooled gradients and finite-step protection

## Question and frozen protocol

[P41](guarded-transfer-audit.md) improved local tradeoffs but did not establish
stable transfer. P42 tests whether estimating the direction over more recordings
and checking actual step responses addresses those two specific shortcomings.
This is a reversible diagnostic on the same two frozen 431,015-parameter P38
controls, not a training trajectory or an architecture-capacity comparison.

Each seed has 16 cases. Each case has six mutually track-ID/source-checksum-disjoint
training batches: four to estimate gradients, one to check the step, and one for
transfer scoring only. Every batch has four music crops and one speech crop plus
their clean identities. New schedule anchors lie halfway between 16 uniform
partitions; all P41 draw indices are excluded. Selection depends only on schedule
geometry and recording identity, never on losses. These recordings were used in
original training, and different cases can reuse recordings. This is not an
independent test or a new full-library exposure claim.

Three directions share the same evaluation material:

1. `single_guarded`: the P41 constrained direction from the first fit batch.
2. `pooled_bisector`: average the four amplitude gradients and four complex
   gradients separately, then form their normalized bisector.
3. `pooled_guarded`: project the pooled bisector onto the eight pooled metric
   non-ascent constraints, using the unchanged P41 solver.

Pooling occurs **before** normalization. It retains the exact local log1p carriers
and original loss weights; it does not redefine the clean target or make the model
learn a DSP output. The guards constrain batch means, not each individual recording.
The single-batch direction is also evaluated against the same four-batch fit mean,
so its finite-step screen is deliberately stricter than its own direction source.
Gradient computation cost is not matched across directions; no training-efficiency
or latency conclusion follows from this comparison.

## Step selection and no-op accounting

All directions are evaluated at the preregistered L2 radii `0.01`, `0.003`, `0.001`
and `0.0003`, in descending order. Recompute actual warmup for every response;
gradients retain the established detached-warmup training graph. Restore the exact
frozen parameters after each case. No optimizer state or checkpoint is updated.

Two fixed policies select the first passing radius. Neither reads transfer scores:

- **Strict:** pooled-fit amplitude and MR must both improve beyond numerical
  allowance; all eight pooled-fit metrics and five separate-guard metrics must
  not increase beyond that allowance.
- **0.1% relative budget:** the same primary progress requirement, with a maximum
  0.1% relative increase in each protected metric, plus numerical allowance.

The numerical allowance is eight float32 epsilons times each metric's baseline
scale, floored at `1e-12` only for zero-scale arithmetic. The 0.1% policy is a
preregistered sensitivity control, **not a perceptual threshold or a production
acceptance criterion**. No tolerance is tuned to a named recording or a counterexample.
Actual changes, including changes permitted by either policy, remain reported.
If all radii fail, select an explicit no-op. No-op stability is not restoration
progress. Both accepted-only and all-case outcomes are reported to expose selection
bias and prevent many rejected steps from making a method appear successful.

## Calibrated raw PCM errors

Alongside unchanged direct/MR spectrum metrics, score raw overlap-add PCM against
the original clean reference, before deployed DSP. Report full-band and >=8 kHz
error RMS using an unwindowed one-sided Parseval transform, with DC and Nyquist
counted once. Per-example dBFS is `20*log10(RMS)`; exact zero is stored as null
representing negative infinity. Group summaries are means of per-example RMS,
not the RMS of a mean waveform. Error RMS and changes in error RMS are distinct
from the waveform change caused by an update.

The common MR supervision region excludes history/startup and final OLA support.
After removing the input pad, physical-source support is `[88, 254)` hops; direct
STFT analysis spans `[88, 256)` hops. Coverage unions source-time intervals across
codec/channel variants within each role and does not double-count identity copies.
These diagnostics do not test synchronized stereo, actual Rust postprocessing,
real-device continuity or listening. Absolute error levels cannot alone establish
audibility, restored spectrum shape, or release qualification.

## Evidence and verification

The fresh registration and all case results are retained under
`data/runs/pooled-updates-p42-20261023/`. Entry points are
`training/scripts/audit_pooled_updates.py` and
`training/scripts/summarize_pooled_updates.py`. Source snapshots, input/checkpoint
hashes, sampled recordings/intervals, coverage and every rejected radius are retained.
No raw training audio or experimental weights are committed.

## Completed results

All 32 cases completed on 2026-10-23. The following counts are improvements on the
transfer batch, among 32 cases. Strict simultaneous improvement is descriptive,
not a requirement that every stochastic update benefit every recording.

| Direction | L2 radius | Amplitude | Complex | MR | Dynamics | Identity MR | All five |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single guarded | 0.01 | 16 | 17 | 15 | 19 | 11 | 2 |
| Single guarded | 0.003 | 16 | 17 | 16 | 19 | 11 | 2 |
| Single guarded | 0.001 | 16 | 17 | 16 | 19 | 11 | 2 |
| Single guarded | 0.0003 | 16 | 17 | 16 | 19 | 11 | 2 |
| Pooled bisector | 0.01 | 16 | 15 | 17 | 20 | 14 | 2 |
| Pooled bisector | 0.003 | 17 | 14 | 17 | 20 | 14 | 2 |
| Pooled bisector | 0.001 | 17 | 14 | 17 | 20 | 14 | 2 |
| Pooled bisector | 0.0003 | 17 | 14 | 17 | 20 | 14 | 2 |
| Pooled guarded | 0.01 | 12 | 21 | 13 | 20 | 17 | 2 |
| Pooled guarded | 0.003 | 13 | 20 | 14 | 20 | 17 | 2 |
| Pooled guarded | 0.001 | 13 | 20 | 14 | 20 | 17 | 2 |
| Pooled guarded | 0.0003 | 13 | 20 | 14 | 20 | 17 | 2 |

The strict policy accepts zero single-guarded updates and only two updates for
each pooled direction; all accepted strict updates come from seed 20261013.
For pooled guarded, **30/32 no-ops are not successes**. The two accepted steps
improve transfer amplitude on one case and MR on one; neither improves all five.

| Direction, 0.1% policy | Accepted / 32 | No-op | Transfer amplitude improves | Transfer MR improves | All five improve |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single guarded | 5 | 27 | 3/5 | 3/5 | 2/5 |
| Pooled bisector | 13 | 19 | 4/13 | 5/13 | 0/13 |
| Pooled guarded | 17 | 15 | 4/17 | 5/17 | 0/17 |

Pooled guarded accepts 9/16 and 8/16 by seed. Among those accepted updates,
transfer amplitude has median changes **+0.01543% / +0.02326%**, and MR
**+0.00629% / +0.01399%**. Complex error improves on 12/17, dynamics on 12/17 and
identity MR on 11/17. Group medians that include no-ops are often zero; the report
separates them from accepted-only statistics rather than interpreting zero as gain.

At radius 0.001, pooled guarded improves both fit spectral metrics on 29/32 cases,
yet the separate step-guard amplitude check rejects 22/32. That amplitude rejection
count remains 22/32 at radius 0.0003. Merely shrinking a step reduces the size of a
regression without reliably changing its direction. This is not only a tiny
identity-error percentage blocking otherwise consistent spectral progress.

## PCM evidence and retained counterexample

For the 17 budget-accepted pooled-guarded updates, mean paired >=8 kHz PCM error
RMS improves on **14/17**, while amplitude error improves on **4/17**. This is a
measured conflict between waveform/complex agreement and magnitude-spectrum
agreement; better high-band PCM RMS alone does not meet the spectrum-first goal.
It does not imply that spectral energy equality guarantees waveform recovery.

Across all 32 transfer cases, median baseline identity full-band error RMS is
6.467e-6 / 6.913e-6 by seed. The largest *increase in mean identity error RMS*
among the budget-selected outcomes is 1.019e-7 / 7.912e-7. These small absolute
changes contextualize relative identity metrics without proving audibility or
justifying a release threshold.

Retained example: seed 20261013, fit anchor 5566, transfer draw 2869, accepted
radius 0.01. Transfer amplitude increases by 0.000351667 and MR by 0.000199318,
while complex error decreases by 0.000172317. Mean paired high-band error RMS
changes from 0.003169325 to 0.003170173, an increase of 8.486e-7; their calibrated
levels are −49.98067 and −49.97834 dBFS. These are levels of the mean per-example
error RMS, not a peak, per-track maximum, or audible-artifact claim. All other
cases, both seeds and rejected radii remain available.

## Actual exposure and next decision

| Seed / role | Paired crops | Distinct recordings | Unique MR source support |
| --- | ---: | ---: | ---: |
| 20261012 / fit | 320 | 188 | 121.837 s |
| 20261012 / step guard | 80 | 76 | 37.292 s |
| 20261012 / transfer | 80 | 75 | 36.827 s |
| 20261013 / fit | 320 | 192 | 112.015 s |
| 20261013 / step guard | 80 | 77 | 37.058 s |
| 20261013 / transfer | 80 | 72 | 36.591 s |

All roles cover the existing codec and channel views and multiple music regions.
Coverage remains sparse and uneven: no funk, no country in seed-12 transfer, and
no blues in seed-13 step-guard/transfer. Exact strata are in `summary.json`.
Do not sum these durations as globally independent material or count identities
as extra original recordings. No new material was acquired or synthesized.

Status: **diagnostic / not qualified**. Pooling and finite-step checks do not make
the amplitude/complex bisector a consistent spectrum-restoration update. This is
not evidence that every pooled optimizer fails or that the model lacks capacity.
The normalized bisector deliberately gives both directions equal geometric status;
it is not a faithful implementation of a spectrum-first priority hierarchy.

The follow-up tests a changed direction: let amplitude plus MR gradients drive
progress, protect dynamics/identity/budget, and keep complex error in the unchanged
finite-step assessment. The [P43 control](spectral-priority-update-audit.md) reuses
these training roles explicitly as a post-hoc mechanism check, not new validation.
No full training, deployed DSP change, runtime claim or promotion follows from P42.

Local full-suite verification passed 676 tests with the documented process-local
FFmpeg path; existing ONNX/MPS warnings remain. Executed source snapshots are
retained. After completion, the runner's list flattening was lint-corrected and
its direction builder made injectable for the separate follow-up; the original
P42 execution and hashes are not rewritten.
