# P74: phase observations inside the small temporal backbone

Registered 2026-11-10 after [P73](capacity-tradeoff-p73.md); all four full training
trajectories and fixed reviews are complete as verified on 2026-11-12. The phase
observation fails the registered two-seed matched-control criterion. Both arms
remain experimental: legacy-relative spectral progress is distinct from this
branch's unsuccessful incremental effect. Trained compute stays below 5 ms, but
four paced completions are late. Default weights remain unchanged. Earlier launch
and progress sections below are historical; the completion review supersedes them.

## Changed mechanism and matched control

P68/P69 supplied phase relations to a final rotation head, leaving the shared
magnitude backbone unable to observe them. P72 widened that magnitude backbone.
P73 found that magnitude recovery amplifies remaining phase mismatch. P74 instead
injects phase relations into band tokens **before** their temporal processing,
so the same history can inform magnitude and complex residual predictions.

Both arms retain token width 24, eight frequency bands, six temporal dilations
(1, 2, 4, 8, 16, 32), readout width 96, the original decoder/budget/adapter and
the existing adjacent/native-support phase head. Both have **216,802 parameters**;
the new projections contribute 12,384 parameters versus P72 small's 204,418.
This is an observation-location comparison, not another twofold capacity study.

The phase arm forms a carrier from input magnitude, using the existing digital
silence convention. Its complex carrier products encode real/imaginary relations
to the preceding frame and preceding frequency bin. Products are normalized by
`1 + carrier * other_carrier`; a smooth numerical norm protects zero amplitude.
They do not unwrap angles, divide by weak magnitudes or use reference information.
Four local observations per bin are projected into each band's 24-dimensional
token before its mixing/TCN stack.

The magnitude control has the same projections, state dimensions and random
initialization. It uses magnitude products duplicated across the four inputs,
providing no new phase information. The two arms have identical tensors at
initialization; zero projections yield the same initial function. The control
tests information rather than merely additional parameter count. Different input
conditioning can still affect optimization; do not infer an architecture ceiling.

Each physical channel has 12,612 state floats, including 258 new previous-carrier
values. There is no future context, added lookahead or output buffering. The
existing 80-hop enhancement warmup and detached 89-hop training burn remain;
warmup audio is not supervised. Fixed buffering remains 256 frames, or 5.333 ms at
48 kHz and 5.805 ms at 44.1 kHz, separate from computation.

## Implementation and derivative validation

`models/phase_temporal.py` defines the two arms. Small optional hooks in the
existing encoder/adapter retain historical default arithmetic. Export signatures
and metadata include the new observation kind, preventing reuse of a graph with
identical weights but different observation arithmetic.

Twenty-three distinct focused tests pass across the new observation, previous
band/native, and old/new Adam-continuation tests. They cover:

- Equal parameters/tensors/initial function, correct state sizes and chunking.
- Exact exclusion of future and other-channel derivatives, with established
  float32 chunk tolerances rather than new bitwise arithmetic assumptions.
- Past phase reaching later magnitude only in the candidate, common-rotation
  invariance of relations and sensitivity to relative phase changes.
- Digital silence, weak input, finite gradients and central finite differences.
- Genuine Adam with persistent moments, exact checkpoint continuation, changed
  observation projections and final-frame repeated-phase diagnostics.

Ruff passes. An initial lint check flagged two unused fixture bindings; these
were renamed before training registration without changing model arithmetic.
The feasibility source snapshot retains the original test source. No registered
execution or quality failure is hidden by this lint correction.

## Untrained native and timing screen

Evidence: `data/runs/phase-temporal-feasibility-p74-20261110/`.
All nonzero stress fixtures, exports, runtime binary, source snapshot, native
traces, timing orders and receipts are retained. No optimizer update occurs here.

Both arms pass sequential ONNX state/reset/silence checks and actual Rust
44.1/48 kHz mono/stereo correspondence. Maximum native complex prediction errors
are 1.059e-5 (magnitude control) and 5.801e-6 (phase); silent-channel peak is zero.
ONNX artifacts are 1,007,154 and 1,012,304 bytes respectively. These remain
experimental diagnostic graphs, not published production weights.

The schedule was fixed before timing: two opposite arm orders, each with both
rates/channels unpaced and 48 kHz stereo paced. Every case excludes its first
100 warmup hops; unpaced cases score 3,000 and paced cases 1,000. All **52,000**
measured hops are retained, with no retry, >=5 ms call, hop-service overrun or
paced late completion.

| Fixture | Maximum frontend + ORT + DSP | Paced mean, two orders | Paced process peak RSS |
| --- | ---: | ---: | ---: |
| Magnitude control | 1.905166 ms | 0.827956 / 0.736486 ms | 32,292,864 / 32,342,016 bytes |
| Phase candidate | 1.730666 ms | 0.966256 / 0.569690 ms | 32,325,632 / 32,391,168 bytes |

Paced p99 spans 1.378–1.533 ms for the phase candidate; process CPU including
startup/warmup is 1.05 / 0.66 seconds over
2.98 seconds wall time. Host load and order differences remain in every raw
report; do not interpret the smaller maximum as proof the phase arm is faster.
This is the direct replay's feasibility screen, not production callback/worker
or physical-device qualification. P72's late completion remains a recorded failure.

## Full training protocol

Evidence root: `data/runs/phase-temporal-training-p74-20261110/`.
Registration binds 95,649 inputs/source files, inherited schedules, evaluation
reservations, runtime assets and the new frozen source. Source/work alias isolation
is checked before registration; input hashes and catalog reservations are rechecked
before training and final evaluation. Original audio stays ignored and local.

Two arms times seeds 20261012 / 20261013, each **16,384 updates**, all fresh.
Each update uses four matched paired/identity batches. This consumes the complete
65,536-draw schedule: 327,680 paired crops and 27,627.408 unique supervised seconds
per trajectory; 182 recordings (157 music, 25 speech), nine codecs and five
channel views, with 80:20 music/speech sampling. This is scheduled finite exposure,
not full-library, artist-independent or new held-out evidence.

The objective is unchanged: balanced equal paired/identity
`log1p(direct original-reference spectral loss) + log1p(raw MR) + log1p(fused MR)`.
No new complex-loss weight, DSP teacher, limited target, auraloss switch or
perceptual term is mixed into this architecture comparison. Realized-complex
supervision remains a separate subsequent control after its own correctness and
deployment checks.

Fresh Adam uses betas 0.9/0.999, epsilon 1e-8, global clip 5 and all parameters.
The first 256 updates warm from 1e-5 to 1e-4; cosine decay ends at 1e-5. There is
no post-step multiplier, veto, best-checkpoint selection or shortened control.
The budget matches established fresh-model training; completion does not prove
convergence. All trajectories must finish and freeze before quality scoring.

Preflight performs two actual TRAIN updates per seed/arm, verifies projection
displacement and phase-history derivatives, and exports each diagnostic state for
native correspondence. Diagnostic states are then discarded. A repeated-final-phase
feature intervention keeps magnitudes/context/targets fixed and examines only
final-frame magnitude, whose current phase is unchanged. The rotation head cannot
change that magnitude. This diagnoses use of phase history, not audio quality;
the altered features are not claimed to come from a physical waveform.

Review retains common fit/history diagnostics, the complete raw/native development
review, all 160 P55 long-form cases, P71 independent spectral/fidelity rulers,
recording-paired uncertainty, codec/identity strata and every counterexample.
Candidate versus matched control must improve native MR and HF SC in both seeds
on the registered recheck; the other fixed native music panel remains visible.
Legacy, untreated and identity DSP are separate comparators. Trained native timing
and blinded materials are generated after all endpoints freeze. Listening ratings,
new source-disjoint tests and actual worker/device evidence remain separate.

## Launch status

The frozen-source process launched at 2026-11-10 00:55:45 Singapore time from
commit `30e8ccb`, with process-bound idle-sleep prevention. All 95,649 registered
bindings passed before real-data preflight. All four preflights completed their
two updates, with 12,336 changed relation-projection values in each arm/seed.
Updated native complex correspondence has maximum error 3.933e-6.

The magnitude controls have exactly zero final-magnitude phase-history derivative
and repeated-phase change on the eight TRAIN probes. Phase-arm maximum derivative
L1 values are 2.555e-5 / 3.466e-5, and repeated-final-phase magnitude RMS changes
are 3.370e-5 / 4.811e-5 dB. These small early effects establish the intended
learning/information path; they do not establish quality improvement or adequate
use of history after full training.

The first formal `20261012-magnitude` trajectory has started from fresh state;
its initial checkpoint hash and finite model/Adam tensors verify. The preflight
completion receipt also verifies. No execution failure is present at launch.
Live status is `progress.json`; `launch.json`, `run.log`, per-trajectory
checkpoints and completion receipts are authoritative. No new restoration
result is available yet; default weights remain unchanged.

## Progress check: 2026-11-11

At 03:01 Singapore time, the first `20261012-magnitude` endpoint has completed
all 16,384 updates. Its endpoint SHA-256 is
`fa97d99b3a451c94ff258095377932d84b7383b9d535e09295f88b646acb2468`.
The second `20261012-phase` trajectory is at a verified 12,288-update checkpoint
(75% of that trajectory). Both seed 20261013 trajectories are still pending.
Total recorded optimization is 28,672 / 65,536 updates (43.75%).

The completed endpoint and current checkpoint hashes verify; both contain finite
model/Adam tensors, finite loss histories and matching step/history lengths.
The training process and its sleep-prevention helper remain live; no execution
failure file is present. Recorded optimization time is 15.097 hours for the
finished control and 10.933 hours so far for the phase arm. These are progress
measurements, not a partial model ranking or convergence result.

No final raw/native comparison has started: the registered runner waits for all
four trajectories and freezes their endpoints before scoring. Default weights
remain unchanged. The status snapshot is retained in
`data/maintenance/cleanup-20261111/training-status.json`; later live progress
continues in the run directory. Workspace maintenance excludes the active run,
its source snapshot, cache, detector, checkpoint and evaluation dependencies.

## Progress check: 2026-11-12

At 02:48 Singapore time, three trajectories have completed all 16,384 updates:
both seed 20261012 arms and the seed 20261013 magnitude control. The final
`20261013-phase` trajectory is at a verified 6,592-update checkpoint. Total
recorded progress is **55,744 / 65,536 updates (85.06%)**.

All four endpoint/current-checkpoint hashes, registration signatures and
step/history lengths verify. Model/Adam tensors and recorded loss histories are
finite; no execution failure is present. The training process and its
sleep-prevention helper are live. All original budgets remain unchanged.

The final arm's recent 256/1,024-update mean durations are 3.014 / 3.042 seconds
per step. At those rates, its remaining training would take approximately
8.20 / 8.27 hours; this is a load-dependent estimate and excludes subsequent
evaluation time. No final comparison or trained runtime grid has started, and
partial trajectories are not used for model selection or quality claims.

The detailed hashes and status snapshot are retained in
`data/maintenance/p74-status-20261112/024808.json`. Default weights remain unchanged.

## Completed review: 2026-11-12

All **65,536 updates** finished: four endpoints at exactly 16,384 updates each,
with 58.587 aggregate optimizer hours. No registered execution failure is present.
Endpoint hashes, registration signatures, finite model/Adam/history values,
completion receipts, native review artifacts and all 20 timing-trace hashes
verify. The training and sleep-prevention processes have exited. No endpoint was
selected early, training extended, score rerendered or timing retried for this audit.

The reusable `training/scripts/summarize_band_capacity.py` now accepts explicit
study/control/candidate names while retaining its P72 defaults. It also uses the
existing deadline accountant's **100-hop timing exclusion**, rather than confusing
it with the model's 80-hop history warmup. Fifteen attribution tests and Ruff pass.
The frozen completion audit is
`data/runs/phase-temporal-training-p74-20261110/completion-review-20261112/`.
Its `receipt.json` binds the audit source, consumed inputs and `summary.json`.
Fit/history diagnostics are bound at audit time; they were not in the original
completion receipt. All original reports and failures remain retained.

| Endpoint | SHA-256 | Final 256-update mean objective |
| --- | --- | ---: |
| 20261012 magnitude | `fa97d99b3a451c94ff258095377932d84b7383b9d535e09295f88b646acb2468` | 0.408713027 |
| 20261012 phase | `18b037221885f8bc9956de8a977c08dc628443b2c9249e24351986590303941c` | 0.408795722 |
| 20261013 magnitude | `5f9683887c26efef36fc7f3b6c7945b87974dff58cb1353e4919fb514c02be20` | 0.407862058 |
| 20261013 phase | `6195df3e0ec8fb5108f808fa51d0bdc8647fe0eb08414d63e3c7b54486d8ae45` | 0.408308324 |

These are unchanged-objective training averages, not validation or convergence
proof. The eight common TRAIN probes also have mixed final objective differences:
0.367931399 to 0.367674507 in seed 12, and 0.367038209 to 0.367426187 in seed 13.
There is no consistent optimization advantage supporting an unchanged extension.

### Registered spectral comparison

All 480 native development roles, 160 long-form cases and 2,080 independent-panel
path records are complete. The long-form panel contains 20 reused recordings,
two excerpts each, three codecs and identity. Aggregate cases by recording;
the table reports percent change in source-equal error, **negative is better**.
Both models have 216,802 parameters and the same training budget.

| Phase versus matched magnitude, long-form paired music | Seed 20261012 | Seed 20261013 |
| --- | ---: | ---: |
| Native multiresolution spectral error (MR) | -0.040630% | +0.165277% |
| Native independent HF spectral convergence (SC) | +0.005631% | +0.034812% |
| Native HF shape | -0.151056% | +0.015127% |
| Native HF envelope | +0.158140% | +0.039396% |
| Native HF dynamics | -0.116998% | +0.023712% |
| Native HF complex error | +0.025417% | -0.060405% |
| Native >=8 kHz PCM RMS error | +0.041574% | -0.133771% |
| Native <8 kHz PCM RMS error | +0.000067% | -0.000131% |
| Native full PCM RMS error | +0.009110% | -0.026894% |
| Independent auraloss MR | -0.051253% | +0.177754% |
| Raw model MR, same long-form recordings | -0.050053% | +0.175866% |
| Raw model HF SC, same long-form recordings | +0.009406% | +0.047969% |

The registered requirement was lower native MR **and** independent HF SC in both
seeds. It fails. Source-paired 95% bootstrap intervals for native MR absolute
deltas are [-0.000435947, +0.000039074] and [+0.000226268, +0.001517775]. HF SC
intervals are [-0.000324364, +0.000467405] and [-0.000361141, +0.000800771]. Thus
the tiny HF SC increases do not establish a confidently positive harm, but neither
provide the preregistered improvement. MR worsens on 6/20 and 10/20 recording means;
HF SC worsens on 7/20 and 11/20. Every source delta and interval remains in
`panel-assessment.json` and the audit summary, including all identity/stereo rulers.

The largest adverse MR recording in both seeds is
`5b0e30010265a782cd18bb1f13250b1d6eaa063073bd10492a9b34a072db11a2`,
with deltas +0.001368801 / +0.006046544. Seed 12's largest adverse HF SC recording
is `7d7d51d147d7867001ec639e2755c7abc083a354b16cf4b31627c99ac8db9701`
(+0.002558174); seed 13's is the preceding `5b0e...` source (+0.004078903).
These are retained counterexamples, not tracks for threshold tuning.

Codec-specific native MR changes are MP3 -0.135452% / +0.085010%,
AAC +0.057731% / +0.375179%, and Vorbis +0.008921% / +0.039719%.
No codec has a two-seed MR win. Identity native MR changes are +0.092308% /
-0.001187%, with their source-bootstrap intervals crossing zero. Favorable
dynamics or coherent-error cells do not override the failed spectral criterion.

### Other development panel and named controls

The other fixed native population contains 25 music and eight speech recordings.
Raw-full review contains 51 music and 32 speech recordings. Keep those populations
distinct; a raw/native stage comparison uses the matched subset explicitly.

| Phase versus matched magnitude, paired MR | Seed 20261012 | Seed 20261013 |
| --- | ---: | ---: |
| Raw full music panel | +0.061589% | +0.242776% |
| Raw matched native music population | -0.134790% | +0.249217% |
| Actual Rust native music | -0.148400% | +0.206354% |
| Raw full speech panel | +0.120904% | +0.036345% |
| Raw matched native speech population | +0.151114% | -0.114981% |
| Actual Rust native speech | +0.158238% | +0.008810% |

The native music result agrees with the long-form seed split. Mixed raw effects
exist before DSP; this does not support blaming fusion alone. Speech transfer also
has a stage-dependent counterexample in seed 13. Native music identity MR changes
-0.755471% / +0.964164%; native speech identity MR changes -3.017527% / +0.796502%.
Relative weak-signal scores need absolute context: speech identity full PCM RMS
is 2.409888e-7 to 2.545635e-7 and 2.502003e-7 to 2.476165e-7. These very small
absolute errors do not establish audible regression or overall restoration.

P74 phase still reduces long-form native MR **3.241320% / 3.448509% versus
historical legacy** and HF SC 4.734931% / 5.023267%. This accumulated spectral
progress is separate from the failed matched-control increment. Against untreated
compressed input, MR improves 6.857547% / 7.056993%, but HF complex error worsens
1.980972% / 2.240707% and HF PCM error worsens 2.313542% / 2.635973% (all 20 recording
means adverse for the latter two). Against actual identity DSP, MR improves
6.496771% / 6.696989%, while HF PCM worsens 2.365285% / 2.687879%.
Reference fidelity and listening therefore remain unresolved. Original lossless
references remain unchanged; there is no limited-reference substitution.

### History is used, without a consistent restoration benefit

All eight registered TRAIN probes show zero magnitude response to phase history
for the magnitude controls, as required. The trained phase candidates now have
nonzero phase-history derivatives on all probes. Median derivative L1 is
1.624886 / 1.033511. Repeating the final phase across history changes final-frame
magnitude by median 0.570284 / 0.429834 dB RMS; ranges are 0.352495–1.046837 /
0.174156–1.286130 dB RMS. These are substantially nonzero learned dependencies,
not evidence that the model ignores the new observation.

This feature intervention holds magnitude/context/targets fixed and is not a
physical-waveform manipulation. It demonstrates use, not useful recovery or a
capacity ceiling. The remaining question is whether the objective and realized
output representation turn that dependence into reference-aligned spectra.
Neither more parameters nor a longer unchanged run is justified by these results.

### Trained compute and continuous deadlines

All **52,000** scored hops finish their compute below 5 ms, with zero service
times reaching a hop period. The measured chain includes frontend, ONNX Runtime
and DSP. Each endpoint has 13,000 scored hops; the paced case has 1,000.

| Endpoint | Maximum compute | Paced mean / p99 | Process peak RSS | Paced late hops |
| --- | ---: | ---: | ---: | ---: |
| 20261012 magnitude | 1.802584 ms | 1.065746 / 1.530376 ms | 32,587,776 bytes | 0 |
| 20261012 phase | 1.762875 ms | 1.051520 / 1.545163 ms | 32,686,080 bytes | 2 |
| 20261013 magnitude | 2.449333 ms | 1.068471 / 1.544560 ms | 32,636,928 bytes | 2 |
| 20261013 phase | 1.727666 ms | 1.059368 / 1.520684 ms | 32,882,688 bytes | 0 |

Paced process CPU including startup/warmup is 1.16–1.17 seconds over 2.98–2.99
seconds wall time. This is local replay resource use, not production-device CPU.
Fixed buffering is still 256 frames: 5.333 ms at 48 kHz / 5.805 ms at 44.1 kHz,
separate from compute and unmeasured device/driver delay.

All four late completions are retained, with zero-based raw hop indices:

| Endpoint / hop | Wake lateness | Compute | Completion lateness |
| --- | ---: | ---: | ---: |
| 20261012 phase / 367 | 3.880961 ms | 1.047334 ms | 2.263961 ms |
| 20261012 phase / 368 | 2.264086 ms | 0.644167 ms | 0.242502 ms |
| 20261013 magnitude / 264 | 0.501287 ms | 2.173125 ms | 0.009536 ms |
| 20261013 magnitude / 356 | 0.770714 ms | 2.449333 ms | 0.554589 ms |

Seed 12 phase has an on-time preceding hop and a two-hop late sequence; hop 369
recovers. Both seed 13 control exceptions recover on the next hop. Late wake and
compute consume the deadline together; stage wall time can include scheduling.
No OS root cause is established. These traces fail continuous-deadline qualification
even though all computation fits a hop. They are not observed device xruns, and
no inference about audible clicks is made. No retry replaces this result.

### Disposition and condition for revisiting

P74 is **complete and unqualified**, with separate quality and paced-runtime
failures. Its changed mechanism is learned but fails the registered restoration
comparison. The evidence does not reject every phase-conditioned architecture.
Twenty blinded listening trials are prepared; no listening ratings, new
source-disjoint test or actual worker/device qualification have been obtained.

The next justified diagnostic is the separately proposed realized-spectrum
supervision path: inspect original-reference high-band magnitude/complex errors
after synthesis and reanalysis, verify correct-reference/silence/weak-energy
gradients and native correspondence, then freeze a changed loss/control before
another full comparison. This is a hypothesis, not an implemented fix or a reason
to weaken spectral priorities. Independently inspect worker wake/headroom and
continuous output behavior. Revisit P74 only with such a demonstrated mechanism,
not unchanged scaling, more steps, favorable seeds/codecs or adjusted thresholds.

[P75](realized-supervision-p75.md) subsequently completes that research and frozen
gradient audit. Existing direct complex gradients already align with realized
ones in the full model; simple loss addition does not resolve the magnitude versus
complex conflict. Its phase-head routing proposal remains untested in training.
