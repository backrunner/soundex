# P74: phase observations inside the small temporal backbone

Registered 2026-11-10 after [P73](capacity-tradeoff-p73.md). The new model and
matched control are implemented. The fixed untrained feasibility grid passes;
full training uses four fresh trajectories under the unchanged P67 objective.
No trained quality improvement or default promotion is claimed.

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
