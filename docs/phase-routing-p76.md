# P76: phase-routed finite Adam and native-output transfer

## Changed hypothesis and fixed controls

[P75](realized-supervision-p75.md) found conflicts between realized magnitude and
complex derivatives. Its phase-head subspace was more favorable, but a Euclidean
gradient is not a persistent-Adam displacement. P76 changes **gradient routing**:
the original joint objective continues to update every base and head parameter;
only the additional original-reference complex derivative is restricted to
`phase_head.*`. This differs from P69's frozen-base phase training.

All four completed P74 endpoints participate, including both seeds and both
observation arms. Architecture, precision, data, DSP and deployed forward graph
remain unchanged. Each endpoint has three matched continuations:

| Arm | Additional head derivative |
| --- | --- |
| `old` | None; exact unchanged P67 objective |
| `realized_phase` | `0.1 * log1p(realized HF complex error)` after synthesis and DSP |
| `direct_phase` | Exact change in direct log risk when its complex coefficient changes from 0.1 to 0.2 |

The latter is a mechanism control: it does not norm-match the auxiliary and does
not isolate loss location alone. No coefficient is selected from P75 results.
Simply detaching predicted magnitudes would be incorrect because the phase head
also consumes base features. Parameter routing is tested explicitly. The common
global clip can still couple the resulting base displacement to the head norm;
later base gradients can also change because the updated head changes output.

## Registered finite-update budget

The runner is `training/scripts/audit_phase_routing.py`; the update implementation
is `training/evaluation/phase_routed_update.py`. The local evidence directory is
`data/runs/phase-routing-p76-20261115/`.

Copy each endpoint's persistent FP32 Adam state, including its final stored rate,
betas, epsilon and step counter. Take exactly two cumulative updates of four
batches each, global clip five, with no line search, displacement scaling, veto or
moment reset. This yields twelve proposals / 24 diagnostic updates. Eight duplicate
canonical updates verify that the old arm reproduces both parameters and Adam
state bitwise. Freeze all twelve proposal hashes before output/transfer scoring.
Record actual displacements by parameter group and the old/auxiliary derivatives
along those displacements. Two updates do not establish convergence or capacity.

Fit uses the first eight inherited TRAIN draws. Transfer uses the first four
later draws whose sources and transitively connected work aliases do not overlap
fit or reserved evaluation sources; selection reads metadata only. Keep the
80:20 music/speech crop mix and equal paired/identity risk. Record actual codecs,
channel views, recordings and unique supervised duration. These sources remain
reused TRAIN material from P74, not independent generalization evidence.

## Output assessment

Score all twelve draws on raw model synthesis and the differentiable DSP proxy.
Render the first fit draw and all four transfer draws through the frozen actual
Rust context-stream binary for each parent and proposal, plus identity DSP.
Untreated compressed PCM is a separate control. All roles are independent mono
views; this is not physical stereo evidence. Original aligned lossless references
are never limited, normalized or replaced.

Retain unchanged MR, independent 512/2048/8192-window HF convergence, shape,
envelope and real dynamics, realized complex/magnitude losses, calibrated HF
prediction/reference/error powers, low-band and full PCM error, temporal error
differences, and paired/identity results. Aggregate paired cases by recording
before bootstrap intervals and retain every adverse source and undefined score.
The independent panel windows are analysis context, not deployment lookahead.

The first role of the fit draw retains a full native trace for each graph. Replay
those actual observations/predictions through the proxy, retaining new loss values
on both PCM outputs. The existing P67 correspondence bounds remain maximum
`2^-14` and RMS `2e-5`; no known-counterexample threshold adjustment is allowed.

A positive finite-transfer diagnostic requires **each** endpoint's native paired
TRAIN-transfer recording means to improve MR, independent HF SC and realized
complex against its own matched old update. Keep all other tradeoffs visible;
coherent improvement through high-band attenuation is insufficient. Undefined
scores are not passing. A favorable result alone does not start full training or
replace default weights. Failure blocks this unchanged recipe, not all related
architectures. Further training requires a separately registered adequate budget.

## Production-worker boundary

The current P74/P76 export has four inputs (`features`, `context`, `condition`,
`state`) and the explicit `p20-context-head-diagnostic` marker. Its native test
uses `ContextHeadStream` directly. The production `RealtimeProcessor` constructs
`SoundExProcessor`; its stateful model contract requires three different inputs,
schema/metadata checks and 64 warmup frames. The experimental stream uses 80
enhancement-warmup hops and its own past-context/detector frontend.

These interfaces are not interchangeable. Relabeling metadata would not supply
missing inputs, reproduce frontend decisions or prove callback safety. Production
already has bounded queues, predictive worker waiting, optional time-constraint
scheduling, dropped-hop state resets and fixed-timeline dry recovery. P76 makes
no scheduling change and does not rerun the failed P74 paced timings.

A follow-up runtime adapter must connect this experimental processor to the
actual shared transport/worker machinery behind a diagnostics-only boundary,
preserving its state/reset/warmup semantics. Then register one fixed callback
replay schedule with submission/queue/compute/presentation and fallback traces,
including deliberate delay, gap and recovery controls. Do not substitute the
legacy production model and call it validation of P74. Physical-device continuity
and listening remain separate requirements in [the real-time standard](realtime-standard.md).

## Status

Implementation and mathematical/control tests are prepared. The preregistered
finite-update/native experiment must finish before any quality conclusion.
