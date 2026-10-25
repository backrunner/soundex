# P57: matched joint training of the existing small model

## Hypothesis and fixed comparison

[P56](trainability-audit-p56.md) found omitted reconstruction directions in frozen
features after complete P54 adapter training. Decoder-only unfreezing changed
little; removing identity protection or increasing learning rate introduced
additional errors. Tiny one-step mean improvements, some near float32 resolution,
do not establish useful quality or a capacity ceiling.

P57 tests whether **cumulative joint adaptation of the existing features and
reconstruction heads** transfers better than adapter-only continuation. No new
parameters, inference operators, context, lookahead or buffering are introduced.
Both scopes use the same 480,809-parameter causal model. Both completed P54
common-loss seeds participate; neither is selected by P55 ranking.

| Setting | Adapter control | Full reconstruction |
| --- | --- | --- |
| Starting weights | Same P54 common endpoint for each seed | Same |
| Trainable parameters | 49,794 | 480,809 |
| Adam state | Fresh, then persistent | Same |
| Learning rate | Fixed 1e-7 | Same |
| Betas / epsilon / clipping | .9/.999 / 1e-8 / norm five | Same |
| Budget | 4096 actual updates per seed | Same |
| Update sampling | Four batches, each four music + one speech with clean identities | Same |
| Objective | Existing common direct spectrum + raw MR, equal paired/identity risk | Same |

The full scope reconnects the adapter to live features. Its budget head receives
original-reference reconstruction gradients, with no DSP/budget-teacher auxiliary.
Phase, envelope and true temporal-change terms remain. No loss weight, hard
threshold, gradient veto or post-Adam displacement multiplier changes. Fixed
evaluation mode prevents training-statistic drift; real warmup is outside
supervision and detached in backpropagation.

Fresh Adam is matched in both scopes because P56 observed a material optimizer-
history effect. The old 1e-7 rate remains; this is not another rate search or a
claim that identity protection can be discarded. The 4096-update budget matches
P54's continuation and tests cumulative effects with broader exposure, not
convergence or architecture superiority. No intermediate winner or budget cut.

## Data and reservation

Reuse the entire registered P54 optimization schedule: 81,920 paired crops per
trajectory, 278 recordings and 287.52 unique MR-supervised minutes after merging
source intervals and removing warmup/overlap. This is scheduled exposure until
the corresponding updates finish. Four trajectories do not multiply unique
audio duration. Source-time, codec, physical sample rate and mono/channel-view
assignments remain fixed. These independent views are not stereo training.

Before launch, verify every selected clean/degraded file checksum and exclude
all registered `evaluation-catalog.jsonl` file/PCM/work aliases using transitive
bridges through processed manifests and the approved catalog. P55's 20 recordings
remain excluded. The registered TRAIN calibration and historical review sets
are not used for optimizer updates. Dataset acquisition, licensing and defaults
do not change.

## Completion, review and runtime

Entry point: `training/scripts/run_joint_training.py`. Local evidence:
`data/runs/joint-training-p57-20261025/`. Registration binds sources, snapshots,
inputs, parents, sampling, review controls, loss code and listening template.
Checkpoints preserve exact moments, parameter scope, history and signature;
parameter/statistic isolation and resumed-step equality are tested.

Finish all four endpoints before comparative review, freeze their hashes, then:

1. Recheck all endpoints and both starts on 32 historical TRAIN music/speech
   batches with the common raw scores and retained per-example errors.
2. Export and verify causal ONNX/state parity. Render the eight evenly spaced
   native batches through the frozen Rust DSP, also retaining historical and
   identity-DSP controls. This adds speech rechecks absent from P55's music set.
3. Measure all four endpoints serially: 3000 hops for each 44.1/48 kHz mono/stereo
   case and 1000 paced 48 kHz stereo hops, totaling 52,000 measured hops. Record
   model+DSP compute, CPU/RSS, service and paced failures; no retry-to-pass.
4. Freeze all eight graph hashes before the complete 160-case P55 stereo recheck:
   four new endpoints, two exact parents, historical weights and identity DSP.
   Use unchanged original references, source-equal scores, strata and source
   bootstrap. Generate a new blinded comparison pack without invented ratings.

P55 recordings are now exposed: this is explicitly a **recheck**, not a new
independent test, even though training remains isolated. The reused renderer
retains its P55 metric/listening protocol; P57 registration and status identify
the new candidate set and recheck scope. Every adverse source/band/codec/identity
result remains visible. No automatic best-checkpoint selection or promotion.

The fixed 256-frame production buffer remains 5.333 ms at 48 kHz, separate from
the <5 ms model+DSP compute target and the 2.667 ms hop service period. Unchanged
graph size is not measured real-time qualification; actual-device and human
listening evidence remain separate requirements.

## Status

Registered and launched on **2026-10-25 at 14:09 SGT**. Supervisor PID 27222 and
its idle-sleep assertion are recorded in `supervisor.json`. Input checks completed
before the first actual adapter-control update; its checkpoint receipt verifies,
parameters changed and optimizer telemetry is finite. All four 4096-update
trajectories are queued in one serial supervisor. This is a launch/update result,
not a quality result; full training, raw/native rechecks and timing remain pending.

Registration binds 79,024 input/source hashes, including selected paired PCM and
all reservation catalogs. Both nonzero P56 full-model preflight exports have
480,809 parameters and 1,972,748 ONNX bytes. Their 320-hop parity checks include
reset and silence; maximum Cartesian errors are 1.213e-7 / 9.424e-8, and maximum
state error is 3.338e-6. These are export checks, not the future trained endpoints'
latency or quality qualification.

Local validation: **773 tests pass**, with 238 existing/export warnings; Ruff
passes all 349 Python files, and 135 English Markdown files pass links/index
checks. Failed P54/P55 quality criteria and the retained P56 operational failure
are recorded in the registry. Default weights remain unchanged.
