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
`data/runs/phase-routing-p76-20261115-v2/`.

Copy each endpoint's persistent FP32 Adam state, including its final stored rate,
betas, epsilon and step counter. Take exactly two cumulative updates of four
batches each, global clip five, with no line search, displacement scaling, veto or
moment reset. This yields twelve proposals / 24 diagnostic updates. Eight duplicate
canonical updates verify that the old arm reproduces both parameters and Adam
state bitwise. Freeze all twelve proposal hashes before output/transfer scoring.
Record actual displacements by parameter group and the old/auxiliary derivatives
along those displacements. Two updates do not establish convergence or capacity.

Reserve the lexicographically last speech work group in the inherited schedule
before selecting fit: take the first eight draws excluding that group, then the
first four draws whose sources and transitively connected work aliases do not
overlap fit or reserved evaluation sources. Selection reads metadata only. Keep the
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

## Preparation and attribution failures

The initial preparation in `data/runs/phase-routing-p76-20261115/` failed before
registration or optimizer updates: its first eight fit draws included all three
speech work groups in the inherited schedule (`vctk:p233`, `vctk:p267`,
`vctk:p340`), leaving no whole source/work-isolated transfer draw. The failed
source snapshot and preparation error remain. V2 fixes selection by reserving
the last speech group before fit selection; it binds the original evidence and
does not weaken isolation, remove speech, inspect audio scores or change losses.
Preparation now creates its evidence directory before selection so that early
failures are recorded automatically. A metadata-only regression test covers
selection and refusal when all remaining speech groups are reserved.

The first post-completion displacement reader incorrectly required optional DSP
shadow stages in ordinary native traces (`KeyError: dsp`). It ran no updates or
renders and changed none of the completed scores. Its source/error remain in
`completion-review-20261115/`. The corrected reader reconstructs active support
from actual input conditions with the same crossover helper as the validated
proxy. A regression test uses an ordinary trace without shadow stages. Results
and a receipt binding the original failure are in `completion-review-20261115-v2/`;
the entry point is `training/scripts/summarize_phase_routing.py`.

## Completed results

All twelve proposals finish their two updates. The eight duplicate canonical
steps reproduce old-arm parameters and persistent Adam state **bitwise**. Every
endpoint freezes before scoring; all input/source/endpoint hashes verify before
and after. There are 4,740 records: 1,920 raw, 1,920 proxy, 850 native and 50
untreated. All 17 traced graphs pass native/proxy correspondence, maximum PCM
error `1.192093e-7`, maximum RMS `1.134029e-8`, below the unchanged bounds.

The actual fit draws are `[2,3,4,5,6,7,9,11]`, transfer `[0,1,8,10]`, with
`vctk:p340` reserved from fit. Fit covers 40 recordings / 18.724499 unique
MR-supervised seconds; transfer covers 20 / 9.283955 seconds. Each has 80% music
and 20% speech, nine codecs and mono/left/right/mid/side views. Genre/region counts
and unknown labels remain in registration. Transfer speech has only one work
group; neither these labels nor recording counts establish artist independence.

Changes below compare **realized phase routing against its matched old update**
on the 20 native paired transfer recordings. Negative means lower error. These
are percentages, deliberately in scientific notation to show their tiny size.

| P74 origin | MR change | Independent HF SC change | Realized complex change | Joint gate |
| --- | ---: | ---: | ---: | --- |
| Seed 12, magnitude observations | +8.118e-7% | +3.688e-6% | -1.484e-6% | Fail |
| Seed 12, phase observations | +5.580e-7% | -1.163e-5% | -2.555e-6% | Fail |
| Seed 13, magnitude observations | -5.192e-7% | +8.029e-6% | +4.006e-7% | Fail |
| Seed 13, phase observations | -3.973e-7% | -1.080e-7% | +3.235e-9% | Fail |

The absolute MR differences are `-1.528e-9` to `+2.376e-9`; all four MR recording
bootstrap intervals cross zero. MR has 13/20, 11/20, 8/20 and 10/20 adverse source
means. HF SC has 14/20, 9/20, 10/20 and 11/20; realized complex has 11/20, 3/20,
12/20 and 10/20. Every source, codec, shape, envelope, dynamics, identity and
calibrated power counterexample remains in `assessment.json` and `records.json`.
Music and speech are also reported separately; no favorable population is selected.

The direct-complex reweighting control is similarly tiny and mixed. Its native
MR changes are `+8.567e-7%`, `+3.022e-7%`, `-3.745e-7%`, `-2.981e-7%`; HF SC
changes `+8.223e-6%`, `-5.731e-6%`, `+1.122e-5%`, `+2.696e-5%`. It does not
establish a useful alternative or a loss-location advantage. Proxy/native signs
can differ at this scale: seed-12 phase HF SC is `+9.966e-6%` in the proxy but
`-1.163e-5%` natively. Native scoring is retained, not replaced with the proxy.

The **unchanged old update versus its frozen parent** is a distinct comparison:
MR changes `-0.001458%`, `+0.000702%`, `-0.015315%`, `-0.011656%`, while HF SC
worsens `+0.035826%`, `+0.030803%`, `+0.046415%`, `+0.037708%`. Those much larger
shared update effects must not be attributed to the tiny new auxiliary. P76 does
not re-evaluate historical legacy superiority or undo P74's legacy-relative gains.

## Why this is inconclusive about the route

The registered joint-transfer gate fails on all four starts, so this unchanged
recipe has not earned a full-training launch. However, these two-step differences
do **not** establish audible degradation, phase-family failure or a capacity limit.
The outcome is an unsuccessful advancement screen with insufficient physical
effect for a meaningful quality conclusion, rather than a demonstrated large loss.

Realized-auxiliary head gradient norms are `1.127e-5`–`3.726e-5`, versus total
gradient norms of roughly 0.85–2.56. None triggers clip five. The actual Adam
displacement has an ascending auxiliary directional derivative in 3/8 steps;
the direct control also has 3/8. The old objective ascends in one step for every
arm, including the unchanged control. Raw gradient alignment alone was therefore
insufficient. These directional derivatives are local approximations, not finite
audio-quality scores.

Against matched old endpoints, realized routing changes head state by L2
`2.709e-5`–`5.051e-5`; base differences are only `1.766e-8`–`3.164e-8` L2 after
the second coupled update. Across all fifty native roles per endpoint, the largest
per-role PCM RMS differences are `1.829e-8`, `2.655e-8`, `1.720e-8`, `1.072e-8`;
the largest sample difference overall is `4.172e-7`. These PCM RMS changes are
comparable in order to the measured FP32 native/proxy discrepancy. They are not
zero, but favorable sign alone is weak evidence of a useful restoration change.

On the one traced fit role per endpoint, active predicted-phase RMS changes are
only `3.775e-6`, `5.636e-6`, `3.112e-6`, `2.394e-6` radians. This is output change,
not reference phase error, and does not extrapolate to all music. Small paired
HF-power decreases (`2.06e-13`–`3.80e-12` PCM power) do not establish either a
substantial attenuation mechanism or recovered detail. Preserve the original
spectral/fidelity rulers; do not relabel near-zero changes as success.

## Next decision

Audit the mapping from phase-head parameter updates to physical phase/PCM before
another loss-weight or longer-budget comparison. The readout explicitly uses
`pi * tanh(logit / 64)`, and copied Adam moments were trained under the old loss.
Those are candidate conditioning mechanisms, **not proven causes**: Adam can
cancel scalar gradient scaling, and weak native HF energy also reduces PCM effect.
A changed-coordinate experiment must first demonstrate forward equivalence and
correct transformation of moments, epsilon and learning rate in its control;
only then change one registered physical update scale. Do not tune a multiplier
to these transfer recordings, scale only a post-Adam displacement, or reset only
one arm's moments and attribute the result to architecture.

This is distinct from P49's oversized frozen-adapter step: P76 has a trained joint
base, explicit native-support phase head and extremely small incremental native
response. It also does not repeat P31's joint change of input/output coordinates.
Any later cumulative training still needs an adequate matched budget and all
frozen endpoints; a two-step diagnostic cannot determine that budget's outcome.
The production-worker adapter remains a separate prerequisite described above.

Validation: 53 focused Python tests across the new routing/reader and existing
spectral, temporal, isolation and evaluation checks; Ruff and English/link/index
checks pass. Only regenerable Ruff/Pytest caches were removed after inventory,
tracked-file and open-file checks: 6,718 logical bytes / 45,056 allocated file
bytes at that cleanup, not a claim about physical APFS free space. Training input,
checkpoint and failure evidence remains intact. No full training, timing retry,
default promotion, listening acceptance or device qualification occurred.
