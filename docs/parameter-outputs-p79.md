# P79: finite parameter contributions and late-producer recovery

## Changed hypothesis

[P78](adam-worker-attribution-p78.md) rules out simple cancellation by Adam's
second moment or uniformly adverse readout momentum. It does not explain how
finite base, hidden phase and readout changes combine in reconstructed audio.
P79 examines that output interaction without another optimizer step, larger
model, coefficient search or changed reference.

Use all four P74 origins and their frozen P77 `gain64_old` and
`gain64_realized_phase` endpoints. Partition the complete state into `base.*`,
`phase_head.0.*` and `phase_head.2.*`. Each block is copied exactly from one
endpoint; all eight combinations are required. The all-old/all-routed states
must be bitwise identical to their anchors. These are diagnostic interventions,
not trained candidates. Freeze all 32 states and graphs before scoring.

## Frozen output protocol

Entry point: `training/scripts/audit_parameter_outputs.py`. Evidence:
`data/runs/parameter-outputs-p79-20261116/`.

Retain the eight P77 fit draws, original aligned lossless targets, paired/identity
roles, 80:20 music/speech sampling and original warmup. Raw-model synthesis and
DSP-proxy output use all eight draws. Actual Rust uses the first fit draw covering
each physical rate, selected from metadata only: draws 2 and 3. This is 640 native
model roles plus 20 identity-DSP controls; 80 anchor roles and 10 identity-DSP
roles already exist and are reused with their hashes. All eight draws retain
untreated controls. These are historical fit recordings, not new generalization
or physical-stereo evidence.

Keep the unchanged P76 FP64 score implementation, spectral shape, MR, HF
convergence, energy, dynamics, coherent and absolute PCM errors. Compare anchor
paths against untreated and identity DSP separately from incremental effects.
Aggregate by recording before bootstrap; report music/speech and paired/identity
separately. Complete three-factor Shapley marginals distribute each interaction
over its participating blocks and sum to the observed routed-minus-old change.
Also report the total change minus the sum of single-block changes as
nonadditivity. Undefined values stay undefined; never pick the best subset.

Show `8 * float32_epsilon * abs(old_mean)` beside incremental effects as P77's
existing numerical ruler, not an audibility threshold or a new acceptance gate.
All new native graphs retain the P67 max `2^-14` / RMS `2e-5` correspondence bounds.
Original P77 failures remain unchanged. No improvement, capacity ceiling or
training decision follows from attribution alone.

## Runtime question

Add an explicit late-producer diagnostic that delays callbacks 256 and 640 by
two and four nominal hop periods, respectively. Keep every sample, the original
nominal clock, bounded queues, fixed 256-frame output delay and gap reset/history
rules. Do not slow later callbacks to hide the compressed windows. Fault tests
must retain missed presentations and distinguish those misses from finite output,
dry-timeline fidelity and recovery. Allocation and audit serialization remain
outside callbacks. This is fault handling, not normal availability qualification.

Freeze eight once-only RT-requested cases: all four P74 parents, inherited 44.1 kHz
synthetic stereo and 48 kHz stored mono mid music, 1,100 callbacks each. Use
`trace_context_worker.py prepare --late-producer` with the completed P78 event
audit and the rebuilt `soundex-worker-replay` binary. Evidence will be retained in
`data/runs/late-producer-p79-20261116/`. No concurrent training, build, hashing or
other benchmark is permitted during the timed grid; there are no timing retries.

Require each injected delay to be observed, with missed presentations, a history
reset and a later accepted fully warmed result before the next fault/end. Retain
finite complete output and the unchanged 256-frame timeline. After two accepted
history-warmup outputs, compare every eligible PCM sample against input delayed by
256 frames through the actual protective limiter. Apply the existing `2e-6`
functional tolerance. This protected dry control is not the lossless restoration
reference or an audibility threshold. Keep all normal availability failures,
processing time, CPU/RSS, resets and warmup duration visible independently.

## Completed output attribution

All 32 frozen interventions finish with zero optimizer updates, 5,860 score
records and 57 new native correspondence checks. Maximum native/proxy PCM
discrepancies are `2.384186e-7` peak and `2.849005e-8` RMS, within the unchanged
bounds. The complete receipt binds 2,814 files. It includes 640 native model roles
and 20 identity-DSP roles, of which 570 are new. Original P77 anchor hashes and
complete input bindings verify; no endpoint is selected or promoted.

Raw/proxy attribution covers 40 reused fit recordings (32 music, eight speech).
Native attribution covers ten of these (eight music, two speech), with both
physical rates and paired/identity roles. These are independent mono channel
views, not physical stereo. Do not interpret differences between eight-draw
raw/proxy means and two-draw native means as isolated DSP effects. This is finite
historical fit exposure, not full-library training or unseen testing.

The following recording-equal paired native changes are routed minus old. Negative
means less error; these very small changes are not perceptual gains. The comparison
uses fit draws 2 and 3, not P77's transfer set, whose failures remain unchanged.

| P74 origin | MR delta | HF SC delta | HF shape delta | Realized complex delta |
| --- | ---: | ---: | ---: | ---: |
| 20261012 magnitude | -5.558e-8 | +8.094e-8 | +1.123e-7 | -3.995e-7 |
| 20261012 phase | -5.922e-8 | -1.128e-7 | +2.000e-7 | -4.373e-7 |
| 20261013 magnitude | -3.124e-8 | +4.196e-8 | +2.167e-7 | -2.223e-7 |
| 20261013 phase | -6.039e-8 | -1.391e-7 | -2.087e-7 | -2.809e-7 |

All four MR intervals span zero and their mean changes are smaller than the
registered numerical ruler (`2.193e-7`–`2.215e-7`). All HF SC means are smaller
than its ruler (`4.847e-7`–`4.867e-7`); only seed-13 phase's HF SC interval excludes
zero. Three HF-shape means worsen and have intervals spanning zero; seed-13 phase
improves on this narrow fit panel. Selecting that origin or a hybrid retrospectively
would not establish a useful training method.

For realized complex error, the readout's Shapley contributions are respectively
`-4.054e-7`, `-4.315e-7`, `-2.256e-7`, `-2.804e-7`: 98.67–101.48% of the net
decrease. Contributions can exceed 100% when another block opposes them. Base and
hidden contributions are much smaller and have mixed signs. All four total complex
intervals exclude zero, but the means remain comparable to the numerical ruler
(`2.243e-7`–`2.255e-7`), far from evidence of audible improvement.

Readout motion therefore reaches reconstructed audio. A blanket hidden/base
obstruction is not demonstrated. HF SC/shape nonadditivity is sometimes as large
as or larger than the endpoint difference, at this tiny numerical scale; do not
assign those spectral changes to one culpable block. Raw and proxy complex means
also decrease at all four origins, with mixed MR/shape/envelope/dynamics effects.
This is consistent with P75's overlap-add coupling warning, not proof that DSP
alone removes an otherwise useful improvement.

Paired native HF PCM RMS decreases only `3.776e-10`–`7.405e-10`; low-band RMS
changes lie between `-2.003e-11` and `+3.765e-11`. All four hop-boundary error means
increase (`3.394e-10`–`1.548e-9`). None is an audibility claim. Identity HF SC and
shape are mixed, while identity MR decreases only `7.987e-9`–`2.186e-8`. Speech has
only two native recordings: its paired MR/complex changes improve for both phase
origins and worsen for both magnitude origins. Preserve these strata and their
uncertainty rather than generalizing a pooled mean to speech or all music.

The models' established restoration behavior is distinct from this increment:
on the same ten native fit recordings, old-anchor MR is 0.229953–0.232213 versus
0.244598 untreated and 0.244704 identity DSP; HF shape is 0.164148–0.168390 versus
0.240398 and 0.238686. Yet realized complex is 0.235161–0.236413 versus 0.218988
and 0.219339, and HF PCM RMS is 0.000998158–0.001001442 versus 0.000973922 and
0.000978812. Better magnitude/shape coexists with worse coherent reconstruction.
These exposed-source controls do not rescore historical legacy or replace P74's
separately reported legacy-relative gains.

Every case, undefined value, recording delta and adverse source remains in
`records.json` and `assessment.json`. Concrete paired native counterexamples:

- `13253a83912ee8d75c892f2167cad1a22abd8529c226d18dbe6c815077cb8bd6`:
  seed-12 magnitude HF SC increases `5.662199e-7`.
- `beb6132a3648420e0a1a2b7ac5f5af0f1bdd4b422df6688be9c37924f964cf7e`:
  seed-12 phase HF shape increases `3.663991e-6`.
- `2b09292cb2dd5f720ece9d2aa0df575d4693786c04813367065d3919d2df6114`:
  seed-13 magnitude HF shape increases `2.034809e-6`.

These relative differences require the accompanying absolute errors; their signs
alone neither qualify nor reject audible quality. No failed P77 gate is relaxed.

## Completed controlled late-producer replay

All eight cases execute once and pass the registered fault-handling check. All
trace/counter reconciliations pass. Each case retains 140,800 finite/bounded output
frames at the fixed 256-frame delay, with worker drain complete and no worker
failure. Every injection produces misses and a history reset, followed by a fully
warmed accepted result. There are 80 initial plus two additional 80-hop warmups
per case. Enhancement unavailability during each 80-hop recovery is approximately
213/232 ms at 48/44.1 kHz; dry output continues, so this is not added output delay.

Each case has 234 eligible protected-dry hops (78 at startup and 78 after each
fault); every compared sample is exactly equal to the expected delayed, protected
input. Maximum PCM alignment error is zero. This excludes the transition hops,
does not assess their perceptual smoothness, and is not a device or listening test.

| Observed quantity | Result |
| --- | --- |
| Fault handling | 8/8 pass; 16/16 injected events observed |
| Unchanged normal availability screens | 0/8 pass on deliberately faulted input |
| Model plus DSP maximum | 1.662792 ms; zero hop-processing overruns |
| Callback p99 / maximum, worst case | 0.027711 / 0.064166 ms |
| Presentation misses | 85; 84 in post-injection bursts, one additional miss |
| Longest miss run | Seven 44.1 kHz hops, 20.317 ms |
| Queue wait / producer lateness maximum | 1.484334 / 15.225419 ms |
| Process RSS | 45.578–46.938 MiB |
| Process CPU, one-core equivalent | 21.00–27.38%, including setup/trace serialization/IO |

The first injection causes misses 257–260 in every case, reset input 259 and
first fully warmed acceptance at callback 340. The second causes misses
641–646 or 641–647, resets input 645 or 646, and recovers at callback 726 or 727.
Queue rejection, obsolete input discard and the first continuity-invalid result
are explicitly accounted for. No queue capacity, output clock or model history is
changed to hide these bursts.

Case `02` (seed-13 magnitude, synthetic 44.1 kHz stereo) also misses callback 558,
outside the injected bursts. Callback 557 begins 2.814158 ms late; input 557 has
only 0.112334 ms until callback 558 starts, despite its processing taking
0.895208 ms. Its expired output is discarded. The input history remains continuous,
so no extra history reset occurs. The registered segment summary reports callback
560 as the fully warmed accepted result after the **last** miss in that segment;
initial recovery from the injected fault already occurred at 340. Keep both facts,
not a fictitious 304-hop history warmup. Raw evidence is in `cases/02/raw.json`.

The complete receipt binds 574 files. RT requests are accepted in all eight cases,
but there is no untraced control, policy-superiority result or physical device
measurement. Fixed buffering remains 5.333/5.805 ms at 48/44.1 kHz, separate from
compute. Earlier timing failures remain. Physical stereo music, a real 30-minute
device session and blind listening are still absent.

## Decision and next distinct question

P79 completes the requested attribution and controlled recovery work. It does not
justify a global moment reset, freezing a supposedly harmful block, another gain
increase, longer unchanged training, or a default-weight replacement. The small
readout step improves its coherent target without consistently improving the
primary reconstructed spectral targets. P39–P46 already retain failures of simple
gradient pooling/projection; repeating those unchanged is also unsupported.

A distinct next prerequisite is an audit of original-reference **phase relations**
at meaningful bins, separately from complex distance. The
[AP-BWE paper, Section III-B](https://arxiv.org/html/2401.06387v2) combines wrapped
instantaneous phase, frequency differences and time differences with amplitude
and complex objectives. It studies speech low-pass bandwidth extension, not our
music-codec streaming task; its throughput does not certify our per-hop budget.

Source inspection matters here: legacy `models/losses.py::phase_gradient_terms`
already implements reference-relative circular frequency/time differences.
The current P74/P77 path instead calls `fused_spectrum.phase_objective`, comprising
direct amplitude/complex/envelope/dynamics plus raw/fused magnitude MR; it does
not call that legacy helper. P68/P74 add phase observations, which are different
from explicit relational supervision. Do not describe this as a newly invented
loss, silently turn old terms on, or assume their old scales transfer.

Before proposing new matched training, register a small fixed diagnostic comparing
the existing phase path, the legacy circular relation and a separately specified
wrapped-distance relation. Check reference zero/stationarity, weak/silent support,
wrap boundaries and antipodal behavior, finite differences, past/current-only
support, phase-head routing, original-reference magnitude/transient conflicts and
native output transfer. Retain the direct lossless target and current spectral
priorities. The new question is derivative information and its actual output
effect, not another complex-loss coefficient or a bigger model. This is a proposed
prerequisite, not an implemented or trained P80 result.

Runtime work can now move to an actual host/device integration with its real
presentation clock, physical stereo music and transition/listening evidence.
The controlled host replay establishes fixed dry-timeline recovery only; it cannot
repair time already lost by the producer or certify uninterrupted enhancement.

Validation: eleven focused Python factorial/event/recovery tests, three native
replay tests, Clippy with warnings denied, Ruff and English/link/index documentation
checks pass. Both complete receipts verify. All frozen evidence is retained; no
training sources, failed runs or checkpoints are deleted.
