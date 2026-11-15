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

## Status

The complete factorial partition and interaction controls pass. Output attribution
and late-producer controls are pending. No new weights or full training are planned
from unreviewed diagnostic results.
