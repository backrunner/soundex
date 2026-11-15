# P77: physical phase-update coordinates and worker integration

## Changed hypothesis

[P76](phase-routing-p76.md) completed correct routed derivatives and copied Adam,
but its incremental native PCM changes were around the FP32 correspondence scale.
This is a conditioning diagnostic, not another unchanged longer training run or
an architecture-capacity comparison. P49 reduced oversized first adapter steps;
P77 examines the final phase readout in a trained joint model with persistent
moments and a tiny incremental response. No new data, loss or DSP teacher is added.

The readout is `pi * tanh((Wh+b)/64)`. Introduce `U=W/64`, `c=b/64` only in that
last layer. Its forward result must remain identical. For Adam equivalence,
transform first moments by 64, second moments by 4096, epsilon by 64 and learning
rate by 1/64. Compute global clipping in the original parameter-gradient units,
then restore the coordinate gradients before Adam. All other parameters, moments,
step counters and rates stay unchanged. Adam's approximate scale cancellation
means that changing the raw derivative alone is not evidence of a fix.

The unit-gain control proves exact forward, model and Adam-state equivalence
after each of two cumulative updates, for both old and realized-routed objectives
at all four P74 origins. Its endpoint must also match the already frozen P76
endpoint bitwise. Only proven-equivalent P76 output records are reused.

The changed factor is **64 times the physical last-layer learning rate**, fixed
from the existing coordinate scale before scoring. It affects both old and
routed objective arms equally. No coefficient search, post-step multiplier,
line search, veto or one-sided moment reset applies. Canonical exports multiply
the readout parameters back by 64 and use the unchanged inference graph/ABI.

## Budget and assessment

Entry point: `training/scripts/audit_phase_step_scale.py`. Evidence:
`data/runs/phase-step-p77-20261116/`. Four origins, two objectives and two physical
gains produce sixteen two-update proposals; eight original-coordinate duplicates
add sixteen equivalence-verification updates (48 updates total). Freeze every
proposal before scoring. This is not adequate full training or convergence proof.

Keep the P76 fit/transfer source/work partition, original references, nine codecs,
80:20 music/speech mix, equal paired/identity risk and detached warmup. All were
historical TRAIN sources; transfer means within-diagnostic isolation, not unseen
generalization. Reused records retain exact hashes and all counterexamples.

Eight gain64 endpoints receive twelve raw/proxy draws and five actual Rust draws
(400 mono roles). Compare gain64 routed against gain64 old, and both against
their same-objective unit controls. Retain spectral shape, HF convergence,
energy, dynamics, complex/absolute PCM, low-band and identity measurements, with
recording aggregation and music/speech reports. No favorable seed/path selection.

A positive advancement diagnostic requires each origin's native paired-transfer
MR and independent HF SC to decrease by more than
`8 * float32_epsilon * abs(control_mean)`, and realized HF complex to decrease,
without undefined scores. This prospective numerical ruler follows P49; it is
not an audibility threshold or a reclassification of P76. Magnitude attenuation
does not establish restored detail. A pass alone does not promote weights or
replace an adequate matched training and listening evaluation.

## Runtime boundary

A separate diagnostics-only adapter reuses the actual callback transport,
worker scheduling, gap resets and fixed-timeline fallback with the experimental
four-input model. Production loading must continue to reject experimental ABI
markers. Worker preparation and inference stay outside callbacks; 80-hop
experimental history warmup remains distinct from the fixed 256-frame delay.
Functional checks cover direct-stream correspondence at both rates/channels,
silent-channel isolation, a real worker sequence-gap reset and rejection by the
production loader. Existing callback allocation/failure tests remain applicable
because the callback transport is shared. These checks are not device continuity.

The once-only host screen freezes four P74 parent graphs (not P77 candidates),
two rates, synthetic mono/stereo and two real mono music views. QoS and requested
time-constraint scheduling alternate in order: 48 cases, 1,100 callbacks each,
the first 100 excluded from the 1,000-callback steady-window report. Availability
must pass both that window and the normal whole sequence excluding only two
padding hops. Keep the existing <5 ms compute, zero hop compute overruns,
callback p99<100 us/max<500 us, miss ratio<=0.1% and burst<=20 ms rules. Freeze
source, binary, input and graph hashes before running after optimizer/build work;
retain every case without retries. Process CPU/RSS includes load/IO/teardown.

## Status

The unit-coordinate tests pass exact two-step forward/model/moment equivalence,
including an active-clipping control and canonical export conversion. The fixed
experiment and runtime adapter checks are pending completion. Defaults unchanged.
