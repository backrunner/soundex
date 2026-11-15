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

## Completed update diagnostic

All sixteen proposals and eight original-coordinate duplicates complete their
fixed budgets. All eight unit controls match both canonical models and persistent
Adam tensors bitwise after each step and match frozen P76 endpoints. All new
native/proxy correspondences pass. The 400 new native roles and reused controls
retain source hashes, original references and recording-level counterexamples.

Native paired-transfer changes below compare **gain64 realized-phase against
gain64 old**, not against historical legacy or an untreated input. Lower is better;
values are percentage changes of recording means, not percentage points.

| Origin | MR | Independent HF SC | Realized complex |
| --- | ---: | ---: | ---: |
| 20261012 magnitude | +0.000004924% | +0.000010676% | -0.000017954% |
| 20261012 phase | +0.000005662% | +0.000006090% | -0.000019640% |
| 20261013 magnitude | +0.000001370% | +0.000006259% | -0.000008928% |
| 20261013 phase | +0.000000875% | +0.000007581% | -0.000005936% |

All four prospective advancement gates fail. The added objective moves its
realized complex metric down in all four means, but MR and independent HF SC move
slightly up. Absolute MR increments are only 2.572e-9 to 1.657e-8. Dynamics is mixed.
These very small differences do not establish audible degradation or improvement.
Their recording-bootstrap intervals and every adverse source remain in
`assessment.json`; aggregate signs must not be extrapolated to all music.

The read-only `completion-review-20261116/` compares retained PCM and canonical
parameters without new training or rendering. Old gain64 versus unit old has
maximum per-role PCM RMS 2.434e-6 and peak 2.638e-5; one fit trace per origin has
active phase-difference RMS 4.990e-5–7.811e-5 radians. The realized-versus-old
increment at gain64 is much smaller: maximum PCM RMS 1.039e-7, peak 1.073e-6,
and fit-trace phase RMS 4.534e-6–8.217e-6 radians. This confirms an actual readout
effect but does not turn the auxiliary into a 64-fold audio improvement. These
are candidate differences, not errors against the original or audibility scores.

Increasing the readout step with the **old objective alone** lowers MR by
0.0000350–0.0000995% relative to unit old, while realized complex increases
0.0000710–0.0001469%; independent HF SC is mixed. Thus the larger step changes the
tradeoff, but does not solve it. This does not revise P74's separate legacy-relative
progress or establish any architecture ceiling. Two updates remain insufficient
to infer convergence. Do not extend this unchanged recipe, search a multiplier
on these transfer recordings or promote these endpoints.

The changed hypothesis is now answered narrowly: coordinate/Adam conversion is
correct, and a physically larger final-readout step alone does not produce the
required joint spectral improvement in this diagnostic. Before another training
budget, inspect the actual head displacement and objective contributions on fit
data to distinguish inherited-moment dominance from conflicting target directions.
Any changed objective must pass derivative/correspondence tests and a newly frozen
matched protocol. A fixed-budget training study still requires an exposure and
convergence rationale; no full training is launched from these tiny signs.

## Completed worker screen and retained measurement failure

Evidence: `data/runs/context-worker-p77-20261116/`, including original raw cases,
`failure.json`, frozen binary/source and `review-20261116/` hash receipt. All
48 scheduled cases execute once: 52,800 attempted input hops and 48,000
post-warmup scored callbacks. The original runner exits unsuccessfully at its
last case; read-only review reconciles all 48 without repeating a timing run.

| Observed quantity | Result |
| --- | --- |
| Existing software screens | 42/48 pass; overall fails |
| Worker model plus DSP maximum | 2.967958 ms; one 48 kHz hop-period overrun |
| Callback p99 / maximum, worst case | 0.017543 / 0.123667 ms |
| Presentation misses | 14 whole normal sequence; 10 after the first 100 hops |
| Longest missed presentation run | 4 hops, 10.667 ms |
| Producer lateness / queue-wait maximum | 5.985417 / 2.004625 ms |
| Process peak RSS, maximum | 39.469 MiB |
| Process CPU, one-core equivalent | 11.38–22.74%, including load/IO/teardown |
| Output | Every case finite/bounded, 140,800 frames, fixed256-frame delay |

Compute below 5 ms does not remove a 2.667 ms hop deadline or establish device
continuity. Maxima were captured before join in the frozen binary and can race
the last metric update, so they are **observed maxima**, not complete final-call
compute certification. The corrected source joins before returning diagnostic
counters. No new runtime claim is made for that changed binary.

Retained failing cases are 03 (44.1 kHz mono, seed 12 phase, QoS, three misses),
21 (48 kHz mono, seed 13 magnitude, RT request, two misses), 31 (48 kHz stereo,
seed 13 phase, QoS, two early misses), 43 (48 kHz music, seed 12 phase, QoS, one
processing-period overrun), 46 (48 kHz music, seed 13 phase, RT request, two early
misses), and 47 (48 kHz music, seed 13 phase, QoS, four misses). Early misses in
31/46 remain failures despite zero post-100-hop window misses. The observed maximum
compute in case 43 does not coincide with a presentation miss; cumulative worker
snapshots do not provide exact per-hop CPU attribution. RT startup requests are
accepted in all 24 requested cases, but this is not a live-policy/workgroup check
or a matched repeated policy-superiority study.

Case 47 rejects one input because its queue is full. The frozen harness incorrectly
waits for all 1,100 attempted inputs; 1,097 processed plus two discarded correctly
account for the 1,099 enqueued inputs. It times out and records drain=false even
though all output frames are present. The review preserves that invalid drain
finding, original failure and independent availability failure. Source now counts
queue rejections explicitly; a regression checks this exact counterexample and
final counters are captured after joining. No timing retry or changed threshold
is used to erase the failure. Both music inputs are stored **mid** mono views,
not physical stereo; the frozen provenance's word "physical" is corrected here
and in the review without altering registration evidence.

Four cases restart model history: total warmup counts are 160, 160, 107 and 128 instead
of 80. Extra dry history recovery is separate from presentation-miss runs and is
not a 10.667 ms bound on all enhancement unavailability. Initial 80-hop history
requires 213/232 ms at 48/44.1 kHz; output continues on its fixed delayed dry
timeline. No listening claim follows from finite output or functional fade tests.

Next runtime work should attribute submission, wake, processing and presentation
events with explicit host/device boundaries, then assess recovery under a newly
registered changed harness or integration. Do not shorten causal warmup, add
buffers or rerun this grid to make its failures disappear. Device 30-minute tests,
physical stereo music, other hosts and blind listening remain absent. No weights
are promoted and no full training is launched from this diagnostic.

Validation includes the full core library/integration suite, actual-worker
functional checks, production rejection and no-default-feature tests, seventeen
focused Python checks, a queue-accounting regression, Clippy with warnings denied,
Ruff and English/link/index documentation checks. The accelerated 30-minute audio
stress test remains explicitly ignored; it is not reported as passed or as a
physical-device test.
