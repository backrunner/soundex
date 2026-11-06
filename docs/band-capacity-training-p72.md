# P72: matched joint band capacity training

Completed 2026-11-06 at 22:55:10 Singapore time; audited 2026-11-07. All four
16,384-update trajectories and fixed reviews finish. The double model improves
long-form native spectral means, with conflicting shape/PCM and development-panel
results and one paced deadline failure. This is useful capacity evidence, not
overall quality, convergence or release qualification. Default weights unchanged.

Registered 2026-11-04 after [P71](evaluation-capacity-p71.md) evaluation calibration
and the single-pass native feasibility screen. The original protocol follows.

## Frozen comparison

- Fresh small 204,418 versus double 408,023 parameters, both fully trainable.
  The only inference architecture change is band token width 24 to 67; phase hidden
  width 32, six temporal dilations, output support and decoder interface stay fixed.
- Seeds 20261012 and 20261013, four trajectories, **16,384 updates each**. Complete
  all four endpoints and freeze hashes before any candidate comparison. No best
  intermediate checkpoint, shortened control or retrospective schedule change.
- Both capacities start fresh with zero residual output heads, not inherited P69
  weights. Historical legacy and identity DSP remain separate reference paths.
- Same full P63/P69 schedule: 65,536 draws, four paired/identity batches per update,
  327,680 paired crops and 27,627.408 unique supervised seconds per trajectory,
  after warmup/overlap accounting. Actual coverage is 182 recordings (157 music,
  25 speech), nine codecs and mono/left/right/mid/side views; 80:20 music/speech.
  This is finite scheduled exposure, not full-library or artist-independent training.
- The 16,384-step budget matches the established fresh P63 model scale. Reaching
  it does not prove convergence; retain update histories and final-window trends.

## Objective and optimizer

Retain P67's original-reference objective: balanced `log1p(direct HF spectral
loss) + log1p(raw reconstructed MR) + log1p(fused reconstructed MR)` with equal
paired/identity roles. The new evaluation panel is not an added training loss.
No DSP teacher, clipped reference, changed loss weights or perceptual proxy.

Optimize **all** model parameters with fresh Adam, betas 0.9/0.999, epsilon 1e-8,
global gradient clip 5. Same 256-update warmup from 1e-5 to 1e-4 and cosine decay to
1e-5. The historical phase helper checks/clips only the phase head, so P72 uses
an explicit full-parameter update helper. Missing/nonfinite gradients stop the
run and retain failure evidence; the training path does not silently fall back
to phase-only optimization.

Real-data preflight performs two updates for all four configurations, checks
encoder parameter changes, exports each diagnostic state and verifies actual
Rust correspondence at both rates and channels. Those diagnostic states are
discarded; full trajectories start fresh. Preflight is not a quality selection.
Tests verify exact model/optimizer continuation and amplitude updates.

## Evaluation and runtime

Apply the frozen [evaluation protocol](evaluation-protocol.md), including
original MR plus independent HF SC/shape/envelope/dynamics, absolute errors,
complex distance and stereo. Primary capacity evidence requires lower source-equal
native paired-music MR and HF SC in both seeds versus the matched small model;
confidence intervals, all adverse recordings and other metric tradeoffs remain
visible. A small waveform regression does not erase a spectral gain, and a
spectral gain does not establish overall quality.

The automatic review retains raw/native source-isolated development reviews,
all 160 P55 long-form cases, untreated/identity/legacy controls, separate codec and
paired/identity strata, and recording-cluster bootstrap intervals. P55 recordings
are repeatedly exposed rechecks. New independent generalization and human ratings
remain separate prerequisites for qualification. Blinded materials include all
endpoints; generating them is not listening evidence.

After training, measure every endpoint through actual Rust: frontend/model/DSP,
CPU/RSS, service and paced deadlines. The 5 ms compute target excludes the fixed
256-frame buffer; hop 128 deadlines still apply. P71's untrained screen cannot
replace these trained measurements or actual-device validation.

## Evidence and failure retention

Local run: `data/runs/band-capacity-training-p72-20261104/`. Registration binds
inherited source bytes, source/work reservations, exact schedule, runtime binary,
package versions, evaluation protocol and source snapshot. Checkpoints preserve
Adam state, full histories and registration signatures; source alias exclusions
are rechecked before launch and input hashes checked at resume/review barriers.
All audio, checkpoints and local reports stay ignored.

Before registration, two Adam-resume tests failed with `KeyError: cutoff` because
the reused fused-loss fixture lacked the detector field required by a model.
No experiment or optimizer update completed in those tests. The original fixture
and failure record are retained in `data/reports/p72-test-fixture-20261104/`.
Supplying the same fixed 14,000 Hz cutoff as the existing phase-training test fixes
the fixture; both capacity resume tests and role/codec aggregation then pass.
This is an invalid fixture, not a model-quality failure; no gate was relaxed.

## Verified launch

Launched at 2026-11-04 02:12:33 UTC (10:12:33 Singapore) from frozen source commit
`6b8f93e`, with a detached Python process and process-bound idle-sleep prevention.
All 94,655 input/source bindings passed verification. All four real-data
preflights completed two updates and both-rate native parity. Encoder changed
value counts are 41,350 for each small seed and 115,439 / 115,437 for the two double
seeds; these establish gradient/update connectivity, not audio quality.

At launch, the first formal `20261012-small` checkpoint was hash-valid, finite and at 1/16,384
updates. Optimizer/history state is retained; all full trajectories start fresh
after preflight. No run failure is present at this check. The authoritative live
status is `progress.json`; trajectories ran sequentially. The completed review
below supersedes the launch status. Defaults unchanged.

## Completed endpoint audit

All four exact 16,384-update endpoints have valid hashes, registration signatures,
complete histories and finite model/Adam tensors. Recorded optimization time totals
58.657 hours. The P72, shared, preflight and long-form completion receipts verify;
all 480 native development roles' artifact hashes also verify. No execution failure
occurred in the registered run. The prior test-fixture failure remains retained.

All 160 long-form cases, 2,080 raw/native/untreated panel records and 20 blinded
listening trials were produced. Listening ratings remain absent. The frozen
endpoints were reviewed together; no intermediate endpoint was selected. The last
four 256-update training windows fluctuate, so the fixed budget alone does not
establish convergence. Final mean training objectives are approximately 0.40934 /
0.40831 for small and 0.40518 / 0.40439 for double, not quality scores.

Read-only completion derivation and source:
`data/runs/band-capacity-training-p72-20261104/completion-review-20261107/`.
Its `summary.json` and `receipt.json` bind endpoints, full matched development
comparisons, the original panel assessment, native timing and the derivation
script. No training, audio rerender or timing rerun was performed during this audit.

## Long-form capacity result

The following is double versus matched fresh small, source-equal paired native
output on the twenty repeatedly exposed P55 recordings. Negative percentages
mean lower error; seeds are 20261012 / 20261013. The panel spans MP3, AAC and Vorbis.

| Ruler | Seed 20261012 | Seed 20261013 |
| --- | ---: | ---: |
| Existing MR | -0.919099% | -0.690461% |
| Independent HF spectral convergence | -0.789985% | -0.362459% |
| Gain-independent HF shape | -1.364731% | +0.239174% |
| HF envelope | -1.805886% | -1.022793% |
| HF dynamics | -0.712747% | -0.443265% |
| Complex HF distance | +0.601882% | +0.421968% |
| HF PCM RMS | +0.980238% | +0.662302% |
| Full PCM RMS | +0.190250% | +0.126236% |

Both MR and HF SC paired recording-bootstrap 95% intervals are below zero in both
seeds. Shape intervals cross zero in both seeds. MR has 4/20 and 3/20 adverse source
means; HF SC has 2/20 and 3/20. Complex HF distance worsens on 15/20 and 16/20 sources,
and HF PCM on 14/20 and 17/20. Keep all counterexamples in `panel-assessment.json`
and `panel-records/`; this is not an across-music generalization claim.

The HF PCM means change from 0.00892637 to 0.00901387 and 0.00895549 to 0.00901481.
The differences of mean RMS errors are 8.750e-5 / 5.931e-5 in linear full-scale
units; these are not RMS amplitudes of the added candidate residual and must not
be interpreted as its audible loudness. Paired low8k RMS increases by 3.902e-7 /
6.464e-7. Relative and absolute evidence are retained together, without a new
known-case tolerance or audibility claim.

Raw reconstruction shows similar magnitude benefits before DSP: MR -1.109892% /
-0.741238%, HF SC -0.886008% / -0.363118%, while HF PCM worsens 1.383628% / 1.300888%.
Thus native gains are not solely a DSP artifact, and the phase/coherent tradeoff
is already present in raw reconstruction.

Relative to historical legacy, double native MR improves 4.064560% / 4.103731%,
and HF SC improves 5.433529% / 5.366893%. Relative to untreated input, MR improves
7.650019% /7.687726% while HF PCM worsens 3.316330% / 3.327068%. These are different
comparators, not conflicting versions of one baseline result.

### Codec and identity exceptions

| Codec, double versus small native MR | Seed 20261012 | Seed 20261013 |
| --- | ---: | ---: |
| MP3 CBR 64 | -1.225061% | -0.589731% |
| AAC LC 96 | -0.891242% | -0.999521% |
| Vorbis q3 | -0.377129% | -0.471648% |

HF SC means improve for all three codec strata in both seeds, but Vorbis shape
worsens 0.242043% / 1.114022%. Identity native MR worsens 0.193581% / 0.550050%; its
absolute changes are 2.929e-5 / 8.330e-5. Identity full PCM error falls slightly,
while HF PCM changes by +2.516e-8 / +5.760e-7. Different rulers disagree; do not
collapse these results into either blanket success or blanket damage.

Concrete retained examples include MR regressions for recording
`f18318fab8c6e1e2bc6d0a18542a7e8a14b788eb77239948de5b646cc95e9be4`
in both seeds (source-mean absolute MR increases 0.00347434 / 0.00320776).
Recording `907a529dcea9466b5ce584c540b72de7a80cedfc98585f4aa01c9a1cef464da5`
has HF shape increases 0.01329560 / 0.03132891. No recording was removed or used
to tune an acceptance threshold.

## Different development panels do not agree completely

The source-isolated development review is also previously used development
material. Full raw review contains 51 music / 32 speech recordings. Actual native
review contains 25 music / 8 speech recordings, with 32 paired music cases per model.
Comparisons below use the same population where attributing native/raw differences.

| Music double versus small MR | Seed 20261012 | Seed 20261013 |
| --- | ---: | ---: |
| Full raw development review | -0.581460% | -0.712210% |
| Raw, restricted to native cases | -0.149765% | -0.252035% |
| Native development review | +0.002340% | -0.290006% |

The first native music seed is effectively flat with a small adverse mean;12/32
and 14/32 native music cases worsen. Paired native speech MR improves 1.011616% /
1.271112%, but clean-music identity native MR worsens 0.680447% / 1.242094%.
All per-recording and per-case results are retained in the completion summary.

The two-primary-metric capacity direction is positive on the full P55 recheck,
but the first seed does not improve MR on the other fixed native music panel.
Do not select only the favorable panel and declare an unrestricted experiment
pass. The evidence supports further investigation of capacity with quality
tradeoffs, not a qualified replacement or a claim that more parameters always help.

## Trained runtime result and failed deadline

All 52,000 registered measured hops are finite and below 5 ms compute, with no
frontend/model/DSP service call longer than its hop period. One paced completion
is late; consequently continuous-deadline qualification **fails** this grid.

| Endpoint | Maximum compute | Paced late hops |
| --- | ---: | ---: |
| Seed 20261012 small | 0.397333 ms | 0 |
| Seed 20261012 double | 0.600208 ms | 0 |
| Seed 20261013 small | 1.534708 ms | 0 |
| Seed 20261013 double | 1.663041 ms | 1 |

The double seed 13 paced 48 kHz stereo trace has one late completion at raw hop 784:
wake lateness 2.185114 ms, frontend 0.042292 ms, ORT 0.646833 ms, DSP 0.009125 ms and
total compute 0.698375 ms. Finish lateness is 0.217405 ms. The observed delayed
wake plus compute exceeds the 2.667 ms period; this does not identify an OS/root
cause or establish that a physical audio device glitched. It remains a runtime
failure, not erased by the <5 ms service result or a synthetic backlog model.

Double paced means are 0.476816 / 0.512339 ms and p99 values 0.538443 / 0.909489 ms.
Whole-process paced CPU is approximately 18.86% / 19.87% of one core, including
startup/warmup; those paced process RSS peaks are 33.17 / 33.42 MiB. Fixed buffering
remains 256 frames (5.333 ms at 48 kHz), excluded from compute. Previous timing
failures and the passing P71 fixture screen remain separate evidence.

## Decision

Retain both small and double endpoints and all counterexamples. The double model
has a modest reproducible long-form spectral benefit, but shape, coherent
reconstruction, identity and cross-panel behavior remain mixed, and pacing fails
once. No default promotion, automatic larger model or unchanged longer training.

Before another training recipe, use these frozen endpoints to locate source/band
and phase errors behind the shape/PCM tradeoff and the development-panel mismatch.
Investigate wake/worker availability separately using the retained timing trace;
do not rerun benchmarks until a pass. A new experiment needs a changed mechanism,
matched controls and frozen evaluation, plus listening and genuinely unexposed
recordings before any broad quality claim.
