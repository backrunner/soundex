# P72: matched joint band capacity training

Registered 2026-11-04 after [P71](evaluation-capacity-p71.md) evaluation calibration
and the single-pass native feasibility screen. This is a new capacity hypothesis,
not a repeated frozen-magnitude phase run. Quality and convergence are pending.

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

The first formal `20261012-small` checkpoint is hash-valid, finite and at 1/16,384
updates. Optimizer/history state is retained; all full trajectories start fresh
after preflight. No run failure is present at this check. The authoritative live
status is `progress.json`; the remaining trajectories run sequentially. Completion,
convergence and all endpoint quality comparisons are pending. Defaults unchanged.
