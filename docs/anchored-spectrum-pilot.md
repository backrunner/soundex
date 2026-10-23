# P45: spectrum-led small-model method pilot

## Decision after calibration

[P44](pcm-fidelity-calibration.md) found that existing identity errors are much
larger than feature/OLA roundtrip noise. They must not be dismissed as numerical
precision. [P43](spectral-priority-update-audit.md) nevertheless provides a local
spectrum-improvement signal worth testing through accumulated updates.

P45 runs a fixed **128 update-attempt method pilot**, not full convergence training,
512-step capacity calibration or a model-size comparison. It uses the two frozen
431,015-parameter P38 controls. Each start has four arms: stored Adam and the P43
spectrum-led direction with **all three** 16/20/24-bit PCM half-step budgets.
There is no selection of a precision level after observing which one passes.
These budgets are engineering sensitivity controls, not audibility limits.

## Matched exposure and update policy

Every arm and seed receives the same newly registered schedule. Each attempt uses
four fit batches and a separate TRAIN guard batch; each batch retains four music
and one speech clip plus identities. Fit and guard recordings/source hashes are
disjoint within an attempt. Entire source-time support excludes P44 calibration,
P39/P41, P42/P43 and P38 development/shifted clips across channel views/codecs/rates.
Recordings may recur across attempts and were used in prior P38 training; neither
the guard nor final P44 recheck is independent testing. Coverage counts and the
union of supervised source seconds are registered before execution.

The Adam arm retains the original objective, stored optimizer moments, learning
rate `1e-4` and gradient clipping at 5. Four-batch gradient averaging differs from
P38's single-batch updates; it matches exposure to the proposed method. Thus it is
an unchanged optimizer/objective control, not an exact P38 trajectory replay.

Each spectrum arm pools the same four raw gradient sets and uses P43's guarded
amplitude/MR direction. It tries radii 0.001 then 0.0003, recomputing actual warmup
for finite-step assessment. The first passing radius is retained; otherwise the
model is restored exactly and the attempt is recorded as a no-op.

- Fit amplitude **and** MR must decrease from the current state beyond the existing
  numerical allowance. No-op and sub-allowance changes are not progress.
- Fit/guard complex error, MR dynamics and budget auxiliary, plus guard amplitude
  and MR, may not exceed the **fixed initial model** metric by more than the
  previously declared 0.1% sensitivity allowance and numerical tolerance.
- For every fit/guard identity example, full-band error RMS may increase above its
  fixed-start error by at most `2^-bits`. The actual output-displacement RMS **and
  peak** from that same fixed anchor must also stay below `2^-bits`.
- The same absolute constraints apply to each paired example's below-4-kHz
  component. This is a fixed low-band fidelity diagnostic, not a claim that every
  codec preserves all such bins. Offline Fourier projection is measurement only;
  it adds no deployment lookahead or model DSP path.

This changes finite-step acceptance, not the original clean supervision target.
Relative identity MR/risk and the mixed total are reported but are no longer
acceptance gates; absolute identity checks replace their role. Complex/dynamics
protection remains. A guard constrains the currently sampled examples, not every
past or possible recording. Fixed anchors prevent rolling budget accumulation
on a given example; they do not establish a global fidelity guarantee.

## Endpoint and failure handling

All eight trajectories must finish the same 128 attempts before any final review.
Accepted updates, rejected radii, guard violations, no-ops, timings and resumable
checkpoints are retained. There is no intermediate best-score selection or early
budget reduction. Resume checks registration/source/input signatures and the
stored optimizer/history; completed endpoints are hash-verified and skipped.

Freeze every endpoint, then recheck all arms and both starts on the same 32 P44
batches. Original P44 baseline results must reproduce exactly. Compare against
the start and matched Adam, separately for music/speech and paired/identity:
direct amplitude/complex/envelope/dynamics, raw MR and calibrated PCM errors.
Keep per-example counterexamples alongside means, and count no-ops separately.

This raw review does not qualify actual Rust DSP, simultaneous stereo, listening
or runtime. Default weights remain unchanged. Native evaluation follows only a
clearly labeled pilot result; the known real-time scheduling failures remain open.

Entry point: `training/scripts/run_anchored_pilot.py`; evidence:
`data/runs/anchored-spectrum-p45-20261023-v2/`. The run is registered and resumable,
with final assessment written automatically after the complete endpoint barrier.

## Execution preflight and retained failure

The original `anchored-spectrum-p45-20261023/` attempt stopped before its first
saved checkpoint: column-shaped batch sample rates broadcast the low-band FFT
mask to an extra batch dimension. A real-data candidate preflight reproduced the
same failure. No trained endpoint exists from that attempt; its registration,
executed source snapshot, traceback and `superseded.json` remain retained.

The correction reshapes rates explicitly to `[batch, 1]`. A regression test covers
both flat and column rates and requires identical two-dimensional output. The v2
run reuses the same scientific policy and schedule, with corrected source hashes.
All four branches undergo a real-batch execution preflight before the new run.

All four corrected branches completed that preflight, and v2 is now training.
The original and corrected schedules/groups match exactly. The initial Adam
trajectory has saved a checkpoint; this is execution evidence, not a quality
result. Live progress and resumable state remain in the ignored run directory.

Scheduled fit exposure is 2,560 paired crops from 436 recordings, with 1,122.537
seconds of unique MR support; guard exposure is 640 crops from 318 recordings,
with 290.503 seconds. Counts include speech, and the two role durations must not
be added without a joint interval union. Scheduled coverage is not completed
training coverage. All nine codecs and 12 genre labels occur, but 214 fit music
clips are unlabeled and 227 have unknown region. Simultaneous stereo remains open.

Local verification after the correction: **688 tests passed**, 223 existing
warnings; Ruff checks and 123 English documentation/link/index checks pass.
