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

All four corrected branches completed that preflight. V2 subsequently completed
all eight 128-attempt trajectories and the frozen raw review. The original and
corrected schedules/groups match exactly. All ten endpoint/review hashes, all
eight checkpoint histories and 216 bound inputs/sources were verified on
2026-10-24. Adam counters advanced from 16,192 to 16,320 as expected.

Per-arm attempted fit exposure is 2,560 paired crops from 436 recordings, with 1,122.537
seconds of unique MR support; guard exposure is 640 crops from 318 recordings,
with 290.503 seconds. Counts include speech, and the two role durations must not
be added without a joint interval union. All draws were evaluated, but rejected
attempts did not update weights. Attempted exposure is not accepted-update
exposure. All nine codecs and 12 genre labels occur, but 214 fit music
clips are unlabeled and 227 have unknown region. Simultaneous stereo remains open.

Local verification after the correction: **688 tests passed**, 223 existing
warnings; Ruff checks and 123 English documentation/link/index checks pass.

## Completed pilot results

The new acceptance policy stalls; it is not a qualified training method. Values
below are changes in mean raw music errors against each frozen starting model,
on the 128 music clips in the reused P44 TRAIN review. Negative is lower error.
Seed order is 20261012 / 20261013. All arms completed 128 attempts per seed.

| Arm | Accepted updates | Direct amplitude | MR spectrum | Direct complex | Identity PCM RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stored Adam | 128 / 128 | −1.25788% / −0.66064% | −0.77642% / −0.46282% | +1.36434% / +0.58386% | +25.69522% / −9.14336% |
| Spectrum, 16-bit ruler | 2 / 3 | −0.00613% / −0.01029% | −0.00403% / −0.00646% | +0.00248% / +0.01114% | +0.31402% / +0.42582% |
| Spectrum, 20-bit ruler | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Spectrum, 24-bit ruler | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

The zero-update endpoints are parameter-identical to their starts. Their unchanged
metrics are no-ops, not successful preservation through restoration. The 16-bit
arm's mean music identity PCM error increases by only 3.2270e-8 / 6.8850e-8 in
absolute RMS units; this does not erase its small spectral gains or certify quality.
Adam's identity changes are +2.6406e-6 / −1.4784e-6, showing seed-dependent tradeoffs.
Its music MR dynamics worsen 0.15291% / 0.07681%; the 16-bit arm changes them by
−0.00027% / +0.00184%. No overall fidelity conclusion follows from amplitude alone.

Speech paired MR improves 0.80147% / 0.48549% with Adam, versus 0.00348% / 0.00638%
with the 16-bit arm. Speech complex error increases in both methods. Music raw MR
against untreated compressed input is lower by 2.49943% / 2.29901% for Adam and
1.74045% / 1.85107% for the 16-bit arm; the unchanged starts were already lower by
1.73649% / 1.84473%. These absolute-reference comparisons prevent crediting the
new method for gains inherited from its starting model.

## Post-hoc gate attribution

At each attempt's last tested radius, all 128 candidates per seed still lower
both fit spectrum objectives. In the 16-bit arms, below-4-kHz output displacement
violates the guard on **125/128 attempts in each seed**. Guard budget-auxiliary
checks also reject 38 / 44 attempts, and identity PCM checks reject 12 / 16;
these counts overlap and must not be added.

Across those low-band violations there are 569 / 515 offending example occurrences.
Every occurrence exceeds the output-displacement **peak** limit, but **none**
exceeds the registered increase limit for low-band error RMS against original
clean PCM. Error RMS actually decreases on 325 / 280 of those occurrences. The
RMS displacement limit is exceeded on only 57 / 27. These are repeated training
occurrences, not distinct recordings or an independent test.

For example, seed 20261012, attempt 1, fit batch 0/example 0 at radius 0.0003 has
low-band output displacement peak 5.1582e-5, above the half-step 1.5259e-5, while
error RMS against the original decreases by 3.2168e-6. The corresponding seed
20261013 case has peak 4.2245e-5 and error RMS decrease 2.3248e-6. Rejecting these
solely for changing the prior output confuses preserving a model output with
approaching the clean reference. Other fidelity checks can still fail: these
examples do not prove that removing the gate would qualify the proposals.

The fixed-anchor rule successfully prevents rolling allowances, but a PCM
quantizer's pointwise rounding bound is not a justified ceiling on every useful
restoration correction. Moreover, the P43 projected direction does not explicitly
optimize the new absolute low-band constraint. A post-step rejection rule alone
does not teach that tradeoff. This is an optimizer/policy mismatch, not evidence
of an architecture capacity ceiling or a reason to repeat more unchanged steps.

Next intervention: distinguish original-reference degradation from output change.
Keep displacement as telemetry; design paired preservation around original-target
band/time errors, while clean identity has its own original-target protection.
Include those constraints in direction construction, test beneficial corrections
and harmful transients explicitly, then preregister a new matched pilot. Do not
retrospectively relax P45 thresholds, select a permissive arm or discard any failed
case. The current candidates remain experimental; no new native/DSP, stereo,
listening or real-time qualification was performed.

Detailed evidence: `assessment.json`, per-arm `updates.json`, per-example `review/`,
`gate-attribution.json` and `final-integrity.json` in the v2 run directory. The
attribution is a post-hoc analysis and did not select or modify any endpoint.
