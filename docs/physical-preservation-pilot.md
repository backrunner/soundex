# P52: original-reference high-band preservation during continuation

## Question and changed mechanism

[P51](adapter-response-audit.md) preserves more compressed-input response than
scalar suppression at matched clean logit power. It does not match physical-band
response or actual clean PCM error. This pilot tests whether a physical,
original-reference auxiliary can preserve clean audio during continued spectral
learning, rather than suppressing an internal proxy after training.

This differs from P46's mean-gradient projection and per-example update gates:
the high-support adapter and encoder are now isolated, finite Adam displacement
is calibrated, and every scheduled update executes without a fidelity veto.
This is a bounded mechanism comparison, not convergence or a capacity test.

## Preregistered controls

- Start from both P50 **persistent** endpoints. This selection names a trajectory
  type, not whichever seed/endpoint scored best. Do not initialize from the
  selectively suppressed P51 head, which would confound initialization and loss.
- For each start, train `control`, `logit` and `physical`: six trajectories with
  128 actual updates each. All use fresh Adam at trajectory start, retained moments
  thereafter, lr 1e-4, betas 0.9/0.999, epsilon 1e-8, gradient clipping at five,
  and exactly 0.001 of each proposed parameter displacement, as in P50.
- Use the same 512 fit draws, four equal batches per update. Reuse the 32 review
  batches, globally recording/source-disjoint within the P50 partition but still
  historical TRAIN. No new independent test or full-library exposure is claimed.
- Keep the original common objective unchanged: paired/identity original-reference
  direct spectra and multiresolution reconstruction, with the existing frozen
  budget auxiliary. The new auxiliary is additional preservation supervision;
  paired high-frequency restoration remains directly supervised.
- `control` adds nothing. `logit` adds mean squared clean affine-head output on
  direct frames 89 onward. `physical` adds clean original-reference high-band
  reconstruction error power after raw OLA and supervision cropping. It includes
  input-dependent units, tanh, phase and overlap, which the logit proxy omits.
- Physical bands are [0,4), [4,8), [8,12), [12,16), [16,24.001) kHz at each clip's
  actual rate. Report all bands; the physical auxiliary sums the four bands above
  4 kHz, equally over identity clips. One-sided Parseval weighting divides FFT
  power by sample count squared, counts DC/Nyquist once, and sums to PCM MSE.
  There is no division by signal level, clipped reference or moving teacher.
- Before any training, calibrate each auxiliary weight separately per seed using
  32 fit batches at indices 0,16,...496. Fix it to 0.1 times the pooled common
  adapter-gradient norm divided by that auxiliary's pooled norm. Zero auxiliary
  gradients are recorded as inactive. No review score or penalty sweep enters
  calibration. Matching initial gradient norms is **not** matching audio fidelity
  or gradient strength later in training; weights and norms remain reportable.

Every arm has the same model/DSP/history/fit schedule and update budget. Common
totals, original-reference spectra and PCM errors judge the candidates together;
augmented training totals are not compared as if they were the same objective.
No early stopping, validation-selected endpoints, repeated timing to pass or
automatic longer run is allowed by this pilot.

## Mathematical and deployment boundaries

The physical auxiliary is a smooth quadratic in original-reference PCM error.
Exact recovery has zero value and zero gradient; silence has no denominator or
square-root singularity. Full-region FFT masks are offline analysis, not an
inference filter or additional deployment lookahead. Target detachment, zero and
weak signals, Parseval units at both rates with odd/even lengths, finite
differences, real model gradients and supervision crop boundaries are tested.

The raw reconstruction is the existing model OLA, not the final Rust DSP chain.
No DSP output is used as a teacher. Final native output, untreated compressed
audio and identity-through-DSP remain essential controls. Mean high-band MSE
cannot guarantee per-example peaks, low-band preservation or temporal continuity.
Keep all original 16/20/24-bit rulers and all counterexamples; do not tune them
using this pilot. Spectral supervision remains responsible for shape, magnitude,
energy and real temporal changes rather than smoothing transients.

Only the existing 49,794 adapter parameters train within the 480,809-parameter
model. Encoder, original decoder, budget and statistics remain frozen. The model
ABI, causal state, high-band support and DSP do not change. Real 80-frame warmup
remains outside supervision; raw OLA uses the preceding guard frame, as before.

## Frozen review and runtime

Finish all six trajectories and register every endpoint hash before any endpoint
fit/review score. Review 32 evenly spaced fit and all 32 review draws per model.
Report music and speech separately, with original-reference direct and MR spectra,
complex/temporal errors, per-band calibrated PCM power, all precision rulers and
same-population comparisons against the zero-adapter start, P50 parent, P51
separated/scalar controls and the new control/logit trajectories.

Export all six endpoints through unchanged sequential/state parity checks. Replay
eight evenly spaced review batches: 480 new mono paired/identity roles. Reuse only
hash-verified P50/P51 native controls on identical clips, including identity DSP.
Native and raw populations differ and must not be subtracted to attribute DSP.
No stereo listening or physical-device qualification follows from mono reviews.

Each new model receives four 3,000-hop 44.1/48 kHz mono/stereo compute runs and
one 1,000-hop paced 48 kHz stereo run: 78,000 scored hops in total. No training
or tests overlap timing. Report CPU/RSS, compute tails and paced delays separately;
retain historical failures. The unchanged 256-frame buffer is 5.333 ms at 48 kHz
and 5.805 ms at 44.1 kHz, outside the user's <5 ms compute target. Worker scheduling
is not changed or considered fixed by this loss experiment.

## Coverage and evidence

The P50 schedule covers 351 fit recordings and 986.623 unique MR-supervised seconds;
83 review recordings and 71.706 seconds. Repetition and calibration add no new
independent audio. The diagnostic mixture remains four music/one speech plus
aligned clean counterparts. Existing genre/region/channel gaps remain; see
[P50 coverage](adapter-continuation-pilot.md#registered-exposure). No dataset,
license scope or default distributed weight changes here.

Implementation: `training/evaluation/physical_preservation.py` and
`training/scripts/run_physical_preservation.py`. Registered local run:
`data/runs/physical-preservation-p52-20261024/`. Inputs/source snapshots, fixed
calibration, checkpoint/optimizer/history receipts and every raw/native artifact
remain ignored locally. Resume must reproduce uninterrupted parameters and Adam
state; completed reviews require receipts and the all-endpoint barrier.

Status: **complete / not qualified**. All six 128-update trajectories, twelve raw
fit/review reports, six exports and 480 new native roles finish. All 736 Python
tests, Ruff and 130-document checks pass. The registered run binds 5,697 inputs;
the completion receipt binds 6,936 artifact files and completed-run re-entry
verifies them. Remote
[CI 40084756741](https://github.com/backrunner/soundex/actions/runs/40084756741)
passes. No default weight is replaced.

### Frozen calibration values

| P50 parent seed | Common gradient norm | Logit weight | Physical weight | Logit/common cosine | Physical/common cosine |
| --- | --- | --- | --- | --- | --- |
| 20261012 | 0.2089280 | 24.7567271 | 885500.5873 | 0.1556110 | -0.3981256 |
| 20261013 | 0.1843543 | 25.2927703 | 1129820.8620 | 0.1717176 | -0.0308839 |

The physical auxiliary uses PCM squared units, with initial pooled gradient norms
2.3594e-8 / 1.6317e-8; its numerically larger coefficient does not imply a larger
calibrated initial gradient. Both auxiliary gradients have exactly 10% of the
common gradient norm after weighting. Their different directions motivate the
comparison but do not establish quality, transfer or future training behavior.
No coefficient was selected using review scores.

## Completed common-objective and restoration results

Every trajectory performs 128 nonzero updates, with finite displacements and
unchanged frozen base parameters. The largest combined gradient norm is 2.4633,
below the fixed clipping limit of five. Endpoints freeze before scoring. Negative
relative error changes below mean improvement; the reference is each seed's
**P50 persistent parent**, not another objective's total or an untouched input.

| Seed / arm | Raw music MR change | Native music MR change | Raw speech MR change | Common total change |
| --- | --- | --- | --- | --- |
| 12 / control | -0.004355% | -0.004520% | -0.003714% | -7.8832e-6 |
| 12 / logit | -0.004273% | -0.004428% | -0.003642% | -7.7737e-6 |
| 12 / physical | -0.000668% | -0.000478% | -0.000951% | -1.6848e-6 |
| 13 / control | -0.002441% | -0.002593% | -0.002091% | -3.7355e-6 |
| 13 / logit | -0.002473% | -0.002634% | -0.002135% | -3.8696e-6 |
| 13 / physical | +0.001550% | +0.001948% | +0.001087% | +1.5423e-6 |

The physical arm loses paired spectral benefit against both matched training
controls in both seeds. Seed 13 also regresses against its own P50 parent, already
on fit: music MR +0.000846%, speech MR +0.001744% and common total +8.1700e-7.
Thus the direction of this failure is not confined to the source-disjoint review
role. It does not establish overfitting, statistical significance of every tiny
change, or an architecture ceiling.

Raw music direct-amplitude changes for physical are -0.001976% / +0.003172%
against the parents; complex changes are +0.002692% / -0.001137%. Music MR dynamics
still increases +0.000780% / +0.000156%. Control and logit add small spectral gains
but retain complex, dynamics and identity regressions. No arm is an overall win.

Against the earlier zero-adapter starts, physical still has raw music MR gains
0.008315% / 0.001495% and native gains 0.009519% / 0.001789%. These include prior
training; they cannot hide the loss of seed-13 P50 benefit. Against untreated native
compressed music, MR is 0.976695% / 0.551333% lower, while MR dynamics remains
1.478931% / 1.425133% worse and HF PCM RMS 2.489620% / 2.525183% worse. Existing
base-model contributions and DSP conflicts remain part of those comparisons.

## Preservation benefits and retained counterexamples

Relative to the new control, the physical arm reduces raw clean-music HF4k error
**MSE** by 6.3138% / 4.4921% and mean full PCM RMS by 0.7519% / 0.2879%. This is a
real preservation tradeoff, not proof of more restoration. Raw clean-speech HF4k
MSE decreases 8.4366% in seed 12 but **increases 9.7911% in seed 13**. Against
the logit control, the seed-13 speech HF4k MSE increase is 21.3182%.

| Physical arm | Raw clean-music full RMS | Change from P50 parent | Raw clean-speech full RMS | Change from P50 parent |
| --- | --- | --- | --- | --- |
| 12 | 7.57910e-6 | +4.13613e-8 | 8.09867e-7 | +1.85928e-8 |
| 13 | 8.79255e-6 | -1.00228e-8 | 4.72708e-7 | +6.92692e-9 |

The physical arms' mean raw clean-music full RMS levels are -102.41 / -101.12
dBFS; speech is -121.83 / -126.51 dBFS. Raw identity HF8k RMS is 2.76048e-6 /
2.32802e-6 for music and 2.54114e-7 / 1.55196e-7 for speech. These use calibrated
PCM and `20*log10(mean clip RMS)`, not unnormalized FFT power. Small absolute
levels neither establish audible restoration nor erase the reference failures.

All six endpoints pass every fit/review 16-bit ruler and every paired-low4k
16/20/24-bit ruler. Identity failures remain, out of 160 examples per role:

| Seed / arm | Fit identity failures, 20 / 24 bits | Review identity failures, 20 / 24 bits |
| --- | --- | --- |
| 12 / control | 71 / 140 | 80 / 136 |
| 12 / logit | 69 / 140 | 77 / 135 |
| 12 / physical | 53 / 139 | 56 / 132 |
| 13 / control | 40 / 115 | 23 / 98 |
| 13 / logit | 38 / 111 | 25 / 94 |
| 13 / physical | 14 / 91 | 10 / 86 |

Native physical clean-music full RMS improves by 2.6230e-8 / 3.4131e-8 versus
the parent, but the absolute mean remains about 0.002865 (-50.86 dBFS), with the
retained identity-DSP control exposing the existing safety/processing discrepancy.
Native clean-speech full RMS **increases 11.005% / 20.781%** to 7.7616e-8 /
7.9425e-8 (about -142.20 / -142.00 dBFS). The absolute increases are 7.6949e-9 /
1.3666e-8: do not translate those large relative percentages into an audibility
claim. Raw and native populations differ (128/32 versus 32/8 music/speech clips
per role); only same-population comparisons support these statements.

## Post-completion attribution: units are correct, balancing is inadequate

A separate, hash-bound fit-only audit replays the **unchanged P50 parents** on
the same 32 calibration draws. It reproduces the registered pooled gradient
norms exactly and decomposes the physical auxiliary by corpus. No endpoint,
weight, tolerance or training budget is changed by this analysis.

- Music contributes **99.98499% / 99.99065%** of the physical auxiliary's scalar
  value despite speech occupying 20% of clips. This percentage describes the
  **new physical term**, not the common spectral loss or a gradient percentage.
  It exposes a strong corpus imbalance in absolute error weighting; the audit
  alone does not prove a causal loudness effect or identify every harmful clip.
- Weighted auxiliary batch-gradient RMS relative to common-gradient RMS is
  **9.0669% / 7.9031%** for physical, versus **3.4006% / 3.6094%** for logit.
  Physical is therefore approximately **2.67 / 2.19 times** as strong by this
  measure, even though the pooled initial vector norms both match 10%.
- On these calibration batches, the physical/common gradient norm ratio ranges
  **0.2334-38.1536% / 0.3133-32.3824%**; logit ranges **1.3677-7.9358% /
  1.3412-9.5792%**. Pooled vector cancellation and batch variation prevent the
  registered matching rule from controlling per-batch strength. These ranges
  concern the 32 calibration draws at the parents, not every later update.

The mathematics and code checks still support the stated original-reference MSE:
zero at correct recovery, calibrated units and finite gradients. The failed design
is the assumption that one unnormalized aggregate error and one pooled-gradient
scale supply balanced preservation. This experiment does **not** isolate physical
representation from reference targeting or establish equal regularization strength
through training. It cannot conclude that reference-aligned constraints are
intrinsically inferior to logit suppression.

## Runtime, resources and next decision

All six exports pass sequential/state parity and native outputs are finite and
aligned. Each retains 480,809 parameters and a 1,972,748-byte graph (about 1.88 MiB).
Across 30 runs and **78,000 measured hops** (81,000 including warmup):

- Maximum compute **1.699375 ms**, largest per-run p99 **1.021122 ms**; zero
  >=5 ms calls and zero compute service overruns.
- **One paced late hop**, lateness **0.447249 ms**, in seed-12 physical. No
  retry-to-pass; previous scheduling failures remain unresolved.
- Rust peak RSS **29.56-32.03 MiB**. Paced whole-process CPU/wall is
  **12.5-24.41% of one core**, including load/warmup. Isolated ORT warmed RSS
  increment is **13.38-15.23 MiB**, measured in a separate process.
- The fixed 256-frame buffer and absence of real-device/listening certification
  remain unchanged. This is a passing compute run, not a scheduling fix.

Do not promote the physical candidate or repeat the same penalty for longer.
The next bounded comparison should first separate music/speech preservation
contributions and calibrate auxiliary strength using batch-gradient variation,
with a fixed fit-only rule and unchanged spectral/control objectives. Verify
weak/silent behavior again before training. Corpus balancing and RMS calibration
are hypotheses to test, not already implemented remedies or quality guarantees.
The small gains and short budget still do not justify a capacity-ceiling claim.

All raw/native outputs, per-example rulers and earlier P50/P51 controls remain.
`interpretation.json` retains common comparisons, calibrated band errors, timing
and every ruler counterexample; `gradient-variation.json` retains per-batch
attribution. Their integrity receipts bind the completed run and local analysis
sources. No execution failure occurred. The quality, balancing and paced-runtime
failures above remain recorded; no further training is launched by this report.
