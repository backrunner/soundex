# P81: metric definitions, numerical interpretation and spectral priorities

Completed 2026-11-16. The core spectral calculations remain useful, but several
definitions and decision practices needed correction. This audit adds independent
within-band temporal error, strict metric completeness and separate spectral versus
coherent evidence. It does not find a hidden overall-quality win in P80, alter
training losses, replace weights or retrospectively pass any registered gate.

## Changed question and evidence

Consulted [P70](baseline-metric-audit-p70.md), [P71](evaluation-capacity-p71.md),
[P80](phase-relations-p80.md) and the [failure registry](failed-experiments.md).
P70/P71 calibrated magnitude, log, energy and phase blind spots. They did not
validate the later P77 scalar epsilon gate, schema intersection or the distinction
between bin dynamics and band-envelope dynamics. P81 tests those specific questions
with analytic controls before descriptively reviewing every saved P80 comparison.
This is a changed measurement question, not another training-budget increase.

Evidence is retained in `data/runs/evaluation-protocol-p81-20261116/`: source snapshot,
registration, 18 synthetic waveform cases at 44.1/48 kHz, exact spectral-vector and
FP32 controls, schema counterexamples, 1,248 comparison interpretations and a
460-file completion receipt. All 144 native-versus-old comparisons are independently
reaggregated with strict schemas and the original recording bootstrap. Every
metric, source delta, interval and undefined/adverse source exactly reproduces P80.
The original eight joint-transfer decisions remain false and immutable.

The separate control evidence is under
`data/runs/evaluation-protocol-p81-controls-20261116/`. Its first 34 tests passed;
an isolated one-frame shape fixture triggered empty temporal-mean warnings in an
unused return field. The fixture now repeats the same vector over three frames,
without changing the shape definition or answer. The original source and log are
retained. Final controls and calibration correspondence are recorded under
`data/runs/evaluation-protocol-p81-validation-20261116/`.
All 44 focused tests pass with warnings treated as errors; every analytic
calibration value reproduces exactly after the fixture correction. Ruff passes,
all 162 Markdown documents pass English/link/index checks, and the P80, P81 audit
and final-validation receipts verify. The tests include original panel behavior,
the untouched phase-loss controls and the existing absolute-fidelity calibration.

## Findings and changes

| Question | Evidence | Action |
| --- | --- | --- |
| Is the core spectral normalization wrong? | Reference errors are zero; half gain has SC 0.5; Nyquist RMS has correct one-sided scaling; weak scaling preserves relative error and scales absolute PCM error | Retain the original magnitude/energy rulers and numbers |
| Does shape mean Hellinger distance? | Half the squared distance of normalized magnitude vectors is **squared** Hellinger; an analytic example returns 0.5, not sqrt(0.5) | Correct naming in documentation/comments; no square-root rescoring |
| Does envelope dynamics cover all spectral motion? | Alternating two equal-energy bins gives envelope/dynamics error 0, while signed-bin dynamics error is sqrt(4/3) | Add independent `*_spectral_dynamics` and its calibrated absolute companion |
| Can STFT scores miss samples? | A 0.25 impulse at sample zero receives zero weight in every left-aligned Hann window; complete PCM error is 0.001953125 RMS | Expose analysis coverage and complete-support PCM error; keep existing boundary/low-band controls |
| Can aggregation hide missing information? | Historical metric-key intersection silently drops a field missing from one case; a NaN opposite `None` can enter the undefined branch | Require strict schemas and finite defined values before pairing; preserve undefined-source and absolute-error reporting |
| Is the scalar FP32 gate a valid error bound? | One representable PCM ULP produces an exactly predicted nonzero FP64 SC change below the historical ruler | Retire that heuristic from prospective interpretation; retain old registered decisions |
| Must phase error always fall for spectral progress? | That conjunction conflates two objectives and rejects every spectral/coherent tradeoff by construction | Report primary spectral direction and coherent/preservation costs separately; no automatic quality approval |

The within-band dynamic error is `||diff(|P|-|T|)|| / ||T||`. Unlike positive-only
onset flux, it retains both rises and falls. It compares with actual reference
changes, not a smoothed target. Each resolution keeps its original hop and its
absolute RMS companion. The three-resolution mean is a fixed diagnostic, not a
derivative per second or perceptual score. Original P80 records do not contain
these new independent fields; the review explicitly marks them missing. Existing
training MR dynamics already penalizes bin-level changes, so this finding does
not mean temporal supervision was absent.

Squared shape alone still ignores gain and can reward an incoherent reconstruction.
Polarity inversion has zero magnitude SC but complex distance 2 in the analytic
control. Gain/phase/noise/transient controls therefore remain separate. A waveform
guard prevents window-edge errors from disappearing without overriding the user's
spectral priority or conflating a small coherent error with audible damage.

### Numerical resolution is not audibility

The exact Nyquist reference amplitude is 0.125; control amplitude is 0.0625 in
FP32. Moving each control sample one representable ULP toward the reference gives:

| Quantity | Value |
| --- | ---: |
| Before / after independent HF SC | 0.5 / 0.4999999403953552 |
| Observed and analytic SC difference | -5.960464477539063e-8 |
| Historical `8 * eps32 * control_score` ruler | 4.76837158203125e-7 |
| Actual incremental PCM RMS | 7.450580596923828e-9 |

The score difference is real for these represented samples even though it is
eight times smaller than the heuristic. This does not establish audible improvement,
the numerical repeatability of every native execution, or the practical relevance
of P80. It demonstrates why that scalar cannot be called a universal error bound.
Likewise a proxy/native implementation difference does not by itself invalidate a
direct native/native comparison. Its relevance requires a matched numerical
analysis, not transferring a PCM bound into an unrelated normalized score.

## What changes in the P80 conclusion?

All eight circular/wrapped relation comparisons have MR and HF SC recording
bootstrap intervals crossing zero on the twenty-recording native transfer recheck.
The correct primary classification is **unresolved**, rather than proof of no
learning, a failed architecture or a useful improvement. All four circular
comparisons have above-zero complex-error intervals; all four wrapped comparisons
have complex intervals crossing zero. These remain descriptive panel intervals,
without multiple-comparison correction or a perceptual-severity interpretation.

Some supporting directions are visible: circular shape intervals are below zero
at both seed-20261013 origins; seed-20261012 magnitude circular envelope dynamics
decreases; seed-20261013 magnitude wrapped envelope error decreases. These isolated
findings do not override uncertain primary results. The original fit-risk decreases,
speech/music disagreements, absolute error scales and every adverse recording
remain in P80. The additional realized-complex arm also has mixed primary/coherent
directions; no origin is selected to claim a win.

This reinterpretation covers the same exposed recordings and saved endpoints.
It is post-hoc, not a newly preregistered experiment or an unseen test. P80 only
made two updates per endpoint; neither its old gate failure nor this interpretation
establishes convergence, overfitting or an architecture ceiling. Historical legacy
was not rescored here. Earlier P72/P74 legacy-relative gains remain distinct from
P80's incremental comparison against its matched old update.

## External checks and scope

[Auraloss's official implementation](https://github.com/csteinmetz1/auraloss/blob/main/auraloss/freq.py)
uses normalized magnitude difference for spectral convergence and exposes log,
linear-magnitude and phase components separately. This supports retaining multiple
spectral rulers; it does not supply an auditory threshold. The local pinned
package remains `auraloss==0.4.0`; no dependency upgrade or newly weighted loss
occurs here. Our independent NumPy panel remains separate from the training loss.

[ViSQOL's official guidance](https://github.com/google/visqol) requires 48 kHz audio
mode, downmixes channels and recommends roughly 8--10-second inputs. The short P80
crop bank is not silently converted into valid ViSQOL evidence. A future pinned
long-form audio-mode score can complement, but cannot replace, physical stereo
checks and blind listening. The
[audio quality metric study](https://arxiv.org/abs/2110.11438) evaluates application
domain dependence against listener ratings; it does not justify treating a generic
spectral objective as a universal perceptual estimator. No new MOS/listening result
is claimed, and no speech-only metric is made the primary music-restoration ruler.

## Next mechanism and prospective requirements

Use [protocol v2](evaluation-protocol.md) for new reports. Keep the primary fixed
MR and independent HF SC, with shape, per-band energy, both kinds of temporal
error, absolute weak-band errors and coherent/preservation costs alongside them.
Validate numerical resolution separately from recording uncertainty and practical
relevance. Do not replace the old threshold with a fitted margin around P80.

The next distinct model prerequisite remains direct feature phase relations versus
relations measured after raw/fused reconstruction. Keep kernels, original targets,
support rules and primary spectral weights fixed; first audit reference/weak
behavior, derivatives and domain-specific conflicts. A successful finite diagnostic
may justify an independently registered adequate-budget matched comparison even
without a statistically resolved two-update transfer gain. No unchanged scaling
retry, loss substitution, default promotion or automatic full training follows
from this metric revision. Runtime code, fixed buffering and compute requirements
remain unchanged; no new timing/device measurements were made.
