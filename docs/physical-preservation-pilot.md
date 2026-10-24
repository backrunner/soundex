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

Status: all 736 Python tests, Ruff and 130-document checks pass. The registered
run binds 5,697 inputs and has started fit-only calibration, followed by all six
trajectories and frozen reviews. No trained result or restoration qualification
is claimed yet; remote CI and the complete experiment remain pending.
