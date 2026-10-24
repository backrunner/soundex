# P54: common baseline comparison and actual spectral training steps

## Questions and boundaries

P48-P53 were bounded adapter mechanisms, not cumulative full-model upgrades.
P49 already showed that a large fit-admissible reset step does not necessarily
transfer. P53 showed that initial gradient-norm calibration does not fix Adam's
actual direction. Do not repeat those probes or add unchanged preservation terms.

This work first compares historical `shape-small`, both P38 seeds, the P52
no-extra-preservation controls and both P53 logit endpoints on identical PCM.
The eight batches are fixed before scoring: P50 review indices 0,4,...28, with
32 music and eight speech mono views plus their clean identities. They are reused
historical TRAIN recordings, not a new independent test. Different inherited
training budgets prevent architecture-superiority claims.

The historical ONNX graph gets only I/O renaming, unused context/condition inputs
and a dummy pass-through state. Original model computation is unchanged, with
exact output/state parity checks. All eight native paths, including identity,
run through the same frozen Rust context replay, ModelGain, SmoothKnee, detector,
warmup and fusion. This diagnostic wrapper is not a production ABI migration.
It tests the old weights under current DSP, not every historical application mode.
Model-only raw OLA and native output are scored against original unclipped PCM on
the same supervised support. Raw low-band outputs of an older DSP-coupled model
must not be confused with its final audible output.

Evidence: `data/runs/common-comparison-p54-20261024/`; no source audio is committed.

### Completed common-material results

All eight native paths completed 80 roles each (640 renders); all raw/native
scores and model hashes are frozen. Lower is better. The two P53 logit seeds
are shown separately; neither was selected as a winner.

| Native metric | Historical shape-small | P53 seed 12 | P53 seed 13 |
| --- | ---: | ---: | ---: |
| Paired music MR | 0.38640593 | 0.38644372 | 0.38810029 |
| Paired music reference dynamics error | 0.21573797 | 0.20633788 | 0.20622839 |
| Paired music auraloss-band score | 0.39116654 | 0.39318929 | 0.39398194 |
| Paired speech MR | 0.32926609 | 0.31322717 | 0.30929670 |
| Clean music MR | 0.07596002 | 0.03265372 | 0.03272669 |
| Clean music full PCM error RMS | 0.00324538 | 0.00286522 | 0.00286493 |

Against old weights, P53 music MR is 0.0098% / 0.4385% worse, while music dynamics
error is 4.3572% / 4.4079% lower and speech MR is 4.8711% / 6.0648% lower. Music
amplitude error is 0.7020% / 0.2544% lower but log-shape error is 6.0558% / 6.7449%
higher. This identifies a spectral-detail tradeoff, not a universally superior
baseline or a demonstrated architecture ceiling. Mean percentages do not replace
recording-level errors, fresh tests or listening.

The 32 music clips represent 25 distinct source recordings; eight speech clips
represent eight sources. Averaging within source first, music MR improves/worsens
on 13/11 sources for seed 12 and 12/12 for seed 13 (one tie each); its mean difference
changes with weighting, so the tiny clip-average gap is not robust superiority.
Music dynamics improves on 23/25 sources for both seeds, but log-shape worsens on
14/25 and 15/25. These counterexamples remain in the per-example scored reports.

DSP is material on exactly these clips: untreated music MR is 0.39026966,
identity-through-DSP is 0.39209190, P53 raw is 0.38355709 / 0.38524007 and P53
native is 0.38644372 / 0.38810029. On clean music the identity-DSP full PCM RMS
error is already 0.00286440, close to P53's native values above; raw P53 error is
7.85e-6 / 7.57e-6. This isolates an existing processing contribution, not a license
to alter the clean reference, eliminate protection, or attribute every error to
one DSP stage. The original targets remain unclipped.

Older raw outputs are much worse before fusion (music MR 0.71025706). This is a
diagnosis of the older DSP-coupled output path, not evidence that its final output
is equally poor. The shared native pipeline is the relevant delivered comparison.

The P52 no-extra-preservation controls and P53 logit endpoints remain nearly
identical in paired spectra, supporting retirement of the unchanged preservation
penalty route. These are alternate experiments, not additive upgrades.

The comparison covers nine codec profiles, both 44.1/48 kHz rates and all five
mono/channel views. Music includes ten reported genres but no hiphop, with only
one China-labelled clip and seven unknown-region clips. These small strata do
not support regional or genre-wide conclusions.

### Answers supported by the evidence

| Question | Answer and next discriminating evidence |
| --- | --- |
| Have recent models surpassed historical weights? | Partly: speech, music dynamics and clean fidelity improve; music spectral shape does not. Overall superiority needs fresh recording tests and listening. |
| Are recent modifications cumulative quality improvements? | No. Most are alternate mechanisms or measurement fixes. P52/P53 paired differences are tiny; preserve useful controls rather than stacking penalties. |
| Is DSP fusion itself wrong? | Not established. Same-material identity DSP isolates material processing error, while model-only supervision stays anchored to original PCM. Native rendering remains required; stage-specific DSP attribution is still needed. |
| Is small-model capacity exhausted? | Not established. Current continuation trains only a linear adapter on frozen features. Broader exposure and sufficient matched optimization precede a separate nonlinear-decoder/unfreezing fit-capacity study. |
| Is the old baseline overfit? | This repeated historical TRAIN comparison cannot answer that. Unequal inherited budgets and prior selection remain confounders; source-disjoint fresh evidence is required. |
| Is the loss mathematically valid and useful? | Reference/gradient/isolation checks pass. Training usefulness is a separate empirical question; both objectives retain common scoring and endpoint barriers. |
| Is real-time delivery solved? | No. This quality comparison is not a timing test. P53's wall-time tail and actual worker/device qualification remain open, independently of spectral gains. |

## Auraloss integration

Pin [auraloss 0.4.0](https://pypi.org/project/auraloss/0.4.0/), Apache-2.0, for
training only. `models/auraloss_spectrum.py` imports the upstream
`SpectralConvergenceLoss` and `STFTMagnitudeLoss`. This is deliberately **not**
stock `MultiResolutionSTFTLoss` and is versioned `soundex-auraloss-bands-v1`:

- Retain 256/1024/2048 Hann windows, quarter-window hops, no centering or padding.
  Windows remain entirely inside the existing supervised raw-OLA crop. They do
  not add inference lookahead or change the 256-frame output buffer.
- Retain physical bands 0-4/4-8/8-12/12-16/16+ kHz; low-band weight .25 and each
  upper-band weight 1, normalized by 4.25. Average resolutions equally.
- Use one-sided Hann/Parseval PCM-RMS scaling and reference-only smooth power
  floors: `1e-4 * reference_full_power + 1e-12`, divided across each band's bins.
  Compute magnitudes as `sqrt(power + per_bin_floor)`. This differs from upstream
  hard-clamped unnormalized FFT magnitudes; it avoids silent denominators and a
  hard-clamp zero-gradient region. Floors are not audibility tolerances.
- Call upstream convergence and log-L1 separately for each example/band, then
  combine with equal weights. Upstream convergence reduces its entire tensor;
  feeding a whole batch would couple loud music and quiet speech denominators.
- No target gain normalization, mel weighting, DSP teacher or phase removal from
  the other supervision terms. New tests check exact-reference zero/gradient,
  silence, weak signals, finite differences, target detachment and batch isolation.

The experiment replaces only the rendered MR term in the training objective.
The direct clean-frequency amplitude, complex, energy and real temporal-change
terms stay fixed, as do equal paired/identity reductions and 4:1 music/speech
draws. The frozen generation-budget term has zero adapter gradient and is omitted
from optimization, but remains in the unchanged historical common scorer.
Never compare totals across the two objective definitions as improvement.

## Actual-step calibration and registered continuation

`scripts/run_spectral_training.py` uses genuine Adam updates, without multiplying
the displacement after `optimizer.step()`. Adam settings are betas .9/.999,
epsilon 1e-8, clip 5; every update averages four batches and retains moments.
Record actual float32 displacement, pre-clip gradient norm, gradient/displacement
inner product, changed parameters and learning rate. Unit tests compare directly
with PyTorch Adam and verify exact optimizer-state continuation.

Use the same two P50 persistent parents, frozen P38 features and high-support
49,794-parameter adapter. Do not increase capacity in this comparison.

Before any outcome, partition the existing P50 fit recordings into optimization
and TRAIN calibration roles by connected recording/source aliases and a fixed
hash. Keep the entire old review role excluded. Draw new source-time crops from
eligible processed pairs instead of recycling the 512-draw finite bank. The
registration records exact crops, codecs, channel views, regions, genres and
unique supervision seconds; this does not expand the underlying source library.

The registered full optimization schedule contains 81,920 paired crops from
278 recordings (157 music, 121 speech), with 17,251.178 unique MR-supervised
seconds, about 287.52 minutes after merging source-time overlaps and excluding
warmup. This is scheduled exposure, not completed training coverage. It spans
nine codec profiles and left/right/mid/side/mono views; independent views are
not simultaneous stereo training. TRAIN calibration samples 61 recordings and
36.823 unique seconds. Its selected music crops omit country, pop and hiphop;
those remain present in optimization but this calibration cannot establish their
generalization. Source-region labels cover China, Japan, Korea, Europe and North
America plus unknown labels; they do not certify cultural/style completeness.

Calibration has 12 fixed trajectories: two parents x two objectives x learning
rates `1e-7, 1e-6, 1e-5`, each 16 persistent updates from a fresh copy of its parent.
Freeze all endpoint hashes before scoring 16 recording/source-disjoint TRAIN
calibration batches. Choose **one shared rate** minimizing the worst endpoint/start
own-objective ratio across both seeds and both objectives, requiring all four
ratios below one. Ties choose the smaller rate. Retain common spectral, phase and
identity regressions; they are not hidden behind this optimization criterion.
If no shared rate descends, record the failure and diagnose it without silently
changing the rule or launching longer training. This is not a generalization gate.

After calibration, reset all four runs to their parents and fresh optimizer
states. Each receives the same selected rate, **4096 actual updates**, and
16,384 common broad-crop batches. The 32-fold continuation budget and increased
unique source exposure address the previous short/fixed-bank limitation; they do
not establish convergence or fully retrain the frozen backbone. No intermediate
review, early winner selection or budget shortening. Finish all four endpoints
before common raw and native review. Convergence needs loss/update trends and
coverage evidence, not merely reaching the step count.

The old and auraloss objectives receive identical architecture, start, optimizer,
rate, data order and budget. This separates loss definition from capacity and
learning-rate changes. Further decoder changes require a separate fit-capacity
diagnosis. No candidate is automatically promoted.

## Status and remaining work

Common comparison and all 12 calibration trajectories completed. All 192 updates
changed parameters, maximum pre-clip gradient norm was 3.1577 (below five), and
all endpoints froze before calibration scoring. The fixed shared-rate rule selects
**1e-7**. The worst endpoint/start own-objective ratios over both objectives/seeds
are 0.999996834 at 1e-7, 1.000026896 at 1e-6 and 1.001649878 at 1e-5.

Larger rates improve some paired spectral means but lose on the calibration
objective, including clean-identity costs. Removing the old post-step multiplier
does not magically permit larger updates: 1e-7 is approximately the effective
rate of the old 1e-4 proposal times .001, with different float32 rounding. The
meaningful new tests are a controlled loss change, persistent measured updates,
broader crop exposure and a 32-fold continuation budget, not a claimed optimizer
bug fix. Short calibration alone does not establish auraloss superiority.

At the selected rate, paired common MR changes on TRAIN calibration are
-0.00269% / -0.00082% for the common objective and -0.00298% / -0.00080% for
auraloss (seeds 12/13). Identity MR still rises; retain these tradeoffs. The
selection rule is an optimization screen, not a fidelity or release gate.

The four registered 4096-update trajectories have been launched with an idle-sleep
assertion. Evidence: `data/runs/spectral-training-p54-20261024/`, including
`supervisor.json`, `progress.json`, `calibration.json` and completion receipts.
Full training, frozen endpoint quality and final timing are **pending**. The
supervised completion command runs all raw/native reviews and then 52,000
diagnostic timing hops only after every endpoint finishes; no concurrent training
inside that process and no retry-to-pass. A running process is not a quality result.

Local verification: 751 Python tests pass (228 existing warnings), Ruff passes,
132 English Markdown documents pass their links/registry checks, and the resolved
Python environment passes `uv pip check`. The actual legacy wrapper also passes
exact feature/state parity and all 640 native render roles finish. Source and
dependency changes do not change the released artifact contract or default weights.

Independent recording evaluation, blinded listening, simultaneous stereo and
real-device qualification remain outstanding. These runs do not resolve P53's
7.40 ms diagnostic wall-time tail or establish its cause. Actual worker/queue
instrumentation is a separate runtime task; no benchmark retry is counted as a fix.

Commands from `training/`:

```bash
python -m scripts.compare_historical_spectra \
  --prior ../data/runs/balanced-preservation-p53-20261024 \
  --legacy ../data/reports/architecture-study-20261011/shape-small/best.onnx \
  --output ../data/runs/common-comparison-p54-20261024
python -m scripts.run_spectral_training \
  --prior ../data/runs/balanced-preservation-p53-20261024 \
  --output ../data/runs/spectral-training-p54-20261024 --calibrate-only
```

For the registered continuation plus automatic frozen review and final timing:

```bash
python -m scripts.complete_spectral_training \
  --prior ../data/runs/balanced-preservation-p53-20261024 \
  --output ../data/runs/spectral-training-p54-20261024
```

It runs only after the shared-rate rule passes. Resumption verifies registration,
package versions and checkpoint/optimizer hashes. Failed diagnostics and original
source snapshots stay in ignored evidence directories.
