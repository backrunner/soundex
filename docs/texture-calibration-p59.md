# P59: critic learning and false-detail calibration

Date: 2026-10-26. This follows the
[P58 attribution](joint-attribution-p58.md) and
[external implementation review/prototype](implementation-review-20261026.md).
It tests whether the small critic learns useful distinctions before adding its
feature loss to a restoration generator. No generator is trained in this study.

## Fixed protocol

Entry point: `training/scripts/calibrate_spectral_texture.py`.
The two completed P57 full generators stay frozen. Each critic starts with the
same seed and 2,818-parameter initialization. Each receives exactly 1,024 Adam
updates at 1e-4, betas 0.9/0.999, epsilon 1e-8, gradient clipping at 5. Every
update uses one paired batch of four music crops and one speech crop. No loss,
critic architecture, primary reconstruction objective or DSP policy changes.

First exclude entire P54 optimization draws containing any calibration or reserved
source/work alias; then select 256 evenly spaced draws from the eligible schedule,
before scores, retaining their exact crops, codecs, physical views and warmup.
Cycle this 1,280-crop bank four times. Repeated visits do not count as new audio.
Use all 16 P54 calibration draws, globally recording/work-alias-disjoint from
the critic's selected optimization material. The fixed P57 generators have
historically seen other crops of some calibration recordings or linked works.
All registered P55 source/decoded/work
aliases remain excluded. Both roles are reused TRAIN material; neither is a
fresh independent test. Registration reports actual unique supervised duration,
recordings, codecs, views, genres and regions, not catalog-size training coverage.

Finish both critic budgets and hash both endpoints before calibration scoring.
Retain random initial critics as controls, plus every source and failure. No
intermediate winner, early stopping, budget reduction or threshold tuning.
An endpoint at 1,024 steps does not establish convergence or a capacity ceiling.
The previous eight-step integration check is not a trained quality baseline.

## False-detail controls

Evaluate 16 fixed evenly spaced optimization draws and all calibration draws,
separately for music and speech. Each control uses the same observed conditioning
and the same paired original lossless reference:

1. Frozen generator output.
2. Untreated observation reconstructed through raw OLA, without deployment DSP.
3. Generator output scaled to match reference RMS separately in 8-12, 12-16 and
   16+ kHz bands. This uses oracle reference information and is only a diagnostic
   for a band-energy shortcut. Exactly absent predicted bands remain absent.
4. Reference with randomized phase above 8 kHz, preserving its whole-excerpt FFT
   magnitudes and its low band. Correct energy alone must not certify detail.
5. Reference with its high band circularly shifted by one-quarter excerpt,
   preserving spectral magnitudes while changing alignment with low-band events.
6. Reference FFT roundtrip, supplying a numerical no-op control.

The last three use a rectangular whole-excerpt FFT; circular edge behavior is
explicit and is not a deployable transform or a claim of listening severity.
DC/Nyquist stay real. Tests check fixed seeds, magnitude/band-energy preservation,
low-band isolation, silence, missing generation and calibrated PCM RMS units.
All bands retain absolute candidate/reference/error PCM RMS alongside critic
scores. Small errors on very weak material do not establish recovery or audibility.

The primary critic diagnostic is reference-minus-candidate logit margin. Report
paired feature distance, real/candidate logits and balanced logistic loss too.
Aggregate cases within a recording, then recordings equally within each corpus.
Random features and trained features have different scales: a larger distance
after learning alone is not evidence of better discrimination.

## Preregistered prerequisite for generator training

Every seed and corpus must meet both conditions on calibration:

- Prediction logistic loss improves over initialization and over the balanced
  constant classifier (`2 * ln(2)` under the sum-of-real-and-fake convention).
- Prediction, energy-matched prediction, phase-scrambled reference and time-shifted
  reference each have positive source-mean margin and a strict majority of sources
  above the magnitude of their reference-roundtrip margin.

This is a mechanism prerequisite, not a restoration quality gate. The roundtrip
comparison handles arithmetic noise; it is not an audible-error allowance.
A critic that recognizes deficient energy but accepts temporally incorrect detail
does not satisfy the proposed texture hypothesis. Such a result must be retained
and diagnosed without retrospectively dropping that control or a weak corpus.
Failure also does not reject every discriminator architecture or paired-feature
objective: ambiguous stochastic phase may not be recoverable from the observation.

## Gradient and optimizer checks

On the 16 fixed fit draws, report common-versus-feature gradient norms and cosine
separately for music/speech and paired/identity. Fix one shared auxiliary scale
across both seeds so its pooled fit gradient RMS is 10% of the unchanged common
objective. Freeze this diagnostic coefficient before any calibration gradients.

Also compute actual fresh-Adam displacements on copies of flattened generator
parameters, at the existing 1e-7 learning rate, for common alone versus common
plus the scaled auxiliary. Report displacement differences and directional
derivatives. Generator tensors themselves remain unchanged. As P53 established,
matching initial gradient norms does not match Adam displacement or establish
finite-step quality. This scale is not a validation-selected training weight.

Calibration gradients are reported only after the scale is fixed; they cannot
change it. Any subsequent generator experiment needs its own registered matched
budget, common-objective scoring, all-endpoint freeze, raw and Rust output review,
identity/phase/temporal/weak-band counterexamples and runtime/listening evidence.

## Execution and evidence

The first preparation attempt (`texture-calibration-p59-20261026`) was stopped by
source isolation before registration or any critic/generator update. Of the
originally selected 256 draws, 150 included aliases of calibration recordings.
Historical source-time separation is not recording separation. The failed source
snapshot and diagnostic remain. The corrected selection retains whole 4:1 draws,
excludes connected calibration/reserved aliases before evenly spaced selection,
and does not inspect quality scores to choose replacements.

The corrected registration is
`data/runs/texture-calibration-p59-20261026-v2/`. It retains 6,635 eligible original
draws and selects 256 of them. Actual critic fit exposure is 156 music recordings
and 49 speech recordings, 1,280 paired crops and **554.812 independent supervised
seconds** (9.247 minutes), excluding warmup and overlapping intervals. Calibration
uses 45 music and 16 speech recordings, 80 crops and 36.823 unique supervised
seconds. Each update retains the 4:1 crop ratio; the unique-recording counts are
not that sampling ratio. Four passes over the fit bank remain 554.812 seconds.

All nine existing codec configurations are represented. Physical views include
mono, left, right, mid and side; they are independent channel views, not a new
stereo training objective. Inherited catalog labels cover classical, blues,
electronic, rock, folk, country, ambient, jazz, hiphop and pop, plus unlabeled
music in fit. Calibration lacks labeled country, hiphop and pop; neither pool
contains a labeled metal crop. Region labels include China, Japan, Korea,
North America and Europe as well as unknown; this is inherited metadata, not a
new audit of musical origin or proof of full regional coverage.

The runner retains source/resource snapshots, source-file and generator hashes,
per-draw fixed-generator PCM cache receipts, both initial critics, optimizer
history, checkpoints every 64 steps, frozen endpoints, per-case review scores,
source-equal summaries and a completion receipt. `--resume` verifies source/input
bindings and optimizer continuity. All generated artifacts stay in ignored
`data/runs/`; original audio and checkpoints are not published.

There is no new deployed graph, added buffer or inference operator. No deployment
timing is rerun for a training-only critic. The existing <5 ms model-plus-DSP target,
2.667 ms hop service requirement and fixed 256-frame buffer remain applicable to
any future restoration checkpoint. Default weights remain unchanged.

## Completed result: prerequisite failed

Both critics completed all 1,024 updates, then froze before scoring. The 3,840
review records cover initial and trained critics, both generators, both roles
and all six controls. Generator updates: **zero**. Both original generator state
dictionaries remain unchanged. All four seed/corpus groups fail the registered
false-detail prerequisite; no generator comparison was launched.

The table counts calibration sources where the trained critic gives the original
reference a higher logit than the control, beyond its roundtrip numerical margin.
Columns are the two fixed generator seeds, separately for music and speech.
The sources are shared across seeds, not independent replications of the corpus.

| Control | Music 20261012 / 45 | Music 20261013 / 45 | Speech 20261012 / 16 | Speech 20261013 / 16 |
| --- | ---: | ---: | ---: | ---: |
| Frozen generator prediction | 42 | 41 | 16 | 16 |
| Band-energy-matched prediction | 38 | 36 | 14 | 14 |
| Reference with scrambled HF phase | **0** | **0** | **0** | **0** |
| Reference with shifted HF time | **4** | **4** | **1** | **1** |

Prediction classification loss does improve: music 1.386425 to 1.385757 /
1.385751, speech 1.386372 / 1.386374 to 1.385063 / 1.385095. All trained values
are below the balanced constant classifier's 1.386294. This small classification
gain does not establish restoration quality. The initial critics already rank
ordinary predictions correctly on the same 42/41 music and 16/16 speech sources;
they also fail every phase-scrambled source. Training does not repair that failure.

Trained reference-minus-control mean margins are negative for phase scrambling
(-0.002320 / -0.002330 music, -0.004486 / -0.004507 speech) and time shifting
(-0.0001801 / -0.0001807 music, -0.0002543 / -0.0002548 speech). This is a
classifier preference, not a listener preference. Energy-matched predictions
mostly remain distinguishable, so the evidence does **not** justify saying the
critic learns only band energy.

The proposed generator loss is paired feature distance, not the classifier logit.
It still penalizes both false-detail controls and is exactly zero at the reference:

| Corpus / seed | Prediction feature loss | Phase-scrambled loss | Time-shifted loss |
| --- | ---: | ---: | ---: |
| Music / 20261012 | 4.6246e-5 | 1.5245e-4 | 2.2855e-4 |
| Music / 20261013 | 4.6426e-5 | 1.5307e-4 | 2.2955e-4 |
| Speech / 20261012 | 7.4030e-5 | 3.5065e-4 | 5.1702e-4 |
| Speech / 20261013 | 7.0574e-5 | 3.5210e-4 | 5.1930e-4 |

Random initial features also penalize those controls. Larger learned feature
distances alone can be rescaling, not improved structure sensitivity. The current
result therefore neither demonstrates a useful learned texture objective nor
shows that the paired objective rewards noise. It fails the preregistered
mechanism prerequisite without disproving all feature-matching approaches.

For physical scale, source-mean 16+ kHz PCM error RMS is 0.001854 / 0.001852 for
phase/time controls in music and 0.001178 / 0.001172 in speech. Prediction errors
are 0.001027 / 0.001031 in music and 0.000508 / 0.000510 in speech. Roundtrip PCM
errors are below 3e-19 in music and 1e-18 in speech; its feature loss and margin
are exactly zero. These calibrated per-excerpt errors retain absolute scale,
but are not audibility thresholds or evidence of restored detail.

### Gradient scale and scope

The shared fit-only coefficient is 902.421492, matching pooled auxiliary gradient
RMS to 10% of the common objective. Calibration mean weighted auxiliary/common
gradient norm ratios are 0.1019 / 0.1021 for paired music and 0.2864 / 0.2968 for
paired speech. Identity ratios are below 0.0003. A single pooled coefficient
does not give equal per-corpus influence. It was frozen before these measurements.

On copied parameters, fresh-Adam displacement changes by a mean 26.01% / 24.95%
relative to common-only steps across the 16 fit draws. Common-loss directional
derivatives are downhill on every draw; texture derivatives are uphill on 3/16
and 2/16. Gradient RMS matching neither matches Adam displacement nor guarantees
improvement of every component. No finite-step generator audio result is claimed.

Local verification: **793 Python tests passed**, Ruff lint/format checks passed,
and the 3,403 registered input/source hashes, completion receipt and all four
initial/trained critic checkpoint hashes were reverified after completion. The
initial isolation failure remains separate from the completed calibration result.
No new raw data, external model code or weights were imported.

## Next changed hypothesis

The current critic only sees observed high-band conditioning and trains on real
references versus fixed-generator errors. It never learns from explicit
phase/time false-detail negatives, and cannot use the discarded low-band timing
and harmonic cues. These are plausible limitations, not isolated causal findings.

A subsequent registered study should separate full observed-band conditioning
from structured negative exposure under matched budgets, retaining the existing
critic and frozen random features as controls. Evaluate classification and paired
feature behavior separately, with original-reference stationarity, silence,
absolute PCM and source/corpus counterexamples intact. Reusing this calibration
pool will be a recheck. That changed study has **not** started; unchanged longer
training, larger restoration models and default weight promotion are unsupported.
