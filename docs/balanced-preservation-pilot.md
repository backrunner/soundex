# P53: corpus balance and batch-gradient calibration

## Question

[P52](physical-preservation-pilot.md) used correct original-reference PCM units,
but its physical auxiliary was almost entirely music. Equal pooled gradient norms
also gave unequal batch-gradient RMS strength. This pilot separates those two
design issues before changing capacity, spectral supervision or deployment DSP.
It does not assume that reducing a preservation penalty proves restoration.

## Registered experiment

Use both P50 persistent parents, the same small high-support adapter, original
lossless references, common spectral objective, 512 fit draws and 128 updates per
trajectory. Each update averages four batches. Fresh Adam at each trajectory start
retains moments thereafter: lr 1e-4, betas 0.9/0.999, epsilon 1e-8, clip five and
0.001 of the proposed displacement. No early stopping, veto or endpoint selection.

Train three arms per seed, six trajectories in total:

| Arm | Preservation term | Fixed initial gradient calibration |
| --- | --- | --- |
| `rms-physical` | Mean identity original-reference raw-OLA error power above 4 kHz | Batch-gradient RMS equals 10% of the common objective's RMS |
| `balanced-physical` | Separate music and speech means of the same PCM error power | Corpus component gradient RMS ratio 80:20; combined RMS equals 10% of common |
| `balanced-logit` | Separate corpus means of squared identity affine-head output | Same 80:20 component ratio and 10% combined RMS |

Reuse all three completed P52 arms for each seed as frozen controls: their starts,
data order, optimizer, update budget, objective and DSP are identical. Reuse is
explicit, not described as new training. Global physical versus P52 physical tests
RMS calibration. Balanced versus global physical tests corpus balancing under
matched combined initial RMS. Balanced physical versus balanced logit compares
physical-reference supervision and its proxy under the same calibration rule.
There is no global-RMS logit arm; do not claim a complete factorial comparison.

## Calibration definition and limits

Before any training, use only the original 32 fit draws at indices 0,16,...496.
For each term, collect its adapter-gradient vector on each draw at the unchanged
parent. Define its RMS as the square root of the mean squared vector norm, not
the norm of the mean vector. Corpus terms are means within that corpus, indexed
by recording metadata rather than assuming the first four rows are music.

For a balanced arm, first multiply each corpus term by its registered sampling
share (0.8 or 0.2) divided by that term's gradient RMS. Then multiply both by one
common factor so that the **combined** gradient RMS, including inter-corpus cross
terms, is 10% of the common spectral objective's gradient RMS. Freeze every
coefficient for the entire trajectory. Report component norms, combined norms,
per-batch ratios, pooled norms and directional alignment. A zero or exactly
cancelled calibration gradient stops preflight and is retained as a failure.

This matches initial aggregate strength, not audible fidelity, each individual
batch, the four-batch optimizer step, or future gradients. The final common factor
preserves the 80:20 component ratio but can make their individual norm fractions
exceed 8% and 2%. It does not equalize scalar loss values or speech/music errors.
Sampling proportions and spectral weights remain unchanged; this experiment does
not adopt per-clip normalization or tune an energy floor using known examples.

The physical loss remains calibrated PCM squared error against the original
unclipped reference, with one-sided Parseval weighting, no signal-level division,
and no DSP teacher. Its exact-reference zero, silence/weak-gradient behavior,
finite differences and supervision crop inherit the P52 checks. New tests cover
metadata permutation, corpus isolation, cancelling batch means, cross terms,
unit rescaling, real-model gradients, equality to a same-weight P52 update and
exact checkpoint/Adam resume. Neither clean logits nor frequency-band power is
a substitute for paired spectral shape, temporal detail or final native audio.

## Frozen review and deployment

Finish all six trajectories and freeze their hashes before endpoint fit/review
scoring. Keep common direct/MR spectral metrics separate from augmented training
totals. Report both corpora, original-reference PCM bands and every existing
16/20/24-bit counterexample against P38 starts, P50 parents, P51 controls and P52
arms. Do not loosen rulers or select a winner from one seed or one metric.

Use the same 32 fit and 32 review draws: 351 fit recordings and 986.623 unique
MR-supervised seconds; 83 review recordings and 71.706 seconds. These are reused
historical TRAIN crops with recording/source-disjoint roles, not independent test
audio, full-library coverage, convergence evidence or a capacity ceiling.

Export all six models with existing sequential/state parity. Each gets four
3,000-hop 44.1/48 kHz mono/stereo compute runs and one 1,000-hop paced 48 kHz stereo
run: 78,000 measured hops. Do not overlap timing with training or tests; retain
all scheduling delays. Replay the same eight native review batches, 480 new mono
roles, with hash-verified P50/P51/P52 controls including identity DSP. Raw and
native populations differ; only compare identical clips within each population.

The 480,809-parameter model, 49,794 trainable adapter parameters, causal history,
80-frame warmup, frame-89 supervision start, ABI and Rust DSP remain unchanged.
The fixed 256-frame buffer is 5.333 ms at 48 kHz / 5.805 ms at 44.1 kHz, separate
from the user's model-plus-DSP compute target below 5 ms. Per-hop throughput and
paced failures remain reportable. This loss experiment does not fix scheduling.

## Related work boundaries

[GradNorm](https://proceedings.mlr.press/v80/chen18a.html) dynamically balances
multitask gradients during training. It supports examining gradient magnitude,
but P53 is a fixed initial calibration, not an implementation of GradNorm or
evidence that its reported generalization results transfer to music restoration.
The 80:20 corpus ratio comes from this registered sampling schedule, not that paper.

[auraloss](https://github.com/csteinmetz1/auraloss/blob/main/README.md) documents
multiresolution STFT losses combining spectral convergence and log magnitude.
This reinforces evaluating more than scalar band energy; P53 retains the existing
common spectral objective rather than swapping its definition during comparison.

[DDSP](https://arxiv.org/abs/2001.04643) demonstrates integration of differentiable
signal-processing modules with neural synthesis. It does not imply that DSP output
should be the training target, nor establish real-time music decompression quality.
Here OLA is the differentiable rendering path, the target remains original PCM,
and unchanged Rust postprocessing is checked separately. These sources motivate
the controls; none supplies a quality guarantee or a <5 ms per-hop measurement.

## Evidence and status

Implementation: `training/evaluation/balanced_preservation.py` and
`training/scripts/run_balanced_preservation.py`. The runner is versioned separately
because prior registrations bind historical source hashes; it retains the tested
P52 training/review lifecycle without changing historical scoring code.

Local run: `data/runs/balanced-preservation-p53-20261024/`. Source snapshots, fixed
coefficients, optimizer histories, all endpoints and raw/native artifacts remain
ignored. No training audio, licenses or default distributed weights change.

Status: **complete / not qualified**. This is a bounded mechanism
comparison. Completion alone cannot qualify restoration, listening, stereo or
real-device performance. Do not repeat an unchanged unsuccessful penalty longer.

### Frozen calibration observations

The registered coefficients were computed before the first update. Both seeds
match combined auxiliary/common batch-gradient RMS at 0.1 to floating-point
precision. Balanced component RMS retains a 4:1 ratio; correlated component
vectors make their sum of norms different from the norm of their sum.

| Parent seed | Common gradient RMS | Global physical weight | Balanced physical music / speech weights | Balanced logit music / speech weights |
| --- | --- | --- | --- | --- |
| 12 | 0.622305279 | 976630.358916 | 771920.448768 / 6363425.311147 | 62.403054 / 11.351330 |
| 13 | 0.516853492 | 1429583.639530 | 1107070.368605 / 5543027.663961 | 62.395309 / 9.873076 |

For balanced physical, the initial component/common RMS fractions are
9.8855% / 2.4714% (seed 12) and 9.6780% / 2.4195% (seed 13). Their combined RMS
is 10%, not the sum of these fractions. Weighted scalar physical loss still gets
only 0.4926% / 0.1870% from speech; gradient balance is not scalar-loss balance.
Calibration batch auxiliary/common norm ratios range up to 0.421. These initial
32-draw statistics neither bound all training updates nor imply audible quality.

Local verification: 740 Python tests pass, including real checkpoint/Adam resume;
Ruff and 131-document checks pass. Remote
[CI 40090565722](https://github.com/backrunner/soundex/actions/runs/40090565722)
passes Rust and export-contract jobs. All six endpoints and their quality/runtime results are reported below.

## Completed spectral and preservation results

All six trajectories finish 128 updates, all 768 updates change parameters, and
the maximum pre-clip gradient norm is 3.1114 (below five). All twelve raw reports,
six parity-checked exports and 480 native mono roles finish. The registration binds
7,822 inputs and the completion receipt binds 8,863 files. There is no interrupted
run, selected early checkpoint, default-weight replacement or timing retry.

Percent changes below are relative to each unchanged P50 parent; negative is
better. These tiny changes are not percentage-point gains or listening evidence.

| Seed | Arm | Raw paired music MR | Native paired music MR | Raw paired speech MR | Native paired speech MR |
| --- | --- | ---: | ---: | ---: | ---: |
| 12 | balanced-logit | -0.0040505% | -0.0041521% | -0.0034836% | -0.0057102% |
| 12 | balanced-physical | -0.0006439% | -0.0005026% | -0.0007900% | -0.0012859% |
| 12 | rms-physical | -0.0004240% | -0.0002035% | -0.0007654% | -0.0011500% |
| 13 | balanced-logit | -0.0024613% | -0.0026081% | -0.0021937% | -0.0038580% |
| 13 | balanced-physical | +0.0018866% | +0.0023159% | +0.0013976% | +0.0028975% |
| 13 | rms-physical | +0.0020736% | +0.0025650% | +0.0014967% | +0.0031016% |

Both physical arms lose paired music MR benefit against the P52 no-auxiliary
control on both seeds. Balanced physical is slightly better than global RMS
physical, but still worse than the old P52 physical arm. Its music MR regression
versus the no-auxiliary control is +0.0037116% / +0.0043282%. Balanced logit is
close to the no-auxiliary control: +0.0003048% / -0.0000199%, not a consistent win.

The seed-13 balanced physical failure is also visible on fit data: paired music
MR +0.0010666%, speech MR +0.0022490% and common total +1.5015e-6 versus the parent.
On review, music direct amplitude is +0.0039010% while complex error improves
-0.0016367%; music MR dynamics still increases +0.0000555%. This is a spectral /
complex-preservation tradeoff, not evidence that extra capacity or an independent
generalization failure caused it. The seed-12 arm improves direct amplitude
-0.0020144% but worsens complex +0.0025438% and MR dynamics +0.0007458%.

Corpus balancing does improve the intended clean-speech preservation diagnostic:

| Seed | Arm | Clean music HF4k MSE versus P52 control | Clean speech HF4k MSE versus P52 control |
| --- | --- | ---: | ---: |
| 12 | balanced-logit | -0.9454% | -11.5840% |
| 12 | balanced-physical | -7.1050% | -19.9135% |
| 12 | rms-physical | -6.6947% | -8.4673% |
| 13 | balanced-logit | -0.0428% | -17.8260% |
| 13 | balanced-physical | -4.8960% | -3.2610% |
| 13 | rms-physical | -5.0283% | +15.5975% |

Balanced physical reduces speech HF4k MSE by 12.5344% / 11.8881% versus P52
physical. Unlike global RMS physical, it no longer worsens this metric against
the no-auxiliary control on seed 13. This is a real preservation improvement,
but does not recover the lost paired spectral benefit.

Absolute error prevents those percentages being mistaken for audible severity.
For balanced physical, mean raw clean-music full error RMS is 7.5623e-6 / 8.7874e-6
(-102.43 / -101.12 dBFS); speech is 8.0176e-7 / 4.6870e-7 (-121.92 / -126.58 dBFS).
These are 20 log10 of mean per-clip PCM error RMS, not FFT power or pooled RMS.
Native clean-music full RMS is 0.00286514 / 0.00286487; the unchanged identity-DSP
control retains its much larger processing/limiting contribution. Native speech
RMS is 7.2366e-8 / 7.4707e-8, increasing 2.4447e-9 / 8.9484e-9 versus the parent.
Keep original unclipped references; neither level proves a listening outcome.

Against untreated compressed music, native balanced physical improves aggregate
MR by 0.9767% / 0.5510%, while MR dynamics worsens 1.4789% / 1.4251% and high-band
PCM RMS worsens 2.4896% / 2.5252%. A waveform regression alone would not reject a
spectral improvement; the combination still lacks overall restoration evidence.
Raw review has 128 music / 32 speech examples per paired or identity role; native
review has 32 / 8. These populations must not be subtracted to attribute DSP.

### Precision rulers and retained counterexamples

All six arms pass every 16-bit ruler. Every review low4k ruler passes at all three
precisions. Fit has one 24-bit low4k peak failure for seed-13 global RMS physical:
case 112, example 1, track `music_library:1453597560-f8f4d240ebfc8e68`, source start
35,296. Peak error increases from 0.0615221369382 to 0.0615222207452, a change of
8.38069e-8 against allowance 5.96046e-8. Preserve the counterexample without
retuning its threshold or interpreting its small delta as a listening judgment.

Identity failure counts below each have denominator 160 per fit/review role.

| Seed | Arm | Fit 20 / 24-bit | Review 20 / 24-bit |
| --- | --- | ---: | ---: |
| 12 | balanced-logit | 68 / 136 | 72 / 136 |
| 12 | balanced-physical | 52 / 133 | 51 / 130 |
| 12 | rms-physical | 51 / 137 | 56 / 131 |
| 13 | balanced-logit | 39 / 106 | 24 / 96 |
| 13 | balanced-physical | 13 / 90 | 8 / 73 |
| 13 | rms-physical | 13 / 89 | 9 / 79 |

All identities, source-time positions and exact failed measurements remain in
`interpretation.json`; the 20/24-bit counts alone do not veto a spectral candidate.
No unchanged/bypassed audio is credited as restoration.

## Runtime failure and resource measurements

All 30 registered runs finish: 78,000 measured hops (81,000 including per-process
warmup). Maximum model-plus-DSP stage wall time is **7.403459 ms**, with **four
>=5 ms calls**, **six service overruns of the hop period**, and **14 paced late
completions** (maximum lateness **5.761414 ms**). The largest per-run p99 is
1.166118 ms. Low typical cost does not satisfy the requested tail/deadline target.

All >=5 ms outliers occur in paced 48 kHz stereo runs:

| Seed / arm | Maximum total (ms) | Largest contributing stage at maximum (ms) | Paced late completions |
| --- | ---: | --- | ---: |
| 12 / global RMS physical | 5.013375 | ORT 4.972875 | 4 |
| 12 / balanced physical | 7.403459 | ORT 7.361625 | 4 |
| 12 / balanced logit | 5.420708 | ORT 5.384333 | 3 |
| 13 / balanced physical | 5.055708 | Frontend 3.324375, ORT 1.709250 | 3 |

The maximum unpaced result is 0.604166 ms. This does not erase paced failures or
prove a scheduling cause. CPU-thread probes were off, QoS was applied and
explicit time-constraint scheduling was off in this diagnostic, as registered.
No SoundEx training/tests overlapped timing; ordinary host/application activity
was uncontrolled. The recorded wall stages cannot retroactively separate active
CPU work from descheduling, blocking or cache effects. Do not blame another app,
model capacity or loss balancing without attribution. This runner does not certify
the production callback/worker/queue or real-device xruns. See the retained
[P40 attribution limits](cpu-time-attribution.md) before another runtime study.

Rust peak process RSS is 29.594-30.141 MiB. Paced whole-process CPU/wall is
16.216-18.644% of one core, including initialization and warmup. Isolated warmed
ORT RSS increments are 13.297-13.406 MiB, excluding frontend/full Rust and not peak
memory. Each ONNX is 1,972,748 bytes (1.88 MiB); parameter counts are unchanged.
The fixed 256-frame buffer remains 5.333 / 5.805 ms at 48 / 44.1 kHz, separate
from the failed <5 ms processing target. No timing run is retried to pass.

## Post-completion update attribution and next gate

After every endpoint and native result was complete, a separate fit-only probe
reconstructed the **first real four-batch update** at each unchanged P50 parent,
with fresh Adam and the frozen P53 coefficients. No trained endpoint, coefficient,
threshold or budget changed. It compares actual float32 applied parameter steps,
not merely raw gradient norms. This one update does not represent later optimizer
states, all recordings, or establish the cause of the complete trajectories.

| Seed | Arm | Auxiliary/common gradient norm on this update | Difference from control step / control-step norm | Step cosine with control |
| --- | --- | ---: | ---: | ---: |
| 12 | balanced-logit | 19.937% | 44.496% | 0.901238 |
| 12 | balanced-physical | 10.183% | 25.864% | 0.966554 |
| 12 | rms-physical | 9.777% | 24.730% | 0.969423 |
| 13 | balanced-logit | 21.526% | 48.829% | 0.881041 |
| 13 | balanced-physical | 4.900% | 25.186% | 0.968288 |
| 13 | rms-physical | 4.800% | 24.844% | 0.969142 |

Step L2 norms remain approximately 2.02e-5. The percentages above describe vector
differences, not 25-49% inflation of the overall step norm.

All first updates reduce the common fit objective, including seed-13 physical
arms that regress at their completed endpoints. Matching calibration RMS does
not match a four-batch gradient, Adam's coordinate-wise preconditioning, the
finite applied update or cumulative transfer. This probe identifies a limitation
of the comparison metric, not a proved optimizer bug or permission to tune to
review scores. The spectral/complex tradeoff is already visible on fit data.

Do not repeat these penalties longer, increase model size, or infer that physical
reference targets intrinsically fail. Corpus balancing fixed an observed imbalance
and reduced clean-speech error, but it did not deliver the requested spectral win.
The next loss investigation should examine **post-Adam update contributions in
this frozen-feature adapter**, with unchanged spectral controls and fixed fit-only
calibration groups. Separate this from P43/P46's earlier Euclidean gradient/projection
studies; do not repeat their no-op gates or relax PCM rulers. Establish a useful
spectral direction before another sustained training comparison.

Independently, runtime investigation should instrument the actual worker/queue
under its production scheduling policy, separating CPU/service/wakeup/queue delay
and observing aligned fallback/recovery. Repeating this paced diagnostic until
it passes would not fix the measured problem. Keep all four >=5 ms samples.

Evidence includes `registration.json`, `calibration.json`, `review-registration.json`,
`complete.json`, `interpretation.json` and `initial-update-attribution.json`, with
separate analysis source/input/output integrity receipts. The new analysis is
explicitly post-completion and never used for candidate selection. Failed results
remain reproducible locally; production weights and dataset licensing are unchanged.
