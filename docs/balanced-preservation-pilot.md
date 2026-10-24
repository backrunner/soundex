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

## Evidence and status

Implementation: `training/evaluation/balanced_preservation.py` and
`training/scripts/run_balanced_preservation.py`. The runner is versioned separately
because prior registrations bind historical source hashes; it retains the tested
P52 training/review lifecycle without changing historical scoring code.

Local run: `data/runs/balanced-preservation-p53-20261024/`. Source snapshots, fixed
coefficients, optimizer histories, all endpoints and raw/native artifacts remain
ignored. No training audio, licenses or default distributed weights change.

Status: **registered protocol; results pending**. This is a bounded mechanism
comparison. Completion alone cannot qualify restoration, listening, stereo or
real-device performance. Do not repeat an unchanged unsuccessful penalty longer.
