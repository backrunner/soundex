# P41: guarded updates and recording-disjoint training transfer

Completed 2026-10-23. Simultaneous gradient constraints improve the **local** tradeoff
but do not establish reliable transfer to other training recordings. At parameter
L2 radius 0.001, five metrics improve together on 19/32 source batches, versus
16/32 for the unguarded bisector; the corresponding transfer counts are 3/32 and
1/32. These strict counts describe local responses, not an audio-quality threshold
or a requirement that every stochastic update improve every recording. No new
trained weights, native output improvements or capacity-limit claims result.

## Changed hypothesis and registration

[P40](update-direction-audit.md) left two questions open: can the update explicitly
protect identity/dynamics, and does its benefit reach other recordings? P41 keeps
the same two frozen 431,015-parameter P38 controls, original loss definitions and
16 source draw indices per seed. Each batch contains four music crops, one speech
crop and their clean identities. This deliberately diagnostic 4:1 mix does not
change the project's production corpus sampling policy.

For each source draw, start half a training schedule cycle later and choose the
first draw sharing neither a track ID nor a source checksum. Selection uses no
loss or gradient values. The transfer material was seen during original training;
it is disjoint from its paired direction-estimation batch, **not an independent
test set**. Different source/transfer pairs can share recordings with one another.

Before scoring, freeze hashes, all 32 draw pairs, three directions and two L2
radii (0.001 and 0.01). Controls are the normalized recorded Adam update and P40
full amplitude/complex bisector. The candidate projects that bisector onto the
intersection `gradient(metric) dot direction <= 0` for all eight unchanged P40
metrics: total, amplitude, complex, paired MR, MR dynamics, identity MR, identity
risk and budget. Normalize the resulting direction for equal-radius comparisons.
There is no line search, best-radius selection, persistent optimizer update or
checkpoint save. Parameters and bound input hashes are verified unchanged.

## Method and its limits

The small solver enumerates active constraint subsets, solves their Gram systems
in float64, and checks primal feasibility, nonnegative multipliers and
complementarity. Zero and duplicate constraints are supported; a zero projection
stays zero. No diagonal regularization silently relaxes constraints. No case here
has a degenerate projection. After casting and normalization, the largest positive
normalized derivative is 2.68e-8 / 5.08e-8 by seed: numerical residue, not exact
floating-point nonpositivity.

The constrained quadratic formulation is related to
[Gradient Episodic Memory](https://arxiv.org/abs/1706.08840), which uses gradients
to protect past tasks locally. [PCGrad](https://arxiv.org/abs/2001.06782) instead
projects conflicting task gradients pairwise. Our independent simultaneous
constraint diagnostic is neither a continual-learning implementation nor a claim
that either paper proves music-restoration quality. The constraints protect only
first-order responses on the batch used to compute them.

Autograd uses the original detached 80-frame history. A central-slope control
holds that state fixed. Actual source and transfer responses recompute real
warmup after each parameter perturbation. The two measurements remain separate;
near-zero constrained derivatives are especially vulnerable to float32 finite-
difference cancellation. No loss implementation or reference target changed.

## Results: every direction and both radii

The five columns below count strictly lower direct amplitude, direct complex,
paired multiresolution spectrum, MR dynamics and identity MR error, respectively.
Each row covers 32 batches, pooling seeds only for counts. Full raw metrics,
including total, identity risk and budget, remain in the reports.

| Radius | Direction / evaluated batch | Amp | Complex | MR | Dynamics | Identity | All five |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.001 | Stored Adam / source | 32 | 6 | 32 | 15 | 5 | 4 |
| 0.001 | Bisector / source | 32 | 32 | 30 | 30 | 17 | 16 |
| 0.001 | Guarded / source | 31 | 32 | 31 | 31 | 22 | 19 |
| 0.001 | Stored Adam / transfer | 26 | 3 | 29 | 12 | 6 | 1 |
| 0.001 | Bisector / transfer | 20 | 14 | 21 | 21 | 12 | 1 |
| 0.001 | Guarded / transfer | 16 | 19 | 18 | 21 | 14 | 3 |
| 0.01 | Stored Adam / source | 32 | 6 | 32 | 15 | 5 | 4 |
| 0.01 | Bisector / source | 32 | 31 | 30 | 30 | 17 | 15 |
| 0.01 | Guarded / source | 32 | 31 | 31 | 30 | 17 | 15 |
| 0.01 | Stored Adam / transfer | 26 | 3 | 29 | 12 | 6 | 1 |
| 0.01 | Bisector / transfer | 20 | 14 | 20 | 20 | 12 | 1 |
| 0.01 | Guarded / transfer | 16 | 18 | 17 | 21 | 14 | 2 |

At radius 0.001, guarded transfer amplitude has median changes of −0.000286% /
−0.000429%, and complex error −0.002342% / −0.001056%, for seeds 20261012 /
20261013. These tiny local changes are insufficient evidence of useful restoration.
Identity MR changes by median −0.1769% / +0.8764%; the corresponding median
absolute dimensionless changes are −4.09e-9 / +4.92e-7. These are not PCM dBFS or
audibility estimates. Percentages and absolute changes have different rankings
because each example has a different baseline.

Even with **fixed history**, guarded source identity MR increases on 9/32 small-
radius probes; with recomputed history it increases on 10/32. Thus warmup changes
alone cannot explain all failures of finite-step protection. Local curvature and
finite precision remain distinct from a first-order feasibility certificate.

Retained concrete counterexample: seed 20261013, source draw 2158, transfer draw
10254, guarded radius 0.001. Amplitude increases 0.06329%, complex 0.01555%, MR
0.02815%, dynamics 0.00744% and identity MR 3.4602%. Identity MR starts at
0.000402942 and increases by 0.0000139427. All other counterexamples, unchanged
values and both radii remain in the local reports; no post-hoc tolerance removes
them. This example does not establish audible damage without native PCM/listening.

## Coverage and interpretation

| Seed / role | Paired crops | Distinct recordings | Unique MR PCM support | Direct analysis support |
| --- | ---: | ---: | ---: | ---: |
| 20261012 / source | 80 | 76 | 37.253 s | 37.702 s |
| 20261012 / transfer | 80 | 74 | 36.901 s | 37.346 s |
| 20261013 / source | 80 | 72 | 36.748 s | 37.186 s |
| 20261013 / transfer | 80 | 75 | 36.627 s | 37.068 s |

Durations union source-time intervals across codecs/channel views within each
row, exclude warmup, and do not double-count clean identity copies. Direct STFT
support and the stricter MR PCM region differ: after accounting for the 128-sample
leading pad, their source hop intervals are `[88, 256)` and `[88, 254)`. MR analysis
windows overlap within that PCM region. Durations are not independent recording
counts and must not be summed across rows as globally disjoint material.

All four groups cover AAC, MP3 and Vorbis recipes, mono/left/right/mid/side views,
and music tagged China, Japan, Korea, Europe, North America and unknown. Each group
contains 64 music and 16 speech crops, plus identity copies. Genre coverage is
sparse: country appears only once in seed-12 source and seed-13 transfer; blues,
funk and hip-hop have gaps. Exact counts and sample intervals are retained. These
views do not form synchronized stereo tests, and roughly 37 seconds per group is
not full-library training or balanced regional/genre evidence.

## Decision, verification and evidence

The guarded direction is **diagnostic / not qualified**. It improves local conflict
handling but does not show consistent cross-recording spectral gains. This does
not prove that multi-step constrained training would fail, nor that the architecture
is at its capacity limit. Do not launch an unchanged Adam continuation or promote
the guarded direction from source-batch counts alone.

The next justified optimizer study would estimate gradients over multiple distinct
training batches and explicitly measure finite-step preservation, with separate
unused training draws for transfer. Freeze those roles and budgets before scoring;
do not tune guards to these counterexamples. A subsequent matched small-model
trajectory needs all endpoints finished and common native/spectral/fidelity review.
No actual Rust restoration, new latency benchmark or listening test occurred here.
The unresolved [worker scheduling question](cpu-time-attribution.md) remains open.

Entry points are `training/scripts/audit_guarded_transfer.py` and
`training/scripts/summarize_guarded_transfer.py`. Evidence is retained in
`data/runs/guarded-transfer-p41-20261023/`: registration, original checkpoint hashes,
source snapshots, 32 complete case records, log, summary and `analysis.json`.
The first post-hoc coverage summary used nominal frame-hop lengths; it is preserved
as `analysis-nominal-frame-accounting.json` with its executed source. Current
coverage distinguishes exact physical MR support and direct STFT support; no
metric values or probes changed. See the [failure registry](failed-experiments.md).

The first full local test run retained 667 passes and two failures: the default
FFmpeg executable could not load its old x265 ABI during mono/stereo resampling
fixtures. This is an environment failure, not a changed loss or codec recipe.
The follow-up passes **669 tests**, using the existing functional FFmpeg 9.0.2 only
in process-local PATH; existing ONNX/tracing and MPS warnings remain.
see [environment guidance](../training/scripts/environment.md). Both test logs
remain in the evidence directory. Ruff and all 119 English-documentation/link
checks pass. The CI dependency-install step now has a ten-minute timeout and
reuses existing executable tools, then verifies both binaries start. The prior
head's delayed CI run eventually completed successfully; it was not canceled.
