# P13: Fixed Segments and Multi-Position Sampling from Same Recording

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

This round of evaluation addresses the generalization issues exposed in P12 regarding fixed segment performance. The experiment modifies only the training position coverage, without expanding model size or altering the loss function.
This is a diagnostic study, not a full library training or a weight release.

## Predefined Protocol

- 40 training audio segments from the same P12 batch: 32 music, 8 speech; each step consists of 4 music + 1 speech, with a corresponding clean identity at the same position.
  Diagnostic risk is 80% music / 20% speech; the 253:1 sampling ratio for the full library remains unchanged.
- Two groups:
  - `fixed`: each segment uses only the original anchor position;
  - `multiposition`: each segment includes the original anchor and 7 additional fixed positions, with balanced rotation.
  The limited position library does not allow infinite random slicing at each step.
- Identical 807,398-parameter model, input phase, P11 loss, Adam optimizer (1e-3), no weight decay, gradient clipping at 5, gain=0, codec, audio order, and batch size.
  Seeds: 20261012 / 20261013.
- Each group runs 2,048 steps per seed; the fixed group reports every 256 steps, with no development set selection points.
  Each segment is exposed 256 times in total; in the multiposition group, each position is seen 32 times, and at 256 steps, each position is seen 4 times.
- New models are created; both groups are trained from the same source code snapshot. After training completion and checkpoint hashing, scoring is performed.

## Position Isolation and Scoring

For each segment, a 32,768-sample window is reserved. The endpoint is selected as the farthest one that does not intersect with the anchor’s supervised region.
The supervised region is defined as `[crop_start + 80*128, crop_start + 32768)`, covering actual frame 89 onward in spectral, waveform, and time-differenced reference support.
No sample from any training position’s supervised region may overlap with the reserved window.
The remaining valid start positions are selected at equal intervals, yielding 7 positions, without selection based on loss or listening preference.

Library audit: 320 positions have zero supervision overlap; 27 positions show input or warmup context overlap.

For short recordings, input and warmup contexts may still overlap with the reserved window; these are recorded individually in the config audit.
This is not equivalent to independent segment holdout. For two short recordings, the reserved endpoints differ from P12, so the shifted total score from P12 cannot be directly compared to P13.
P13 uses identical reserved endpoints for both groups.

All 40 original anchors, 40 reserved positions, and 89 reused development segments are scored.
Music and speech scores are averaged separately, then combined at 80/20 ratio.
Each scoring group is evaluated with half of the paired recovery and half of the identity components.
The development set remains reused for validation — not an independent test, and no overfitting proof is provided.

Each scoring group performs real-time Rust DSP playback using the exact 4-music + 1-speech anchor and its identity.
Total of 300 real-time replays (30 baseline + 4 checkpoints × 30) are conducted.
Additionally, a proxy score is recorded for the multiposition training library as a whole, though it is not directly compared to the fixed group's mean across different samples.

## State and Reproducibility

Four training runs, 300 real-time Rust DSP replays, and automated pairwise comparisons have been completed. Default weights are not replaced.
The run completed at 09:33:50 UTC (Singapore time: 17:33:50) on 2026-10-13.
Source code snapshot: `279d7548222f75b7b3b4281116ad34c789650d70`.
Run root directory: `data/runs/crop-p13-20261013/`. All original audio, cache, checkpoints, and reports are excluded from submission.

Both groups used:
`training/scripts/run_subset_study.py --sizes 40 --crop-study`
with arguments `--crop-positions 1` and `--crop-positions 8`, sharing the same parent config, Rust detector/replay, and ONNX replay contract.
After training, the following command was executed:

```sh
PYTHONPATH=training training/.venv/bin/python training/scripts/analyze_crop_study.py \
  --study data/runs/crop-p13-20261013/study
```

No development set was used to modify configuration prior to checkpoint fixation.
During evaluation, paired recovery, identity degradation, and high-frequency spectral convergence, transients, energy, and low-frequency behavior were observed, along with consistency across seeds.
Model architecture and real-time execution path remain unchanged.
No new real-time pass or blind listening verdicts are generated in this round.

Verification: 8 real-data smoke tests per group were fully executed — training, Rust replay, and hash-verified comparison summaries.
427 local Python tests and Ruff lint/format checks passed.
[CI for frozen version](https://github.com/backrunner/soundex/actions/runs/38375024429) passed.
For the fixed group, seed 20261012 at step 256 matches the P12 checkpoint exactly (maximum difference: 0), confirming no drift due to extended sampling tools.
Final status is recorded in `pipeline.json`, logs in `pipeline.log`.
Re-execution of the frozen version summary script confirms all checkpoints, scores, sampling tables, and completion receipts pass hash validation.

## Official Results

The following scores are for fixed, common objectives — lower is better.
Note: These scores do not represent audio quality recovery percentages.
The exposure count per position differs between groups, as this is a deliberate component of coverage variation.

| Seed | Step | Original Anchor: Fixed / Multi-position | Reserved Position: Fixed / Multi-position | Cross-Recording Dev Set: Fixed / Multi-position |
| --- | ---: | ---: | ---: | ---: |
| Initial Model | 0 | 0.279887 | 0.286961 | 0.271475 |
| 20261012 | 256 | 0.279933 / 0.282780 | 0.292147 / 0.293856 | 0.338419 / 0.305075 |
| 20261013 | 256 | 0.277783 / 0.405433 | 0.291418 / 0.334142 | 0.341236 / 0.415565 |
| 20261012 | 2048 | 0.261841 / 0.269019 | 0.330395 / 0.286107 | 0.511675 / 0.408393 |
| 20261013 | 2048 | 0.261074 / 0.270330 | 0.317800 / 0.286693 | 0.446079 / 0.394683 |

The multiposition group shows a relative reduction of 20.18% / 11.52% in development set scores compared to the fixed group, but still remains 50.43% / 45.38% above the initial model.
The reserved position score drops only by 0.30% / 0.09% — essentially unchanged from baseline — and cannot be interpreted as a robust breakthrough.
At 256 steps, the two seeds show opposite trends in development set performance, so it cannot be claimed that multi-position sampling performs better under all budgets.
No single seed or checkpoint should be selected based on these results.

### Music and Speech

The cross-recording development set consists of 33 music segments / 26 recordings, 56 speech segments / 54 recordings.
The table shows per-linguistic mean scores. Final combined score is still weighted 80% music / 20% speech.

| Corpus | Initial Model | Multi-position (2048, seed 20261012) | Multi-position (2048, seed 20261013) |
| --- | ---: | ---: | ---: |
| Music | 0.294182 | 0.466435 | 0.449274 |
| Speech | 0.180645 | 0.176225 | 0.176322 |

Speech shows a ~2.4% drop in common score, while music performance deteriorates significantly.
Paired recovery error increases from initial 0.588365 to 0.734712 / 0.719533; identity degradation increases to 0.198158 / 0.179015.
Thus, the degradation is not solely attributable to identity weights.

Post-hoc per-fragment analysis:
- Seed 20261012: 24/33 music segments have worse scores than initial;
- Seed 20261013: 25/33 music segments have worse scores than initial.
Three specific segments contribute approximately 78.1% / 76.6% of the net error increase — two blues recordings and one unlabeled segment.
These are diagnostic signals, not grounds to conclude a genre failure, delete these segments, reset thresholds, or declare success without further validation.
The count of such fragments should not be equated with the number of independent recordings.

### Real DSP Validation

All 300 real-time replays completed. Maximum absolute error between proxy and Rust waveform is
`1.430511474609375e-6`, below the predefined tolerance of `2e-5`.
This confirms proxy-to-proxy alignment, but does not validate audio quality or real-time performance.

For the multiposition final model, using only the original 4 music + 1 speech and its identity, the output compared to the baseline identity model:

| Native Metric Change (lower is better) | Seed 20261012 | Seed 20261013 |
| --- | ---: | ---: |
| Paired high-frequency spectral convergence error | -0.004476 | -0.004637 |
| Paired high-frequency transient error | +0.003048 | +0.001569 |
| Identity high-frequency spectral convergence error | +0.054860 | +0.022179 |
| Identity high-frequency transient error | +0.058260 | +0.024531 |

Note: This native subset consists only of the original 4 music + 1 speech and its identity — not the full 89 development segments.
Local improvements in high-frequency energy and spectral shape are accompanied by transient and identity degradation.
The experimental state remains unchanged.

## Judgment and Next Steps

Expanding training position coverage within the same recording helps mitigate degradation in fixed segment fitting, but does not resolve the issue of cross-recording music generalization.
The problem cannot be entirely attributed to fixed segments, nor can it be concluded that model capacity is insufficient.
No new blind listening conclusions or inference performance certifications are generated.
Model structure and real-time execution path remain unchanged.

Next steps:
Prioritize auditing the spectral scale of failed music segments, weak high-frequency normalization, and misfilling behavior in relation to training coverage.
On the training side, investigate whether similar patterns exist.
Distinguish between improvements in scale/stability and representation capability before registering any single-factor experiments.
Retain all development samples, original lossless references, and identical evaluation metrics.
Do not extend current model training, nor establish genre-specific rules based on a small number of development counterexamples.

Full numerical results: `study/crop-comparisons.json`
Per-group files: `scaling-comparisons.json`, `evaluation-256.json`, `evaluation-2048.json` contain per-fragment and native metric details.

Both real-data preflights use 8 updates. The fixed arm at step 256 reproduces the corresponding P12 checkpoint with maximum tensor difference 0. The final endpoint is step 2048; its lower training error does not establish cross-recording restoration.
