# P17: Consistency Training with Learning Rate Warmup Comparison

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

## Pre-Run Protocol

P16's second seed degraded on the full training corpus across both primary and auxiliary objectives, as well as the combined objective, when stepping from 256 to 2,048 steps.

This round focuses solely on evaluating whether a small learning rate update can mitigate this optimization issue. No expansion of training scale, no modification of loss function, and no extension of the original training duration are performed.

- **Starting point**: The fixed `step-256.pth` checkpoint from P16 consistency runs on seeds 20261012 and 20261013.
- Full model restoration, including Adam first- and second-order momentum states, step counters, and random state initialization; only the learning rate is modified.
- For each seed, compare `lr=1e-3` vs. `lr=1e-4`, each running for 256 steps to reach a total of 512 steps; four distinct runs in total.
- Continue from `crop_draws[256:512]`, maintaining auxiliary coefficient at 0.1365237430457469.
- Maintain 40 audio recordings × 8 positions, 4 music + 1 speech, and corresponding identity labels, original model, loss, and `clip=5`.
  This is a diagnostic sampling rate and does not alter the official training corpus's speech mixing rules.
- Evaluate all 320 training positions at total steps: 256, 320, 384, 448, and 512; each position assigned equal weight.
- Record paired recovery, identity, primary objective, auxiliary objective, and combined objective per position, preserving per-position primary objective and step-by-step update logs.
- For the original learning rate branch, at update points prior to 320, 384, 448, and 512 steps, the primary and auxiliary objectives must reproduce the original P16 logs with absolute tolerance of 1e-7; the full training score at shared starting points must match the prior report within 1e-6.
- Checkpoint hashes are recorded before evaluation. All fixed-timepoint scores are reported, without selection of the best point or seed.
- Perform 60 Rust DSP re-plays using the five predefined training anchors and their corresponding identities—two shared starting points and four endpoints.
  This does not constitute a new candidate ONNX streaming inference validation.

No development audio scoring is read, no new held-out recordings are used, no generalization is claimed, and no default weights are replaced.

## Evaluation Criteria

A numerical consistency tolerance of 1e-6 is used—this is not a threshold for perceptible improvement. For further investigation of small updates, the following conditions must be met:
- Both seeds must show lower primary and combined objectives at the fixed endpoints compared to their respective starting points and the original learning rate branch.
- Both paired recovery and identity metrics must not degrade relative to the starting point.
- Additionally, report the worst error at intermediate fixed timepoints and the number of times it exceeds the starting point.
  A better endpoint alone does not constitute overall stability; strict monotonicity across all optimization steps is not required.

If the small learning rate update fails to improve performance on the full training corpus, further investigation into actual output update magnitude and target coupling is required.
If short-term improvements are observed, this only indicates that the continuation window supports a viable optimization direction—further work must be logged for longer training and cross-recording rechecks.

This round does not prove that the P16 degradation in later stages was solely due to learning rate, nor does it prove sufficient capacity or that the learning rate is optimal.

## Execution and Output

Script: `training/scripts/run_consistency_lr_study.py`.
Full state restoration testing uses a dropout-enabled model, with continuity preserved during training. Parameter values, Adam momentum and counter states, random state recovery, and branch training without modifying the original checkpoint are verified.

Experiment directory: `data/runs/consistency-lr-p17-20261013/`.
Input files and code hashes are recorded prior to execution; local weights, raw audio, and reports are excluded from push.
Training source code snapshot: `16450b17fa3cd117c7956ddb5e6b930485c54a0d`.
Four short runs, full fixed evaluations, and 60 original DSP re-plays were completed at 22:40:47 (Singapore Time) on 2026-10-13.
The script `training/scripts/analyze_consistency_lr_study.py` rechecks source files, input files, completion credentials, checkpoints, scores, and summary consistency, and outputs `study/comparisons.json` according to the previously registered rules.

## Results: Small Learning Rate Update Passes Short-Term Optimization Benchmark

Primary error (not on development set, not after auxiliary term addition) across the full training corpus:

| Seed | Start (256) | Original lr=1e-3 (512) | Small lr=1e-4 (512) | Relative improvement vs. start |
| --- | ---: | ---: | ---: | ---: |
| 20261012 | 0.290827 | 0.287360 | 0.279703 | -3.82% |
| 20261013 | 0.289273 | 0.291007 | 0.279112 | -3.51% |

Small learning rate endpoints are 2.66% and 4.09% lower than the original learning rate endpoint, respectively.
Both seeds pass all four required checks, including improved combined objective and no degradation in paired recovery or identity metrics.
All eight pre-update logs from the original learning rate branch are reproduced with tolerance 1e-7; the shared starting point scores match the P16 baseline within 1e-6.

Per-seed performance at endpoint:

| Metric | 20261012: Start → End | 20261013: Start → End |
| --- | ---: | ---: |
| Paired recovery | 0.560540 → 0.554782 | 0.556942 → 0.553433 |
| Identity damage | 0.021114 → 0.004623 | 0.021604 → 0.004791 |
| Auxiliary term | 0.116055 → 0.105803 | 0.111606 → 0.105066 |
| Combined objective | 0.306671 → 0.294147 | 0.304510 → 0.293456 |

Paired recovery improves by 1.03% / 0.63%, identity damage drops by 78.10% / 77.83%.
Of the total 74.12% / 82.73% reduction in primary objective, over 70% is attributable to reduced identity damage.
This constitutes evidence of optimization and fidelity improvement—not evidence of significant recovery of missing high-frequency content.

Among the four intermediate and endpoint checks, the small learning rate branches for both seeds maintain primary error below their respective starting points.
In contrast, the original learning rate branch exceeds the starting point in 1/4 and 3/4 of the checks, respectively.
For seed 20261013, the original learning rate reaches 0.348863 at step 384, while the small learning rate version reaches only 0.279661 at the same point.
However, at step 320, the paired recovery and auxiliary term for seed 20261013 briefly exceed the starting point, and the primary error slightly increases from step 384 to 448.
Thus, while the reduction in rebound at the checked points is evident, we cannot claim monotonic stability across all steps or metrics.

## Native DSP and Evidence Boundaries

Maximum absolute deviation across 60 Rust DSP re-plays: **2.384186e-7**.
For the fixed five training pairs (4 music + 1 speech), relative to their 256-step starting points:

| Small learning rate endpoint change (negative = better) | 20261012 | 20261013 |
| --- | ---: | ---: |
| High-frequency spectral convergence error | +0.000467 | -0.002911 |
| High-frequency transient error | +0.001526 | -0.004503 |
| High-frequency energy absolute error (dB) | +0.022896 | +0.245379 |

Directional inconsistencies in spectral shape and transients; energy errors slightly increase.
These results cannot be substituted by average loss across the full training corpus.
For the same fixed subset, identity-level high-frequency spectral shape, transient, and energy errors all improve—further supporting the primary benefit as reduced degradation.

These few training segments do not represent music quality across recordings, nor do they reflect stereo output, real-world device behavior, or blind listening validation.

No development set re-evaluation, independent testing, blind listening, or candidate ONNX export was performed.
No runtime graph or DSP modifications were made, and no default weights were replaced.
Therefore, the results of this short-term study cannot be interpreted as surpassing the initial identity baseline or resolving all post-P16 issues.

Next steps include registering a separate, fixed-budget, two-seed low-learning-rate continuation run, maintaining full training corpus validation, and post-freeze rechecking cross-recording music and native high-frequency detail metrics.
No development score points should be selected.
If only identity continues to improve while paired high-frequency recovery shows inconsistent gains, no further identical long training should be pursued—instead, shift focus to music coverage and output representability comparisons.
Current results do not support direct model scaling.

## Verification

456 local Python tests passed; Ruff lint/format passed.
Training implementation CI: `38414650100`, Analysis implementation CI: `38415016014`—both passed.
