# P8: Linear Reconstruction Objective and Independent Group Verification

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

Building upon [P7 Loss Audit](energy-objective-audit.md). In this round, the P7 additive energy two-branch model is fixed, and we compare only the training objectives—no model scaling, loss modification, or new inference features are introduced in the same experiment. The following rules are fixed prior to formal training and remain unchanged throughout, with default weights preserved.

## Reconstruction Objective

For each sample and each frequency band, we exclude the first 80 hops. Let `f` be the reference full-spectrum RMS at 1e-4, with an absolute lower bound of 1e-7 DFT magnitude, consistent with P6 and P7. Define `D² = mean(target²) + f²`, where the mean is taken over the frequency points and effective time frames in the band. The smooth norm is defined as:

`N(error) = sqrt(mean(error²) / D² + 1e-6) - 1e-3`.

- **Amplitude reconstruction term**: `N(prediction - target)`, jointly constraining both energy and intra-band detail.
- **Reference dynamic term**: `N(diff(prediction) - diff(target))`, preserving the sign of differences, with the denominator still using the reference magnitude energy.
- **Total score**: amplitude reconstruction + 0.25 × reference dynamic. Weighting: 0.25 for 0–4 kHz, 1.0 for each of the other four bands, normalized by 4.25.

This replaces the previous sum of individual components with a unified linear reconstruction objective. **The old log-sum score is not re-added**. Historical metrics—original energy, normalized shape, dynamic, and log1p amplitude—are still recorded for analysis, and the old total score is stored separately as `factor_total` for interpretability. When the reference is correct, the new loss approaches zero; the smoothness term prevents undefined gradients at zero residual; the floor avoids infinite amplification when reference magnitude is extremely small.

All reference statistics are used solely for training and scoring and are not passed into inference.

Validated against audit-level counterexamples: correct spectral shape with only 1% amplitude yields a loss of **0.23271**; correct RMS with flat spectral shape yields **0.30610**, and such detail degradation is no longer rewarded. Amplitude scaling of 0.1x and 10x results in **0.21153 / 2.11741**, respectively—no longer approximately equivalent. These results are mathematically verified, not musical quality judgments.

## Comparison and Data

| Group | Model | Training Objective | Shared Selection Target |
| --- | --- | --- | --- |
| energy_control | P7 rms_residual | Original P6/P7 component loss | P8 Linear Reconstruction Total Score |
| linear_reconstruction | Identical model | P8 Linear Reconstruction Total Score | P8 Linear Reconstruction Total Score |

Both sub-experiments use seeds `20261012` and `20261013`, trained four times from scratch, each with 2048 updates and batch size 4. All initial tensors are identical under the same seed, resulting in identical initialization across runs. Total parameters: 807,398. No P7 weights are loaded. Optimizer: AdamW with lr=1e-4, cosine decay to 0.1×, weight decay=1e-4, gradient clipping=5.
Input remains past-only with 1024/2048 point history, 80-hop warmup, and audio/track isolation preserved.

For each seed, 8192 samples are drawn from `seed+60000`, with 1024 samples serving as identity control. Music and speech sampling, and identity substitution, remain unchanged. Validation occurs every 512 steps, with shared selection using music/speech ratio 253:1 and paired/identity ratio 87.5:12.5. Selection at step 0 is allowed, but does not imply successful recovery. If the best performance occurs at the final step, it does not imply convergence.

New and old loss values cannot be directly compared across objectives. Formal comparison uses the same new objective to recompute both groups.

## Group Checks Cannot Be Overridden by Overall Averages

The shared total score is solely responsible for selecting study checkpoints. After selection, additional per-group checks are performed independently. These checks are not iteratively reselected based on results, nor do they offset poor performance in one group with gains in another.

**Weak band definition**: a band where reference clip-band RMS is ≤ the previously defined `f`, and only bands above 4 kHz are considered.
For comparison and candidates, the reference grouping must be identical. Fragments without weak bands do not dilute the average of weak groups to zero.
Per-recording segment averages are computed first, then 2000 paired bootstrap resamples are performed.

- **Development music high-frequency SC and energy error**: must show improvement over both baseline and compressed input under the same budget.
- **Paired and identity music weak-band reconstruction error**: reported separately, with no degradation allowed.
- **Music and speech identity reconstruction error**: reported separately, with no degradation allowed.
- **Music and speech preservation of low-band SC, high-frequency flux, and paired speech reconstruction error**: checked individually.

"Support for improvement" requires the upper bound of the 95% confidence interval to be ≤ -1e-6. "Support for no degradation" requires the upper bound to be ≤ +1e-6.
1e-6 is a numerical tolerance, not a perceptual one. If fewer than two recordings are available, the result is marked as *unsupported*. If the confidence interval is insufficient to support a conclusion, it is marked *inconclusive*. Only when all checks are supported is `all_groups_supported=true` set.
Thus, unchanged output, insufficient samples, or overall average improvement do not automatically pass.

This is a conservative diagnostic, not a statistical equivalence proof or a production promotion. Reuse of development recordings does not imply independent generalization. Relative error in low-power bands does not substitute for perceptual quality. When any check fails, full details are reported—no failure samples are suppressed or thresholds altered.

## Workflow and Acceptance Boundaries

Loss testing covers: reference optimal, weak/noisy gradient, finite difference, batch independence, warmup exclusion, real reference dynamics, MPS/CPU correspondence, and the previously cited counterexamples. Group regression validation must not mask damage to weak groups, and paired, reference grouping, and missing sample handling must be correctly processed. The model has no structural changes, and single/dual-channel ONNX state consistency is verified.

After formal training, automated execution of:
- Pairwise audio comparison
- Full group validation checks
- First seed candidate ONNX export
- Per-hop CPU cost and independent RSS
is performed. Benchmarking runs are executed *after* training and are not concurrent. Then, 890 real-time Rust DSP re-plays are conducted, including corresponding seed P3, identity DSP, and phase-matched pairs: model paired with P3 phase, and candidate paired with input phase.

Retained metrics include dynamic HF, fixed HF, and five-band fixed window raw/analyzed outputs. Proxy error requirements are ≤ 2e-5.

These group checks only evaluate raw magnitude. Final validation still requires actual DSP output gain, identity output playback, independent evaluation, blind listening, and full real-time stability—no automatic model replacement is permitted based solely on magnitude-based pass.
The model remains constrained by ≤2M parameters and <8 MiB memory. Frontend + inference latency and full output latency are reported separately.

```sh
PYTHONPATH=training training/.venv/bin/python training/scripts/run_linear_study.py \
  --parent data/runs/energy-p7-20261012/study/config.json \
  --output data/runs/linear-p8-20261012/study \
  --cache data/runs/factor-p6-20261012/study/cache

PYTHONPATH=training training/.venv/bin/python training/scripts/analyze_envelope_study.py \
  --study data/runs/linear-p8-20261012/study --benchmark

PYTHONPATH=training training/.venv/bin/python training/scripts/replay_factorized_study.py \
  --study data/runs/linear-p8-20261012/study \
  --phase-study data/runs/causal-p3-20261012/study \
  --replay-binary target/debug/soundex-prediction-replay
```

Executed from the frozen source code. Configuration, sampling, checkpoints, and pipeline stages are preserved locally. Source audio and experimental weights are not pushed.

## Final Results (2026-10-12)

All four groups completed 2048 steps. Checkpoints and evaluation hashes match those in the request. Shared development selection scores are as follows, with lower being better:

| Checkpoint | Seed 20261012 | Seed 20261013 |
| --- | ---: | ---: |
| Step 0 | 0.494611 | 0.494611 |
| energy_control, Step 2048 | 1.920612 | 1.789279 |
| linear_reconstruction, Step 2048 | 1.188453 | 0.888879 |

The new objective achieves lower error than the previous objective at its final step, but still does not surpass the initial state. All group selections converged to Step 0—this is not due to group checks rejecting and then reverting; the average selection score itself failed. For each seed, 13 group-level checks were evaluated: 9 were supported for no degradation, and 4 were inconclusive (improvement not statistically significant). The lack of degradation in near-identity models does not constitute recovery.

890 Rust DSP re-plays have been completed. Maximum proxy/original error is 4.54e-6. However, using the selected Step 0 magnitude model, no quality gain is demonstrated. Candidate model has 807,398 parameters, ONNX size: 3,271,096 bytes (3.12 MiB).
Frontend + ORT peak performance (P99) is 0.400 ms, observed maximum 1.120 ms. Independent process peak RSS increase: 20.97 MiB.

These represent diagnostic overhead and do not cover detector, DSP, audio adapter, or device-level performance. They do not constitute evidence of complete output latency or perceptual quality.

No default weights were replaced. Future [Optimization Process Audit](linear-optimization-audit.md) will examine first-step overshoot, small-sample fitting, and energy coordinate input scale conditioning. No improvement was achieved by relaxing group thresholds.

Exact registered seeds: `20261012/20261013`. The high-band weights are 1 each, with low-band weight 0.25 and normalization by 4.25; this is a reference-reconstruction objective, not a mandate to suppress all enhancement.
