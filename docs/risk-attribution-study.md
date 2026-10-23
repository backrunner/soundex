# P36: Impact of Error Compression and Identity Weighting Reduction

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

Following [P35 Broad-Risk Study](broad-risk-study.md), this round completes a phased audit of the deployed endpoint with frozen weights, then fills in two additional comparative configurations. No new loss term is introduced, and the deployed DSP remains unchanged. Four new official trajectories have been completed, and all automatic rechecks are finalized. The error compression module retains spectral gains on two seeds, while the identity weighting reduction incurs a higher fidelity cost. However, neither combination passes the full study screening.

## Degradation Localization

Actual Rust frontend and ORT predictions are performed on the four P35 endpoints and all 89 existing held-out pairs. A total of 712 paired/identity roles are rechecked. Comparisons are made between raw OLA, band-fused, unclipped gating, actual clipping, and the native final output. All stages are evaluated against the original unclipped reference. The maximum synthetic error is `2.98e-7`, with no tolerance failure.

The unclipped stage is used solely for diagnostic counterfactual analysis and is not part of the deployed configuration. The order of gating and clipping is unchanged.

Changes in music dynamic error relative to the recording mean are as follows:

| Stage | Seed 20261012 | Seed 20261013 |
|---|---:|---:|
| Raw ORT Prediction (original OLA) | +0.641% | −0.008% |
| After band fusion | +0.720% | +0.067% |
| After unclipped gating | +0.720% | +0.067% |
| Final output | +0.696% | +0.064% |

Dynamic degradation in the first seed already exists in the original model prediction. The second seed shows slight improvement in the original prediction, which turns into minor degradation only after fusion. Voice dynamics also degrade by approximately 0.816% / 0.108%. Therefore, future work must address both model training and fusion—errors cannot be attributed solely to DSP.

Additional identity generation in music is already present in the raw prediction stage. Band fusion reduces the absolute dynamic error, but the final clipping introduces a new relative error compared to the original reference. It is inappropriate to claim improvement by removing safety clipping, nor can a large identity ratio be used to infer perceptual quality. The absolute PCM magnitude and worst-case fragment performance are detailed in the P35 report.

For each seed, 16 evenly spaced batches from the full training sequence are selected. Gradient measurement is performed only on frozen weights, with no parameter updates or selection of the optimal validation model. The current optimization gradient is decomposed into four components: non-dynamic reconstruction, reference dynamic, identity protection, and budget. Each batch is individually verified to ensure the sum of these components matches the original target gradient. For the non-dynamic reconstruction and identity protection directions, 14/16 and 15/16 batches show negative cosine values, with median values of **−0.495 / −0.648**. The non-dynamic and dynamic components show directional conflicts in 6/16 and 5/16 batches, respectively. This supports a diagnostic of target trade-offs, though the limited training cropping gradients do not independently prove causality, overfitting, or the effectiveness of increasing any particular weight.

Local evidence: `data/runs/risk-stages-p35-20261022/`, `data/runs/risk-gradients-p35-20261022/`. This remains a post-hoc diagnosis using reused held-out/training materials and is not an independent generalization test. All existing counterexamples are retained.

## Fixed Two-Factor Comparison

Since P35 modified both error compression and identity weighting, the existing two combinations cannot isolate individual contributions. P36 completes two additional configurations to evaluate where the recovery benefits originate and whether reduced identity weighting can alleviate degradation.

| Error Method | Identity Weight 1.0 | Identity Weight 0.3 |
|---|---|---|
| Raw direct + MR | P35 `control`, frozen | P36 `identity_mid`, new training |
| Per-example log1p(direct) + log1p(MR) | P36 `compressed`, new training | P35 `compressed_identity_mid`, frozen |

All combinations maintain a fixed paired coefficient of 0.5, with the budget term kept at 0.1× and unchanged in paired/identity ratio. Each new configuration uses the two designated seeds, trained from random initialization for **16,192 steps**, with four new endpoints. Model, optimizer, sampling order, speech proportion, historical warmup, and supervised region are identical to P35. No P35 checkpoints are reused for fine-tuning.

Historical controls are pre-validated for full evidence and endpoint hashes, and are tied to reference assessments, initialization hashes, and data manifests. Model, data construction, loss, and optimizer execution must be identical to P35. The new initialization must exactly match the P35 initialization for the same seed. Reports on the impact of compression and identity weighting reduction, as well as interaction terms computed from common MR scores, are provided. No comparison of total loss or percentage differences is used to infer interaction effects.

## Validation and Acceptance

After completing training and freezing the four new endpoints, native PCM, consistency, common MR, and real-time baselines are executed, followed by unified analysis with the fixed P35 units. The original prediction and actual DSP output are always reported together, with separate analysis for music and speech, paired and identity, per-recording gains, and counterexamples. The used recordings remain from the reused held-out set and no new independent test claims are made.

The study screening criteria for the new combinations follow P35: for both seeds, the average native music MR and the median difference in per-recording error must improve. Music and speech identity MRs, as well as dynamic values for each role, must not degrade. Combinations that fail must be fully reported and used for diagnosis—no group selection based on intermediate metrics, no budget reduction, and no automatic default weight override. Dynamic terms continue to follow the true reference changes, and smoothing transient behavior to improve scores is not permitted.

Code entry points: `training/scripts/run_risk_attribution_study.py` and `training/scripts/analyze_risk_attribution_study.py`. The explicit protocol, objectives, and evaluation callbacks from the broad coverage executor are reused. Checkpoint recovery, full endpoint freezing, Rust/ORT/DSP validation, and hop-by-hop baselines remain unchanged. Original audio, checkpoints, and detailed reports are retained in the ignored directory; only code and source/experiment documentation are submitted.

## Execution Status

Pre-check `data/runs/risk-attribution-p36-preflight-20261022-v2/` is complete: two new combinations, each with four steps, cover music/speech, 44.1/48 kHz, paired/identity, 16 native consistency roles, and 16 additional native MR roles. All 651 Python tests pass, and Ruff lint/format passes. Tests include exact gradient after factor change, preserved budget term, interaction effect units, per-recording alignment, merging of both sub-results, and rejection of input baseline changes. The 188 execution dependencies' hash is identical to the preflight and formal training.

Formal directory: `data/runs/risk-attribution-p36-20261022/`; registered SHA-256:
`0900a05c633afbb0529715e66b88e105c90ebfbff999487d0150d07bcf593fad`.
An independent supervision process has completed normally, with state saved every 64 steps during training. Entry points are `supervisor.json`, `study/progress.json`, and `pipeline.log`. Evidence is preserved on failure; no automatic budget reduction or selection of intermediate best models occurs. Pre-check short training is not counted toward formal initialization or sufficient training conclusions.

## Complete Results (2026-10-22)

All training and automatic rechecks were completed at **10:51:22 Singapore Time, 2026-10-22**, taking approximately **8 hours 7 minutes**. Each of the four new endpoints completed 16,192 steps. All update history, endpoint/assessment evidence hashes, and full input registration have been verified. The supervision process exited normally with no pipeline errors. The four P35 reference endpoints remain unchanged.

Below are per-recording mean values for common MR v1 and actual Rust/ORT/DSP outputs, relative to the same seed P35 control; negative values indicate error reduction. Values in parentheses list results for the two seeds in order, and comparisons across different total loss percentages are not permitted.

| Configuration | Music Spectrum | Speech Spectrum | Music Dynamics | Music Identity Spectrum |
|---|---:|---:|---:|---:|
| Only error compression, identity 1.0 (P36) | **−1.417% / −1.274%** | −1.773% / −2.028% | +0.214% / +0.242% | +5.331% / +1.629% |
| Only identity reduced to 0.3 (P36) | −1.031% / −1.142% | −3.020% / −2.437% | +0.823% / +0.295% | +85.285% / +29.419% |
| Compression + identity 0.3 (P35) | −2.126% / −1.118% | −4.342% / −3.072% | +0.696% / +0.064% | +36.550% / +2.338% |

The native music MR for only compression is **0.378741 / 0.378982**, representing a relative reduction of **1.657% / 1.594%** compared to uncompressed input. The original model OLA shows a relative reduction of **1.669% / 1.613%**. Among 26 rechecked music pieces, 20 improved and 6 degraded in the first seed, while 17 improved and 9 degraded in the second seed, with median per-recording differences improving in both. For the identity reduction-only case, the original OLA music MR decreased by 1.308% / 1.444% relative to control, and the native output decreased by 1.272% / 1.462% relative to input. Thus, both factors independently produce recovery gains—neither can be credited with all the benefit.

The speech dynamic for only compression still increases by **0.092% / 0.066%**, but speech identity MR drops by **96.648% / 90.398%**. For only identity reduction, speech dynamic increases by 0.632% / 0.286%, and speech identity MR increases by **1883.597% / 719.121%**; the latter has a very small baseline and must be interpreted in conjunction with the next section's PCM absolute magnitude.

Within the compression target, reducing identity from 1.0 to 0.3 shows inconsistent effects on music MR across the two seeds: the first seed sees further improvement, while the second seed shows slight degradation. The interaction term for music native MR is **+0.001239 / +0.004985** (in common score units, not percentages); for speech, it is **+0.001452 / +0.004492**. The combined gain is not simply the sum of the individual gains.

For compression-only: gradient clipping count is **1 / 0**, maximum norm **10.491 / 4.708**; for identity-only: **63 / 46**, maximum norm **401.072 / 745.066**. Compared to P35 control's 59 / 45, extreme gradient reduction is primarily associated with error compression, while identity reduction does not resolve this issue. Full budget does not prove convergence.

## Absolute PCM, Native Consistency, and Latency

The native output relative to the same material identity DSP in PCM RMS, averaged per recording:

| Configuration | Music | Speech |
|---|---:|---:|
| Only compression | 2.28e-6 / 2.89e-6 | 4.65e-8 / 5.92e-8 |
| Only identity reduction | 4.14e-5 / 1.85e-5 | 1.40e-6 / 6.53e-7 |

For compression-only, the worst-case single fragment music is `2.29e-5 / 2.32e-5` (−92.81 / −92.70 dBFS), and for identity-only, it is `3.42e-4 / 9.46e-5` (−69.31 / −80.48 dBFS). For speech, the worst-case single fragment for compression-only is `4.61e-7 / 7.28e-7` (−126.73 / −122.76 dBFS), and for identity-only, it is `1.74e-5 / 6.74e-6` (−95.19 / −103.43 dBFS). These are actual PCM differences, not unnormalized FFT power. No perceptual inference is drawn from these absolute values, nor are thresholds adjusted post hoc based on magnitude.

For paired music, relative to the original lossless PCM RMS, compression-only yields `0.0134995 / 0.0135008`, compared to the control of approximately `0.0134978`. This slight increase does not negate the spectral gains. Music identity error relative to the original lossless also includes the impact of identity DSP's own clipping/synthesis, and cannot be conflated with new model modifications.

A total of **712 native PCM roles + 712 additional native MR roles** are newly evaluated, resulting in a maximum synthetic error of **4.17e-7**, with zero tolerance failure. A total of **280,000 hops** are tested: maximum processing time is **1.278 ms**, highest P99 is **0.517 ms**, zero instances ≥5 ms, and zero instances exceeding hop period. Peak RSS is **31.69 MiB**. The `paced` system still has **1 instance of late completion, 0.467 ms**. This does not justify claiming zero scheduling delay or device authentication. Lower latency does not imply faster model structure—testing conditions differ. A fixed 256-frame buffer remains at 5.333 ms @48 kHz / 5.805 ms @44.1 kHz, plus additional overhead.

## Next Steps

Both new configurations fail pre-registration screening due to dynamic or music identity degradation. Only error compression retains music gains and better controls identity changes compared to the identity-only and P35 combinations, making it a suitable starting point for further study. However, it does not outperform P35 across all metrics and is not yet a qualified release weight. Future work will prioritize identifying remaining high-frequency dynamic errors and a few identity counterexamples, with continued use of the same material and direct comparison between original prediction and deployed output. The two sub-results from the held-out set do not replace independent generalization, blind listening, or real-device stability validation. This round does not result in promotion of a default weight.

Subsequent work completed: [P36 Temporal Dynamics and Identity Counterexample Diagnosis](risk-temporal-diagnosis.md). Confirmed that gains include improvements in missing high-frequency input, and dynamic degradation exists in both raw prediction and fusion stages. Identity mean degradation is primarily driven by a few extremely weak reference band counterexamples. See detailed analysis by frequency band, per-recording, absolute PCM, and supervision path.

The unchanged small architecture has 431015 parameters.
