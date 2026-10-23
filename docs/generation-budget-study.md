# P24: Complete Comparison of Generated Energy Constraints and Error Scales

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

This round builds upon [P23 Absolute Error Audit](capacity-outlier-audit.md). The audit revealed subtle, high-frequency mis-generation artifacts and amplification of weak reference signals in relative scoring—both of which require separate handling. P24 does not set thresholds based on individual track performance, does not remove known counterexamples, and does not equate low absolute error with inaudibility or successful recovery.

Subsequent [P25 Complete Inference and Coverage Audit](context-deployment-audit.md) confirmed the persistence of genuine audio perturbations in the Rust frontend → ORT → DSP chain, and identified limited supervised coverage and unbounded residual budget issues.

## Complete Validation Results (2026-10-17 15:56 SGT)

All recovery validations have concluded. We have re-verified inputs, source code, sampling rates, frozen endpoints, and **74 completed verification reports**.
25 relative spectral metrics were replayed (identity control + 24 endpoints), covering 4,450 roles; an additional 4,450 candidate roles and 178 cached identity DSP controls were replayed, resulting in a total of **9,078 actual Rust role replays**.
All 24 endpoints completed ONNX consistency checks and 10,000 hop cost measurements.

Using the same Rust-based observation spectrum, the maximum synthetic proxy error was `3.57628e-7 < 2e-5`, with no failures. The original training frontend still exhibits **one unique role failure** (retained in both replay workflows), with a peak error of `2.44714e-5`. This failure was not eliminated. These validations involve externally generated spectra passed through real DSP—not equivalent to end-to-end live inference of the full production frontend + ORT model pipeline. Prior reused recordings remain under development review and are not considered independent test cases.

Below are fixed results for **16,192-step endpoints and two seeds**, averaged, across 33 musical fragments and 26 recordings, aggregated by clip. Lower error is better, with waveform RMS computed against original clean reference and full-amplitude PCM units:

| Configuration | DSP High-Frequency SC | DSP High-Frequency Flux | High-Frequency Energy Error (dB) | Paired Waveform Error RMS | Identity Perturbation vs. Identity DSP RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Identity Model + DSP | 0.548761 | 0.717164 | 8.4801 | 0.01251822 | 0 |
| control | 1.901090 | 1.684844 | 7.7538 | 0.01251827 | 0.000002187 |
| balanced | 10.175414 | 8.131500 | 5.5594 | 0.01262628 | 0.000197319 |
| budget | 0.565158 | 0.753107 | 7.0350 | 0.01254734 | 0.000096268 |
| joint | 7.947084 | 5.898544 | 5.3525 | 0.01260774 | 0.000167033 |

The budget configuration shows a significant relative improvement over the control, though still higher than the control in several metrics. Compared to the identity model + DSP, the budget still exhibits a 2.99% higher high-frequency SC, a 5.01% higher high-frequency flux, and a 0.233% higher waveform RMS error. Energy error is reduced by approximately 17.0%, but this does not justify claims of improved spectral shape, fine detail, or overall audio quality. When aggregated by recording, budget achieves a high-frequency SC of 0.566754 and flux of 0.753341, compared to 0.561468 and 0.733050 for the control—same conclusion. Low-frequency SC remains largely unchanged, indicating no notable low-frequency improvements.

As an independent baseline, the unprocessed compressed input shows a waveform RMS error of 0.01240408 against the original clean reference and 0.00350853 for speech. The identity DSP shows higher music errors, attributable to clipping and synthesis artifacts; the budget configuration does not achieve overall superiority over the unprocessed compressed input. No modifications or clipping of the clean reference were applied to mask these errors.

For the budget branch, 26/33 music fragments and 23/33 music fragments show relative proxy error reductions in paired comparisons, respectively. However, only 9/33 and 2/33 show better waveform RMS against the original clean reference. Different metrics evaluate different objectives. The majority of relative score improvements cannot substitute for waveform or perceptual judgment, nor can the observed relative degradation be fully attributed to audible distortion.

The old main counterexample `d2599868e6a472eff47c1674` saw its identity perturbation drop from -97.15 / -92.10 dBFS in the control to -175.51 / -144.20 dBFS in the budget. However, a new counterexample `64f035bd9dd0d09364fae6e0` (starting at 53568) now exhibits identity perturbations of -52.21 / -52.68 dBFS across both seeds—approximately 1.115% / 1.056% of the original signal RMS, compared to -105.58 / -105.13 dBFS in the control. This represents actual PCM-level modifications, not just relative score amplification due to weak references. Blind listening has not yet been conducted, so this cannot be directly labeled as audible burst noise. No threshold or removal of this case is proposed at this time.

For speech, the budget's high-frequency SC decreased from 0.291963 to 0.282180, but the average paired waveform error increased from 0.00350853 to 0.00351375. Both balanced and joint configurations show trade-offs between spectral and waveform error. For shifted music, the common proxy total decreased from 0.301122 to 0.296100 in budget; balanced and joint achieved 0.438069 and 0.359974, respectively. The new PCM metrics for shifted music are derived from proxy outputs; these validations do not constitute sufficient evidence of overall audio quality gains.

All 30 historical relative quality comparisons failed complete screening. Neither the absolute waveform errors nor the original signal perturbations support a formal weight increase. **This round of validation is complete, but no new qualified weights have been produced; the default model remains unchanged.**

### Measured Latency and Continuity Boundaries

Final endpoint, dual-channel, 10,000 hop, single-threaded CPU ORT + Python causal FFT frontend:

| Configuration | Two Seeds p99 Range | Observed Maximum | Virtual Deadline Violations / 20,000 Hops |
| --- | ---: | ---: | ---: |
| control | 1.377–1.474 ms | 3.313 ms | 16 |
| balanced | 1.132–1.172 ms | 5.795 ms | 16 |
| budget | 1.795–2.949 ms | 11.848 ms | 347 |
| joint | 1.749–2.912 ms | 44.586 ms | 432 |

The budget branch shows a new ORT process memory increment of approximately 65.3–65.7 MiB; excluding the full Rust and FFT frontend, and not representing peak memory. The file size is about 45.5 MiB; the formal product size protocol has not yet been migrated.

This long-duration test revealed latency spikes not observed in short tests. We cannot continue claiming zero latency or announce a stable model + DSP latency below 5 ms. The virtual deadline count includes queueing effects and is not equivalent to physical hardware xrun. These measurements do not include the full DSP worker, host scheduling, or load isolation. Thus, the worst-case values cannot be directly interpreted as the model's inherent computational upper bound. Future work requires controlled long-duration tests with the full worker, a real 2.667 ms hop deadline, and actual device validation. The fixed 256-frame buffer remains separately accounted for in output latency.

Next steps prioritize generalization of original audio enhancement artifacts, consistency of real-time inference frontends, and tail-end processing latency. This round provides no sufficient basis for further expanding capacity or extending training on the same objective.

## Training Completion and Recovery Verification Records (2026-10-17)

Eight trajectories completed all 129,536 updates by **08:03 SGT**, with 24 endpoints frozen. Eight training completion reports, full update counts, and checkpoints were re-verified. The automatic pipeline paused at **08:04 SGT** during actual Rust replay—this was not a training failure. The original directory and source code at commit `7944d7a` remain unchanged.

The table below shows the complete **training pool**, averaged across two seeds, using the same P23 scoring scheme. No direct comparison of new and old total scores across configurations is valid, and no training error should be used to infer development or audio quality conclusions:

| Configuration | Paired Recovery Error | Identity Fidelity Error | Common Total |
| --- | ---: | ---: | ---: |
| control | 0.576004 | 0.004114 | 0.290059 |
| balanced | 0.542203 | 0.050916 | 0.296559 |
| budget | 0.571044 | 0.000383 | 0.285713 |
| joint | 0.560202 | 0.036299 | 0.298250 |

The budget configuration shows a ~1.50% reduction in training total and a ~90.7% reduction in identity error compared to control. While balanced and joint show lower recovery error, they exhibit higher identity fidelity error. This training table does not replace the full development validation above and cannot justify early selection of the budget configuration.

The blocking case is `joint-2048-20261012` development index 29, paired role `57bace181ff17ad01b58d050`, starting at 47518. The maximum PCM difference in the original proxy was `2.4471432e-5`, exceeding the established `2e-5` threshold; RMS difference was `1.32111e-6`, with actual output peak at ~0.1701. No NaN or clipping occurred. The double-precision proxy remained at `2.44781e-5`, and no fix via float64 conversion or threshold relaxation resolves the issue.

Using the newly added offline `soundex-analysis-trace` to export production `StftAnalyzer` observation spectra, we confirmed a phase difference between Rust and PyTorch FFT at extremely low magnitude points. For example, at frame 232 / bin 49, observed magnitude is approximately -111 dB (FFT magnitude coordinate, **not PCM dBFS**), with a phase difference of ~0.01321 rad and fusion weight ~0.4943. When the model generates high-amplitude components, this phase discrepancy is amplified by the unit vector fusion.

With frozen prediction, input PCM, detector, gating, clipping, and Rust replay binary unchanged, only the observed phase from the proxy fusion was replaced with the actual Rust value. The maximum error dropped to `3.72529e-8`; when the full observation spectrum was replaced, the error became `4.47035e-8`. The original identity role bypass had zero error throughout. The current source code reconstruction of the replay binary differs from the frozen binary by at most `1.49e-8`. The original failure was not eliminated by reconstruction. Evidence is saved in `data/reports/p24-parity-investigation-20261017/diagnosis.json`.

A new review-only pipeline is located at `data/runs/generation-budget-p24-review-20261017/`, with frozen verification source code `a14d9e4`. It re-validates all 24 fixed endpoints, identical data, and actual Rust output:

- The original `proxy_max_error` and `training_frontend_parity_passed` are preserved; the failure remains explicitly false.
- A new synthetic consistency check based on the same Rust observation spectrum is added, requiring maximum error ≤ `2e-5`.
- Model predictions and quality references remain unchanged; no modification to the original training loss; no report overwrites in the original registry.
- This isolates the error between the observation frontend and DSP synthesis. It does not resolve differences between training and deployment inputs, nor does it constitute full Rust frontend + ORT real-time model acceptance. Deployment still requires validation of the real frontend's impact on model output.

The new path passed actual replay validation of the original failure fragment, 529 Python tests, Rust analysis tooling, Clippy, and formatting checks. The recovery pipeline does not automatically promote weights.
The recovery pipeline started at **15:19 SGT** and ended normally at **15:56 SGT**; historical supervision PID `36153` and validation subprocess PID `36157` have both terminated. Status and logs are preserved in the new directory under `study/progress.json`, `pipeline.json`, and `pipeline.log`.

## Fixed Experiment

All four configurations use a causal complex head with width=640, re-initialized from the same seed identity. This fixes the large model capacity and isolates the impact of the target function and generation energy branches.

| Configuration | Main Loss / Consistency Normalization | Input-Driven Generation Budget | Parameter Count |
| --- | --- | --- | ---: |
| control | P23 Original Scheme | Fixed global RMS (full frequency) | 11,917,698 |
| balanced | Full frequency + band reference power | Fixed global RMS (full frequency) | 11,917,698 |
| budget | P23 Original Scheme | Learnable band budget with supervision | 11,924,103 |
| joint | Full frequency + band reference power | Learnable band budget with supervision | 11,924,103 |

Two seeds (`20261012` / `20261013`), 16,192 steps per seed, total 129,536 updates. Full use of the P23 2,088 clip training pool, 320 common clips, original PCM hash, and frozen sampling sequence. Each step includes 4 music + 1 speech, paired/identity equally weighted; Adam 1e-4, no weight decay, gradient norm clipping 5, complex residual slope=1/64. Execution order: joint → control → balanced → budget.

After all eight trajectories complete, 24 endpoints are frozen at 2,048 / 8,192 / 16,192 steps. Subsequent development materials are then re-verified. Primary conclusions are based on final endpoints; intermediate endpoints are described only in terms of learning trajectories, with no best checkpoint selected. The 16-step preflight uses only training data and discards its weights; no early stopping at 512 steps. NaN/Inf, input drift, or execution faults must halt and preserve recovery information. Completion does not imply convergence.

## Generation Branch

In P23, the residual unit was `global_rms * sqrt(local_power + 1)`. Even in very weak high-frequency bands, the full-frequency RMS served as the generation unit. In P24, a new branch predicts five `softplus` non-negative budgets from the same causal encoder's latent space, interpolated linearly across 2 / 6 / 10 / 14 / 20 kHz centers, with edge continuation. The new unit becomes `global_rms * sqrt(local_power + budget(frequency)^2)`.

Initial budget value is set to 1, preserving original output and residual scale. Main trunk initialization is verified per tensor. The budget has no hard upper bound and does not multiply by input energy from suppressed bands, so it can reconstruct completely missing high-frequency components. This is a learnable generation scale—not a calibrated confidence probability, nor a strict output energy upper bound. Local terms still allow modification of existing frequency inputs. Sustainable, stable, and actual recovery benefits must be confirmed in subsequent validation.

During training, the residual RMS per band per frame, relative to the input global RMS, is computed from the difference between the original clean complex spectrum and input. The identity teacher is zero. For both prediction and teacher, `log1p` is used with Smooth L1 loss, fixed weight 0.1. This weight was not optimized over the development set. The teacher is only used in training supervision; the forward pass does not receive targets, track ID, or identity labels. The first 80 frames are used for pre-warmup; the last 9 frames are excluded from budget supervision. The budget factor includes both branch and auxiliary supervision, and its performance cannot be attributed to any single component without additional ablation.

## New Objectives and Common Evaluation

For balanced/joint, the band amplitude and reference change error denominator uses `reference_band_power + reference_global_power + 1e-14`. Band weights remain low-frequency 0.25, others at 1, divided by 4.25; change error coefficient remains 0.25. This links weak-band errors to the overall reference signal scale, without setting dB thresholds for counterexamples. This alters the trade-off between weak and strong bands—whether this weakens fine-detail recovery must be answered by results.

The primary objective retains 0.25× relative waveform error, plus an additional 0.25× full-amplitude PCM smooth RMS error. The clean-complex consistency auxiliary term also includes stabilization via full-frequency reference power in the denominator to avoid bias when the main objective changes; auxiliary weight remains `0.1365237430457469`.
Paired comparisons remain against original clean reference; identity comparisons remain against the output of the identity model passed through real DSP. Safety clipping error against the original reference is reported separately and cannot be masked by clipping the clean reference.

All configurations retain the original P23 relative scoring and add common balanced proxy and absolute PCM evaluation. Internal group optimization totals cannot be directly compared as quality improvements. Verification includes:

- 89 development and 40 shifted paired fragments—all previously used development validations, not independent tests.
- All 4,450 actual Rust DSP role replays, preserving raw model and DSP post-output spectral, energy, transient, and low-frequency metrics, plus identity model + DSP control.
- Full absolute PCM replay of all development data: calibrated RMS/dBFS, peak, per-band error, relative full-reference RMS; compared against original clean and identity DSP, with unprocessed compressed input vs. clean reference. Unnormalized FFT power is not considered PCM dBFS. For shifted data, new PCM metrics are explicitly labeled as DSP proxy.
- Separated by music/voice, paired/identity; all rows preserved, RMS in linear units, mean, p95, and maximum per clip/recording; no sorting via average dB. Old relative scores are preserved and cannot automatically trigger release.
- Each endpoint undergoes non-zero ONNX state consistency, mono/stereo, 44.1/48 kHz, silence/reset checks, 10,000 hop timing with single-threaded CPU ORT + causal FFT frontend, and independent process RSS.

No new "inaudible" threshold is assumed, and no automatic weight promotion path exists. True recovery, perceptual quality, full end-to-end real-time cost, and independent recording generalization are all separate, independent pieces of evidence. Diagnostics do not include the full Rust worker/DSP and cannot be used to claim model + DSP <5 ms. The fixed 256-frame buffer and 2.667 ms hop deadline remain in effect. Original materials, paired data, reports, and experiment weights remain in the ignored directory; authorization sources and formal protocols remain unchanged.

## Execution and Status

Source entry point: `training/scripts/run_budget_study.py`; preflight entry point: `training/scripts/preflight_budget_study.py`.

```sh
PYTHONPATH=training training/.venv/bin/python training/scripts/run_budget_study.py \
  --parent data/runs/full-capacity-p23-20261015/study \
  --output data/runs/generation-budget-p24-20261016/study
```

The official background execution uses the current `source/training` frozen copy. Modified working tree code cannot be used to recover previously registered experiments. Status and logs are stored in `pipeline.json`, `study/progress.json`, and `pipeline.log`.

A full background pipeline was initiated at **2026-10-16 16:59 SGT**; frozen code at `7944d7a`, supervisor PID `85896`, training subprocess PID `85898`. Process IDs are snapshots at startup; subsequent status is based on the state file. Training began at `joint/20261012`, without reusing 16-step preflight weights. This record is preserved; full results are shown above. Startup verification confirmed 64/16,192 steps completed and saved Adam/RNG checkpoints; all historical loss and gradient values are finite. Current source code passed 527 Python tests, Ruff checks, and formatting validation.

Preflight evidence is stored in `data/reports/budget-p24-preflight-20261016/`: 16 steps per group, using the first 16 draws of the full sampling sequence as training data, no development set scoring. The new budget head weights started from zero and produced non-zero updates, confirming gradient connectivity—this does not imply correct learning of recovery needs. 50 actual Rust DSP role replays achieved a maximum proxy error of `2.38419e-7 < 2e-5`. All four non-zero ONNX endpoints passed 320 hop state consistency; maximum Cartesian error was `1.33280e-7`.

M5 Max, single-threaded CPU ORT + causal FFT frontend, dual-channel, 1,000 hop measurement:

| Configuration | p99 | Maximum | Virtual Hop Timeout |
| --- | ---: | ---: | ---: |
| control | 1.157 ms | 1.240 ms | 0 |
| balanced | 1.109 ms | 1.160 ms | 0 |
| budget | 1.112 ms | 1.132 ms | 0 |
| joint | 1.141 ms | 1.167 ms | 0 |

The new branch ONNX is 47,742,559 bytes (~45.5 MiB), explicitly bypassing size thresholds; the formal product size protocol has not yet been migrated. This table does not include the full Rust worker/DSP, physical hardware, or real-time playback pressure. Thus, we cannot claim that the final model + DSP meets a 5 ms latency requirement. Full endpoint 10,000 hop re-measurements are shown above; long-duration testing has observed timeouts.

Exact seeds: `20261012 / 20261013`. The initial generation budget is 1, preserving the original initial output and residual scale; backbone initialization is checked tensor by tensor.
