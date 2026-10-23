# P23: Capacity Comparison Under Full Training Budget

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

The user explicitly requested on 2026-10-15 to complete sufficient training before evaluating large model performance.
The results from [P22](latency-capacity-study.md) at 512 steps are retained for early diagnosis; this round of training is **not** halted due to failure of the 512-step check, nor is an increase in initial audio degradation interpreted as the ultimate performance ceiling of the large model.

## Final Results (2026-10-16 09:13 SGT)

All eight trajectories have completed 16,192 steps, totaling 129,536 updates. After 24 endpoint freezes, 4,450 Rust role replays, development/recheck validations, and export cost measurements were completed. The input, source code, sampling, checkpoints, and 57 finalized review reports were re-verified using a frozen analyzer. The maximum Rust/proxy error is `1.08034e-5 < 2e-5`; all comparisons failed full quality checks, and no formal weights were replaced.

This stop is the **official end of a pre-registered experiment**, not an early stop at 512 steps or a failed execution.

### Full Training Pool & Seed Averages:
- Identity initialization (main): **0.292682**
- Anchored / Scaled224 / Scaled384 / Scaled640: **0.284743 / 0.287859 / 0.288519 / 0.290059**

Longer training indeed reduced training error, but this did not translate into overall gains in development music quality.

Below are the **16,192-step, two-seed averages, aggregated across 33 music development clips**. These are reused development rechecks, not independent tests. Lower error is better. The table presents proxy and actual DSP metrics separately—different metric types are not conflated into a single objective:

| Configuration | proxy paired | proxy identity | DSP High-Frequency SC | DSP High-Frequency Flux | DSP High-Frequency Energy Error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Identity Model + DSP | 0.588365 | 0 | 0.548761 | 0.717164 | 8.4801 dB |
| Anchored Control | 0.624674 | 0.059516 | 0.612967 | 0.755468 | 8.1772 dB |
| scaled224 | 0.823622 | 0.209859 | 1.615558 | 1.491498 | 7.2266 dB |
| scaled384 | 0.986274 | 0.334769 | 2.057559 | 1.783899 | 7.4695 dB |
| scaled640 | 0.951500 | 0.306329 | 1.901090 | 1.684844 | 7.7538 dB |

Aggregation by recording also failed to show improvement. Lower energy error does not imply better spectral shape or transient recovery.
For speech, the total proxy score for scaled models ranges from **0.18060 to 0.18063**, compared to initial **0.180645**—a negligible gain.
For shifted music, the total score ranges from **0.30184 to 0.30286**, compared to initial **0.301122**—also showing no overall benefit.

However, it is incorrect to claim that every music segment degraded: for each scaled capacity and each seed, **29–30 out of 33** clips show improved paired proxy metrics compared to initialization. The main anomalies are concentrated in the same clip `d2599868e6a472eff47c1674`, starting at position 40244. Two counterexamples contributed up to **101–106% net degradation**, though the rest of the improvements partially offset this degradation.

This known counterexample **must not be deleted**, nor can post-hoc threshold adjustments be applied to allow passage. It is also invalid to infer global waveform burst or perceptual degradation from average relative error alone.

In this counterexample, the 12–16 kHz and above reference power levels are approximately **-111 dB and -119 dB**, respectively, within the current frequency band metric space. Taking the Scaled640 configuration and seed `20261012` as an example:
- Identity actual DSP output high-frequency SC: **28.86** (vs. original **34.25**)
- Paired values: **31.83 / 34.95**

Relative metrics amplify errors under weak references, but they also document the generation of high-frequency content that should have been preserved. The DSP did not eliminate this issue, and consistency between proxy and deployed outputs remains unverified.

These power readings are **not calibrated waveform dBFS values** nor subjective listening conclusions. Further validation via absolute error and human listening is required.

### Final End-to-End Latency (Single-threaded CPU ORT + Causal FFT Preprocessing)
- Two-channel, 10,000 hop per seed:

| Configuration | p99 Range | Observed Maximum |
| --- | ---: | ---: |
| scaled224 | 0.300–0.429 ms | 2.440 ms |
| scaled384 | 0.567–0.586 ms | 1.142 ms |
| scaled640 | 1.104–1.175 ms | 2.466 ms |

The virtual per-hop timeout count for scaled endpoints remained zero—this does **not** imply that the full real-time DSP worker or hardware has met the 5 ms budget. No blind listening, continuous physical device streaming, or independent new recording evaluation has been conducted.

### Conclusion:
Increasing training budget is beneficial, but under the current full budget and fixed optimization settings, **wider scale alone has not resolved failure in weak energy bands**. Future work should prioritize auditing absolute and relative errors in low-frequency bands, output residual scales, and original audio fidelity before deciding on method adjustments.
It is inappropriate to continue scaling based solely on training loss, nor can we conclude that all longer training is ineffective. Original evidence is preserved in `study/comparisons.json` and `study/review/`.

## Fixed Scheme

| Configuration | Parameter Count | Steps per Seed |
| --- | ---: | ---: |
| Anchored144 Control | 807,398 | 16,192 |
| scaled224 | 1,748,578 | 16,192 |
| scaled384 | 4,594,818 | 16,192 |
| scaled640 | 11,917,698 | 16,192 |

Two seeds: `20261012` and `20261013`, totaling **129,536 updates**. Each configuration's budget is **31.625×** that of P22 and equals the full budget of P20. Execution order: **Scaled640 → Scaled384 → Scaled224 → Anchored**. After completing all configurations for the first seed, the second seed is executed.

All training begins from the same identity initialization. The first 512 steps reuse the identical sampling prefix from P22. No checkpoint selection is made based on performance, and no P21/P22 registration, checkpoints, or conclusions are overridden. Due to the lack of element-wise tensor sharing across different widths, per-seed and per-width backbone initializations are still verified via hash.

The full 16,192-step sampling sequence and 2,088-crop training pool from P20's frozen state are reused, along with a common 320-crop set.
253 music recordings and 8 speech recordings, with 4 music and 1 speech per step, paired/identity-weighted.
Model architecture, DSP proxy, main loss, clean-complex auxiliary term, and weights remain unchanged. The scaled head continues to use `2*tanh(z/64)`, with fixed Adam learning rate at `1e-4`, no weight decay, and gradient norm clipping at 5.

This round isolates the effect of training budget—no re-tuning of loss, learning rate, or hyperparameter search based on development set performance.

No early stopping is triggered by 512 steps, training loss, or clipping ratio. NaN/Inf, input/source drift, or runtime failures will halt execution and record errors. A checkpoint is saved every 64 steps, including Adam and RNG state. Reaching the budget does **not** imply convergence; final decisions on whether further training is warranted must be based on learning curve analysis.

## Post-Training Evaluation

Fixed at 2,048 / 8,192 / 16,192 endpoints, with primary conclusions based on **16,192 steps**. Intermediate endpoints are used for trajectory comparison only, and no endpoint is selected as optimal. No quality monitoring during training is used to decide which scale to retain. Every 2,048 steps, the shared training pool is observed; at the end, the full training pool is observed for each endpoint.

After all eight trajectories complete, validation reports and total update counts are verified, and **24 endpoint hashes are then frozen**. Following this, the system automatically executes:

- **89 development paired clips** and **40 shifted clips** for proxy evaluation. These recordings were previously reused and are explicitly labeled as development rechecks—**not** new independent tests.
- Initial control plus 24 endpoints: all development clips undergo actual Rust DSP replay, totaling **4,450 roles**.
  For each, the model's raw output and post-DSP high-frequency spectral shape, energy, transient, low-frequency fidelity, and original audio damage are analyzed, separately by music and speech, aggregated by clip and recording, and the worst-case examples are preserved.
- Each endpoint undergoes non-zero ONNX continuous state consistency checks: two-channel CPU inference with causal FFT preprocessing (10,000 hop), and independent ORT RSS snapshots.
- Comparative analysis across all scales and against the anchored baseline, with attention to differences between 224→384 and 384→640 under same budget.

Full numerical validation, blind listening, independent new recording evaluation, and continuous physical device validation remain distinct phases.
The current diagnostic export does **not** include full real-time DSP worker timing. It cannot be used to claim that the model + DSP meets the user’s <5 ms computation budget. All new weights remain experimental and are **not automatically promoted** to production.

## Execution

Command:
```sh
PYTHONPATH=training training/.venv/bin/python training/scripts/run_full_capacity_study.py \
  --parent data/runs/latency-capacity-p22-20261015/study \
  --p20 data/runs/head-p20-20261014/study \
  --output data/runs/full-capacity-p23-20261015/study
```

Local directory: `data/runs/full-capacity-p23-20261015/`
Source code and audio/sampling inputs are frozen with hash.
Training and auto-recheck status are tracked in `study/progress.json`, supervision status in `pipeline.json`, and logs in `pipeline.log`.
Original training materials and checkpoints are not pushed.

## Launch Log

- Started at **23:01 SGT, 2026-10-15** on local machine, with source code hash `4fa004c` frozen.
- Historical supervision PID `98142` (now terminated normally) was replaced by the new run using the same directory's frozen source.
- Actual subprocess and stage progression are governed by `pipeline.json` and `study/progress.json`.
- First run: **Scaled640**, seed `20261012`.
- Full Python test suite passed 505 cases; Ruff and formatting checks passed.

The registered seeds are `20261012 / 20261013`. Execution uses `training/scripts/run_full_capacity_study.py` from the frozen `source/training` tree. The subsequent [absolute-error/stage audit](capacity-outlier-audit.md) retains the original scores while separating small absolute high-band generation from amplified relative errors above 16 kHz; no thresholds or release status change.
