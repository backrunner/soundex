# P34: Loss Scale and Identity Preservation Diagnostics for a Single Small Model

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

2026-10-21. Based on the [P33 Mechanism Diagnosis](broad-spectrum-diagnosis.md), the user requested a preliminary investigation using a single, small-parameter model, with actual training results to guide iterative tuning. This round uses a fixed width of 96, with a total of 431,015 parameters and a consistent architecture. Only the training objective is compared; no expansion, no DSP modifications, and no default weight promotion are performed.

## First Round: Initial Setup

| Configuration | Direct Spectral and MR Risk | Paired Coefficient | Identity Coefficient | Budget Auxiliary |
|---|---|---:|---:|---:|
| control | Maintains original P33 mr_add target | 0.5 | 0.5 | 0.1 |
| compressed | Each sample takes log1p of two separate targets | 0.5 | 0.5 | 0.1 |
| compressed_identity | Same as above | 0.5 | 0.05 | 0.1 |

`log1p` is a smooth, monotonically increasing, dimensionless compression of risk, with derivative `1/(1+risk)`. No samples are removed, clean references are unchanged, and risk is not clipped to zero. This compression also weakens gradients for difficult real recovery samples, and must be evaluated using the original, uncompressed MR v1 and all counterexamples. Compression does not guarantee resolution of scale issues across frequency bands. The identity coefficient of 0.1 is a tenfold sensitivity contrast from the pre-registered baseline—not a presumed optimal value or published threshold. The paired coefficient is not re-normalized upon identity changes; the budget head objective and auxiliary coefficient remain unchanged.

All three groups start from the same seed `20261012` and the same initial tensor; Adam optimizer with learning rate 1e-4, no decay, global gradient norm capped at 5, for **1,024 steps**, trained sequentially. Fixed at 16 batch cycles, each batch contains 4 distinct music pieces and 1 speech sample, plus corresponding clean identity. 80-frame no-grad warmup period, with supervision region retained as `[89*128, 32768-128)`. Evaluation occurs only after all endpoints are frozen—no early selection, pruning, or budget adjustment during training.

This is a learnability diagnostic under a single seed and a finite, curated dataset, and cannot be used to declare convergence, sufficient training, capacity limits, or architectural superiority. After completion of all three groups, a new corrected experiment is registered based on shared scoring; historical results are not overwritten.

## Data and Acceptance Criteria

The fitting pool is selected from the full update sequence of the P33 seed `20261012`, with 16 batches sampled at equal intervals, matching the mechanism diagnosis: 64 music clips / 60 speech recordings, 16 speech clips / 16 speech recordings. After removing warmup and overlaps, the independent supervision time is **29.97 seconds of music, 7.28 seconds of speech**; it is not valid to claim that the entire 253-song music library participated in this training. Each trajectory has 4,096 music and 1,024 speech clip exposures, with repeated exposures not counted as new coverage.

Additionally, 16 batches of training domain recheck fragments are selected: 64 music clips / 59 speech recordings, 16 speech clips / 15 speech recordings, with independent supervision time of **29.43 seconds of music, 7.40 seconds of speech**. All segments are verified by source checksum and source time. The full input window including warmup does not overlap with the fitting pool; overlapping candidates are only considered by shifting backward to the next batch, not by score. Speech clips may overlap, and all are drawn from existing training domain data—thus, they cannot be considered independent validation or cross-recording generalization.

Mono, left, right, mid, side, and all weak-energy cases are retained, including the side fragments that dominated the previous average. The fitting pool contains only two mono music clips; the recheck pool contains no mono music. True stereo synchronization is not covered. Nine encoding bitrates and sample rates of 44.1/48 kHz are retained; no new sources, raw data, or licensing changes are introduced.

Each endpoint is evaluated separately: original OLA, identity DSP, actual Rust frontend → ORT → DSP. Results are reported separately for fitting vs. non-fitting training domains, music vs. speech, paired vs. identity objectives. The original MR v1 and all component-level metrics—including per-recording mean, median change, absolute PCM and band-level errors, and all counterexamples—are preserved. The total loss across different optimization objectives is not ranked. All actual native playback and standard hop-by-hop baselines are maintained; latency remains subject to the same requirements: <5 ms computation, 2.667 ms hop throughput, and fixed 256-frame buffer. Without real device or blind listening evidence, the experiment remains in a reported state.

## Implementation and Pre-Check

Loss entry point: `training/models/risk_balance.py`
Training entry: `training/scripts/run_risk_balance_study.py`
Result analysis: `training/scripts/analyze_risk_balance_study.py`
The same old objectives, model, and deployment chain are retained; new experiments only modify risk aggregation.

**44 tests passed**, including risk derivative, finite difference, silence/weak-energy reference gradients, high-frequency drop direction, identity weight not altering paired gradient, and verification of existing MR/direct spectral/budget exports.
Each group undergoes independent pre-check for two steps, with exports and 60 complete native PCM and 60 complete native MR roles generated. Pre-checks validate the pipeline only and are not used for group selection or quality judgment.

First round directory: `data/runs/risk-balance-p34-20261021/`
Pre-check directory: `data/runs/risk-balance-p34-preflight-20261021/`
Original audio, experiment weights, and detailed reports are retained in ignored directories. Audio, sampling, 184 execution dependencies, and endpoint hashes are registered, and the execution source code is frozen.

```sh
rtk proxy env PYTHONPATH=training training/.venv/bin/python \
  training/scripts/run_risk_balance_study.py \
  --parent data/runs/broad-spectrum-p33-20261020/study \
  --output data/runs/risk-balance-p34-20261021/study \
  --native-binary data/runs/small-mechanisms-p28-20261019/binaries/soundex-context-replay
```

## First Round Results

All three configurations complete 1,024 steps, with **960 roles** of native PCM and additional MR playback completed. The maximum synthetic error is **4.17e-7**, with zero tolerance failure. The following values are based on original MR v1 and lossless reference, not compressed scores; means are computed equally across recordings, while medians are per-recording relative input changes.

| Configuration | Fitted Music Deployment MR Mean | Fitted Music Median Change | Non-fitted Training Domain Music Deployment MR Mean | Non-fitted Music Median Change |
|---|---:|---:|---:|---:|
| control | 2.236254 | −1.119% | 0.495104 | −0.709% |
| compressed | 2.234127 | −0.286% | 0.497587 | −0.121% |
| compressed_identity | 2.218000 | −4.725% | 0.495518 | −1.082% |

Only the compressed configuration shows conservative error behavior. The non-fitted music mean is **0.501% worse** than control.
The identity weight of 0.1 recovers more spectral shape: 49 music pieces outperform control, 8 underperform; 35 music pieces outperform, 24 underperform, though the non-fitted mean remains **0.084% worse** than control.
It is not valid to declare overall superiority over control based on median or win count; absolute scores across different data sets cannot be cross-ranked.

For the 0.1 identity weight configuration, the envelope and log-spectrum error of non-fitted music show improvement, though amplitude and dynamic errors increase. The lossless identity deployment MR increases from control's 0.015594 to **0.032744**, and the RMS perturbation relative to identity DSP increases from **3.44e-5 to 1.49e-4** (per-recording mean). This absolute metric does not equate to audible quality, but the preservation trade-off is clear—no promotion is justified. The non-fitted speech spectrum mean improves by 3.07% and requires further evaluation alongside music, transients, and identity.

Weak side examples remain in the fitting pool. Although the original OLA mean appears to improve significantly, most of the benefit vanishes upon deployment. The raw mean cannot replace final recovery performance. All other samples, components, and native absolute band-level errors are preserved in `assessment.json`, `spectrum-review/`, and `native-audit/`.

Optimization mechanism is validated: control triggers 50 gradient clippings, with maximum norm **41.19**; compressed and compressed_identity trigger zero clippings, with maximum norms of **2.05 / 1.88**. Additionally, on the frozen P33 endpoints, a single-sample reconstruction gradient for a known anomalous side fragment drops from 50.43 to **0.458**, while normal samples still retain effective gradients. However, optimization stability does not imply better recovery performance—this round has not resolved the trade-off.

Standard baseline: 21 processes, **210,000 hops**. Zero instances ≥5 ms, maximum **2.744 ms** (control), p99 maximum **0.785 ms**, one service exceeds hop, 4 paced completions are delayed; peak RSS reaches **31.875 MiB**. This is not a physical device validation and does not cover long-tail counterexamples from prior P33. The fixed 256-frame buffer remains at 5.333 / 5.805 ms, and is counted separately. The full first-round pipeline is complete; default weights remain unchanged.

## Second Round: Intermediate Identity Weight

Based on first-round results, we introduce **compressed_identity_mid**: identity relative weight set to **0.3**, maintaining paired coefficient at 0.5 and identity coefficient at 0.15. This is a post-hoc addition and registered prior to training, representing the next debugging step—not a fabricated fourth group, nor can reused recheck fragments be called a new held-out set.

Starting from identical initialization, using the same fitting and recheck lists, architecture, optimizer, and 1,024-step budget; the frozen endpoints from the first round are reused for equal-budget comparison. Budget auxiliary, DSP, and shared scoring remain unchanged. The first-round assessment, registration, and endpoint hashes are recorded and verified for data, budget, and structure consistency.

Second round directory: `data/runs/risk-balance-p34-followup-20261021/`
The first-round source code is frozen in the original directory; subsequent replay must use its `source/training` and cannot use modified source code to pass old dependency checks.

Second round completed: music deployment MR, fitted pool at **2.227201**, non-fitted training domain at **0.493596**, both down from control by **0.405% / 0.305%**. Median changes: −2.680% / −1.081%; 41 music pieces outperform control, 18 underperform. For non-fitted music, amplitude, envelope, and log-spectrum improve by 0.322% / 0.123% / 0.657%, though dynamic error increases by **0.210%**, and identity MR increases by 3.017%. The RMS perturbation relative to identity DSP is **3.27e-5**, slightly below control's 3.44e-5. Native lossless PCM error is preserved and cannot be conflated with DSP control as a primary objective. Speech paired MR improves by 0.062% over control—no broad domain improvement is observed.

Second round completes 320 complete native PCM roles and 320 MR roles, with synthetic maximum error at **2.98e-7**, zero tolerance failure. 70,000 hops: maximum **1.494 ms**, p99 maximum **0.787 ms**, zero instances ≥5 ms, zero service over hop, zero paced completion delay. Single-baseline does not replace device validation.

### Post-hoc Splitting of Missing and Overfilled Spectra

Actual Rust playback is performed for all four endpoints across all 160 paired clips, resulting in 640 roles. Using identical 256/1024/2048 Hann/Parseval scaling, and classifying frequency bands (4–24 kHz) by whether the original input amplitude is below, above, or equal to the reference, the error is computed per category. Classification does not read candidate predictions. Exact equality (including silence) frequency points are preserved and not excluded by threshold.

This is a normalized squared amplitude error, not the total MR score, nor wave-level PCM dBFS. Absolute calibrated amplitude spectrum error is retained, and true PCM and band-level RMS are preserved in native reports. Evidence is stored in `data/runs/risk-directions-p34-20261021/` and is not used to retroactively modify original scores or thresholds.

For the intermediate weight group, on non-fitted music: the error for "missing" decreased by **1.561%**, but the new overfill term increased from 0.001974 to 0.005743, causing the total normalized squared error for missing points to increase by **1.109%**. The absolute amplitude spectrum error for missing points decreased by 1.214%, indicating that relative scale of weak references affects judgment—thus, the relative increase cannot be interpreted as audible severity. For overfilled points, the normalized error decreased by **2.063%**. Therefore, the overall MR improvement is modest and cannot be fully attributed to successful inpainting of missing spectra.

## Third Round: Constraining Only Overfilled Points

On the foundation of the second round—fixed structure, data, initialization, 1,024 steps, and identity relative weight 0.3—we introduce `compressed_identity_guard`. The auxiliary term applies a smooth, normalized amplitude constraint to `max(predicted_amplitude - reference_amplitude, 0)` at frequencies where the original input amplitude is below the reference, using the same smoothing and normalization as the original MR. The constraint is applied across three scales and four high-frequency bands, with equal weighting; the total paired coefficient is `0.5 × 0.1`. Thresholds are not adjusted based on track, counterexample, or candidate outcome.

This auxiliary term has no gradient when the predicted amplitude does not exceed the reference—thus, it does not directly penalize normal fill-in. Shared parameter changes may indirectly influence other points, so actual training must be performed to validate. The reference is only used in training supervision. The original direct + MR loss continues to handle missing energy, overfilled points, silence, phase, and dynamics—this auxiliary term cannot replace the overall objective.

Current test suite: **47 tests passed**, including new validations for reference/underfilled zero gradient, normal/weak-energy overfill direction and finite difference, and silence behavior.
Pre-check and training directories for this round: `risk-balance-p34-guard-preflight-20261021/` and `risk-balance-p34-guard-20261021/`.
The third round is complete. Results do not outperform the second round in overall trade-offs.
Fitted music deployment MR: **2.219717**, non-fitted training domain: **0.494063**, improving by 0.739% / 0.210% over control. However, the non-fitted music performance regresses by **0.095%** compared to the intermediate weight group, with only 32 pieces outperforming control and 27 underperforming. Dynamic error increases by **1.346%** over control, and identity MR increases by **42.687%**. The RMS perturbation relative to identity DSP reaches **6.96e-5**, higher than the intermediate weight group's 3.27e-5. Speech paired MR improves by 0.847%, but fails to offset degradation in music and fidelity.

320 complete native PCM roles and 320 MR roles are completed, with synthetic maximum error at **2.98e-7**, zero tolerance failure. 70,000 hops: maximum **1.927 ms**, p99 maximum **0.794 ms**, zero instances ≥5 ms, zero service over hop, or paced completion delay. No gradient clipping occurs; maximum norm is 2.17.

Therefore, the third round's overfill constraint is not included in the default training path and is not promoted. The implementation is retained as an explicit experiment; mathematical verification of reference and gradient is not sufficient to claim improved recovery on new segments. Five configurations, 800 paired native rechecks, are independently validated and results are stored in `data/runs/risk-directions-p34-guard-20261021/`. Missing, overfilled, exact-equal, overfilled, and underfilled relative and absolute spectral amplitude errors are preserved.

This recheck is complete. The actual deployment overfill power for non-fitted music increases by **75.97%** relative to the second round; the average calibrated spectral amplitude error power increases from **6.21e-8 to 2.72e-7**—this cannot be claimed as waveform PCM error or audible severity. The underfilled term decreases by 4.135%, missing point total normalized squared error decreases by 0.939%, but overfilled point error increases by 0.735%; indicating a more aggressive fill and potential for additional artifacts.

To distinguish between implementation error and migration failure, two frozen endpoints are re-computed for the same training auxiliary term: on the original 4:1 music/speech batch, the fitted pool's weighted overshoot drops from **0.00171748 to 0.00091856** (a ~46.5% drop), while the non-fitted training domain increases from **0.00042628 to 0.00091151** (a ~113.8% rise). This supports the conclusion that "the auxiliary term actively participates in fitting, but benefits are not migrated"—not due to a simple gradient direction error. The root-mean-square aggregation, the above-squared error aggregation, and the deployment processing chain all differ and must be reported separately. Evidence is in the third round's `guard-component-audit.json`.

## Final Loop and Future Boundaries

Three rounds, five configurations, each with 1,024 steps, are completed—**5,120 total steps**—maintaining the same 431,015-parameter structure, single seed, identical fitting data, and budget. All five endpoints complete actual output validation before judgment. Full native consistency spans 1,600 roles, with an additional 1,600 native MR scoring roles. 350,000 hops standard timing: zero instances ≥5 ms, maximum 2.744 ms, one service exceeds hop, and 4 paced completions are delayed.

Confirmed progress includes reduced dominance of anomalous gradients and the identification of a slightly better intermediate weight configuration. The non-fitted music MR mean improves by 0.305%—not a significant breakthrough or independent generalization proof. The third round reveals that additional reasonable loss terms may exacerbate fitting and migration trade-offs, and thus cannot be blindly added or iteratively adjusted without caution.

Future work will maintain the small model, prioritize broader training coverage, and validate the intermediate weight scheme with a second seed. Simultaneously, tracking of missing points, overfill, dynamics, and identity will continue. The limited dataset provides only about 37 seconds of independent supervision—insufficient to replace comprehensive coverage. Reused training domain rechecks have already participated in debugging and cannot be considered new held-out validation. No new background training is initiated; production models, DSP, and official weights remain unchanged. Blind listening or real device validation has not been conducted.

For the 0.1 identity-weight arm, 35 unfitted music recordings improved over control and 24 regressed. The three first-round arms each used 1,024 updates of 4 music crops and 1 speech crop plus clean identity. Compressed arms triggered 0 gradient clips. Across all three rounds the 350,000-hop total retained 1 service overrun and 4 paced late completions; successful shorter runs do not erase those failures.
