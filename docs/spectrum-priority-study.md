# P32: Spectral-First Multi-Resolution Supervision

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

2026-10-20. As confirmed by user, **spectrum fidelity is now the primary training and evaluation objective, with waveform error serving as auxiliary evidence of perceptual quality**. It is not valid to reject a candidate solely due to a minor increase in per-sample waveform error. Nor should approximate energy per band be equated to spectral shape, transient fidelity, or audio quality restoration. All waveform and spectral results from [P31](phase-coordinates-study.md) are retained. The new priority does not retroactively modify prior experimental protocols or conclusions.

## New Supervision Objective

The predicted FFT256 complex spectrum is first synthesized using the same OLA (Overlapping and Adding) process as the original reference, then analyzed using 256, 1024, and 2048-point Hann STFTs with hop sizes equal to one-fourth of the FFT size. PCM serves as an intermediate representation for multi-resolution spectral analysis, and no per-sample waveform fitting term is computed. This ensures supervision focuses on synthesizeable audio spectra, avoiding improvements that only appear on a single-frame spectrum output.

The reference is strictly aligned **original, unclipped clean PCM**, without any DSP processing such as detection masking, gating, limiting, or model prediction. Paired and clean identity are treated equally in weighting; all candidates use the same reference and normalization.

| Spectral Component | Weight | Purpose |
|---|---:|---|
| Relative magnitude spectrum error | 1.0 | Recover correct energy at frequency locations and spectral shape |
| Smoothed log-spectrum error | 0.25 | Provide additional supervision for weak harmonics or spectral details |
| Per-band energy envelope error | 0.1 | Recover time-varying energy dynamics |
| Magnitude spectrum time-differential error | 0.25 | Track transient dynamics of the reference, not reward smoothing of real changes |

All three scales are equally weighted. Frequency bands remain 0–4, 4–8, 8–12, 12–16, and 16–24 kHz, allocated according to physical sampling rates of 44.1/48 kHz; low-frequency bands are assigned 0.25 weight, while higher bands are assigned 1.0, normalized by 4.25. Single-sided spectra are calibrated to RMS per bin using Parseval and Hann window energy normalization.
The denominator floor is fixed at 1e-4 plus 1e-12 (absolute power), derived solely from the reference; log terms use per-band per-bin floors. Smoothed norm is defined as `sqrt(x + 1e-6) - 1e-3`.
This floor is not a post-hoc pass threshold per track; all weak bands report both relative and absolute errors.

The supervision region is fixed at `[89*128, frames*128-128)` and matches the original native recheck. Analysis uses `center=False`, no zero-padding, no out-of-bounds extension, and at least two full windows per scale. The first sample includes the contribution from the previous OLA frame; this frame falls within the prediction protection zone and is not part of the 80-frame warmup. Reference-synthesized rounding errors are independently verified and cannot be artificially zeroed by shifting or clipping the reference.

The 2048-point window is used only for offline training and scoring; **it does not increase deployment FFT size, lookahead, or buffering**. The model continues to infer using causal FFT256/hop128, and performance under continuous streaming is independently measured.

## Fixed Baselines

| Group | Optimization Target |
|---|---|
| control | P29/P31 raw direct objective + original budget auxiliary terms |
| mr_add | Original direct objective + new multi-resolution spectral objective + original budget auxiliary terms |
| mr_primary | New multi-resolution spectral objective + 0.02 direct complex fidelity term + original budget auxiliary terms |

All groups use the **same 431,015-parameter, 96-width relative coordinate model**, with phase input disabled; shared parameters and initialization are identical. Each group runs for **1,024 steps**, Adam with learning rate 1e-4, gradient norm capped at 5, and budget auxiliary term set to 0.1.
No model is selected based on intermediate performance, no early stopping, and no extension of any single candidate.

`control → mr_add` isolates the addition of multi-resolution supervision; `mr_add → mr_primary` removes the old single-scale magnitude, envelope, and dynamic terms and reduces the direct complex term from 0.1 to 0.02—this is a clear re-prioritization, not a simple weight ablation. No optimization of DSP-posted scores or per-sample waveform distance is performed.

The full set of 20 fixed training clips from P29/P31 is retained: 16 music, 4 speech, with four music and one speech per batch, plus clean identity. Each batch cycles four times. Supervision duration remains approximately 7.32 seconds for music and 1.85 seconds for speech.
This is a **diagnostic of learnability, not a proof of full corpus training, convergence, or generalization**. First, we verify whether the spectral objective can be learned, then decide whether to proceed to full cross-recording training. Repeated rechecks of these clips do not constitute proof of overfitting.

## Unified Evaluation and Reproducibility

All six final endpoints are frozen before unified scoring. The `spectrum-review/` directory compares the following four paths under the same multi-resolution spectral objective: unprocessed compressed input, identity DSP, original model OLA, and actual Rust frontend → ORT → DSP output. Per-scale, per-band, per-component, and per-track results are preserved separately for music/speech, paired/identity; no single weighted score is considered, nor are different group totals directly compared.

Spectral total score and magnitude, shape, energy, and dynamic fidelity are primary evidence of recovery. Phase, PCM, low-frequency, high-frequency (absolute), channel consistency, and continuity remain auxiliary and integrity indicators. Minor waveform degradation does not negate spectral gains.
No automatic weight promotion occurs, and training clip scores are not interpreted as perceptual or independent test conclusions.

After 240 full native rechecks, an additional 240 full native rechecks are run for multi-resolution spectral evaluation. Subsequently, all six endpoints undergo mono/stereo, 44.1/48 kHz, and two additional stereo 48 kHz tests, plus paced stereo, totaling 420,000 timed hops. All results with >5 ms or hop-level timeouts are preserved. A fixed 256-frame buffer is reported separately; the new loss has no deployment overhead and does not imply actual tail latency meets specification.

```bash
rtk proxy env PYTHONPATH=training training/.venv/bin/python \
  training/scripts/run_spectrum_priority_study.py \
  --parent data/runs/phase-coordinates-p31-20261019/study \
  --output data/runs/spectrum-priority-new/study \
  --native-binary data/runs/small-mechanisms-p28-20261019/binaries/soundex-context-replay \
  --prepare-only
```

After registration, the full `source/training` is copied using `source_hashes`, and the frozen snapshot is executed with `--prepare-only` removed. Source code, data, binaries, and endpoint hashes are all bound to registration; failures are preserved as evidence and not overwritten.

## Current Verification Status

Implementation, targeted numerical testing, and independent four-step prechecks have been completed. True 20-track clip reference validation confirms:
- Maximum multi-resolution spectral loss: **1.91e-8**
- Maximum gradient magnitude: **2.49e-7**
- All finite values; 80-frame warmup gradient is zero.

Synthetic signal testing covers correct reference, silence, weak energy finite-difference, equal energy at wrong frequency, real transients, phase reversal without magnitude change, and DSP masking independence.

A total of two pre-processing failures occurred during full testing, due to the system FFmpeg lacking `libx265.216.dylib`. Isolated revalidation using the project's existing imageio-ffmpeg 7.1 passed all 620 test cases (including 12 new ones). Test environment paths and outputs are preserved in the precheck directory.
Isolated revalidation passed **620 test cases** (12 new). Ruff lint/format passed. No system dependencies or existing training pair data were modified.

Precheck directory: `data/runs/spectrum-priority-p32-preflight-20261020/`. All three groups completed four-step, 120 native consistency roles and 120 multi-resolution native scoring roles; maximum synthesis error was 2.38e-7, with zero tolerance failure.
The first four steps of control show identical loss and gradient records as in P31. Precheck endpoints are not included in the official comparison and are not used for group selection.

**All six 1,024-step diagnostics and full rechecks are complete**: `data/runs/spectrum-priority-p32-20261020/`.
178 source code and configuration dependencies are frozen. Code commit: `9050d4a82417881341129742719329ff92e5c075`. Registration SHA-256: `96256e92472c7a36b54e107059e4af4a65d87a3513fcc5133a5f5d05ea1d5748`.
Ended at 04:45:06 (Singapore time), total process duration ~33 minutes 34 seconds. Supervision process exit code: 0. Remote CI for code and documentation passed. Post-hoc aggregation scripts and results are saved locally at `evaluation-20261020/`. 307 hash bindings were re-verified, including 178 frozen source codes/configurations, input data, six endpoints, scoring reports, and 42 raw timing reports. Default weights were not replaced.

## Final Results: Spectral Improvement Confirmed, Scope Limited to Fitting Clips

The table below shows per-track mean values under the same MR v1 objective and original, lossless reference. Lower is better. The two numbers per row correspond to seeds 20261012 and 20261013. No cross-optimization target comparison of total loss is performed.

| Path / Configuration | Music (Original OLA) | Music (Actual Rust + ORT + DSP) | Speech (Actual Rust + ORT + DSP) |
|---|---:|---:|---:|
| Unprocessed Compressed Input | 0.230262 | 0.230262 | 0.229617 |
| Identity DSP (Control) | — | 0.229449 | 0.229617 |
| control | 0.190957 / 0.190064 | 0.194972 / 0.194151 | 0.215313 / 0.214404 |
| mr_add | **0.182656 / 0.182574** | **0.190388 / 0.190225** | **0.212683 / 0.211707** |
| mr_primary | 0.183585 / 0.184070 | 0.190942 / 0.191822 | 0.217605 / 0.219406 |

`mr_add` shows a **4.35% / 3.94% reduction** in magnitude spectrum error compared to the same budget control for music original OLA, and a **2.35% / 2.02% reduction** for actual deployed output. For speech, the deployed output shows a **1.22% / 1.26% reduction**. Relative to unprocessed input, music deployment spectral scores dropped by **17.32% / 17.39%**, representing training clip-level error improvement—not a general audio quality improvement percentage.
Of the 16 music tracks, 10/11 outperformed control, 1/1 matched, and 5/4 degraded. 14/15 outperformed the input. For speech, 3 out of 4 tracks outperformed control.
The worst relative increase over control for music was only **1.14% / 1.15%**, with all counterexamples preserved and no threshold-based filtering applied.

Per-component changes for music actual deployment under `mr_add`, relative to the same seed control, with negative values indicating improvement:

| Component | Seed 20261012 | Seed 20261013 |
|---|---:|---:|
| Magnitude spectrum error | -2.30% | -1.83% |
| Log-spectrum shape error | -3.94% | -3.47% |
| Energy envelope error | -11.74% | -7.53% |
| True time variation error | **+0.76%** | -0.31% |

Across all three FFT analysis scales, the 8–12, 12–16 and 16–24 kHz bands show magnitude error improvements in both seeds, with per-bin reductions ranging from ~1.97% to 6.34%. This is not limited to total energy in a single FFT band. However, time-variation gains remain unstable, and both seeds show higher dynamic errors than the unprocessed input (input: 0.111535, mr_add: 0.115889 / 0.115655).
Thus, we cannot claim transient recovery has been achieved. 0–4 kHz magnitude error changes are near zero (-0.001% to +0.011%), and absolute PCM low-frequency error remains under review.

`mr_primary` shows a lower music total score than control, but speech scores increase by **1.06% / 2.33%**. All four speech tracks under `mr_primary` underperform control. One seed shows a 6.11% increase in 16–24 kHz magnitude error for music.
Therefore, current evidence supports further validation of the **mr_add combination supervision**, and does not support a significant weakening of the prior spectral supervision and complex fidelity term. This comparison represents a combination change and cannot be attributed solely to phase weighting.

The difference in music total score from original OLA to deployed output: control ≈ 0.0040, mr_add ≈ 0.0077.
The full deployment chain retains partial gains but also offsets some—this cannot yet be attributed to any specific gate, budget header, or limiter. Component-level ablation is required.
No direct removal of DSP or alteration of reference should be used to improve scores.

## Waveform, Fidelity, and Consistency Auxiliary Evidence

Music full-band PCM RMS error:
- Input: 0.009219998
- Identity DSP: 0.009223736
- Control: 0.009684719 / 0.009706785
- `mr_add`: **0.009927955 / 0.009896091**
- `mr_primary`: 0.009992523 / 0.009992904

`mr_add` shows a ~2.51% / 1.95% increase relative to control and a ~7.68% / 7.33% increase relative to input. This represents a phase and waveform cost that must be retained and addressed further. It cannot be claimed as improved, nor can this metric alone be used to deny spectral gains.

`mr_add` 0–4 kHz absolute PCM RMS error: 0.005953493 / 0.005952591, very close to control's 0.005952284 / 0.005952965.
Per-band RMS/dBFS, per-track results, and original reference power are preserved in native reports.
MR spectral score is not dBFS and cannot be used to infer whether weak-band noise is audible.

Clean identity music deployment MR: 0.006214 / 0.006148; identity DSP itself: 0.005492.
Speech: 2.13e-5 / 9.07e-6. This confirms that the system still produces changes on inputs that do not require repair.

All 240 native consistency roles passed synthesis tolerance, with maximum synthesis error 2.98e-7. An additional 240 actual native roles completed multi-resolution scoring. This is code-path and numerical evidence—not blind listening, physical device, or independent recording generalization.
1024 steps only completed pre-registration budget; end logs still show variation, so convergence is not claimed.

## Real-Time Performance: Low Typical Latency, Long-Tail Not Met

| Scope | Hops | p99 Range | Max Computation Time | >5 ms Count | >1 hop period count |
|---|---:|---:|---:|---:|---:|
| All six endpoints, 42 tests | 420,000 | 0.169–1.406 ms | **45.003 ms** | **91** | **221** |
| Among 48 kHz stereo paced | 60,000 | 0.651–1.060 ms | **8.330 ms** | **7** | **36** |

Paced mode had 619 late completions, maximum delay 12.713 ms; completion latency includes scheduling wait and cannot be simply added to computation time, nor can it be equated directly to physical device clipping.
`mr_add` paced p99 values were 0.683 / 0.651 ms, both below 5 ms, but each scenario had 12 instances of >5 ms.
No single paced result can be used to claim compliance.
All test outputs are finite; peak RSS was 30.88–33.30 MiB. Paced CPU usage was ~10.42%–17.73% (user+system/wall, including initialization and warmup), not net model memory or overall system CPU.
No fixed time constraints, idle host, or load comparison was performed—thus, long-tail cannot be attributed to model computation, system scheduling, or external load.

Fixed buffer remains at 256 frames, i.e., 5.333 ms at 48 kHz and 5.805 ms at 44.1 kHz, plus computation and scheduling overhead. Multi-resolution loss does not increase deployment buffer.
**Typical computation has margin, but strict <5 ms and continuous hop-by-hop delivery has not passed all tests.**

## Next Steps

Retain `mr_add` and match `control`. The next round will first register a broader training/development audio split, with identical sufficient budget and fixed spectral metrics. Focus will be on validating cross-recording gains and transient degradation.
No further optimization on these 20 fitting clips will be pursued before claiming generalization.
Parallel work will focus on component-level ablation of fixed-endpoint DSP modules, and on locating real-time long-tail with scheduling and load evidence.
Generalization, transient recovery, continuous stream, and blind listening evidence will be required before formal weight promotion.
This round only completed result validation and archiving; no training has been initiated, and no model has been promoted.

**Next update**: Upon user authorization, the [P33 Broad-Spectrum Study](broad-spectrum-study.md) has been launched, retaining `control`/`mr_add`, identical initialization, and more generous budget. Additionally, P32 actual native performance has been completed in phased localization.
All original P32 results and protocols remain unchanged.

The historical protocol date is 2026-10-20. Both seeds 20261012 / 20261013 were retained for every arm. The per-bin magnitude reductions of 1.97%–6.34% are improvements; the range is not a positive error increase.
