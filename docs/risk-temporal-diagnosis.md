# P36: Diagnosis of High-Frequency Loss, Temporal Dynamics, and Identity Counterexamples

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

2026-10-22 — Analysis of the frozen P36 `compressed` model (only error compression, identity weight = 1.0) and the corresponding P35 `control` model with the same seed. Seeds are 20261012 and 20261013, respectively. Both models have 431,015 parameters and were trained for 16,192 steps. No weight updates, DSP deployment, or threshold filtering occurred. Full training and real-time results are available in [P36 Two-Factor Comparison](risk-attribution-study.md).

**Conclusion:** The restored performance indeed includes improvements in the recovery of high-frequency content, but the compensation for temporal dynamics remains inaccurate. Dynamic degradation already exists in the raw model output and is further amplified by spectral fusion. The degradation in identity spectrum mean is primarily driven by a single counterexample from a very weak reference band. These are distinct issues and should not be addressed collectively through overall reduction of enhancement magnitude, simple smoothing of output, or removal of low-energy samples.

---

## Scope and Methodology

- All 89 existing held-out verification segments (33 music, 56 speech) were used. For each recording, segments were first aggregated, then averaged per recording. Music comprises 26 tracks; segment count and recording count must not be conflated.
- For the four new endpoints of P36, 712 actual Rust frontend/ORT paired and identity roles were executed, decomposing the original OLA, band-level fusion, unclipped gating, clipping, and native output. The maximum error between stage proxies and native output was `4.17e-7`, with no tolerance failure. 169 frozen Python source files from training were preserved and verified via hash consistency.
- Additional 356 native paired inferences were performed on P35 control and P36 compressed at four endpoints, analyzing input frequency loss/gain and temporal dynamic errors. These are not new training samples or independent test cases.
- All stages were scored against the original, unclipped, lossless reference. Identity DSP was used solely for isolating additional modifications. All counterexamples were retained; no per-track or post-hoc thresholds were applied to weak energy content.

---

## Which Frequencies Were Recovered?

Common MR v1 music native output, per-band relative error compared to the same-seed control:

| Band | Amplitude Error | Log-Spectral Shape Error | Dynamic Error |
|---|---:|---:|---:|
| 8–12 kHz | −0.128% / +0.097% | −1.104% / −0.798% | +0.084% / +0.173% |
| 12–16 kHz | −0.387% / −0.369% | −2.112% / −2.165% | −0.059% / −0.017% |
| 16 kHz to Nyquist | **−3.752% / −3.279%** | **−10.206% / −10.391%** | **+0.964% / +0.933%** |

The 0–4 kHz metrics remain nearly unchanged; 4–8 kHz amplitude variation is approximately −0.001%. Results for spectral shape and amplitude show inconsistency: amplitude is closer to reference, but this does not imply accurate recovery of timing or magnitude changes.

Additionally, missing frequency points were defined based on whether input amplitude was below reference, independent of model output. This mask was fixed and not adjusted by model predictions, and weak frequency components were not excluded. For 256/1024/2048-point Hann analysis and the four bands above 4 kHz, normalized squared amplitude error (based on reference power and fixed normalization floor) is shown below:

| Music Missing Frequency Diagnosis | Seed 20261012 | Seed 20261013 |
|---|---:|---:|
| Total missing region error change (native output) | −3.861% | −3.661% |
| Still deficient amplitude error change | −3.973% | −3.755% |
| Unnormalized missing region error power change | −0.963% | −0.961% |
| New over-amplitude error power (native output) | 5.15e-8 | 5.54e-8 |
| Control new over-amplitude error power | 1.90e-8 | 1.90e-8 |

Speech missing region normalized error also decreased by 8.132% / 8.552%. This supports the claim that the gain includes actual compensation of missing spectral content, not merely suppression of excess energy. However, new over-amplitude errors increased, so we cannot claim precise reversal of compression. This power-based diagnosis differs from the square-root, weighted approach of the common MR metric and cannot replace the primary scoring metric. Calibrated spectral amplitude power differences are not equivalent to PCM dBFS waveform differences and thus cannot be used to infer audible quality.

---

## Where Does Dynamic Degradation Occur?

Common MR dynamic metrics for music, per-recording average relative to control:

| Stage | Seed 20261012 | Seed 20261013 |
|---|---:|---:|
| Raw model OLA | +0.150% | +0.173% |
| After band fusion | +0.217% | +0.249% |
| After unclipped gating | +0.217% | +0.249% |
| Final native output | +0.214% | +0.242% |

Therefore, fixing the DSP alone is insufficient, and attributing all issues to model capacity is premature. The actual fusion process uses linear amplitude interpolation and phase interpolation on the unit circle—this cannot be directly explained by complex-spectrum linear mixing. In the 33 music segments, no active switches or gating events occurred within the supervised region, yet band masks still varied over time. This rules out time-varying gate changes as the cause. The contribution of mask variation, phase fusion, and overlap synthesis has not yet been disentangled, so temporal correlation cannot be interpreted as a confirmed fusion mechanism.

Native music dynamics degraded in 16 out of 26 recordings in seed 20261012 and 19 out of 19 in seed 20261013—this is not isolated to a single anomaly. Speech native dynamic average also degraded by 0.092% / 0.066%; both seeds show 26 recordings with degradation, 19/21 improved, and the rest unchanged. Thus, future supervised experiments must retain both music and speech validation. The relatively weak right-hand segment (`heiyaoyao-adaa234fc0509f0b-a73ef28fdfb5440b`) contributed 36.9% / 45.8% of all positive dynamic degradation. However, the normal-voiced electronic track `sideways-city-bdb28d27eb409b05` also shows clear degradation in both seeds. Its reference full-band RMS is −9.36 dBFS, and the candidate relative to identity DSP shows an additional PCM RMS of −52.75 / −51.95 dBFS—this cannot be explained by "all signals are extremely weak."

In the second seed, the 16 kHz and above reference RMS for this electronic track is −39.37 dBFS, while the native output is −51.29 dBFS, and identity DSP is −61.57 dBFS: the compensation adds back missing energy, but still significantly underperforms. This is not evidence that high frequencies are being uniformly suppressed, nor does it constitute a perceptual conclusion.

---

### Temporal Dynamics Diagnosis Breakdown

Let P, X, T denote the output, input, and reference STFT magnitudes, respectively. Define
`q = Δ(P − X)` and `r = Δ(T − X)`. Then:

`||q − r||² = ||r||² + ||q||² − 2〈q, r〉`

This measures whether the model's compensation follows the required temporal changes in the reference, without penalizing the reference's own transient behavior. Under identical fixed analysis window, band, and normalization, the native music output results are:

| Metric | Control (both seeds) | Only Compressed (both seeds) |
|---|---:|---:|
| Compensation change power `||q||²` | 0.0001580 / 0.0001636 | 0.0003857 / 0.0003661 |
| Alignment term `2〈q, r〉` | 0.0001124 / 0.0001411 | 0.0003127 / 0.0002430 |
| Final change error `||q−r||²` | 0.0638452 / 0.0638221 | 0.0638726 / 0.0639227 |

The alignment term increases, but is insufficient to offset the added compensation. In the raw OLA, the alignment term exceeds the compensation power. In this squared error metric, raw OLA shows a slight improvement over control, but the primary dynamic metrics still degrade. The square root and normalized aggregation differ between the two, so one cannot be used to override the main conclusion. The unnormalized temporal error power shows one seed improved and the other degraded—both perspectives must be retained.

All algebraic identities were verified: the decomposition of the identity, zero error in reference output, zero compensation when input is unchanged, a same-phase half-amplitude difference having one-quarter of the squared error, and non-zero output on silence reference being penalized. Additionally, missing regions in normal/weak/silent references show zero error. These are algebraic consistency checks and do not constitute gradient validation of a new training loss or perceptual pass condition.

---

## Identity Degradation Cannot Be Judged by Mean Percentages Alone

The average identity spectral error for music relative to control still increases by 5.331% / 1.629%, but per-recording distribution shows:

| Seed | Improvements | Degrades | Stays Flat | Median Deviation |
|---|---:|---:|---:|---:|
| 20261012 | 17 | 5 | 4 | −5.37e-6 |
| 20261013 | 18 | 4 | 4 | −2.52e-5 |

The weak right-hand segment alone accounts for **91.4% / 92.5%** of all positive MR degradation. Its full-band reference RMS is −42.43 dBFS, with 16 kHz and above reference at −143.72 dBFS; the candidate relative to identity DSP shows a full-band additional PCM RMS of **−111.51 / −109.78 dBFS**. In seed 20261012, the 16 kHz and above relative error to raw reference is −112.43 dBFS. These are calibrated PCM-level values; this relative fraction counterexample is not the largest absolute PCM error segment.

Therefore, we cannot conclude from mean percentage that audible pops are present, nor can we claim restoration success simply because the absolute magnitude is small. All counterexamples and prior study results are preserved. Future work should simultaneously examine relative spectral shape, absolute error, and blind listening.

---

## Next Steps Recommended

Following the [Model and DSP Boundary Review](dsp-model-boundary-review.md), we found that even perfect prediction can be rewritten by current fusion mechanisms. Priority has been adjusted to first audit and simplify the fusion process before deciding on deployment supervision. The following three training configurations were initially proposed but remain uninitiated and are not to be executed as immediate next steps.

Current MR dynamic metrics are based on time-differenced amplitude errors between prediction and reference, not a simple penalty for output change. P35/P36 primary spectral supervision occurs along the raw model OLA path; actual DSP output dynamic behavior is not directly optimized in this round. Existing evidence supports first investigating supervision path and objective trade-offs, and does not support a "parameter insufficiency" conclusion.

We recommend maintaining a small model, error compression, and identity weight at 1.0, and conducting three equivalent-budget combinations:
- Original control target
- Enhanced reference dynamic supervision on raw OLA
- Additional reference dynamic supervision on deployed path

The latter two should first be validated on training materials for comparable scale and gradient budget, then pre-registered for formal training. The direct lossless spectral reconstruction goal must be preserved—no deployment fidelity objective should replace high-frequency recovery. All groups start from the same initialization and are frozen and uniformly rechecked.

Prior to training, validate the exact reference, silence/weak-energy, transients, finite-difference, and the correspondence between fusion/limiting proxies and actual Rust output. The deployment chain may inherently prevent output from matching the unclipped reference; such irreducible errors must be reported separately and cannot be circumvented by modifying the reference. The causal mechanism behind stage differences has not yet been identified, so no fusion mechanism should be modified simultaneously. Additionally, we do not reuse previously failed over-protection terms for additive tuning.

This analysis concludes with diagnosis only—no new training was initiated, and no default weights were promoted. All materials are reused held-out rechecks and cannot prove independent generalization. No new blind listening or real-device continuous audio evidence is available.

---

## Local Evidence and Recheck

Detailed outputs are preserved in the ignore directory; raw materials, weights, or experiment JSON are not submitted:

- `data/runs/risk-stages-p36-20261022/`: Full stage audits and frozen training source code for all new endpoints.
- `data/runs/risk-diagnosis-p36-20261022/recording-band-audit.json`: Per-recording, per-band, win/loss, medians, positive/negative contributions, and native PCM magnitude levels.
- Same directory: `directions-registration.json`, `directions-*.json`: Fixed input masks, endpoint/binary/script hashes, and full paired spectral and temporal change decompositions.
- Same directory: `metric-check.json`, `mask-inventory.json`: Algebraic validation and per-segment gate/mask statistics.
- Same directory: `aggregate.py`, `directions.py`, `masks.py`: Post-hoc diagnosis scripts.
  The `directions.py` script uses hash-bound historical `risk-directions-p34-guard-20261021/audit.py` with pure scoring functions; it does not execute the historical experiment or training.

Scripts are run from the root repository, using `training/.venv/bin/python` and
`PYTHONPATH=data/runs/risk-attribution-p36-20261022/source/training`,
`OMP_NUM_THREADS=1`. The stage audit entry point is `training/scripts/audit_spectrum_stages.py`, specifying P36 study, the above output directory, and the registered `soundex-context-replay` binary.

The residual decomposition is `||q − r||² = ||r||² + ||q||² − 2〈q,r〉`; the cross-term `2〈q,r〉` measures alignment, not merely added energy.
