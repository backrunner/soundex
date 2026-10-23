# P6: Band-Energy Envelope and In-Band Spectral Shape Specialization

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

Building upon the reference intervention evidence from [P5](envelope-study-20261012.md), the following protocol is formally registered prior to training.

This round of evaluation maintains the core structure and objective of the original amplitude study—no skipping of phase, DSP, blind listening, or real-time deployment thresholds.

---

## Control Design

Three groups, each with two seeds (`20261012 / 20261013`), are trained using **2048 updates, batch size 4**. All training begins from scratch.

| Group | Output Structure | Training Objective |
| --- | --- | --- |
| direct_legacy | P5 direct amplitude residual head | P5 compressed amplitude + time-differenced |
| direct_factored_loss | Same direct amplitude head | Separated energy/spectral shape objectives |
| factorized | Frame-wise 5-band RMS head + 129-bin in-band shape head | Same per-component objectives |

All three groups share the same P5 past-only 1024/2048-point input, 4-hop update stride, width=144 TCN backbone, and online bandwidth conditions.
**Training and all metrics exclude the first 80 hops**, covering the full historical context of long windows and TCN dynamics; future data is not read.
All groups share identical initial backbone tensor states, and the final amplitude residual layer is uniformly zeroed, starting from an approximate identity function.

The `factorized` group adds only a single 288→5 energy head (1445 parameters), with no significant expansion in model size.

Each band's shape is normalized by positive amplitude seeds via RMS, and the energy head predicts the logarithmic RMS correction relative to the input per-band RMS.
A small floor of approximately 1e-8 is retained in the relative per-band RMS input, allowing complete band absence;
the output relative to full-spectrum RMS is clipped to the range [-30, 2]. This is **not** an oracle: during inference, no clean reference is available.
Changes in energy for a given band do not alter its normalized shape, and changes in shape do not affect the predicted energy for that band—both properties are independently verified.
Under extremely weak input, only a numerical floor is preserved; no mathematical claim of strict zero output is made. Finite values and gradients are confirmed.

---

## Objectives, Selection, and Data

Bands remain: 0–4, 4–8, 8–12, 12–16, 16–Nyquist kHz; low-band weight = 0.25, high-bands = 1, normalized by 4.25.
A reference supervision region is defined as 1e-4 of the full-spectrum RMS (minimum 1e-7 DFT magnitude):

- **Energy term**: Absolute difference of smoothed per-frame per-band log-RMS.
- **Spectral shape term**: Absolute difference of smoothed per-bin RMS-normalized amplitudes; weighted by confidence in reference band power relative to floor.
- **Dynamic term**: Signed difference of log-band RMS time-differenced between prediction and reference, preserving transient behavior.
- Retain the P5 compressed amplitude term. Total loss = energy + 0.5×spectral + 0.25×dynamic + 0.25×P5 amplitude.

The floor is used for smoothing and normalization, not to alter the reference. The new and old loss values are not directly comparable.
Every 512 steps, the three groups are scored using **identical per-component objectives**: 253/254 music/voice and 1/254, then mixed 87.5% paired + 12.5% identity.
The near-identity initialization at step 0 may also be considered a baseline, avoiding forced promotion of degraded training results.
AdamW optimizer with lr=1e-4, cosine decay to 0.1×, weight decay=1e-4, gradient clipping=5.

Existing, verified lossless sources and audio/origin checksum isolation are used. P5 weights are not loaded.
Each seed generates 8192 fixed samples using `seed+40000`; across the three groups, identical inputs are used, with 12.5% replaced by native identity input.
All groups retain mono, speech, and weak high-frequency recordings; no data is removed due to failure or threshold adjustment.
Development set and training probes follow existing definitions; not considered independent test, and historical test scores are not accessed in this round.

---

## Pre-Specified Evaluation Criteria

We compare:
- `direct_factored_loss` vs. `direct_legacy` (objective effect)
- `factorized` vs. `direct_factored_loss` (structural effect)

Evaluations are performed via audio pairing bootstrap, with two seeds reporting independently.
Primary evidence is improvement in high-frequency speech (SC) and energy error, alongside review of flux, compressed input, native identity, low-frequency, and speech components.
Overall loss reduction alone is insufficient for promotion.
If the two seeds disagree or intervals cross zero, uncertainty is retained.
If only one group shows improvement, gains are not attributed to the new structure.
No minor metric fluctuations are used to infer perceptual quality.
Maintaining compressed input or winning at step 0 is not considered a recovery success.

After freezing the factorized weights, historical features are re-diagnosed; no new checkpoint is selected.

Model and loss testing cover reference optimal, weak energy/no-speech gradients, finite differences, MPS/CPU, causality, and block/channel isolation.
The training framework adds recovery consistency and product integrity checks.
Model size remains ≤2M parameters; ONNX diagnostic size <8 MiB.
CPU hop cost and full audio latency/stability are reported separately.

To avoid being confined to raw amplitude metrics, a separate offline Rust DSP diagnosis is conducted (not involved in selection):

For each seed, using the frozen P3 control phase, we compare:
- P3 original output
- Identity DSP
- Three amplitude-replacement models
- Factorized with input phase

All use the same first 80 hops of original input, model gain, and smooth-knee.
Scoring is clipped to `(80+9)*128 : 32768-128`.
All development fragments are reused, and results are reported separately for music/voice, cutoff frequency HF metrics, and fixed 4 kHz+ performance.
Rust output is verified against proxy error ≤2e-5.
This represents offline feature prediction through a real-time synthesis chain; it does **not** imply the new model has been integrated into live streaming, nor does it constitute blind listening.

---

## Reproduction

```sh
PYTHONPATH=training training/.venv/bin/python training/scripts/run_factorized_study.py \
  --parent data/runs/envelope-p5-20261012/study/config.json \
  --output data/runs/factor-p6b-20261012/study \
  --cache data/runs/factor-p6-20261012/study/cache
```

Configuration, sampling, input validation, detector, and source code hashes are preserved locally; checkpoints include full optimizer and RNG.
Source code and documentation are submitted; data, cache, checkpoints, and diagnostic weights are retained in ignored directories.
Default audio model remains unchanged.

---

## Pre-Training Boundary Correction

The initial `892e618` run was terminated early and is excluded from final evaluation.
A complete probe for missing band recovery revealed:
- The original lower bound of -18 was above the logarithmic value of input seeds with 1e-8 floor, causing energy gradient clipping.
- Zero-amplitude symmetric seeds caused spectral head gradients to vanish.
- Corrected to a floor of -30, with a small positive bias added to shape seeds.
- Added strict-zero input and actual 1e-10 feature floor regressions, requiring both energy and spectral heads to have correct direction and non-zero finite gradients.
- The corrected batch uses a new directory, frozen source code, and full six-group re-run.
- Sampling seeds, training budget, loss, and selection rules remain unchanged.
- Only input features cached by audio/source/detector hash are reused; no old weights or evaluation conclusions are reused.

---

## Six Groups with Corrected Version Completed

Final source: `186301b`, directory: `data/runs/factor-p6b-20261012/study/`.
Each seed has 8192 samples, 1024 of which are identity controls; all six groups complete 2048 updates and select step 2048.
Best steps occur at budget boundary—convergence is not proven, and longer training may still yield benefit.
This round's architectural conclusions are limited to the current budget.
Below are 33 music and 26 speech development fragments, averaged equally, with fixed bands and 80-hop exclusion zone.

"Common objective" refers to paired music sub-score, not the weighted sum of speech/identity after selection.

| Group / Seed | Common Objective ↓ | High-Frequency SC ↓ | High-Frequency Flux ↓ | High-Frequency Energy Error (dB) ↓ |
| --- | ---: | ---: | ---: | ---: |
| Compressed Input | 1.12013 | 0.54790 | 0.71550 | 8.83169 |
| direct_legacy / 12 | 0.85916 | 2.11832 | 1.37725 | 5.06870 |
| direct_factored_loss / 12 | 0.85139 | 3.48862 | 1.88541 | 5.01535 |
| factorized / 12 | 1.04358 | 1.06465 | 1.01669 | 7.54628 |
| direct_legacy / 13 | 0.86796 | 2.16476 | 1.45297 | 5.17911 |
| direct_factored_loss / 13 | 0.84235 | 3.71767 | 1.92805 | 4.87881 |
| factorized / 13 | 1.04413 | 0.70118 | 0.79353 | 7.66464 |

**Conclusion: Structural effect exists, but no overall recovery benefit is established.**

- **Only loss change**: Per audio pairing, high-frequency SC increases by **+1.4285 / +1.7056**, flux increases by **+0.5483 / +0.5030**; 95% bootstrap intervals for both are above zero. Energy error reduction intervals cross zero.
  Average energy improvement cannot cover spectral or transient degradation.
- **Same loss, two-branch structure**: SC average difference **-2.6396 / -3.3452**, flux **-0.9859 / -1.2840**, energy error **+2.7493 / +3.0601 dB**.
  First seed's SC/flux intervals cross zero; second seed's are below zero. Both energy error intervals cross zero.
  Thus, large fragment averages cannot be equated to stable per-audio advantage.
- **Compressed input**: factorized energy error pairing difference **-1.4859 / -1.3226 dB**, intervals below zero; SC increases **+0.4930 / +0.1290**, both intervals above zero.
  Still fails to meet primary recovery criteria.

Native identity music shows factorized HF energy error of **1.755 / 1.312 dB**, indicating failure to fully avoid unnecessary modifications.
Speech SC: **0.388 / 0.341**, energy error: **0.975 / 0.773 dB**—separately recorded, not replaced by music average.
Music probe common objective: **0.9524 / 0.9614**; development: **1.0436 / 1.0441**.
Probe and development differ—this difference cannot be directly attributed to overfitting.
This round reused development set; no independent test or blind listening evidence is added.

---

## Scale and Independent Performance Diagnostics

Factorized model has **807,398 parameters**, 1445 more than direct head; ONNX diagnostic size: **3,277,209 B (3.125 MiB)**.
First seed export SHA-256: `1825019a26cdf24e5dd05dd32c1a1decc3d8a852f0cadb2efef660de683414fb`.
Maximum PyTorch/ORT continuous state difference: **8.34e-7**; mono/stereo export verified.

Apple M5 Max, AC, low-power mode disabled; after official training exit, single-threaded ORT, 5 seconds per group, pre-warmup 96 hops.
Includes Python FFT frontend; excludes bandwidth detection, DSP, queue, callbacks, or device.

| Sample Rate / Channel | Frontend + Inference P99 ms | Max ms | Exceeds hop budget |
| --- | ---: | ---: | ---: |
| 44.1k / mono | 0.14831 | 1.11242 | 0 |
| 44.1k / stereo | 0.25569 | 1.00104 | 0 |
| 48k / mono | 0.15299 | 0.54092 | 0 |
| 48k / stereo | 0.22220 | 0.27633 | 0 |

CPU time per audio frame approximates single-core usage of **4.23%–7.29%**, estimated as irregular tight loop.
Separate NumPy/ORT process: peak RSS **43.55 MiB**, after load **62.94 MiB**, after inference **64.34 MiB**, peak increase **20.80 MiB**.
This cannot be used as a proxy for full Rust deployment memory.
Amplitude probe size and compute cost meet current budget; no claim is made for full audio latency <8 ms or stable streaming output.

**351 tests passed locally.** Cross-platform CI failed a unit test due to random backbone convergence at exactly 120 steps; now fixed by freezing irrelevant backbone and independently training two output heads with double precision, preserving original loss improvement threshold, and accurately verifying missing band learnability.
Real training remains FP32/MPS, unchanged in budget or result.
`c1ac5ae` remote Rust and export-contract CI passed.

Checkpoints, evaluations, ONNX, and audio are preserved locally; default weights remain unmodified.
Evidence is itemized in the study directory: `summary.json`, `comparisons.json`, `diagnostic-cost.json`, `diagnostic-memory.json`.

---

## Actual DSP Output Recheck

All 89 development fragments (33 music, 56 speech) × 6 paths × 2 seeds → **1068 Rust synthesis replays** completed.
Maximum proxy/original waveform difference: **4.54e-6**, below target 2e-5.
Below is the music fragment average, using cutoff frequency from list to define HF.
Cannot be directly subtracted from prior 4 fixed high-frequency band averages.

| Path / Seed | High-Frequency SC | High-Frequency Flux | High-Frequency Energy Error (dB) | ΔSI-SDR dB |
| --- | ---: | ---: | ---: | ---: |
| P3 Original / 12 | 1.2202 | 1.1883 | 6.0548 | -0.3859 |
| Identity DSP / both | 0.7245 | 0.8813 | 7.6110 | -0.0742 |
| Direct Head + Per-component loss + P3 Phase / 12 | 1.1086 | 0.9894 | 6.5500 | -0.2901 |
| Dual-branch + P3 Phase / 12 | 1.2162 | 1.2353 | 6.5593 | -0.5579 |
| Dual-branch + Input Phase / 12 | 1.1658 | 1.2720 | 6.4193 | -0.5222 |
| P3 Original / 13 | 1.1764 | 1.1259 | 6.2568 | -0.3828 |
| Direct Head + Per-component loss + P3 Phase / 13 | 1.0006 | 0.9728 | 6.4715 | -0.2890 |
| Dual-branch + P3 Phase / 13 | 1.1441 | 1.1219 | 6.0845 | -0.2565 |
| Dual-branch + Input Phase / 13 | 1.0406 | 1.2057 | 5.8915 | -0.1957 |

Per 26 speech pairs, dual-branch + P3 phase vs. P3 original:
- SC difference: **+0.1303 / +0.0105**
- Energy error difference: **+0.6889 / -0.0561 dB**
All 95% intervals cross zero—cannot claim superiority over P3.
Relative to direct head with same loss, output flux increases by **+0.2366 / +0.0938**, intervals: **[0.0361, 0.4956] / [0.0181, 0.1900]**—indicating that raw amplitude advantage cannot be directly converted to product benefit.
Switching to input phase improves ΔSI-SDR relative to P3 phase by **+0.0452 / +0.0754 dB**, but does not simultaneously improve dynamic HF flux.
Phase swapping is not a validated general solution.

This output validation includes real DSP, but amplitude is pre-computed; no live streaming integration, device long-term stability, or blind listening has been completed.
No checkpoint or release threshold was adjusted based on this recheck.

---

### In-Band, Time-Window Aligned Recheck

Dynamic cutoff HF and fixed band averages reflect different regions. To eliminate metric confusion, an additional time-aligned recheck is performed on the same 1068 combinations, saved as `dsp-aligned-*`, not overriding initial results, and not treated as new independent samples.
Original amplitude and actual output are re-analyzed using 256 FFT / 128 hop, same origin, same reference, five fixed bands; only windows fully within common scoring window (frame 89:254) are retained.
Output includes synthesis delay, no additional zero-padding.
Time-axis regression with identity is added, requiring near-zero SC, flux, and energy error for the same signal.

| Path / Seed | Original SC | Output SC | Original Flux | Output Flux | Original Energy Error (dB) | Output Energy Error (dB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct Head / 12 | 3.4923 | 2.8985 | 1.8895 | 1.8574 | 5.0159 | 5.0599 |
| Dual-Branch / 12 | 1.0731 | 0.9477 | 1.0212 | 0.9675 | 7.5603 | 6.8937 |
| Direct Head / 13 | 3.7199 | 2.9005 | 1.9352 | 1.9395 | 4.8848 | 5.1041 |
| Dual-Branch / 13 | 0.7046 | 0.6562 | 0.7943 | 0.7905 | 7.6809 | 6.8182 |

Same P3 phase applied. Under identical metrics, dual-branch maintains average SC/flux advantage after DSP, though at a cost in energy error.
Per-audio pairing: output SC difference **-2.0940 / -2.4900**, 95% intervals **[-5.0348, 0.0533] / [-5.9363, -0.0735]**; output energy error difference **+1.9935 / +1.9197 dB**, both intervals cross zero.
Thus, differences in metrics across different metrics cannot be attributed to phase or DSP.

Per-band output further localizes:
Above 16 kHz:
- Direct head SC: **8.0881 / 8.2311**; dual-branch: **2.4906 / 1.3531**
- Energy error: **11.1106 / 11.2320 dB** vs. **16.5793 / 17.0778 dB**
- Dual-branch average signed energy deviation: **-7.6078 / -12.6903 dB**
But absolute error is significantly higher—indicating mixed over/under-compensation.
These are post-hoc development set diagnostics, not new promotion criteria.

---

## Judgment and Next Steps

This round fails to achieve overall recovery goals, nor does it prove model capacity insufficiency, baseline overfitting, or pure DSP error.
Default weights remain unchanged. Prior investigation of current trade-offs is prioritized before deciding on extended training.

Next, we test two distinct hypotheses before full-scale training:

1. **Does missing-band energy parametrization limit learning?**
   Current approach anchors per-band prediction to input RMS, requiring large dynamic range under severe truncation.
   Compare explicit input log-band energy vs. absolute envelope prediction independent of missing-band input.
   Use identical shape head, loss, and budget to isolate structural effect.
   This remains a hypothesis—negative deviation in this round does not confirm the issue.

2. **Can energy and synthesizable fine details be improved together?**
   Fix per-band, per-window evaluation. Separate preserved vs. damaged bands.
   Evaluate raw amplitude, actual synthesis, and natural weak high-frequency speech.
   Re-check synthetic spectral/transient supervision under matched controls; phase may be introduced if needed.
   Do not misinterpret differences in dynamic HF and fixed-band performance as DSP-induced loss of benefit.

Future rules must be registered prior to new training. Retain both seeds, per-recording intervals, identity/voice controls, and real-time budget.
Independent testing, blind listening, and full streaming stability remain necessary final evidence.

---

## Final Verification

All six group best checkpoints and evaluation hashes are consistent.
Each seed has 8192 samples, 1024 identity controls, and identical digest.
85 frozen source code files pass hash validation.
Current model, loss, and official training script match registered source code exactly.
