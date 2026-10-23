# P5: First, Validate the Learnability of Magnitude Spectral Shape

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

Building upon [Restoration Root-Cause Audit](restoration-root-cause-20261012.md), this experiment records findings prior to formal training.

We begin by comparing representation methods, without continuing P4 weights, without adding power penalties, and without treating magnitude error as an audio quality metric.

## Fixed Protocol

- Three groups: `short / tiled / past`, each with two seeds (`20261012 / 20261013`), initialized from identical random weights.
  Each group undergoes 1024 updates, batch size 4, AdamW learning rate 1e-4, cosine decay to 0.1, weight decay 1e-4, gradient clipping at 5.
  Each seed shares a fixed sampling scheme of `seed+30000`; source/track/SHA-256 isolation follows the previously approved validation list.
  Training audio/speech remains at 253/254, 1/254, with every eighth sample replaced by the corresponding native identity input.
- Common 256 FFT / 128 hop; magnitude input eliminates sensitivity to global phase inversion; explicit online cutoff, active, and sample rate are provided.
  TCN width 144, dilation 1/2/4/8/16/32, shared history warmup for 64 hops, no backpropagation or scoring during warmup.
- The `past` group includes Hann-windowed FFT magnitude from past and current 1024/2048 points, normalized by `256/N`;
  Updated every four hops, others remain at the most recent value. Frame `t` only accesses `(t+1)*128`, not future frames.
  The `tiled` group constructs a long window by repeating the current 256-point frame; updated at the same frequency. The `short` group has zero-long window slots.
  All three groups have identical parameter counts, though the `short` group's inactive input weights do not equate to the same effective capacity; `tiled` is used as a control.
  The `short` group also includes TCN temporal history; the comparison is between the frequency resolution of past audio and its encoded representation, not whether history is present.
- Output is `abs(input_magnitude + input_RMS * 2*tanh(delta))`, allowing identity and filling missing frequency bins.
  This experiment does not predict phase, does not use DSP/waveform loss, and does not output a directly deployable audio model.
- Fixed frequency bands: 0–4, 4–8, 8–12, 12–16, 16–Nyquist kHz.
  Log1p magnitude L1 loss and signed time-differenced L1 loss with time-weight 0.25 and low-frequency band weight 0.25; each of the four high-frequency bands has weight 1, divided by 4.25.
  Reference spectrum RMS (1% of reference spectrum RMS, minimum 1e-5 DFT magnitude) is used for scale compression; clean reference is not modified.
  This total score is not directly comparable to P2/P3/P4 results.
- Every 256 steps, a development validation is performed; the music and speech are weighted and then the best model is selected from paired 87.5% and identity 12.5%.
  Final report includes raw magnitude SC, positive flux, energy deviation/absolute error, and reference absolute spectral power.
  Band and loss settings are consistent; reference spectral power is in DFT units and cannot be converted to waveform dBFS.
  Separate reports are provided for training probe, development music, speech, and identity, as well as ablation of historical features with fixed weights.
  This round does not load historical test scores, nor can any new independent test be claimed.

## Interpretation and Next-Stage Thresholds

Compare `past` and `short`, `tiled` using audio pairing; pre-requirement: both seeds must show consistent improvement in the primary magnitude target during development.
Additionally, verify SC, flux, weak-band energy, and identity control. Confidence intervals crossing zero are considered insufficient evidence; small mean differences are not considered breakthroughs.
Ablation of `past` weights should support actual utilization of history; such ablation is not used to re-select weights.
Development probe and development results are judged separately for optimization deficiency and generalization gap; a 1024-step result cannot be used to deny the capacity ceiling of long-window representations.

Only when spectral shape discrimination is supported will the next stage proceed to full magnitude and phase, and actual Rust DSP.
At that point, full audio quality, blind listening, fixed time axis, and real-time continuity validation will be re-run.
This phase allows measurement of ONNX and frontend per-hop cost; Python frontend latency does not equal Rust callback cost.
Long windows without lookahead do not imply complete latency is acceptable.
All constraints remain: ≤2M parameters, ONNX <8 MiB; default model unchanged.

## Implementation and Reproducibility

`training/scripts/run_envelope_study.py` saves frozen configuration, source code hash, data list, detector, sampling, complete optimizer/RNG checkpoint, and best weight hash.
On recovery, source, configuration, and sampling signatures are verified.
Run lock prevents concurrent execution in the same directory; data and experiment outputs are preserved in ignored directories.

```sh
PYTHONPATH=training training/.venv/bin/python training/scripts/run_envelope_study.py \
  --parent data/runs/causal-p4-20261012/study/config.json \
  --output data/runs/envelope-p5-20261012/study
```

## Completed Results

All six configurations completed 1024 steps; training source code is frozen at `7d57c8d`.
The `short` group selected seed 2 at step 768; all others selected at 1024 steps. Below are the development music **33 fragments / 26 recordings** averaged equally.
These raw magnitude, fixed-band metrics cannot be directly compared to the dynamic region DSP metrics from prior P4 experiments.

| Group / Seed | Primary Magnitude Target ↓ | High-Frequency SC ↓ | High-Frequency Flux ↓ | High-Frequency Energy Error (dB) ↓ |
| --- | ---: | ---: | ---: | ---: |
| Unprocessed Compressed Input | 0.285981 | 0.54717 | 0.71683 | 8.77455 |
| short / 12 | 0.270985 | 2.26158 | 1.65550 | 5.25112 |
| tiled / 12 | 0.271799 | 2.01782 | 1.47198 | 5.22810 |
| past / 12 | 0.272362 | 1.99483 | 1.39421 | 5.33293 |
| short / 13 | 0.270130 | 1.83306 | 1.61707 | 5.29745 |
| tiled / 13 | 0.271843 | 1.77597 | 1.48814 | 5.25324 |
| past / 13 | 0.270899 | 1.83056 | 1.35158 | 5.24923 |

Within the same recording, fragment differences are first averaged, then bootstrapped across 26 recordings (2000 samples, fixed seed).
Relative magnitude target differences between `past` and `short` are **+0.001971** (95% CI: -0.000847 to +0.004841) and **+0.001457** (-0.001641 to +0.004545). Neither meets the pre-defined requirement for improvement in both seeds.
Directional differences relative to `tiled` are inconsistent (+0.000571 / -0.001183), with both confidence intervals crossing zero.
These intervals are derived from reused development recordings and are not independent tests or corrected for multiple comparisons.

We cannot conclude that long windows are completely ineffective: the flux pairing difference between `past` and `short` is **-0.28878 / -0.27278**, both intervals below zero.
When the `past` weights are replaced with short window or repeated frame features, flux degrades.
This supports the model's dependence on historical representations and a transient benefit, but does not constitute a primary magnitude or spectral shape improvement.
The `past` group also fails to outperform `short` on the training music probe magnitude target, inconsistent with a simple "good training gain, only overfitting in development" explanation.
With only 1024 steps, this result does not prove the long window structure is unlearnable or has reached capacity limits.

Additionally, we audited the warmup boundary: after 64-hop TCN initialization, full historical context requires approximately 80 hops.
The original protocol maintains 64 hops for training and selection, without post-hoc weight swapping; for each group, we further exclude the first 80 hops of scoring.
Even after this exclusion, the magnitude pairing difference between `past` and `short` remains **+0.002026 / +0.001617** (both intervals cross zero), and flux difference remains **-0.30713 / -0.26499** (both below zero), unchanged.
Future structural experiments must record full historical context warmup in both training and scoring, not just the TCN's 64-hop baseline.

For speech magnitude targets, `short/tiled/past` are: first seed **0.19709 / 0.19872 / 0.19962**, second seed **0.19869 / 0.19930 / 0.19842**—no consistent advantage.
For native identity music, `past` magnitude target is **0.03734 / 0.03104**, with high-frequency energy error **1.943 / 1.732 dB**, still showing unnecessary modifications.
All six groups reduce magnitude target and energy error compared to compressed input, while increasing high-frequency SC/flux.
Thus, success cannot be declared based on total loss or energy recovery.

## New Positioning: Band-Level Intensity and In-Band Detail Require Separate Evaluation

Post-training interventions, not involved in training or selection, nor used as deployable scores.
For the `past` group, two seeds are fixed and evaluated using clean reference per-frame per-band L2 norm, or per-band normalized magnitude shape; all other factors remain unchanged, with 0–4 kHz unchanged.
Development music results:

| Intervention | SC, seed 12 / 13 ↓ | Flux, seed 12 / 13 ↓ |
| --- | ---: | ---: |
| Original Prediction | 1.99483 / 1.83056 | 1.39421 / 1.35158 |
| Correct per-frame per-band energy, original per-band shape | 0.45821 / 0.46665 | 0.66890 / 0.67390 |
| Original energy, correct per-band shape | 1.82348 / 1.65441 | 1.90284 / 1.71622 |

**Per-frame per-band energy generation is a major source of current relative spectral error.**
It is not a simple global gain adjustment; the oracle uses per-frame, per-band reference information and cannot be treated as a feasible controller.
Simultaneously, the per-band magnitude target is reduced to approximately 0.160, while per-band energy is raised to about 0.209:
The merged compressed magnitude target and relative spectral error reflect different trade-offs—correct numerical values do not imply alignment with product objectives.

In the first seed's native identity input, average energy above 16 kHz is elevated by approximately **3.80 dB**, though low-frequency SC is only 0.00076.
Small absolute errors in weak bands can amplify relative SC—this underscores the need to preserve both absolute spectral power and energy deviation.
We cannot directly extrapolate audible distortion, nor can we eliminate counterexamples by raising the evaluation floor.

The next structural experiment should explicitly separate **band-level energy envelope** and **in-band normalized detail**, introducing native, non-generated controls.
These should be compared under shared budget against the short window magnitude head; first demonstrate improvement in both strength and spectral shape before advancing to phase, texture, and DSP.
This round did not blindly extend the old path or extend a 1024-step negative result into a denial of the entire architecture.

## Scale and Computational Cost

All three groups have **805,953 parameters**.
The first seed's `past` diagnostic export is **3,254,457 B (3.104 MiB)**, SHA-256 `ffd4eb3a34792ab9516e71aeff9fdde3434e8aa81cdc6a4f5e7fbe87bbd28c08`.
Maximum per-frame difference in single-channel/stereo continuous state (PyTorch–ORT) is **1.19e-7**.

Apple M5 Max, AC, low-power mode disabled, single-thread ORT; after formal training, each native sample rate/channel combination is tested for 5 seconds, with 96-hop warmup, synthetic input, stereo using batch=2.
Frontend includes short window FFT, long window FFT every four hops, and Python conversion; does not include online bandwidth detection, DSP, queue, callback, or device overhead.

| Sample Rate / Channel | Frontend + Inference P99 ms | Max ms | Exceeds Hop Computation Budget |
| --- | ---: | ---: | ---: |
| 44.1k / mono | 0.12267 | 0.32558 | 0 |
| 44.1k / stereo | 0.20694 | 1.09763 | 0 |
| 48k / mono | 0.11975 | 0.20967 | 0 |
| 48k / stereo | 0.20304 | 1.33758 | 0 |

Using CPU time consumption to estimate audio rate, the total load is approximately **3.48%–6.21% of single-core usage**, not actual callback CPU measured.
Peak RSS during PyTorch export/validation is 290.83 MiB; separate NumPy/ORT process:
- After import: peak 43.50 MiB
- After load: 61.94 MiB
- After inference: 63.02 MiB
- Peak increase: 19.52 MiB
These are distinct process metrics and do not represent the full Rust deployment RSS.
Scale and operator cost are feasible, **but no conclusive evidence exists for acceptable full latency, continuous audio quality, or production-grade real-time stability**.

All **330 test cases passed locally**; remote CI passed for training and analysis source code `524b122`.
Original audio, cache, checkpoints, and per-fragment reports were not submitted. Primary local outputs are located at:
`data/runs/envelope-p5-20261012/study/`: `summary.json`, `comparisons.json`, `factorization-audit.json`, `warmup-sensitivity.json`, `diagnostic-cost.json`, `diagnostic-memory.json`.
Experiment weights do not replace the default model.
