# P20: Magnitude vs. Complex Residual Output Heads

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

This round investigates whether output representations are a limiting factor in the degradation and spectral bleed-through issues identified in the cross-recording fidelity analysis of [P19 Failure Analysis](coverage-failure-analysis.md). The experiment retains local weights and does not replace the default model.

## Correction and Comparison

The shared backbone remains a causal TCN with width 144, using 1024/2048-point FFT context updated every four hops, with no future input. The encoder, time layer, and decoder hidden layers are initialized with identical seeds and verified via tensor-level hash checks. All three heads start from an identity output and are independently retrained:

| Group | Output Representation | Parameters |
| --- | --- | ---: |
| anchored | Original five-band independent energy and magnitude shapes, preserving input phase | 807,398 |
| amplitude | Real-valued residual in continuous frequency units, preserving input phase | 805,953 |
| complex | Same real-valued residual with added imaginary component, enabling phase correction | 824,658 |

New heads use the global RMS and a nine-point smoothed local power to form residual units:
`unit = global_rms * sqrt(local_normalized_power + 1)`, with real/imaginary coordinate increments given by
`2 * tanh(head) * unit`. Even completely missing high-frequency bands can still be leveraged through full-band energy; the model is not restricted to a minimal noise floor due to zero local input. The imaginary component at DC and Nyquist is fixed to zero.
A stable magnitude transformation uses an absolute `1e-15` value, below the input magnitude contract floor; no value is introduced that grows with full-band energy, which would otherwise suppress output in empty frequency bands. The new test directly validates the expressiveness of completely absent bands.

The initial training pilot uses local units and relative magnitude terms; empty-band fixtures exposed these limitations, so the configuration was corrected before formal evaluation on the development set. The old pilot is retained only for debugging and is not used in any official configuration comparison.
All three groups maintain the same backbone size; the complex head adds 18,705 parameters, so this is not a fully equivalent parameter comparison, and differences cannot be solely attributed to phase.

## Loss Reference

The original DSP rendering objective, individual components, and weights remain unchanged. After phase learning, the auxiliary term "lossless magnitude + compressed input phase" reference interferes with legitimate phase correction, so **all three groups now use a lossless complex reference**.
The identity sample continues to use the actual input as reference; the first 80 frames use the original input for model pre-warmup, and supervision begins at frame 89.
The reference is only used for training scoring and does not enter the inference input.

The auxiliary term still compares the inconsistency between predicted and reference outputs before and after synthesis and reanalysis, with weight `0.1365237430457469`. It does not treat the actual transient content of the reference audio as an error to be eliminated. Historical calls preserve the original input phase to avoid inadvertently altering the P16–P19 reproduction. P20 explicitly selects clean phase. Therefore, **P20 must retrain the anchored control; direct comparison with P19's combined loss is invalid for evaluating performance**.

## Fixed Experiment

- 253 training music pieces paired with 8 speech recordings, 2,088 fixed crops; 4 music + 1 speech per step, each material providing both paired and identity pairs, with equal weighting. This 20% is inherited from the diagnostic experiment's fixed risk ratio and does not represent the final product's training speech proportion; limitations in rare genre and speaker coverage remain.
- Seeds `20261012` and `20261013`; all three groups use the identical sampling order from P19 wide. Adam optimizer with `1e-4`, no weight decay, gradient norm clipping at 5, 16,192 updates per trajectory, totaling 97,152 updates.
- Pre-fixed endpoints at 2,048 and 16,192; all groups complete training and freeze endpoint hashes before development recheck. No checkpoint is selected based on development set performance. A model is saved every 64 steps, including optimizer, RNG, and history, enabling exact resumption.
- 89 paired development fragments plus 40 shifted fragments; these are reused diagnostic materials and not independent tests. Initial DSP control includes 12 endpoints, totaling 2,314 actual Rust DSP role replays.
- Input lists, 700 actual clean/degraded PCM files, sampling tables, source code, and strategy files are all verified via hash. Segregated by recording and original source. Original audio, paired data, and experiment weights are excluded from Git.

Fixed screening compares anchored/amplitude/complex under the same budget, with results reported per seed.
For music paired recovery and native high-frequency SC/flux, both must outperform the matched control and initial DSP. Identity error, energy error, and low-frequency distortion must not exceed the matched control, using a tolerance of `1e-6`.
Clip and recording results are equally weighted and preserved; failure cases and worst-case instances are recorded individually, and no mean can mask severe failures.
This is a screening step for further research and does not constitute a blind listening, continuous real-time stability, or release acceptance test.

## Execution and Verification

The final configuration first runs a 5-fragment pilot on the training set, with 64 steps per group per seed. It checks actual backpropagation, 70 Rust DSP replays, per-hop ONNX, silence, reset, and mono/stereo behavior.
Training set fit results do not represent generalization performance. The final pilot (`head-p20-smoke-v3-20261014`) is complete; the following is a **4 training music fragments and average over two seeds**, with all errors lower being better:

| Head | Paired Recovery Error | Identity Error | Rust High-Frequency SC | Rust High-Frequency Flux |
| --- | ---: | ---: | ---: | ---: |
| Initial DSP Control | 0.600599 | 0 | 0.519299 | 0.700891 |
| anchored | 0.572044 | 0.003797 | 0.495191 | 0.667126 |
| amplitude | 0.541397 | 0.045024 | 0.457691 | 0.668928 |
| complex | 0.474004 | 0.060756 | 0.394379 | 0.622839 |

The complex head shows a relative paired error reduction of approximately **17.1%**, and a high-frequency SC reduction of about **20.4%**. However, identity error increases significantly, so **this result fails the fidelity screening and does not meet the goal or demonstrate improved generalization**. The amplitude group shows the same trade-offs, indicating that removing hard frequency band constraints alters recovery strength and risk profile. No listening evaluation is provided.

The final pilot achieved a maximum proxy waveform error of `2.98023e-7 < 2e-5` during 70 Rust role replays. The ONNX diagnostic size for the complex head is 3,337,561 bytes (~3.18 MiB). Single-threaded stereo inference with Python context has p99 of 0.1948 / 0.1953 ms, max 0.2113 / 0.2175 ms; all 1,000 timed hops per seed remained under 48 kHz hop duration of 2.667 ms.
Separate ORT inference p99 is 0.1821 / 0.1803 ms, with explicit stereo state size at 72,576 bytes.
Additionally, six independent ORT-only subprocesses were pre-warmed with 100 stereo hops each, comparing actual RSS before and after model loading:
- Complex head shows an incremental RSS increase of approximately **20.1–20.2 MiB** (total RSS ~62 MiB, including Python/NumPy/ORT);
- Old head shows an incremental RSS increase of approximately **21.3–21.4 MiB**.
This snapshot includes session, allocator, and state buffer, excluding PyTorch, FFT frontend, or full Rust DSP. It is not a peak guarantee. Evidence is captured in the pilot root directory `isolated-resources.json` and `measure_rss.py`.

This is not a fixed latency or real-time stability certification, and historical context does not introduce forward-looking assumptions.

Formal execution began at 22:08 SGT on 2026-10-14, with frozen source code commit `6002609`, local supervisor PID `63472`. The latest stage is governed by `pipeline.json` and `study/progress.json`.
All three groups, two seeds, total 97,152 updates, completed by 03:01:57 SGT on 2026-10-15, including all endpoint rechecks and benchmark export. No candidate passed the pre-registration screening; default weights were not replaced.
All Python tests passed: **481 passed**, Ruff passed; 20 targeted tests cover reference correctness, finite difference, silence, channel isolation, causality, non-zero ONNX head, empty band capability, and precise breakpoint recovery.

The diagnostic ONNX receives current spectrum, external causal context, detection conditions, and per-channel history. It is explicitly marked as experimental and **cannot be directly loaded into the current SoundEx runtime as a formal artifact**.
Benchmarking uses single-threaded CPU ORT + Python FFT frontend, reporting p50/p95/p99/max, count of hops exceeding 48 kHz hop duration, and state size. Peak RSS is explicitly annotated as the high-water mark of the entire study and is not misattributed to model incremental memory. This test does not include sound card, callback queues, or full Rust pipeline.

```bash
PYTHONPATH=training training/.venv/bin/python training/scripts/run_head_study.py \
  --p19 data/runs/music-coverage-p19-20261014/study \
  --output data/runs/head-p20-20261014/study
```

Actual background execution uses the frozen source code from `data/runs/head-p20-20261014/source/training`.
This directory includes `pipeline.log`, `pipeline.json`, `study/progress.json`, `study/registration.json`; after training, it automatically performs native review, exports cost metrics, and generates a summary in `study/comparisons.json`.

## Complete Results and Failure Analysis (2026-10-15)

A re-execution of the frozen analyzer verified inputs, source code, checkpoints, 31 completion credentials, and all 2,314 Rust role replays. Maximum proxy waveform error: `1.02893e-5 < 2e-5`; 12 endpoints failed screening.
No new training was run, no screening threshold was adjusted, and no optimal development set weights were selected.

The following table presents the average over 16,192 steps, 33 development music fragments per seed, averaged across two seeds. These are reused development recheck materials and not independent generalization tests; all values are lower being better:

| Output | Paired Recovery Error | Identity Error | Rust High-Frequency SC | Rust High-Frequency Flux |
| --- | ---: | ---: | ---: | ---: |
| Initial Identity + DSP | 0.588365 | 0 | 0.548761 | 0.717164 |
| anchored | 0.624674 | 0.059516 | 0.612967 | 0.755468 |
| amplitude | 0.978874 | 0.447615 | 2.311690 | 1.994917 |
| complex | 1.057290 | 0.459039 | 2.621984 | 2.385622 |

The complex head reduces high-frequency energy error from initial 8.48 dB to approximately 6.69 dB, but spectral shape and transient errors increase significantly—indicating increased energy does not equate to correct content recovery. Degradation is already evident at 2,048 steps, not just in late training.
Results are consistent when weighted by recording; the music fidelity in 40 shifted fragments is weaker than in the anchored group.
Speech averages show no equivalent degradation, though speech and music coverage, recording count, and sampling weights differ, so music effectiveness cannot be extrapolated.

### Not Just Overfitting: Training Itself Has a Scale Problem

The shared fixed training set initial main loss is `0.282046`. Final values:
- Anchored: `0.271969 / 0.271196`
- Amplitude: `0.314983 / 0.297434`
- Complex: `0.327231 / 0.308479`

The new heads perform worse than initialization on the training set, and auxiliary and combined loss show no improvement. This cannot be explained as "training is good, only validation is bad," nor does it imply model capacity is insufficient.

Throughout training, gradient clipping ratios:
- Anchored: both seeds ~ **1.32%**
- Amplitude: **86.52% / 86.76%**
- Complex: **86.49% / 86.82%**

Median gradient magnitude: old heads ~ `0.48–0.53`, new heads ~ `27.6–31.1`. All updates passed finite value checks and no NaNs were observed. However, "finite" gradients do not imply a reasonable optimization scale.

Additional read-only gradient diagnostics were added, with fixed seed `20261012` for the first 16 training batches:

| Initialization: all three heads output and loss identical | Main Loss Gradient Norm Median | Weighted Auxiliary Term Gradient Median |
| --- | ---: | ---: |
| anchored | 0.4096 | 0.0163 |
| amplitude | 35.8754 | 0.8585 |
| complex | 36.6036 | 1.1441 |

The new head's main gradient is approximately **88–89 times larger** than the old head—this divergence exists before overfitting occurs.
Primary pressure comes from the main objective, not merely from disabling consistency auxiliary terms.
The derivative of output near zero is `2 * unit`; `unit` is at least the full-band RMS, while weak-band loss is normalized by its own energy and a small relative floor—both factors contribute to high sensitivity to tiny residuals.
Same Adam learning rate and same clip threshold do not imply equivalent effective update scale.
Frequent clipping is not the sole root cause, but combined with initialization gradients and training degradation, points toward a parameterization/optimization condition issue.

### Tail Failures Persist—Cannot Be Attributed to DSP Bleed

Of 33 music fragments, the final complex head had 21/23 paired errors better than the initial control, but a few severe failures dominate overall degradation. The two worst cases contributed approximately **95.0% to 100.6%** of net paired degradation.
Over 100% indicates that the net improvement from other fragments partially offsets the loss from bad cases—this is not a counting error.

In the common worst-case example `d2599868e6a472eff47c1674`, identity diagnosis shows:

- Anchored: two sub-versions had blended phase error `0.0738 / 0.2218`, OLA after `0.2236 / 0.8239`;
- Complex: `22.9599 / 13.4942`, OLA after reduced to `14.2738 / 10.0603`.

The error in the complex head existed before synthesis; OLA is not the sole source of the new damage.
The OLA and limiter post-processed errors for these three diagnostic fragments are identical, so limiter is not the cause.
The magnitude residual head also failed, indicating increased phase freedom is not sufficient. This round does not prove the complex route is inherently invalid.

The worst-case reference has very weak high-frequency content, and normalization metrics amplify relative errors. The pre-limiter waveform RMS error for the complex head is approximately **0.315% / 0.251%** relative to the same-stage identity baseline RMS—not a global waveform explosion of tens of times.
These ratios are not in dBFS or subjective listening conclusions. Absolute, relative, and subsequent blind listening evidence must be preserved.
No sample should be removed to pass the experiment, nor should real damage be treated as numerical error.
Extreme SC values alone cannot be used to justify severe clipping.

### Cost and Next Steps

The final complex head remains at 824,658 parameters and 3,337,561 bytes ONNX. For both endpoints, double-channel ORT + Python frontend p99 is **0.2145 / 0.2153 ms**, max **0.2610 / 0.2444 ms**; all 1,000 timed hops remained under 48 kHz hop duration.
Cost is not the reason for failure. This evidence still does not represent full Rust frontend integration, long-term callback, sound card, or blind listening validation.

**No further training of these weights is continued; no direct scaling is performed.**
Next steps should first calibrate output local sensitivity on training data, update magnitude and learning rate to keep weak-band damage under control, while preserving the ability to generate content in completely missing bands.
A parameterization approach can be compared—e.g., "reduce initial output slope while preserving final expressible range"—to avoid sacrificing generation upper limits for a clean output.
First, validate multi-batch training with natural bandwidth limits and low-energy materials, confirming fixed training error reduction and controlled identity damage before freezing the scheme for full cross-recording revalidation.
Natural absence of high frequencies and compression loss remain independent issues requiring content/retrograde conditions and paired negative examples—not hard-coded thresholds for specific failing tracks.

This additional diagnostic uses only frozen weights and no parameter updates. Original results are stored in
`data/runs/head-p20-20261014/study/`; diagnostic scripts, results, and hash credentials are in
`data/reports/head-p20-analysis-20261015/`, including `gradient-probe.json`, `stage-probe.json`, and `summary.json`. Training data and checkpoints remain excluded from Git.

The retained native parity bound was `2.98023e-7`, below the registered `2e-5` tolerance.
