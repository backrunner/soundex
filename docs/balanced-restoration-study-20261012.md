# P4: Smooth DSP, Reference Peak, and Bidirectional High-Frequency Energy Supervision

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

2026-10-12. This round is a controlled training experiment; default weights are not replaced, and no claims are made regarding generalization or perceptual quality improvements.

## Motivation and Fixed Baselines

[P3](weak-band-study-20261012.md) introduced unidirectional excess energy constraints that reduced unnecessary high-frequency content, but also weakened recovery of missing energy. The [limiter baseline](limiter-study-20261012.md) demonstrated that the original DSP itself alters the correct signal. Thus, this round fixes both a new smoothing DSP and shared training budget, differentiating between peak supervision conflicts and high-frequency recovery deficiencies, and avoids treating the old baseline as correct.

| Training Group | Deployment & Training Proxy | Peak Constraint | Additional High-Frequency Energy Term |
| --- | --- | --- | --- |
| `smooth_control` | `smooth-knee` | Original absolute threshold 0.855 | Unidirectional excess energy from P3 |
| `reference_peak` | `smooth-knee` | Allow reference's own corresponding peak | Unidirectional excess energy from P3 |
| `balanced` | `smooth-knee` | Allow reference's own corresponding peak | Bidirectional log-band power difference |

All groups use the same 12.5% identity input control and start from the corresponding seed's P3 **control** checkpoint. Parent weights are not selected based on held-out recheck results. Seeds are fixed at 20261012 and 20261013; each group undergoes 1024 updates, batch size 4, with a new AdamW optimizer, learning rate decaying from 1e-4 to approximately 1e-5 via cosine schedule, and gradient clipping set to 5.
Each seed uses the same frozen sampling sequence (seed + 20000), maintaining a music/speech base sampling ratio of 253/254, 1/254.
The identity control replaces one sample every two updates, without altering the source sampling proportion.

The model remains the original TCN: 639,906 parameters, 256/128 FFT/hop, explicit causal history, native 44.1/48 kHz; no additional lookahead or runtime reference signal is introduced. Output size is approximately 2.48 MiB. The new loss is used solely for training.

## Loss Definition and Limitations

Base spectral shape, complex spectra, energy, reference transient, and low-frequency fidelity terms retain P2 weights.
The new peak tolerance is defined as `max(0.855, abs(clean[t]))`, penalizing only the portion of the synthesized signal exceeding this value.
This term no longer requires the model to suppress legitimate peaks in the unclipped reference; a deployment safety limiter remains at a 0.95 upper bound.
**The reference audio is not clipped**, and when a safety upper bound conflicts with the original reference, waveform error is still truthfully included.
Per-sample amplitude tolerances do not constrain sign or phase; phase is jointly supervised by the base complex spectrum and waveform term.

The bidirectional term uses single-sided Parseval power from a 256/512/2048-point Hann window, fixed coverage across 8–10, 10–12, 12–14, 14–16, 16–18, 18–20, and 20–24 kHz (up to Nyquist). Frame-by-frame, band-by-band comparison:

`abs(log1p(predicted_power / floor) - log1p(reference_power / floor))`

`floor = max(reference_full_frame_power * 1e-6, 1e-10)`, with gradient stop applied at the reference end.
Averaged across bands, frames, samples, and three analysis dimensions, with an additional weight of 0.25, consistent with P3.
Above the floor, proportional over- and under-compensation are approximately equivalent; near the floor, strict symmetry is not claimed.
The correct reference yields zero additional error and zero additional gradient; silence is preserved with limited fidelity.
The power term cannot determine phase, nor can it verify correct fine details—joint supervision with other errors and perceptual judgment is required.

The training proxy uses real Rust input detection traces, gating, and band masking; supervision excludes a 64-frame warmup plus one full OLA support frame. Final selected models are separately evaluated using actual Rust processing.

## Selection and Generalization Boundaries

All groups use the same `causal-balanced-band-v1` target selection checkpoint:
`0.875 * paired_validation + 0.125 * identity_validation`, scored every 256 steps.
The total loss across new and old versions is not directly comparable. Parent weights were also scored using this common target prior to update.
Training domain probes are fixed at 32 music tracks and 8 speech recordings, not involved in selection.
Development validation remains at 89 paired segments / 80 recordings, isolated by original recording; additional verification includes workgroup and original file hash isolation.

After selecting the checkpoint, a fixed SHA is used to score historical held-out rechecks: one paired file per recording, totaling 39 recordings (21 music, 18 speech), processed over the full 2–4 second paired file. **This is not the entire song.**
These recordings were previously used in prior diagnostic studies and cannot be considered independent test cases, nor can they guarantee absence of overfitting.
Reports are stratified by source, codec, sample rate, channel role, genre, and region; confidence intervals are computed under resampling.
Independent mono/left/right segments are not stitched into pseudo-stereo; this metric does not replace paired stereo validation.
The validation set lacks country, pop, hip-hop, funk; the recheck set lacks Chinese, country, hip-hop, jazz—conclusion coverage is thus limited.

Actual output comparisons use identical `steady_*` metrics and windows: 72 hops and 256 samples are removed from the start and end, respectively. Full-segment metrics are retained, and the two are not mixed.
Comparisons include the original compressed input, P3 parent weights with new DSP, and the three P4 training groups.
Lower energy error cannot mask degradation in spectral shape, transient, low-frequency, or identity fidelity.

## Execution and Validation

Formal training began at 10:50 (Singapore time) on 2026-10-12 at local MPS, PID 7865.
All six groups completed 1024 steps, exported, and underwent actual Rust scoring. Training and caffeinate processes have since terminated.
`summary.json` is marked as `complete`; no weight promotion occurred.

Source code is frozen at `6fefd0416531ec5775a1a8add00b0608244758d3`; independent worktree:
`data/runs/causal-p4-20261012/source`. Training materials and results are stored only in the local ignore directory.

- **Formal results**: `data/runs/causal-p4-20261012/study/`
- **Logs**: `data/runs/causal-p4-20261012/train.log`
- **Launch credentials**: `data/runs/causal-p4-20261012/launch.json` (PID, UID, start time, command)
- **Configuration, data manifests, source code, parent weights, sampling, and independent copies of the Rust binary** are all registered with hashes; any mismatch prevents continuation.
- **Local full Python regression**: 313 passed, including new loss parsing, finite differences, silence, recovery direction, peak semantics, Rust DSP replay, and CPU/MPS cross-checks; Ruff checks passed.
- All three MPS 4-step smoke runs completed training, validation, ONNX/ORT export validation, and actual Rust scoring.
- CPU/MPS probe loss values are consistent; gradients in differentiable regions are verified with relative norm error <0.002.
  L1 exact zero-error breakpoints do not require identical second-order gradients; waveform or loss error tolerances were not relaxed.
- The inference structure remains unchanged; prior performance metrics are diagnostic evidence from previous weights only.
  New weight promotion requires binding to a new SHA and re-measurement of real-time performance, continuous stream stability, stereo, and perceptual quality—no inheritance of release status.

Replication command (from repository root, P3 completed):

```bash
PYTHONPATH=training training/.venv/bin/python -m scripts.run_balanced_study \
  --parent data/runs/causal-p3-20261012/study \
  --output data/runs/causal-p4-20261012/study \
  --stream-binary target/release/soundex-stream-eval \
  --detector data/runs/causal-p3-20261012/study/binaries/soundex-gain-data
```

A formal continuation requires a frozen worktree and preserved original registration configuration; the above path is provided as an example for first-time runs.

## Complete Results: No Promotion, No Extension of the Same Recipe

All six groups achieve their optimal checkpoints at step 1024. This endpoint alone is not a justification for further training.
Changes in the common loss have not yet stabilized into overall audio quality gains.

| Seed | Group | Parent Weight Common Loss | Optimal Common Loss | Paired Validation | Identity Validation | Training Probe |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 20261012 | smooth_control | 2.35421 | 2.36112 | 2.62992 | 0.47955 | 2.78284 |
| 20261012 | reference_peak | 2.35421 | 2.36122 | 2.63001 | 0.47970 | 2.78277 |
| 20261012 | balanced | 2.35421 | 2.36363 | 2.62502 | 0.53387 | 2.75503 |
| 20261013 | smooth_control | 2.39020 | 2.35497 | 2.63492 | 0.39532 | 2.82193 |
| 20261013 | reference_peak | 2.39020 | 2.35498 | 2.63493 | 0.39533 | 2.82192 |
| 20261013 | balanced | 2.39020 | 2.35154 | 2.62306 | 0.45091 | 2.79126 |

The table below presents **21 music recordings** from historical held-out rechecks, one paired file per recording, within a stable region.
Lower SC and flux error, lower energy error (dB), and higher ΔSI-SDR (dB) are better.
This is not an independent test or blind listening score.

| Seed | Path | High-Frequency SC | High-Frequency Flux Error | High-Frequency Energy Error (dB) | ΔSI-SDR dB |
| --- | --- | ---: | ---: | ---: | ---: |
| — | Compressed Input | 0.69998 | 0.86321 | 4.48301 | 0.00000 |
| 20261012 | P3 control + smooth DSP | 0.67436 | 0.89514 | 3.25407 | -0.14791 |
| 20261012 | smooth_control | 0.68574 | 0.89178 | 3.45624 | -0.14580 |
| 20261012 | reference_peak | 0.68582 | 0.89182 | 3.45575 | -0.14581 |
| 20261012 | balanced | 0.70296 | 0.90122 | 3.32315 | -0.14900 |
| 20261013 | P3 control + smooth DSP | 0.67109 | 0.89008 | 3.31591 | -0.14623 |
| 20261013 | smooth_control | 0.66945 | 0.88414 | 3.43176 | -0.14448 |
| 20261013 | reference_peak | 0.66946 | 0.88414 | 3.43168 | -0.14448 |
| 20261013 | balanced | 0.68658 | 0.89305 | 3.26843 | -0.14813 |

Among the two seeds, the `balanced` group shows a **reduction of 0.1331 dB and 0.1633 dB** in energy error compared to the same budget `smooth_control`, respectively.
However, high-frequency SC increases by **0.01722 and 0.01713**, flux error increases by **0.00944 and 0.00891**, and ΔSI-SDR decreases by **0.00319 and 0.00365 dB**, respectively.
Low-frequency preservation LSD increases by **0.08557 and 0.08398 dB**, respectively.
These small waveform differences lack perceptual significance and cannot be used as standalone disqualifications.
No net gain in spectral shape, transient, or energy is established in this round. Further root cause analysis is provided in [Root Cause Audit](restoration-root-cause-20261012.md).
Relative to P3 parent weights, both seeds show degradation in high-frequency SC, flux, and ΔSI-SDR for the `balanced` group, with energy error either increasing or decreasing.
Relative to the compressed input, the `balanced` group shows slightly worse high-frequency SC in the first seed; both seeds show transient errors still above input.

Pairwise recording bootstrap (10,000 iterations, per seed independently computed) is saved locally at
`data/reports/p4-analysis-20261012/summary.json`. For example, the `balanced–control` flux difference for the first seed has a 95% confidence interval of **[0.00226, 0.01789]**, and energy error difference of **[-0.36297, 0.09832] dB**.
Thus, the mean energy improvement cannot be described as a confirmed stable gain. The intervals are exploratory and not subjected to multiple comparison correction.
Development set includes multiple segments per recording; main report point estimates are averaged per segment, while paired analysis first averages per recording—these two estimation methods are not mixed.
All parent-child baseline comparisons for compressed input are line-by-line identical, and have been verified for window, recording set, and metric consistency.

The development music `balanced` high-frequency SC is **1.28395 / 1.19048**, significantly worse than the compressed input of **0.69958**, lacking generalizability.
This also demonstrates that showing only average energy compensation in held-out music can mask counterexamples.
Speech results are preserved separately and are not masked by extremely low speech sampling ratios or overall mean.
Variations in training domain probes and development set exist across seeds; no new independent recordings were conducted, so overfitting cannot be ruled out, nor is there evidence that results stem from overfitting to the old baseline or model capacity limits.

**Conclusion**: Semantic correction of peak constraints can be retained as an experimental capability, but is not the primary gain of this round.
The bidirectional power term passes both mathematical and numerical checks, yet still cannot ensure correct spectral and transient detail restoration.
Next steps must prioritize isolating stable envelope, transient, and phase errors, and validate them on training domain learnability and independent recordings—rather than continuing to add energy weights, training duration, or adjusting thresholds based on test tracks.
Default model and default DSP configuration remain unchanged.

Speech-stratified balanced high-frequency SC (development / held-out recheck) are **0.67312 / 0.22408** and **0.68102 / 0.22191**, respectively.
Complete energy, transient, waveform, and low-frequency metrics are retained in local results.
These results do not alter the music-based promotion conclusion.

## Real-Time CPU and Diagnostic Performance of New Weights

Fixed measurement for `balanced-20261013`, SHA-256:
`f07c185fe5502db09bed1add3dcdc897df1d9a776d0c4c0283b5ed61b9a8a29f`.
Model: 639,906 parameters, ONNX 2,599,024 bytes (2.48 MiB).
The process runs after training completion; Apple M5 Max / macOS / AC / lowpowermode=0, ORT single-thread, smooth-knee / Model gain.

Core: 5 seconds per combination, warmup 96 hops:
- Mono P99: **0.101–0.105 ms**
- Stereo P99: **0.185–0.203 ms**
- Slowest hop: **1.995 ms**
- No hop deadline miss or non-finite output.
- Core peak RSS: **35.52 MiB**.
Formal gate returns exit code **2**—this is diagnostic, and far from meeting the 1800-second per combination measurement requirement; cannot be described as a formal performance pass.

Real-time adapter uses native 44.1/48 kHz, mono/stereo real music PCM, 128-frame callback,
real-world wall-clock 30 seconds per combination. Fixed alignment buffer remains at 256 frames; no new model lookahead is introduced.

| Sample Rate / Channels | Callback P99 (µs) | Callback Max (µs) | Conservative Software Overhead (ms) | worker miss |
| --- | ---: | ---: | ---: | ---: |
| 44100 / 1 | 12.667 | 278.625 | 6.084 | 0 |
| 44100 / 2 | 13.708 | 129.042 | 5.934 | 0 |
| 48000 / 1 | 12.166 | 104.750 | 5.438 | 0 |
| 48000 / 2 | 14.041 | 170.125 | 5.503 | 0 |

All four configurations output full expected frame counts, with no NaN/Inf, queue overflow, callback deadline overrun, or producer late hop.
Peak output sample values do not exceed **0.942538**, and continuous output passes verification.
Maximum additional latency is **6.084 ms**, meeting current 8 ms target and 10 ms redline, though not reaching the initially proposed 5 ms.
This includes only software path (library/adapter); does not include sound card, drivers, or host buffering.

Real-time process peak RSS: **39.81 MiB**; `/usr/bin/time -l` records 120.25 seconds wall-clock, 22.52 seconds user + 2.33 seconds system, approximately **20.7% of one CPU core**, including PCM I/O and scheduling—this is not solely model operator computation.
All metrics are bound to this specific weight and host; not extrapolated to x86, mobile devices, or other hosts.
All reports, measurements, binary hashes, and logs are saved in `data/reports/p4-analysis-20261012/`.

No actual long-duration playback on real sound cards, blind listening, or new independent recordings were conducted.
Short-term output continuity and model audio quality eligibility are judged separately: current performance budget is feasible, but audio quality gains do not meet promotion criteria.
