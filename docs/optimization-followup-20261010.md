# 2026-10-10 High-Frequency Recovery Optimization Log

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

## Decision Target and Current Assessment

Target defined in [AGENTS.md](../AGENTS.md) and [High-Frequency Inpainting Design](inpainting-objective.md):
Recover missing high-frequency content from compressed audio, and through collaboration between model and DSP, achieve output that closely approximates the original lossless reference.
Waveform improvement is a constraint only; it does not serve as a substitute for effective high-frequency recovery.
Candidate developments and reports are retained locally and do not constitute independent testing, blind listening, or formal release eligibility.

All continuation training begins from native lossless music/VCTK at epoch168, loading only the generator. Optimizers and discriminators are reinitialized; GANs are not used. Recipe `7fda9e58…`, manifest set `a1401918…`, and original training configuration remain identical. Actual music/voice samples are drawn from 253/254 and 1/254. No new or reintroduced datasets (MUSDB, Slakh) are used.
Parent checkpoint SHA-256 is `a4fed7e7f4a463c01213d2a0705286c85a63b8e3b34c6681dcd9124cf7530650`.
Source code divergence is still publicly disclosed per [Loss Audit](loss-audit-20261010.md), and no old artifact bytes are rewritten.

## Prior Single-Factor Continuation Training

Two candidates were trained:
- 4x higher high-frequency weight
- Freezing BatchNorm statistics

Each completed 8 epochs / 1024 updates, selected at epoch1 and epoch4. Both ONNX files are 3,309,786 bytes (13 cross-runtime numerical cases passed).

Common CPU FP32 single-threaded, 896 validation rows, V2 weights-based comparison:

| Metric | Original epoch168 | 4x High-Frequency Weight | Frozen Statistics |
|---|---:|---:|---:|
| V2 Total | 1.568438 | 1.573078 | 1.566426 |
| High-Frequency Amplitude (dB) | 10.871966 | 10.930850 | 10.854841 |
| Raw Waveform L1 | 0.006845969 | 0.006845560 | 0.006844368 |

Rust-based validation uses a fixed coverage set of 89 rows / 80 recordings:
- Music: 33 rows / 26 recordings
- Speech: 56 rows / 54 recordings
This coverage includes all selected recordings, genres, speakers, and language/codec/sample rate combinations.
This set is not representative of training sampling distribution nor expected playback distribution.

| Deployment Metrics | Original epoch168 | High-Frequency Weight Candidate | Frozen Statistics Candidate |
|---|---:|---:|---:|
| High-Frequency LSD (dB), lower is better | 20.297498 | 20.218301 | 20.523769 |
| High-Frequency Energy Error (dB), lower is better | 4.121040 | 4.123657 | 4.232135 |
| SI-SDR (dB), higher is better | 28.261367 | 28.269559 | 28.284253 |
| Low-Frequency Preservation LSD (dB), lower is better | 3.940090 | 3.975398 | 3.790242 |

The high-frequency weight candidate shows a paired high-frequency LSD improvement of -0.079197, with 95% CI [-0.186249, 0.053353]; low-frequency preservation error increases by +0.035308 [0.004249, 0.066008].
The frozen statistics candidate shows a high-frequency LSD increase of +0.226272 [0.067155, 0.415446], with corresponding increase in high-frequency energy error.
Neither approach improves upon baseline weights. For music, the high-frequency LSD improvement of -0.147317 remains statistically non-significant (crosses zero); for speech, degradation occurs by +0.446422 [0.239853, 0.712943].

Isolating input phase preservation yielded unchanged magnitude output per bit, with SI-SDR increasing by +0.138593, but high-frequency LSD increasing by +1.152563 [0.444060, 1.948102]. This result does not support attributing performance degradation to architecture, BatchNorm, or training deficiency alone.
No single factor can be isolated as the root cause.

## Neural / DSP / Hybrid Pathway Comparison

Using the same original epoch168 ONNX, same Rust binary, and 89 rows / 80 recordings:

| Deployment Metric | Compressed Input | Neural | DSP | Hybrid |
|---|---:|---:|---:|---:|
| High-Frequency LSD (dB) | 45.260553 | 20.297498 | 28.010917 | 20.387972 |
| High-Frequency Spectral Convergence | 0.766668 | 0.764139 | 1.496539 | 0.764212 |
| High-Frequency Energy Error (dB) | 5.604920 | 4.121040 | 4.157431 | 4.212022 |
| High-Frequency Forward Spectral Flux Error | 0.896177 | 1.045593 | 1.823120 | 1.041811 |
| SI-SDR dB | 28.707313 | 28.261367 | 28.592249 | 28.265583 |

The hybrid path constrains DSP subband power and retains model phase, reducing degradation compared to direct amplitude/phase mixing. The final hybrid path shows a transient metric change of -0.003782, 95% CI [-0.008717, -0.000893], but a high-frequency LSD increase of +0.090474 [0.069217, 0.112537], and energy error increase of +0.090983 [0.070792, 0.111446]. **This path is not activated as default.**

The standalone DSP shows particularly poor performance in music spectral convergence and transient error, indicating that transferring texture across frequency bands, controlling energy, and output safety do not guarantee correct detail recovery. All paths have limited output, peak <0.95, and zero error in any arbitrary block; this does not constitute evidence of absence of audible artifacts.

Reintroduced high-frequency spectral convergence and flux metrics reveal: reducing weak-frequency LSD and expanding visible spectrum does not necessarily improve high-frequency structure or transient behavior—comprehensive comparison against lossless reference is required. All confidence intervals are nominal 95% ranges from development diagnostics and not independently validated through repeated tuning.

## Objective v3 Short-Term Validation

Objective v3 focuses on amplitude, spectral shape, energy, and temporal detail of missing high frequencies, as well as post-reconstruction spectral fidelity, while preserving waveform, effective phase, preserved frequency bands, and seam continuity. Specific formulas, initial gradients, and missing differentiable DSP proxies are detailed in [Objective Design](inpainting-objective.md). Training code is frozen at `3cfaceb`, with 8 causal frames, frozen statistics, 8 epochs / 1024 updates; new scores are compared under identical V3 conditions.

8 epochs completed, selected at epoch8. Identical 896 validation rows, CPU FP32 single-threaded, V3 comparison:

| V3 Metric | Original epoch168 | New epoch8 |
|---|---:|---:|
| total | 1.043180 | 1.037225 |
| High-Frequency Spectral Convergence | 0.254824 | 0.253302 |
| High-Frequency Energy Envelope (dB) | 1.539460 | 1.518960 |
| High-Frequency Temporal Change | 0.259837 | 0.253542 |
| Post-Reconstruction High-Frequency Spectral Convergence | 0.267116 | 0.258646 |
| High-Frequency Amplitude (V3 Reference Baseline) | 5.442011 | 5.466953 |
| Relative Dry Waveform Error | 0.913570 | 0.914328 |

New total improves by ~0.57%, post-reconstruction high-frequency error improves by ~3.17%; raw waveform L1 decreases from 0.006829523 to 0.006829151—insufficient to claim overall fidelity improvement. Phase and seam terms degrade; V3 high-frequency dB baseline does not equal the unclipped magnitude error of the original.

Same 89 rows / 80 recordings validated via actual Rust:

| Deployment Metric | Original epoch168 | V3 epoch8 | Paired Change & 95% CI |
|---|---:|---:|---|
| High-Frequency LSD (dB) | 20.297498 | 18.735151 | -1.562347 [-2.050557, -0.966657] |
| High-Frequency Spectral Convergence | 0.764139 | 0.748853 | -0.015286 [-0.023188, -0.008368] |
| High-Frequency Energy Error (dB) | 4.121040 | 3.992897 | -0.128142 [-0.330731, 0.059175] |
| High-Frequency Flux Error | 1.045593 | 1.048841 | +0.003248 [-0.015506, 0.018863] |
| SI-SDR dB | 28.261367 | 28.239244 | -0.022123 [-0.045407, -0.003317] |
| Low-Frequency Preservation LSD (dB) | 3.940090 | 4.114803 | +0.174713 [0.076362, 0.271892] |

Overall high-frequency LSD improves by ~7.70%, spectral convergence improves by ~2.00%, primarily driven by speech improvements; overall score is not weighted by 99.61% music playback share. Per category:

| Paired Change | Music (33 rows / 26 recordings) | Speech (56 rows / 54 recordings) |
|---|---|---|
| High-Frequency LSD (dB) | -0.226640 [-1.054935, 0.869795] | -2.349459 [-2.744221, -1.950998] |
| High-Frequency Spectral Convergence | -0.008341 [-0.024760, 0.006173] | -0.019379 [-0.026653, -0.011960] |
| High-Frequency Energy Error (dB) | -0.433949 [-0.905795, -0.039037] | +0.052065 [-0.129025, 0.231624] |
| SI-SDR dB | -0.059254 [-0.117705, -0.013111] | -0.000243 [-0.008552, 0.007973] |
| Low-Frequency Preservation LSD (dB) | +0.249416 [0.093948, 0.451828] | +0.130691 [0.020067, 0.244346] |

Music shows progress in high-frequency energy matching, though spectral shape improvements remain uncertain. Waveform and preserved band fidelity incur cost. **V3 is retained as a candidate for further high-frequency recovery optimization; epoch168 remains the baseline. Default or release weights are not replaced.**
Next steps should prioritize verifying cutoff frequency, loudness control, and reconstruction proxy differences to reduce preserved band contamination, and improve music transient fidelity—not merely extending training or optimizing SI-SDR.

New ONNX is 3,309,786 bytes (3.16 MiB), SHA-256 `c07ada11f603609962168e40db62d8692c642e1636c48daaf92e2cd57bf39f83`.
13 PyTorch/ORT/Rust numerical cases passed. All 162 BN statistics buffers and parent weights are bit-identical. 236 tensor parameters updated and finite. Arbitrary block output error is zero; all outputs are bounded and peak <0.95. This model has only undergone development validation and cannot be considered independent testing, blind listening, or artifact-free.

## Workload Scheduling Diagnostics

Added worker memory histogram to separate queueing and processing p99 reporting; histograms are not read or written in callbacks.
Native real-time threads retain existing strategy: 500 µs and 50 µs polling, idle self-spin, fixed 256-frame output latency, and failure fallback unchanged.

Prior 4-condition 30-second scheduling trials with parent model and music PCM:

| Strategy | 44.1 mono / stereo / 48 mono / stereo late callback count | Decision |
|---|---|---|
| Histogram-enabled existing strategy | 4 / 11 / 5 / 54 | Failed entire group |
| Native real-time idle spin 75 µs | 124 / 2 / 1 / 1 | Not adopted; code reverted |
| QoS only, normal thread spin | 239 / 35 / 45 / 155 | Not adopted; explicit diagnostic toggle retained |

Existing strategy had 44.1 mono callback max of 718.25 µs, exceeding 500 µs threshold. Other conditions maxed at 35–119 µs; worker processing p99 upper bound ~1.575–1.800 ms.
No combination of conditions can be stitched together to produce a "fully passing" result. Training and quality validation concluded prior to timing, but process snapshots recorded ~198.6% single-core CPU usage. This is a host-level diagnostic under background load and cannot be used to attribute all latencies to a single process, histogram, or model computation. No other application or system process was terminated.

30-minute **sample time** accelerated failure/recovery continuity test passed; still lacks real audio hardware, 30-minute wall-clock, x86_64 reference machine, blind listening, or formal audio quality certification. Software latency targets, redlines, callback behavior, and worker availability remain independently verifiable, per [Realtime Standard](realtime-standard.md).

## Processing Latency, Resource Usage, and Continuous Output

Apple M5 Max, CPU ORT single-thread, 256 FFT / 128 hop, same real PCM; each condition runs in a separate process, pre-warmed with 32 hops, then processes 10 seconds of **audio duration**. This is a non-real-time throughput measurement, not 10 seconds wall-clock or real audio card validation. Per-hop p99 (µs):

| Path / Weight | 44.1 mono | 44.1 stereo | 48 mono | 48 stereo |
|---|---:|---:|---:|---:|
| Neural / epoch168 | 943.792 | 1691.458 | 785.709 | 1655.166 |
| DSP-only / no model load | 18.000 | 11.375 | 7.125 | 11.792 |
| Hybrid / epoch168 | 968.083 | 1424.458 | 1025.958 | 1314.208 |
| Neural / V3 epoch8 | 551.792 | 1192.042 | 623.292 | 1511.292 |

DSP and Hybrid reuse existing frequency spectrum, no new frame buffer latency introduced; STFT sync itself introduces 128-sample delay. Background load varies across runs, so Hybrid's lower individual p99 cannot be interpreted as negative computational cost from DSP, nor can a single run's speed claim imply V3 performance gain. Snapshots from before model processing show background CPU load from PerfPowerServices, Node, Paseo Renderer, etc. Training and quality assessment completed; no other processes terminated.

| Path / Weight | Per-process peak RSS (MiB) | RTF Range | 10-second audio user + sys CPU (seconds) range |
|---|---:|---:|---:|
| Neural / epoch168 | 35.62–36.08 | 0.201–0.405 | 2.04–4.05 |
| DSP-only | 8.25–8.34 | 0.00177–0.00350 | 0.03–0.06 |
| Hybrid / epoch168 | 35.59–35.72 | 0.208–0.383 | 2.07–3.88 |
| Neural / V3 epoch8 | 35.52–35.75 | 0.177–0.390 | 1.83–3.93 |

RSS from `/usr/bin/time -l` peak of entire short process, including construction and PCM read; RTF is total hop wall-clock time summed / audio duration, **not CPU utilization percentage**. CPU time resolution is ~0.01 seconds, especially for DSP-only, which provides only a coarse reference. Spectral has no model load or inference; its executable still compiled with ORT support; no independent ORT-free binary size measurement derived.

Subsequently, 10-second wall-clock, 128-frame beat output diagnostics were conducted:

| Path / Weight | Four-condition worker late callback count | Max software increase latency budget (ms) | Group availability |
|---|---|---:|---|
| Neural / epoch168 | 0 / 1 / 0 / 0 | 5.845 | Short-term pass |
| Hybrid / epoch168 | 0 / 0 / 0 / 0 | 5.889 | Short-term pass |
| Neural / V3 epoch8 | 10 / 0 / 2 / 0 | 5.832 | Failed |

All conditions produced exactly the expected number of frames, no NaN/Inf, and callback budget passed. V3 callback p99: 1.708–4.125 µs, max 10.084–32.458 µs. However, 44.1 mono worker latency rate of 0.2904% > 0.1%, with 8 consecutive hops / 23.220 ms > 20 ms, and 1 input queue overflow. This condition uses existing smooth dry-sound fallback, fixed timeline with no frame loss. **Finite values and frame count correctness does not imply absence of audible artifacts, nor does a fixed delay <8 ms constitute full end-to-end real-time qualification.** 48 mono had 2 late callbacks (0.0534%), longest 5.333 ms; all other V3 conditions had no lateness.

These budgets are fixed at 256-sample software output delay plus callback overhead, excluding audio card, system audio buffer, input device, or DAC. Run durations and host load vary; the 10-second pass cannot override the earlier 30-second failure evidence. No real device or 30-minute wall-clock validation performed; candidate weights and Hybrid remain in experimental status.

## Engineering Checks

Python suite passed 232 tests; Rust workspace tests, fmt, Clippy (warnings as errors), and no-default-feature builds passed. Spectral passed actual processing without ORT mode; all three modes passed startup, saturation fallback, and non-finite input cleanup callbacks with no Rust heap allocation. 13 candidate export consistency cases and these engineering checks only verify numerical and behavioral consistency—cannot replace audio quality or continuous real-time certification.

Commit `acd9bf9` remote [CI](https://github.com/backrunner/soundex/actions/runs/38055656325) passed two checks. Training source code fixed at `3cfaceb`; runtime measurements bound to respective ONNX and binary hashes. Documentation and callback extensions in subsequent commits are version-controlled separately.

## Local Evidence

Original metrics, weights, checksums, paired bootstrap, process timing, source code, and binary receipts are preserved in `data/reports/optimization-followup-20261010/`. Training runs are saved in `data/runs/`. These directories are ignored in Git; original materials and experimental weights are not pushed.
Unused 61-second repeat PCM has been cleaned; original 31-second PCM and generation sources remain preserved.

The earlier continuation increased preserved-low-band error by 0.035308 [0.004249, 0.066008]. Frozen statistics increased high-band LSD by 0.226272 [0.067155, 0.415446]. Keeping input phase increased SI-SDR by 0.138593 dB but worsened high-band LSD by 1.152563 [0.444060, 1.948102]. The DSP comparison used the same `Neural` path and original epoch168. The follow-up DSP change increased high-band LSD by 0.090474 [0.069217, 0.112537] and energy error by 0.090983 [0.070792, 0.111446], so it remained disabled by default. The host snapshot recorded `PerfPowerServices` at about 198.6% of one CPU core. See the [inpainting objective](inpainting-objective.md) for the subsequent objective design.
