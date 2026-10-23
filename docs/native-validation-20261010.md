# Lossless Music Training Candidate: First Validation – 2026-10-10

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

**Conclusion:** The model has learned high-frequency recovery capabilities, but this validation is insufficient to justify the release of recovered weights for production use.
**Recommendation:** Prior to further training, adjust the scaling and weighting of the loss function, and refine deployment strategies for bandwidth, phase, and working thread allocation. Conduct small-scale controlled experiments before continuing training for hundreds of additional rounds. Current evidence does not independently indicate model architecture failure.

The subsequent [Loss Audit and 8-Round Controlled Retraining](loss-audit-20261010.md) has been completed: minor improvements in phase and waveform fidelity, with high-frequency errors rising again. The original weights are retained. This document also records metadata correction and real-time scheduling comparisons; the present report preserves the original results from the first diagnostic run.

---

## Weights and Loss

Training completed at 01:00 Singapore time, consisting of 200 rounds and 25,600 updates, with random initialization—no prior weights or teacher model.
The selected candidate checkpoint from round 168 is used for validation. Total loss: 11.6241, high-frequency error: 10.8692.
At round 200, high-frequency error increases slightly to 10.9424, slightly above the minimum observed. The average high-frequency error across rounds 90–100 was 11.4775, and the last 10 rounds averaged 10.9740. The additional 100 completed epochs improved this average by approximately 4.4%, suggesting a plateau.

| Metric (from 896 fixed validation samples) | Direct Compressed Input | Best Model |
| --- | ---: | ---: |
| High-frequency magnitude error (masked-weighted dB absolute error) | 36.3279 | 10.8692 |
| Phase error (energy-weighted `1 - cos(delta)`) | 0.53944 | 0.81271 |
| Steady-state waveform L1 (normalized PCM) | 0.0068255 | 0.0068452 |
| Total weighted loss (validation, excluding GAN term) | 36.4364 | 11.6241 |

Each validation row ID matches exactly with the fixed validation set in the checkpoint. The total loss is dominated by high-frequency components and should not be interpreted as a perceptual quality score. The G loss during training includes the GAN term and is therefore not directly comparable to the validation total loss.
Currently, the waveform term contributes only about 0.000685 to the total loss, while the high-frequency term contributes approximately 10.87. Before imposing stronger waveform or phase constraints, the loss components must be normalized by magnitude and gradient analysis must be performed to avoid merely increasing a still negligible weight.

Candidate checkpoint SHA-256: `a4fed7e7f4a463c01213d2a0705286c85a63b8e3b34c6681dcd9124cf7530650`.
Local ONNX size: 3,309,786 bytes (3.16 MiB), SHA-256: `cd0546e54c4477195a2a6567192df515285b86c42b286baf4a9e40f90160d063`.
13-point PyTorch / ONNX Runtime / Rust numerical consistency checks passed; existing policies remain unchanged.

---

## Independent Test Set Sampling Diagnostics

The full test set contains 7,891 rows. For this round, deterministic sampling rules were frozen: one unit selected per corpus, genre, encoding, and sample rate, with all test recordings and channel roles fully populated. A total of 209 rows were generated, covering 21 music tracks, 18 speech samples, 10 encoding configurations, and both 44.1 kHz and 48 kHz sample rates.
Production-grade Rust gate, STFT, cross-band processing, loudness/phase handling, clipping, and flush operations were applied. Testing was performed on full segments, 128-frame blocks, and irregularly segmented portions. No Python waveform implementations were substituted.
The original test list was not modified; all paired file hashes have been audited. The following values represent the fixed diagnostic set.

| Metric | Result |
| --- | --- |
| High-frequency LSD (compressed → enhanced) | 43.23 → 17.93 dB |
| High-frequency LSD difference by track (95% CI) | −31.16 to −18.38 dB |
| Music/speech high-frequency LSD difference | −25.83 / −21.90 dB |
| SI-SDR (compressed → enhanced) | 24.21 → 23.58 dB, average drop of 0.63 dB |
| Maximum absolute difference between full and two block types | 0 |
| Non-finite output rows | 0 |

Improved high-frequency spectral content does not necessarily imply perceptual benefit. In one blues recording, the average high-frequency LSD increased by 2.69 dB. This genre has only one independent test piece and cannot be generalized to the entire blues category. Additional independent test pieces are required for genres such as blues, pop, and funk. Degradation in phase, waveform, and high-bitrate input should be included in the next round of fidelity comparisons.

---

### Recheck of Low-Frequency and Ultra-peak Metrics

The original low-frequency LSD mean was 4.47 dB, exceeding the current 0.25 dB threshold. However, this metric is based on a weighted average of extremely sparse frequency points and does not imply a 4.47 dB change in actual low-frequency loudness. Additionally, for 115 physical channel rows (mono/left/right), the first and last 1,024 samples were removed, and the complex STFT energy below the cutoff frequency minus 500 Hz was measured:
- Average absolute energy gain change: only 0.0026 dB (peak: 0.0588 dB)
- Relative residual input low-frequency energy dB ratio change: average −50.91 dB

This recheck confirms minimal actual low-frequency changes. The original metric requires further evaluation in conjunction with energy and bandwidth estimates. No release threshold was adjusted, and no full quality acceptance is claimed based on this result.

There are 273 existing over-range samples in physical channel inputs; 4 are retained in output, with no new overflows. Most other overflow cases originate from energy-normalized virtual mid-channel roles, which may themselves exceed full-scale. These cannot be directly attributed to model clipping. Offline retained input overflows still do not meet the current zero-clipping release criterion and require review of the dry path.

The left/right random snippet start points differ, so they cannot be stitched into a synchronized stereo signal. Additionally, two independent stereo test pieces (rock/electronic) were fully encoded to MP3 128, aligned, and 12-second segments cropped to verify both sample rates—yielding 4 stereo scenarios. Block-level differences were all 0, no overflows, and stereo image width errors increased slightly by 0.0015–0.0065 dB. Phase errors also increased slightly, but these cannot replace blind listening.
Further tests on two sample rates and mono/stereo silence/pulse patterns (8 scenarios): output frame count accurate, no non-finite values, silence peak at 0.

---

## On-Device Performance and Continuous Output

Hardware: Apple M5 Max, ORT single-thread, FFT 256 / hop 128.
Serial baseline: 10 seconds per scene. Real-time adapter uses a 30-second MP3 128 audio stream from held-out music, with four scenes tested. These are short-term engineering diagnostics and do not involve physical audio devices, nor do they include x86 or 30-minute stress testing.

| Scenario | Serial P99 Software Latency | Real-Time Adapter Conservative Overhead | Enhanced Output Not Arrived |
| --- | ---: | ---: | ---: |
| 44.1 kHz / 1 ch | 3.474 ms | 5.963 ms | 0.019% |
| 44.1 kHz / 2 ch | 3.940 ms | 5.823 ms | 0.271% |
| 48 kHz / 1 ch | 3.220 ms | 5.473 ms | 0.018% |
| 48 kHz / 2 ch | 3.711 ms | 5.357 ms | 0.320% |

The serial P99 includes 128-sample algorithm processing and waiting time. The real-time adapter overhead includes 256-sample fixed buffer and observed maximum callback latency, excluding existing audio card buffering. These two paths represent different access methods and cannot be used to infer that the ~3.9 ms serial latency is representative of end-to-end latency in asynchronous adapters. All four real-time outputs maintain accurate frame count, with 0 callback timeouts, queue overflows, or non-finite outputs. The stereo enhancement lateness rate of 0.27%–0.32% exceeds the current 0.1% availability target. This missing coverage is not reflected in output dropouts, and no output pause is observed. However, the system cannot be claimed to be always available or free of audible artifacts.

External `/usr/bin/time -l` measurement shows peak process RSS:
- Serial: 39.09 MiB
- Real-time adapter: 36.11 MiB

Stereo processing consumes approximately 0.34–0.37 CPU core budget. The four-scenario real-time test (user + system) / wall time ≈ 0.49 cores, including overhead from frame provisioning and timing tests. This cannot be used as a pure model CPU consumption metric.

---

## Next Steps and Evidence Boundaries

1. **Retain the checkpoint from round 168 as baseline.** Prior to any further action, conduct controlled comparisons on the fixed validation set—specifically evaluating phase/waveform fidelity constraints, loss normalization, and conservative frequency band controls—before deciding on a limited low-learning-rate retraining or full retraining. A small loss reduction is not justification for indefinite continuation.
2. **Refine or supplement metrics for weak-energy frequency points**, separating physical channels, virtual roles, and pre-existing overflow cases. Add synchronized stereo test cases. A single threshold relaxation cannot substitute for actual fidelity or perceptual validation.
3. **Address real-time thread deadline availability issues**, preserving current non-drop-frame output guarantees, and then conduct extended and device-level testing.
4. **This test result is strictly for diagnostic purposes and must not be used as training data.** Final acceptance based on these results should be evaluated using a separate, untouched set of samples not involved in the tuning process. Reusing test sets as evidence of generalization is invalid.

**Pending tasks:**
- Full 7,891-row deployment scoring
- ViSQOLAudio evaluation (not available on this machine)
- Human blind listening
- 30-minute stability test
- x86 performance benchmarking
- Full artifact rights and ownership verification

Therefore, the weights remain candidates for local experimentation only.
Original audio, paired files, reports, audition WAVs, and ONNX files are preserved in the ignored `data/` directory. Only source metadata and report documents are shared.

**Local evidence directory:** `data/reports/native-validation-20261010/`
Contains: fixed sampling list, loss curves, compressed baseline comparison, 13-point parity checks, 209-row diagnostics, overflow/bandwidth rechecks, physical channel energy retention, stereo/boundary test cases, two performance reports, and resource measurement logs. See also [Training Record](native-training-20261009.md).
