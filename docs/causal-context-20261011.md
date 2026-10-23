# Causal Context Experiment

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

The subsequent work on continuous reconstruction and multi-resolution supervision has been completed, as reported in [multires-reconstruction-20261011.md](multires-reconstruction-20261011.md). This round introduces a real DSP warmup and employs unified CPU recheck for quality assessment and diagnostic analysis.

In this iteration, a history of the past eight frames is introduced to a small-scale spectral refiner to verify whether temporal cues can improve high-frequency recovery. Both matched experiments completed eight rounds of training, followed by two additional rounds of exploration focused solely on training the temporal branch. **No overall improvement in recovery quality was achieved, and the default weights are not replaced.** The causal model aligns with the local short-term resource budget, but exhibits degradation in high-frequency energy recovery, weak historical dependencies, and persistent discrepancies in real-frame numerical audits beyond budget.

## Conclusions and Next Steps

Using the same CPU validation protocol, the causal model achieves a total loss that is approximately **0.050% lower** than frame-by-frame training, yet **0.195% higher** than the parent model. In historical held-out rechecks, the causal model shows increased high-frequency energy errors relative to frame-by-frame training—specifically, **+0.162 dB for music and +0.110 dB for speech**, both with positive 95% confidence intervals. Evidence for spectral convergence and transient improvements remains unstable and unconfirmed. After removing the startup segment, degradation persists, and cannot be fully explained by differences in warmup behavior.

Freezing the parent model and optimizing only the temporal parameters for two rounds reduces CPU loss by **0.013%** compared to the parent, though removing real history only changes the total loss by approximately **5e-7**. This marginal change does not justify extending training with the same configuration, nor does it confirm that the temporal structure has learned meaningful information. Short-term training with a single seed cannot rule out the potential of alternative causal structures.

From a resource perspective, 923,526 parameters and 3.622 MiB ONNX fall within the established scale budget. Testing four parallel 60-second causal paths on this machine introduces a new software latency of **5.344–5.848 ms**, with zero worker misses. While this meets the threshold for short-term testing, it does not imply performance on sound cards, cross-platform compatibility, or 30-minute stability. Quality and numerical integrity remain independently unverified.

The next round of experiments will prioritize the following hypotheses, **none of which were achieved in this round**:

1. Extend continuous reconstruction supervision by incorporating longer-window multi-resolution spectral and transient reference errors to verify gradient effectiveness in the temporal branch. Continue to preserve independent music/voice and non-attenuated band constraints.
2. Reduce the discrepancy between the training DSP proxy and actual Rust behavior, including cutoff, gain control, and state evolution. Audit raw model predictions and DSP-posted outputs separately to identify sources of high-frequency energy and transient degradation.
3. Compare within the same latency budget causal long-window auxiliary encoding, subband representations, and similar structures. Longer past observations do not increase lookahead, and must be re-measured for computational overhead. First, verify that historical information is indeed effective before expanding parameters or training duration.

Original research and sources are detailed in [architecture-study-20261011.md](architecture-study-20261011.md). The following hypotheses represent experiment priorities supported by current data—not yet proven to improve audio quality.

## Architecture and Protocol

- The parent model is the `shape-small` epoch 4 from the prior round, with 905,734 parameters. The causal branch adds 17,792 parameters, for a total of 923,526 parameters. Retains original gain and circular spectral shape representations, with bounded magnitude and phase correction.
- Shared spectral encoding across nine frames, using depthwise temporal convolutions and zero-initialized mixing to combine the current latent. The original U-Net processes only the current frame; no future frames, cyclic hidden states, or iterative generation are used.
- Artifact/Architecture 1.4: Input `[B,2,9,129]`, output `[B,2,1,129]`, float32, opset17, dynamic batch size 1/2. Input is ordered from oldest to current. The graph is `stateless`, meaning no implicit recursive state; the host remains responsible for maintaining per-channel spectral history.
- Rust advances history on every observed hop, including silence and bypass. Spectral history is batched per physical channel to prevent crosstalk due to variable active batch positions. Resets, worker index gaps, and model reloads clear the history. Cold history is set to -200 dB / 0 rad. The first eight hops output align with dry, and only the ninth hop allows enhancement and fade-in.
- Historical padding of approximately 21.33/23.22 ms is used to extend startup time, not to add playback delay. Output still adheres to a fixed 256-sample real-time adapter timeline. ORT control thread pre-warmup does not advance audio history.
- Caching uses only pre-allocated memory; callbacks are zero-allocated and non-blocking. Inference remains within the worker. Historical processing increases memory and inference workload, and must be measured for CPU, RSS, hop-by-hop latency, miss rate, and recovery—results cannot be inferred from parameter count alone.

## Fair Training and Selection

`prepare_temporal_study.py` generates frame-control and causal branches from the same audit-approved parent checkpoint. Native, lossless data, splits, 253/254 music and 1/254 speech samples, and V6 loss are preserved. Each group undergoes eight rounds / 1,024 updates, with LR 2e-5, 128-step warmup/cosine, clip1, frozen BN, and no GAN. Both groups use eight real warmup frames and eight supervised frames, with identical cropping and validation at 896 rows. The control group skips the first eight frames of supervision; the total score from the prior round without warmup cannot be directly compared.

Explicit `--initialize-temporal-refiner` introduces only four new tensors, with zero mixing weights and biases. It does not inherit the old optimizer. Shared parameters and RNG are kept consistent. FP32 rounding differences in nine-frame GEMM and single-frame computation may exist, and migration checks use strict absolute tolerance, not bit-level identity.

Loss continues to use the existing short-segment DSP proxy. This round only modifies temporal observation and has not yet implemented longer multi-resolution supervision or learnable DSP gain, to avoid confounding structural gains.

Each group is selected by validation total, with high-frequency magnitude as tiebreaker. A single CPU FP32, 8+8 crop, V6 total is used to select candidates, with historical hash of the frozen model as a tiebreaker. All previously used test recordings are explicitly labeled as held-out rechecks. Total score, actual Rust audio quality, real-time performance, and subjective listening are independent evidence.

## Validation and Evidence

Test coverage includes: non-zero influence of past frames, future isolation, frame-by-frame and window consistency, dynamic batch, PyTorch/ORT/Rust numerical parity, history axis and channel switching, cold start/reset/reload, irregular chunking, real worker saturation recovery, and zero-allocated Rust callbacks. `temporal-probe.onnx` is a synthetic protocol tester, not a restoration model.

Experiment configuration and checkpoints are written to `data/runs/causal-context-20261011/`. Quality, resource, and selection reports are written to `data/reports/causal-context-20261011/`. Original audio and experimental outputs are stored in ignored directories. All results are tied to actual evaluation model hashes.

## Training and Export Results

Frozen source code: `52ca6b93f12f9cb8298fd649106768c615c81fb7`. Both groups complete eight rounds / 1,024 updates. Using fixed CPU FP32 rules, **causal** is selected.

162 BN buffers are bit-identical to parent weights. All tensors are bounded. Numerical checks are listed separately; diagnostic plots are not claimed as qualified exports.

| Path | Parameters | ONNX MiB | best epoch | CPU V6 total |
|---|---:|---:|---:|---:|
| parent | 905,734 | 3.550 | 4 | 1.34916061 |
| frame-control | 905,734 | 3.550 | 6 | 1.35246801 |
| causal | 923,526 | 3.622 | 4 | 1.35179364 |

| Path | checkpoint SHA-256 | ONNX SHA-256 |
|---|---|---|
| frame-control | `e7d4782f1d9d8d270861cf6de8513d7783295c58a255096f17c0dff20f132841` | `d0ecce2d2410a3cb3b3c47fc1d0823bfd826bac41e50d15de71784db72ad03c2` |
| causal | `f15e78b4b09111c6d3deb02558f8d36420bd2048b3c3b9f35af032a7f92a56b5` | `71cb4ffae8fa4812c5c3db84f3f17e60b18a476c5c596a0843b8c256253b9523` |

### Numerical Consistency Status

| Model | 13-item parity | Failed 512-frame batch / 256 | Deployment qualification |
|---|---|---:|---|
| frame-control | Not passed | 8 | Not passed |
| causal | Passed | 10 | Not passed |

Original v2 numerical budget remains unchanged, with thresholds unmodified. Frame-by-frame comparison is rejected by standard exporters; a local `diagnostic.onnx` is generated for analysis. The causal model passes fixed export checks, but broader real-frame audits still exceed budget and are therefore retained only as diagnostic artifacts. Future audio quality and resource reports describe actual diagnostic behavior and do not imply deployment or release qualification.

Real-frame audits use a nine-frame window, covering two sample rates, 13 validation tracks, 512 output frames, and 256 dual-sample cases. Future or padded history is not used to simulate continuity. Passing 13 cases is insufficient to replace this check. Current budget is defined as relative RMS ≤ 2e-4 and inverse FFT peak ≤ 6.1035e-5 (see [numerical budget](parity.md)). Actual maximum inverse FFT difference: per-frame ~2.239e-4, causal ~9.000e-5.

FP64 comparisons and different ORT optimization levels indicate some drift stems from FP32 rounding sensitivity. This does not automatically imply audible artifacts, nor should "numerical drift" be misinterpreted as training damage. While explicit BN fusion probes reduce per-frame audit failures from 8 to 3, it still fails the budget and is not included in official exports. No thresholds were relaxed, nor was a path added to standard exporters to skip checks. Quality degradation is further supported by actual Rust output evidence.

### Raw Prediction vs. DSP Proxy Breakdown

Same 896 rows, unweighted, lower is better; units are non-comparable. `reconstruction_*` are differentiable proxies; short-segment initialization and online state differences remain.

| Metric | parent | frame-control | causal |
|---|---:|---:|---:|
| high_band_magnitude | 5.547667 | 5.561699 | 5.559763 |
| high_band_energy | 1.601419 | 1.572136 | 1.589927 |
| high_band_spectral_convergence | 0.252784 | 0.252682 | 0.252565 |
| high_band_temporal | 0.250041 | 0.250523 | 0.249548 |
| low_band_identity | 3.416124 | 3.338085 | 3.417504 |
| phase_frequency_gradient | 0.784702 | 0.885894 | 0.821448 |
| phase_time_gradient | 0.797891 | 0.793175 | 0.799949 |
| reconstruction_high_energy | 1.661806 | 1.690546 | 1.679997 |
| reconstruction_high_spectral_convergence | 0.273374 | 0.277979 | 0.276262 |
| reconstruction_temporal | 0.272218 | 0.273066 | 0.273255 |
| spectral_consistency | 0.183919 | 0.191896 | 0.186510 |

### Historical Dependency Diagnostics

With the causal model fixed, replacing all nine input positions with the current frame results in a CPU total change from 1.35179364 to 1.35179366. This is an out-of-distribution historical dependency probe, not a retrained control. The original candidate selection is frozen. This result is used solely to inform independent development set exploration and does not result in any back-feeding to the original paired experiment.

## Actual Rust Quality Comparison

Neural / MatchEdge, fixed development set of 89 rows / 80 segments, 33 rows / 26 music, 56 rows / 54 speech. High-frequency (HF) metrics: lower is better; SI-SDR higher is better. Pairwise confidence intervals (CIs) are bootstrapped per track/split_group/source at 2,000 iterations, 95%, with no multiple comparison correction.

### Development Set Music

| Model | HF Spectral Convergence | HF Energy Error (dB) | HF flux | SI-SDR dB |
|---|---:|---:|---:|---:|
| frame-control | 0.909382 | 4.023011 | 1.404272 | 26.623741 |
| causal | 0.900147 | 3.926692 | 1.400837 | 26.618250 |

### Development Set Speech

| Model | HF Spectral Convergence | HF Energy Error (dB) | HF flux | SI-SDR dB |
|---|---:|---:|---:|---:|
| frame-control | 0.653674 | 4.432797 | 0.807768 | 29.265411 |
| causal | 0.654100 | 4.554872 | 0.807965 | 29.267906 |

## Held-Out Historical Recheck

78 rows / 39 segments, 21 music, 18 speech, including AAC160 and two sample rates. Track/split_group/source checksums and train/validation isolation are verified; artist-level isolation is not claimed. This is repeated-validated historical material, **not a new independent test**. Music coverage lacks Chinese, country, hip-hop, jazz, etc. Prior to scoring, all model hashes are bound. Both model structures are recorded in parallel, and no reselection is made based on this.

### Held-Out Music

| Model | HF Spectral Convergence | HF Energy Error (dB) | HF flux | SI-SDR dB |
|---|---:|---:|---:|---:|
| causal | 0.721379 | 2.797578 | 1.068937 | 25.778961 |
| frame-control | 0.725154 | 2.635807 | 1.077788 | 25.772837 |
| original | 0.745696 | 2.503481 | 1.119915 | 25.751111 |
| parent | 0.728370 | 2.821492 | 1.089917 | 25.772151 |
| v6 | 0.733257 | 2.782127 | 1.092556 | 25.770247 |

Causal vs. frame-control:

| Metric | Average Change | 95% CI |
|---|---:|---|
| enhanced_high_spectral_convergence | -0.003775 | [-0.020836, +0.007734] |
| enhanced_high_band_energy_error_db | +0.161772 | [+0.039963, +0.277121] |
| enhanced_high_flux_error | -0.008851 | [-0.033852, +0.010318] |
| enhanced_si_sdr_db | +0.006124 | [+0.001188, +0.013725] |

### Held-Out Speech

| Model | HF Spectral Convergence | HF Energy Error (dB) | HF flux | SI-SDR dB |
|---|---:|---:|---:|---:|
| causal | 0.546814 | 3.162596 | 0.691594 | 30.015756 |
| frame-control | 0.547795 | 3.052518 | 0.688465 | 30.012989 |
| original | 0.553484 | 2.934310 | 0.684675 | 30.009073 |
| parent | 0.544782 | 3.199011 | 0.692443 | 30.015958 |
| v6 | 0.547228 | 3.208142 | 0.691184 | 30.016087 |

Causal vs. frame-control:

| Metric | Average Change | 95% CI |
|---|---:|---|
| enhanced_high_spectral_convergence | -0.000981 | [-0.003784, +0.001854] |
| enhanced_high_band_energy_error_db | +0.110078 | [+0.042986, +0.180141] |
| enhanced_high_flux_error | +0.003129 | [-0.000244, +0.006977] |
| enhanced_si_sdr_db | +0.002767 | [+0.001087, +0.004686] |

## Startup Segment and Steady-State Supplement

To avoid treating the eight-hop dry warmup as a gain in temporal information, all paths uniformly truncate the first 0.25 seconds and last 256 samples, then fully process and score. This protocol is recorded before held-out scoring, and full-length metrics are preserved for reference, without influencing selection or tuning. The following remains historical held-out diagnostics.

| Corpus | Metric | Causal vs. Frame-Control | 95% CI |
|---|---|---:|---|
| Music | enhanced_high_band_energy_error_db | +0.124436 | [+0.005181, +0.239638] |
| Music | enhanced_high_flux_error | -0.023870 | [-0.076889, +0.007634] |
| Music | enhanced_high_spectral_convergence | -0.009715 | [-0.037038, +0.007059] |
| Music | enhanced_si_sdr_db | +0.001091 | [-0.002744, +0.005192] |
| Speech | enhanced_high_band_energy_error_db | +0.089212 | [+0.028807, +0.150668] |
| Speech | enhanced_high_flux_error | +0.004391 | [+0.000764, +0.008664] |
| Speech | enhanced_high_spectral_convergence | +0.000314 | [-0.002318, +0.003080] |
| Speech | enhanced_si_sdr_db | +0.002385 | [+0.000765, +0.004219] |

## Targeted Optimization: Added Temporal Branch

A matched eight-round experiment probe, where all nine input positions are replaced with the current frame, shows only a total loss change of approximately **2.29e-08**, indicating no discernible benefit from historical information. Non-zero parameters do not imply learning to use history. Therefore, a targeted development set exploration is added: starting from the same `shape-small` parent, all parent parameters and BN buffers are frozen, and only the 17,792 temporal parameters are updated, with LR 1e-3, warmup 32/cosine. This changes the optimization scope and learning rate and is not considered a third arm of the original paired experiment. No reselection or new held-out scoring is added. The training source code is `9fd534d7ce6c4d446737865ee457d9b91803a292`, best epoch 2.

| Item | Value |
|---|---:|
| Parent Model CPU Total | 1.349160610 |
| Targeted Temporal Branch CPU Total | 1.348985210 |
| Relative to Parent | -0.01300% |
| When Repeating Current Frame | 1.348985707 |
| Change After Removing Real History | +0.000000497 |
| Parameters / ONNX MiB | 923,526 / 3.622 |

All parent tensor values are bit-identical: `true`. 13-item parity is preserved: `true`. This exploration is for development diagnostics only and has not undergone new held-out, 512-frame numerical recheck, or real-time resource acceptance. It does not replace default weights.

| Development Corpus | HF Spectral Convergence | HF Energy Error (dB) | HF flux | SI-SDR dB |
|---|---:|---:|---:|---:|
| Music | 0.896314 | 3.902270 | 1.391737 | 26.598961 |
| Speech | 0.653902 | 4.634331 | 0.808085 | 29.269911 |

Targeted checkpoint SHA-256: `6419d9d82679eab398c5116a790f2f1d7e400afe3b611d9871ef4af0fc4c3789`.
Targeted diagnostic ONNX SHA-256: `9b49b3210a3d616adb6a5080a70babc0c9800271649cd9acd4ef58f930f05c87`.

## Relative Compression and Low-Frequency Preservation

Below shows causal output minus compressed input, all relative to the same lossless reference. Lower HF metrics are better, higher SI-SDR is better. High-frequency recovery cannot be interpreted as success based on coverage or LSD reduction alone.

| Held-Out Corpus | Metric | Change | 95% CI |
|---|---|---:|---|
| Music | high_band_energy_error_db | -1.147499 | [-1.837656, -0.464422] |
| Music | high_spectral_convergence | +0.000373 | [-0.074853, +0.105814] |
| Music | high_flux_error | +0.194616 | [+0.072430, +0.379076] |
| Music | si_sdr_db | -1.167524 | [-2.193579, -0.425649] |
| Speech | high_band_energy_error_db | -0.017767 | [-0.816191, +0.701631] |
| Speech | high_spectral_convergence | +0.039663 | [+0.001161, +0.085179] |
| Speech | high_flux_error | +0.068196 | [+0.036020, +0.104973] |
| Speech | si_sdr_db | -0.116365 | [-0.228763, -0.045449] |

Music high-frequency energy approaches reference, but transient and full waveform fidelity degrades. Speech spectral and transient quality also shows no improvement. This again demonstrates that "recovery of high frequencies" does not equate to "recovery closer to lossless."

Low-frequency relative energy residuals are approximately -53.41 dB (music), -59.37 dB (speech) compared to input. Absolute power changes are about 0.00568/0.00204 dB. Music has only 21 rows/15 segments with sufficient effective frequency content, so extrapolation to 42 rows is not valid. Low-frequency preservation LSD remains weak at ~3.48/3.67 dB, and cannot be inferred from small power differences.

## Continuity and Listening

Both development outputs and causal held-out outputs contain no non-finite values. For 128-frame and 31/512/997/7 irregular chunking, maximum error is zero. Additional testing on two synchronized stereo rock/electronic original tracks, full MP3 encoding/decoding and aligned, 12-second segments, two sample rates; causal model performs 10 cases of silence, pulse, and mono input. Local listening page: `data/reports/causal-context-20261011/listening/index.html`. Five conditions: lossless reference, compressed input, original epoch168, frame-by-frame comparison, and causal model. 24-bit WAV, full RMS matching with peak protection, not LUFS-matched. Irregular chunking maximum error is zero; 10 boundary cases have correct length, finite output, and silence and empty right channel remain exactly zero. Synchronized stereo causal output has width error of ~0.021–0.056 dB, correlation coefficient error 0.00096–0.00220, and inter-channel phase error 0.0166–0.0794 rad. Relative to frame-by-frame results, both improvements and regressions are present. Listening scores have not been filled; `listening_performed: false`. No subjective validation is claimed.

## Inference Latency and Resource Usage

M5 Max, CPU ORT single-thread; measured after training and quality assessment; machine has background loads from media analysis, graphics programs, and simulators, not an isolated benchmark. Two groups × two sample rates × mono/stereo × three rounds of synchronized testing, interleaved, each case processing 10 seconds of real music. Mix of different channels and sample rates prevents division of extreme values as performance ratios.

| Model | Sync p99 ms Range | Max RSS MiB |
|---|---:|---:|
| frame-control | 0.579–1.408 | 37.92 |
| causal | 0.576–1.533 | 38.64 |

Each group runs four cases × 60 seconds wall-clock pacing, same 61-second real music PCM, default pre-warmup, and scheduling request.

| Model | Software Latency (ms) | CPU % (Single Core Normalized) | RSS MiB | callback p99 μs | callback max μs | worker miss |
|---|---:|---:|---:|---:|---:|---:|
| frame-control | 5.341–5.858 | 45.08 | 38.48 | 5.667 | 52.917 | 3 |
| causal | 5.344–5.848 | 52.04 | 38.78 | 5.875 | 46.292 | 0 |

The frame-by-frame group has three misses, all isolated to a single hop (longest 2.667 ms), while the causal group has zero misses. All cases achieve 8-case availability within budget. The frame-by-frame group does not achieve zero misses, but the causal group achieves zero. All cases produce correct output length, no non-finite values, and peak ≤ 0.95. Raw callback timing arrays, miss bitmaps, and budget formulas are re-verified. During loss enhancement, time alignment with dry axis is maintained. CPU/RSS reflects the entire four-case process, including ORT/DSP/reporting cache. Original time arrays, miss bitmaps, and input/model/binary hashes are separately re-verified. Worker delivery availability does not imply that the model enhances every hop—silence, bandwidth bypass, and causal history padding produce dry output. This is a short-term simulation test; no sound card testing, 30-minute stability, or x86_64 real-world performance conclusions are drawn. The fixed 256-frame software timeline does not include device/driver/host buffering; additional buffer insertion must still be within the 10ms red line.

## Evidence and Delivery

- Source CI: [38120407415](https://github.com/backrunner/soundex/actions/runs/38120407415), Rust/export-contract both succeeded; local Python 267 items passed.
- `data/runs/causal-context-20261011/`: frozen source code, configuration, training logs, and checkpoints.
- `data/reports/causal-context-20261011/`: selection, hash binding, CPU validation, actual Rust audio quality, historical diagnostics, listening, and raw resource records.
- Data, experiment checkpoints, and ONNX files remain in Git-ignored directories; no new audio sources, external pre-trained weights, or teacher outputs were introduced.

The opening comparison's music high-band energy errors increase by 0.162/0.110 dB; both 95% confidence intervals are positive. These are regressions, despite small loss changes. The common warmed scoring region excludes the first 0.25 seconds and final 256 samples.

Fixed/irregular block comparisons have maximum error 0; the causal worker has 0 misses in the short paced test. The framewise worker has 3 isolated single-hop misses; the causal group has 0. Reducing extended parity failures from 8 to 3 still does not pass; 0 failures are required under the unchanged tolerance. No 0-miss short run replaces long-duration/device evidence.
