# 2026-10-11 Training Objectives, Branch Interactions, and Real-Time Startup Optimization

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

## Objectives and Fixed Conditions

Continuing the goal of high-frequency spectral inpainting and aiming to closely approximate original lossless audio through DSP post-processing. This round isolates three key components: deployment level control, complex reconstruction/spectral consistency training objectives, and amplitude/phase branch interaction. New objectives and architecture definitions are defined in [Objective Design](inpainting-objective.md).

Both continuation training runs are initialized from the V3 generator at epoch 8. The parent checkpoint SHA-256 is:
`74c0e6fecaa8c968d94b37e6bd4f9b2e77c608f3b9e4fe53158013d996021942`; the original epoch168 baseline remains unchanged and was not automatically promoted to release weights. Data, recipe, split, and sampling remain unchanged: 253/254 music/speech and 1/254, with no new or reintroduced MUSDB/Slakh datasets.
The recipe SHA-256 is `7fda9e58ab583617f3514f9dd9cfcf87f29871ae0587e91ca29ac0ddaf6b29d4`, manifest set is `a14019181a6303417a72285b45f4e28988b349ab9d164f39b64b82f44150dab4`.

The V4 control group training code is frozen at `22a3999`, and the structure group at `adc3030`. Both use the same seed, 6 rounds / 768 updates, 8 training causal frames, frozen BatchNorm, no GAN, and a new optimizer. One group uses only V4, while the other adds interaction from the encoding layer branches. Prior trial runs in the structure group revealed that the initialization of additional projections introduces extra RNG consumption, affecting subsequent training clipping—this has been halted and audit records preserved, and is not included in candidate comparisons. The `adc3030` branch isolates zero-projection construction in CPU RNG; testing confirms that legacy and new structure initialization parameters, subsequent RNG states, and clipping random numbers are bit-identical. Format reorganization does not alter the training module AST; the control branch training logic remains unchanged, and no source code or checkpoint modifications occurred during runtime.

## Deployment Level Comparison

Fixed at 89 rows/80 recordings (33 rows/26 music, 56 rows/54 speech), using the same Rust binary, identical model, and only varying the high-frequency level strategy:

| Deployment Metric | epoch168 edge | epoch168 model | V3 edge | V3 model |
|---|---:|---:|---:|---:|
| High-frequency LSD dB | 20.297498 | 20.540240 | 18.735151 | 18.780958 |
| High-frequency spectral convergence | 0.764139 | 0.742517 | 0.748853 | 0.728236 |
| High-frequency energy error dB | 4.121040 | 4.968638 | 3.992897 | 4.896352 |
| High-frequency flux error | 1.045593 | 0.980123 | 1.048841 | 0.982692 |
| SI-SDR dB | 28.261367 | 28.326201 | 28.239244 | 28.318059 |
| Retained low-frequency LSD dB | 3.940090 | 3.762295 | 4.114803 | 3.928191 |

`edge` refers to the existing cutoff edge matching (target -6 dB), `model` refers to the retained model-predicted level. All other processing—detection, frequency separation, phase, OLA, clipping, wet/dry—remains identical. The model-level adjustment does not imply full removal of the DSP chain.

Compared to `edge`, the V3 `model` shows a paired recording clustering 95% confidence interval: for music, flux error decreases by 0.169183 [-0.299764, -0.052240], SI-SDR increases by 0.187464 [0.058281, 6.0202], and low-frequency retention error decreases by 0.224433 [-0.369913, -0.087962]. However, music high-frequency energy error increases by 1.085755 [0.413926, 1.836600]. Overall, high-frequency energy error increases by 0.903454 [0.576024, 1.213531], and speech spectral convergence and LSD also degrade.

Thus, explicit diagnostic options are retained and no default level strategy is changed. This demonstrates that deployment loudness processing impacts restoration results, but cannot attribute all shortcomings solely to this stage. Coverage differs from actual playback and training ratios, and must be analyzed per music and speech separately—generalized averages cannot serve as a valid metric for music product performance.

## Benchmark Training Results

Both groups complete 6 rounds / 768 updates. The best epoch selected per group, based on fixed objectives, is epoch 4. Candidates remain in experimental status. Same CPU FP32, single-threaded, 896 validation rows, V4 weights are re-scored; V3 is also re-scored against V4, without direct cross-version total comparison. Full Rust chain is independently validated using the previously fixed 89 rows.

V4 control group vs. V3 in internal validation: total score drops from 1.132916 → 1.127071 (a 0.516% decrease), complex high-frequency reconstruction error decreases from 0.364203 → 0.356900, low-frequency damage decreases from 0.000917255 → 0.000873758 (a 4.74% drop), spectral consistency drops from 0.155117 → 0.138621 (a 10.63% drop). High-frequency energy term increases from 1.518960 → 1.587272, indicating that improved consistency does not replace energy calibration.

Full Rust Neural/MatchEdge output (same binary SHA-256 `ed36f7d9186fdd064bc0b89913a5eebdb7513aa1f706d812e1c365cc69172305`):

| Overall Metric | epoch168 | V3 | V4 Control Group | V4 + Interaction |
|---|---:|---:|---:|---:|
| High-frequency LSD dB ↓ | 20.297498 | 18.735151 | 18.624710 | 18.727317 |
| High-frequency spectral convergence ↓ | 0.764139 | 0.748853 | 0.747230 | 0.742148 |
| High-frequency energy error dB ↓ | 4.121040 | 3.992897 | 4.310981 | 4.282062 |
| High-frequency flux error ↓ | 1.045593 | 1.048841 | 1.030345 | 1.029359 |
| SI-SDR dB ↑ | 28.261367 | 28.239244 | 28.258130 | 28.248918 |
| Retained low-frequency LSD dB ↓ | 3.940090 | 4.114803 | 4.064021 | 4.038035 |

Control group vs. V3, music-specific: spectral convergence drops by 0.014663 [-0.028615, -0.000761], flux drops by 0.051552 [-0.076747, -0.023767], SI-SDR increases by 0.031656 [0.005967, 0.056321]; energy error increases by 0.323052 [0.147936, 0.516934]. Low-frequency retention error decreases by 0.050782 [-0.088921, -0.015641], though still increases relative to original epoch168 by 0.123931 [0.008546, 0.238353]. Speech spectral convergence increases by 0.006061 [0.002418, 0.009643]. These trade-offs prevent direct replacement of the baseline.

Structure group achieves a total score of 1.121569, a 1.002% drop from V3 and a 0.488% drop from the control group. Missing band spectral convergence (0.248394) and time error (0.247262) both outperform the control group (0.251008/0.249465); phase error increases from 0.854260 to 0.860808. Complex high-frequency reconstruction (0.357208) is nearly identical to the control group (0.356900), so a lower total score does not prove overall structural improvement. Both final checkpoints retain all 162 BatchNorm buffers from the parent model, and all 8 new interaction projections have learned non-zero weights.

All intervals are nominal 95% confidence intervals derived from recording clustering, fixed seed, and 2,000 bootstrap samples. The development set has been used multiple times for optimization and cannot be used to claim independent generalization or perceptual improvement.

Structure group vs. control group: all six music metrics show nominal intervals crossing zero, providing no evidence of stable musical benefit. For speech, spectral convergence decreases by 0.004145 [-0.007566, -0.000224], flux decreases by 0.003261 [-0.006705, -0.000301], but high-frequency LSD increases by 0.146335 [0.061137, 0.241098], SI-SDR decreases by 0.008808 [-0.017085, -0.002319]. Overall, LSD increases by 0.102606 [0.015817, 0.189517], SI-SDR decreases by 0.009213 [-0.018550, -0.000602]. No structural benefit can be claimed based on lower total or better average spectral convergence.

Both candidates produce finite outputs; block-level differences of 128 and irregular 31/512/997/7 with synchronized output show maximum absolute differences of zero. The structure group achieves a peak value of 0.893697 across all rows, below the 0.95 upper bound. These are correctness checks and do not replace sonic or perceptual validation. Full per-codec, per-sample rate, per-role, and per-category breakdowns are preserved. Next-phase quality optimization will prioritize high-frequency energy calibration, phase/ reconstruction constraints, and deviations from actual level chains. Independent structural toggle and fixed-data validation remain in place; no blind pursuit of additional training rounds or network scale expansion is recommended.

## Local Export and Sources

| Candidate | Parameter Count | ONNX Size | ONNX SHA-256 |
|---|---:|---:|---|
| V4 Control Group | 806,276 | 3,309,786 bytes (3.16 MiB) | `8255ea58ac2c05c9f6602972e95e667aeb4bdfed907139b8b604e4cad2ae6e06` |
| V4 + Interaction | 848,900 | 3,464,970 bytes (3.30 MiB) | `59bcdaa1a918b8dcb15737107050fc7dedc90518e1b40d7cad580ed49db7748a` |

Each group passes 13 PyTorch/ORT/Rust consistency test cases; approved under defined unit and phase error budgets, not requiring floating-point bit-for-bit equality. Training checkpoint SHA-256s are respectively:
`a152f00c6907221a4af80986364f621c1d884e9df0af22ba001ff419a574ccab` and
`9c9798247253e5197ae1f778b191043d7c3b32f2303c321c7ec7084fae00c219`.
Model, checkpoint, and original reports are retained locally and have not been promoted to official weights.

## Real-Time Startup Preparation

At startup, on the worker, prior to native real-time scheduling, a warmup of up to 2 active batches (batch=1/2) is performed, 4 times per batch; no analysis, synthesis, or DSP history is advanced. A thread is constructed to wait for worker completion, with a maximum timeout of 10 seconds. Audio callbacks do not read from ready channels, do not wait, and do not buffer. On failure, the construction thread is released and the system falls back to existing continuous dry audio. On timeout, the request returns an initialization error; no join occurs in the callback.

Fixed latency remains at 256 samples: 5.805 ms at 44.1 kHz, 5.333 ms at 48 kHz. Warmup count and time are reported independently and excluded from the stream input hop processing and queuing histogram. Explicit diagnostics can be disabled via `SOUNDEX_REALTIME_WARMUP=0`, though this is not used in default integration.

## Computational and Memory Diagnostics

Apple M5 Max; diagnostics executed after training and numerical scoring, each in a new process, using the same real PCM, Neural/MatchEdge, 32 pre-warmup hops, 10-second audio acceleration. Measured via `/usr/bin/time -l`. This is a single-run diagnostic under desktop load, not a controlled experiment for performance variance; no other applications were terminated. Environmental receipts include per-process CPU snapshots and `pmset` data, and no claims of zero background load or fixed power mode are made.

| Model | Sample Rate/Channels | hop p99 ms | Max ms | processing RTF | User+Sys seconds | Peak RSS MiB |
|---|---|---:|---:|---:|---:|---:|
| V3 | 44100/1 | 0.811 | 1.456 | 0.186 | 1.89 | 35.38 |
| V3 | 44100/2 | 1.462 | 2.335 | 0.359 | 3.63 | 35.45 |
| V3 | 48000/1 | 0.779 | 1.174 | 0.202 | 2.05 | 35.72 |
| V3 | 48000/2 | 1.428 | 1.795 | 0.386 | 3.91 | 35.66 |
| V4 | 44100/1 | 0.784 | 1.555 | 0.185 | 1.88 | 35.42 |
| V4 | 44100/2 | 1.525 | 2.635 | 0.361 | 3.63 | 35.70 |
| V4 | 48000/1 | 0.561 | 0.778 | 0.195 | 1.99 | 35.44 |
| V4 | 48000/2 | 1.381 | 1.897 | 0.385 | 3.90 | 35.62 |
| V4 + Interaction | 44100/1 | 0.804 | 1.047 | 0.194 | 1.97 | 36.28 |
| V4 + Interaction | 44100/2 | 1.258 | 1.725 | 0.362 | 3.69 | 36.59 |
| V4 + Interaction | 48000/1 | 0.758 | 1.092 | 0.209 | 2.13 | 36.64 |
| V4 + Interaction | 48000/2 | 1.547 | 2.770 | 0.408 | 4.11 | 36.59 |

Runtime/bench source code version is `1bcb091`; `adc3030` only adjusts Python initialization RNG and testing, with Rust unchanged. Saved synchronized/continuous bench binary SHA-256s are respectively:
`2a59104812d649808be5be153de71a92f0998e3d39144d4242a2b7e772b27736` and
`1bd0d12b73e8e3e31e20a6edffbe28187d0a356bb552955987d24416f9961e4c`.

RTF is wall-clock time per hop divided by audio duration, not CPU usage. CPU seconds and RSS refer to the entire process (including loading, warmup, PCM, and test tools), and cannot be directly interpreted as per-library overhead. The structure addition increases parameters by 5.29%, and RSS increases by approximately 0.86–1.20 MiB compared to the control group. Single-hop p99 fluctuations do not prove faster performance. At 48 kHz stereo, the structure group reaches a maximum 2.770 ms, exceeding the 2.667 ms hop period; actual stream performance, including sustained overruns and fallback, must be evaluated, not just p99.

## Continuous Stream Diagnostics

Same bench binary and real PCM, FFT256/hop128/callback128, no additional buffer access. V3 cold/start and warm-start each run 20 seconds, while both candidates run 30 seconds per item, across all four sample rate/channel combinations. This run explicitly requests `SOUNDEX_REALTIME_PRODUCER_RT=1`, simulating native real-time producer requests, all of which are accepted. Unlike previous short tests, which used a non-interactive producer, this result cannot be attributed to warmup improvements.

| Path | Sample Rate/Channels | Callback p99/max µs | Worker p99 upper bound ms | Overrun Count | Max consecutive error ms | Fixed latency + callback max ms |
|---|---|---:|---:|---:|---:|---:|
| V3 Cold Start | 44100/1 | 2.208/5.208 | 0.650 | 0 | 0.000 | 5.810 |
| V3 Cold Start | 44100/2 | 2.000/5.500 | 1.350 | 0 | 0.000 | 5.810 |
| V3 Cold Start | 48000/1 | 2.208/10.583 | 0.800 | 0 | 0.000 | 5.344 |
| V3 Cold Start | 48000/2 | 2.042/4.625 | 1.100 | 0 | 0.000 | 5.338 |
| V3 Warmup | 44100/1 | 1.959/5.833 | 0.650 | 0 | 0.000 | 5.811 |
| V3 Warmup | 44100/2 | 2.083/5.334 | 1.275 | 0 | 0.000 | 5.810 |
| V3 Warmup | 48000/1 | 2.000/6.834 | 0.825 | 0 | 0.000 | 5.340 |
| V3 Warmup | 48000/2 | 2.125/6.917 | 1.225 | 0 | 0.000 | 5.340 |
| V4 Warmup | 44100/1 | 2.167/12.667 | 0.800 | 0 | 0.000 | 5.818 |
| V4 Warmup | 44100/2 | 2.042/7.958 | 1.375 | 0 | 0.000 | 5.813 |
| V4 Warmup | 48000/1 | 1.959/8.250 | 0.875 | 0 | 0.000 | 5.342 |
| V4 Warmup | 48000/2 | 2.042/6.084 | 1.150 | 0 | 0.000 | 5.339 |
| Interaction Warmup | 44100/1 | 2.125/8.250 | 0.950 | 0 | 0.000 | 5.813 |
| Interaction Warmup | 44100/2 | 2.042/6.292 | 1.425 | 0 | 0.000 | 5.811 |
| Interaction Warmup | 48000/1 | 2.000/7.084 | 0.850 | 0 | 0.000 | 5.340 |
| Interaction Warmup | 48000/2 | 2.167/5.291 | 1.225 | 0 | 0.000 | 5.339 |

All 16 original `callback_samples_ns` and `worker_presentation_missed` values were independently recalculated: counts, p99, maximum consecutive error, and latency/availability policy metrics match the report. Zero overruns, zero queue overflow, zero callback period timeout, complete frame output, finite duration, peak ≤0.95—all meet the short-term software policy. Worker histograms are binned at 25 µs; the table above shows p99 upper bounds. V3 cold and warm-start both show zero overruns, so this test cannot claim that warmup reduces overruns; warmup provides a clear readiness boundary before construction completion.

Per-process resource totals are aggregated in order across the four configurations. CPU usage is user+sys time divided by total wall-clock time, normalized per core, including test tools and producer/worker—this does not represent peak per-channel CPU usage:

| Group | Wall-clock seconds | CPU seconds | Avg per core % | Peak RSS MiB |
|---|---:|---:|---:|---:|
| V3 Cold Start | 80.55 | 23.61 | 29.31% | 35.91 |
| V3 Warmup | 80.13 | 23.89 | 29.81% | 36.09 |
| V4 Warmup | 120.12 | 35.78 | 29.79% | 36.17 |
| Interaction Warmup | 120.13 | 37.49 | 31.21% | 37.14 |

This round runs at a maximum of 30 seconds per item, simulating timing, not actual device performance under 30 minutes of continuous playback. No xrun evidence on x86_64 hardware is available. Real-world integration would require reevaluation if additional buffering is introduced, as it must respect the <10 ms latency threshold. Full records are stored locally at `cost/raw-audit.json`, tied to the original model, PCM, binary, and raw logs.

## Engineering and Evidence Boundaries

The added interaction structure passes dynamic ONNX batch export testing. Rust callback allocation, failure fallback, reset, and signal continuity checks pass. A remote initial check uncovered a missed Python formatter check, which has since been fixed. Both Rust and Python components of `adc3030` passed remote CI (run 38070448340); Python: 240 passed / 1 skipped, and contract validation in Python passed.

Original reports, CPU/MPS metrics, weights, binaries, and source receipts are preserved in `data/reports/optimization-20261011/` and `data/runs/`, and are excluded from Git. Without independent audio quality testing, blind listening, or continuous real-world playback on actual audio hardware, no claim of official restoration weights is made. Training raw audio is not pushed.

The V4 control versus V3 music SI-SDR change was +0.187464 dB, with nominal interval [0.058281, 0.360202]. Both candidates had maximum chunking differences of 0 for 128-frame and irregular chunks. All six music intervals for the interaction model versus control crossed 0.
