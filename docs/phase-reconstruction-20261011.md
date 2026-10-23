# Phase Fusion and Continuous Waveform Reconstruction Comparison

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

This round yielded a noteworthy DSP improvement: by fusing phases based on residual input power—without altering the model's predicted magnitude, edge gain, or output latency—the waveform fidelity and certain high-frequency recovery metrics were enhanced. A full restoration of the original signal remains unachieved. The default strategy and official weights were not replaced. No new neural training or claimed loss reduction was introduced.

## Methodology and Fixed Conditions

The same `shape-small` model, 905,734 parameters, ONNX 3,722,845 bytes (3.550 MiB),
uses C1, FFT256/hop128, Neural/MatchEdge, 1000 Hz crossover; gain calibration disabled.
Model SHA-256: `bdef9d1269ea4b8b1e8e5e04bd6d25032b817f5401e13c51e2c064d721503782`.
Implementation source code: `19aa0eecadda43713e998a0aeabb406b9257befe`.

Three phase strategies are fixed, with identical magnitude and MatchEdge recursion:

- `model`: Original unit vector phase fusion, unchanged by default.
- `observed`: Full-frequency use of input phase as phase source, ablation. Still uses model-predicted magnitude, not dry bypass.
- `surviving-power`: Primary candidate. Model phase weight is set to
  `w_phase = w_crossover * (1 - min(A_input / A_generated_after_gain, 1)^2)`.
  When generated magnitude is ≤ input magnitude, the weight is 0; when input is missing, model phase is used. Zero input yields output 0.

The residual power ratio is heuristic and not calibrated phase reliability. It does not access unclipped references, does not fit parameters, does not smooth inter-frame variations, does not introduce lookahead or new caching. Weights are reused per channel from the time they were constructed. Only supports Neural; Spectral/Hybrid pairing is explicitly rejected.

Unit tests verify that all three strategies produce identical magnitude and gain when passed to IFFT, preserve phase in the retained bands, and actually undergo enhancement. Final reconstructed energy may still vary, as phase affects real-valued spectral projection and the overlap with adjacent windows. The old comment for `PhaseSmoother` has been corrected: it only performs intra-frame fusion and does not provide temporal phase consistency guarantees.

Development scoring criteria were frozen prior to evaluation, model, and selection rules: only if the primary candidate shows no statistically supported degradation in music-centric metrics and at least one high-frequency metric shows statistically supported improvement, will it proceed to held-out recheck. This condition is not equivalent to or proof of release. The `observed` strategy was not substituted for the primary candidate after seeing results. No hyperparameter tuning was performed based on held-out results.

## Development Set and Original Phase Diagnostics

Same 89 rows / 80 recordings: 33 rows / 26 music tracks, 56 rows / 54 speech segments.
This is a development diagnostic, not an independent test. Pairwise file hashes and isolation from training recordings have been verified.
All scoring uses real Rust output, including detectors, gain, fusion, OLA, gating, and clipping.
The default path shares the exact same 4,272 metric values as the prior round—metric consistency, not PCM byte-level comparison.

An additional NumPy-based periodic Hann FFT + CPU ORT probe was run on the same audio, skipping the first 32 frames, using unclipped high-frequency power-weighted `1-cos(phase_pred-phase_clean)` (0 is best, 2 is worst):

| Corpus | Input Phase | Model Original Phase |
|---|---:|---:|
| Music | 0.534109 | 0.975867 |
| Speech | 0.614377 | 0.977454 |

This indicates that the original model phase is significantly suboptimal, and useful information remains in the input. This probe does not include Rust's detection, fusion, gain, or clipping—thus it is not evidence of deployment consistency or independent test performance. It cannot be inferred that copying the input phase across the full frequency band would necessarily yield better results. In fact, the full-frequency `observed` strategy showed a deterioration in high-frequency spectral convergence—from 0.896469 to 0.938182, a difference of +0.012144 to +0.075401 (95% CI). Phase and magnitude, along with window overlap, must be evaluated jointly.

Actual output of the primary candidate (lower error is better; higher SI-SDR is better):

| Development Metric | Music (model) | Music (surviving-power) | Speech (model) | Speech (surviving-power) |
|---|---:|---:|---:|---:|
| High-frequency spectral convergence | 0.896469 | 0.890703 | 0.654183 | 0.644661 |
| High-frequency energy absolute error (dB) | 3.916874 | 3.664596 | 4.641319 | 4.329585 |
| High-frequency flux error | 1.391960 | 1.397953 | 0.808059 | 0.796096 |
| SI-SDR dB | 26.597193 | 26.790031 | 29.270062 | 29.328735 |
| Retained band LSD (dB) | 4.031109 | 3.787984 | 4.008570 | 3.673020 |

95% bootstrap CI by track clustering: music energy difference −0.252278 dB (−0.517444 to −0.028016),
SI-SDR difference +0.192838 dB (+0.043346 to +0.397447); both spectral convergence and flux changes cross 0.
Thus, it meets the pre-defined recheck threshold, though music transient improvements or absence of degradation are not proven.
After removing the first 0.25 seconds and the last 256 samples, the music energy difference drops to −0.208368 dB (−0.440836 to +0.027766), still insufficiently robust. Speech improvements in key metrics show no interval crossing 0.

## Held-Out Recheck

Reused the prior 78 rows / 39 recordings (42 music rows / 21 tracks, 36 speech rows / 18 segments),
each covering two sample rates and codec variants within and outside the held-out set.
Grouped by track, split group, and original file checksum, isolated from training and development validation.
Previously used in multiple diagnostic rounds—thus constitutes a held-out recheck, not a fresh independent test.
At 22:11:56 (Singapore time), the primary candidate and development report hashes were frozen before this round began.

| Held-Out Music Metrics | Original epoch168 | shape-small original strategy | surviving-power |
|---|---:|---:|---:|
| High-frequency spectral convergence | 0.745696 | 0.728370 | 0.722453 |
| High-frequency energy error (dB) | 2.503481 | 2.821492 | 2.607615 |
| High-frequency flux error | 1.119915 | 1.089917 | 1.076957 |
| SI-SDR dB | 25.751111 | 25.772151 | 25.927686 |
| Retained band LSD (dB) | 3.455125 | 3.420168 | 3.029181 |

Relative to the original model strategy, music energy average decreased by 0.213878 dB, with a CI of −0.488168 to +0.060180; both spectral convergence and flux show intervals crossing 0. SI-SDR increased by 0.155535 dB (+0.036727 to +0.362978), while retained band LSD decreased by 0.390987 dB (−0.504739 to −0.282840).

Relative to the original epoch168, music spectral convergence decreased by 0.023244 (−0.044643 to −0.001839), flux decreased by 0.042957 (−0.079012 to −0.013399), and SI-SDR increased by 0.176575 dB; however, energy error remained higher by 0.104133 dB (−0.216905 to +0.420045), so a full superiority cannot be claimed. This comparison includes neural modifications to `shape-small`; only the comparison against the `shape-small` original strategy isolates the phase factor.

Relative to the same model baseline, speech performance: spectral convergence 0.544782 → 0.503865, energy error 3.199011 → 2.862136 dB, flux 0.692443 → 0.643190, SI-SDR 30.015958 → 30.053379 dB. All improvement intervals remain non-crossing 0.
Music and speech are reported separately; actual usage benefit is not inferred from diagnostic row proportions.

For direct compression pass-through, there remains a notable gap: held-out music energy error reduced by only 1.337463 dB, but flux error increased by 0.202636 (CI +0.065450 to +0.427575), and SI-SDR decreased by 1.018799 dB (−1.919280 to −0.348657). Thus, while some improvements over the old model exist, it still falls short of near-lossless or overall goal achievement.
Across 57 physical mono/left/right rows, low-frequency absolute power variation averaged only 0.003397 dB; low-frequency residual relative to input averaged −62.539 dB; LSD changes do not imply equivalent low-frequency loudness changes.

## Continuity and Coverage Limitations

Across 89 development rows and 78 held-out rows of the primary candidate, and with 128-frame and irregular 31/512/997/7 block divisions, the maximum error across all strategies is zero. All outputs are bounded. In four synchronized stereo scenarios (two rock/electronic tracks, 44.1/48 kHz, fully aligned MP3-encoded, 12-second clips), all block-level errors are zero.
The primary candidate peaks at 0.861610, with average stereo image width error of 0.002579 dB and channel phase error of 0.006122 rad.
With insufficient sample volume, minor differences cannot be declared as image improvement.

Additionally, 10 silent, pulse, and mono-side input scenarios were tested: frame count correct, non-finite values zero, silence and uninput channels strictly zero.
Zero Rust heap allocation during construction/reset and active worker hop checks passed. Callbacks continue to perform only queue and fixed-time-axis processing.
These do not substitute physical device checks for pops or dropouts.

20 anonymous 24-bit WAV files (4 segments × 5 conditions) were generated. RMS matched first, then common attenuation applied to prevent clipping; not LUFS-matched.
Scoring table is empty, `listening_performed: false`.
Development music lacks pop/country/hip-hop/funk; held-out music lacks country/hip-hop/jazz and Chinese works.
Results cannot be generalized to all genres, regions, or all joint-stereo encoders.

## Real-Time Resource Performance

After completing all quality and phase diagnostics, the same release binary was run on Apple M5 Max using CPU ORT single-threaded, with identical real PCM interleaved measurement. Synchronized across three rounds × four sample rates/channel combinations × 10 seconds per case, including full analysis, inference, and DSP.
The `min_bandwidth_ratio` was set to 1.0; actual quality score remained at default 0.85.

| Strategy | Full hop p99 range (ms) | Synchronous process peak RSS (MiB) |
|---|---:|---:|
| model | 0.551–1.942 | 38.03 |
| surviving-power | 0.672–1.660 | 38.34 |

Each strategy also underwent four configuration × 60-second beat runs, with worker pre-warmup and macOS real-time scheduling requests enabled:

| Strategy | Full chain software latency (ms) | CPU single core (%) | Process peak RSS (MiB) | callback p99 μs | worker miss / queue overflow |
|---|---:|---:|---:|---:|---:|
| model | 5.346–5.824 | 46.95 | 42.44 | 4.833 | 0 / 0 |
| surviving-power | 5.346–5.831 | 47.19 | 38.89 | 5.167 | 0 / 0 |

All eight cases met the 8ms target, 10ms redline, callback budget, and enhancement availability. Longest continuous processing gap was 0.
Maximum callback latency for the primary candidate was 26.0 μs. Fixed 256-sample adapter latency remained unchanged; phase strategy added no new buffering.
CPU is user+system/wall time; RSS includes test framework and report caching. No claims are made about faster or more memory-efficient performance due to fluctuation.
Original callback arrays, miss bitmaps, PCM/ONNX/binary hashes, and actual strategy fields have been verified. Background load snapshots are retained.
Chain software latency does not include sound card, drivers, or host buffer overhead; any added buffer must be accounted for separately.
This round does not represent physical sound card, 30-minute continuous device operation, or x86_64 validation.

## Engineering and Next Steps

This round did not modify data, neural weights, training objectives, or the Apache-2.0 path. Original materials remain on local disk.
Rust and export-contract code in source commit [38145958140](https://github.com/backrunner/soundex/actions/runs/38145958140) passed;
276 Python tests passed locally; Rust workspace, Clippy, no default backend compilation, and formatting checks passed.
No existing ONNX was repackaged into new training weights, and no claim of PyTorch/Rust proxy consistency was made.

Next steps should focus on residual representations preserving input phase information and joint training with magnitude. Model selection should be based on real waveform reconstruction performance.
The current phase probe does not support simply increasing phase loss weight or directly discarding the phase branch.
New training must use this frozen DSP candidate as a baseline, supplemented with additional, previously unused recordings and subjective listening evaluation.
This work has not yet been implemented. The current candidate is only enabled via explicit `PhasePolicy::SurvivingPower` or via evaluation tools.

Local full materials: `data/reports/phase-reconstruction-20261011/`.
Source code entry point for reproducing development comparison: `training/evaluation/compare_phase.py`, using identical selection, two manifests, fixed ONNX, and the newly built release `soundex-stream-eval`.
