# Smooth Clipping: DSP Comparison with Fixed Weights

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

2026-10-12, commit `aeae49e`. The baseline audit [baseline-generalization-20261012.md] identified that clipping itself introduces additional high-frequency content.
This round maintains all weight freezing, modifying only optional DSP modes to validate the cause and benefit; the default remains `legacy`.
No weights were promoted, and overall audio quality is not declared acceptable.

## Implementation and Boundaries

The original curve abruptly changed slope from 1 to 0.1 at `knee = 0.9 * ceiling`, and then applied a further compression on the output already limited by the worker.
The new `LimiterMode::SmoothKnee` preserves the same `knee` and `ceiling`, modifying only the shoulder region as follows:

```
|x| <= 0.9c: y = x
|x| > 0.9c:  y = sign(x)*c*(0.9 + 0.1*tanh((|x|/c - 0.9)/0.1))
```

The shoulder region starts with a slope of 1 and zero second derivative, asymptotically approaching the upper limit. The default `c` remains 0.95.
This is a stateless, per-sample nonlinear function, introducing no new latency, lookahead, buffering, allocation, or threading.
Signals within legal bounds above the knee are still modified—thus, it is not fully transparent—and true peak reconstruction between sample points is not guaranteed.
No safety margin was expanded to gain score.

The real-time mixer applies a single dry protection, directly receiving the wet output from the worker after it has been clipped, using the original 128-sample shared fade.
It does not re-compress the mixed result. The final clamp is retained to prevent overflow during recovery when a fixed correction is added to a suddenly changing dry signal.
The mode is determined at construction time and does not support hot-swapping in callbacks.
Stereo channels are processed independently; peak-levels at high peaks are not guaranteed to remain consistent across left/right balance.
Channel isolation, phase inversion, and shared recovery timing have been tested.
This is not a stereo-linked dynamic limiter with lookahead.

Entry point: `SoundExConfig.limiter_mode`. The streaming scorer uses `SOUNDEX_EVAL_LIMITER_MODE`, and benchmarks use `SOUNDEX_LIMITER_MODE` explicitly.
Reports are bound to this mode. The Python `limit/render` provides a matching mode for deployment proxy validation.
P3 has always used legacy; this round did not modify it during execution, and the smooth-knee results were not claimed as retrained weights.
Future switching of training objectives requires separate registration and cannot reuse the old loss total.

## Fixed Samples and Selection

`training/scripts/run_limiter_study.py` requires P3 completion, first registering the model, binary, input hash, and sample window before executing all paths.
7 model/gain combinations × 2 clipping modes × 2 sample sets = 28 reports.

- Identity model, old `shape-small` MatchEdge / Model gain.
- P3 control and `excess_identity`, each with two fixed seeds for validation; selection is not based on out-of-sample scores.
- Development set: 89 tracks / 80 recordings, 33 music tracks / 26 recordings; 32768-sample center crop.
- Historical out-of-sample 39 recordings, each preserved with full encoding and intact files; 21 music tracks.
  Each file contains a 2–4 second segment.
- Stable steady-state interval from hop 72 to 256 samples before end; reports are clustered by `track_id` for bootstrap.
- Different from the prior round's baseline audit (78 out-of-sample selections), no cross-table subtraction is allowed.

A new scorer for the legacy mode is reconstructed, and across four P3 candidates, evaluated on all development paths for spectral shape, transient, energy, and SI-SDR metrics.
The difference from the frozen P3 scorer is zero across all metrics—preventing any drift in the baseline from being misinterpreted as a benefit of the new mode.

## Audio Results

Fixed-weight comparison on historical out-of-sample music. Arrow to the right is smooth-knee:

| Path | High-Frequency Spectral Shape ↓ | High-Frequency Transient ↓ | High-Frequency Energy Error (dB) ↓ | SI-SDR Relative to Input (dB) ↑ |
| --- | ---: | ---: | ---: | ---: |
| Identity DSP | 0.70159 → 0.70030 | 0.88145 → 0.87035 | 4.15524 → 4.39902 | −0.29939 → −0.13762 |
| Old MatchEdge | 0.66171 → 0.65646 | 0.99114 → 0.97985 | 2.87073 → 2.92152 | −0.37155 → −0.21395 |
| Old Model Gain | 0.65433 → 0.65087 | 0.90772 → 0.89622 | 3.94947 → 4.11817 | −0.34477 → −0.18615 |
| P3 Control / Seed 1 | 0.67888 → 0.67436 | 0.90480 → 0.89514 | 3.15370 → 3.25407 | −0.30885 → −0.14791 |
| P3 Control / Seed 2 | 0.67511 → 0.67109 | 0.89943 → 0.89008 | 3.20499 → 3.31591 | −0.30719 → −0.14623 |
| P3 excess_identity / Seed 1 | 0.66342 → 0.65932 | 0.88667 → 0.87661 | 3.46710 → 3.60667 | −0.30501 → −0.14379 |
| P3 excess_identity / Seed 2 | 0.65872 → 0.65524 | 0.87940 → 0.87012 | 3.66740 → 3.85356 | −0.30302 → −0.14163 |

Input spectral shape: 0.69998, transient: 0.86321, energy error: 4.48301 dB.
All paths show improvement in spectral shape, transient, and SI-SDR mean values, though energy error increases.
The identity model also shows the same trade-off: harmonics introduced by the original limiter may mask missing high-frequency energy, so a single energy metric cannot determine that the old curve is correct.
This does not prove that every segment improved, nor that all newly generated harmonics are harmful—further listening and independent validation are still required.

Development music follows the same trend: identity path spectral shape 0.77050 → 0.71507, transient 1.04263 → 0.87396,
SI-SDR change: −0.14647 → −0.07260 dB.
P3 excess_identity on both seeds: spectral shape 1.09259 → 1.08476 / 1.03185 → 1.02246, still below input 0.69958.
We cannot package DSP improvements as evidence that the model has resolved the issue of extra harmonic generation.

For out-of-sample speech, under the new mode: input spectral shape 0.21943, P3 seeds at ~0.22103/0.22104,
SI-SDR difference ~−0.00356 dB; old MatchEdge at 0.33089, SI-SDR difference −0.02211 dB.
Results are reported separately for music and speech; speech performance near identity does not dilute the music issue.

## Correctness Verification

- Rust workspace tests, Clippy, and builds without ORT feature passed.
- For known sine waves, peak-square error is less than 10% of the original curve; verified for slope, odd symmetry, monotonicity, zero/peak ceiling, and finite upper bounds—beyond just output clipping.
- Python/Rust replay at two sample rates covers overload, silence, gate control; smooth-knee finite-difference gradients pass.
- Mono/stereo native sample rates, short blocks, irregular blocks, finalize/reset, silence, phase inversion, and single-clip consistency passed.
- Callback and worker mode: zero Rust heap allocation; NaN/Inf, late packets, dropped packets, recovery, and queue saturation tests passed.
- Each mode ran 1800 seconds of accelerated sample time, producing 86,400,000 frames; 640 intentionally corrupted packets were rejected, zero queue overflow; known slow sine maximum step 0.01625.
  Actual runtime per test case ~1 second, not 30 minutes of card or perceptual validation.

## Performance and Delivery

Apple M5 Max / macOS / AC, low-power disabled. Model fixed to P3 excess_identity seed 1,
639,906 parameters, 2,599,024 bytes (2.479 MiB), SHA-256 `f325b54ad21d4b6f6f733c69057938c62ba9c8b81a239689865cc1af39a7efad`.
Both modes measured sequentially in order: legacy → smooth-knee.
No concurrent training, quality scoring, or compilation.
This is a single sequential diagnostic and cannot be used to infer statistical significance of minor differences.

Synchronous worker measurements at various sample rates and channel counts, 5 seconds per test, pre-warmed 96 hops, with enforced actual inference:

| Mode | Mono p99 Range | Stereo p99 Range | Max Processing Time | peak RSS |
| --- | ---: | ---: | ---: | ---: |
| legacy | 0.108–0.110 ms | 0.175 ms | 1.272 ms | 35.50 MiB |
| smooth-knee | 0.099–0.102 ms | 0.205–0.214 ms | 1.143 ms | 35.48 MiB |

No hop deadline misses were observed. Exit code 2 is due to explicit marking as diagnostic and less than 1800 seconds per group; formal release threshold remains failure, and short tests cannot be marked as passed.
Synchronous throughput under full load uses nearly one core—cannot be considered normal real-time playback usage.

Callback uses a single, real, lossless audio file synthesized as MP3 CBR128, verified by SHA for all four native sample rates and channel configurations.
Mono is the average of both channels; stereo matches the original file; no independent channel stitching, looping, or synthetic padding.
Each mode runs 30 seconds wall-clock, 128 frames per callback.

| New Mode Use Case | callback p99 | callback max | worker miss | Longest Miss | Fixed Delay + Measured Callback Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 44.1 kHz mono | 10.209 µs | 105.208 µs | 0 | 0 | 5.910 ms |
| 44.1 kHz stereo | 13.916 µs | 349.459 µs | 0 | 0 | 6.154 ms |
| 48 kHz mono | 10.333 µs | 89.333 µs | 0 | 0 | 5.423 ms |
| 48 kHz stereo | 14.125 µs | 159.417 µs | 4 / 0.0356% | 10.667 ms | 5.493 ms |

The 48 kHz stereo case also recorded one producer schedule delay; this case was not removed or rerun to achieve full green.
All four cases meet: callback p99 < 100 µs, max < 500 µs, miss ≤ 0.1%, burst ≤ 20 ms, new software delay < 8 ms.
These do not include external integration buffers or device-level latency.
All groups output frame count accurate, zero NaN/Inf, zero callback deadline overruns, peak at 0.94361, never exceeding 0.95.
During miss, the same time-aligned dry signal is output—enhanced unavailability, not audio drop.
Legacy mode had zero worker misses, callback p99 7.792–14.291 µs, maximum additional delay 5.892 ms.

The entire callback benchmark process (`/usr/bin/time -l`) averaged 19.6% and 19.1% CPU usage on a single logical core, peak RSS at 36.86 MiB and 44.625 MiB respectively.
This is a single observation and cannot be used to infer that the limiter allocation caused the RSS difference.
The short test CPU difference cannot be interpreted as performance gain.
Both cases remain below the 50 MiB RSS budget.
The new code shows zero Rust allocation in both callback and worker, with separate regression tests.
Model parameters, ONNX size, and fixed 256-frame adapter delay remain unchanged.

These are numerical and callback evidence. A single music snippet or short Apple Silicon test cannot replace 30-minute physical DAC, x86_64, or blind listening validation.
This round does not support the conclusion that "all scenarios are stable and acceptable."

All reports and PCM files are stored in the ignore directory: `data/reports/limiter-study-20261012/`:
`registration.json`, `benchmark-registration.json`, `benchmark-exits.json`, and per-mode JSON/log files.
No model promotion, `independent_test=false`, `listening_performed=false`, `audio_device_tested=false`.

Next round should fix the new deployment proxy, distinguish between valid reference peaks and actual over-generation, and re-define recovery and fidelity objectives.
Models must not rely on compensating for old limiter distortion to gain points, nor should excess weights be increased without justification.

The [baseline attribution audit](baseline-generalization-20261012.md) identified limiter-generated high frequencies. The old knee is `knee=0.9*ceiling`. Recomputed original P3 scores differ by 0; silence checks also preserve 0. Local checks passed 303 Python tests. The paced input consists of 4 rate/channel PCM combinations from the same aligned 61-second recording, verified by SHA.
