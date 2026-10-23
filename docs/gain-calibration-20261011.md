# Input-conditioned High-Frequency Gain Calibration

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

This round addresses the fixed MatchEdge target proportion identified in V7, maintaining the shape-small neural weights unchanged. A calibration module with only 10 coefficients is fitted for input-conditioned adjustment. Development and quality validation have been completed, but actual high-frequency recovery has regressed—no replacement of the default strategy is made.

## Conclusion

The calibrated parameters are not adopted as default values. While internal target MAE decreased by approximately 20%, the actual error in high-frequency energy for music and speech increased by 0.7394 dB and 0.5189 dB, respectively. The 95% confidence intervals (CI) for both differences do not cross zero; spectral convergence at high frequencies also deteriorated.

Transient and SI-SDR performance improved slightly, though this gain comes at the cost of weaker high-frequency filling and cannot be considered a successful recovery of the target. This round confirms a gap in the training objective: fitting per-frame clean high-frequency RMS or input edge does not necessarily translate into improved waveform reconstruction performance. The actual reconstruction path includes gain clamping and smoothing, crossover regions, phase, and OLA (overlap-add), and the detection cutoff and quality scoring do not align with the same frequency boundary. The degradation cannot be attributed to any single factor.

The next iteration should directly optimize the high-frequency spectral shape, energy, and transient characteristics in a joint manner, auditing preserved frequency points and phase information—rather than continuing to increase the gain calibrator or neural network based on raw energy ratios. This subsequent change has not yet been implemented.

## Fixed Protocol

- Use actual ChannelProcessor per-hop analysis, including causal padding, silence reset, detection, and context readiness.
  Lossless signals are used only for label generation and do not participate in feature extraction or gating. They are not a ported approximation of a detector.
- Features include bias, cutoff/nyquist, sample_rate/48000, six relative RMS values per preserved subband to edge, and residual high-frequency relative to edge RMS. dB features are divided by 20 and truncated to [-3, 3].
- Training labels are the dB difference between clean high-frequency and input edge, relative to a baseline of −6 dB, clipped to ±24 dB.
  A 10-coefficient Huber ridge regression is fitted (3 dB threshold, 8th-order IRLS); output is clipped to ±6 dB, final target range [-12, 0] dB.
  The non-silent gain target remains in the [0.25, 2] range, with 0.1 smoothing, low-frequency fusion, phase, clipping, and fixed output time axis preserved.
- Only training recordings are used; per recording and per sample rate, up to two codecs are selected, with first 4 seconds, skipping first 32 hops, and sampling every 8 hops.
  Only active segments with edge > 1e-4 and clean_high > 1e-7 are included in fitting. All recordings are equally weighted, with music/voice ratio 253/254 and 1/254.
  Approximately 20% of the training set is reserved for calibration validation, split by recording hash. Ridge values tested: 0.001, 0.01, 0.1, 1.0. Final re-fitting is performed on the full training set.
- Neural topology remains fixed at shape-small, SHA-256 `bdef9d1269ea4b8b1e8e5e04bd6d25032b817f5401e13c51e2c064d721503782`.
  The calibration file hash is frozen prior to actual Rust quality assessment. The existing 89-line development set is reserved for subsequent real output scoring and is not used to re-tune coefficients or select regularization—nor is it considered an independent test.
- For the same material, compare the old MatchEdge and calibrated version; evaluate quality, numerical values, continuity, real-time cost, and perceptual quality separately.
  This calibration adjusts gain only; it does not address neural phase or spectral shape prediction, nor does it modify the cutoff frequency detector.

## Reproduction and Integration

Build with: `cargo build --release -p soundex-core --bin soundex-gain-data --bin soundex-stream-eval`.
Run in training directory: `scripts/fit_gain_calibration.py --manifest <music-manifest> --manifest <speech-manifest> --binary ../target/release/soundex-gain-data --output <new-directory> --context 1`.
Input must be a pre-reviewed, per-recording paired list; original audio is not included in Git.

Integration occurs in the control thread via `calibration::read_gain_calibration(path)`, assigned to `SoundExConfig.gain_calibration`. Default is `None`, so existing application output remains unchanged. Only supports Neural/MatchEdge, FFT256/hop128/crossover1000.
The calibration JSON is not an ONNX alternative and must be loaded alongside the corresponding model, recipe, source, and validation record. Unknown schema, invalid feature contracts, or unbounded coefficients will cause loading rejection.

Local fitting, development scoring, and resource logging are written to `data/reports/gain-calibration-20261011/`.
No new audio, external teachers, or pre-trained models are introduced; calibration parameters remain unpublicized experimental results.

## Fitting and Actual Results

Source code implementation: `cb5ab5642aee81ce0808b6f212516a5f4a6bdac2`, clean workspace during fitting.
Acquisitioner SHA-256: `ca96497e646823ae49838c47be68fb318762b1e59019c1b1f0e4f578691cd1ca`.
Trained on 253 music and 193 speech recordings, totaling 1784 paired segments, 188,871 valid frames.
354 recordings used for internal fitting, 92 for internal selection—no overlap in recordings. After selection, re-fitted on full training set.
Speech recording count does not determine training weight: total speech weight remains 1/254.

| ridge | Internal Validation MAE (dB) |
|---|---:|
| Fixed −6 dB target | 7.864431 |
| 0.001 | 6.402326 |
| 0.01 | 6.289884 |
| 0.1 | 6.336464 |
| 1.0 | 6.945389 |

Selected ridge = 0.01. Final JSON is 455 bytes, 10 coefficients, SHA-256 `5b1db4881a2ca8780fca22c46e335efe31919bc73c784bf6941effe40482b55c`.
Neural graph remains 905,734 parameters, 3.550 MiB.
No retraining, export, or neural weight modification occurred—thus, no new neural network loss or ONNX numerical qualification conclusions are drawn.
Unweighted training frame prediction correction mean: −2.568 dB.
Approximately 27.21% of predictions reached the −6 dB boundary, 2.61% reached the +6 dB boundary.
This suggests the calibrator primarily acts as a suppressor—not a successful high-frequency enhancement.

### Actual Rust Output

89 lines / 80 segments from existing development set; 33 lines / 26 music, 56 lines / 54 speech.
No new independent test cases were added. Music development scoring still lacks pop, country, hip-hop, funk.
Both paths use the same neural graph, decoding, and detection parameters.
Higher HF spectral convergence is better; higher SI-SDR is better.
Pairwise 95% CI computed via bootstrap over 2000 samples per recording cluster; no multiplicity correction applied.
Old strategy on 89 lines × 52 common metrics matches exactly with previous binary results (not equivalent to byte-level PCM checksum).

#### Music

| Path | HF Spectral Convergence | HF Energy Error (dB) | HF flux | SI-SDR dB |
|---|---:|---:|---:|---:|
| Compressed Input | 0.839516 | 5.488903 | 1.051229 | 27.563202 |
| Old MatchEdge | 0.896469 | 3.916874 | 1.391960 | 26.597193 |
| Calibrated MatchEdge | 0.905943 | 4.656305 | 1.376495 | 26.634760 |

| Metric | Calibrated vs. Old Strategy | 95% CI |
|---|---:|---|
| enhanced_high_spectral_convergence | +0.009474 | [+0.003466, +0.015112] |
| enhanced_high_band_energy_error_db | +0.739431 | [+0.443927, +1.100273] |
| enhanced_high_flux_error | -0.015464 | [-0.032971, -0.001463] |
| enhanced_si_sdr_db | +0.037567 | [+0.012677, +0.076037] |

#### Speech

| Path | HF Spectral Convergence | HF Energy Error (dB) | HF flux | SI-SDR dB |
|---|---:|---:|---:|---:|
| Compressed Input | 0.723740 | 5.673287 | 0.804808 | 29.381521 |
| Old MatchEdge | 0.654183 | 4.641319 | 0.808059 | 29.270062 |
| Calibrated MatchEdge | 0.663640 | 5.160244 | 0.807289 | 29.281420 |

| Metric | Calibrated vs. Old Strategy | 95% CI |
|---|---:|---|
| enhanced_high_spectral_convergence | +0.009457 | [+0.005823, +0.013140] |
| enhanced_high_band_energy_error_db | +0.518926 | [+0.317933, +0.722585] |
| enhanced_high_flux_error | -0.000770 | [-0.002068, +0.000515] |
| enhanced_si_sdr_db | +0.011357 | [+0.004834, +0.020089] |

### Beyond the Start Segment

Full segments are processed first, then the first 0.25 seconds and last 256 samples are removed according to predefined rules—not by reselecting segments.

| Corpus | Metric | Calibrated vs. Old Strategy | 95% CI |
|---|---|---:|---|
| Music | enhanced_high_spectral_convergence | +0.009702 | [+0.003496, +0.015679] |
| Music | enhanced_high_band_energy_error_db | +0.713398 | [+0.413505, +1.046030] |
| Music | enhanced_high_flux_error | -0.010776 | [-0.019981, -0.003060] |
| Music | enhanced_si_sdr_db | +0.038667 | [+0.010468, +0.083219] |
| Speech | enhanced_high_spectral_convergence | +0.009395 | [+0.005880, +0.013209] |
| Speech | enhanced_high_band_energy_error_db | +0.528269 | [+0.317477, +0.736792] |
| Speech | enhanced_high_flux_error | -0.000793 | [-0.001948, +0.000227] |
| Speech | enhanced_si_sdr_db | +0.011491 | [+0.004898, +0.019704] |

### Continuity, Imaging, and Listening

The calibrated path has zero error across all 89 lines, with maximum error at 128 and 31/512/997/7 irregular blockings and full segment processing. No NaN/Inf values.
Two full stereo MP3 pairs were truncated to 12 seconds, covering 44.1kHz and 48kHz.
Ten cases of silence, pulse, and mono input show consistent frame counts and finite behavior; silence and mono channels remain at zero.
These are software-level checks; no listening or real-device validation has been completed.

| Genre | Sample Rate (Hz) | Imaging Width Error (dB) | Channel Correlation Error | Channel Phase Error (rad) |
|---|---:|---:|---:|---:|
| rock | 44100 | 0.0057220 | 0.0005639 | 0.0075224 |
| rock | 48000 | 0.0004005 | 0.0000348 | 0.0075130 |
| electronic | 44100 | 0.0022278 | 0.0001718 | 0.0045250 |
| electronic | 48000 | 0.0019050 | 0.0001609 | 0.0049484 |

Local file `data/reports/gain-calibration-20261011/listening/index.html` provides 4 segments × 4 conditions: compressed input, lossless reference, old strategy, and calibrated strategy.
24-bit WAV, full segment RMS matched, peak protection enforced—no LUFS matching.
Scoring sheet is empty, `listening_performed: false`.

## Engineering Verification

273 Python tests, Rust workspace tests, Clippy, and no-default-feature builds passed.
[Rust and export-contract CI](https://github.com/backrunner/soundex/actions/runs/38138687327) passed successfully.
Zero-coefficient calibration matches the old path sample-by-sample; reset recovery is consistent, silent channels do not leak, and invalid configurations are rejected.
Acquisition interface and production processor exhibit identical per-hop cutoff/gate behavior; changes to clean labels do not alter input features or gating.

## Real-Time Resource and Limitations

M5 Max / CPU ORT, single-threaded; measured after fitting, actual quality, and stereo checks.
Same PCM, 2 sample rates × mono/stereo, 3 rounds of interleaved synchronous processing, 10 seconds per case.
Stress test sets `min_bandwidth_ratio` to 1 to expand enhancement coverage; quality assessment remains at production default 0.85.
Machine background load snapshot preserved; no unrelated user processes stopped.

| Strategy | Synchronous Processing (p99 ms range) | Peak RSS (MiB) |
|---|---:|---:|
| baseline | 0.576–1.323 | 37.92 |
| calibrated | 0.566–1.298 | 37.92 |

Additional 4 cases × 60 seconds wall-clock pacing, fixed 256-sample output latency, same as PCM and pre-warmup.

| Strategy | Software Added Latency (ms) | CPU% (per core, normalized) | RSS Peak (MiB) | callback p99 μs | worker miss | Longest Miss Segment (ms) |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 5.352–5.825 | 48.36 | 38.77 | 5.125 | 0 | 0.000 |
| calibrated | 5.346–5.822 | 48.15 | 42.75 | 4.708 | 0 | 0.000 |

All eight cases met the 8ms target, stayed within 10ms red line, and satisfied callback and enhancement availability budget.
CPU/RSS values represent the entire process (four cases), including inference, DSP, and report buffering.
The three rounds are not used to assert faster inference; background load and gating affect measurements.
Calibration introduces no lookahead or new buffering. Raw callback arrays, miss maps, input/model/binary hashes, and actual loaded coefficients were all verified.
Average inference time is not used as a proxy for continuous delivery.
Software latency does not include sound card, driver, or host buffer overhead; any added buffer must be accounted for separately.
This round includes only short-term pacing simulations—no real device 30-minute, x86_64 performance, or subjective listening validation has been conducted.

All 89 calibrated outputs match whole-file processing under both fixed and irregular blocks with maximum error 0. The 10 silence/impulse/one-sided-input cases preserve frame counts and finite output; silent material and empty channels remain 0. Stress timing sets min_bandwidth_ratio to 1, while quality evaluation retains 0.85; the reported 0-miss short test is not device qualification.
