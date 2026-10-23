# Spectral Representation and Capacity Experiment: Training, Audio Quality, and Real-Time Cost

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

Three original structural experiments and a V6 control re-computation were completed in this round. Recent paper insights and design trade-offs are referenced in [Architecture Survey](architecture-research-20261011.md). All candidates remain in local experimental state and have not replaced default weights.

## Fixed Protocol

- Training code: `7cfaf78490f385712a65fbcb5f198b49622fc593`; parent model is V4 epoch4.
- Each group runs 8 epochs / 1,024 updates, under the same V6 loss, LR 2e-5, warmup128/cosine, clip1, frozen BN, and no GAN.
- Same 300 songs of reviewed music and VCTK; music/speech sampling at 253/254 and 1/254; no external weights, data, or teacher targets introduced.
- FFT256/hop128; model processes frames independently; 8 frames used for reconstruction loss; no new lookahead, model state, or DSP buffering added.
- The control reuses the identical configuration, parent model, and sampling setup from the V6 run; this round involves CPU re-computation, export, and parity verification, matching the original results exactly. This is not a multi-seed replication experiment.
- Each group is selected by validation total; then, within the same CPU FP32 896-line total, candidates are chosen based on the same hash freeze, followed by historical held-out set scoring.
- Recording-level cluster bootstrap: 2,000 iterations, 95% confidence interval; no multiple-comparison correction across models or metrics; conclusions are exploratory.

## Model and CPU Verification

| Path | Parameters | ONNX MiB | best epoch | V6 total | Relative to Control |
|---|---:|---:|---:|---:|---:|
| control | 806,276 | 3.156 | 4 | 1.348875 | +0.000% |
| polar-small | 905,734 | 3.550 | 4 | 1.342853 | -0.446% |
| shape-small | 905,734 | 3.550 | 4 | 1.342215 | -0.494% |
| shape-large | 1,399,942 | 5.436 | 4 | 1.340974 | -0.586% |

Fixed selection: **shape-large**. All 13 components passed PyTorch/ORT/Rust parity checks; 162 parent BN buffers matched bit-by-bit, with all parameters bounded.

| Path | Checkpoint SHA-256 | ONNX SHA-256 |
|---|---|---|
| control | `22c39a78baca386c2b30d34641f02dea7cf83a331f00eefd8390d769f0a8a433` | `18ea464cfcc4c978b252f1b6ad04ce5e1fc427dc225b58759b6a8e69d2b57336` |
| polar-small | `aa1cdf23610e286e0783ea4f75c7305e2e5569e7774f94ccf5f56ba6a5cfba04` | `9fccf1068e8958b2d51abb3af2f7e0dbfcc8fd529e73c738327a6dbd69c2a630` |
| shape-small | `a5166a6f56239a31a48986725188aa19c1c199c565904c76aca0f160a19f393a` | `bdef9d1269ea4b8b1e8e5e04bd6d25032b817f5401e13c51e2c064d721503782` |
| shape-large | `8d5795cfe62a572e2221c657997951f6fb3b843911219087584e006b2218a53c` | `a9a5096d1cc87eaf3c151ce9d25fc4dc1f4abe33db84b7a94076739c66fa78d3` |

## Raw Predictions and Differentiable DSP Proxy

The following are per-frame, unweighted, CPU FP32 validation results for the same 896-line test. All are lower is better; units and scales differ across components and cannot be directly summed. The first seven constrain raw model predictions; `reconstruction_*` terms are post-processed with gain, fusion, and OLA; `spectral_consistency` measures spectral coherence. Definitions and normalization are as defined in [V6 Loss](inpainting-objective.md#v6-deployment-aware-continuation-experimental). The proxy uses static cutoff frequency, short fragment first-frame gain initialization; online detection, gain history, gating, limiting, and fallback still differ from the real Rust processing.

| Metric | control | polar-small | shape-small | shape-large |
|---|---:|---:|---:|---:|
| high_band_magnitude | 5.475147 | 5.457680 | 5.430957 | 5.422090 |
| high_band_energy | 1.598176 | 1.583369 | 1.599671 | 1.587778 |
| high_band_spectral_convergence | 0.250706 | 0.250230 | 0.248926 | 0.248802 |
| high_band_temporal | 0.247553 | 0.247322 | 0.246518 | 0.246944 |
| low_band_identity | 3.390601 | 3.365368 | 3.395655 | 3.401853 |
| phase_frequency_gradient | 0.830409 | 0.840599 | 0.791815 | 0.794893 |
| phase_time_gradient | 0.773541 | 0.774771 | 0.783691 | 0.774716 |
| reconstruction_high_energy | 1.712001 | 1.715422 | 1.713754 | 1.713874 |
| reconstruction_high_spectral_convergence | 0.275842 | 0.273303 | 0.273768 | 0.273490 |
| reconstruction_temporal | 0.276602 | 0.270178 | 0.273999 | 0.274598 |
| spectral_consistency | 0.193432 | 0.193712 | 0.187107 | 0.186872 |

Selected candidate shows reduced high-frequency magnitude and energy errors, but a slight increase in reconstructed high-energy error; overall loss improvement does not imply simultaneous improvement across all objectives.

## Actual Rust-Developed Audio

Fixed 89 lines / 80 segments: 33 lines / 26 music segments, 56 lines / 54 speech segments; Neural/MatchEdge default processing. First three metrics are lower is better; SI-SDR higher is better. All arms use the same input checksum, identical binary, and configuration.

### Music

| Path | HF Spectral Convergence | HF Energy Error (dB) | HF Flux Error | SI-SDR dB |
|---|---:|---:|---:|---:|
| control | 0.904906 | 3.895419 | 1.405710 | 26.598670 |
| polar-small | 0.901904 | 4.043868 | 1.385432 | 26.622565 |
| shape-small | 0.896469 | 3.916874 | 1.391960 | 26.597193 |
| shape-large | 0.906221 | 3.913919 | 1.398139 | 26.601327 |

### Speech

| Path | HF Spectral Convergence | HF Energy Error (dB) | HF Flux Error | SI-SDR dB |
|---|---:|---:|---:|---:|
| control | 0.655987 | 4.634038 | 0.807928 | 29.271228 |
| polar-small | 0.661214 | 4.826734 | 0.809219 | 29.274686 |
| shape-small | 0.654183 | 4.641319 | 0.808059 | 29.270062 |
| shape-large | 0.654774 | 4.687198 | 0.808060 | 29.268554 |

## Historical Held-Out Recheck

78 lines / 39 segments (21 music, 18 speech; two conditions per segment), including 44.1/48 kHz and AAC160. These recordings were previously used for diagnosis; **this is not a new independent test**. Track, split_group, and source checksums were verified to be isolated from training/validation. Artist isolation is not claimed.
Music lacks coverage in Chinese, country, hip-hop, jazz, and other genres; cannot extrapolate to all regions or styles.

Table below shows `candidate` = shape-large, `original` = native epoch168, `parent` = V4 epoch4, `v6` = current control.

### Music

| Path | HF Spectral Convergence | HF Energy Error (dB) | HF Flux Error | SI-SDR dB |
|---|---:|---:|---:|---:|
| candidate | 0.725250 | 2.824839 | 1.080050 | 25.770295 |
| original | 0.745696 | 2.503481 | 1.119915 | 25.751111 |
| parent | 0.742269 | 2.876600 | 1.117913 | 25.760181 |
| v6 | 0.733257 | 2.782127 | 1.092556 | 25.770247 |

Pairwise differences (candidate minus V6 control), 95% CI:

| Metric | Mean Change | Interval |
|---|---:|---|
| enhanced_high_spectral_convergence | -0.008006 | [-0.033373, +0.007345] |
| enhanced_high_band_energy_error_db | +0.042712 | [-0.044855, +0.116463] |
| enhanced_high_flux_error | -0.012506 | [-0.051913, +0.010046] |
| enhanced_si_sdr_db | +0.000048 | [-0.006876, +0.006199] |

### Speech

| Path | HF Spectral Convergence | HF Energy Error (dB) | HF Flux Error | SI-SDR dB |
|---|---:|---:|---:|---:|
| candidate | 0.544562 | 3.222398 | 0.692287 | 30.016262 |
| original | 0.553484 | 2.934310 | 0.684675 | 30.009073 |
| parent | 0.538776 | 3.188172 | 0.694500 | 30.017263 |
| v6 | 0.547228 | 3.208142 | 0.691184 | 30.016087 |

Pairwise differences (candidate minus V6 control), 95% CI:

| Metric | Mean Change | Interval |
|---|---:|---|
| enhanced_high_spectral_convergence | -0.002666 | [-0.004364, -0.000989] |
| enhanced_high_band_energy_error_db | +0.014256 | [-0.024662, +0.061573] |
| enhanced_high_flux_error | +0.001103 | [-0.000763, +0.003256] |
| enhanced_si_sdr_db | +0.000176 | [-0.001304, +0.001940] |

## Continuity and Channel Check

All developed outputs and selected candidate held-out outputs show limited deviation; fixed 128-frame and irregular 31/512/997/7 block comparisons against offline processing show maximum differences of zero.
Two original stereo rock/electronic music tracks were used: full MP3 encode/decode and alignment performed first, then 12-second segments compared at two sample rates for stereo imaging. Ten silence/pulse/single-side input cases were individually verified.
Anonymous 24-bit WAV and scoring sheets are available at `data/reports/architecture-study-20261011/listening/index.html`. RMS matching with peak protection is used; not LUFS-based. Scoring sheets remain unfilled and cannot be used to claim perceptual approval.

## CPU Inference and Streaming Resource

M5 Max, CPU ORT single-thread. Resource measurements taken after all training and quality assessment processes complete.
Synchronous test: 4 models × 2 sample rates × mono/stereo × 3 rounds; each processing 10 seconds of real compressed music; model order interleaved.
Below are the 12-run ranges, across different channel/sample rate configurations; the extremes cannot be interpreted as paired performance ratios.

| Path | Synchronous Processing p99 ms Range | Max RSS MiB |
|---|---:|---:|
| control | 0.558–1.425 | 35.84 |
| polar-small | 0.699–1.438 | 37.88 |
| shape-small | 0.586–1.534 | 38.12 |
| shape-large | 0.710–1.501 | 44.16 |

Control and selected candidate each run four combinations × 60 seconds wall-clock pacing. Same 61-second real MP3 music PCM used, default warmup and scheduling request.

| Path | Software Added Latency (ms) | CPU % (single-core normalized) | Peak RSS MiB | callback p99 μs | callback max μs | worker miss |
|---|---:|---:|---:|---:|---:|---:|
| control | 5.341–5.814 | 29.53 | 36.20 | 2.167 | 9.709 | 0 |
| candidate | 5.342–5.817 | 37.97 | 48.55 | 2.708 | 12.042 | 0 |

CPU/RSS represents the entire four-case benchmark process, including ORT, DSP, tester, and report caching; CPU is the average across all four scenarios. Fixed 256-frame latency remains unchanged. Raw callback time arrays, miss bitmaps, and input/model/binary hashes are separately verified.
All eight paced cases passed the <8 ms target, <10 ms redline, callback and enhanced availability checks. Raw data re-computation results are consistent. These represent added software processing latency, excluding sound card, system, or host buffering.
This is a short-term simulated clock test; no physical sound card testing was performed, and results do not substitute for 30-minute Apple Silicon/x86_64 release validation. The unselected branch only passed synchronous performance screening.

## Result Judgment and Next Steps

**This round does not yield sufficient overall audio quality improvements to replace the baseline.** All three new branches learned non-zero corrections; CPU validation total decreased by 0.45%–0.59%, but no stable ranking of performance translates into actual musical quality improvement.
We cannot claim to have reached SOTA, nor can we treat this single-seed, 8-round continuation as a performance ceiling for any model structure.

- **Representation Comparison**: `shape-small` shows a 0.127 dB average reduction in music high-frequency energy error compared to `polar-small` with same parameter count; 95% CI is [-0.313, +0.055], still uncertain. Music SI-SDR decreased by 0.0254 dB (CI: [-0.0557, -0.0028]). Speech spectral convergence and energy improved, but SI-SDR slightly decreased. Circular encoding shows local value but is not yet proven superior in music.
- **Capacity Comparison**: `shape-large` vs. `shape-small` shows all music metrics—spectral convergence, energy, flux, and SI-SDR—show zero-crossing intervals. Speech energy error increased by 0.0459 dB (CI: [+0.0191, +0.0739]). With a 73.6% increase in parameters over `shape-small`, no clear benefit is observed. This result does not support further expansion.
- **Selected Candidate vs. V6**: On historical held-out music, four key metrics show zero-crossing confidence intervals; speech spectral convergence slightly improved. Compared to native epoch168, held-out music shows a 0.02045 drop in spectral convergence, 0.03986 drop in flux, and a 0.32136 dB increase in high-frequency energy error (CI: [+0.05322, +0.58591]). The baseline and experimental state are retained.
- **Compressed Input vs. Output**: Held-out music shows a 1.120 dB reduction in high-frequency energy error (CI: [-1.805, -0.438]), but a 0.20573 dB increase in flux error (CI: [+0.07539, +0.41170]) and a 1.176 dB drop in SI-SDR (CI: [-2.201, -0.432]). Speech spectral convergence and flux also degraded. Energy gain does not equal detail correction; a large isolated LSD gain cannot mask these issues.

Low-frequency diagnosis on 57 real mono/left/right held-out segments shows average absolute power gain error of 0.00339 dB, maximum 0.06088 dB; maximum low-frequency residual power ratio is -27.916 dB (V6: -28.453 dB); no full-scale overflow. This indicates small changes, not strict identity.
The low-frequency mask ends 500 Hz below the measured cutoff, computed from FFT256/hop128 and edge-cropped.

In four stereo segments, candidate's stereo imaging width error vs. lossless reference ranges from 0.02070 to 0.05961 dB; channel correlation error from 0.000955 to 0.002170; phase error from 0.01651 to 0.07954 rad. Width and correlation slightly underperform V6, though electronic phase error is slightly higher and only two tracks are covered.
Ten boundary cases show correct output length and finite bounds; silence and empty channels are strictly zero. These checks do not constitute blind listening validation.

During performance testing, the project's training and quality assessment has concluded, though the host still has background loads from games, browsers, and Spotlight. Environment snapshot is preserved.
Three rounds of synchronous testing with interleaved model order help observe variance, but are not a significant isolation experiment. Do not divide extreme values across different sample rates/channels to infer acceleration.

Next round will prioritize **small-scale causal temporal context + longer multi-resolution reconstruction supervision**, followed by independent study of learnable band-specific gain alignment with actual DSP. Past frames provide envelope, harmonic evolution, and transient cues; current branch expansion only increases static mapping, not observation. Time state requires training warmup, export protocol, per-channel reset, and frame drop recovery to be realized; still constrained by ≤2M parameters, <8 MiB, and real-time standards. This structure was not yet implemented in this round; future implementation and empirical validation will be reported in [Causal Context Experiment](causal-context-20261011.md). Results from this page are not included in its evaluation. Before further expansion, new independent recordings are needed, missing genre/region coverage must be supplemented, and blind listening must be conducted to avoid continued tuning on repeatedly rechecked development materials.

## Evidence Location

- `data/runs/architecture-study-20261011/experiments/`: configurations, training logs, best/final checkpoints.
- `data/reports/architecture-study-20261011/`: selection rules/hashes, CPU verification, parity, actual audio comparisons, cross-arm analysis, and raw resource records.
- CI: Run [38081774072](https://github.com/backrunner/soundex/actions/runs/38081774072), Rust and export-contract both succeeded; local Python passed 260 tests.
- Training audio, experimental checkpoints, ONNX, and reports remain in Git ignored directory.

The fixed 128-frame and irregular 31/512/997/7 block outputs both match offline
output with maximum difference 0. The separate stereo probe includes 10 silence,
impulse and one-sided-input cases. All 10 boundary cases preserve output length and
finite values; silence and empty channels are exactly zero. These are numerical
checks, not blind listening.
