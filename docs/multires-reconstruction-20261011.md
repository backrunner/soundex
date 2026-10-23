# Continuous Reconstruction and Multi-Resolution Supervision Experiment

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

The input-conditioned gain calibration has been completed: direct acquisition of production analyzer states, verifying the effectiveness of bounded gain targeting on final output.

Following the addition of causal history in the prior round, improvements in high-frequency energy and transient response remain unimpressive. In this round, we introduce continuous reconstruction supervision and gain state initialization on the training side, without altering the deployment structure, FFT256/hop128, nine-frame causal input, or the fixed 256-sample output time axis. Two groups have each completed four rounds; overall audio quality still shows trade-offs, and no existing weights are replaced.

## Conclusions and Next Steps

The multi-resolution group achieves a unified CPU total of 1.679951, which is 0.35% lower than the long-sequence baseline but 1.21% higher than the parent model before this continuation. In actual Rust output, music transient error relative to the baseline decreases by 0.02054 (95% CI does not cross zero), but high-frequency energy errors for music and speech increase by 0.12760 dB and 0.07904 dB, respectively (both CIs do not cross zero). Speech energy improves by 0.51069 dB relative to the parent model, while SI-SDR slightly decreases by 0.01069 dB. These results do not justify a claim of overall superiority over the baseline.

This round has implemented long-sequence, real-time DSP warmup, and multi-resolution supervision, yet no balanced recovery benefits have been achieved. The sensitivity to nine-frame history remains low; a short training session with a single seed does not invalidate the importance of temporal structure. Next, we prioritize calibrating the cutoff and gain state between training and deployment. Fixing MatchEdge at a constant value would suppress part of the model's amplitude learning. Discrepancies between the list cutoff and Rust online analysis remain. We should retain low-frequency and stereo constraints in bounded, input-conditioned high-frequency calibration, conduct item-by-item comparisons, and only then decide whether to expand or perform long training. Simply disabling gain is not a validated solution. These subsequent strategies remain unimplemented and cannot be considered gains from this round.

## Modifications and Boundaries

- V7 extends observation from 8-frame model history to 32-frame real DSP history, then supervises 32-frame output. MatchEdge starts from the Rust reset value of 1 and is propagated recursively; historical predictions are detached and not used for target error training, and BN statistics are frozen. This avoids approximating the entire segment's initial state with the first frame's instantaneous target gain.
- The 32-frame DSP warmup provides only limited history; under non-silent, stable conditions, the initial state influence decays at 0.9^32, still not equivalent to the full runtime state. The cutoff used here does not include online change detection, gating fade-in, or limiter.
- Full OLA support window of 3968 samples (32 frames × 124 samples) is used for supervision. A 512/1024/2048-point Hann analysis is applied. New metrics include high-frequency spectral convergence, per-window energy, signed time variation relative to reference, and preserved band complex constraints. High-frequency floor and normalization depend solely on the reference; excess energy and smoothing of true transients are penalized.
- These long windows are used only for loss. The model remains a finite causal refiner with 923,526 parameters, with no added lookahead. Whether it meets performance and continuity requirements will be judged by actual exported waveform measurements.

## Fixed Protocol

From the same `shape-small` parent checkpoint `a5166a6f56239a31a48986725188aa19c1c199c565904c76aca0f160a19f393a`, a zero-initialized time branch is migrated. Native, lossless data, splits, music-to-voice ratios, and source obligations remain unchanged. Training samples: 253/254 music, 1/254 speech; fixed validation set of 896 rows (768 music + 128 speech). Data recipe hash is `7fda9e58ab583617f3514f9dd9cfcf87f29871ae0587e91ca29ac0ddaf6b29d4`, and list set hash is `a14019181a6303417a72285b45f4e28988b349ab9d164f39b64b82f44150dab4`. Both arms freeze BN, use no GAN, batch size 16, 2048 draws per round, 4 rounds / 512 updates, LR 2e-5, 64-step warmup/cosine, total 262,144 supervised frames. Both groups observe identical-length sequences and use the same sampling seed.

- `long-control`: long sequence and DSP history; multires weights set to zero.
- `multires`: on top of the above, uses 0.5 spectral convergence, 1.0 energy (scaled by 20 dB), 0.25 temporal error, and 1.0 preserved band complex error. Original V6 weights remain unchanged.

Only the pre-defined epoch 4/final checkpoint is used; best epoch is not selected based on individual training objectives. Both groups and the parent model use the same complete V7 CPU FP32 target, fixed 896 validation rows, and cropped re-evaluation. Before scoring, hash binding is applied, followed by actual Rust development set audio quality comparison. This round is a development set experiment; reused materials are not considered independent tests. Low loss, export values, actual outputs, real-time performance, and listening quality are all verified, and no default weights are automatically replaced.

## In-House Outputs

- `data/runs/multires-reconstruction-20261011/`: frozen source code, recipe, training logs, and checkpoints.
- `data/reports/multires-reconstruction-20261011/`: real audio MPS gradient checks, numerical values, audio quality, and resource reports.
- No new data sources, external code/teacher, or weights are introduced. Audio and experimental outputs remain in the ignored directory.

## Training and Unified CPU Recheck

Training source code: `a27550acc142c7a91441fd1068bd6056df38cd7f`. Both groups complete 4 rounds / 512 updates, using the pre-defined final checkpoint, without selecting best epoch based on individual training objectives. Data, validation rows, and final random/sampler states are consistent; no additional record of crop position is kept. All saved tensors are finite, with 162 BN buffers matching the parent model bit-for-bit.

| Model | Unified Complete V7 Total | Parameters | ONNX MiB |
|---|---:|---:|---:|
| parent | 1.65981154 | 905,734 | 3.550 |
| long-control | 1.68588339 | 923,526 | 3.622 |
| multires | 1.67995108 | 923,526 | 3.622 |

According to the pre-defined common CPU rule, `multires` is selected. Quality and numerical qualifications are still evaluated separately. The previous V6 total cannot be directly compared to this table.

| Model | checkpoint SHA-256 | ONNX SHA-256 |
|---|---|---|
| long-control | `95b1d8efd32dd8bed61f4ad0d36a1be01aef66842d4d25da8de12f40aed6a09c` | `8728b0bb16e7dddeac350207eb495406efbf24a0c6ccdd5e1da53c22df8439c1` |
| multires | `23ae3b355bdd246d42c77f7c7eef29e81844d8a9c9c323e712cec57cd2a118d8` | `dcff2cf6320ccda229aac1d494c1343b9c7ff9f16c18ca05c3901a69ff6ef121` |

### Raw Predictions and Reconstruction Errors

All metrics are unweighted; units differ and cannot be directly summed. Reconstruction is a training proxy and does not replace Rust results.

| Metric | parent | long-control | multires |
|---|---:|---:|---:|
| high_band_magnitude | 5.471625 | 5.537713 | 5.520186 |
| high_band_energy | 1.588409 | 1.518020 | 1.515842 |
| high_band_spectral_convergence | 0.269687 | 0.275008 | 0.274148 |
| high_band_temporal | 0.269243 | 0.275634 | 0.274474 |
| reconstruction_high_energy | 1.760752 | 1.765453 | 1.780173 |
| reconstruction_temporal | 0.295483 | 0.302373 | 0.301002 |
| multires_high_spectral | 0.314071 | 0.325744 | 0.322020 |
| multires_high_energy | 0.874968 | 0.867459 | 0.874309 |
| multires_high_temporal | 0.251973 | 0.258892 | 0.256156 |
| multires_low_complex | 0.000905 | 0.000958 | 0.000932 |

## Actual Rust Development Set Results

89 rows / 80 segments of recordings, including 33 rows / 26 music segments and 56 rows / 54 speech segments; Neural/MatchEdge. Same material and processing parameters, paired with 95% CI bootstrapped across 2000 recording clusters. No multiple comparison correction was performed. This is reused development material, not an independent test. Music lacks coverage in pop, country, hip-hop, and funk. Higher-frequency error is better, higher SI-SDR is better.

### Music

| Path | HF Spectral Convergence | HF Energy Error dB | HF flux | SI-SDR dB |
|---|---:|---:|---:|---:|
| original | 0.932430 | 3.956168 | 1.445967 | 26.577054 |
| parent | 0.896469 | 3.916874 | 1.391960 | 26.597193 |
| previous-causal | 0.900147 | 3.926692 | 1.400837 | 26.618250 |
| long-control | 0.936649 | 3.754156 | 1.470854 | 26.616312 |
| multires | 0.933971 | 3.881757 | 1.450317 | 26.648028 |

Difference: multires minus long-control

| Metric | Change | 95% CI |
|---|---:|---|
| enhanced_high_spectral_convergence | -0.002678 | [-0.007331, +0.001780] |
| enhanced_high_band_energy_error_db | +0.127601 | [+0.002676, +0.269719] |
| enhanced_high_flux_error | -0.020537 | [-0.034334, -0.006820] |
| enhanced_si_sdr_db | +0.031716 | [+0.006088, +0.065342] |

### Speech

| Path | HF Spectral Convergence | HF Energy Error dB | HF flux | SI-SDR dB |
|---|---:|---:|---:|---:|
| original | 0.664968 | 4.218196 | 0.809657 | 29.253909 |
| parent | 0.654183 | 4.641319 | 0.808059 | 29.270062 |
| previous-causal | 0.654100 | 4.554872 | 0.807965 | 29.267906 |
| long-control | 0.646320 | 4.051588 | 0.809396 | 29.255910 |
| multires | 0.649821 | 4.130626 | 0.810042 | 29.259369 |

Difference: multires minus long-control

| Metric | Change | 95% CI |
|---|---:|---|
| enhanced_high_spectral_convergence | +0.003501 | [+0.002281, +0.004713] |
| enhanced_high_band_energy_error_db | +0.079039 | [+0.033859, +0.126880] |
| enhanced_high_flux_error | +0.000646 | [-0.000759, +0.002077] |
| enhanced_si_sdr_db | +0.003459 | [-0.002030, +0.008905] |

## Comparison After Removing Startup Segment

Process full stream, then uniformly crop first 0.25 seconds and last 256 samples. Rule is fixed before Rust scoring and does not influence candidate selection. Below remains the same development set.

| Corpus | Metric | multires minus long-control | 95% CI |
|---|---|---:|---|
| Music | enhanced_high_spectral_convergence | -0.000286 | [-0.004643, +0.003931] |
| Music | enhanced_high_band_energy_error_db | +0.160368 | [+0.022067, +0.312735] |
| Music | enhanced_high_flux_error | -0.009124 | [-0.016487, -0.002121] |
| Music | enhanced_si_sdr_db | +0.036565 | [+0.006286, +0.080934] |
| Speech | enhanced_high_spectral_convergence | +0.003691 | [+0.002090, +0.005243] |
| Speech | enhanced_high_band_energy_error_db | +0.136789 | [+0.077236, +0.202041] |
| Speech | enhanced_high_flux_error | -0.000521 | [-0.001923, +0.000782] |
| Speech | enhanced_si_sdr_db | +0.003472 | [-0.002150, +0.008841] |

## Historical and DSP Diagnostics

Replace all nine observation positions with the current frame, fix the model, and measure output variation across 512 real validation windows. This is an out-of-distribution sensitivity probe, not a post-hoc ablation or audio quality metric. Maximum output variation is used to supplement overall loss diagnosis, avoiding cancellation of positive and negative errors in the mean.

| Model | Maximum Change dB | Maximum Phase Change rad | Maximum Complex Relative RMS |
|---|---:|---:|---:|
| long-control | 0.0012898 | 0.0000486 | 0.0001132 |
| multires | 0.0012830 | 0.0000467 | 0.0001155 |

Fixed MatchEdge exhibits a predictable scaling limit: when the target gain has not reached clamp and reaches steady state, the generated magnitude multiplied by the matched gain leads to high-frequency RMS approaching a fixed −6 dB ratio relative to low-frequency edge RMS. The overall magnitude scaling is then canceled by the gain. In constant-spectrum probes, correct high-frequency magnitude is 0.8; when predicted magnitude is scaled to 90%/100%/110%, the 32-frame warmup output mean is 0.5032/0.5040/0.5047. This is a mechanism-level diagnosis and cannot be used as evidence of audible artifacts or as justification for disabling gain. V7 does not eliminate this limitation.

| Corpus | Rust Analysis Mean Cutoff vs. List Hz | Analysis Enhanced Frame Ratio |
|---|---:|---:|
| Music | -1484.23 | 0.9854 |
| Speech | -1534.94 | 0.6418 |

This table comes from full-spectrum mean of the same configuration Rust analyzer; it is not a per-hop trajectory. It suggests geometric bias in the proxy and cannot be used to prove frame-level cutoff consistency.

## Numerical, Continuity, and Listening

| Model | 13-item fixed parity | Real 512-frame failure batch / 256 | Deployment Numerical Qualification |
|---|---|---:|---|
| long-control | True | 6 | False |
| multires | True | 4 | False |

No relaxation of v2 numerical budget. Both groups pass 13 fixed export checks, but real-frame audits still exceed limits and remain as local diagnostic plots. Actual audio quality and resource metrics do not equate to deployment eligibility. Numerical over-budget does not automatically imply audible artifacts.

| Model | Number of Playback Rows | Maximum Irregular Chunk Error | Non-finite Output |
|---|---:|---:|---:|
| long-control | 89 | 0.000000000 | 0 |
| multires | 89 | 0.000000000 | 0 |

Additional testing on two validation-originated stereo rock/electronic original tracks, fully MP3-encoded and aligned, 12-second segments extracted at 44.1/48 kHz; multires also tested on 10 cases of silence, pulse, and mono-channel. Listening report available at `data/reports/multires-reconstruction-20261011/listening/index.html`. Five conditions: lossless reference, compressed input, parent model, long-sequence baseline, and multires. Full RMS matching, peak protection, 24-bit WAV; not LUFS-matched. Scoring sheet is empty, `listening_performed: false`.

## Real-Time Resource

M5 Max, CPU ORT single-thread; measured after training, quality, and history probing, with machine background load snapshot preserved. The machine is not idle; the startup snapshot includes indexing and other application loads; no unrelated user processes were terminated. Each group performs 2 sample rates × mono/stereo × 3 rounds, interleaved hop-by-hop processing, 10 seconds of real music per case.

| Model | Per-hop processing p99 ms range | Synchronous RSS MiB peak |
|---|---:|---:|
| long-control | 0.578–1.592 | 38.14 |
| multires | 0.571–1.802 | 38.16 |

Each group also performs 4 cases × 60 seconds wall-clock pacing, pre-warmup, 44.1/48 kHz, mono/stereo, using the same real PCM.

| Model | Software Added Latency ms | CPU% per Core (normalized) | RSS MiB peak | callback p99 μs | worker miss | Longest Miss Segment ms |
|---|---:|---:|---:|---:|---:|---:|
| long-control | 5.343–5.819 | 42.52 | 38.52 | 4.292 | 0 | 0.000 |
| multires | 5.340–5.814 | 38.92 | 38.55 | 3.292 | 0 | 0.000 |

CPU/RSS represents the entire 4-case process, including inference, DSP, and report caching. Raw callback arrays, miss bitmaps, input/model/binary hashes, and budget formulas are separately verified. The fixed 256-sample software latency does not include device, driver, or host buffering. Availability does not imply every hop is enhanced; silence/bypass/history padding can still produce aligned dry outputs. This is a short-term simulation test and does not include real audio cards, 30-minute runtime, or x86_64 real-world performance validation.

All eight real-time cases meet 8ms target, 10ms red line, callback, and enhancement availability budget. The multi-resolution group has a maximum callback latency of 9.709 μs; raw records have been verified.

## Code Verification

271 Python tests passed; real audio MPS three-step gradient checks all yield finite values. The Rust and export-contract components of the training source code passed CI at [GitHub Actions Run #38122345171](https://github.com/backrunner/soundex/actions/runs/38122345171). Original audio, checkpoints, and diagnostic ONNX are not pushed. Source code and result documentation are submitted under BackRunner identity.

## Comparison with Compressed Input and Numerical Error Magnitudes

Development set mean under lossless reference; lower HF error is better, higher SI-SDR is better.

| Corpus | Path | HF Spectral Convergence | HF Energy Error dB | HF flux | SI-SDR dB |
|---|---|---:|---:|---:|---:|
| Music | Compressed Input | 0.839516 | 5.488903 | 1.051229 | 27.563202 |
| Music | multires | 0.933971 | 3.881757 | 1.450317 | 26.648028 |
| Speech | Compressed Input | 0.723740 | 5.673287 | 0.804808 | 29.381521 |
| Speech | multires | 0.649821 | 4.130626 | 0.810042 | 29.259369 |

Music shows reduced high-frequency energy loss, but spectral convergence, transient, and overall SI-SDR performance remain inferior to the compressed input path. Thus, energy gains cannot be considered a recovery of fine details. Low-frequency LSD relative to reference decreases, though the low-frequency preservation LSD relative to input remains at 3.9829 dB (music) and 3.9804 dB (speech); this logarithmic metric is sensitive to weak frequency points and does not equate to a 4 dB change in low-frequency loudness. More complete band preservation and subjective validation are still required.

| Model | Maximum Absolute Difference dB | Maximum Phase Difference rad | Maximum Complex Relative RMS | Maximum Inverse FFT Difference |
|---|---:|---:|---:|---:|
| long-control | 0.00214481 | 0.00366241 | 0.00031431 | 0.00005560 |
| multires | 0.00102997 | 0.00337374 | 0.00039626 | 0.00002042 |

Both groups have actual inverse FFT errors below the existing 6.1035e-5 budget, though feature-level violations remain. The multi-resolution group's maximum complex relative RMS is approximately 3.96e-4 (budget 2e-4). This result operates within existing, explainable tolerances and does not imply audible artifacts. This round's independent quality trade-offs mean no numerical policy changes were made for candidate release. If budget adjustments are made, they should be based on FP64, end-to-end waveform, and cross-platform results outside the candidate set.

## Stereo and Boundary Details

The following covers only two synchronized stereo validation recordings and does not generalize to full catalog.

| Genre | Sample Rate Hz | Stereo Width Error dB | Channel Correlation Error | Channel Phase Error rad |
|---|---:|---:|---:|---:|
| rock | 44100 | 0.0057476 | 0.0005662 | 0.0075230 |
| rock | 48000 | 0.0004083 | 0.0000355 | 0.0075128 |
| electronic | 44100 | 0.0022256 | 0.0001716 | 0.0045248 |
| electronic | 48000 | 0.0019247 | 0.0001625 | 0.0049484 |

These stereo width errors are slightly higher than the corresponding compressed input, indicating no improvement in stereo imaging. All four multires stereo cases show irregular block error 0. All ten boundary cases have consistent input/output frame counts and are finite. Silence and mono inputs maintain zero in silent channels. Pulse peaks are approximately 0.8594, with no full-amplitude overflow. These checks do not replace real audio card or subjective listening validation.

The subsequent [input-conditioned gain calibration](gain-calibration-20261011.md) is a separate 20261011 study and does not retroactively change this experiment. Fixed validation has 768 music +128 speech rows. The four stereo probes have irregular-block difference 0, and the 10 boundary cases retain finite correctly sized output with silent channels at 0. There is no added lookahead: 0 future frames are read, and buffering is still reported separately. These finite probes do not certify 0 deadline misses for every host or replace the historical <10 ms software limit and later model-plus-DSP <5 ms requirement.
