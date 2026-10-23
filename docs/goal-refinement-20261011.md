# Goal Recheck: Energy, Phase, Structure, and Deployment

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

This round continues the optimization of high-frequency recovery with respect to an original lossless reference. The evaluation focuses on the final audio produced by the model after processing through Rust DSP. It is not sufficient to declare success based solely on training total, spectral coverage, or limited continuous output.

## Completed Diagnostics and Methodological Decisions

A set of 89 development validation rows and 80 actual recordings, each with Rust output, shows that the V4 default path has an average signed high-frequency energy difference of −3.659 dB. Among 33 musical rows, 21 are missing content, and all 56 speech rows show under-recovery; musical average: −2.086 dB, speech average: −4.585 dB. Some musical rows exhibit over-recovery, so a uniform gain increase cannot resolve this issue.

After disabling edge matching, the overall difference drops to −4.909 dB, which cannot be resolved by simply preserving the model's gain. These values are row-level averages; the high speech proportion in this diagnostic set does not imply a training sampling bias or actual usage distribution.

A new signed energy deviation metric is introduced: positive values indicate over-recovery, negative values indicate under-recovery. The original absolute energy error metric remains unchanged, and both are reported together to prevent cancellation of positive and negative deviations being misinterpreted as accurate recovery.

We also performed a DSP ablation on direct complex spectral interpolation. Compared to the original magnitude/phase fusion, the V4 high-frequency absolute energy error increases by 0.187 dB (95% CI: 0.095–0.282), low-frequency LSD retention improves by 0.033 dB (0.026–0.041), and high-frequency flux error decreases only by 0.0062. This approach was not adopted. The experimental patch and binary remain locally stored in the audit directory; no production DSP modifications were made.

## Should Temporal Structure Be Expanded Immediately?

Using 64 training musical recordings to fit a fixed ridge regression, we evaluated 26 development validation musical recordings, predicting high-frequency energy in four bands using 10 coarse-band energy features from current and past frames. All methods used the same 128 effective frame anchors/recording. Standardization used only training data and did not include future frames.

| History Frame Count | Validation MAE | Relative Single-Frame Difference (95% CI by Recording Cluster) |
|---|---:|---:|
| 1 | 9.416 dB | — |
| 4 | 9.391 dB | −0.089 to +0.042 dB                                          |
| 8 | 9.415 dB | −0.108 to +0.103 dB                                         |

This linear probe provides no clear benefit from increasing historical frame count. It does not test non-linear temporal networks, phase history, or actual audio restoration. Therefore, it cannot be used to reject all temporal structures. In this round, we proceed without altering the stream state contract, introducing circular phase inputs instead, to avoid expanding network scale, resetting logic, or increasing runtime cost without evidence.

Degradation and deployment divergence remain a significant risk. Research on filter mismatch in degraded audio (e.g., [Filter Mismatch Study](https://arxiv.org/abs/2011.07274)) shows that changes in degradation distribution can affect generalization. This round maintains the existing real codec synthesis recipe to isolate loss from structural change; no data from the paper or changes to the established music/voice sampling ratios were introduced.

## Controlled Experiments

### Data Coverage Boundaries

This round performs independent validation on recordings as specified in the manifest. Music is divided into training (253), validation (26), and held-out (21) tracks.
- Training: rock (49), electronic (48), classical (47), ambient (27), folk (22), pop (13), blues (7), jazz (6), country (5), hip hop (2), funk (1), untagged (26).
- No genre assumptions based on titles.
- Development validation excludes pop, country, hip hop, and funk; held-out excludes country, hip hop, and jazz, and contains no Chinese-origin works.
Thus, current acceptance does not cover all target genres or regions.

Quality statistics from source files show that 171 of the 253 training musical tracks have less than `1e-4` power above 16 kHz, and 44 have less than `1e-6`. This reflects only the full-spectrum energy distribution and cannot be used to infer lossy origin. It also cannot treat naturally low-frequency music as junk. This suggests that future work should examine the actual proportion of high-frequency detail preserved in training clips, and consider sampling active high-frequency segments from quiet samples while maintaining the user-specified music/voice ratio. No recipe or sampling changes were made to avoid confounding the controlled experiment.

### Configuration and Source

Both experiments start from the V4 base structure at best epoch4, with frozen BN statistics, no GAN, and reinitialized optimizer. Each group runs for 8 rounds / 1,024 updates, with a learning rate of 2e-5. The original 300 musical tracks and VCTK sources, licenses, and splits remain unchanged; training sampling is music 253/254, speech 1/254.

- `calibrated`: original structure, V5 reconstruction targeting high-frequency and subband energy and reference phase changes.
- `circular`: same V5 objective, with added sin/cos phase inputs at the first layer, adding 144 new parameters.

Both training sources use the same codebase: `db738344d033f4515ead4220c2e6403a72de2259`, with parent checkpoint SHA-256: `a152f00c6907221a4af80986364f621c1d884e9df0af22ba001ff419a574ccab`. Formulas, gradient audits, and migration methods are detailed in [Goal Specification](inpainting-objective.md).

A fixed 896 development validation rows are used for CPU FP32 V5 total comparison between the parent model and candidates, without direct comparison to historical V4 total. Actual development audio uses the same 89 rows, processed with identical Rust binary and default Neural/MatchEdge pipeline, reporting results by recording cluster, music/speech grouping, and block-level consistency.

## Pre-emptive Agreement for Held-Out Recheck

Prior to selecting any candidate, a freeze rule is applied: from training and development validation, select 21 music and 18 speech recordings that are track-, split_group-, and original file checksum-disjoint from each other. For each, two deterministic segments are selected, covering 44.1/48 kHz, and one training codec and one AAC160 held-out codec, totaling 78 rows.
A single candidate is selected based on the unified development V5 total and its model hash is frozen. Only after this step is the held-out scoring performed—no model is selected or re-tuned based on held-out results.

These 39 recordings were previously used in the baseline diagnosis, so this round is referred to as **held-out recheck**, not a new independent test. The left/right segment start points differ, so they cannot be stitched into synchronized stereo. Additionally, two original stereo works are fully encoded in MP3, aligned, and truncated to 12 seconds for comparison of imaging and block-level results. This supplementary set includes only two tracks (rock/electronic), two sample rates, and does not cover all joint-stereo codecs or perceptual qualities.

Complete local evidence is stored in `data/reports/goal-refinement-20261011/`. Raw data, experiment weights, and intermediate reports are not submitted to Git.

## Training and Deployment Results

Both groups complete 8 rounds. Unified CPU FP32 V5 total:
- Parent model: 1.258466
- Calibrated candidate: 1.255589 (−0.229%, best epoch4)
- Circular input candidate: 1.256967 (−0.119%, best epoch7)

Both groups pass 13 out of 13 PyTorch / ORT / Rust parity checks; 162 BN buffers match the parent model exactly, and all parameters are finite. The circular input adds new channels with non-zero learned weights. These checks do not constitute audio quality approval.

| Candidate | Parameters | ONNX Bytes | checkpoint SHA-256 | ONNX SHA-256 |
|---|---:|---:|---|---|
| calibrated | 806,276 | 3,309,786 | `79cf39af89f7887a2478e4ed1685c187cffc02b74327392fbfcae58ade3e0f1e` | `c4bb4b1b29b0575b893300fa0d57e7b0361e3116d57c82edd5294e0bc155bc83` |
| circular | 806,420 | 3,310,673 | `3f861a8adc1570dbe0cb946bfa576452038af8e171610453cce8f350c7485cfc` | `2b670a43398a78dcecac8f4e5959fa8fa14a31f30c071e1d402cdceac474a799` |

### Development Music: 33 rows, 26 recordings

All outputs are actual Rust Neural/MatchEdge results. Higher SI-SDR is better; lower error is better. "Original model" refers to the native epoch168 selected from 200 training rounds, not a compressed direct-through baseline.

| Metric | Compressed Direct | Original Model | V4 Parent Model | calibrated | circular |
|---|---:|---:|---:|---:|---:|
| High-frequency spectral convergence | 0.8395 | 0.9324 | 0.9094 | 0.9160 | 0.9225 |
| High-frequency absolute energy error (dB) | 5.4889 | 3.9562 | 3.8453 | 3.6485 | 3.4087 |
| High-frequency flux error | 1.0512 | 1.4460 | 1.4044 | 1.4236 | 1.4523 |
| SI-SDR dB | 27.5632 | 26.5771 | 26.5495 | 26.5210 | 26.5175 |
| Retained band LSD (dB) | 0 | 3.9196 | 4.1346 | 4.1946 | 4.1944 |

The calibrated candidate shows a 0.197 dB reduction in music energy error relative to V4 (95% CI: −0.347 to −0.058), but a 0.019 increase in flux (+0.007 to +0.035), and a 0.028 dB drop in SI-SDR (−0.048 to −0.012). Relative to the original model, its spectral convergence drops by 0.0165 (−0.0313 to −0.0038); the energy improvement crosses zero. The circular input further improves frequency recovery, but fails to synchronize improvements in spectral shape, transient fidelity, or waveform accuracy—thus cannot justify structural replacement.

Development speech: 56 rows / 54 recordings. The calibrated candidate achieves a spectral convergence of 0.6474, slightly below V4's 0.6517; energy error of 4.3683 dB, lower than V4's 4.5854 dB; SI-SDR of 29.2513 dB, slightly below V4's 29.2650 dB. The music-to-speech ratio in this diagnostic set does not imply proportional training or usage benefit.

### Error Scale and Deployment Discrepancy Recheck

Among 33 musical rows, 24 have low high-frequency power in the evaluation band below 1e-4 of total frequency. Per-row metrics are amplified by these extremely weak bands. Using the 1% full-band norm from training as a baseline, we perform a **post-hoc diagnosis**: the calibrated candidate shows a −0.0332 difference in high-frequency spectral convergence relative to direct-through (CI: −0.0816 to +0.0114), which does not prove overall improvement or degradation; flux error remains at +0.0380 (+0.0113 to +0.0659).
No changes were made to original metrics, candidate selection rules, or release thresholds—this diagnosis does not prove perceptual insignificance.

In the same dataset, the mean difference between online bandwidth and paired static cutoff is approximately −804 Hz; the worst-case relative spectral error case has very low original high-frequency power. These suggest that natural narrowband content, detector behavior, and training masking differences should be further studied. However, mean comparisons do not prove detector-induced errors, and a blanket bypass cannot be justified as "improvement."

### Held-Out Recheck: 78 rows, 39 recordings

2026-10-11 02:25:29 (Singapore) — the `calibrated` hash is frozen according to the V5 total rule, and only compared against the original model and V4 parent model. No selection of `circular` was made based on the following results.

| Held-Out Music Metrics (42 rows / 21 tracks) | Original Model | V4 Parent Model | calibrated |
|---|---:|---:|---:|
| High-frequency spectral convergence | 0.7457 | 0.7423 | 0.7617 |
| High-frequency energy error (dB) | 2.5035 | 2.8766 | 2.8433 |
| High-frequency flux error | 1.1199 | 1.1179 | 1.1417 |
| SI-SDR dB | 25.7511 | 25.7602 | 25.7402 |

The energy improvement observed in the development set does not generalize stably to held-out music: relative to V4, the energy difference is −0.033 dB (CI: −0.145 to +0.084), with spectral convergence and flux deteriorating by +0.0194 and +0.0238, both above zero. The music energy error relative to the original model increases by 0.340 dB (+0.082 to +0.600).
Held-out speech: 36 rows / 18 tracks — spectral convergence improves to 0.5315 (V4: 0.5388), energy error drops to 3.0137 dB (V4: 3.1882). This indicates performance variation across datasets, and cannot generalize speech gains to music performance.

Relative to the compressed direct-through, held-out music energy error reduces by 1.102 dB (CI: −1.862 to −0.281), but flux increases by 0.267 (+0.097 to +0.534), and SI-SDR drops by 1.206 dB (−2.259 to −0.442). This remains a result of energy restoration, not full detail recovery. A significant reduction in high-frequency LSD cannot be declared as success.

For 57 physical mono/left/right rows, the average low-frequency absolute power change is only 0.00339 dB, and residual power relative to input low-frequency average is −56.92 dB. There is no new or retained full-scale overflow. Retained band LSD values cannot be directly interpreted as actual low-frequency loudness changes. This round does not use them to determine severe low-frequency distortion.
The calibrated candidate's 89 development rows and 78 held-out rows, and the circular input's 89 development rows, show a maximum difference of zero across 128 frames and irregular blockings, with no non-finite outputs.

## Stereo, Continuity, and Resource Performance

Two original stereo rock/electronic works × 44.1/48 kHz, resulting in 4 synchronized MP3 scenarios:
- The calibrated candidate shows zero block difference, finite output, and peak value of 0.8773.
- Imaging width error average: 0.04114 dB (original model: 0.03605, V4: 0.03992), and channel phase error: 0.04772 rad (original model: 0.04758, V4: 0.04773). These differences are insufficient to claim imaging or perceptual improvement.

Additionally, 10 quiet, pulse, and mono-input scenarios are tested. Output frame count is correct, and no non-finite values occur. Silence output and the absent channel are strictly zero. These checks do not replace actual device testing for pop/clip or listening quality.

Local `listening/` has generated 20 anonymous 24-bit WAV files (4 segments × 5 conditions), including original lossless, compressed direct, original model, V4, and selected candidate. Each condition first matches RMS to reference, then jointly attenuated to avoid clipping; not LUFS-matched. The scoring sheet has not been filled, and no automatic claim of blind listening pass is made.

After all training and quality scoring, on Apple M5 Max, the same release binary, true PCM, and ORT are used with single-threaded measurement. Each scene runs 10 seconds with 32 hops pre-warmup; four sample rate/channel combinations:

| Model | Full-synchronized processing p99 range | Synchronous process peak RSS |
|---|---:|---:|
| V4 parent model | 0.933–1.843 ms | 36.05 MiB |
| calibrated | 0.973–1.677 ms | 35.88 MiB |
| circular | 0.746–1.642 ms | 35.72 MiB |

This is a single sequential measurement; the fluctuation does not prove that adding sin/cos features reduces latency. Synchronous processing includes full hop analysis/inference/DSP, not isolated ORT operations, and should not run within audio callback.

For real-time adapter, each candidate is run over four scenarios, 30 seconds per, with work thread pre-warmup and macOS native real-time scheduling requests; true PCM hash is fixed. Results:

| Candidate | Fixed latency + max callback time | Callback p99 max | Real-time process peak RSS | Total CPU per core |
|---|---:|---:|---:|---:|
| calibrated | 5.340–5.813 ms | 2.625 µs | 38.53 MiB | 30.45% |
| circular | 5.341–5.817 ms | 2.959 µs | 36.28 MiB | 31.43% |

All eight scenarios achieve zero worker deadline misses and zero queue overflow. Output length, finite value, and clipping checks pass. Individual recalculations of raw callback times and miss bitmaps confirm p99, longest continuous miss, availability, and latency statistics; both 8 ms target and 10 ms redline are met. Audio buffer remains fixed at 256 frames, with no added lookahead. CPU is the total test process `(user+sys)/wall`, including test framework and startup; different RSS values do not prove that circular input is more memory-efficient. Environment, binary, audio, and model hashes are preserved.

These are local short-term audio callback simulation results; they are not end-to-end physical card delays, nor 30-minute device continuous validation. Rust/Python CI passes are confirmed; Python: 248 passed / 1 skipped. Numerical, engineering continuity, and subjective recovery quality remain separately recorded.

## Decision and Next Steps

**No overall recovery goal has been achieved, and no comprehensive superiority over the original model or compressed direct-through has been proven.** This round does not release or replace the baseline. The V5 and circular input remain available as reproducible experimental options. The best point for the original structure occurs at epoch 4; for circular input, at epoch 7. Eight rounds under a fixed budget only demonstrate performance in this round and do not prove that longer training is without value. However, there is no evidence supporting continued blind expansion of training rounds under the same objective.

Future training priorities are derived from the counterexamples of this round:

1. **Reduce the gap between training reconstruction proxy and real-time DSP**, especially in online cutoff frequency and edge-level matching. Begin with per-example differentiable proxy/execution comparison, then adjust the target. Natural narrowband, weak high-frequency, and high-bitrate inputs should be preserved as counterexamples—do not replace recovery with a blanket reduction in enhancement magnitude.

2. **Improve correct transient and spectral shape fidelity under retained band and waveform differences**, and separately report both frequency recovery and error details. For extremely weak references, report absolute scale, not relative metrics or arbitrary new thresholds, as a basis for perceptual judgment.

3. **Expand validation with rare genres and previously unrepresented regions**; verify the effective high-frequency information density in clipped segments. Investigate sampling active segments from quiet or naturally narrowband samples. Maintain the existing speech proportion and source licensing path.

4. **If still constrained, explore non-linear, past-frame-based temporal or phase structures**. Linear energy probes are insufficient to rule out their value. Such a structure must be implemented with full state, reset, export, rollback, and real-time cost validation—cannot be replaced simply because training loss decreased.

These represent the direction of future experiments. This round did not achieve or validate any new stateful temporal network, modify official thresholds, or obtain blind listening or 30-minute physical device validation results.

Of 33 music evaluation rows, 24 have lossless reference power in the evaluated high band below `1e-4` of full-band power. Weak-band relative errors require this context. Development irregular-block output differences are 0; the separate synchronized stereo calibrated-path block differences are also 0. Neither equality replaces listening.
