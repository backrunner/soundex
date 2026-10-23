# V6: Deployment-Aware Gain and Transient Reconstruction with Controlled Continuation

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

This round addresses the discrepancies between the training proxy and actual DSP behavior exposed in the [V5 recheck](goal-refinement-20261011.md), by incorporating Neural/MatchEdge gain into the reconstruction loss and adding a transient constraint based on a lossless reference. The base model and Rust algorithm remain unchanged; this round's candidate is derived from local experimental weights. Eight training rounds, export, real-time stream processing, ablation, and one-minute live testing have all been completed.

## Experimental Constraints

- **Frozen source code**: `e6946b2e037cce8afb81bc3653623f4b14409def`.
- **V4 parent checkpoint SHA-256**: `a152f00c6907221a4af80986364f621c1d884e9df0af22ba001ff419a574ccab`.
- **8 rounds**, each with 2,048 samples, batch size 16, totaling 1,024 updates; learning rate 2e-5, 128-step warmup, cosine decay, gradient clipping 1.
- **Only generator is warmed up**; optimizer is reinitialized, no GAN, 162 BatchNorm state buffers are frozen.
- **Dual-stream architecture on raw frequency axis**, 806,276 parameters, FFT 256 / hop 128; no additional model state or future frame prediction.
- Model calls during training and validation remain frame-by-frame independent; only 8 consecutive frames are used to construct the reconstruction loss.
- Native, lossless 300-song music and VCTK datasets are preserved, with a training split ratio of 253/254 music to 254/254 speech (1/254 speech). The template ratio is illustrative; actual runtime uses configuration parsed from the parent checkpoint, with identical canonical data hash.
- Recipe SHA: `7fda9e58ab583617f3514f9dd9cfcf87f29871ae0587e91ca29ac0ddaf6b29d4`;
  Manifest set SHA: `a14019181a6303417a72285b45f4e28988b349ab9d164f39b64b82f44150dab4`.
- Fixed 896 development validation samples (768 music, 128 speech), with V6 selecting the lowest epoch as candidate.
  After export, the model hash is frozen before undergoing a held-out recheck on 78 tracks (39 segments).
  These recordings were previously used for diagnosis and are not new independent tests; there remain gaps in genre and regional coverage.
- All original training data, checkpoints, ONNX outputs, audio results, and full reports are retained in the ignore directory.

## Loss and DSP Boundary

[See V6 Formula and Implementation Boundary](inpainting-objective.md#v6-deployment-aware-continuation-experimental)
Retains V5's original magnitude, energy, phase, and change error; the reconstruction path additionally uses Rust's identical cutoff frequency binning, edge reference RMS, −6 dB target ratio, [0.25, 2] gain target, and 0.1 smooth recursive update.
Only frequency bins with crossover weight greater than zero are subject to gain, followed by polar coordinate fusion, causal OLA, and re-analysis.
A new weight of 0.25 is introduced for time difference error, while supervising both attack and decay, without rewarding constant spectra.

Since short training segments lack prior gain history, the first-frame target gain is used to initialize the detached value—this is a steady-state approximation and cannot be considered equivalent to full deployment. The cutoff frequency remains from static pairing metadata; online detector, gate, limiter, wet/dry ramp, and asynchronous fallback are still validated against actual Rust output.
The V6 identity generator score is also processed through the gain proxy, not a dry baseline.
No direct comparison between V5 and V6 total scores is made.

Among 64 training samples, the L2 mean of the new reconstruction time term gradient (weighted by parameter) is 4.7030; spectral convergence is 3.3550, raw magnitude is 1.7504, and phase is 2.4490—all bounded.
This check is used to detect significant imbalance, not to prove optimal weight configuration.

## Ideal Input Diagnostics: DSP Also Has Error Sources

On the same 64 training samples, we replace predicted magnitude and phase with lossless reference values using CPU FP32. This is an undeployable diagnostic input, evaluated using the V6 proxy, not a test score or achievable upper bound.
All following metrics are lower is better, and comparisons are only made within the same metric definition, not across versions:

| Input Condition | High-Frequency Spectral Convergence | High-Frequency Complex Error | High-Frequency Energy Error (dB) | Time-Difference Error |
|---|---:|---:|---:|---:|
| V4 Prediction + MatchEdge Proxy | 0.246946 | 0.340057 | 1.396734 | 0.242808 |
| Correct Magnitude + Predicted Phase + MatchEdge | 0.203532 | 0.352519 | 1.197768 | 0.194629 |
| Predicted Magnitude + Correct Phase + MatchEdge | 0.236687 | 0.241924 | 1.398647 | 0.208724 |
| Correct Magnitude + Correct Phase + MatchEdge | 0.182366 | 0.186335 | 1.205929 | 0.135519 |
| Correct Magnitude + Correct Phase, no edge gain | 0.050710 | 0.060730 | 0.118810 | 0.045789 |

It is evident that correct magnitude cannot fully compensate for complex errors introduced by predicted phase; a fixed edge energy ratio still deviates from the correct reference. The last row retains errors in retained frequency bands and crossover regions.
For existing V4 predictions, removing edge gain increases the high-frequency energy error from 1.3967 dB to 1.5338 dB, so it cannot be inferred that the gain control should be disabled. This supports further research into adaptive gain, and it cannot replace real-time stream processing validation.

## Training and Deployment Results

All 8 rounds completed; the 4th round is selected based on predefined rules. The following uses the same 896 samples, same V6 objective, and CPU FP32 single-threaded re-computation:

| Metric | V4 Parent Model | V6 Candidate |
|---|---:|---:|
| V6 total | 1.351980 | 1.348875 |
| Raw High-Frequency Magnitude Error (dB) | 5.485374 | 5.475147 |
| Raw High-Frequency Time Difference | 0.249465 | 0.247553 |
| Reconstructed High-Frequency Spectral Convergence | 0.276458 | 0.275842 |
| Reconstructed High-Frequency Complex Error | 0.392754 | 0.383451 |
| Reconstructed Low-Frequency Complex Error | 0.001027 | 0.000985 |
| Reconstructed High-Frequency Energy Error (dB) | 1.702155 | 1.712001 |
| Reconstructed Time Difference | 0.279467 | 0.276602 |
| Relative Waveform Error | 0.916551 | 0.915600 |
| Spectral Consistency | 0.188507 | 0.193432 |

Total score decreased by approximately 0.230%; improvements are concentrated in complex reconstruction, low-frequency residuals, and transients, while high-frequency energy and spectral consistency show slight degradation.
This small drop in total score cannot be interpreted as overall audio quality meeting standards.

Candidate checkpoint SHA-256: `22c39a78baca386c2b30d34641f02dea7cf83a331f00eefd8390d769f0a8a433`.
ONNX SHA-256: `18ea464cfcc4c978b252f1b6ad04ce5e1fc427dc225b58759b6a8e69d2b57336`.
13 out of 13 PyTorch/ORT/Rust parity checks passed; all 162 BatchNorm buffers match the parent model bit-for-bit, and all parameters are finite.

### Fixed Development Audio: 89 Tracks, 80 Segments

Actual Rust Neural/MatchEdge; 33 music tracks / 26 segments, 56 speech tracks / 54 segments.
The first three metrics are lower is better, higher is better for SI-SDR. Dry is the unenhanced compressed input.

**Music**

| Path | High-Frequency Spectral Convergence | High-Frequency Energy Error (dB) | High-Frequency Flux Error | SI-SDR dB |
|---|---:|---:|---:|---:|
| Compressed Input | 0.839516 | 5.488903 | 1.051229 | 27.563202 |
| Raw Epoch168 | 0.932430 | 3.956168 | 1.445967 | 26.577054 |
| V4 | 0.909426 | 3.845271 | 1.404426 | 26.549455 |
| V5 calibrated | 0.915956 | 3.648472 | 1.423574 | 26.521046 |
| V6 | 0.904906 | 3.895419 | 1.405710 | 26.598670 |

**Speech**

| Path | High-Frequency Spectral Convergence | High-Frequency Energy Error (dB) | High-Frequency Flux Error | SI-SDR dB |
|---|---:|---:|---:|---:|
| Compressed Input | 0.723740 | 5.673287 | 0.804808 | 29.381521 |
| Raw Epoch168 | 0.664968 | 4.218196 | 0.809657 | 29.253909 |
| V4 | 0.651650 | 4.585417 | 0.809905 | 29.265028 |
| V5 calibrated | 0.647373 | 4.368251 | 0.811117 | 29.251306 |
| V6 | 0.655987 | 4.634038 | 0.807928 | 29.271228 |

V6 shows improvement over V5 in music spectral shape, flux, and SI-SDR, though high-frequency energy is worse.
Spectral shape and flux clustering shows 95% bootstrap confidence interval crossing zero, so the mean improvement cannot be declared as statistically significant.
SI-SDR difference: +0.07762 dB (95% CI [0.01740, 0.16547]); energy error difference: +0.24695 dB (95% CI [-0.05950, 0.60770]).
Speech high-frequency spectral convergence and energy are worse than V5.
No evidence of improvement over V4 in music energy or flux.
No claim of surpassing the baseline can be made based solely on V5 comparisons.

Using the same canonical metrics and 2,000 track-level bootstrap. High-frequency reference bias amplifies spectral convergence and flux; thresholds, denominators, and acceptance rules remain unchanged.
All 89 candidates show maximum deviation from offline output of zero under 128-frame and irregular chunking `31,512,997,7`.

### Held-Out Recheck: 78 Tracks, 39 Segments

42 music tracks / 21 segments, 36 speech tracks / 18 segments; each segment tested at both 44.1/48 kHz, including AAC 160 held-out codec.
Training and validation track, split_group, and source checksums do not intersect with these recordings, though they were previously used for diagnosis—thus not a new independent test, nor artist-disjoint.
Candidate selection receipt was frozen prior to scoring; no model, DSP, metric, or selection rule was modified after evaluation.

**Music**

| Path | High-Frequency Spectral Convergence | High-Frequency Energy Error (dB) | High-Frequency Flux Error | SI-SDR dB |
|---|---:|---:|---:|---:|
| Compressed Input | 0.721006 | 3.945078 | 0.874321 | 26.946485 |
| Raw Epoch168 | 0.745696 | 2.503481 | 1.119915 | 25.751111 |
| V4 | 0.742269 | 2.876600 | 1.117913 | 25.760181 |
| V5 calibrated | 0.761708 | 2.843306 | 1.141697 | 25.740227 |
| V6 | 0.733257 | 2.782127 | 1.092556 | 25.770247 |

**Speech**

| Path | High-Frequency Spectral Convergence | High-Frequency Energy Error (dB) | High-Frequency Flux Error | SI-SDR dB |
|---|---:|---:|---:|---:|
| Compressed Input | 0.507151 | 3.180363 | 0.623398 | 30.132120 |
| Raw Epoch168 | 0.553484 | 2.934310 | 0.684675 | 30.009073 |
| V4 | 0.538776 | 3.188172 | 0.694500 | 30.017263 |
| V5 calibrated | 0.531532 | 3.013707 | 0.692345 | 30.014774 |
| V6 | 0.547228 | 3.208142 | 0.691184 | 30.016087 |

Compared to V5: music spectral convergence difference is −0.02845 (95% CI [−0.04592, −0.01230]), flux difference is −0.04914 (95% CI [−0.07977, −0.02449], average drop ~4.3%), SI-SDR +0.03002 dB (95% CI [0.00508, 0.06689]), energy difference −0.06118 dB (CI crosses zero).
Compared to V4, music flux also decreased by −0.02536 (95% CI [−0.05742, −0.00471]), other metrics have CI crossing zero.
Compared to raw epoch168, music energy error **increased by 0.27865 dB** (95% CI [0.03230, 0.50896]); speech spectral shape and energy also degraded relative to V5.
Therefore, no claim of overall superiority over the original baseline or overall success can be made.

All 78 tracks output is bounded; both chunking modes show zero deviation from offline output.
Low-frequency absolute energy variation across 57 mono/left/right physical channels has a mean of 0.003380 dB; low-frequency residual power/input mean is −57.151 dB.
No new output exceeds full-scale. Low-frequency LSD does not imply volume change.

### Dry and DSP Ablation

Relative to unenhanced compressed input, V6 reduces music high-frequency energy error by 1.16295 dB (95% CI [−1.87244, −0.41811]), but increases flux error by 0.21824 (95% CI [0.06976, 0.46325]) and reduces SI-SDR by 1.17624 dB (95% CI [−2.20386, −0.43075]).
Development music also shows transient/waveform degradation in the same direction.
Thus, local superiority over V4/V5 does not equate to successful de-compression.

On the same 89 development samples, we perform actual Rust gain ablation: a diagnostic test, not used to reselect the held-out candidate or adjust default settings.
With MatchEdge disabled and direct model magnitude used:

| V6 Music Path | High-Frequency Energy Error (dB) | High-Frequency Flux Error | SI-SDR dB |
|---|---:|---:|---:|
| MatchEdge | 3.895419 | 1.405710 | 26.598670 |
| Model gain | 5.286458 | 1.267314 | 26.748162 |

Music energy error increases by 1.39104 dB (95% CI [0.74283, 2.06328]); speech increases by 0.94717 dB (95% CI [0.62758, 1.27927]).
When energy is underfilled, flux and SI-SDR can improve—indicating that reducing enhancement does not guarantee recovery success.
Default MatchEdge is retained.

### Stereo and Boundary

Two existing rock/electronic held-out recordings are fully MP3-128 encoded and aligned, then a 12-second stereo snippet is extracted, tested at both 44.1/48 kHz.
All four candidates are bounded; both chunking modes show zero deviation from offline output.
Peak maximum is 0.87734.
Not constructed by concatenating left/right training segments from different start points.
10 use cases (silence, pulse, mono, single channel) produce correct output length and bounded results; silence and uninput right channel are exactly zero.
Average panning width error is 0.03938 dB (V4: 0.03605, original model: 0.03992); inter-channel phase error is 0.04776 rad (original model: 0.04758, V4: 0.04773).
Sample size is small; cannot extrapolate to all stereo music.
These checks do not replace real playback or subjective listening.

New 20 anonymous 24-bit WAV files and local player are located at:
`data/reports/deployment-refinement-20261011/listening/index.html`.
Five conditions: lossless, compressed input, original model, V4, V6; scored based on peak RMS matching and common peak decay, not LUFS.
No scoring sheet was completed; no blind listening was conducted; materials are not considered evidence of acceptable listening quality.

### Resource and One-Minute Continuous Output

M5 Max, local CPU ORT single-threaded; benchmark source code is `d837c5e`, model source code remains `e6946b2`.
Training, audio evaluation, and resource measurement are executed sequentially; no overlap during measurement.
44.1/48 kHz × mono/stereo, 60 seconds continuous, warmup enabled, producer/worker real-time scheduling requests all accepted.

| Metric | V6 Measured |
|---|---:|
| ONNX Size | 3,309,786 bytes（3.16 MiB） |
| Synchronous full hop p99, across four configurations | 0.642–1.396 ms |
| Synchronous process peak RSS | 35.70 MiB |
| Paced additional software latency range | 5.3399–5.8129 ms |
| Maximum callback p99 across four configurations | 2.500 μs |
| Maximum callback max across four configurations | 7.917 μs |
| Paced overall CPU, single core percentage | 32.08% |
| Paced overall peak RSS | 41.11 MiB |
| worker deadline misses / queue overflows | 0 / 0 |

Synchronous V4 p99 is 1.010–1.726 ms, peak RSS 35.86 MiB; only one sequential run was performed, so cannot claim V6's architecture is faster.
Paced CPU = (user + system) / wall, covering 240.39 seconds of full process, including startup, test tool, and report caching—not overall machine CPU percentage; RSS includes test tools.
No direct causal comparison to prior 30-second test resource numbers.

All four use cases pass bounded values, output length, continuous output checks, enhancement availability, and callback budget.
Original callback nanosecond sequences and worker miss bitmaps were independently recalculated for p99, miss/burst, and latency; all 4 passed.
Fixed latency remains at 256 samples, meeting <8 ms target and <10 ms redline. No new model lookahead or buffering.
This is not a physical sound card, full device end-to-end, cross-platform, or 30-minute release validation.

The first one-minute paced test reused only the first 31 seconds of the old PCM, encountering input EOF at 31.42 seconds—this is not counted as passed.
Receipt saved in `cost-short-input-failed/`. Two benchmark input loaders were updated with required frame count checks (including full hop and synchronized warmup); old short files were rejected in 0.032 / 0.033 seconds.
The new PCM is derived from the original full recording, re-encoded to MP3, aligned, and truncated from the 10th second onward for 61 seconds.
First 31 seconds match the old file byte-for-byte, no looping or stitching; original SHA, PCM SHA, and alignment result saved in `realtime-pcm/receipt.json`.
This fix only affects test tools and does not alter the audio product processing path.

## Current Decision

Retain the raw epoch168 baseline; V6 is selected as a local experiment candidate.
Measurable progress is achieved in transient reconstruction on held-out music, but trade-offs remain in development and held-out, music and speech.
High-frequency energy and input fidelity have not met overall standards.
The best round among 8 is round 4; the next four rounds did not further reduce the selection target.
No evidence supports extending the same recipe for additional training rounds.
No proof of architectural ceiling has been established.

Next steps prioritize validating learnable/conditioned gain control to reduce conflict between fixed edge ratios and actual high-frequency energy;
simultaneously, consider using only non-linear temporal context from past frames to supervise phase and transient.
New controlled experiments are required, maintaining no lookahead, fixed latency, and callback budget.
Additional independent evaluation of genre/region and subjective listening evidence is needed.
These are future directions, not functionalities achieved in this round.

## Recheckable Materials

Complete local report: `data/reports/deployment-refinement-20261011/`.
Training logs: `data/runs/deployment-continuation-20261011/`.
Source code and configuration have been submitted; original data and experimental weights are not pushed.
Full local Python suite passed 254 tests; CI for source code [38079133782](https://github.com/backrunner/soundex/actions/runs/38079133782) shows successful Rust and export-contract execution.

The run completes 8 epochs and selects epoch 4; later epochs do not further improve the selection objective. Fixed versus irregular block comparisons have maximum differences 0 for development, 0 for held-out and 0 for the synchronized stereo probes. These numerical equalities do not establish listening quality.
