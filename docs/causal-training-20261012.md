# P2: Training with a Unified Objective Using Three Main Branches

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

October 12, 2026, following [P0/P1](causal-prototypes-20261011.md). Code is frozen at
`50dd268c621acad9e48a446eda3ed786c058afec`. Formal ablation testing has been initiated on the local MPS hardware.
**Six runs have been completed, which does not imply surpassing the baseline or readiness for release**. Default weights were not replaced.
[P3: Weak High-Frequency and Generalization Control](weak-band-study-20261012.md) has been launched.

## Fixed Experiment Setup

| Item | Value |
| --- | --- |
| Main Branch | Freshly initialized from the original `shape-small`, original GRU, original TCN |
| Parameter Count | 905,734 / 643,458 / 639,906 |
| Seed | 20261012、20261013 |
| Budget per Group | 4096 optimizer updates, batch size 4, total 16,384 paired crops |
| Cropping | 32,768 samples, 256 hop, native 44.1/48 kHz |
| Model Warmup | First 64 hops with no gradients; subsequent state detach |
| Supervised Region | Raw spectral input at 192 hops; waveform reconstruction skips first 65 hops, retains 191 hops |
| Optimizer | AdamW, learning rate 3e-4, weight decay 1e-4, cosine decay to 0.1, gradient norm capped at 5 |
| Checkpointing | Every 512 steps and at end; saves `last` and `best` (lowest fixed validation score) |
| DSP Strategy | Model gain + Model phase; runtime decisions driven by no reference audio |
| Validation | Fixed 89 development validation tracks selected as before; each centered and cropped; separate reporting for music and speech |
| Selection Rule | Music mean ×253/254 + speech mean ×1/254; no automatic promotion |

The legacy main branch uses the dual-stream structure of original `shape-small`, with gain/shape refiner, interaction and circular input disabled.
During training, only the supervised region updates BatchNorm; validation uses frozen statistics. It has no historical state.
The new main branch explicitly maintains per-channel physical history, with warmup state not introducing output latency.

This round evaluates performance under a fixed training budget, without claiming to have exhausted the optimizer or capacity limits of any architecture.

The previously approved music/speech manifest is reused, with recipe
`7fda9e58ab583617f3514f9dd9cfcf87f29871ae0587e91ca29ac0ddaf6b29d4`.
There are 300 music tracks in total, with 253 assigned to training; 193 speech recordings are used for training.
Music tracks are evenly distributed, maintaining a music-to-speech source probability of 253/254 and 1/254 respectively.
Each seed pre-froze row, start point, and common gain (−6 dB to +3 dB); all three main branches are fully reused.
Actual source sampling count, covered recordings, and effective supervised seconds are logged in `draws-<seed>.json`; probabilities are not exact quotas.
The first seed uses 16,318 music and 66 speech crops, covering 314 unique music/speech recordings.
Each group accumulates approximately 145.20 minutes of waveform supervision (cumulative over repeated crops, not new independent material).
Cross-split leakage is prevented by track ID and original checksum; the first read of paired PCM is verified for checksum, sample rate, length, and finite value.
Prior cached data is also checked for PCM integrity. MUSDB, Slakh, and FMA datasets were not re-introduced.

## New Objective: causal-reconstruction-v1

This is an independent version and cannot be directly compared to the total loss of V5/V7. All main branches use identical weights and normalization.

| Component | Weight |
| --- | --- |
| Raw high-frequency magnitude / complex / energy / time variation | 1 / 0.2 / 0.1 / 0.25 |
| Raw preserved band complex identity | 0.1 |
| Waveform error of synthesized proxy | 1 |
| Output high-frequency magnitude / time variation / energy | 1 / 0.5 / 0.1 |
| Output preserved band complex identity | 1 |
| Overlimit penalty before clipping | 0.1 |

Multi-resolution output spectra use 256, 512, and 2048-point analysis with hop size set to one-fourth of each window;
temporal constraints match the lossless reference for change, avoiding static smoothing in place of transient recovery.
High-frequency and waveform components target lossless references; preserved bands are constrained to identity with respect to input.
Normalization uses fixed scale and energy floor from reference or input, and no prediction amplification is allowed to suppress scores.
Waveform term is normalized by original lossy error, with 1% reference RMS and absolute floor to prevent divergence near identity or silence.

The proxy per-hop uses actual Rust detector cutoff/active thresholds, reproducing band magnitude and complex vector mixing, Hann overlap-add, soft clipping, and fade-in/fade-out.
The detector sees only the input; both parameters passed to the trace tool use the same lossy PCM to avoid clean labels entering the input condition.
**Dry and lossless waveform targets directly use aligned PCM**, without STFT re-synthesis.

Known bounds of the proxy:

- MR loss band weights use average cutoff from the supervised region; synthesis path still uses actual cutoff per hop.
- Each crop starts from cold DSP state; the first 64 hops disable enhancement and are warmed up; this is not a long stream state test.
- Mono and stereo tracks are trained independently per channel, not combined into a pseudo-stereo output.
  Stereo imaging, panning, and long-term continuity must be separately validated on real paired streams.
- Floating-point analyzer, FFT, and windowing implementations have minor differences; no claim of bit-level consistency across the entire proxy.
- The legacy main branch does not wait for 64-hop state during runtime; training uses a common supervised region, and actual Rust scoring reports both full track and a stable interval after 72 hops, to avoid startup differences from contaminating architectural conclusions.

## Pre-Run Verification

A short training run (8 updates × batch 4 × 1 seed) was conducted for all three main branches, completing limited gradient updates, checkpointing, export, and actual Rust waveform scoring.
ONNX sizes are 3,722,892 / 2,590,696 / 2,599,024 bytes, all within parameter ≤2M and single-file <8 MiB.
This short run does not serve as evidence of architectural superiority or audio quality improvement.

Pressure inputs for the training proxy and external Rust prediction playback include saturated pure tones, silence, and reactivation:

| Sample Rate | Max Absolute Waveform Error | RMS Error |
| --- | --- | --- |
| 44.1 kHz | 9.996e-5 | 3.724e-6 |
| 48 kHz | 9.435e-5 | 3.305e-6 |

Test constraints: peak atol 2e-4 (rtol 2e-5), RMS <2e-5.
Pure tone observations at near-zero frequency points are sensitive to FP32 window phase errors, and when mixed with non-zero generated energy, exact phase equality is not required.
Finite values, 0.95 clipping, silence tails, and OLA consistency are separately asserted; no relaxation of semantic or temporal checks.

Fixed weights, one real cached crop, forward and backward pass probing on CPU/MPS:
maximum difference in total loss across the three main branches is 3.815e-6, gradient relative L2 differences are 2.20e-6 / 3.61e-6 / 2.88e-6, all small.
This is a single-batch numerical probe and does not guarantee long-term training consistency.
PyTorch MPS FFT still prints internal resize deprecation warnings; current computation and backward pass succeeded.
Environment version is recorded in immutable configuration and not silently enabled with CPU fallback.

## Local Execution and Post-Training Thresholds

Run directory: `data/runs/causal-p2-202612/`

- `launch.json` records PID, UID, start time, command, and source code SHA; `train.log` contains continuous runtime logs.
- `config.json` and `provenance.json` store fixed objectives, environment, source code, manifest, selection, and binary hash.
- `progress.json` tracks current group and training step; per-group saves checkpoint, validation details, `best.onnx`, and `rust-validation.json`; only after all six groups complete is the overall `summary.json` marked as `complete`.
- On failure, a `failure.json` is written; identical commands can be resumed, but changes in configuration or source code are rejected; file locking prevents duplicate launches.
  On recovery, original failure records are preserved. No cached data, raw data, or experimental weights are submitted.

Pre-run evidence is stored in `data/reports/causal-p2-20261012/smoke-final/` and `numeric-audit.json`.
Source code and methodology documentation are released under Apache-2.0; no new external data or model licenses were introduced.

After each group, a real Rust Model-gain path score is automatically computed and reports results for high-frequency, energy, transient, low-frequency, and music/speech.
Current reporting is bootstrapped by segment; a [recheck of the first seed](causal-p2-review-20261012.md) identified and corrected the issue to report per-track clustering intervals, without altering the frozen training source code.
Repeatedly used development data is not claimed as independent validation.
Post-training, candidate models must be frozen and uniformly rechecked against baseline, original compressed input, DSP paths, historical ablation, true stereo, full segments, blind listening, and per-hop CPU, RSS, and real-time availability.
`listening_performed` remains `false`. Before full post-training weight validation is completed, no claims of P1-level performance or release readiness are made.
Real-time software target remains under 8 ms, redline under 10 ms, with no inference, allocation, or blocking during callback.

## Final Validation Results for Two Seeds

| Main Branch | seed 20261012 | seed 20261013 |
| --- | ---: | ---: |
| legacy | 2.662400 (4096 steps) | 2.675942 (3584 steps) |
| GRU | 2.485908 (2048 steps) | 2.483496 (3584 steps) |
| TCN | 2.462378 (4096 steps) | 2.476988 (4096 steps) |

All six groups trained to 4096 steps; values in the table represent the best checkpoint selected under fixed rules.
The new main branch demonstrates superior performance in the same objective across both seeds, though this does not imply that the output is universally superior to the old model or compressed input.
P3 selects TCN as the controlled continuation carrier, without claiming that TCN audio quality is significantly better than GRU.

Exact registered geometry: seed 20261012 (and 20261013), 4096 updates at batch four, 16384 paired draws per arm; each crop has 32768 samples and 256 hops. Local validation passed 291 Python tests; the existing 30-minute stress test remained ignored. These checks do not replace the <10 ms historical software redline or device qualification. The retained run directory is `data/runs/causal-p2-20261012/`.
