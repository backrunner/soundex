# P25: Realistic Inference Frontend, Budget Pathways, and Training Coverage Audit

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

This round builds upon [P24 Full Validation](generation-budget-study.md), focusing first on error localization and deployment cost analysis. No intermediate best endpoints are selected, default weights are not replaced, and historical failure checks are not reinterpreted.

---

## Confirmed Limitations in Training Coverage

P24’s 2,088 cropped segments do not represent a complete music corpus:
- Music: 253 paired rows, 8 positions per row
- Speech: 8 paired rows, 8 positions per row

Each song has only one encoding version, one channel view, and a source snippet of approximately 3 seconds. Although the full pool includes 9 encoding configurations, this does not imply that every song participated in multiple encoding training scenarios.

By merging actual supervised intervals by source file, channel view, and native sample rate, and excluding 89-hop warmup/boundary regions and overlap, we obtain:
- **482.328 seconds (8.04 minutes)** of music supervision
- **15.949 seconds** of speech supervision

Paired and identity reference intervals are shared, so they are not counted as separate audio segments. This total represents the union of mono-channel supervised intervals—not the total length of the media library, nor the sum of all repeated exposures across rounds.

The current training list includes:
- 71,327 music paired rows, 253 recordings
- 13,896 speech paired rows, 193 recordings
All cover 9 encoding configurations. P24 did not utilize all of these rows.

This indicates that the next iteration must explicitly distinguish between **limited-pool fitting** and **broad-coverage training**. Current evidence suggests generalization risks, and coverage statistics alone cannot prove that expanded sampling will lead to performance improvement.

Reproducible entry point: `training/scripts/audit_crop_coverage.py`
Evidence saved in: `data/runs/context-deployment-p25-20261017-v3/crop-coverage.json`, tied to original registration and script hash.

---

## Realistic Rust Input-to-Output Path

A new path, `soundex-context-replay`, is introduced, explicitly accepting experimental four-input ONNX. The formal model loading protocol remains unchanged. This path executes in sequence:
- Production short-window analysis and bandwidth detection
- Rust causal 1024/2048-long window context
- Single-threaded CPU ORT execution
- Production spectral fusion, phase processing, OLA, clipping, and channel sharing gating

It uses the production `ChannelProcessor` and `GateMixer`, without introducing a simplified DSP version as a substitute.

Each physical channel maintains its own history; silence and bypass paths also execute the model and update state. Long windows update every 4 hops and retain only past and current PCM. The first 80 hops of output are aligned with the dry signal.

A critical historical condition is preserved: P24’s input cache zeroed the active features from the first **64 hops**. This is distinct from the **80-hop model warmup**. The initial access check failed due to failure to reproduce this convention. After correction, a full recheck was performed, and the old failure directory is retained.

In this round, all 16,192 endpoints for four configurations and two seeds were fixed and fully replayed, along with 89 development pairs across two roles, totaling **1,424 full inference replays**. The supervision interval remains `[89*128, 32768-128)`. All recordings were used in development and remain labeled as development rechecks.

Per-item reports retain:
- Detection conditions
- Context differences
- Spectral differences between ORT/training implementation
- Synthesis error against observed spectra
- Absolute PCM error relative to clean/original and identity DSP/old proxy

The observed-spectrum consistency threshold remains at `2e-5`, unchanged from historical standards.

All 1,424 roles are now complete:
- Maximum detection difference: **0**
- Maximum relative RMS difference in long-window context: `2.00472e-7`
- Maximum PCM difference in observed-spectrum synthesis: `2.98023e-7`
- No failures

The full frontend’s maximum per-fragment RMS difference relative to the old training proxy is `0.00011284`, preserved and not masked.

Average RMS error of original clean waveform for music final endpoints across two seeds:
- **control**: `0.01251827`
- **balanced**: `0.01262628`
- **budget**: `0.01254735`
- **joint**: `0.01260773`

These values are consistent with P24 results.

The full inference pipeline does not eliminate mis-enhancement of the original signal, nor does it cause any candidate to outperform the unprocessed compressed input.

Quality recheck directory: `data/runs/context-deployment-p25-20261017-v3/audit/`
This is an offline replay of the full computation chain and does not include production real-time queues, frame drop recovery, device callbacks, or blind listening.
The fixed real-time buffer remains at **256 frames**; the 128-frame delay inherent in offline stitching must not be misrepresented as total adapter latency.

---

## Budget Does Not Constrain All Residuals

For all `budget` and `joint` development roles, we performed a common ablation:
- Freeze the model and all downstream processing
- Set the generation budget to zero
- Analyze only structural behavior

This ablation cannot produce real-world audio quality judgments and cannot be considered a viable deployment-level rejection enhancement strategy.

P24 defined residual magnitude as:
`scale * sqrt(local_power + budget²)`

When budget is zero, the residual becomes:
`scale * sqrt(local_power)` — which still allows modification of existing frequency bins.

In key identity counterexamples, index 33:

| Seed | Full Rust Extra Perturbation RMS (original) | After Zero Budget (proxy) |
| --- | ---: | ---: |
| 20261012 | 0.00245088 | 0.00035585 |
| 20261013 | 0.00232160 | 0.00042886 |

The original model’s full frontend switch error in this case was only about `1.62e-6 / 1.27e-6` RMS — significantly smaller than the original audio perturbation. The frontend difference is real, but it does not explain the observed residual enhancement in these two cases. Index 83 also shows non-zero residual after zero budget. All cases are retained; no thresholds are added for these recordings.

A new optional `budget_scope="residual"` is introduced, using:
`scale * sqrt(local_power + 1) * budget`

This incorporates the full residual into the continuous budget.
- When budget = 1, behavior matches the old structure
- When budget = 0, behavior is strictly identity
- Missing bands can still be generated via full-band scale

No new parameters are introduced, and the budget is not interpreted as a probability or perceptual threshold. The old checkpoint continues to default to `budget_scope="global"`, preserving prior results.

This candidate has been verified for:
- Zero budget
- Full expressiveness under missing frequency bands
- Finite gradient under weak energy
- Finite difference behavior
- ONNX single-step and state/reset consistency

However, this structure was not trained during P25. It was fully trained and rechecked in [P26](broad-budget-study.md), yet still failed to achieve overall recovery gains.

The budget predictor itself still errs on unseen clean audio; its correction range cannot replace broad coverage training or generalization validation.

---

## Cost Testing and Next-Phase Training

Timing is independent of quality auditing and is initiated only after audit completion to avoid CPU contention.

Eight fixed final endpoints were tested at 44.1/48 kHz, mono/stereo. Each test includes:
- 100 pre-warmup hops + 10,000 measurement hops
- 48 kHz stereo tested twice, with model execution order varied

Additionally, paced measurements were conducted for control and budget seeds, with fixed arrival clocks. Total: **52 independent processes**, each bound by model, binary, and execution script hash.

Metrics recorded:
- Frontend / ORT / DSP / total compute distribution
- Count of ≥5 ms events
- Per-hop timeout counts
- Virtual backlog
- Paced completion lateness
- Process CPU and peak RSS

All tests used the same macOS `USER_INITIATED` QoS request as production, with explicit success logging. No claims are made about production worker time-constrained scheduling or device-level testing.
Input is fixed synthetic multi-voice and envelope variation, used solely for cost evaluation, not for quality assessment.

Evidence stored in: `data/runs/context-cost-p25-20261017-v2/`

All 52 entries are now complete and have undergone hash verification of original timing files. Total: **520,000 measurement hops**, plus **5,200 pre-warmup hops**. All QoS requests succeeded. Output is bounded, with peak synthesis amplitude of approximately **0.27617**.

These results do not constitute validation of real music audio—no evidence of burst noise, imaging, or device continuity is provided.

For 48 kHz stereo, three runs (two seeds, 60,000 measurement hops per configuration):

| Configuration | 6x p99 range (ms) | Max (ms) | ≥5 ms count | Single-hop over | Timeout (with virtual backlog) |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 1.834–3.622 | 23.772 | 34 | 1,255 | 2,195 |
| balanced | 1.903–2.566 | 24.662 | 16 | 248 | 486 |
| budget | 1.566–3.469 | 9.989 | 25 | 1,180 | 2,020 |
| joint | 1.901–2.573 | 10.223 | 7 | 216 | 349 |

Peak RSS for these dual-channel processes: **82.7–84.7 MiB**, including model load overhead—this is not incremental memory.

Without rhythmic pacing, process CPU time / wall-clock time is approximately **94–97% of single-core** (including load and warmup).

While mean and p99 values show room for improvement, the observed spikes and hop-level timeouts do not meet continuous processing requirements. They do not justify declaring stable sub-5ms performance for model + DSP. Virtual backlog counts are not equivalent to audio device xruns and cannot be used as a proxy for real callback availability.

For paced 48 kHz stereo, 10,000 hops per run:
- **Budget** seeds had **529 / 382** late completions, max lateness **4.934 / 10.325 ms**
- **Control** had **610 / 134** late completions, max **2.568 / 2.349 ms**

These include sleep/wake and scheduling overhead, not per-model computation time.
Budget’s per-hop p99: **3.178 / 2.996 ms**, max **5.945 / 5.400 ms**
Four paced runs averaged ~52–58% CPU usage, peak RSS ~84.0–84.8 MiB.

The pacing engine continuously processes all inputs. However, production worker timeout drops, state resets, and dry alignment backoffs are not included in this measurement. Thus, the lateness rate cannot be directly interpreted as playback dropout rate or enhancement availability.

Timing was only initiated after quality audit completion, but the host was not in an isolated idle environment: background GUI applications and compilation jobs ran during the test. Load snapshots and per-run `load average` values are preserved in local reports. All configurations show tail behavior, so maximum values cannot be used to claim that the budget branch is inherently slower. Similarly, peak spikes cannot be suppressed to claim compliance.

Next phase should treat **data coverage** and **budget scope** as independent variables:
- Maintain two seeds
- Use identical, sufficient, and consistent training budgets
- Share the same objective and fixed final endpoints
- Compare limited-pool vs. broad-coverage and `global` vs. `residual` structures

Broad coverage must sample multiple source positions, encodings, and channel views, preserving the established corpus proportions.
- Exclude held-out recordings and protected intervals
- Report actual exposure, not just repeated cropping of the same segment

P25 did not initiate new training. [P26](broad-budget-study.md) completed new sampling, isolation testing, and full comparative registration, finishing training and validation on **2026-10-18**. Results are detailed in the P26 report—**no new weights are deemed acceptable**.

The new structure, training frontend, and real-world output chain must be evaluated as a unified system.
Reducing identity perturbation does not equal success.
High-frequency spectral shape, energy, transients, original waveform, preserved bands, stereo, cross-recording consistency, and blind listening remain essential and must be preserved.

**Verification**:
- 541 Python tests passed
- New timing report JSON serialization regression checks passed
- Rust workspace: 127 tests passed, 1 existing test skipped
- Clippy, no-default-feature checks, Ruff, and formatting checks passed
- CI remote run `207c23d` [passed](https://github.com/backrunner/soundex/actions/runs/39018178680)
- Real device and perceptual validation remains pending.

The music endpoint means in configuration order are `0.01251827 / 0.01262628 / 0.01254735 / 0.01260773`; these are final-output errors against original clean PCM, not added error against identity DSP.
