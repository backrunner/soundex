# P19: Controlled Comparison with Expanded Music Recording Coverage

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

## Problem and Scope

P18 exhibited degradation across music recordings even after convergence in optimization; its diagnostic training set consisted of only 32 music tracks, lacking representation of blues, funk, and hip-hop. This round of evaluation investigates whether expanding the music recording coverage leads to tangible improvements in recovery performance—without treating the addition of new tracks as a guarantee of quality.

Existing frozen data and previously approved source reviews are used; no new audio material is downloaded, and development or test counterexamples are not introduced into the training set.

- **Small set**: Original 32 music tracks + 8 speech segments
- **Large set**: Full 253 music tracks + same 8 speech segments
- The remaining 26 validation and 21 test tracks remain isolated.
- For each track, only one existing paired segment is selected, along with eight protected time positions. This does not involve training on all time intervals and codec combinations across the 253 full songs.

| Category | Small Set | Large Set |
| --- | ---: | ---: |
| ambient | 7 | 27 |
| classical | 6 | 47 |
| electronic | 7 | 48 |
| rock | 4 | 49 |
| folk | 1 | 22 |
| country | 1 | 5 |
| jazz | 1 | 6 |
| pop | 3 | 13 |
| blues | 0 | 7 |
| hiphop | 0 | 2 |
| funk | 0 | 1 |
| unannotated | 2 | 26 |

## Selection and Sampling

- The anchor/crop positions for the original 32 music tracks and 8 speech segments remain unchanged. The new position library includes the old one as a subset.
- 221 additional tracks are selected in order of `track_id`, and matched cyclically to the original music by codec ID and sample rate. For each, a row is selected based on the SHA-256 hash of `row_id`; no target spectrum or development score is used in selection.
- Mono is used for monaural content; for stereo, the mono template is mapped to the mid channel, and other channel roles are preserved. A total of 12 such channel role mappings are defined, each logged individually. No cross-timestamp stereo channel fusion is performed.
- Time positions retain the relative start point of the template, then apply the original protected window and eight-position rules. Real-time pre-checks do not require start point rollback.
- Music and speech use separate balanced sampling streams; each recording's crop is independently sampled per batch. For the same seed, both sets share a common update budget, with speech recordings and positions being fully identical across sets. No more than one music track per batch is repeated.
- Each batch consists of 4 music tracks + 1 speech, and each has its own identity. This is a single-factor diagnostic ratio and does not alter the official training configuration that mixes genres with speech. The large set still exhibits imbalanced genre and source distribution, so it cannot be considered a full musical genre coverage.
- Codec and sample rate templates for new and old sets are made as closely matching as possible, though 253 is not an integer multiple of 32, resulting in minor rounding discrepancies. Source channel layouts also differ; actual distributions are logged, and no claim is made that an exact joint distribution is achieved.
- Supervision does not intersect with the preserved input window; there remain 27/320 and 186/2,088 training positions where input or warmup context overlaps with the preserved window. This is a limitation of the existing protocol and cannot be claimed as a fully non-overlapping audio segment.

## Initialization and Budget (Fixed Before Training)

Two seeds: 20261012 / 20261013. Each seed trains one small and one large set, starting from the same seed-generated 807,398-parameter model, with initial tensor hashes recorded. Fixed hyperparameters: `lr=1e-4`, Adam optimizer without decay, `clip=5`, identical primary loss and auxiliary coefficient `0.1365237430457469`; auxiliary coefficients are not re-tuned, and DSP is unchanged.
P18 previously trained for 256 steps at `1e-3`, and cannot be directly used as a substitute for the low-learning-rate small set comparison in this round.

| Comparison | Small Set | Large Set | What It Answers |
| --- | ---: | ---: | --- |
| Same update count | 2,048 steps | 2,048 steps | Benefit of expanded music coverage under identical updates and speech sampling |
| Same music exposure | 2,048 steps | 16,192 steps | Result per track after 256 random sampling per track; different update budgets |

Each music position appears 32 times at the equal music exposure endpoint. At the 2,048-step mark, the large set sees each track appear 32–33 times, and each position 4–5 times. A total of four training trajectories and 36,480 updates are logged.

**The second comparison is not equal full corpus exposure:**
- The small set has each speech segment appearing 256 times.
- The large set, with longer budget, sees each speech segment appear 2,024 times.
- Voice performance gains/losses must be reported separately; any difference due to longer training cannot be attributed solely to music coverage expansion.

## Scoring and Interpretation

- The original 320 training positions are evaluated at both 2,048 steps and final evaluation; additionally, the full position library (320 / 2,088) is checked at each registered comparison endpoint.
- Mean performance across different training sets is not claimed as a direct improvement over the common dataset.
- No pre-selection of points is made based on training or development results. All fixed checkpoints are completed and hashed before development scoring begins.
- Seven conditions are evaluated:
  1. Initial identity DSP
  2. Two small set runs at 2,048 steps
  3. Two large set runs at 2,048 steps
  4. Two large set runs at 16,192 steps
- All 1,246 Rust DSP re-plays are performed on paired and identity development samples, reporting per-segment high-frequency spectral shape, energy, transient behavior, low-frequency fidelity, and per-track/per-recording mean performance.
- P18 recovery thresholds are applied: the large set must show improvement in music primary loss, paired recovery, native high-frequency spectral shape, and transient fidelity over both small sets and initial identity DSP, and must exceed both. Low-frequency and energy must not degrade relative to the small set. Tolerance remains at `1e-6`, not interpreted as perceptual threshold; degradation in any segment is reported.
- Reused development recordings are not independent tests; passing this screening only supports further testing and blind listening. If only identity or energy improves while recovery details degrade, no further training is added along this route; instead, evaluation moves to output representational expressiveness analysis.

## Execution and Output

- `training/data/coverage_study.py`: Nested music/position library with independent balanced sampling.
- `training/scripts/run_music_coverage_study.py`: Logs, recoverable training, fixed endpoint native playback.
- `training/scripts/analyze_music_coverage_study.py`: Product validation and comparative analysis between the two budgets.
- Experiment directory: `data/runs/music-coverage-p19-20261014/`.

Single process, single PyTorch CPU thread; position libraries are reused as read-only; shared indexing views are used for scoring to avoid re-copying large tensors. A checkpoint is saved every 64 steps, preserving model, Adam optimizer, random state, and sampling positions.
New tests for nested position protection, mono, codec matching, exact exposure, speech consistency, sample rate invariance, and model interruption recovery are verified at the per-parameter, per-momentum, per-update level.

Raw weights and original audio are retained locally and ignored. Deployment graph, default weights, or real-time buffering are not modified.
Frozen predictions replayed via Rust do not replace the per-hop performance or real-device validation of candidate ONNX models.

Pre-launch validation: 465 local Python tests passed, Ruff lint/format passed. Selection of 261 recordings corresponds to 522 clean/degraded PCM files, all of which passed actual checksum, sample rate, length, mono shape, and finite value checks. Results are saved locally in `data/runs/music-coverage-p19-20261014-preflight.json`, including source manifest and configuration hash.
This does not constitute legal opinion or master tape authentication; existing archived source reviews continue to be used.

## Launch State

At 00:39 (Singapore time), the background queue was launched, PID `5235`, source snapshot `b2b8a2d04de21ccda9e528107cce6442bfab7367`. The actual registration phase was completed, followed by sequential execution of:
- Two small set seeds
- Two large set seeds
- Full fixed endpoint revalidation
- Automatic analysis

Real-time status is available in `pipeline.json` / `study/progress.json`; logs are in `pipeline.log`.
No results from this round have been produced yet; default weights remain unchanged. Remote CI status reflects the actual repository execution.

## Mid-Check (2026-10-14 00:58, Singapore Time)

Both small set seeds completed 2,048 steps. Primary loss on the common 320 training positions:

| Seed | Initial Value | Small Set End | Relative Change |
| --- | ---: | ---: | ---: |
| 20261012 | 0.28204553 | 0.26948623 | -4.45% |
| 20261013 | 0.28204553 | 0.27041567 | -4.12% |

The large set has completed preparation for 2,088 positions and is now training on the first seed; at the time of check, it had passed over 448 steps, with a fixed endpoint of 16,192 steps per seed. These metrics reflect only the common training library error and do not yet include development set or native DSP comparison results.
No conclusion can be drawn that expanded coverage improved recovery quality.

The main branch cleanup did not alter the running source snapshot, optimizer, data, or model. Queue PID 5235 / training PID 5994 continues to run. After cleanup, all frozen input hashes in `pipeline-plan.json` were individually verified. All checkpoints and validations proceed as originally scheduled, regardless of intermediate scores.

## Final Results (2026-10-14 02:35, Singapore Time)

All four training trajectories, six fixed candidates, initial control, and 1,246 Rust DSP re-plays have been completed. The frozen analyzer re-validated sources, inputs, sampling, initialization, checkpoints, and completion receipts. All four budget/seed combinations failed the pre-registered recovery screening criteria.
The maximum per-sample error between native output and training proxy is `1.290440559387207e-5`, below the replay threshold of `2e-5`.

The following values are mean values over reused development music segments—lower is better. Primary loss is the original product reconstruction proxy; spectral and transient metrics come from actual Rust output. These do not represent blind listening or independent, untested evaluations.

| Condition | Music Primary Loss | High-Frequency Spectral Convergence Error | High-Frequency Transient Error | High-Frequency Energy Error (dB) |
| --- | ---: | ---: | ---: | ---: |
| Initial Identity DSP | 0.294182 | 0.548761 | 0.717164 | 8.480117 |
| 32 Tracks / 2,048 / Seed 12 | 0.437930 | 0.719934 | 0.837148 | 7.096696 |
| 32 Tracks / 2,048 / Seed 13 | 0.411146 | 0.666151 | 0.808389 | 7.380139 |
| 253 Tracks / 2,048 / Seed 12 | 0.299495 | 0.548391 | 0.717183 | 8.339510 |
| 253 Tracks / 2,048 / Seed 13 | 0.299090 | 0.550844 | 0.718649 | 8.326952 |
| 253 Tracks / 16,192 / Seed 12 | 0.409889 | 0.717505 | 0.829274 | 7.542976 |
| 253 Tracks / 16,192 / Seed 13 | 0.382140 | 0.675445 | 0.806718 | 7.376077 |

At the 2,048-step mark, expanded coverage significantly reduces degradation in the small set development set, though the music primary loss remains above the initial identity control. Spectral and transient fidelity show no consistent dual-seed improvement.
Relative to the small set, music primary loss decreases by approximately 6.4% / 7.1%, but relative to the initial control, it increases by about 39.3% / 29.9%. High-frequency spectral convergence error increases by about 30.7% / 23.1%, and transient error increases by about 15.6% / 12.5%. When aggregated by track, final music primary loss, spectral, and transient metrics still outperform the initial identity control. The conclusion is not dependent on fragment counting.

The large set with extended budget shows reduced relative degradation compared to the small set, but its development music identity loss remains at 0.135301 / 0.110671, while the initial proxy is zero. Speech primary loss decreased from 0.180645 to 0.175249 / 0.175040.
However, speech gains do not equate to music recovery gains, and the extended budget increases speech exposure.

Therefore, insufficient coverage is a contributing factor, but this round does not support it as the sole cause. While overall training error decreased, music performance degraded under longer training, indicating persistent generalization and fidelity issues. These findings do not conclusively point to capacity limits, a loss term error, or an inherent architectural flaw.
No additional training is added under this configuration. The next step should be a representational expressiveness analysis of output consistency, evaluating paired recovery, identity fidelity, and cross-recording generalization, while maintaining model size, causality, and real-time budget.

Default weights remain unchanged. This round produced no new ONNX latency/resource tests or scored blind listening results.
Passing the replay consistency check does not imply real-time deployment or audio quality acceptance.

Post-hoc failure analysis is complete: errors are concentrated in a small number of recordings, and on the current P19 weights, adjacent band interference is reproduced through stage decomposition and real Rust intervention.

Historical review coverage was 89 development clips and 40 protected windows from the same recordings. The identity proxy initially has error 0. The completed review dated 2026-10-14 is followed by the [coverage failure analysis](coverage-failure-analysis.md).
