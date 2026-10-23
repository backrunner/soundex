# P26: Expanded Actual Training Coverage and Complete Residual Budget

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

2026-10-19 Decision: The 640-width large model is retired from future expansion training; the frozen endpoint remains as historical baseline.
Future mechanism positioning, external method research, and new path boundaries are detailed in [P27](head-collapse-audit.md).

2026-10-17 Registration and initiation; completed training and automated validation by 18:15 (UTC+8) on 2026-10-18;
2026-10-19 Verification of records and report hashes completed.
Codebase based on the frozen snapshot `4b06399`;
Local evidence directory: `data/runs/broad-budget-p26-20261017-v2/`.
All eight trajectories completed, but no candidate achieved overall recovery gains—default weights remain unchanged.

## Final Results

129,536 updates completed, 24 endpoints frozen, and 82 completion confirmations verified.
Endpoint proxy/DSP rechecks completed; for eight final endpoints, **1,424 full Rust frontend → ORT → DSP replay sessions** and **56 independent cost processes, 560,000 measured hops** were executed. Task exit code: 0.
All full replays show consistent detector conditions, with identical Rust observations. The maximum synthetic error across all metrics is `2.980232e-7`, below the specified threshold of `2e-5`. The observed deployment divergence is insufficient to explain the current quality stagnation.

Final endpoint music recheck results are presented below, representing mean values across two seeds and 33 music clips.
High-frequency spectral metrics derive from predictions replayed through real DSP; PCM data comes from the full Rust frontend and ORT path.
Lower values are better; no direct cross-metric comparison is valid, and reuse of recordings cannot be considered independent generalization testing.

| Path | RMS Error (Original Unclipped Reference PCM) | High-Frequency Spectral Convergence Error | High-Frequency Change Error | High-Frequency Energy Error (dB) |
| --- | ---: | ---: | ---: | ---: |
| Compressed Input, Unprocessed | 0.01240408 | 0.54787658 | 0.71375772 | 8.84387 |
| Identity Model + DSP | 0.01251822 | 0.54876080 | 0.71716419 | 8.48012 |
| finite_global | 0.01253625 | 0.55228936 | 0.72667301 | 7.62436 |
| finite_residual | 0.01253075 | 0.55275789 | 0.72480686 | 7.89268 |
| broad_global | 0.01251822 | 0.54874172 | 0.71716622 | 8.47238 |
| broad_residual | 0.01251822 | 0.54838372 | 0.71720389 | 8.43005 |

The music output from the broad coverage groups differs from the identity DSP by only approximately `8.63e-8` and `1.22e-7` RMS, respectively—behavior close to no enhancement. Such minimal degradation does not constitute successful restoration.
The finite pool group shows partial improvement in high-frequency energy, but degradation occurs in spectral shape, change error, and waveform error; seed differences are evident. The maximum RMS perturbation in identity for the finite pool groups is `0.001663` and `0.001822`, respectively.
The marginal relative spectral improvements in the broad coverage group fail to translate into overall waveform recovery. All pre-registered dual-seed comparative analyses failed to pass.
Voice output under broad coverage remains nearly identical to the identity DSP, with no demonstrated voice restoration benefit.

This supports the next step of investigating whether the shared objective encourages solutions close to identity—particularly the interaction between residual terms, identity constraints, and gradient competition among budget supervision. The root cause remains unconfirmed, and neither the capacity nor the broad coverage data can be dismissed.
Prior to further training, a diagnostic of training set objectives, gradients, and fit should be conducted to avoid repeating longer training or adjusting thresholds for specific validation tracks.
Blind listening, real-device continuity, and new held-out recordings remain unverified.

Cost testing shows that **all 560,000 measured hops are below 5 ms**, with a maximum observed value of 4.532 ms.
For 48 kHz stereo, three rounds without pacing: p99 = 1.121–1.171 ms, peak = 3.252 ms; paced: p99 = 2.200–2.352 ms, peak = 4.532 ms.
Peak RSS for stereo channels: 84.2–84.7 MiB; paced process CPU time/wall-clock time is approximately 68.0–72.5% of a single core, including loading and warmup.
These are local observations and do not guarantee performance across all hosts or loads. Differences from P25 cannot be directly attributed to structural optimization.

All measurements still show **136 individual updates exceeding their corresponding hop period**. Among them, 80,000 hops in the 48 kHz stereo paced case have 131 instances exceeding 2.667 ms, with 375 including wake-up scheduling delays, maximum delay 2.267 ms.
Actual callback queues, expired discards, dry backoffs, and device xruns are not included in this baseline. Therefore, a computation <5 ms does not equate to real-time continuity compliance. The 256-frame fixed buffer is separately evaluated.

Evidence: `study/summary.json`, `comparisons.json`, `native-audit/summary.json`, `native-cost/summary.json`; original reports and all weights remain locally ignored and unmodified, preserving the frozen experiment.

P25 identified two issues: the finite diagnostic pool provided only about 8 minutes of independent music supervision; when the original generation budget was zero, local power still allowed model modifications to input.
This round contrasts data coverage and budget scope to determine whether expanded audio exposure and full residual constraints are effective.
Reduced original signal perturbation does not equate to successful restoration—improvements in relative high-frequency spectral shape, energy, transient behavior, and full output error must also be achieved.

## Fixed Experiments

| Group | Training Sample | Budget Scope |
| --- | --- | --- |
| finite_global | Corrected finite pool | Historical global path |
| broad_global | Broad coverage sampling | Historical global path |
| finite_residual | Same finite pool | Full residual |
| broad_residual | Same broad coverage sampling | Full residual |

All four groups are 640-width, approximately 1,192M parameter complex heads, with two sub-versions: `20261012` and `20261013`. Each trajectory trains from scratch for **16,192 steps**, totaling **129,536 updates**.
Shared backbone initialization; identical data factors and sampling sequences; no reuse of P24 weights.
Endpoint fixed at 2,048 / 8,192 / 16,192 steps; final endpoints are primary comparison targets.
All eight trajectories completed, and only after 24 endpoint hashes were frozen were the held-out recheck datasets scored. No intermediate model was selected.

Loss, weights, and optimizer remain identical: historical rendered main + `0.1365237430457469 × clean complex auxiliary + 0.1 × budget auxiliary`;
Adam learning rate `1e-4`, no weight decay, gradient norm capped at 5.
Both budget definitions are identical when budget value is 1; the new definition disables full residual when budget value is 0.
This round does not alter loss normalization or expand model size, to isolate the two experimental factors.

Each update includes four distinct music clips and one speech segment, paired/identity-weighted, with no added gain perturbation.
This continues the **20% speech sampling ratio** from P24, not establishing it as a permanent standard for all product training.
Broad coverage uses randomized loop over recordings, within each recording, sampling encoding/channel/pairing rows by manifest weights, then uniformly selecting start offsets. It simultaneously expands source position, encoding, channel, and speech recording coverage—no single subfactor can be isolated as the cause of group-level performance.

## Actual Coverage Scope

The following represents the training-time frozen sequence coverage—**this does not imply that all audio has been fully trained**. Time spans the supervision interval `[89×128, 32768−128)`, merged across encoding, sample rate, and channel views using original source time; paired and identity share reference, with no overlap counting. This is more conservative than channel-view duration.

| Sampling Group / Seed | Music Recording | Independent Music Supervision (minutes) | Speech Recording | Independent Speech Supervision (minutes) |
| --- | ---: | ---: | ---: | ---: |
| Corrected Finite Pool (shared across two seeds) | 253 | 8.01 | 8 | 0.262 |
| Broad Coverage 20261012 | 253 | 331.14 | 193 | 14.94 |
| Broad Coverage 20261013 | 253 | 330.40 | 193 | 14.95 |

Each broad coverage seed includes 64,768 music clips and 16,192 speech clips; music spans 40,138 / 40,333 paired rows, including 9 encoding levels, 44.1/48 kHz, mono/left/right/mid/side.
This represents more thorough use of existing paired databases, with no new sources or changes to original material licensing.
Adjacent updates are independently constructed; left/right at different times are not considered the same stereo segment.

Training music region tags: China 25, Japan 41, Korea 50, Europe 27, North America 79, unknown 31. These are source/author library tags and do not guarantee local stylistic representation.
Genre distribution remains unbalanced: rock 49, electronic 48, classical 47, ambient 27, folk 22, pop 13, blues 7, jazz 6, country 5, hip-hop 2, funk 1, unlabeled 26.
Broad coverage does not artificially balance rare genres. The existing test split contains no Chinese music, so claims of regional generalization cannot be made.
Repeated validation/test sets remain only considered held-out rechecks.

## Isolation and Loading

Training/validation/testing splits are verified by recording, source checksum, and work group.
For each recording, the full input time interval of shifted probe is protected across all encodings, channels, and sample rates; precise fractional conversion of sample clocks is used, with codec alignment truncation of clean prefix accounted for.
The pre-warmup portion of training input must not overlap with protected intervals.

Previously, the finite pool only restricted supervision overlap, explicitly allowing pre-warmup overlap. This round identified **27 clips and 19 paired rows** requiring stricter movement rules—24 music, 3 speech. This cannot be directly attributed to old label leakage.
Prior to evaluating new model scores, these were moved to the nearest unused valid start point based on geometric distance, preserving original row, codec, channel, recording, and sampling index.
The shared training diagnostic pool was also corrected accordingly. All corrections are stored in `study/coverage.json`, so the finite group is now a corrected baseline, not a point-by-point replication of P24.

Data is loaded in batches with lazy loading; up to 128 prepared clips are cached, reusing existing disk cache, and no new random clips are written to NPZ.
On first read, each PCM file checks manifest checksum; clip generation checks sample rate, length, finite values, and mono-channel geometry.
Existing material quality and licensing reviews remain valid; actual per-clip checks occur at read time, and short-run checks cannot be interpreted as full data traversal.

## Execution and Acceptance

Four groups each completed a 4-step training set integration check, verifying actual loss, backpropagation, finite gradients, saving, and loading.
Weights are retained as local diagnostic evidence and are not used in official trajectories, nor for early stopping or quality filtering.
Training saves recovery state every 64 steps; at 2,048 steps and fixed endpoints, a common 320-crop training diagnostic pool is scored—this score is not considered indicative of overall broad coverage risk.
At recovery, model initial values, optimizer, RNG, configuration, and sequence hash are cross-verified.

Automated queue order:

1. Eight full training trajectories, 24 pre-registered endpoints frozen.
2. Unified proxy metrics and Rust-observed DSP synthesis rechecks, reporting music/voice, paired/identity.
3. Full export of endpoints, status, and per-hop ORT checks; eight final endpoints perform full Rust frontend → ORT → DSP.
4. Sequential test of full computation chain: 44.1/48 kHz, mono/stereo; 48 kHz stereo repeated three times, all final endpoints undergo paced long-term testing, totaling 56 processes, 560,000 measured hops. Reports include model + DSP computation, per-hop deadlines, backlog, CPU/RSS; real callback, device continuity, and blind listening remain to be verified.

Non-finite counts, input drift, or runtime errors halt execution and write error logs—no silent budget reduction occurs.
Number of completed steps does not imply convergence; low loss does not imply acceptable audio quality, nor can it substitute for user-defined requirements of computation <5 ms and sustained throughput validation.
The 256-frame fixed output buffer is reported separately and not included in user-defined model + DSP computation time.

Local state entry: `supervisor.json`, `pipeline.log`, `study/progress.json`; if failure, see `study/pipeline-error.json`.
Only code and source/protocol documentation are submitted; training audio, sampling sequences, checkpoints, and experiment reports remain in the ignored directory.
Registration SHA-256: `48301b33c361931b1cb4843874f00a1f517eee2c98c3d0f0c8cad977bf2eb5b7`.

On startup verification, the first registration was missing the required `native_count` for the final analyzer. As a result, the first trajectory was automatically stopped after 256 steps, preserving the original directory and `superseded.json`.
Version 2 added hop count contract fulfillment and included actual registration and hash drift regression testing, re-freezing code and registration, and restarting all budget from scratch.
No modifications to prior registration or model scores were made, and no use of the first version weights for continuation training.

Startup validation: **548 full Python tests passed**, Ruff and formatting checks passed.
This round confirmed completion receipts and original report hashes, document links, and diff format; no changes to training code were made.
Full training and automated recheck completed; real device continuity and perceptual validation remain unverified.

The exact registered seeds are `20261012/20261013`. These workstation observations
do not guarantee compute below 5 ms on every host or background load.
