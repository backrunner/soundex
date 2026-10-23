# P35: Broad Coverage Error Compression Evaluation for Small Models

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

This round builds upon [P34 Three-Round Diagnostics](risk-balance-study.md), maintaining the same **431,015-parameter** model, and investigates whether the benefits of limited clipping under moderate identity weights can be reproduced across broader training coverage and a second seed.

Four formal training runs and full automated rechecks have been completed. Both seeds show improved music and speech spectral errors, but degradation in dynamic and identity fidelity persists, failing the pre-registration screening criteria and not advancing to default weights.

---

## Motivation and Fixed Baseline

In P34, `compressed_identity_mid` achieved a ~0.305% reduction in common MR error on untrained music segments compared to its contemporaneous control, though both dynamic and identity spectral fidelity degraded. The addition of unilateral overcompensation penalties reduced performance on the target training set while worsening performance on out-of-distribution segments. Therefore, P35 does **not include** this penalty. Instead, we first validate stability across training coverage and seed variation, and refrain from interpreting the limited clipping results as general recovery gains.

| Item | Fixed Configuration |
|---|---|
| Architecture | Same as P34: width96 complex residual + generation budget, 431,015 parameters |
| Seeds | 20261012, 20261013; two independent runs per seed, initialized from identical random parameters |
| Steps | 16,192 steps per group per seed, total 64,768 updates; no inheritance from P34 weights |
| Optimizer | Adam, learning rate 1e-4, no decay, gradient norm clipping at 5 |
| Sampling | Fully reuses P33's per-seed sampling order; 4 distinct music and 1 speech segment per step, plus corresponding identity references |
| Time Axis | hop128 / FFT256; 80-frame real history warmup excluded from supervision; original spectral supervision region preserved |
| Deployment | Identical frozen Rust/ORT/DSP binary; multi-resolution windowing used during training does not extend to deployment lookahead |

Both objectives are supervised by raw, unclipped reference signals, with identity supervision provided via input:

- `control`: `0.5 × (mean_paired(direct + MR) + mean_identity(direct + MR)) + 0.1 × budget`
  This corresponds to **P33's `mr_add`**, not P33's `control`.
- `compressed_identity_mid`: For each example, take `log1p` of direct and MR outputs separately, then average across roles; paired coefficient remains 0.5, identity coefficient is 0.15; budget term remains `0.1 × budget`.

Error compression and identity downweighting constitute a combined intervention—neither can be isolated or attributed independently. Total training scores are not directly comparable across groups. Evaluation always uses the unclipped, version-consistent, and weight-matched MR v1. Training callbacks are simplified to remove non-optimized branches; verified to produce identical objectives and gradients to P34.

---

## Data and Evaluation Boundaries

Each full trajectory covers 253 music recordings and 193 speech recordings. Statistical reporting is based on actual sampled rows and source time intervals, excluding warmup and overlapping periods; no repeated exposure is counted as new material.

Independent music supervision duration per seed: 331.14 / 330.40 minutes; speech: 14.94 / 14.95 minutes. These represent per-seed coverage and cannot be summed directly, as time intervals between seeds may overlap.

Recheck consists of 89 paired segments: 33 music, 26 music recordings, 56 speech, 54 speech recordings. Training and recheck are isolated by recording and source validation, though these validation segments were previously used in prior diagnostics and must be considered **held-out rechecks, not independent testing**. Continues to preserve P33's genre, regional, mono, and stereo coverage gaps; no new audio sources are introduced, and existing licensing boundaries remain unchanged.

Pre-registration evaluation rules and all input hashes were fixed prior to any training endpoint completion and freezing:

1. Common MR score from unprocessed compressed input, identity DSP, original OLA, and actual Rust/ORT/DSP output.
2. Per-music/speech, per-paired/identity, per-recording breakdown of amplitude, spectral shape, envelope, true dynamic, and all counterexamples.
3. Direct comparison of raw lossless PCM and identity DSP PCM, plus actual inference/synthesis consistency.
4. Full endpoint reporting of native latency, RSS, CPU, per-hop timeout, and paced execution validation.

Pre-registration criteria for candidate advancement: **both seeds** must show superior paired music MR performance over their respective controls, with a negative median per-recording error difference, and **no degradation** in music/speech identity MR or dynamic fidelity across recordings. All failure cases are reported; non-compliance does not trigger automatic replacement of default weights. Full budget convergence is not implied, nor does recheck imply blind listening or device certification.

---

## Execution and Recovery

Entry script: `training/scripts/run_broad_risk_study.py`, with unified result aggregation via:
`training/scripts/analyze_broad_risk_study.py`. Execution depends on a frozen local directory at `source/training`; data, experiment weights, and reports are excluded from Git.

Pre-check directory: `data/runs/broad-risk-p35-preflight-20261021/`, containing 4 steps per group, including music/speech, 44.1/48 kHz, paired/identity, 16 native consistency roles and 16 additional native MR roles. All 16 roles completed, with maximum synthesis error of `2.38e-7`, zero tolerance failure. This pre-check does not participate in formal initialization.

Full Python **646 tests passed**, Ruff lint/format passed; tests use isolated imageio/FFmpeg, without modifying system tools.

A recovery test was performed after interruption at step 64. Correct absolute sampling position was verified at step 65. Final parameters, optimizer state, and training history matched exactly with a non-interrupted run.

Formal directory: `data/runs/broad-risk-p35-20261021/`, frozen with 186 execution dependencies. Registration SHA-256: `c940631597c3981102e9ba4c40d589746595594a119e91c33f7da1e26f05e126`.
Current progress can be viewed via `study/progress.json` and `supervisor.json`; execution logs are recorded in `pipeline.log`. Models, optimizers, and history are saved every 64 steps. Before recovery, protocol, source code, input, and initialization hashes are verified. After full training, a serial pipeline automatically executes freezing, export, native/spectral recheck, baseline, and synthesis. On encountering numerical, integrity, or runtime failures, errors are preserved; no automatic candidate fallback, budget reduction, or retry is performed.

---

## Complete Results (2026-10-22)

Supervision completed normally at **01:11:26 Singapore Time, 2026-10-22**, lasting approximately 10 hours 24 minutes. All four trajectories completed 16,192 steps, totaling 64,768 updates. All endpoints have been re-verified for SHA-256 integrity, continuity of update history, and full evidence hash alignment—no failures recorded.

The table below shows actual Rust/ORT/DSP output, aggregated by recording, using common MR v1. Lower is better.

| Metric | Seed 20261012 | Seed 20261013 |
|---|---:|---:|
| Music: vs. control | 0.384185 | 0.383874 |
| Music: candidate | 0.376018 | 0.379584 |
| Music: vs. control (relative) | **−2.126%** | **−1.118%** |
| Music: vs. unprocessed compressed input | −2.364% | −1.438% |
| Music: wins/losses per recording | 20 / 6 | 20 / 6 |
| Speech: vs. control (relative) | **−4.342%** | **−3.072%** |
| Speech: vs. unprocessed compressed input | −6.031% | −4.727% |
| Music dynamic error: vs. control | +0.696% | +0.064% |
| Speech dynamic error: vs. control | +0.586% | +0.261% |
| Music identity MR: vs. control | +36.550% | +2.338% |
| Speech identity MR: vs. control | −36.788% | +65.186% |

The original OLA music MR degraded by 2.570% / 1.374% and speech by 4.916% / 3.605% relative to control. Thus, gains extend to direct prediction—deployment retains a portion of this benefit. This does not imply that DSP is optimal or that missing information has been perfectly reconstructed. Both seeds improved median music per-recording performance, but neither met the pre-registration requirement of **no degradation in dynamic or identity fidelity**.

Native music amplitude error decreased by 1.780% / 1.038%, log-spectral shape error decreased by 5.103% / 2.243%, and envelope error decreased by 2.123% / 0.485%. Descriptive breakdown of per-band reports: first average across same-recording segments and three analysis resolutions, then across recordings.
- 16–24 kHz: amplitude error dropped by 5.684% / 3.025%, spectral shape error dropped by 15.372% / 7.272%, but dynamic error increased by 2.500% / 0.347%.
- 8–12 kHz: amplitude error increased by 0.588% in first seed.

This suggests the need to investigate high-frequency temporal dynamics and additional generation behavior under identity inputs. It is premature to conclude full high-frequency detail recovery based solely on average spectral shape scores.

Control gradient clipping count: 59 / 45; candidate: 2 / 1. Maximum gradient norm dropped from 404.927 / 367.100 to 6.509 / 6.106. The compression objective reduced training gradient magnitude, but the combined intervention cannot disentangle the contributions of error compression and identity downweighting, nor can it be used to assert convergence.

---

## Absolute Error and Real-Time Boundaries

Music identity output vs. identity DSP PCM RMS, averaged per recording:
- Control: `2.50e-6 / 5.09e-6`
- Candidate: `1.67e-5 / 6.77e-6`

Worst-case single segments: `1.92e-4` (−74.35 dBFS) and `6.49e-5` (−83.76 dBFS). These are actual PCM differences, not unnormalized FFT power. The relative percentage scores above must not be interpreted as audible severity, nor should small absolute values justify post-hoc approval. The identity DSP's own clipping/synthesis errors remain independently reported.

Speech identity output vs. identity DSP PCM RMS, averaged per recording:
- Control: `2.50e-7 / 2.13e-7`
- Candidate: `2.06e-7 / 2.81e-7`
Worst-case segment: `1.78e-6 / 3.31e-6` (−115.00 / −109.60 dBFS). Again, this does not justify auditory judgment or approval.

Paired music relative to original lossless PCM RMS: from ~`0.013498` to `0.013502 / 0.013499`. This minor shift does not constitute a failure in spectral candidate.

Total of **712 native PCM roles + 712 additional native MR roles** completed across four endpoints. Maximum synthesis error: `2.98e-7`, zero tolerance failure. Baseline coverage: **280,000 hops**.

- Maximum processing time: **1.948 ms**; peak P99 across scenarios: **0.808 ms**; zero instances ≥5 ms; zero instances exceeded hop period.
- Paced test: **1 late completion**, 22.23 µs. Cannot be interpreted as zero deadline misses or fully validated continuous output.
- Peak RSS: **31.77 MiB**; for 48 kHz stereo paced process, CPU usage accounted for 19.4%–23.6% of runtime (per logical core, including test tools), not callback overhead.
- Fixed 256-frame buffer remains at **5.333 ms @48 kHz / 5.805 ms @44.1 kHz**, not included in above processing time. This round satisfies 5 ms compute requirement, but does not imply end-to-end latency below 5 ms.

The results confirm that small models achieve repeatable spectral improvements under existing held-out recheck conditions. However, overall fidelity, independent recording generalization, blind listening, and real-device stability remain unverified. Next steps include diagnosing the sources of dynamic and identity degradation, preserving all counterexamples and frozen endpoints, and **not relying on this recheck to override official weights**.

Subsequent work has completed staged real-deployment validation and training gradient diagnostics, and the [P36 Factor Attribution Study](risk-attribution-study.md) has been initiated to disentangle the effects of error compression and identity downweighting. The P35 endpoints remain frozen and preserved as a baseline reference for future comparisons.
