# P33: Broad-Spectrum Supervised Contrasting

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

The subsequent [fixed endpoint diagnosis](broad-spectrum-diagnosis.md) for music has been completed: majority of randomly sampled training music was evaluated, with only marginal gains. Weaknesses in reference scale imbalance and associated fixes, as well as identity gradient competition, were identified. No weight updates were applied.

**2026-10-20.** On the fitting cropping from [P32](spectrum-priority-study.md), multi-resolution supervision was introduced, improving magnitude spectrum, spectral shape, and energy. However, transients, waveform, and generalization remain unimproved. This round verifies whether the supervision gains can extend to broader training coverage and to validation sets isolated by recording, without expanding the model or retraining with P32 weights.

## Completed Results (2026-10-21)

All four trajectories of 16,192 steps, 64,768 updates, 712 native consistency roles, and an additional 712 native MR scoring roles, plus 280,000 hop-time samples, have been completed. Ended at **01:00:02 (Singapore time)** on **2026-10-21**, with a total runtime of approximately **13 hours 35 minutes**. Supervision exit code: 0. Hash validation of 731 evidence files—including frozen dependencies, registration records, sampling sequences, endpoints, completion receipts, and raw reports—was performed. All four full update records are continuous and bounded. Post-hoc analysis was conducted in the local directory `evaluation-20261021/`, without modifying historical registration or scoring.

**No transferable musical spectral advantage was observed; weights were not promoted.** The table below uses identical MR v1, original unclipped references, and aggregates per recording before equal-weight averaging. Values are listed in order: seed 20261012 / 20261013, with lower being better. This differs from P32's dataset, so direct subtraction of absolute scores cannot be used to infer degradation.

| Configuration | Music Original Model OLA | Music Actual Model + DSP | Speech Actual Model + DSP |
|---|---:|---:|---:|
| Unprocessed Input | 0.385123 | 0.385123 | 0.328045 |
| Identity DSP | — | 0.384780 | 0.328045 |
| control | 0.384114 / 0.384142 | 0.383840 / 0.383825 | 0.326358 / 0.325249 |
| mr_add | 0.384482 / 0.384154 | 0.384185 / 0.383874 | 0.322253 / 0.322442 |

The music actual output under `mr_add` shows a **relative increase of 0.090% / 0.013%** compared to control. Magnitude, log-spectral shape, and envelope errors slightly increased, while dynamic error decreased marginally. Out of 26 music pieces, 5/15 outperformed control, and 21/11 regressed—no consistent cross-seed advantage exists. The gains relative to input are minimal: control deployment scores decreased by 0.333% / 0.337%, while `mr_add` decreased by **0.244% / 0.324%**. The original OLA music model under `mr_add` declined by only **0.166% / 0.251%**. This result cannot be attributed solely to DSP cancellation of model gains; the model itself shows weak high-frequency recovery across broader recordings.

For speech, the deployment spectral mean under `mr_add` decreased by **1.258% / 0.863%** relative to control, primarily in magnitude, spectral shape, and energy—though dynamic error increased by **0.184% / 0.175%**. Out of 54 speech samples, 21/33 outperformed control, and 26/13 regressed, with 7/8 tied. The mean gain does not imply overall improvement—this result still fails to establish acceptable restoration quality. While identity errors for music and speech are smaller than control, the input changes are less significant, and this does not replace the need for lossy music restoration.

Waveform serves as auxiliary evidence: the original input PCM RMS error for music was **0.013353316**, while identity DSP yielded **0.013497808**, control yielded **0.013497885 / 0.013497944**, and `mr_add` yielded **0.013497831 / 0.013497843** (per recording average). The difference between `mr_add` and identity DSP output is only about **1.07e-5 / 1.93e-5 RMS**—nearly indistinguishable. Absolute RMS/dBFS, low-frequency components, and per-recording counterexamples are preserved in native reports. Such minor modifications cannot serve as a successful restoration criterion.

Native synthesis maximum error: **2.98e-7**; 712 roles achieved zero synthesis tolerance failure. Current evidence does not support attributing weak recovery to Python/Rust synthesis inconsistency. The budgeted output is not zero, but its mean is insufficient to infer gradient or residual suppression. The last 1,024 steps' shared MR logs show no stable dual-seed advantage; these are batch scores combining music/voice and identity, varying with updates, and not frozen endpoint re-scoring. Convergence or capacity upper bound is not claimed. This recheck fails to distinguish between competitive objectives, optimization stagnation, training distribution shifts, or generalization failure. It cannot be used to assert that the historical baseline was not overfit.

Standard timing results (still using frozen binary, standard QoS):

| Range | Hop Count | p99 Range | Max Computation Time | ≥5 ms | Exceeds Own Hop Period |
|---|---:|---:|---:|---:|---:|
| All | 280,000 | 0.101–0.803 ms | **8.445 ms** | **2** | **3** |
| Subset: 48 kHz stereo paced | 40,000 | 0.783–0.803 ms | **8.445 ms** | **2** | **3** |

The paced mode completed 16 times with lateness. Two instances exceeding 5 ms occurred: one from control seed 20261012 (8.445 ms), and one from `mr_add` seed 20261012 (6.153 ms); the latter seed reached a maximum of 3.056 ms, still above the 2.667 ms hop. Peak RSS: 30.91–32.88 MiB; paced single-core equivalent CPU: 18.84%–19.26% (including initialization and warmup). Fixed 256-frame buffer remains at 5.333 ms (48 kHz) or 5.805 ms (44.1 kHz), with computation and scheduling time separately accounted. Previous P32's additional scheduling diagnostics cannot cover these results, nor can they prove P33 is real-time qualified.

Next steps should first fix the same batch of training and recheck validation samples, verifying effective high-frequency updates, budget, and identity gradient competition, separating "what could not be learned during training" from "where training gains fail to generalize." Then, targeted corrections can be decided. The P32 fusion phase findings remain valuable, though fusion is no longer considered the sole bottleneck for P33. No new round of training or model/DSP modifications were initiated. Blind listening and real device output remain unverified.

## Fixed Training Protocol

| Item | Fixed Setting |
|---|---|
| Model | 431,015 parameters, width 96, relative coordinate complex residual, full residual budget, phase input disabled |
| Configuration | control: original direct spectral supervision + budget auxiliary; mr_add: additional MR v1 |
| Initialization | full training from scratch; all tensor initializations identical across seeds |
| Seeds | 20261012, 20261013; two configurations, four trajectories |
| Budget | 16,192 steps per trajectory, 64,768 total; final endpoint as primary comparison |
| Optimizer | Adam 1e-4, no decay, gradient norm capped at 5 |
| Batch | four distinct music pieces + one speech segment, plus corresponding clean identity; both roles equally weighted |
| Data | reused P26's frozen broad-coverage sequences; same update order for both configurations under same seed |
| Supervised Region | real 80-frame pre-warmup, scoring window `[89*128, 32768-128)` |

MR v1 uses the same FFT256/1024/2048, weights, floor, and unclipped reference as P32. No per-sample waveform optimization terms, and no adjustment to loss, thresholds, or DSP based on current validation tracks. Budget auxiliary weight remains at 0.1. Old deployment proxy scores are only recorded and not included in either optimization objective. 20% speech is carried over from prior experiments; final product data proportion is not redefined. Training continues on the previously validated small-model CPU path, single-threaded.

All four trajectories completed and endpoint hashes frozen before validation scoring begins. No mid-process selection, early termination, or isolated retraining occurred. Only due to finite count, integrity drift, or runtime errors were any stops made. Completion of budget does not imply convergence. A recovery state is saved every 64 steps; recovery reads from the frozen sequence by absolute step number, and cannot restart from the first batch.

## Coverage and Isolation Plan

Below are the pre-registered sampling coverage; all four trajectories have now completed their full sequence and can be used as actual training exposure. This does not equate to full time coverage of the original music library. Independent supervision time is encoded by merging raw source time, sampling rate, channel, and fragment overlap, excluding pre-warmup; paired/identity is not double-counted.

| Seed | Music Recording | Independent Music Supervision (minutes) | Speech Recording | Independent Speech Supervision (minutes) |
|---|---:|---:|---:|---:|
| 20261012 | 253 | 331.14 | 193 | 14.94 |
| 20261013 | 253 | 330.40 | 193 | 14.95 |

Each trajectory sampled 64,768 music and 16,192 speech segments, involving approximately 40,000 paired rows. Covers 9 training encoding levels, 44.1/48 kHz, mono/left/right/mid/side. Channel views from different starting points are processed independently and not combined into stereo. Existing native lossless materials and licensing records are retained; no new sources such as MUSDB, Slakh, or lossy libraries are introduced. Original audio, paired data, and experimental weights remain in the ignored directory.

At registration, checks were performed on recording, source checksum, work/split group cross-division isolation; training input windows (including pre-warmup) avoid all development and shifted protected intervals from P26. On first PCM file read, manifest checksum is verified, and checks for finite values, mono geometry, length, and sample rate are performed. No claim is made that all random crops were individually read. RAM crop caching is limited to 128, and random crop NPZ writes are not continuous.

Unified recheck uses the existing validation set of **33 music fragments / 26 recordings** and **56 speech fragments / 54 recordings**, all isolated from training recordings. However, these validation recordings were reused in prior experiments and must be labeled **held-out recheck**, not a fresh, independent test. They cannot prove absence of development set selection bias. This round does not open the test set.

Music validation region tags include: China 4, Japan 3, South Korea 4, Europe 2, North America 10, Unknown 3. Genres: ambient 2, blues 3, classical 4, electronic 4, folk 4, jazz 1, rock 5, unlabeled 3. This validation list lacks pop/country/hip-hop/funk and mono music. It cannot be used to claim full genre, regional, mono, or true stereo synchronization generalization. These tags refer to reviewed source libraries, not regional popular styles.

## Evaluation Order and Acceptance

1. All four final endpoints are frozen; all configurations are re-scored against the same target.
2. Compare input, identity DSP, original OLA model, and actual Rust frontend → ORT → DSP outputs. Primary focus: MR spectrum and magnitude, spectral shape, envelope, and dynamic behavior across scales/bands. Report per-recording and per-fragment averages, and all counterexamples.
3. Secondary checks: original lossless PCM, phase, low-frequency, high-frequency, and identity perturbation. Minor waveform degradation should not be used to negate spectral gains, nor should overall scores mask transient or weak energy degradation.
4. 712 full native consistency roles and an additional 712 native MR scoring roles. Retain proxy and prediction playback as auxiliary; proxies cannot replace native outputs.
5. Total of 28 processes, 280,000 hops: mono/stereo, 44.1/48 kHz, repeated stereo and paced. Use frozen standard binary; new scheduling diagnostics are independent experiments and do not mix with prior standard benchmarks. Report computation, per-hop timeout, fixed 256-frame buffer, CPU/RSS.

No automatic promotion of weights. Real callback/device, continuous output, and blind listening remain to be independently verified. Local scores cannot substitute these evidences.

## Implementation and Validation Launch

Entry point: `training/scripts/run_broad_spectrum_study.py`. Full Python test suite passed **632 cases**, Ruff lint/format passed. Uses a frozen `imageio-ffmpeg` test path; no system FFmpeg or existing material modifications. Added tests cover training/validation list separation, sampling ratio, recording deduplication, protected time intervals, and recovery indexing.

Pre-check directory: `data/runs/broad-spectrum-p33-preflight-20261020-v2/`. Two configurations, four steps, two sample rates, music/voice, paired/identity; 16 native consistency roles and 16 additional native MR roles completed. Synthesis maximum error: **2.38e-7**, zero tolerance failure. Pre-check only validates integration flow, not for selection. Initial version incorrectly used training-specific coverage statistics during registration; upon encountering validation rows, it correctly rejected. Corrected to use independent validation list statistics, preserving original training isolation rules. Initial version did not trigger parameter updates.

Formal run directory: `data/runs/broad-spectrum-p33-20261020/`. 180 execution dependencies frozen, SHA-256 registration: `d929c79add4b2efe1fa699ca3929719b9860d3f042e684173aa5754f40505611`. Entry point: `supervisor.json`, `pipeline.log`, `study/progress.json`. Training, all endpoint freezes, unified recheck, MR native scoring, and standard benchmarking are executed in sequence by background supervisor process; failures are retained, no automatic budget reduction or model switch.

After completion of scheduling diagnostics, the official background training has finished (historical supervisor PID 44897). The pre-check four steps or P32 fitting weights were not used for initialization of the official training.

## P32 DSP Stage-by-Stage Localization

Additionally, 20 pairs of 6 endpoints, 240 roles, were re-verified with actual Rust trace analysis from P32, evidence stored in `data/runs/spectrum-stages-p32-20261020/`. All stages used clean, unclipped reference scoring, including identity. Actual frontend and ORT predictions were used to avoid attributing training frontend differences to DSP. Final stage output vs. real Rust PCM had a maximum error of **2.98e-7**. No deployment processing chain modifications were made.

Music MR error, reported per seed:

| Stage | control | mr_add |
|---|---:|---:|
| Actual Predicted OLA | 0.190991 / 0.190088 | 0.182680 / 0.182608 |
| OLA after band fusion | 0.195040 / 0.194216 | 0.190431 / 0.190258 |
| Unclipped gating diagnosis | 0.195040 / 0.194216 | 0.190431 / 0.190258 |
| Final Real Output | 0.194972 / 0.194151 | 0.190388 / 0.190225 |

The primary divergence occurs in **band fusion**; gating changes are negligible. Limiting amplitude slightly benefits `mr_add` in average spectral score. Actual deployment applies separate dry/wet clipping before cross-fading; the "unclipped gating" in the table is a diagnostic counterfactual, not a production-level clipping sequence. The original model already includes a budget header, so this analysis does not isolate the budget header. This analysis identifies stages, not proving that removing fusion would improve overall performance.

Future work should verify fusion masking, junction phase cancellation, and low-frequency fidelity before deciding whether to modify fusion or add deployment consistency supervision. P33 retains fixed DSP in this round to avoid confusing loss comparisons.

## Real-Time Tail Localization

Among the original 91 ≥5 ms events in P32, the longest latency phase was ORT (77 times), frontend (8 times), and DSP (6 times); a 45 ms extreme at 44.96 ms occurred within the ORT phase. This measures wall-clock time, including preemption, and cannot be used to claim a pure 44.96 ms CPU inference. ORT was fixed with intra/inter-thread isolation; no direct thread configuration error was identified.

It was discovered that the prior independent benchmark only requested USER_INITIATED QoS, while the official worker defaults to macOS audio time-constraint scheduling. A new **diagnostic-only, explicitly enabled** control was added: identical new binary, four P32 control/mr_add endpoints, same 48 kHz stereo paced input, each model tested under QoS and audio real-time scheduling, 80,000 hops total. Interleaved test order, recording acceptance, host load, output consistency, and all timeouts were logged—without overriding prior data.

Both modes pre-allocated timing arrays; real-time requests allowed only paced execution, and thread priority was restored after loop exit. No changes were made to the official worker, model, or DSP. The new/old binaries do not constitute a single-factor control; only the two scheduling modes within the same binary can be directly compared, and still not a physical device certification.

The control is complete. Evidence is stored in `data/runs/scheduling-p32-20261020/`. No concurrent training or build tasks occurred during execution; other application and system loads were uncontrolled, and per-process load average is preserved. All four processes requesting real-time scheduling were accepted by the system.

| Mode | Hop Count | p99 Range | Max Computation Time | ≥5 ms | Exceeds 2.667 ms | Paced Completion Lateness |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary QoS | 40,000 | 0.807–0.819 ms | 1.735 ms | 0 | 0 | 1 |
| Audio Time-Constraint | 40,000 | 0.732–0.751 ms | 0.856 ms | 0 | 0 | 0 |

Final audio differences across models in both modes were **exactly zero**. Peak RSS: 31.39–31.67 MiB; single-core equivalent CPU (including initialization/warmup): ordinary QoS 21.91%–22.44%, real-time 14.88%–21.41%. This is not a net increase in RSS or overall system CPU percentage. Fixed buffer remains at 256 frames, 5.333 ms at 48 kHz.

The real-time tail latency is smaller, but **ordinary QoS does not reproduce P32's 45 ms tail**, so the 91 prior timeouts cannot be attributed solely to lack of real-time scheduling. The issue is not resolved. The original P32 results remain valid. Future work requires load stress and real worker/callback measurement. This round's real-time benchmark and additional scheduling comparison are reported separately; no better result is used to cover a worse one. The diagnostic binary release build, Clippy, formatting, and actual runtime passed. The protection check against rejecting non-paced real-time requests also passed.

The completed supervisor exit code is 0; all frozen endpoint and evidence checks are recorded separately from the failed quality screen.
