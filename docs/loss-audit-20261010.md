# Loss Design Audit and Controlled Optimization (2026-10-10)

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

Original 200 epochs of training have been completed. This iteration addresses identified deficiencies in phase and waveform fidelity during validation, modifies the training objective, and conducts a separate short-range continuation training. The original best epoch 168 is retained; new versions of the loss cannot be directly compared to prior values.

## Confirmed Issues

The original objective directly summed dB amplitude error, dimensionless phase error, and L1 error of linear PCM, without proper scaling. This led to a severely weakened waveform supervision: when evaluating the same 64 training samples (4 batches, 16 per batch) at epoch 168, BatchNorm was fixed in evaluation mode, and gradients for all generator parameters were backpropagated individually to compute:

| Weighted Term | Old Objective Gradient L2 Mean | New Objective Gradient L2 Mean |
|---|---:|---:|
| Missing high-frequency amplitude | 66.2119 | 2.8252 |
| Phase | 3.7809 | 4.0069 |
| Low-frequency identity regularization | 3.6962 | 0.1780 |
| Crossover-band continuity regularization | 1.6195 | 0.0805 |
| Waveform | 0.0004574 | 2.9851 |

This is not a GAN gradient audit, nor does it represent a guarantee over the full training distribution. It serves to identify scale imbalances and calibrate initial weights. Small-scale gradient balancing does not imply perceptual improvement. Actual training uses BatchNorm batch statistics, so gradient audits must also be reported under training mode—eval mode gradients cannot be used as a proxy for real training behavior. The original objective used per-batch effective frequency point normalization, which assigned different training weights to samples with different bandwidths, while validation averaged across rows equally.

Rechecking the same training samples under **training mode** yields:

| Weighted Term | Old Objective | New Objective |
|---|---:|---:|
| Missing high-frequency amplitude | 41.2248 | 2.6464 |
| Phase | 7.6588 | 1.6849 |
| Low-frequency identity regularization | 4.3980 | 0.2318 |
| Crossover-band continuity regularization | 1.0464 | 0.0523 |
| Waveform | 0.0005131 | 0.6741 |

This confirms that scale issues persist in actual training: the waveform term in the old objective was ~80,000 times weaker than the high-frequency amplitude term, while in the new objective, the gap is reduced to ~4×. Regularization terms remain small and do not require strictly equal gradients. Both modes used the same parent weights, fixed seed, and training data—no test set was used.

## Objective v2

The new objective first normalizes per-sample effective frequency points, then takes the batch mean, aligning with sampling probability and per-row validation. Terms with zero bandwidth return zero. Current controlled configuration is:

```text
total = high_dB / 20
      + 0.2 * phase / 2
      + 0.1 * low_identity_dB / 20
      + 0.05 * crossover_dB / 20
      + waveform_relative
```

20 dB is a defined reference scale—not restricting amplitude terms to [0,1]. Phase uses a periodic `1 - cos(delta)` with range [0,2]; it is energy-weighted and excludes phase values below -60 dB relative or below -120 dB absolute, which are considered meaningless. The initial candidate phase weight was 1, but empirical gradient measurement was ~20.03, indicating over-weighting; the final value is kept at 0.2.

The waveform term uses actual deployment-level band/phase blending and causal Hann overlap-add. FFT256/hop128, 4-frame continuous, supervises only 384 samples with full overlap support, excluding the 128-sample ends due to cropping. Each sample is normalized as:

```text
scale = max(L1(degraded_OLA, reference_OLA), 0.01 * RMS(reference_OLA), 1e-4)
waveform_relative = mean(L1(predicted_blended_OLA, reference_OLA) / detach(scale))
```

The denominator derives only from input and reference; generators cannot reduce the loss by modifying the denominator. When neither lower bound is triggered, the compressed input value remains 1 and the reference is fully restored to 0; when either bound is triggered, the dry base can fall below 1. These lower bounds suppress gradients for silence and near-identity inputs, meaning gain invariance is not strictly enforced at very low levels. The original PCM L1 is still recorded separately.

Low-frequency identity and crossover-band terms are regularizers: deployment preserves effective low-frequency content, and these cannot be used to replace post-deployment low-frequency energy or residual checks. The training proxy does not approximate Rust's dynamic loudness matching, clipping, or dry/wet gating. Final quality must still be verified through real Rust stream processing.

The original Objective v1 formula and metrics remain compatible. Historical configurations without version tags continue to execute v1 logic and do not silently alter the training definition of prior checkpoints.

Phase mixing previously encountered floating-point zero vectors at opposite phase, 50% weight. The current implementation explicitly avoids undefined `atan2(0,0)` branches and instead computes gradients along the deployment fallback path. For 100,000 boundary samples, forward pass matches the old implementation bit-for-bit, and all gradients are finite. The current PyTorch 2.12.1 implementation on CPU/MPS does not reproduce non-finite gradients, confirming this is a robust boundary behavior safeguard—not a proven cause of failure in prior training. The 8-epoch continuation training was run before this safeguard was implemented.

## Controlled Continuation and Selection

- **Configuration**: [`balanced_continuation.yaml`](../training/configs/balanced_continuation.yaml), 8 epochs, 2048 samples per epoch, batch size 16, learning rate 2e-5, 128-step warmup.
- Inherits only the generator from original native data training at epoch 168. Optimizer, learning rate trajectory, and discriminator are reinitialized; GAN components are disabled to isolate changes in supervision objectives—this is not a GAN ablation proven optimal.
- No changes to model architecture, FFT/hop, recipe, data list, source ratio, or 896-row fixed validation set. No diagnosed test samples are included in training.
- New checkpoint is selected based on the new objective total, with high-frequency amplitude ranked by equal division; selection strategy is bound to configuration hash. All original metrics are preserved, and independent judgment is required to determine if the new checkpoint regresses relative to the parent.
- `--resume` requires strict configuration consistency. Target modifications must begin with `--initialize-generator-from` to start a new training trajectory; model, feature protocol, data, and validation members are verified, and parent checkpoint SHA-256, parent source code, and initialization chain are recorded. Loading is rejected if parent parameters are non-finite.
- Local run: `data/runs/balanced-native-continuation-20261010-1935/`; frozen source code is `b4ed52c`, with an independent detached worktree; original epoch 168 is fully preserved.

## Validation Scope

Testing covers the old objective formula, new per-sample weighting, baseline relative error, gain scaling, silence/identity finite gradients, periodic phase, checkpoint strategy, and continuation compatibility. Formal release requires additional validation: continuous streaming, perceptual and blind listening, real device, and cross-platform testing. A drop in training loss does not imply these thresholds have been met.

Gradient details, parent weight comparisons, new experiment logs, and weight preservation are stored locally in `data/` and not pushed. Raw training audio is not shared. New weights inherit the parent's data provenance chain and still require separate review before release.

## Completed 8-Epoch Comparison

8 epochs, 1024 updates completed. Among 8 candidate continuations, the new objective total was minimized at epoch 6. CPU float32 single-threaded reevaluation of parent and candidates was performed; all 896 row IDs matched exactly, avoiding micro-differences between CPU and MPS:

| Metric (lower is better) | Parent epoch168 | Continuation epoch6 |
|---|---:|---:|
| New objective total | 1.568438 | 1.571623 |
| High-frequency amplitude (dB) | 10.871966 | 10.970338 |
| Phase | 0.812382 | 0.799784 |
| Original waveform L1 | 0.006845969 | 0.006844864 |
| Relative waveform error | 0.913996 | 0.913265 |
| Low-frequency identity regularization | 3.530340 | 3.569211 |
| Crossover-band regularization | 4.781529 | 4.806951 |

Phase improved by 1.55%, waveform improved by ~0.016%, high-frequency error increased by ~0.905%, and new total increased by ~0.203%. This is not a holistic improvement and is insufficient to justify long-term continuation or default replacement.

A further evaluation of the audio from this batch was processed through a real Rust stream: 89 rows, 80 recordings were selected by fixed hash, covering all recordings, music genres, roles, and corpus/codec/rate combinations in the 896-row fixed validation set. This is a development validation and not an independent test set or full-scale release evaluation. Both parent and continuation weights used the same Rust binary:

| Deployment Metric | Parent Weight | Continuation | Paired Change & Audio Clustering 95% CI |
|---|---:|---:|---|
| SI-SDR (dB, higher is better) | 28.2614 | 28.2813 | +0.0199 [0.0085, 0.0353] |
| High-frequency LSD (dB, lower is better) | 20.2975 | 20.5187 | +0.2212 [0.1109, 0.3464] |
| Low-frequency retention LSD (dB, lower is better) | 3.9401 | 3.8501 | -0.0900 [-0.1338, -0.0532] |

Relative to compressed input, the continuation shows a ~0.426 dB reduction in SI-SDR, so the small gain over the parent cannot be described as improved overall waveform fidelity. LSD is sensitive to weak energy bands and must be paired with energy/residual metrics and blind listening. All candidate outputs are bounded, with peak values not exceeding 0.95. Both chunking modes and offline outputs match sample-by-sample.

This coverage includes 33 music rows / 26 recordings and 56 speech rows / 54 recordings. The overall mean does not reflect training sample proportions or expected playback content distribution. Music and speech are reported separately to avoid masking issues in one domain by a high speech proportion:

| Paired Change (continuation minus parent) | Music | Speech |
|---|---:|---:|
| SI-SDR dB | +0.0398 [0.0140, 0.0738] | +0.0082 [0.0024, 0.0160] |
| High-frequency LSD (dB) | +0.1386 [-0.1282, 0.4406] | +0.2699 [0.1760, 0.3613] |
| High-frequency energy error (dB) | +0.2018 [0.0214, 0.3718] | -0.0003 [-0.0641, 0.0509] |

Music shows minimal waveform benefit, with a rise in high-frequency energy error—another reason not to promote this as a default weight.

The continuation ONNX is 3.16 MiB, SHA-256 `55475b40e9a47c1a55c4802c979414fc3a4e7bd147685abd6b27b2cf4ada2fd4`, and 13 PyTorch/ORT/Rust consistency cases passed. **The original epoch168 is retained as baseline; the continuation weights are only considered experimental candidates.** The scale and per-sample weighting adjustments in Objective v2 have empirical justification, but specific weights and long-term performance have not been proven optimal. 8 epochs are insufficient to disentangle the effects of GAN components, optimizer resets, and BatchNorm state changes. Future work should first conduct single-factor controls and deployment phase/bandwidth error analysis.

## Output Safety and Scheduling Comparison

The original serial/offline dry path had no gain limiting, potentially outputting unclipped PCM after decoding. The current serial processor now applies gain limiting once to both dry and enhancement paths before cross-fading, and enforces strict peak protection—no additional soft limiting is applied to already-limited enhancement paths. The asynchronous safety mixer retains its own final limiting. Paths below the soft limit threshold (default 0.855) remain bit-identical; peak excursions are soft-limited even if they do not exceed 1.0. Single/dual-channel over-amplitude input and consistency with offline output are verified. Real-time callback zero-allocation and fault fallback tests pass.

Work thread experiments retained the original fixed 256-frame latency, without adding queue or buffer, and does not trigger OS wake-up signals from callbacks. All timing measurements were performed serially after training and gradient audits, using the same hash-bound music PCM; the producer is a host clock simulation, not actual audio hardware.

- **Candidate A**: Wakes 300 microseconds early, 50 microseconds polling, native real-time thread spins up to 75 microseconds. Stereo improvement observed, but mono and 48 kHz conditions not fully passed—rejected.
- **Candidate B adopted**: Wakes 500 microseconds early, 50 microseconds polling, native real-time thread continues to disable idle spinning; regular thread maintains at most 200 microseconds spin. In 30-second separate tests, the parent weights showed 0/0 late deliveries (stereo and mono), down from 17/21 in prior tests.
- Repeated 30-second full tests on parent weights: 13/1/2/2 (44.1 mono, 44.1 stereo, 48 mono, 48 stereo), at 0.126%/0.0097%/0.0178%/0.0178% respectively. **44.1 kHz mono still exceeds the 0.1% threshold and cannot be declared acceptable as a whole.** Combining better conditions from different tests to create a single pass is not permitted.
- The continuation weight passed 9/1/3/0 on the full four conditions in short test, meeting all short-term thresholds, but this weight failed the overall quality comparison and cannot be used to replace the parent.

Parent weight retested software latency budget (fixed latency plus max callback overhead) was 5.827/5.827/5.424/5.360 ms, callback p99 at 4.458/4.292/5.166/4.291 microseconds. All native real-time requests were accepted; no output errors, non-finite values, queue overflow, or callback overruns occurred, and peak values remained within limits. The worst-case consecutive delay was one hop, and the dry path backfill maintained temporal continuity and output continuity. Request acceptance does not imply the OS will never downgrade the thread.

These are development machine diagnostics and have not undergone 30-minute wall-clock, real audio card, x86_64, blind listening, or power mode validation. Latency, continuity, and enhancement availability remain independent thresholds; mono edge cases and host scheduler jitter require further optimization and long-term validation.

## Model Resource Comparison

On the same Apple M5 Max, CPU ORT single-thread, 256 hop pre-warmup, 10-second wall-clock per condition, serial measurement, both models have 806,276 parameters and 3,309,786 bytes (3.16 MiB). Below are synchronized full-processing times, not asynchronous adapter overhead:

| Condition | Parent/Continuation p99 (ms) | Parent/Continuation RTF (single-core time budget) |
|---|---:|---:|
| 44.1 kHz mono | 0.552 / 0.542 | 0.1766 / 0.1758 |
| 44.1 kHz stereo | 1.025 / 1.073 | 0.3358 / 0.3364 |
| 48 kHz mono | 0.541 / 0.541 | 0.1922 / 0.1915 |
| 48 kHz stereo | 1.020 / 1.018 | 0.3650 / 0.3648 |

No deadline misses or non-finite outputs were observed. `/usr/bin/time -l` peak RSS was 40,878,080 / 40,845,312 bytes (~39 MiB). This does not indicate statistically significant resource differences—only that no additional cost was observed. The parent model's full asynchronous four-condition test took 120.16 seconds wall-clock, 58.66 seconds user, 1.68 seconds system, averaging ~0.50 core CPU, peak RSS 38,109,184 bytes (~36.34 MiB); this CPU includes both producer and host test framework and cannot be considered pure model overhead.

Synchronization benchmark exit code 2 was due to `diagnostic_run`, unrecorded power mode, and insufficient release duration—this was not rewritten as a formal release pass. Original timing, resource receipts, model hashes, and per-condition reports are preserved locally in `data/reports/objective-audit-20261010/`.

## Source Receipt Fix

This audit also uncovered that the original training checkpoint had `source_git_sha` as `dd0d15ab`, while the actual frozen source code receipt was `abe0e497`: the archive directory was nested within the main repository, and the old implementation would read up to a changed main repository HEAD. A point-by-point comparison of 106 version-controlled training files in the original archive against `abe0e497` Git archive showed zero differences.

The original checkpoint and ONNX are preserved in raw bytes and hashes, with clear distinction between declared metadata and actual frozen versions. Now, archives read the full frozen version from `source-receipt.json` and record the receipt hash; archives without receipts no longer claim the main repository version. Independent Git worktrees read their own HEAD. This continuation uses the latter approach, and the checkpoint has been verified to bind to `b4ed52c`.

## Next Steps and Experiments

Future work is now focused on high-frequency spectral inpainting and model addition of DSP to approach original lossless audio. The new Objective v3 formula and processing proxy boundaries are detailed in [Objective Design](inpainting-objective.md). Training, actual Rust comparison, resource, and continuous output results are detailed in [Follow-up Experiment](optimization-followup-20261010.md). Historical values of the old loss remain unchanged in this document and are not directly comparable to the new total.
