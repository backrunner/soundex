# P21: Residual Sensitivity Calibration and Model Scale Comparison

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

In response to the issue in [P20 Full Results](restoration-head-study.md) where approximately 87% of updates triggered gradient clipping and where the fixed training error remained above initialization, this round first calibrates the optimization scale before comparing model capacity.
It is incorrect to treat a model with difficult optimization under the same learning rate as having reached a proven capacity ceiling.

User permission to lift the upper bound on model size was granted on 2026-10-15, with a separate study [P22: Latency-First Scaling](latency-capacity-study.md) now in place.
This page, P21, is now frozen, preserving the original budget and continuation conditions; new expansions do not retroactively modify ongoing experiments.

## Current Results

At 15:31 SGT on 2026-10-15, the SGT (Sensitivity Gradient Tuning) calibration process ended normally with status `calibration_rejected`.
All three scaled configurations and both seeds failed pre-registration checks and did not initiate the 8,192-step phase.
While paired error recovery showed a slight improvement, the main training pools for both common and full training sets did not fall below initialization.
The calibration significantly reduced gradient clipping, but audio fidelity degradation remained evident.
Future, larger-scale results under the same budget are detailed in [P22 Final Report](latency-capacity-study.md).
The execution and pre-check data presented below are historical records and do not represent ongoing training.

## Changes and Fixed Comparisons

The residual coordinate was changed from `2 * tanh(z)` to `2 * tanh(z / 64)`.
This reduces the derivative near zero, but does *not* scale the final amplitude by a factor of 1/64:
the maximum coordinate range remains ±2, and the complete absence of frequency bands still generates energy.
The value `1/64` was fixed prior to this round, based on a diagnosis of P20's initial gradient being amplified by 88–89×;
it was not derived from a search of current development recheck results.
Whether output amplification risk has been controlled remains to be determined by subsequent training.

All causal TCN, context update, DSP, main loss, clean-complex auxiliary objectives, and weights were retained from P20.
No additional frames or output buffering was introduced.
Checkpoints explicitly save width, head, and slope; exported models are restored per configuration, using original defaults from prior P20 checkpoints.

| Calibration Configuration | width | Parameters | Residual Slope | Adam Learning Rate |
| --- | ---: | ---: | ---: | ---: |
| Anchored Control         | 144 | 807,398 | Original Head  | 1e-4 |
| Unscaled Control         | 144 | 824,658 | 1 | 1e-4 |
| Lowrate Control          | 144 | 824,658 | 1 | 1e-4 / 64 |
| scaled96 | 96 | 430,050 | 1/64 | 1e-4 |
| scaled144 | 144 | 824,658 | 1/64 | 1e-4 |
| scaled224 | 224 | 1,748,578 | 1/64 | 1e-4 |

The largest complex model is approximately 2.12× the parameter count of the 144-width baseline, still below 2M.
For identical width and seed, shared main backbone initialization hash; tensor shapes differ across widths, so element-wise initialization weights are not claimed to be identical.
The lowrate and scaled144 configurations separate the effects of learning rate adjustment from output parameterization adjustment.

## Two-Phase Execution

All configurations and subsequent endpoints were pre-registered prior to training initiation.
The 253 music recordings, 8 speech recordings, and 2,088 cropped segments from seeds `20261012` and `20261013` with identical sampling order were inherited from P20.
Each step consists of 4 music + 1 speech, with each material paired with both paired and identity counterparts, splitting risk evenly—this maintains the diagnostic sampling ratio and does not alter the final product's speech-to-music training proportion.
Adam uses no weight decay, with gradient norm clipping at 5.

**Calibration Phase:** Six configurations × two seeds × 512 steps, totaling 6,144 updates.
Simultaneously evaluated on 320 cropped common and 2,088 cropped full training pools, monitoring recovery, original audio degradation, combined loss, gradient clipping ratio, and per-scale export cost; development sets were not scored, and no seed or endpoint was selected as optimal.

The pre-defined continuation criteria require all three scaled configurations and both seeds to satisfy:

- Main, paired, and combined losses on common and full training sets all improved over initialization; full training set main and paired losses also decreased;
- Identity performance on both common and full training pools must not degrade relative to the anchored control (within numerical tolerance 1e-6);
- Cropping ratio must not exceed 25%, serving as a diagnostic upper bound during optimization, not as a quality acceptance standard;
- Parameter count must not exceed 2M, with per-image memory under 8 MiB; single-threaded, stereo diagnostic path p99 latency per hop must remain below 48 kHz.

If any configuration or seed fails, the pipeline writes `calibration_rejected` and halts—no automatic transition to long training, and no subset is packaged as a full capacity conclusion.
This threshold is a training process filter, not a release or perceptual quality benchmark.

**Post-Training Phase:** Automatically executed only if the above screening passes.
For `anchored144` and `scaled96/144/224`, two seeds, each is retrained from a common identity baseline for 8,192 steps (65,536 updates), with fixed endpoints at 2,048 and 8,192.
No weights are migrated or selected from the calibration phase.
After full training completion and freezing of 16 endpoint hashes, 89 development paired segments and 40 shifted segments are scored.
Initial control plus 16 endpoints, totaling 3,026 Rust role replays.

Each scale is compared against the anchored, 96→144, and 144→224 budget-matched results.
Per-clip and per-recording metrics are aggregated separately for music and speech, preserving per-fragment degradation and worst-case examples.
These data are reused development rechecks and cannot prove absence of overfitting.
Recovery main error, actual DSP high-frequency spectral shape, and transient behavior must outperform matching controls and initial DSP, and original audio fidelity, energy, and low-frequency response must not deteriorate.
These results still cannot replace independent new recordings, blind listening, or continuous device validation.

## Completed Prechecks

Training source code `f441f8e` is frozen.
Prior to actual training, each of the three scaled configurations completed 16 pre-registered training batches.
Non-zero weight ONNX continuity, silence, reset, 44.1/48 kHz, and mono/stereo were verified.
For each configuration, five pairs of training materials were replayed in actual Rust execution, totaling 30 roles, with maximum proxy error `2.38419e-7 < 2e-5`.
This is not a result of the 512-step screening or a conclusion on cross-recording quality.

On local Apple M5 Max, single-threaded CPU ORT, with Python causal FFT frontend, 1,000-hop stereo timing:

| Scale | ONNX | p99 / hop | Max / hop | ORT Independent Process RSS Increment |
| --- | ---: | ---: | ---: | ---: |
| 43.0 × 10,000 | 1,759,277 B（1.68 MiB） | 0.1613 ms | 0.1800 ms | 12.78 MiB |
| 82.5 × 10,000 | 3,337,766 B（3.18 MiB） | 0.1993 ms | 0.2147 ms | 20.11 MiB |
| 174.9 × 10,000 | 7,033,446 B（6.71 MiB） | 0.2991 ms | 0.3133 ms | 28.50 MiB |

RSS increments are measured after model loading and 100-hop stereo warmup, relative to a fresh-process baseline after ORT initialization but before model loading;
includes session, allocator, and input/state buffers, excluding Python FFT frontend, PyTorch, or full Rust.
These are snapshot increments, not peak values. All graphs remain diagnostic ABIs with external context input and cannot be directly considered formal SoundEx artifacts; timing does not represent fixed output latency, callback availability, or actual end-to-end sound card performance.

The median gradient across 16 prechecks was 0.751 / 0.875 / 0.993, triggering clipping 1 / 2 / 3 times respectively.
These statistics cannot be directly compared to the P20 full 16,192-update performance due to different budget contexts; the 512-step screening would provide matched results for each control.
Unit tests independently verified initialization Jacobian scaled by 1/64, identity output invariance, preserved expressible output range, finite gradient under silence, and checkpoint recovery consistency between old and new.
All Python tests passed: 492 passed, Ruff checks passed.

## Local Execution

At 15:00:35 SGT on 2026-10-15, the background calibration has started, with supervisor PID `16646`.
Latest status is available in `data/runs/capacity-p21-20261015/pipeline.json` and `study/progress.json`;
specific training updates are in `study/calibration/progress.json`;
post-screening status is in `study/full/progress.json`.
No default weights were replaced.

The local root directory contains the frozen source code at `source/training`, `preflight`, `pipeline.log`, and `study`.
`study/registration.json` and the registration files for both phases contain fixed inputs, source files, sampling, configurations, and endpoints;
`calibration-decision.json` records the rationale for continuation or stop.
Training audio, paired data, and checkpoints are not pushed.

```bash
PYTHONPATH=data/runs/capacity-p21-20261015/source/training \
training/.venv/bin/python \
  data/runs/capacity-p21-20261015/source/training/scripts/run_capacity_study.py \
  --p20 data/runs/head-p20-20261014/study \
  --output data/runs/capacity-p21-20261015/study
```

The command uses exclusive locking; if a run is already active, it will not restart.
Breakpoint recovery preserves model configuration, optimizer, RNG, and sampling.
Post-completion validation files are required for continuation; source files or input drift are rejected.

The registered seeds are `20261012 / 20261013`. Each configuration replays 5 paired training examples in the stated preflight; those examples do not establish held-out quality.
