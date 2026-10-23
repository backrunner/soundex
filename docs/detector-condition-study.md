# P38: small-model detector-condition training controls

**Completed 2026-10-23 at 00:14:38 Singapore time:** all training and automatic
reviews finished. Fixed detector conditions regressed music spectra in both seeds;
neither output path passed the registered research screen. No weights were promoted
and this fixed-condition route will not be expanded unchanged.

Following [P37](fusion-boundary-audit.md), this study isolates dependence on explicit
cutoff/active conditions. It does not change the main loss, enlarge the model or use
DSP output as a supervision teacher.

## Fixed configuration

| Item | Setting |
| --- | --- |
| Control | P36 compressed: direct lossless spectrum plus raw-OLA multiresolution objective, identity weight 1.0 |
| Intervention | Same objective; normalized cutoff=0.5 and active=1 inside the model, retaining the real sample-rate condition |
| Model | Same complex-residual structure, width 96, 431,015 parameters |
| Initialization | Identical tensors between arms per seed; P36 initialization hashes checked; trained from scratch |
| Budget | Seeds 20261012/20261013, 16,192 steps per arm/seed, 64,768 total |
| Data/sampling | Full P36 schedules: four music plus one speech excerpt and corresponding identity roles; no new sources |
| Optimization | Single-thread CPU, Adam 1e-4, gradient norm cap 5; 80-frame no-gradient model warmup |
| Assessment | Freeze every endpoint before common scoring; reused recording-isolated held-out material, not independent generalization evidence |

Condition transformation occurs in `RestorationHead.forward_details`, starting at
the first warmup frame and included in the exported ONNX graph. The policy is stored
in checkpoint configuration and export metadata. It is not a Python-only training
change. Rust detection, gating and the input timeline are unchanged. Old models
default to native conditions, with unchanged parameter shapes and initialization.
Removing explicit detector information does not remove bandwidth cues from the audio.

The objective matches P36 compressed: per-example `log1p(direct) + log1p(MR)`, the
original paired/identity coefficients, plus 0.1 times the residual-budget objective.
The target is original unclipped lossless audio. No gradient passes through complete
safety limiting or failure policies, and dynamics weights are not changed concurrently.

## Review protocol

After all four final endpoints freeze, run export, native PCM/proxy correspondence,
common MR, CPU timing and the two predetermined output paths: current and phase-only.
Phase-only removes magnitude remixing while retaining phase fusion, detection and
gating. Compare candidate/control within the same path and seed; do not select the
best score across paths.

Each path requires both seeds to improve native music MR means and median per-recording
differences, without regressions in music/speech identity or either role's dynamics.
Retain every band and absolute-PCM counterexample. Report raw prediction, native
output, untreated compressed input and identity DSP separately. Even passing this
research screen would not publish weights: independent recordings, stereo, continuous
streams and blind listening remain necessary. A completed budget does not prove
convergence.

## Device correction and startup

The 2026-10-22 startup audit confirmed that **the frozen P36 executor already used
CPU**. `run_direct_spectrum_study.train` creates CPU models/batches without moving
them to MPS; the outer executor sets one thread. Treating the Metal compiler-service
failure as a blocker for this experiment was incorrect. P38 explicitly registers
the same CPU route, rather than changing devices to bypass a failure.

The independent startup MPS probe failed with `MTLCompilerService` connection error
141; it cannot be reported as passing. This did not affect CPU training or Rust CPU
ORT validation. The system FFmpeg dynamic-library issue was handled using the existing
local FFmpeg in the process PATH, without changing the system installation.

Executor: `training/scripts/run_detector_condition_study.py`. Formal local directory:
`data/runs/detector-condition-p38-20261022/`; separate preflight:
`data/runs/detector-condition-p38-preflight-20261022/`. Source, configuration, manifests,
initialization, schedules and native binary are hash-bound. Recovery state is saved
every 64 steps. Failures retain evidence; no automatic budget changes, retries or
intermediate-best selection occur. Status is in `supervisor.json`,
`study/progress.json` and `pipeline.log`; checkpoints are under
`study/<arm>-<seed>/latest.pth`.

The two four-step preflight endpoints completed training, ONNX export, 16 native PCM
roles, 16 additional MR replays and 32 fusion-path replays. Maximum fusion-proxy
error was `2.98e-7`, with unchanged tolerance. CPU tests: **653 passed, seven MPS
items unverified**. Coverage includes nonzero-model condition invariance/sample-rate
gradients, history equivalence, checkpoint policy, 320-hop export and per-path,
two-seed screening. Short preflight loss was not used to predict full-run quality.

All **191 execution-dependency hashes** match preflight/formal execution. Registration
SHA-256: `5919c8c0c45891c00006ebce694f173c700f14bd5cebf4f6cc0fb990afa2a18d`.
The supervisor records child exit success/failure without automatic restart or
rewriting P36/P37 evidence. Initial process-bound caffeinate failed with
`Failed to create PreventUserIdleSystemSleep assertion`; sleep prevention was not
active then. Training continued and step 256 was verified; the machine needed to
remain awake.

## Completion and resume integrity

All four trajectories completed **16,192 steps**, totaling **64,768 updates**. The
third finished October 22 at 21:50:59. At the user's request, the fourth paused at
step 2,624 for a Mac restart. It resumed with identical weights, optimizer and sampling
sequence; the first 2,624 history records and the three completed receipts were
unchanged. Sleep prevention worked after restart. Final supervisor exit code: 0.

All endpoint hashes, full update histories, registered inputs, 191 frozen dependencies,
native scoring receipts and raw timing-report hashes were verified. Weights and
training values are finite. Gradient clipping occurred 1 / 0 times in controls and
2 / 1 times in fixed-condition arms. Every control model tensor equals its matching
P36 `compressed` endpoint, reproducing the control behavior exactly.

## Quality results

Common MR v1, aggregated by recording, against original unclipped lossless audio.
Changes below compare the fixed-condition arm with the **same seed and output path's
control**; negative means lower error. Cells list seeds 20261012 / 20261013.
The 89 reused held-out clips cover 26 music and 54 speech recordings, not a new
independent test.

| Stage/path | Music MR | Music dynamics | Speech MR | Speech dynamics |
| --- | ---: | ---: | ---: | ---: |
| Raw model OLA | **+0.276% / +0.629%** | +0.185% / −0.113% | −0.099% / −0.162% | +0.165% / +0.093% |
| Current native DSP | **+0.195% / +0.558%** | +0.122% / −0.137% | −0.057% / −0.145% | +0.136% / +0.040% |
| Phase-only native DSP | **+0.242% / +0.576%** | +0.166% / −0.087% | −0.052% / −0.172% | +0.171% / +0.055% |

Current-path music MR changes from `0.3787409 / 0.3789824` to
`0.3794783 / 0.3810963`. Recording outcomes are **13 wins/13 losses and six wins/20
losses**; neither seed improves the median recording difference. Phase-only also
fails the two-seed music-mean criterion. Against untreated compressed input, fixed
current still reduces error by 1.466% / 1.045%, versus controls' 1.657% / 1.594%.
Failing to beat the control does not mean every metric is worse than untreated audio.

Above 16 kHz, current output amplitude error increases **0.656% / 1.525%** and log-shape
error **0.602% / 4.169%**. Raw OLA amplitude error increases 0.596% / 1.659%.
Seed two's high-band dynamics and PCM RMS improve approximately 0.525% / 0.578%,
which does not cancel primary amplitude/shape regressions. Current 0–4 kHz PCM error
increases only about 0.0002%; broad low-frequency destruction does not explain the
main regression.

The largest recording accounts for approximately 33.1% / 45.1% of positive paired
music error increments. The paired failure is not wholly one weak-bin/recording
outlier. In contrast, about 99.0% of seed one's positive music-identity increment
comes from the known `heiyaoyao-adaa234…` case; these mechanisms must be separated.

## Identity preservation and absolute errors

Current music identity MR changes **+8.285% / −3.175%**; speech identity changes
**−42.392% / +53.658%**. Seeds disagree. Much music-identity error against the original
reference includes existing safety limiting. Identity DSP on the same material
separately measures additional model changes.

Recording-mean identity PCM RMS relative to identity DSP:

| Material | Control | Fixed conditions |
| --- | ---: | ---: |
| Music | 2.28e-6 / 2.89e-6 | 1.75e-6 / 2.94e-6 |
| Speech | 4.65e-8 / 5.92e-8 | 3.27e-8 / 7.83e-8 |

Worst fixed-condition music-clip added RMS: `1.70e-5 / 2.84e-5`
(−95.39 / −90.93 dBFS); speech: `5.08e-7 / 1.40e-6`
(−125.88 / −117.06 dBFS). For seed one's known weak-high-band counterexample,
added PCM rises from `2.66e-6` (−111.51 dBFS) to `5.34e-6` (−105.45 dBFS).
These are calibrated PCM values, not FFT power. Relative spectral increases do not
establish audible severity; small absolute errors do not replace restoration gains
or blind listening. All counterexamples remain and thresholds were not revised.

## Native correspondence and compute

All endpoints completed **712 native PCM roles, 712 additional common-MR replays
and 1,424 two-path replays**. Native synthesis/fusion proxy differences are at most
**2.98e-7**, with zero tolerance failures under the original `2e-5` threshold.
The regression already exists in raw predictions; a Python/Rust correspondence
failure is not supported by these measurements.

Current-path CPU ORT plus DSP covers **280,000 hops**, 44.1/48 kHz, mono/stereo and
paced cases. All controls/candidates remain; failures were not replaced by reruns.

| Metric | Result |
| --- | ---: |
| Maximum compute | **5.411 ms** |
| Highest scenario p99 | **1.135 ms** |
| Calls ≥5 ms | **2** |
| Service time over hop period | **11** |
| Paced completion delays | **27** |
| Maximum paced lateness | **4.584 ms** |
| Peak process RSS | **31.56 MiB** |

Both ≥5 ms calls occur in seed one's fixed-condition 48 kHz stereo paced case.
A control also has four service overruns and six paced delays. Thus strict <5 ms
compute and continuous hop deadlines are **not satisfied**. Fixed order, OS
scheduling and background load may affect the tail; these results cannot attribute
slower inference to constant conditions alone. Low p99 does not erase maxima or
lateness. Process user+system totals 44.87 s and wall time 136.71 s, including startup
and paced waits; this is not callback CPU utilization. Phase-only quality review is
not a renewed performance qualification for that path.

Fixed buffering remains 256 frames: 5.33 ms at 48 kHz and 5.80 ms at 44.1 kHz,
separate from compute. Real-device, full callback/queue and blind-listening validation
are incomplete; glitch-free operation is not established.

## Decision and follow-up

1. Retain explicit bandwidth conditions; stop unchanged expansion of the fixed-condition
   route. P37 frozen-inference sensitivity did not predict retraining gains. This does
   not prove detector conditioning is always harmless or isolate active alone from
   the joint intervention.
2. Separate DSP interface effects from learning. Raw predictions already regress,
   while P37's magnitude-remixing issue remains real. Preserve direct lossless targets
   and native conditions; do not hide training regressions by selecting another path.
3. Do not blindly extend or enlarge these runs. Diagnose high-band magnitude/shape,
   phase and reference-dynamics gradients on fixed training crops before designing
   a new small-model comparison. Neither convergence nor an architecture ceiling is
   established merely by finishing the registered budget.
4. Diagnose timing separately with the same frozen binary/weights and balanced order,
   stage attribution and actual worker deadlines/fallbacks. Whole-file mean time is
   not acceptance evidence.

Both path screens failed. Default weights are unchanged; no subsequent training was
launched. Local supplements: `post-review/bands.json` and
`post-review/verification.json`, relative to the formal run. Audio, checkpoints and
reports remain ignored. Subsequent [P39 gradient](objective-conflict-audit.md) and
[timing](interleaved-timing-audit.md) diagnostics are recorded separately.
