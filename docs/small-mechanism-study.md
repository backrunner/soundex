# P28: Initial Validation of Deactivation Mechanisms with a Small Model

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

2026-10-19, following [P27: Localization and External Method Survey](head-collapse-audit.md).
This round fixes the width at 96, does not restore the 640-width large model training, and does not pursue parameter scaling as a solution.
We first validate three mechanisms: decoder deactivation, budget supervision gradient, and phase input, before deciding whether to implement a new DSP incentive branch.

## Fixed Control Set

| Group | Unique Intervention | Parameter Count | Pre-check ONNX |
|---|---|---:|---:|
| `control` | Small-scale complex residual head, original GELU, shared budget gradient, magnitude input | 431,015 | 1.688 MiB |
| `decoder_norm` | Add per-frame LayerNorm between decoder Linear and GELU | 431,207 | 1.689 MiB |
| `budget_detach` | Block gradient flow from budget auxiliary loss to shared latent | 431,015 | 1.688 MiB |
| `phase_input` | Add zero-initialized projection of observed complex carrier in encoder | 455,783 | 1.784 MiB |

All groups use the full residual budget `budget_scope="residual"` with `residual_slope=1/64`, identical shared parameter initialization, zero output initialization, and the same random data schedule. The `phase_input` group adds 24,768 new parameters, so its performance cannot fully isolate phase information from the added parametrization effects. This round does not include a ranking by scale.
These are single-factor controls; no interaction effects among the three modifications are tested, nor are all modifications combined and attributed in a single experiment.

The `budget_detach` group uses the same budget head and identical budget prediction values, but computes the auxiliary supervision via `budget_head(latent.detach())`. Deployment output and main recovery loss gradients to the budget head or shared features remain unchanged from the baseline—no budget is disabled, no path is bypassed, and no enhancement is reduced.

The `phase_input` group combines the current input phase (sin/cos) with normalized log magnitude carrier, then projects it through a bias-free 258→96 projection and adds it before the LayerNorm in the encoder. The feature floor for digital silence is first subtracted, and no arbitrary phase is injected into zero signals. Only past and current state inputs are accessed; reference phase is used solely for existing training supervision. The original TCN accumulates historical context; no forward-looking, channel-sharing, or additional frame buffering is introduced.
LayerNorm is computed only along the feature dimension, with no cross-time statistics or training/inference running statistics differences.

## Training and Data Protocol

- Two fixed seeds: `20261012` and `20261013`, each with **2,048 updates**, totaling eight trajectories and 16,384 updates.
  This is a diagnostic budget for small-scale mechanism validation, not a sufficient convergence or capacity ceiling benchmark. It should not be ranked against P26's 16,192-step results, nor should any group's budget be reduced during the process.
- Each seed reuses the first 2,048 draws from P26's broad schedule; each step includes 4 distinct music recordings + 1 speech, paired with corresponding clean identity, paired/identity weighting. The codec, sample rate, channel view, and source time rules are preserved.
  Data remains from pre-verified native, lossless music and established speech materials—no third-party weights or data are introduced.
- The entire input window and development/shifted source intervals are isolated; cross-codec, cross-sample rate, and cross-channel view checks are maintained.
  Recording, source, and work-group divisions are preserved for audit. Actual independent supervision duration is logged in `coverage.json`; coverage cannot be substituted by number of tracks, nor can mono L/R/M/S views be treated as paired stereo evaluations.
  Both seeds involve 253 music and 193 speech recordings, across 9 codec formulations; after deduplication by source time, music supervision totals 3565.78 / 3565.40 seconds, speech supervision totals 531.73 / 527.10 seconds.
- Unified Adam optimizer with `1e-4`, no weight decay, gradient norm capped at 5; loss includes main reconstruction term plus `0.1365237430457469 × clean-complex consistency` and `0.1 × budget risk`.
  No changes to loss definition, weights, band normalization, or DSP are introduced; GANs are not used.
- Identical 80-hop history warmup, supervision begins at frame 89, and warmup gradients are not backpropagated. Training cache uses 128-crop LRU; new random crops are not written to disk; a small amount of pre-verified cache is retained in this run to avoid reconstructing large, previously cleaned caches.
- All groups complete full training, and final checkpoints are frozen with hash; no best checkpoint selection or early stopping is performed.
  Numerical divergence, source drift, or runtime errors must trigger immediate termination with record.

## Validation and Judgment

The fixed training monitoring set only reports optimization progress. Final mechanism diagnosis uses the first seed's draws at 0, 682, 1364, and 2047, totaling 16 music and 4 speech clips:
Recorded GELU pre- and post-activation, budget, per-band raw and post-processed residuals, identity DSP differences, and reference oracle.
For the first batch of fixed samples, we decompose the shared, decoder, and budget gradient components of five loss terms; for the budget isolation group, actual training gradients are verified via routing.
For normalization groups, activation pre-statistics use actual GELU inputs—never the LayerNorm input.

Eight final endpoints undergo unified recheck:

1. Recheck using the same previously reused development/shifted clips, reporting music, speech, raw predictions, and model + DSP outputs.
   Development is not a new independent test; results cannot be used to claim absence of overfitting.
2. Retain unprocessed compressed input, identity model passing through real DSP, and original lossless reference.
   Check HF spectral convergence, energy, transients, low-frequency fidelity, identity perturbation, and absolute PCM RMS/dBFS.
   Recovery success is not judged by weak-band LSD or minor enhancement metrics.
3. Replay 1,602 roles from original prediction, jointly verified against baseline across 1,780 roles; then perform **1,424 full Rust frontend → ORT → DSP role replays**, preserving frontend differences and observed synthesis differences.
4. All eight exports undergo 320-hop ONNX state, silence, and reset consistency validation; full pipeline performance is tested under 44.1/48 kHz, mono/stereo, repeated measurement, and paced long streams, totaling 56 processes and 560,000 measurement hops.
   Report <5 ms computation ceiling, 48 kHz/hop128 deadline of 2.667 ms, CPU/RSS, and scheduling lateness.
   Fixed 256-frame buffer is included in total output latency; synthesis performance stimulation is not a device or listening validation.

Separate judgment of recovery effectiveness and mechanism effectiveness.
Activation recovery, non-zero gradients, or training loss reduction only indicate mechanism intervention effectiveness.
Only when cross-seed high-frequency recovery, band preservation, and original audio fidelity are supported can we proceed to full-scale training or DSP structure experiments.
Current relative metrics serve only as screening tools; all absolute errors and counterexamples are retained.
No rule relaxation, no seed selection, no automatic scale expansion, and no automatic replacement of official weights are permitted.

## Engineering Pre-check and Execution

Pre-check directory: `data/runs/small-mechanisms-p28-preflight-20261019/`.
Each group runs 4 steps, completing training, mechanism diagnosis, checkpoint freezing, joint scoring, native replay, and export, followed by 16 full Rust frontend → ORT → DSP role replays.
Pre-check weights are discarded and not used for candidate selection or quality proof.
It covers the full pipeline but does not run formal long-term performance measurement.

All four exports pass 320-hop state/reset/silence consistency checks; 16 full Rust role replays pass observed synthesis checks.
Complete local test results: **569 items passed**, Ruff lint/format passed.
Fixed bugs in old resume test stubs for new keyword parameter compatibility, and added precise resumption validation for isolation groups.

New structural validations include: shared initialization and old checkpoint compatibility, budget auxiliary gradient isolation with unchanged main gradient, negative common offset decoder gradients, phase input validity/causality, finite difference, silence/weak signal gradients, block/reset/channel isolation, non-zero model ONNX and real Rust consistency, and endpoint timing selection.

Formal training is driven by `training/scripts/run_small_mechanism_study.py`, running from a previously submitted frozen codebase.
Run path, registration hash, and startup status are recorded in the subsequent status and locally in `provenance.json`.

Post-training mechanism diagnosis, native recheck, and latency testing are executed in sequence via the same pipeline.

```bash
rtk proxy env PYTHONPATH=training training/.venv/bin/python \
  training/scripts/run_small_mechanism_study.py \
  --parent data/runs/broad-budget-p26-20261017-v2/study \
  --output data/runs/small-mechanisms-new/study \
  --native-binary target/release/soundex-context-replay --prepare-only
```

When executing the full pipeline, first save a snapshot of `source/training`, then launch from that snapshot.
Do not resume from a modified `main` branch after the registered state.
`--preflight` is fixed as a separate 4-step pre-training material check and must use a distinct directory.
Native playback remains an experimental interface; official artifacts and default weights remain unchanged.

## Launch Record

The formal pipeline started on **2026-10-19 03:29 +08**, in directory
`data/runs/small-mechanisms-p28-20261019/`.
Frozen training source code comes from commit `435024a86a4c80e53a19c2f26d6494ee3d0c274c`, registered SHA-256 as
`d99d33421be0cb94ddcc861b22e95856f97feeb025dae467114b32c080caa8a0`.
At startup, supervisor PID 4967, pipeline PID 4969, caffeinate PID 4970; PIDs are for internal tracking only.
Latest progress can be viewed in `study/progress.json`, logs in `pipeline.log`, exit status in `supervisor.json`.
The pipeline completed normally at **2026-10-19 06:04 +08** (exit 0); all eight trajectories completed 2,048 steps, and all mechanism diagnosis, unified recheck, eight exports, full Rust replay, and long-stream timing tests were completed.

## Final Results

**None of the three interventions passed the pre-registered dual-seed improvement screening; no candidates advanced.**
Below are clip mean values from reused development music clips; lower HF SC is better,
PCM error refers to the original lossless reference. This is not a new independent test, nor a listening-based conclusion.

| Configuration | Raw Prediction HF SC (Seed 12 / 13) | DSP Output HF SC (Seed 12 / 13) | Full Rust Output PCM Error RMS (both seeds ~) |
|---|---:|---:|---:|
| Identity Control | 0.54787658 | 0.54876080 | 0.012518218 |
| control | 0.54782270 / 0.54796378 | 0.54877521 / 0.54888899 | 0.012518218 |
| decoder_norm | 0.54787194 / 0.54797595 | 0.54883172 / 0.54886515 | 0.012518218 |
| budget_detach | 0.54824934 / 0.54805025 | 0.54919308 / 0.54895948 | 0.012518218 |
| phase_input | 0.54782296 / 0.54796493 | 0.54877604 / 0.54888656 | 0.012518218 |

Unprocessed compressed input music has PCM error of **0.012404078**; after passing through identity DSP, it becomes 0.012518218; all eight models remain around this value—no evidence of error reduction.
Music has 33 clips, 26 recordings; speech has 56 clips, 54 recordings.
Speech unprocessed error: 0.00350853224; identity DSP: 0.00350853253; the eight models range from 0.00350853221 to 0.00350853271—no substantial recovery evidence.
Original reference is preserved; limited-amplitude signals are not replaced with recovery targets.

Complete Rust output vs. identity DSP music differences show a clip mean RMS of only **2.77e-8 to 2.04e-7**.
The main result remains close to identity output, not effective high-frequency restoration.
All candidate DSP outputs have HF SC slightly worse than the initial identity control—no recovery benefit is evident.
Individual seed improvements or total loss changes cannot be interpreted as deployment gains.

### Correction to Mechanism Judgment

Among 20 fixed training clips, all eight models show GELU near-zero and negative saturation ratios of 0.
The small model control group does not reproduce the large model's extensive deactivation observed in P27, yet still produces near-zero residuals.
Therefore, **GELU deactivation is not a sufficient explanation for near-identity output**; this does not negate its impact on prior large models, nor can the 2,048-step result be used to infer that longer training would never deactivate.

The normalization group shows raw normalized residual RMS of 6.46e-5 to 1.08e-4, about 2.05e-5 higher than control, though post-budget and post-DSP values remain very small.
The budget isolation group indeed shows an auxiliary shared gradient of 0, but no better recovery results are achieved.
The phase input group shows metrics nearly identical to the control group.
These results do not support direct stacking of modifications or scale expansion.
Initial training samples still show conflicts between recovery and identity/consistency gradients, but a single-batch diagnosis cannot prove this is the sole cause—nor can it justify removing fidelity constraints.

Next, we should first use small-scale, training-material-limited learnability diagnostics to identify where—between original residual target, budget scaling, and deployment loss—the effective restoration stalls, before deciding on new DSP incentive or spectral shape control branches.
Original predictions and full deployment outputs must be judged separately, and low-frequency, transient, silence, and identity constraints must be preserved.
This is a recommendation for the next step; no new training was initiated, and no new DSP incentive branch was implemented.

### Native Consistency and Real-time Overhead

All 1,424 full Rust frontend → ORT → DSP role syntheses pass observed synthesis checks, with maximum absolute error of **2.98e-7**; maximum ORT complex spectrum error per input is **3.87e-5**, separately retained and not conflated with synthesis error.
Export consistency passing does not imply effective recovery.

On an Apple M5 Max, 56 measurements total **560,000 hops**, with 0 computations exceeding 5 ms.
Maximum observed latency is **3.6204 ms**. For 24 non-paced, 48 kHz, stereo measurements, p99 is 0.1625–0.1764 ms; for paced, 8 measurements, p99 is **0.9022–1.0165 ms**, CPU usage 19.1%–24.4% per core, peak RSS 31.53–32.00 MiB.
Paced stream has 80,000 hops, with **5 instances exceeding 2.667 ms** and **32 instances of late completion**; these are distinct statistics and cannot be summed or directly labeled as device clipping.
Fixed 256-frame buffer at 48 kHz remains **5.33 ms**, not included in the above computation.
Synthesis workload meets the <5 ms requirement, but no real device, callback, or blind listening validation was performed.

Results are saved in `study/comparisons.json`, `mechanism/`, `joint-review/`, `native-audit/`, and `native-cost/`.
1,133 file hashes were verified, including 34 training, recheck, and export receipts, frozen source code, eight endpoints, and native reports—all matched.
Recorded in local `completion-audit-20261019.json`; no rehashing of training audio was performed.
CI [build](https://github.com/backrunner/soundex/actions/runs/39159181414) passed.
Experimental artifacts remain in ignored directories; official weights and default access behavior remain unchanged.

Both seeds `20261012 / 20261013` are retained. Zero update or identity behavior is a diagnostic control, not recovered audio.
