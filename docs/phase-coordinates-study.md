# P31: Phase Input and Complex Residual Coordinates Comparison

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

2026-10-19, building upon [P30 Phase Swap Diagnosis](phase-swap-audit.md). Reference phase improved waveform fidelity, but pairing reference magnitude with predicted phase amplified errors. This round evaluates whether the model can more effectively learn high-frequency complex signals. Reference phase swapping does not yield deployable benefits and cannot replace the current training and actual Rust output validation.

## Fixed Design

| Group | Phase Input | Residual Coordinates | Parameter Count |
|---|---|---|---:|
| relative | Disabled | Relative input phase | 431,015 |
| relative_phase | Enabled | Relative input phase | 455,783 |
| cartesian | Disabled | Global real/imaginary parts | 431,015 |
| cartesian_phase | Enabled | Global real/imaginary parts | 455,783 |

Fixed width of 96. Phase input follows the observation complex carrier from P28: log-amplitude normalized by RMS across all frequencies, multiplied by sin/cos phase components, and projected into the encoder with zero-initialized projection. Thus, weak observed frequency bins still contribute weak phase information. This round does not claim to recover true phase in missing frequency bands. The phase-aware branch adds 24,768 parameters (~5.75%), enabling comparison of two factors and their interaction, though it cannot fully isolate phase information from added capacity.

The global coordinate group directly predicts residual differences in physical real/imaginary components, then converts to a common polar coordinate output interface. This differs from first predicting local residuals and then rotating them by input phase. The same amplitude units, budget head, residual slope, DC/Nyquist real constraints, and causal state are preserved. No lookahead or buffering is introduced. The old checkpoints remain defaulting to relative coordinates. All shared tensors are verified as identical under seed; zero output head and zero-phase projection maintain exact identity initialization.

Each group uses two fixed seeds (20261012, 20261013), with **512 steps**, Adam optimizer at 1e-4, gradient norm capped at 5. All use the direct spectrum loss from P29 raw group with 0.1 budget supervision. Version, frequency splitting, normalization, magnitude/complex/envelope/dynamic weighting remain unchanged. The old DSP loss is used only for unified scoring and does not participate in optimization. Full training budget is 4 × 2 × 512 updates; no early stopping or extension of any group based on intermediate performance.

Continues to use the 20 pre-fitted training clips from P29: 16 music recordings, 4 speech recordings, one speech per four music steps, with corresponding clean identity references, cycled over four batches; 80-frame warmup with no gradients, supervision starting from frame 89. De-duplicated audio coverage totals approximately 7.32 seconds of music and 1.85 seconds of speech. **This is an early learnability diagnostic, not a full music library training, independent test, convergence proof, or architectural capacity upper bound.** Audio source gaps in recording, region, and channel coverage remain as in [P29](direct-spectrum-study.md) and cannot be extrapolated beyond this dataset.

## Validation and Execution

Entry point: `training/scripts/run_phase_coordinates_study.py`. On registration, binds parent experiment, audio, and tool hashes, and freezes Python/JSON/YAML/TOML runtime dependencies. Pre-check runs in a separate directory, per seed, per group, for four steps; weights are used only for integration validation and do not enter the 512-step experiment or contribute to group selection.

```bash
rtk proxy env PYTHONPATH=training training/.venv/bin/python \
  training/scripts/run_phase_coordinates_study.py \
  --parent data/runs/direct-spectrum-p29-20261019/study \
  --output data/runs/phase-coordinates-new/study \
  --native-binary data/runs/small-mechanisms-p28-20261019/binaries/soundex-context-replay \
  --prepare-only
```

After registration, copies the `source_hashes` dependencies to `source/training`, runs the same command from the frozen snapshot, and removes `--prepare-only`; the runtime code does not follow workspace changes. Checkpoints can be verified for recovery; frozen results are not overwritten. Any run failure must preserve both error logs and completed evidence and cannot be falsely claimed as a full workflow completion.

After full training and freezing of all eight endpoints, a unified set of evaluations are performed: direct spectrum scoring, real DSP playback, joint PCM recheck, ONNX consistency, and 320 role rechecks from Rust frontend → ORT → DSP. Results are reported separately for music/speech, paired/identity, raw prediction vs. final output, and all counterexamples. Identity DSP and unprocessed input remain unchanged. Phase and spectral shape improvements must be validated alongside high-frequency absolute PCM error, transient behavior, and preservation in key frequency bands.

Subsequently, all eight endpoints undergo native performance testing: mono/stereo, 44.1/48 kHz, two additional stereo 48 kHz runs, and paced stereo 48 kHz per endpoint, totaling 560,000 hop timings. Reports include full model + DSP computation, CPU/RSS, number of times exceeding 5 ms or hop period, and paced latencies; a fixed 256-frame buffer is separately counted as output delay. This is still not proof of continuous audio device operation, perceptual quality, or formal product qualification.

## Current Status

**All eight 512-step training runs, unified native rechecks, and performance tests are complete**, located at
`data/runs/phase-coordinates-p31-20261019/`. Frozen source code corresponds to commit
`20272ed377814dcdf4e4d75bcf9b015409c9080b`, comprising 175 runtime dependency files.
Registration SHA-256: `d8cae527bfcfbbb497df9202838a8bef7af72ad13c5a764a6daa221388af5937`.
Background supervision process exited normally at 23:32:55 UTC+8 on 2026-10-19, with total runtime of approximately 19 minutes and 24 seconds.

Locally, **608 test cases passed** (15 new), Ruff lint/format passed. Coverage includes zero initialization, old checkpoint compatibility, global residual direction, silence/low-energy/phase wraparound gradient, finite difference, causal chunking, channel isolation, reset, and two new configurations' 320-hop ONNX sequential consistency checks.

Independent preflight `data/runs/phase-coordinates-p31-preflight-20261019/` completed four steps per group, all exported, and 160 native processing roles; maximum synthetic error: 2.98e-7, zero synthetic tolerance failed. Maximum absolute difference between real observation input and ORT/Torch complex output: 6.10e-5, retained separately—this is not PCM synthesis error and should not be conflated with a single threshold. The control group's first four steps show identical loss and gradient records to P29 raw. The preflight only proves integration feasibility, not improved audio quality or meeting 5 ms latency requirements.

## Unified Evaluation on 2026-10-20

**No overall audio restoration benefit was established; no default weights were promoted.** The following are descriptive rechecks of fixed training clips; all eight endpoints are retained, with no selection based on results. It is not possible to deny architectural capacity based on 512-step results, nor can a lower training loss or higher high-frequency performance be considered a success in isolation.

### Magnitude Improvement Has Not Translated to Complex or Waveform Fidelity

For music, clip mean (across 16 different recordings, equivalent to current recording mean), lower is better.
Range spans two seeds, no seed selection is made. Spectral energy error is relative, not PCM dBFS.

| Group | Original Prediction High-Frequency Energy Error (dB) | Final Rust Full-Band PCM RMS Error | Final Rust High-Frequency PCM RMS Error |
|---|---:|---:|---:|
| Unprocessed Compressed Input | 8.1553 | 0.009219998 | — |
| Identity Model + DSP | — | 0.009223736 | 0.006118215 |
| relative | 1.5170～1.7485 | 0.009236265～0.009246550 | 0.006134616～0.006147514 |
| relative_phase | 1.3312～1.6193 | 0.009248983～0.009267036 | 0.006150753～0.006172374 |
| cartesian | 8.0676～8.1092 | 0.009223736 | 0.006118214 |
| cartesian_phase | 8.0563～8.1009 | 0.009223736 | 0.006118213～0.006118214 |

After introducing phase input, the relative coordinate group's music final waveform error increased by **0.138% / 0.222%** compared to the control, and by **0.314% / 0.510%** compared to unprocessed input. Final high-frequency spectral energy SC improved from control's 0.391291 / 0.381931 to 0.397256 / 0.395255, and flux error increased from 0.616682 / 0.606590 to 0.631848 / 0.631158. Improved magnitude energy coexists with these regressions.

The overall direct objective paired mean (80% music, 20% speech) decreased from control's 0.229088 / 0.225849 to 0.227525 / 0.222259; however, the complex component increased from 0.234412 / 0.236973 to 0.240113 / 0.242174. Numerical validation confirms this, though it does not imply that the current multi-objective weighting can constrain this trade-off. Cross-comparison across different metric units is invalid.

Speech-specific recheck also failed to establish waveform benefit: Identity DSP has PCM RMS error of 0.002475175; relative: 0.002478980 / 0.002479879; relative_phase: 0.002480731 / 0.002480421; both global coordinate groups are nearly identical to the input. Final error mean for clean identity music: Identity DSP: 0.000487582; relative: 0.000499157 / 0.000500623; relative_phase: 0.000500924 / 0.000498563; global coordinate groups at ~0.000487700–0.000487719. The Identity DSP itself contains clipping artifacts and is not a clean reference of zero; all original references and counterexamples are preserved and not re-clipped.

### Global Coordinate Group Does Not Form Effective Residuals

Post-evaluation, a read-only diagnostic was conducted across all eight endpoints and four fixed training batches; no further fitting occurred. After normalization by full-frequency RMS, the original complex residual of the global group remained only **2.75e-5 to 3.61e-5**; relative group batches ranged from 0.00184 to 0.0240. The divergence is already significant before DSP processing, and cannot be attributed to the DSP correctly canceling all predictions.

All endpoints show zero negative saturation in the decoder. The global group retains non-zero gradients, and phase projection is non-zero, though the first fixed batch's encoder gradient norm was only 4.75e-5 to 1.22e-4—significantly weaker than the relative group. Budget mean was 0.0238–0.0249, while relative group averaged 0.1396–0.1684, and residual logits were substantially smaller. These results indicate early optimization stalled near the identity solution and do not individually prove the mechanism of gradient cancellation or architectural limits. In the last two 64-step windows, the global group showed only minor raw loss changes, while the relative group continued to decrease—thus, the model has not converged.

### Performance: Low Typical Latency, Long Tail Falls Short of Strict Requirements

All **560,000 hops** completed, with finite value checks passed. Per-test case total computation p99 ranged from 0.107 to 0.965 ms; paced stereo 48 kHz p99 ranged from **0.490 to 0.965 ms**. However, there were **9 instances exceeding 5 ms and 13 instances exceeding the hop period**, with a maximum of **14.096 ms**. Among them, paced stereo at 80,000 hops had 8 instances over 5 ms, 10 over hop, and **79 instances of late completion**, with a maximum completion latency of 15.557 ms. Service latency, scheduling overhead, virtual queue prediction, and actual device xrun are distinct metrics and cannot be conflated.

Peak RSS during native measurement was **31.03 to 34.20 MiB**, not reflecting model memory overhead. Paced process CPU time/wall-clock time was approximately **11.4% to 20.7% of single-core**, including load and pre-warmup. Non-paced tests ran at full speed, and CPU usage cannot be equated to actual playback load.

Host one-minute load rose from early ~3.10 to ~10.29–14.42 during paced phase; scheduling contention remains a pending factor and cannot be used to excuse or attribute tail latency differences to architecture. The global phase group did not timeout in this round, and cannot be declared to have passed continuous stream qualification based on only two short cases. **This round does not prove strict <5 ms or device-free stuttering.** The fixed 256-frame buffer remains at 5.333 ms (48 kHz) and 5.805 ms (44.1 kHz), separate from computation time.

### Completeness and Next Steps

Hash verification of 567 sources, frozen source code, checkpoints, replay receipts, and original timing files all match. For 320 full Rust frontend → ORT → DSP roles, maximum synthetic difference is **2.38e-7**, zero tolerance failed; the maximum absolute difference between observed ORT/Torch complex spectra is separately retained at 6.10e-5. Remote CI for implementation submission and startup documentation passed. No blind listening or physical device validation was conducted.

Reproducible analysis and per-recording differences are available locally at
`data/runs/phase-coordinates-p31-20261019/evaluation-20261020/`:
`summarize.py` / `assessment.json` aggregate all endpoints and latency; `mechanism.py` and `mechanism-*.json` retain activation, budget, and gradient diagnostics; computations use frozen P31 dependencies.

**Next Steps:** Recommend a small-scale learnability and objective trade-off audit: why magnitude improvement occurs alongside phase degradation under direct complex/waveform constraints, and whether global coordinate gradients, budget scaling, and phase representation are effectively aligned. Pre-register budget comparisons without selecting "best" tracks from this training set or directly expanding the model. Performance should be retested under stable, recordable conditions on all candidates, preserving all long-tail evidence. This P31 evaluation has not yet initiated these new experiments; default DSP and formal weights remain unchanged.

Upon user-defined spectral priority, [P32](spectrum-priority-study.md) has been launched accordingly: fixed model structure, isolated multi-resolution spectral supervision, waveform as auxiliary check. P31 historical results and frozen protocol remain unchanged and are not retroactively adjusted to reflect new priority.
