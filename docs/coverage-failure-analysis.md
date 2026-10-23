# P19 Failure Localization: Cross-Recording Fidelity Failure and Band Interference

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

## Conclusion

The issues with P19 are not merely due to a small music library, nor are all recordings exhibiting degradation over time. While overall training error decreases and most development music pairs show only minor improvements in paired error, a small number of previously unseen recordings exhibit severe original signal damage—outweighing these marginal gains. Stepwise and actual Rust intervention further confirm: when the model attempts to correct weak frequency bands, the overlap-add (OLA) reanalysis reveals interference between adjacent bands, significantly amplifying errors in severe cases.

This is the actual failure chain observed in this round—not a proof that any new architecture can resolve it:

1. **Natural narrowband, lossless inputs can still activate enhancement paths**; the cutoff frequency does not distinguish between natural bandwidths and compressed degradation.
2. The current shared predictor learns small identity corrections on training recordings, but produces incorrect corrections on certain previously unseen recordings.
3. Output is determined by per-band amplitude correction, preserved input phase, and overlapping windowing; small per-frame spectral errors do not guarantee small overall output. In severe cases, the primary damage originates from adjacent bands, not directly from the corrected band itself.
4. Relative normalization of weak reference bands amplifies these damages in the loss function; real Rust outputs also show spectral bias, so problems cannot be resolved simply by adjusting the denominator or relaxing thresholds.

The initial control for this round was a zero-correction identity mapping. Its performance cannot be interpreted as "overfitting better." This also does not re-verify the overfitting issue in early training baselines—these should be considered separately.

## Scope and Recheck

All original results for P19 are frozen; no model updates, no loss modifications, no checkpoint reselection.
A new staged diagnosis is added, covering the first registered window per recording in the training set (253 music + 8 speech), and all 89 reused development fragments.
On the training side, two final models are inspected; on the development side, two seeds are evaluated at 2,048 → 16,192 steps, totaling 878 cases and 1,756 paired/identity roles.

The training-side staged diagnosis includes only one fixed window per recording, not claiming full coverage of the 2,088-position library. Training error statistics are drawn from the original experiment's completed position library scoring.

For all development cases, the target values must differ from the frozen scores by no more than `2e-6`. This threshold is met.
Maximum waveform error after reorganized band residuals: `2.98e-8`, maximum complex spectral error: `2.87e-7`.
For the three worst music fragments per seed, a five-band only/without diagnosis is performed.
Additionally, 24 real Rust re-replays are conducted, based on the primary error band sources.
Proxy/native maximum per-sample error: `2.98e-7`, below the original `2e-5` threshold.

These are post-hoc diagnostic checks—not new independent tests, no listening evaluations, and not a basis for candidate release approval.

## Training Gains and Development Tail Damage

| Metric | Seed 20261012 | Seed 20261013 |
| --- | ---: | ---: |
| Full training music paired error, 2,048 → 16,192 steps | 0.622491 → 0.608755 | 0.622672 → 0.608035 |
| Full training music identity error, 2,048 → 16,192 steps | 0.005085 → 0.003182 | 0.004660 → 0.003731 |
| Development music paired error, initial → final | 0.588365 → 0.684476 | 0.588365 → 0.653610 |
| Development music identity final error | 0.135301 | 0.110671 |
| Development music paired improvement/degradation fragment count | 22 / 11 | 21 / 12 |
| Development music paired error change median | -0.001172 | -0.002101 |
| Worst two fragments contribute to net increase in main music error | 92.17% | 82.64% |

The denominator in the last row refers to the **net increase** across all fragments, including both positive and negative changes, not the percentage of total error.
Two severe cases originate from different recordings: `9d95e7f035f878fa1a6ab421` (blues) and `d2599868e6a472eff47c1674` (unlabeled). These cannot be removed to declare success—they represent valid input failures. Removing these two fragments does not eliminate the problem: after removal, the average main error of the remaining music still increases by approximately 0.055735 / 0.048173. This deletion analysis is not used to modify the evaluation set.

The paired error for the protected 32 training music windows decreased from 0.602244 to 0.589545 / 0.589813. These windows still have overlapping input context constraints. Evidence supports a generalization gap across recordings—complete temporal generalization cannot be claimed verified. Speech improvements do not substitute for music recovery.

## Location of Amplification

In the same frame range from 89 to end of valid region, the following shows identity spectral components (including reference change error, excluding waveform term).
For identity, each stage uses the output of the initial stage as reference.

| Range / Seed | Spectral after mixing | After OLA reanalysis | After clipping |
| --- | ---: | ---: | ---: |
| Training music fixed window / 12 | 0.002355 | 0.003025 | 0.003008 |
| Training music fixed window / 13 | 0.002737 | 0.003259 | 0.003229 |
| All development music fragments / 12 | 0.035712 | 0.137102 | 0.135115 |
| All development music fragments / 13 | 0.047008 | 0.112079 | 0.110523 |
| Blues severe fragment / 12 | 0.115361 | 2.109583 | 2.109583 |
| Blues severe fragment / 13 | 0.102157 | 1.418578 | 1.418578 |

The primary transfer in the blues fragment is **12–16 kHz → above 16 kHz**. Removing only the 12–16 kHz correction reduces full identity loss from 2.109781 / 1.418697 to 0.055444 / 0.052442; the real Rust high-frequency spectral error drops from 4.118242 / 2.746885 to 0.101922 / 0.095809.
When only the source band is retained, the original damage is largely reproduced. Pre-clipping peak ≈ 0.691, below the 0.855 starting point, with gate=1—thus, this counterexample is not caused by limiter or gating switch.

Another severe fragment primarily transfers as **4–8 kHz → 8–12 kHz**. Removing the source band reduces identity loss from 1.780225 / 1.370762 to 0.238690 / 0.299738; pre-clipping peak ≈ 0.022, with no clipping involved. This fragment also exhibits direct damage from other bands, so removing a single source band does not fully resolve the issue.

No threshold on tracks or frequency band closures was established. Such development counterexamples should not be used to justify training gains. [P15 training intervention](band-transfer-study.md) has already shown that hard-band closures degrade paired recovery performance.

## What the Loss, Sampling, and Structure Reveal

- **Loss Implementation:** P19 actually uses `FixedFitObjective`, averaging 50% paired and 50% identity errors per batch, and also computes actual rendered spectral, time-variant reference, and waveform errors. It is not limited to supervising unrendered magnitude. The `RenderedObjective` includes historical 1/8 identity normalization, which is not part of this training path. Auxiliary term is reference perception `(P-I)(Zpred-Zref)`, not requiring the true transient to be eliminated. Recomputation confirms consistency; historical finite-difference, identity, and silence alignment checks pass. However, this does not prove the current weights are optimal, nor that they align with subjective audio quality.
- **Error Scale:** In the blues counterexample, RMS reference magnitude above 16 kHz is approximately `1.01e-4`, predicted error RMS is about `1.66e-3 / 1.11e-3`, full-spectrum reference RMS ≈ 1.819. Relative to weak band, this is large—about 0.091% / 0.061% of full spectrum; this is a DFT magnitude ratio, not overall loudness, audibility, or burst evidence. Actual spectral bias exists, but subjective severity assessment still requires blind listening and waveform measurement.
- **Training Statistics:** Across both seeds, only 155 / 169 updates triggered norm > 5 clipping, approximately 0.96% / 1.04%. Median gradient norm ≈ 0.552 / 0.498; isolated peaks exist, but no non-finite values occurred. The result cannot be explained by "generalized gradient explosion," nor can these statistics alone rule out all optimization issues.
- **Data Coverage:** The 253 tracks still represent one codec/channel configuration per song, with eight fixed short windows—this does not equate to learning across all songs or all degradation conditions. Training now includes blues; severe cases involving Vorbis q3, MP3 VBR q6, and AAC 192 are all observed codecs—not solely due to the left-out AAC 192. The activation rate of enhancement in identity input is 65.2% in training and 72.4% in development—this reflects bandwidth detection behavior, not compression recognition accuracy. Cases with reference below 16 kHz are rare: 18/253 in training, 3/33 in development. Training does not lack weak high-frequency content. Genre diversity does not guarantee coverage of spectral edges, transients, or multi-channel joint conditions.
- **Structural Limitation:** The current head independently controls energy and magnitude shape per five hard frequency bands, then reuses input phase. It does not guarantee consistent cross-frame correction synthesis at the representation level. Constraints in the training loss must generalize to new inputs. The P19 auxiliary term does not ensure this. This structure is a suspect point worth further investigation, though not yet proven to be the sole root cause. [P11 direct fitting](fixed-fit-diagnostic.md) already shows room for improvement in fixed-phase magnitude, so the claim that "phase correction is required for recovery" is not supported by current evidence, nor can it be directly inferred from small parameter count.

## Hypotheses to Validate Next

No further long-running training with identical configuration will be attempted. Error threshold relaxation should not be used as a path to promotion. Priority should be given to two separate controls:

1. **Output Representation:** With fixed data, backbone, starting point, and budget, compare the current independent per-band magnitude head against a complex residual head that enables cross-band coordination and phase correction. First verify training fit and synthesis transfer, then freeze and recheck generalization. Complex heads do not automatically ensure consistency—must be empirically verified. A single re-analysis/synthesis projection is insufficient, as the current deployment already performs synthesis, and the current loss already supervises the rendered output.
2. **Fidelity Generalization:** Isolate a comparison with more codec/channel/time-position rotation and expanded spectral edge conditions in training. Record per-recording risk distribution and assess whether tail damage is reduced. Maintain existing clean identity baselines; do not treat "adding back identity" as an untried method, nor use development tracks to set training thresholds.

Each path must simultaneously improve paired recovery and original signal fidelity. Simply reducing correction back to identity is not success.
The model continues to adhere to ≤2M parameters, ONNX <8 MiB, causality, and existing real-time budget. Any new exported model must undergo re-measurement of inference cost—results from this diagnostic round cannot be reused. No model or deployment path was modified in this round.

## Reproducibility and Output

Original P19 root directory: `data/runs/music-coverage-p19-20261014/study`.
New report: `data/reports/coverage-failure-20261014/`, containing registration, six stage reports, native interventions, summary, and input/code/source/checkpoint hash verification; audio and weights are not pushed.

```sh
PYTHONPATH=training training/.venv/bin/python training/scripts/audit_coverage_failure.py \
  --study data/runs/music-coverage-p19-20261014/study \
  --output data/reports/coverage-failure-20261014
PYTHONPATH=training training/.venv/bin/python training/scripts/replay_coverage_failure.py \
  --study data/runs/music-coverage-p19-20261014/study \
  --audit data/reports/coverage-failure-20261014
PYTHONPATH=training training/.venv/bin/python training/scripts/analyze_coverage_failure.py \
  --study data/runs/music-coverage-p19-20261014/study \
  --audit data/reports/coverage-failure-20261014
```

The first command requires a new output directory; during reproduction, modify the output/audit paths accordingly. Before audit, record the source; after execution, verify source code has not changed. During summary, recheck original inputs, source code, and stage report hashes.
Verification: 29 loss decompositions, band interventions, synthesis consistency, coverage sampling, and break recovery tests passed.

The retained codec counterexamples include the stated 160 kbps condition; codec labels and paired-file evidence remain in the frozen per-case reports.
