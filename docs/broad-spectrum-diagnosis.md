# P33 Fixed Endpoint Mechanism Diagnosis

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

Following [P34 Small Model Tuning](risk-balance-study.md), three rounds and five budget endpoint scenarios have been completed.
The observed gains and costs associated with risk compression, identity weighting, and over-completion constraints remain under active evaluation, with all candidate hypotheses retained in experimental status.

On 2026-10-21, building upon [P33 Full Training Results](broad-spectrum-study.md). This iteration does **not** include training, nor any adjustment to production DSP or default weights. Conclusion: **The majority of randomly sampled music exhibits weak recovery performance, and cannot be attributed solely to generalization failure; imbalanced weak-side signal scaling and competition between restoration and identity objectives warrant immediate attention.** No definitive causal link has been established through training intervention, and thus we do not conclude either model capacity deficiency or convergence.

---

## Methodology and Evidence Boundaries

Four fixed endpoint configurations are maintained. For each seed, 16 batches are selected at equal intervals across 16,192 actual update sequences.
Each seed includes 64 music + 16 speech training clips, plus corresponding identity references. Control and `mr_add` variants use identical clip segments.
Music coverage spans 60/57 recordings, speech spans 16/15 samples. All samples appear in training, but this is not a full re-evaluation of the training set, nor an independent test.

Prior to execution, segment selection and endpoint hashes are recorded. Normal prediction and budget-1 inference ablation are then run, generating 1,280 role predictions. For each model, gradients are analyzed across the fixed batches 0, 5, 10, and 15—totaling 16 batches.
The 80-frame real history, supervised region, original unclipped reference, and MR v1 are retained. Raw model OLA is reported; this Python-based diagnostic is not considered a new Rust or hardware validation. Neither normal prediction nor budget ablation results in parameter updates.

Cross-verification confirms 180 current execution dependencies match the P33 frozen source code hash; manifest and sampling file integrity checks pass. Actual cropping is validated by the original loader via PCM checksum. Weight hashes remain unchanged before and after execution.
For the most extreme relative error case, a post-hoc explicit tagging and full batch sample retention is performed.

Local evidence: `data/runs/p33-mechanism-audit-20261021/` containing `registration.json`, `selection-*.json`, four endpoint reports, `summary.json`, `outlier.json`, and execution script.
Experimental reports, scripts, and data remain in the ignored directory and are not uploaded; raw audio or experimental weights are not shared.

---

## Training-Seen Music Does Not Show Universal Improvement

Using common MR v1, performance is evaluated per recording average—lower is better. The two seeds differ in sample composition, so absolute values cannot be directly compared across seeds.

| Seed | Input Mean | Control Mean | mr_add Mean | Control Per-Recording Median Change | mr_add Per-Recording Median Change |
|---|---:|---:|---:|---:|---:|
| 20261012 | 2.245261 | 1.575598 | 1.508436 | −0.379% | **−0.165%** |
| 20261013 | 0.359720 | 0.358622 | 0.357525 | −0.225% | **−0.248%** |

Median values represent the median relative change per track, not the ratio of means. Two tracks per seed scored zero in input evaluation; percentages are undefined for these. They are preserved in both mean and per-recording reports, excluded from median percentage calculations.
It is incorrect to infer training success from a 33% drop in input mean for the first seed: **99.77% of the mr_add mean improvement in this seed stems from a single side channel clip**—primarily through noise suppression, as detailed in the next section.

Per-channel recheck results remain: mr_add left/right/mono mean improvements are 0.187%/0.252%/0.085% and 0.959%/0.458%/0.182%; only two mono clips per seed are available, insufficient to establish monaural coverage adequacy.
Mid, side, and all degraded examples remain preserved.

Thus, the current evidence better supports the claim that **"a majority of music lacks effective fitting"**, rather than indicating strong performance on training music that degrades only on new recordings. This assessment applies only to the fixed diagnostic set and does not invalidate concerns of overfitting, distribution shifts, or historical validation selection bias.

---

## Weak Side Signals Reveal a Loss Scaling Issue

The most extreme case involves training music at AAC 96 kbps, 44.1 kHz, side-view. Post-hoc batch inspection reveals:

| Metric | Value |
|---|---:|
| clean PCM RMS | 3.7961e-5（**−88.41 dBFS**） |
| Input relative clean PCM RMS error | 0.00181175（−54.84 dBFS） |
| mr_add output PCM RMS error | 0.00090682（−60.85 dBFS） |
| Input → mr_add (MR v1) | 115.5232 → 71.4138 |
| Direct + MR gradient norm (0.1 batch weight) | **50.43** |
| Max corresponding gradient norm of other four paired samples | **0.01651** |

This segment produces a gradient magnitude approximately **3,054 times greater** than the maximum observed in the other paired samples within the same batch.
Normalization against a reference scale amplifies the relative weight of compression artifacts in extremely weak side signals; while global gradient clipping limits total norm, it fails to balance directional contributions within the batch.

The primary benefit is reduction in 8–12 kHz noise: RMS error drops from 0.001765 → 0.000862.
Concurrently, absolute errors in 0–4, 4–8, and 16–24 kHz bands **increase**. Therefore, the overall score decline cannot be interpreted as successful high-frequency inpainting, nor can the −88 dBFS level be used to infer perceptibility or justify removal of this counterexample.
These frequency bands are evaluated using rectangular FFT projections from the supervised region and correct PCM scaling—used solely for diagnostic purposes, not for deployment filtering.

Historically, across four training tracks, there were **186 pre-clipping gradient norm exceedances** above the original threshold of 5. Of these, **180 occurred in batches containing side signals**. Each trajectory had approximately 58% of batches containing side content.
The mr_add variants achieved peak gradients of 404.93 and 367.10, respectively. However, these events represent only about **0.29%** of the total 64,768 updates. Importantly, "batch contains side" does not imply the side signal caused the gradient spike—thus, it cannot be inferred that side signals dominate the training process or explain all stagnation.

Side, weak-energy, and silence content must be preserved. A re-evaluation of normalization and gradient contribution is required. During training, the side signal scale should be compared against the original stereo reference energy, or a joint objective should be defined that accounts for absolute error in the compressed domain. The specific form must be pre-registered and validated through silence, weak-energy, reference-zero gradient, finite-difference, and non-homologous left/right channel checks.
No exclusion rule should be established for this piece, nor should scoring criteria be altered to artificially improve results.

---

## Target Competition Exists, But Cannot Be Directly Removed

Among the eight pre-selected batches for mr_add, the cosine of shared parameter gradients between paired restoration and identity remain consistently **negative**, ranging from **−0.863 to −0.158**. At the budget head, the range is **−0.995 to −0.227**.
Identity shared gradient norms range from **7.1% to 174.7%** of paired gradients—enough in some cases to significantly influence direction.

In contrast, the cosine of direct spectrum and added MR gradient contributions is **+0.564 to +0.996**.
This does not support the interpretation that "new spectral objectives are generally opposed to the original direct spectrum objectives." The direction and relative strength of budget-augmented gradients vary per batch, and thus cannot be used to attribute all issues to auxiliary terms.

Negative cosine values indicate local competition, but do not imply identity design error, nor establish causal training failure. In the next round, under identical architecture, data, and sufficient budget, a comparative analysis of target scaling and gradient ratios should be conducted.
The original lossless identity, low-frequency response, transient fidelity, stereo imaging, and continuity should be preserved—rather than being removed outright.

---

## Excluded Simple Explanations

- **Decoder-wide inactivation is not the primary manifestation in this small model.** For mr_add music, the GELU negative saturation ratio is only 0.001% / 0.010%, approaching zero at 0.090% / 0.015%. This does not support the inference that the previous large model's inactivation pattern applies here, nor does it prove the architecture is robust.
- **Forced budget amplification does not resolve the issue.** After enforcing a budget of 1, mr_add music MR mean increased from 1.50844 → 2.06841, 0.35753 → 0.36329; identity increased from 0.000164 → 0.025945, 0.000476 → 0.028966—clearly degrading fidelity. This is a post-hoc inference ablation, not a retraining under fixed budget, and thus cannot be used to reject all budget modifications.
- **We cannot blame DSP alone.** The original OLA improvement in this training set is already weak. The P33 held-out recheck shows only 0.166% / 0.251% improvement in OLA. The P32 fusion loss remains, though it is not the sole issue in this round.
- **No regression in loss values was observed within the tested set.** Re-running MR, direct spectrum, and residual budget tests yields **37 passing cases**, including gradient, silence/weak-signal, and derived checks. Five ONNX warnings persist. Mathematical behavior passing does not imply that the objective trade-offs are suitable for the real training distribution.

---

## Next Round Priorities

1. **Address training scale issues**: Differentiate between restoring missing energy and suppressing extraneous artifacts, and report gains per frequency band. Establish a unified scale for both absolute and relative error in weak side signals. Retain MR v1 joint scoring and all counterexamples.
2. **Isolate and compare identity gradient ratio and output parameterization** in the small model. Current output is modeled as "**budget × small slope residual × input phase carrier**". Direct high-frequency amplitude and envelope prediction should be compared against independent phase residual predictions—this remains a hypothesis to be tested, not a justification for increased energy as equivalent to spectral recovery. No changes should be combined into a single composite.
3. **Confirm, under sufficient budget and dual-seed training, that most training music shows measurable improvement** before evaluating generalization via recording isolation recheck. Subsequently, validate actual Rust + DSP, transient response, stereo fidelity, blind listening, and per-hop latency.
   No evidence supports immediate model expansion or unmodified continuation training.

Industry methods may support exploration of alternative generative representations, but performance claims cannot be directly transplanted:
- [BEHM-GAN](https://arxiv.org/abs/2204.06478) uses complex spectra and adversarial learning, trained on classical solo piano and historical recordings;
- [Streaming SEANet](https://arxiv.org/abs/2010.10677) employs feature and adversarial losses on 8→16 kHz speech, with 16 ms streaming latency.
Neither demonstrates that 48 kHz multi-track compression recovery, 128-hop processing, or model + DSP <5 ms latency is achievable.
Currently, priority is to correct identified scale and target competition issues. Generative priors may be explored in future, but should not be immediately replaced with large-scale GANs.

No new training or re-verification of latency has been initiated. The P33 timeout, incomplete blind listening, and equipment acceptance conclusions remain valid.

The frozen-budget inference ablation sets budget to 1. Gradient analysis uses
registered batch indices 0, 5, 10 and 15 for each model. The two seeds respectively
have 2 and 1 music recordings with zero input score: retain them in recording reports
and means, but exclude undefined relative percentages from percentage medians.
