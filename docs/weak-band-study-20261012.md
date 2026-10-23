# P3: Weak High-Frequency Constraint, Identity Control, and Generalization Verification

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

Following [P2 Seed Recheck](causal-p2-review-20261012.md), this round treats the baseline as a subject for validation, and does not consider matching the prior model as a sign of recovery success. The goal remains to closely approximate lossless reference material, with actual performance gains over the unprocessed input.

Source code: `ee06142`, running in an isolated checkout; no modifications to the training script were made during P2.

---

## Loss and Architecture Review

A new loss function `causal-weak-band-v1` is introduced, retaining all components of the original `causal-reconstruction-v1`: full reconstruction, transient, low-frequency, and waveform terms, plus a new penalty term with weight 0.25 for excess energy in the output frequency bands. Reference data is used only during training; no reference is read during inference.

- Fixed frequency bands at 8/10/12/14/16/18/20 kHz up to Nyquist, with thresholds unchanged across tracks, genres, or counterexamples.
- Uses actual synthetic proxy outputs, computing single-sided Parseval power per band across three FFT scales: 256, 512, and 2048, with Hann window power normalization applied to compensate for windowing effects; the final value is the average across the three scales.
- Per-band/frame loss is defined as `log1p(relu(Poutput−Pclean)/(Pclean+floor))`.
  - `floor = max(1e-10, 1e-6 × total reference power)`; the normalization term is detached to prevent gradient backpropagation.
  - The lower bound corresponds to −60 dB relative and −100 dBFS absolute power; we do not enforce a strong global 1% amplitude floor.
- The reference signal has no such penalty, and the gradient of this term is zero. Missing high-frequency content is still penalized by the original reconstruction term.
- Output cannot be made dry or artificially suppressed in high frequencies as the optimal target. This term alone cannot distinguish noise texture or phase artifacts; it must be combined with existing spectral shape, complex-valued outputs, transients, actual output, and blind listening.

**Verification performed:**
- Finite-value checks for silence and weak-band signals at two sample rates.
- Zero penalty and zero gradient for the extra term.
- Finite difference gradient check passed.
- Manual Parseval validation for sine waves confirmed.
- High-frequency reconstruction in the presence of clean reference outperforms both dry and artificially boosted high-frequency versions.
- CPU/MPS total loss difference on a single real-cropped test: 6.68e−6; gradient relative L2 difference: 5.25e−5—both negligible.
- This is a numerical probe, not a claim of exact loss trajectory alignment. Full Python test suite passed 300 cases.
- MPS internal resize warning remains active; not silenced, and CPU fallback not enabled.

All GRU and TCN architectures maintain the ability to express identity, and add energy only in specified high-frequency bands. Causal, channel-independent, irregular chunking, and backward silence handling are preserved. No new architectural parameters, inference operators, state variables, or audio buffers are introduced.
TCN remains at 639,906 parameters; short training ONNX model size: 2,599,024 bytes—within the pre-defined volume budget.

We cannot claim that any lossless target can be exactly expressed: the residual range of `2*tanh` is bounded by the RMS of the input spectrum.
Among 256 training clips and 89 development clips, the proportion of high-frequency real and imaginary components exceeding the output representable range is approximately 0.0863% and 0.0118%, respectively. After projection into the representable range, the resulting squared error relative to target energy is about 0.236% and 0.525%. This is particularly pronounced when input energy is near zero.
This indicates that a small subset of targets is constrained by hard amplitude limits, but does not support the conclusion that "increasing model size will solve the main degradation."
It also does not guarantee that any mapping within the output range can be learned by a finite network.
This round isolates the influence of loss and data control; structural limits are preserved for independent ablation in future work to avoid confounding causes.

---

## Three Fixed Groups, Two Seeds

Each seed is initialized from the P2 TCN best checkpoint, with identical AdamW optimizer state; no inheritance of prior optimizer state.
Each group undergoes 1024 updates per batch of 4, learning rate 1e−4, cosine decay reduced to 0.1, gradient norm capped at 5.
Validation and checkpoint saving occur every 256 steps, up to a maximum of 6 groups; no infinite continuation is allowed due to poor performance in any one group.

| Group | Training Objective | Sampling |
| --- | --- | --- |
| control | Original v1 | Original paired lossy → lossless |
| excess | v1 + new energy penalty | Identical to control |
| excess_identity | Same as excess | Every second step, one slot replaced with clean→clean, 12.5% coverage |

All groups share the same random cropping and gain modulation (−6 dB to +3 dB), with per-recording equalization applied. Music/voice source probabilities remain at 253/254 and 1/254, respectively.
The identity group does not increase update budget; the replaced slots are independent of content or evaluation results, resulting in 12.5% reduced exposure to lossy samples—this is part of the ablation design.
Clean→clean transitions are reprocessed through the input detector, and the original lossy cutoff is not disguised as an identity input.
No runtime label is provided to the model indicating "this is an identity sample."

All groups are scored using the **same new objective**:
`0.875 × original paired validation score + 0.125 × identity validation score`,
with scores aggregated by music/voice source weight. The lowest-scoring checkpoint is preserved, and both `base_total` and the new term are retained.
Raw total scores from different training objectives should not be used to draw structural conclusions. The original parent weights are also re-scored under the same objective before update.
All results remain in experimental status; score degradation does not trigger promotion.

---

## Preventing Learning Only from Validation Counterexamples

- Beyond track/source checksum isolation, we verify that split_group/work_group does not cross validation boundaries.
- 40 fixed training audio probes (32 music, 8 speech) are used; each validation is re-computed under the same objective. This monitors only the gap between training and development domains, and is not used in checkpoint selection. The loss of a random training batch cannot substitute this comparison.
- The original 89 development validation clips and corresponding clean→clean controls are fixed; metadata includes music/voice, encoding, sample rate, channel role, genre, and region.
- For each test recording, a full, encoded pair is preserved—39 recordings total. These are only evaluated after the best checkpoint is frozen and scored via actual Rust evaluation; they are not used in checkpoint selection. This is a historical held-out recheck, not a new independent test.
- All actual scoring is clustered by `track_id` using bootstrap; a bug in P2's automatic reporting (which previously aggregated by row) has been fixed.
- Stereo output is not constructed from independent channels. The full paired files remain short (2–4 seconds), and are not claimed to represent full songs, long streams, physical DACs, or synchronized stereo validation.

Gaps remain: 253 music training, 26 validation, 21 test.
Validation lacks country/pop/hip-hop/funk; test lacks Chinese, country/hip-hop/jazz.
Therefore, we cannot confirm generalization across all genres or regions, nor can we guarantee absence of overfitting.
The purpose of the identity group is to detect and reduce over-generation and selection bias; independent validation on truly unseen sources still requires additional data.

---

## Engineering Evidence and Execution

All three small-scale groups (CPU and MPS) completed training, export, and actual Rust paired/identity scoring. The latest smoke test also executed a held-out recheck on full paired files and the same training probe objective. This is **not** a claim of quality improvement.
Reports are stored in `data/reports/causal-p3-20261012/`, including `capacity-audit.json`, `mps-audit.json`, and `smoke-frozen/`.

During performance tooling setup, a Cargo rebuild replaced the shared target binaries for streaming scoring and detection. A hash check rejected one smoke launch. The binary was recovered from dependencies and restored to match the P2-registered SHA exactly.
The P2 cache timestamp was earlier than the rebuild, and no P2 scoring report was generated during that interval; the original training remained unchanged.
P3 now copies the binary to its own `binaries/` directory and verifies the hash; future builds will not alter previously registered experiments.

Formal run started at local time 2026-10-12 02:53. All six groups are now complete. Output directory: `data/runs/causal-p3-20261012/study/`; log file: `train.log`. No weights were promoted automatically.

Formal entry point: `training/scripts/run_weak_band_study.py`, which requires all six P2 groups to be complete before launch.
Command usage: `--parent <P2 directory> --output <independent output directory> --detector <registered detector>`.
All source code, parent checkpoint, draws, environment, loss constants, and selection rules are fully logged. File locks prevent duplicate launches; interruption is recoverable.
Training PCM, cache, checkpoints, and reports are not pushed; source code remains under Apache-2.0 license.

Overfitting and clipping attribution in the baseline is referenced in [Baseline Generalization Recheck](baseline-generalization-20261012.md).
Existing DSP can corrupt identity inputs; P3 maintains a frozen pipeline and cannot claim resolution of application issues based solely on this round's loss improvement.
Real-time cost measurement must use a frozen candidate SHA and must avoid mixing with training or concurrent evaluation—results from such runs cannot be claimed as deployment-ready. No model promotion occurred in this round.

---

## Post-Training Judgment

All results use the original deployment clipping limits. The following does not include future smooth-knee experiments.

| Seed | Group | best step | Common Selection Loss | Paired Validation Loss | Identity Validation Loss |
| --- | --- | ---: | ---: | ---: | ---: |
| 20261012 | control | 768 | 2.25825 | 2.47907 | 0.71252 |
| 20261012 | excess | 768 | 2.23960 | 2.46637 | 0.65220 |
| 20261012 | excess_identity | 768 | 2.22188 | 2.47485 | 0.45113 |
| 20261013 | control | 256 | 2.27838 | 2.50421 | 0.69758 |
| 20261013 | excess | 256 | 2.26598 | 2.49802 | 0.64171 |
| 20261013 | excess_identity | 256 | 2.24232 | 2.50031 | 0.43641 |

The inclusion of identity samples reduced the common selection score, but the primary benefit comes from the identity subset. The paired recovery score does not fully surpass the excess-only group. Best steps for both seeds occur early—before the final step—and there is no evidence supporting indefinite continuation.
Training probes contain only paired samples and must be compared against development set **paired** scores. The common selection score with 12.5% identity weighting cannot be used to compute generalization gaps directly.

Historical held-out music: 21 recordings, each with a full paired file. Actual Rust steady-state performance:

| Seed / Group | High-Frequency Spectral Shape ↓ | High-Frequency Transient ↓ | High-Frequency Energy Error (dB) ↓ | SI-SDR vs. Input (dB) ↑ |
| --- | ---: | ---: | ---: | ---: |
| 20261012 / control | 0.67888 | 0.90480 | 3.15370 | −0.30885 |
| 20261012 / excess | 0.67014 | 0.89659 | 3.33452 | −0.30683 |
| 20261012 / excess_identity | 0.66342 | 0.88667 | 3.46710 | −0.30501 |
| 20261013 / control | 0.67511 | 0.89943 | 3.20499 | −0.30719 |
| 20261013 / excess | 0.66924 | 0.89319 | 3.38091 | −0.30555 |
| 20261013 / excess_identity | 0.65872 | 0.87940 | 3.66740 | −0.30302 |

Reference input values: spectral shape 0.69998, transient 0.86321, energy error 4.48301 dB.
The new constraint reduces spectral and transient errors in both seeds, but weakens energy recovery. Relative input transient and waveform fidelity still fall short of target.
Development music exhibits higher spectral distortion than input, and we cannot declare a general performance improvement or universal applicability.
The clean→clean baseline spectral error dropped from 0.85325/0.81154 in control to 0.60229/0.55983—still not transparent.
The identity input baseline SI-SDR is near theoretical upper limit; the large dB gap is not indicative of general lossy music quality ranking.

Next step: isolate DSP effects, see [Smooth Knee Comparison](limiter-study-20261012.md).
This round did not increase model size; constrained high-frequency recovery must be re-designed under a new deployment proxy, not simply by increasing the penalty weight.
