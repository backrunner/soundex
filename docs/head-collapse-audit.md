# P27: Residual Decoder Deactivation Diagnosis and Convergence Path

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

**Follow-up progress:** [P28 Small Model Mechanism Comparison](small-mechanism-study.md) has implemented three single-factor interventions and completed fixed-budget training, without establishing recovery gains. The next step is [P29 Direct Spectral Supervision](direct-spectrum-study.md). Below retains the positioning and planning from the P27 timepoint; the new DSP incentive branch remains unimplemented.

**2026-10-19:** Training of the 640-width, 11,924,103-parameter model from P26 has been halted.
No overall recovery gain was established across eight complete training trajectories or native deployment rechecks. The endpoint of the large model has been shifted to historical baselines and is no longer loaded into default weights or considered for expansion in the next round. This represents an engineering trade-off in the current representation, structure, and objective design—not a conclusion that "the large model is ineffective for audio restoration."

---

## Verified Mechanisms

Diagnostic tool: `training/scripts/audit_head_collapse.py`.
Prior to scoring, a fixed selection rule is registered:
Take the first seed from P26, and select draw indices 0, 5397, 10794, and 16191 from the broad training schedule.
Each batch consists of 4 music and 1 speech segments, with corresponding clean identity inputs added.
Twenty paired cropped input windows are evaluated, fully isolated from the development/shifted intervals.
Eight final endpoints are inspected; no training, no new candidate selection, no held-out scoring is performed.
All following results are derived from this specific training mechanism and cannot be extrapolated to overall music quality or used to infer overfitting.

1. **GELU Hidden Layer Deactivation in the Residual Decoder.**
   Across four broad endpoints and four paired input batches, the proportion of activations with `abs(activation) < 1e-8` is **72.47%–99.62%**.
   Pre-activation mean values range from **−7.03 to −16.34**; 67.15%–99.30% of pre-activation values are below −6.
   These are diagnostic thresholds and are **not used** in deployment, training release, or track curation.
   Output `tanh` values with `abs(scaled_logit) > 3` are all zero—this cannot be attributed to tanh saturation.
   The hidden layer has already entered a large portion of the negative saturation region of GELU under current distribution.
   Expansion has not translated into effective residual prediction.

2. **Not due to budget being fully zeroed, nor due to detector bypass.**
   Broad endpoint budget RMS ranges from **0.0312 to 0.1899**.
   Paired activation frame proportion ranges from **80.0% to 88.1%**.
   RMS difference between the output and identity DSP in PCM space is only **8.2e-9 to 4.0e-7**.
   Five-band analysis shows that some original residual content in low frequencies is lost during the existing band fusion process;
   the absence of effective high-frequency residual is inherently small.
   It is therefore invalid to claim that raw, unprocessed high-frequency residual volume can be used to compensate for missing high frequencies.

3. **Same DSP and target still allow for improvement.**
   The 20 paired cropped inputs show the following mean spectral error after deployment proxy:

| Diagnostic Input | Spectral Error After Deployment | Relative to Identity DSP |
|---|---:|---:|
| Identity Input   | 0.433288 | — |
| Reference Magnitude + Input Phase | 0.319498 | −26.26% |
| Reference Magnitude + Reference Phase | 0.184423 | −57.44% |

   Reference inputs are used only for oracle-level diagnosis and are **not deployable**.
   These results are from pooled training diagnostics on Torch DSP proxies—**not** new model performance, native Rust evaluation, or perceptual quality.
   Full Rust pipeline results for P26 are available in [Completion Report](broad-budget-study.md).
   Improvement potential exists, but this table does **not prove** that the current network can predict the oracle.

4. **Loss conflict and shared gradient imbalance exist, but no evidence supports removing identity constraints.**
   On the first batch of fixed training samples, gradients from paired spectrum, paired waveform, identity, complex consistency, and budget were computed under original training weights.
   At the four broad endpoints, the budget supervision gradient norm for shared parameters was **176, 22, and 4847 times** that of the paired spectrum term in three cases, and 0.93 times in one.
   The cosine of shared gradients between paired spectrum and identity were negative in three cases and positive in one.
   This is local evidence from a single batch and final endpoint—**cannot be used to conclude** that the entire training process is dominated by budget or identity loss.
   Particularly, loss magnitude does not imply gradient magnitude; the presence of budget branch learning does not imply effective residual branch performance.

**Source and Hash:** `data/runs/head-collapse-p27-20261019-v2/`, SHA-256 hash:
`33a4e4f745604658cfeb5d723e0778f3b0ad059bc665424222ad4af85dd3eb42`.
All eight checkpoints match the final endpoint hash of P26.
The first version of diagnosis is preserved in `data/runs/head-collapse-p27-20261019/`; the second version includes pre-activation values and per-band residual localization.
Neither version altered the P26 model, objectives, registration, scoring, or frozen source code.

---

## Limitations in Representation and Objective

The current `RestorationHead` is termed a "complex head," but `EnvelopeProbe.encode` only accepts:
short-window magnitude, long-window magnitude, global scale, and detection conditions.
Phase information is only rotated at the output and is **not fed as a learned feature** into the network.
This prevents the model from distinguishing between inputs with identical magnitude but different phase or instantaneous frequency.
In missing bands, relying on original phase rotation for carrier generation may not yield stable output.
This is a structural limitation and does not justify attributing all degradation to it.

The primary objective performs regression on magnitude and waveform after DSP, with a complex auxiliary term constraining `(analysis(synthesis(ΔZ)) - ΔZ)`.
This is **not** a direct spectral recovery error.
Therefore, "having complex output and consistency loss" does not imply sufficient phase modeling or high-frequency generation supervision.
It remains uncertain whether high-frequency point-by-point regression, identity preservation, and shared budget gradients collectively lead to conservative behavior—this hypothesis requires controlled ablation in the next phase.
The current diagnosis identifies a deactivation location, but does **not claim** it is the sole cause.

---

## External Methods and Applicability Boundaries

Review of papers and official implementations up to 2026-10-19.
No third-party pre-trained weights, datasets, or source code were introduced into training or released artifacts.
Any future integration would require independent verification of actual file licenses and training sources.

| Method / Source | Mechanisms that may be borrowed | Limitations and Trade-offs for SoundEx |
|---|---|---|
| [Opus 1.6 BWE](https://opus-codec.org/demo/opus-1.6/); [Paper](https://arxiv.org/html/2412.11392v2); [Official BBWENet](https://github.com/xiph/opus/blob/main/dnn/torch/osce/models/bbwe_net.py) | ~370K parameters for non-linear excitation, time-varying filtering, and shaping; use of instantaneous frequency features; spectral adversarial loss after pre-training | Limited to 8–20 kHz speech extension; cannot serve as a general music restoration weight. Paper uses 10 ms feature frames and ~0.27 ms signal path latency—cannot be ported to our 128-frame input with added 128-frame delay. Prioritize borrowing DSP excitation and small controller design separation. |
| [AERO Paper](https://arxiv.org/abs/2211.12232) and [Official Implementation](https://github.com/slp-rl/aero) | Real/Imaginary spectrum U-Net, reconstruction, and adversarial/feature matching supervision, including music experiments | Official music experiments involve MUSDB18-HQ at 11.025→44.1 kHz resampling—does not equate to multi-codec music restoration. No direct reuse of their dataset or full architecture. |
| [AP-BWE Paper](https://arxiv.org/abs/2401.06387) and [Official Model](https://github.com/yxlu-0102/AP-BWE/blob/main/models/model.py) | Dual-branch input (magnitude and phase), mutual interaction, magnitude-phase discriminator | Official generator uses kernel-7/padding-3 convolutions reading future time positions—requires causal re-implementation. CPU 18.1× real-time is throughput, not proof of hop-level deadline compliance. Main evidence is speech-based. |
| [DSP-informed BWE](https://arxiv.org/abs/2407.15624) | Separation of excitation spreading and time-varying filtering, controlled by acoustic features | 8→48 kHz speech experiments; music and polyphonic effects require local validation. Supports splitting "high-frequency generation" and "envelope control" for independent validation. |
| [PEBE，2026](https://www.mdpi.com/2076-3417/16/3/1439) | Parallel branches for encoding damage repair and bandwidth extension, trained independently | Designed for encoded speech; not general music restoration. Modified AP-BWE baseline still shows 15 ms algorithmic delay; we can only borrow task decomposition, not the buffer design. |
| [DeepFilterNet2](https://arxiv.org/abs/2205.05474) and [Official Repository](https://github.com/Rikorose/DeepFilterNet) | Low-cost envelope and complex filtering, full real-time engineering implementation | Primarily for speech denoising; filtering existing frequencies ≠ generating missing ones. Serves as a real-time implementation reference, not a baseline for successful high-frequency restoration. |
| [A2SB, 2025](https://research.nvidia.com/labs/adlr/A2SB/) and [Official Code](https://github.com/NVIDIA/diffusion-audio-restoration); [Single-step Music Flow Matching, 2026](https://arxiv.org/abs/2601.14356) | Generative modeling for music restoration and spectral contour conditioning | Closer to music tasks, but lacks evidence of hop-by-hop causality with CPU <5 ms. Single-step does not imply low latency. Retained for offline quality research only—excluded from current deployment path. |

Source code snapshots are preserved in the ignored directory `data/research/20261019-upstream/` for reading only.
Git blobs reviewed:
- Opus BBWENet: `66c8d92ad665b1d4c0793e289e7e37a445bbccd4`
- AP-BWE model: `ca89fff7d62eb9818176a17ba93eb56e939a3586`
- AERO model: `c50a01f51b17a9f4e3abed83d326bba8d0006ca5`

---

## Next Experimental Sequence

No further training or expansion of the old 640-width model will be initiated.
Prioritization will be on validating compact causal controllers and DSP excitation / spectral shaping pathways:
- Separate supervision for preserved band restoration and missing band generation
- Maintain shared control over band fusion and low-frequency protection
- Enhance controller with reliable complex and instantaneous frequency information
- High-frequency carrier generation must **not** rely solely on unreliable phase from missing bands

Multitrack music, noise-like high frequencies, transients, and stereo continuity must be tested independently—**cannot be assumed to follow speech-based single-fundamental assumptions**.

First, use small-scale, matched diagnostics to isolate:
- "Decoder normalization / negative region gradient"
- "Budget gradient sharing"
- "Phase information in input"

Do **not** modify all three simultaneously and then attribute success to one.
Record gradient and effective activation per supervision target per branch, and confirm that the model actually generates high-frequency content before proceeding to full budget, multi-seed training, and frozen endpoint recheck.
These structural and experimental designs remain unimplemented and thus **cannot be considered** as new weights having been trained.

Loss remains centered on original lossless reference: spectral, energy, transient, and preserved band fidelity.
If discriminators or feature matching are introduced, they are used **only during training**, not in inference.
However, any gains must be evaluated jointly with pseudo-high-frequency artifacts, waveform errors, and blind listening.
**No improvement in oracle performance guarantees improved real-world prediction.**
Forcing output of more high frequencies does not equate to improved restoration quality.

Ultimately, the system must still pass:
- Full Rust pipeline
- Hop-by-hop throughput
- Paced deadlines
- CPU/RSS and stability validation

Model + DSP computation <5 ms (excluding fixed buffer);
for 48 kHz / hop128, the service budget remains **2.667 ms**.
The new path has **not yet provided evidence** of meeting this requirement.
A fixed 256-frame buffer is still required and must be included in the total output latency.

---

## Reproducibility of Diagnosis

```bash
rtk proxy env PYTHONPATH=training training/.venv/bin/python \
  training/scripts/audit_head_collapse.py \
  --study data/runs/broad-budget-p26-20261017-v2/study \
  --output data/runs/head-collapse-audit-new
```

The script uses original training data and reference targets, preserving source code, checkpoints, and selection rule hashes.
Results are stored in the ignored directory.
Original model, default weights, training loss, and Rust DSP were **not modified**.
Passed 548 Python unit tests and Ruff checks.
Fully executed diagnostics across eight endpoints, and after deleting old cache, re-generated a batch of inputs to produce **identical results** as before cleanup.

The registered set contains 20 paired crops. The output-tanh saturation diagnostic `abs(scaled_logit)>3` is zero throughout; this is distinct from the observed negative GELU saturation. Paired active-frame coverage is 80.0%–88.1%; RMS change relative to identity DSP spans `8.2e-9` to `4.0e-7`.
