# P29: Direct Lossless Spectral Supervision and DSP Dependency Diagnosis

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

The subsequent [P30 Phase-Swap Diagnosis](phase-swap-audit.md) has been completed, identifying phase limitations and recovery ceilings in the fusion stage.

**2026-10-19**, building upon [P28](small-mechanism-study.md). The goal is to determine why the model still approaches identity output. First, we assess the learnability of a small model on fixed training materials, without expanding model size or selecting formal weights.

---

## Confirmed Supervision Path

The primary component of the current P28 `FixedFitObjective` applies the following pipeline: model output is passed through a detection mask, amplitude-phase fusion, OLA, gating, and limiter, then re-analyzed in the frequency domain and compared against an original lossless reference. The paired target does not refer to the DSP output; however, the main objective does not directly constrain the model's spectrum prior to DSP processing. The identity role uses identity DSP as a fidelity reference.

The `consistency_loss` enforces consistency between the corrected synthesis and re-analyzed spectrum, but does not directly constrain amplitude or complex spectrum reconstruction. As such, the current objective suffers from a dependency on gradient transmission through the deployment chain. However, this alone does not establish that the DSP is the sole cause.

Historical `RenderedObjective` included raw metrics, which do not imply that the current optimizer performed backpropagation on them. In this round, we analyze the actual call chain `run_head_study → FixedFitObjective` to ensure that reported fields are not being used as training objectives.

---

## Direct Reference Target

A new file, `training/models/direct_spectrum_objective.py`, has been introduced. The model input and deployment structure remain unchanged, with only the training process now using raw, lossless amplitude, complex spectrum, band energy envelope, and inter-frame dynamics:

`raw = amplitude + 0.1 × complex + 0.1 × envelope + 0.25 × dynamics`

The five bands are 0–4, 4–8, 8–12, 12–16 kHz and 16 kHz–Nyquist. The low-band weight is 0.25 and each other band has weight 1, normalized by the total. Per-band error is normalized by reference-band power plus `1e-4` times full-reference power and an absolute floor of `1e-14`, using a `1e-3` smooth norm. All scales and weights are fixed before execution, not fitted to known counterexamples. Calibrated absolute PCM error remains a separate native assessment.

Direct supervision is not multiplied by detector mask/active and does not pass through DSP. Identity targets the original input; paired roles target the original lossless spectrum. Magnitude-weighted complex distance constrains meaningful phase without assigning meaning to silent-bin phase angles. Dynamics compares prediction/reference frame changes, rather than penalizing the reference's own transients. Reference scales are detached throughout and never enter the model or deployment inference.

This is an explicit, direct reconstruction objective. Compared to the previous deployment objective, it introduces new error metrics and normalization. It is not merely an experiment that cuts the DSP Jacobian path. Results are only valid for the specific combinations tested and cannot be overinterpreted.

---

## Four Matching Control Groups

All four groups use the same width-96, 431,015-parameter complex head, identical `residual_slope=1/64`, residual budget, and initial parameters. No P28 normalization or phase input interventions are introduced.

| Group | Optimization Objective | Residual Budget |
|---|---|---|
| `deployed` | Old deployment objective + original-weight complex consistency + 0.1 budget supervision | Learning |
| `raw` | Direct lossless spectrum + 0.1 budget supervision | Learning |
| `joint` | Direct lossless spectrum + full old deployment objective + 0.1 budget supervision | Learning |
| `raw_unit_budget` | Direct lossless spectrum | Fixed at initial 1; budget head not trained |

The `raw` group is compared against `joint` to evaluate the impact of adding the deployment objective. `raw` is also compared against `raw_unit_budget` to assess the effect of learning and auxiliary supervision on the budget head. The latter is not a simple change in budget value or auxiliary gradient. The budget is fixed, with identical inference graph and parameters; only the budget head is frozen during training. All groups retain low-frequency and identity constraints, and cannot rely on noise injection or corruption of the lossless input to achieve low scores.

Two seeds: `20261012` and `20261013`, each with **512 steps**, Adam optimizer at `1e-4`, no weight decay, gradient norm capped at 5. An 80-frame no-gradient warmup is used, and supervision begins at frame 89. This is an early fitting diagnosis and does not replace full training or justify claims of capacity, phase modeling, or architectural limits. After completion, all eight endpoints are frozen and scored uniformly. No intermediate point is selected, and no group's budget is temporarily reduced.

---

## Diagnostic Materials and Evidence Boundaries

Fixed inputs are drawn from the first seed schedule at draw indices 0, 682, 1364, and 2047—exactly four batches, each containing 4 music clips and 1 speech clip, plus corresponding clean identity references. The training cycles through these four batches, with only the initialization changed between seeds. No alteration to the diagnostic materials occurs. The entire input window is isolated from existing development/shifted source time.

Materials consist of **16 music recordings and 4 speech recordings**. After removing historical warmup and deduplication by source interval, actual music supervision totals approximately **7.32 seconds**, and speech supervision totals **1.85 seconds**. Coverage includes eight codec recipes and mono/L/R/M/S views. This is not a complete music library, nor a full stereo quality alignment test. Its purpose is to determine whether the model can fit known materials—no independent test or "no overfitting" conclusion is implied.

Unified records of per-band and per-component spectral errors, old deployment metrics, per-batch gradient norm and direction, and differences between model and identity DSP outputs are tracked. Both paired and identity comparisons are evaluated. Final results are verified against the original objective, and no cross-group ranking is performed. The original amplitude spectrum and true Rust output HF SC, energy, and transients are preserved.

Additional fixed-ratio interpolation from input spectrum to lossless complex spectrum is performed at `0/.01/.1/.5/1` for directional validation only. This uses reference information and is not deployable. Pre-checks show that all four batches' direct objectives decrease monotonically with interpolation, reaching zero at full reference. The DSP-post recovery objective also decreases, but does not reach zero, indicating residual recovery space and inherent DSP error. This does **not** constitute evidence that the model has learned the improvements.

---

## Execution and Validation

Entry point: `training/scripts/run_direct_spectrum_study.py`, supporting `--prepare-only` and independent `--preflight` (4 steps per group, single seed). Pre-flight checks are not used for candidate selection or audio quality reporting. All input and source code hashes are logged, and a `source/training` snapshot is saved before training begins. The run starts from a frozen source code. Breakpoints save model, optimizer, objective signature, and training history. Any signature drift or non-finite value must trigger an error.

The historical replay directory name `development` is used only for the training clips in this round. The registered `review_scope` explicitly marks the training fitting materials. After completion, eight exports are produced, each validated for 320-hop state/reset/silence consistency, followed by 320 full Rust frontend → ORT → DSP role replays. This round does not repeat the long-term performance experiments from P28, nor does it provide any real-time or audio quality certification for new candidates.

Structural, state, and fixed buffer configurations remain unchanged. If a recovered signal is valid, further full training, cross-recording recheck, real-time latency, continuity, and perceptual validation are still required.

The official comparison was launched on **2026-10-19 14:38 +08** from commit
`796d838560648c2ffef585aa7b96a866e68c936e`, frozen source code, directory
`data/runs/direct-spectrum-p29-20261019/`, with SHA-256 hash:
`68d98a0bc92326b02de5032d6c8ced613905ee9aaf8d7e4f254406ea1c3b5a76`.
Process supervisor PID 84735, training PID 84736. Current status can be viewed in `supervisor.json` and `study/progress.json`. The first launch failed due to incorrect virtual environment interpreter parsing; it exited before training began, with the attempt logged in `supervisor-attempt-1.json`. A correct interpreter was then used to restart.

**586 tests passed locally**, Ruff lint/format passed. New direct target reference zero error, silence/weak-signal gradient, finite difference, reference transient, phase error, DSP mask independence, and exact continuity across all four groups were verified. The first version of pre-flight revealed an old tool's excessive dependency on P10 cache directory; after correction to prioritize the current explicit cache, the independent `direct-spectrum-p29-preflight-20261019-v2` passed fully. All four exports, 160 full native role syntheses, passed identical observation checks with maximum absolute error of **2.38e-7**; maximum ORT complex spectrum error of **5.34e-5**, independently reported. All pre-flight weights were not used for quality selection.

The official pipeline completed normally at **2026-10-19 14:52 +08** (exit 0): all eight trajectories completed 512 steps, final endpoints were uniformly rechecked, and all eight exports and 320 full Rust role replays were completed. Original audio and experimental outputs remain in the ignored directory.

---

## Final Results: High-Frequency Recovery Achieved, But Recovery Precision Still Insufficient

All results apply only to the repeated training on the 16 music and 4 speech clips described above. The four groups have different optimization objectives; the following metrics are evaluated using common indicators, without comparing individual total loss. No pre-defined quality promotion or independent test was conducted, so no claims of generalization, convergence, or full training effectiveness can be made.

The table below shows music clip mean values, with the slash-separated values representing seed `20261012` and `20261013`, respectively. Lower error is better. The HF SC values for raw prediction and DSP output use the same replay report; PCM values use full Rust frontend → ORT → DSP.

| Configuration | Raw Prediction HF SC | Raw Prediction High-Frequency Energy Error (dB) | DSP Output HF SC | Full Rust PCM Error RMS |
|---|---:|---:|---:|---:|
| Initial Identity Control | 0.430315 | 8.1553 | 0.430129 | 0.009223736 |
| deployed | 0.380854 / 0.373307 | 3.9457 / 3.5397 | 0.386107 / 0.374710 | 0.009223956 / 0.009223969 |
| raw | 0.376954 / 0.362712 | 1.7485 / 1.5170 | 0.391291 / 0.381931 | 0.009236265 / 0.009246550 |
| joint | 0.365301 / 0.369889 | 3.1713 / 3.4722 | 0.373114 / 0.378548 | 0.009224300 / 0.009224012 |
| raw_unit_budget | 0.414607 / 0.357858 | 1.6043 / 1.3862 | 0.417105 / 0.377242 | 0.009239957 / 0.009242278 |

The unprocessed music PCM error is **0.009219998**; the identity DSP value is slightly higher at **0.009223736**. The raw group shows a **~0.14% / 0.25% increase** in waveform error compared to identity DSP, despite improvements in magnitude and energy. These percentages do not represent audible degradation. The RMS difference between raw and identity music output is **0.0003685 / 0.0005607**, approximately **14 / 19 times higher** than the deployed group—no longer resembling the near-floating-point-noise output of P28.

The paired direct target scores (fixed 4:1 music-to-speech mix) dropped from initial **0.249942** to **0.229088 / 0.225849** for raw, a reduction of **~8.3% / 9.6%**. However, the per-component complex error within the same direct target increased from **0.224356 to 0.234412 / 0.236973**, indicating that the overall score reduction does not equate to joint magnitude and phase fidelity. Complex error alone cannot attribute all degradation to phase. The HF flux error in music DSP output increased from identity's **0.602888** to **0.616682 / 0.606590** in raw, and to **0.593535 / 0.593778** in deployed. This supports continued focus on phase and transient reconstruction, not just higher HF energy.

**Speech-only report**: Unprocessed PCM error is **0.00247517461**, identity DSP is **0.00247517467**, and raw full Rust output is **0.00247898025 / 0.00247987867**, showing slight degradation. Original prediction HF SC drops from **0.218055 to 0.203486 / 0.195704**, and DSP output from **0.208642 to 0.203823**. Improved amplitude spectrum does not equate to waveform recovery.

---

### Judgment on the DSP Dependency Hypothesis

- The direct target indeed prompts the small model to learn more prominent high-frequency supplementation and reduces energy error—previously lacking direct supervision was a valid issue. Adding the full old deployment objective causes output changes to shrink significantly.
- The old deployed objective improves HF SC on fixed training samples, indicating it does not fully block gradient flow. The random cropping used in P28 and the small-sample fitting in this round cannot be used for cross-budget or generalization ranking.
- The `raw` group shows fidelity issues both before and after DSP—this does not imply correct prediction; the signal is corrupted by the DSP. A fixed budget does not stabilize music spectrum or transient errors, and differences remain between the two seeds.
- The `joint` group performs well on some deployment metrics but does not outperform the control on either seed. No seed is selected for promotion. The direct and old objectives alter error metrics and scales, so they cannot be used to prove that the DSP Jacobian is the sole cause.

**Next Steps**: Continue using these training materials for non-deployment amplitude/phase swap diagnosis: predict magnitude with reference phase, reference magnitude with predicted phase, and compare against original predictions along the same path to isolate contributions from phase, magnitude, OLA, and fusion. Reference information is strictly for diagnosis and must not enter inference. No automatic training extension, new experiment launch, or default weight override was initiated.

---

### Engineering Recheck

All 320 full native role syntheses passed with maximum absolute error of **2.38e-7**; maximum ORT complex spectrum error of **5.34e-5**, independently reported. No synthetic tolerance was relaxed to pass cases. Final checks confirmed frozen source code, 34 completed receipts, eight endpoints, and native audit logs/reports—all **462 hash values** matched. No rehashing of original audio occurred. Descriptive summary is stored locally in `completion-audit-20261019.json`, with full per-component and counterexample reports preserved. CI [build](https://github.com/backrunner/soundex/actions/runs/39211136683) passed.
No additional long-duration latency, real device continuity, or blind listening validation was performed. No new real-time or audio quality certification is provided.

The exact seed pair is `20261012 / 20261013`. The separately reported same-input ORT complex-spectrum maximum error is 6.10e-5; it must not be confused with a PCM synthesis-parity tolerance.
