# P14: Spectral Loss, Gradients, and Synthesis Chain Audit

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

## Conclusion

This round did not modify the training objective, model architecture, default weights, or runtime behavior. The audit reveals:

- Spectral loss decomposition matches the original implementation in both numerical values and gradients; no arithmetic errors were detected within the scope of this round.
- Development music exhibits actual spectral misfilling. Reducing high-frequency weights or increasing the lower bound of normalization does not constitute a direct fix.
- In severe counterexamples, the per-frame spectral content after inter-band mixing is analyzed again via OLA (Overlap-Add) and shows significantly amplified errors.
  - Neither of the two counterexamples entered the clipping region. Next, we should prioritize identifying inconsistencies across frames and inter-band crosstalk.
- This does not indicate a bug in the OLA algorithm itself, nor does it prove that this mechanism explains all generalization failures.
  - The current model already computes loss through DSP output; what is revealed here is its failure mode on unseen recordings.
  - It is no longer valid to treat "adding a DSP loss term" as an untried solution.

## Audit Scope and Evidence

Using the P13 multi-location set as input, with fixed initial model and two seeds at step 2048, no new candidates were selected.
All 320 training positions (256 music, 64 speech) and 89 reused development snippets (33 music, 56 speech) are covered; a total of 1,227 prediction pairs and 2,454 paired/identity role cases.
Training positions still originate from 40 audio recordings—thus, the 320 positions do not represent 320 independent recordings.

For each case, we recorded:
- RMS values of reference and output per band
- Absolute error
- Relative full-spectrum error
- Normalization lower bound
- Amplitude and time-difference loss
- Rendered amplitude coordinate gradients
- Gradients returned from DSP after receiving raw predictions
- Model parameter gradients
- Cosine angle between paired recovery and identity parameter gradients

All gradients were computed from a frozen model on individual pairs—**not** from historical gradients in the original training minibatch or Adam updates.
No claims are made that the original training experienced gradient explosion or frequent clipping based on these gradients.

Additionally, for two seeds that ranked in the top three worst development music snippets overall, we replayed the initial and final models:
- 3 snippets × 2 roles × 3 models = 18 real-time Rust DSP diagnostics.
- This constitutes a post-hoc counterexample diagnosis and cannot be used as a new quality validation set or for generalization across music.
- Maximum proxy/original waveform absolute error: `2.980232238769531e-7`, which is less than the original `2e-5` tolerance.

The primary audit code is frozen at commit `81257f95415adff11e5d4d1e4c46397435311170`.
P13 checkpoint, completion receipts, sample lists, scoring files, manifest, and native contract hashes have all been verified.
Post-processing and stage-by-stage scripts independently record source code hashes.
Stage comparisons use the same frame range from 89 to the end of valid frames.
For identity, each stage uses the output from that stage’s initial pass as reference, avoiding contamination from prior limiter distortion.

## Results

### Training vs. Development Gap

All values below are per-case averages within music. Identity loss measures additional degradation relative to the initial DSP.

| Metric | Seed 20261012 | Seed 20261013 |
| --- | ---: | ---: |
| Paired recovery loss (training) | 0.572272 | 0.572836 |
| Paired recovery loss (development) | 0.734712 | 0.719533 |
| Identity loss (training) | 0.009352 | 0.012118 |
| Identity loss (development) | 0.198158 | 0.179015 |
| High-frequency (16 kHz+) in development identity | 0.146681 | 0.135564 |

Initial paired recovery loss: training = 0.598178, development = 0.588365; initial identity delta is zero.
Development identity degradation accounts for approximately 74% / 76% of the loss in the 16 kHz+ spectral components.
These are error ratios, not subjective perceptual quality loss ratios.

In training music, 38 out of 256 (14.8%) have reference energy below the normalization lower bound in the 16 kHz+ band.
In development, only 3 out of 33 (9.1%) fall below the threshold.
High-frequency content is not entirely absent from the training set; further investigation into local spectral edges, temporal dynamics, and model correction behavior is required.
Adding sampling weight for "weak high frequencies" alone is insufficient.

The median L2 norm of final per-pair risk parameter gradients in training is 0.399 / 0.378, with maximums of 2.697 / 3.938.
Corresponding development maxima are 77.528 / 59.830.
In severe counterexamples, the cosine between paired recovery and identity gradients ranges from 0.956 to 0.9997:
- On these cases, both losses are largely aligned, so the degradation cannot be simply attributed to conflicting loss objectives.
- This does not rule out gradient conflicts across different recordings, nor does it imply instability in the training history.

### Where Amplification Occurs

The following compares only spectral components (excluding waveform terms), including time-differenced reference bands. Lower values are better.

| Music Range / Seed | Post-mix per-frame spectrum | After OLA re-analysis | Final limited output |
| --- | ---: | ---: | ---: |
| Training / 20261012 | 0.009937 | 0.009505 | 0.009327 |
| Training / 20261013 | 0.012853 | 0.012231 | 0.012057 |
| Development / 20261012 | 0.117986 | 0.203507 | 0.197676 |
| Development / 20261013 | 0.107732 | 0.184107 | 0.178389 |

Severe blues counterexample `9d95e7f035f878fa1a6ab421`:

| Seed | Post-mix spectral | After OLA | Final spectral | Peak before limiter |
| --- | ---: | ---: | ---: | ---: |
| 20261012 | 0.092319 | 2.273647 | 2.273647 | 0.690754 |
| 20261013 | 0.061887 | 2.087301 | 2.087301 | 0.690739 |

The scoring region has gate=1. Both peaks remain below 0.855—the limiter threshold—so the amplification occurs during OLA synthesis, **not** due to limiter or gate switching.
The final 16 kHz+ energy relative to identity reference is elevated by approximately 25.10 dB / 24.45 dB.
Given the weak reference bandwidth, this relative energy change cannot be directly interpreted as a perceptible loudness change or verified audible noise.
Another unlabeled counterexample shows primary degradation in the 8–12 kHz band, also without triggering clipping.
No special rules should be applied only to the 16 kHz+ band.

## Next-Phase Direction

1. Perform per-band isolation correction of residual errors on frozen predictions, and measure error propagation after OLA synthesis per band.
   Confirm whether the degradation stems from adjacent-band crosstalk, inter-frame inconsistencies, or other mechanisms.
   Perform parallel diagnostics on the training side.
   Development counterexamples are for diagnostic use only and **must not be used in training**.

2. If confirmed as synthesis inconsistency, register a single-factor control experiment:
   "Original objective + added synthesis consistency auxiliary constraint."
   - Fix model, P13 multi-location sampling, budget, and seed.
   - Auxiliary term scale must be calibrated using training data only.
   - Do not require smoothness, nor penalize transient changes that exist in the reference.

3. Continue using original lossless references and fixed P13 metrics for acceptance.
   Report actual high-frequency recovery and identity degradation.
   Retreating to identity output is not a success.
   If no improvement across recordings, shift toward broader audio coverage or structural alignment.

No new weight training was initiated in this round.
We must complete the above mechanism isolation before introducing auxiliary terms, to avoid baseless loss modifications based on a single average.
Model parameter count, export budget, and real-time execution path remain unchanged.
No new blind listening or real-time pass judgment has been conducted.

## Reproducibility and Local Artifacts

Root directory: `data/runs/loss-audit-p14-20261013/` (all content ignored; no materials or checkpoints pushed)

- `audit/registration.json`, `audit/summary.json`: Input registration and output hash records.
- `audit/development-{initial,final-20261012,final-20261013}.json`, `audit/training-{initial,final-20261012,final-20261013}.json`: Full per-case data.
- `audit/analysis.json`: Per-linguistic and per-percentile summaries.
- `render-stages-common.json`: Stage-by-stage comparison across identical valid frame ranges; this version is used for official stage-level diagnosis.

Scripts: `training/scripts/audit_loss_scale.py`, `analyze_loss_scale.py`, `audit_render_stages.py`
Production loss and inference implementations remain unchanged.

Verification: 441 local Python tests passed. Ruff lint/format and diff checks passed.
Coverage includes: loss value and gradient consistency, silence and low-energy cases, finite difference, frozen weights, frame alignment across stages, and final output matching original render.
[Main audit frozen code CI](https://github.com/backrunner/soundex/actions/runs/38394842452) passed;
latest remote CI for post-processing scripts is valid as of the repository's current state.

Complete per-case evidence is retained in `audit/{training_bank,development}-{initial,final-20261012,final-20261013}.json`, under the local audit root. It includes the original counterexamples; no threshold is fitted to a single case.
