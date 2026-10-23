# P30: High-Frequency Magnitude/Phase Swapping and Synthesis Stage Diagnostics

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

2026-10-19, following [P29 Direct Spectrum Supervision Comparison](direct-spectrum-study.md).
P29 introduced direct high-frequency output targets, but improvements in complex spectrum, transient, and final waveform errors were not synchronized.
This round focuses on error localization—no new model training, no loss modification, no DSP changes, and no default weight adjustments.

---

## Fixed Protocol

- Uses all eight final endpoints from P29's frozen state: four groups, two seeds, without selecting the best-performing seeds.
- Uses the same full set of 20 pre-fitted training crops (16 music, 4 speech), each augmented with clean identity.
  - No new independent test set; material, recording, encoding, and channel coverage boundaries remain consistent with P29.
- The model observes only the original input and existing causal history, predicts once, then performs output replacement. Reference signals are not fed into the model state.
- Interventions apply only to **above 4 kHz, starting at frame 80**; all variants retain identical low-frequency predictions and 80-frame pre-warmup.
  The 4 kHz hard boundary is for measurement only and is not a new frequency segmentation strategy.

| Variant | High-Frequency Magnitude | High-Frequency Phase | Purpose |
|---|---|---|---|
| candidate | Model prediction | Model prediction | Original candidate |
| reference_phase | Model prediction | Original lossless reference | Validate phase accuracy |
| reference_amplitude | Original lossless reference | Model prediction | Validate magnitude limits and energy impact under erroneous phase |
| reference_high | Original lossless reference | Original lossless reference | High-frequency reference upper bound control; low frequencies remain as candidate |
| identity | Original input | Original input | Common identity DSP control; replayed once only |

Swapping reference phase does not prove the model can predict that phase. Upper-bound control is not a deployable quality result.
Below 4 kHz, the candidate and DSP remain active, so `reference_high` does not equal full-band lossless reconstruction.

---

## Measurement Path

All variants use the same original input detector, mask, gate, and smooth-knee limiter. Spectra are replayed via **real Rust DSP** offline:
8 models × 4 variants × 20 crops × 2 roles + 40 role identities = **1,320 native roles**.
This is not a new end-to-end frontend → ORT pipeline or real-time computation test; P29's actual frontend → ORT → DSP evidence remains independent.
Swapped reference spectra cannot be used to train audio recovery models.

Absolute RMS/dBFS errors against clean PCM are reported per stage:

1. Raw prediction OLA (diagnostic proxy, no masking or clipping).
2. After masking and magnitude-phase fusion OLA.
3. After gate, before clipping.
4. Final Rust output.

All raw and final magnitude spectra, HF SC, energy, flux, and low-frequency errors are preserved, along with all counterexamples, separated by music, speech, paired, and identity.
Clip mean and recording mean PCM errors are also preserved.
Differences between Rust and Torch frontend outputs, and between observed synthesis and actual synthesis, are separately reported—no synthesis tolerance is relaxed.

Complex spectrum squared error is further decomposed as:

`|P − T|² = (|P| − |T|)² + |P||T| × |exp(iΔφ) − 1|²`

The second term represents magnitude-weighted circular phase error, avoiding the misinterpretation of arbitrary phase angles in silence as major issues.
This decomposition is in unnormalized FFT coefficient power—**not PCM dBFS**.
Absolute audio power is reported through a separate Parseval/window-energy-calibrated analyzer.
No conclusions about perceptibility are drawn from weak-band relative scores.

---

## Implementation and Reproducibility

Entry point: `training/scripts/audit_phase_swap.py`; implementation of swapping and error decomposition resides in
`training/evaluation/phase_swap.py`. No reference information is introduced into production inference or training objectives.

```bash
rtk proxy env PYTHONPATH=training training/.venv/bin/python \
  training/scripts/audit_phase_swap.py \
  --study data/runs/direct-spectrum-p29-20261019/study \
  --output data/runs/phase-swap-new/study --prepare-only
```

Input, endpoints, and source code hashes are first registered, then a copy of `source/training` is duplicated and executed from a frozen snapshot.
The snapshot must include Python, JSON, YAML, and TOML runtime dependencies; copying only `.py` files misses shared parity policies.
The first snapshot failed during import due to missing JSON; after correction, it ran successfully in an independent directory:
`data/runs/phase-swap-p30-20261019-v2/`, where original registration was preserved.

**593 tests passed locally**; 7 new checks cover frequency band swapping, component-level behavior, 44.1/48 kHz geometry, warmup and low-frequency invariance, reference input preservation, complex error decomposition, phase wrapping, and silence numerical behavior.
Ruff lint/format passed. Frozen source code corresponds to commit
`2bccd4e5378e072d1b20fa40e7fad921072f731e`; registered SHA-256 is
`3c6616f2c97cd38effb9f8904444374d9ef2a1e0dde67f5d98b3da6069221c9b`.

---

## Final Results

**All 1,320 native roles completed. Phase is a recoverable constraint, but magnitude remains inaccurate; the current fusion strategy still limits the recovery amplitude achievable from high-frequency reference inputs.**
The following are descriptive diagnostics for the pre-fitted training crops—**not new model gains or generalization results**.

### Magnitude and Phase Must Be Recovered Together

Final PCM error for raw music crops (lower is better):

| High-Frequency Output | Seed 20261012 | Seed 20261013 | vs. Original Candidate |
|---|---:|---:|---|
| Original Candidate     | 0.009236254 | 0.009246582 | — |
| Predicted Magnitude + Reference Phase | 0.009004067 | 0.008913488 | ↓ 2.51% / ↓ 3.60% |
| Reference Magnitude + Predicted Phase | 0.009793689 | 0.009778959 | ↑ 6.04% / ↑ 5.76% |
| Reference Magnitude + Reference Phase | 0.007693806 | 0.007693808 | ↓ ~16.7% / ↓ 16.8% |

All eight endpoints show the same directional trend: reference phase reduces average waveform error; swapping reference magnitude instead increases average error.
While the magnitude spectrum HF SC improves significantly in the latter case (e.g., raw seed 12 drops from 0.391291 to 0.248378), waveform quality degrades.
Thus, evaluating only magnitude or energy leads to incomplete conclusions.
This is a group-mean trend—not guaranteed for every individual crop or frequency bin.

Speech results are equally distinct: raw seed errors were 0.002478979 / 0.002479873; after reference phase swap, they dropped to 0.002456333 / 0.002454350 (↓ ~0.91% / ↓1.03%); swapping only reference magnitude increased to 0.002494670 / 0.002493247.

Complex error decomposition confirms phase cannot be blamed alone: for raw seed 12 music, the unnormalized coefficient error power is 0.052683, with magnitude term 0.037437 and phase term 0.015246.
After setting phase to reference, magnitude error persists.
Note: this decomposition reports error power—not PCM dBFS.
The percentage improvement in reference swap is not equated to new model performance or perceptual gain.

---

### Fusion Strategy Imposes Practical Limitations

Below is the absolute PCM RMS error for raw seed 12 music, high-frequency band:
Each crop first aggregates the four HF bands' calibrated error powers, takes the square root, then computes the clip mean.
This is not a direct sum of per-band RMS.

| High-Frequency Output | Raw OLA | Masked/Amplitude-Phase OLA | After Gate (pre-clipping) | Actual Rust Output |
|---|---:|---:|---:|---:|
| Original Candidate     | 0.0061401 | 0.0061336 | 0.0061336 | 0.0061346 |
| Predicted Magnitude + Reference Phase | 0.0048476 | 0.0058436 | 0.0058436 | 0.0058468 |
| Reference Magnitude + Reference Phase | 0.0002499 | 0.0042226 | 0.0042226 | 0.0042304 |

Another seed shows identical trends. The primary divergence occurs during the masking/phase-fusion stage; gate application within the supervised region shows no mean change, and limiter addition is minor.
Current processing retains many input frequency components below cutoff—preserving tonal fidelity but also preserving compression artifacts.
This table does not disentangle contributions from masking geometry versus magnitude-phase interpolation—no specific implementation detail can be blamed, nor can it be attributed to a limiter failure.

For the current candidate, fusion slightly reduces HF error; its original HF synthesis error is still higher than the identity input by ~0.0061169.
Thus, it cannot be claimed that the model predicted correctly—only DSP error remains.
The reference HF OLA still exhibits ~0.00025 error, attributable to fixed boundaries, windowing, and low-frequency retention—not a zero-error upper bound for full-band reference reconstruction.

---

### Engineering Consistency

The RMS of original PCM for all eight candidates matches P29's corresponding replay **exactly** (maximum deviation: 0).
All roles undergo identical observation-based synthesis validation using actual Rust-observed spectra, with maximum error **2.98e-7**, zero failures.
A single case exceeds the prior 2e-5 tolerance:
`joint-512-20261013 / reference_amplitude / music paired`, row `20ccaa00c2dd3cf5497dcc87`, start at 34254, peak error **2.265e-5**.
After preserving original prediction and replacing with actual Rust-spectrum, the synthesis difference drops to **2.38e-7**.
Original frontend differences and failure flags are fully preserved—no threshold relaxation or frontend divergence suppression.

174 frozen source code/configurations, 8 endpoints, and 9 reports were verified; total 191 hashes.
Summary stored locally in `completion-audit-20261019.json`; each counterexample preserved in `study/reports/`.
No new training, model promotion, listening validation, or real-time certification occurred.

---

## Next-Step Structural Comparison

P29's model did not include phase input: the encoder primarily reads magnitude and its history, with complex residual rotated relative to input phase at output.
Missing-band input phase may be unreliable; even if phase input is added, existing carrier features degrade as amplitude at that frequency diminishes.
This is a visible representation limitation in the source code.
However, this round's oracle cannot prove that any alternative architecture is superior.

The next round should fix the direct objective, scale, and small model size, and compare along two dimensions:

1. Whether the encoder uses observed complex/phase information.
2. Whether residuals use input-phase-relative coordinates, or direct global real/imaginary coordinates.

Using matched initialization, identical two seeds, and pre-registered training budget, first validate learnability on training materials, then proceed to full cross-recording training.
Prior P28 phase input intervention operated under an older deployment objective and cannot substitute this comparison.
Adding phase branches introduces minimal parameters—must be evaluated against capacity changes, and cannot be used to rank large vs. small models.
The original DSP remains for shared deployment control; separate audit of repair pathways inside and outside detection bands is retained.
Post-verification of prediction fidelity, we may assess whether to retain frequency band repair or generate missing bands—**no wider masking should be adopted as default for the base model**.

The next [P31 Phase/Coordinate Comparison](phase-coordinates-study.md) has completed four-group, two-seed, fixed 512-step learnability diagnostics and unified recheck:
Phase input yields no waveform benefit; global coordinate sets remain near-identity.
This does not alter the current oracle conclusion, nor does it prove the model has learned reference phase or that this architecture has sufficient capacity.

All phase swaps were scored from frame 89 after the common warmup. Substituting reference phase reduced the reported magnitude-only error from 0.002478979 / 0.002479873 to 0.002456333 / 0.002454350, approximately -0.91% / -1.03%.
