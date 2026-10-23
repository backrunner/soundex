# P7 Loss and Audit of Weak High-Frequency Energy Compensation

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

**Conclusion:** Current evidence points to **spectral energy calibration failure and a compromise in average target allowance across samples**.
No evidence of reversed loss gradient direction; the significant degradation in high-frequency SC cannot be directly attributed to failure of the spectral shape head.
No change to default weights, no exclusion of failing recordings, and no reclassification of reused development set as independent generalization validation.

---

## Scope and Reproduction

Audited the four frozen best-weighted models from [P7](energy-coordinate-study-20261012.md), verifying checkpoint integrity and evaluation hash consistency.
Re-ran CPU-based predictions on 33 music development segments across 26 recordings, with corresponding 33 identity inputs and 32 training probe recordings;
Total of 392 prediction records (2 structures × 2 seeds).
Recomputed results match original evaluation within rtol/atol = 1e-4.
Per-component validation of loss value and gradient was performed, including 44.1/48 kHz, silence, and low-energy cases.
All training parameters, loss function, data splits, and candidate selection remain unchanged.

```sh
PYTHONPATH=training training/.venv/bin/python training/scripts/audit_energy_objective.py \
  --study data/runs/energy-p7-20261012/study
```

Full local results: `data/runs/energy-p7-20261012/study/objective-audit.json`.
Report retains checkpoint summaries, script overviews, all recordings, per-band decomposition, gradient checks, and reference interventions—raw data is not submitted.

---

## 1. High Mean Driven by Tail Amplification, But the Problem Is Not Limited to the Tail

After aggregating across recordings, both seeds show degradation in Spectral Consistency (SC) for **11/26** development music recordings.
Three specific recordings account for **97.0% / 94.5%** of the total positive SC degradation.
Excluding these three for sensitivity analysis, the remaining 23 recordings still exhibit an average SC difference of **+0.0740 / +0.1787**.
This is not a formal score update, nor a license exception, nor limited to three independent fragments—these three cover five development segments.

The weak reference amplifies relative spectral error, but actual predicted energy also increases. Using the full-spectrum DFT power of each reference fragment as baseline, for six high-frequency clip-bands below -80 dB, the candidate average energy deviation is **+37.35 dB / +33.77 dB**, compared to **+8.06 dB / +6.97 dB** for the reference.
Candidate predicted power averages at **-57.96 dB / -61.54 dB** relative to full-spectrum. These are **spectral-domain relative powers**, not dBFS, and cannot be used to infer perceptual quality or device output noise.

The -80/-60/-40 dB thresholds are used only for descriptive reference strength and do not enter inference, training, data filtering, or release thresholds.

---

## 2. Raw Spectral Error is Primarily Energy-Related, Not Shape-Related

After freezing predictions, only offline reference-based per-frame or per-band RMS or normalized spectral shape replacement was applied.
Retained 0–4 kHz band; used identical 80 hop exclusion zones. Below is the average SC across 33 music development segments, equally weighted:

| Candidate Path | Seed 12 | Seed 13 |
| --- | ---: | ---: |
| Raw Prediction | 4.4792 | 3.5576 |
| Correct band energy only, preserve shape | 0.4497 | 0.4510 |
| Correct band shape only, preserve energy | 4.3054 | 3.3888 |

When provided with correct energy, SC drops to **0.4294 / 0.4335**, indicating that the candidate shape is not inherently superior to the reference.
However, shape differences cannot explain the majority of the SC increase—from ~0.7 to 3–4.
SC is a magnitude spectrum reconstruction error, encompassing both energy and shape; prior terminology "spectral shape error" was overly broad.

These reference interventions are not deployable and do not represent new model audio quality. Their purpose is diagnostic—identifying the source of error.

---

## 3. No Symbol Errors Detected; Overall Decline Arises from Compensation Across Samples

For all high-frequency frames in development music, we screened frames where predicted power exceeded reference by more than 1%, and examined `sum(dL/dp * p)`—the derivative along the logarithmic magnitude gain direction:

- For both seeds, **10,449 / 12,486** over-predicted frame-band pairs exhibit **energy-decreasing gradient directions**.
- No instances of reverse reward increasing energy were observed.
This check was performed on frozen predictions and does not prove that the backbone can simultaneously satisfy all gradients.

Per-component summation aligns with the original function. When reference energy is correct, loss is zero. Therefore, this failure cannot be attributed to a symbolic error in the loss function.
Loss increases on failing recordings, but due to improvements in other recordings, the overall mean still declines:

| Music Segment Group / Structure | Seed 12 Total Loss | Seed 13 Total Loss |
| --- | ---: | ---: |
| Three tail recordings (5 segments / reference) | 0.3010 | 0.2859 |
| Same segments / candidate       | 1.0024 | 0.9727 |
| Remaining 28 segments / reference | 1.1688 | 1.1888 |
| Same segments / candidate       | 0.9061 | 0.9525 |

---

## 4. Target Ordering Indeed Conflicts with Recovery Metrics

Two fixed synthetic counterexamples validate design trade-offs, not per-track tuning:

- **Single sparse high-frequency band**, correct shape but magnitude at only 1% of reference: total loss = **1.0881**, SC = **0.99**.
  Replaced with flat spectrum having correct RMS: loss drops to **0.2809**, but SC increases to **1.3019**.
- Correct shape, magnitude at 0.1× or 10× reference: total loss ≈ **0.5448 / 0.5449**,
  but SC is **0.9 / 9.0**. Log-energy error is approximately symmetric, while linear reconstruction error is asymmetric.

Mathematically, this is not an error—it does not imply we should abandon energy compensation.
It demonstrates that the current objective function cannot independently guarantee near-lossless recovery.
The normalized shape term largely removes amplitude scale, and cannot compensate for this ordering issue.
The retained log1p magnitude term has low weight.

Additionally, the reference RMS floor is set at 1e-4 (absolute floor at 1e-7 DFT magnitude).
Of the six weakest clip-bands, **99.3%** of frames fall below this floor; average shape confidence is only **0.067**.
This weighting provides numerical stability and prevents overfitting to extremely weak noise.
Candidates are still penalized after exceeding the floor, so these bands are not "fully unmonitored."
Directly removing the floor would allow tiny references to dominate training—this is not a viable fix.

---

## 5. Issues Also Appear in Training Distribution and Identity Inputs

For 32 training probe recordings, candidate SC increases from reference **0.6131 / 0.5614** to **1.9307 / 1.8832**.
In seven high-frequency clip-bands below -80 dB, average energy deviation reaches **+24.00 dB / +23.24 dB**.
Thus, this is not merely an isolated failure in the development set or a simple training-to-development generalization gap.
Probes are fixed crops from training recordings and cannot be generalized to the full training set average.
They also do not justify claims of overfitting.

Native identity inputs show degraded SC on **19/26 and 20/26** recordings, with loss increasing on **23/26 and 26/26**.
Introducing correct energy correction reduces candidate identity SC from **4.2159 / 3.2823** to **0.1363 / 0.1340**.
This supports the hypothesis of erroneous high-frequency energy compensation in natural, weakly excited regions.
It has not yet been proven that the model fails to distinguish between natural absence and compression cropping.

This analysis only evaluates magnitude prediction; whether identity inputs, after actual DSP, produce audible differences requires separate playback and listening validation.
Therefore, this result cannot be directly interpreted as the final audio product being equally degraded in amplitude.

---

## Next Steps for Handling

1. **Retain energy, spectral shape, and reference transient supervision**. Test the addition of a normalized reconstruction term with numerical floor and actual amplitude error alignment.
   Prioritize synthetic counterexamples with reasonable ordering, reference-optimized, and bounded weak-energy gradients.
   Do not simply increase the weight of existing terms.

2. **Treat identity and naturally weak high-frequency content as distinct, grouped constraints**, to prevent cancellation by overall mean.
   Define grouping rules using training references and fix them prior to new experiments.
   Do not treat the three development recordings as special cases, nor provide reference labels to inference.

3. **Conduct a new round of identical architecture/budget objective comparison**, evaluating both energy recovery and overcompensation, and comparing raw predictions against true DSP.
   If calibration continues to fail, conduct a separate analysis of input energy representation or recovery confidence branches—avoid modifying loss and structure simultaneously.

**Current evidence is insufficient to attribute the primary cause to model capacity limits, baseline overfitting, or phase/DSP issues.**
This investigation serves as diagnostic and regression verification only.
No new training cycle has been initiated, and no modifications to the objective or promotion criteria have been made.
