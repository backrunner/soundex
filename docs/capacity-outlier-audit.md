# P23 Weak High-Frequency Anomalies: Absolute Error and Stage-by-Stage Verification

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

**Conclusion**: Two phenomena coexist: the model genuinely generates extremely weak high-frequency components; and existing relative spectral metrics significantly amplify these small absolute amplitude errors. The previously observed large amplification factors in high-frequency spectral content cannot be interpreted as equivalent perceptible distortions or burst noises. Similarly, counterexamples cannot be dismissed or scores adjusted arbitrarily based solely on these metrics. At this time, no new weights have undergone blind listening or full-end-to-end equipment validation. No modifications were made to the training loss, model architecture, DSP pipeline, or acceptance thresholds in this round.

---

## Verification Scope and Methodology

Fixed: four P23 model variants, two seeds, and all final 16,192-step checkpoints. For each of the 89 development clips, paired and identity roles were re-evaluated, resulting in **1,424 records**. Reconstruction of the original loss and frozen report consistency was verified. Training set: 2,088 cropped segments; development set: 89 clips; shifted reference: 40 additional clips for cross-band spectral coverage analysis.

Four actual Rust reference cases were included: primary anomaly, secondary anomaly, and mid-range music/speech examples selected based on reference high-frequency content proportion. Eight models were replayed in both paired and identity modes, under identical identity DSP conditions, yielding **128 total role re-plays**.

Maximum waveform deviation from proxy and reference was measured at `2.98023e-7 < 2e-5`.

- Supervised region: retained original warmup and output alignment, length 21,248 samples (~0.443 seconds at 48 kHz). Stereo channels were not concatenated into a stereo stream.
- Absolute RMS was computed over the full supervised segment using PCM values.
- Band-level analysis used 4,096-point Hann windowed power spectra, with endpoint weighting and window energy normalization.
- Unit PCM amplitude corresponds to full-scale 1; unit peak sine RMS = -3.0103 dBFS.
- Unit tests cover 44.1/48 kHz sampling, band assignment, identity processing, silence, amplitude scaling, and invalid input handling.
- 4,096-point FFT used solely for offline audit and does not alter the model’s real-time FFT256/hop128 configuration or introduce latency.

Zero power is represented as `null` in JSON (in dB), not artificially injected with noise floor.
The unnormalized DFT power values reported in prior reports must not be interpreted as equivalent to the current PCM dBFS.

---

## Absolute Magnitude of the Primary Anomaly

`d2599868e6a472eff47c1674`, starting at sample 40244, extracted from a right-channel snippet of the original author’s WAV file.
Original audio RMS: **-42.43 dBFS**, peak amplitude ≈ 0.02166.
All values below are differences between final output and identity DSP under identical conditions — **not** the total compression error between MP3 input and lossless reference.

| Model | Seed | Extra Residual RMS | % of Original RMS | Residual Peak |
| --- | --- | ---: | ---: | ---: |
| anchored control | 20261012 | -93.04 dBFS | 0.295% | 0.000168 |
| anchored control | 20261013 | -83.12 dBFS | 0.923% | 0.000326 |
| 175k | 20261012 | -93.56 dBFS | 0.278% | 0.000124 |
| 175k | 20261013 | -105.87 dBFS | 0.067% | 0.000035 |
| 459k | 20261012 | -91.84 dBFS | 0.338% | 0.000191 |
| 459k | 20261013 | -92.79 dBFS | 0.303% | 0.000144 |
| 1,192k | 20261012 | -97.15 dBFS | 0.184% | 0.000095 |
| 1,192k | 20261013 | -92.10 dBFS | 0.329% | 0.000166 |

The largest model exhibits a smaller absolute perturbation than the anchored control for the same seed, yet performs significantly worse in the prior relative spectral metric. This directly demonstrates that the two metrics rank differently — it does not imply that the large model is universally inferior in all aspects.

High-frequency band residuals (identity, two seeds):

| Band | Seed 20261012 | Seed 20261013 |
| --- | ---: | ---: |
| 0–4 kHz | -181.31 dBFS | -177.73 dBFS |
| 4–8 kHz | -106.39 dBFS | -101.93 dBFS |
| 8–12 kHz | -98.78 dBFS | -93.64 dBFS |
| 12–16 kHz | -118.13 dBFS | -109.95 dBFS |
| 16 kHz–Nyquist | -103.83 dBFS | -98.89 dBFS |

The identity DSP reference shows only about -138.98 dBFS / -143.72 dBFS above 12–16 kHz and 16 kHz, respectively.
Introducing even a tiny absolute energy in these bands produces large relative deviations. These values reflect actual mis-generation, but do not imply perceptible degradation. Playback gain, device masking, and perceptual context were not evaluated in this measurement.

In paired input, the largest model introduces additional deviations of -93.75 dBFS / -88.61 dBFS relative to identity DSP. The final output's total error relative to the original lossless is approximately -58.78 dBFS — this is not equivalent to "new distortion introduced by the model."

---

## Where Does It Occur?

- Input detection average cutoff: **~6,389 Hz** for original audio; active proportion and supervision gate both at 1.0.
  For paired input, cutoff is approximately **5,918 Hz**. Even narrowband natural audio enters the enhancement path.
- At the same stage, the largest model exhibits spectral fidelity error (raw: 9.1147 / 18.3991, blended: 9.1145 / 18.3985, OLA after: 6.8449 / 12.9832).
  High-frequency mis-generation is already present in the output; blending preserves it, while OLA reduces relative error.
- Gated, limited, and OLA decomposition results are identical. The segment falls well below the clipping threshold and does not trigger any clipping.
  Difference between original and proxy/limited-waveform is on the order of ~1e-9 RMS.
- No non-finite values, clipping, or gate-switching events were observed.
  Evidence from short clips does not substitute for real-time continuous playback testing.
  Therefore, we cannot conclude that the entire real-time system is free of burst noise.

**Source-level mechanism**: Residual generation in the output header uses the formula
`scale * sqrt(local_normalized_power + 1)`, with a minimum threshold of the current frame’s full-spectrum RMS.
Thus, even if a high-frequency component is extremely weak, a non-zero prediction can generate energy from the full-spectrum scale — explaining why truly absent bands can be filled in.
This design behavior aligns with the observed failure, though structural ablation was not performed to confirm it as the sole cause.

---

## Loss and Material Coverage

The denominator of the relative loss is `reference_band_power + floor²`, where
`floor = max(global_spectral_rms * 1e-4, 1e-7)`.

This formulation avoids division-by-zero, but assigns equal weighting to each high-frequency band — even when the reference signal is extremely weak.
The high-frequency bands above 12–16 kHz in the primary anomaly are dominated by this floor term.

The largest model’s identity spectral component is approximately **99.98%** above 8 kHz, and the absolute residual in these bands falls within the small range shown in the table above.
Mathematical computation and formula remain consistent.
Treating this relative score as a proxy for perceptual severity or as a basis for cross-model ranking is insufficient.

In the training pool of 2,024 music clips, 12–16 kHz and above had relative floor values in **89 out of 156** clips; those with relative values below the primary anomaly reference or full-spectrum average were **37 out of 40**.
This shows exposure to weak high-frequency content — but exposure does not equate to mastery in suppressing mis-generation.

We cannot conclude that the original source material was damaged during encoding.
Historical master recordings are not proven to have undergone lossy conversion by this audit.

Across all 33 development music clips, the scaled model’s identity residual PCM RMS median is approximately **1.29e-6 to 1.76e-6**, with a maximum of ~2.56e-5.
The anchored model median is ~5.08e-6 to 8.00e-6, with a maximum of ~2.05e-4.
Under this absolute fidelity scale, the large models are not universally worse.

On the other hand, average waveform RMS error (relative to identity DSP):
- Identity DSP: 0.08307205
- Largest model (two seeds): 0.08307298 / 0.08307749

No overall waveform recovery benefit is demonstrated.
Therefore, we cannot declare new weights acceptable based solely on smaller absolute residuals.

---

## Next Steps and Deliverables

Future work must retain both absolute PCM and band-level error, full-spectrum relative error, spectral shape, and transient characteristics across all materials.
Avoiding conclusions based solely on a relative metric near zero is essential.
Any tolerance thresholds must be pre-registered under a unified rule and validated with new recordings and subjective listening calibration.
Do not raise thresholds for this case as a basis for acceptance.

Model improvements should prioritize input-driven band energy estimation and confidence in generation — preserving the ability to reconstruct genuinely missing high frequencies.
Do not achieve low-damage scores through blanket suppression of high-frequency output.

**Implementation**:
- `training/evaluation/absolute_audio.py`
- `training/scripts/audit_capacity_outliers.py`

No modifications were made to the frozen P23 source code or results.
Final local report: `data/reports/capacity-p23-absolute-audit-20261016-v2/`

- `measurements.json` contains full per-band and per-segment breakdowns
- `reference-coverage.json` includes coverage scan results
- `registration.json` and `summary.json` link source code, model, and result hashes

Audio files:
- `audio/case-071-identity/` and `audio/case-071-paired/`
  Provide original lossless, input, identity DSP, candidate outputs, and differential FLOAT WAVs (unamplified differences).
  All files are gain-aligned and sample-aligned.
  These are ~0.44-second localized segments — **no subjective blind listening or ABX testing was conducted**.
  These results do not constitute proof of perceptual quality.

**Future work**: Independent long-form listening tests and new recording validation are still required.
All original audio and derived files are preserved locally.

The stage audit replays all 8 models. PCM full scale is 1. The weak reference bands discussed above are 12–16 kHz and above 16 kHz; their small absolute energy must be reported alongside relative errors.
