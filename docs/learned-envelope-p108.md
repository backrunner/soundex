# P108 learned causal envelope versus a matched per-bin control

P108 follows [P107](missing-band-baseline-p107.md). P107 measured a 9.3 dB envelope
gap between the best frozen model and a reference envelope applied to a copy-up
patch. P108 trains a small causal GRU to predict that envelope from compressed
input alone, alongside a capacity-matched per-bin output control. Evaluation uses
an offline Python proxy of a deployable fill. No Rust implementation, default
weight or release decision follows from this report.

## Changed hypothesis

P5-P7 factorized envelope, shape and energy *inside* the per-bin generator, kept
the existing magnitude/phase fusion, and found tradeoffs. P108 changes both the
output and the synthesis:
- **Output:** 33 half-ERB band energies above 4 kHz (`band`), or 107 per-bin
  energies above 4 kHz (`bin`, the matched control).
- **No phase:** the model predicts no phase.
- **Synthesis:** a hop-coherent copy-up of the 4 kHz-to-cutoff band, added with
  energy max(predicted - present, 0), so input is never attenuated.

Both arms use the same features, data, loss (L1 in dB on reference energies, plus
0.5 x L1 on adjacent-frame changes, 32 warmup frames excluded), budget and seeds.

## Design

- **Model.** Linear(131->128) + 2-layer GRU(128) + linear head. The band arm has
  219,297 parameters and the bin arm 228,843; the P74 parent has 216,802. Each
  frame takes only the current frame's per-bin log power, relative to its
  sub-4 kHz loudness, plus that loudness and a sample-rate flag. History lives
  only in the GRU state.
- **Data.** From the TRAIN split of the 300-work library: 16,000 rows (235 works,
  9 codecs, 44.1/48 kHz, mono/left/right/mid/side views) with three 256-frame
  segments each (48,000 segments; about 8.7 h of supervised frames, counting
  channel views of a crop separately). 18 works matching any reserved evaluation
  catalog (P55/P85/P87/P88/P89) by file hash, decoded-PCM hash, `split_group` or
  `work_id` are excluded. P88 works are absent from the library. Validation uses
  1,000 rows from 26 validation works.
- **Budget.** 16,384 AdamW updates, batch 64, cosine 1e-3 to 1e-5, ±10 dB
  gain augmentation. Seeds 1 and 2. Final endpoints only.
- **Fill proxy** (`training/evaluation/envelope_fill.py`). An exact port of the
  Rust bandwidth detector (peak -60 dB top-down scan, alpha 0.1, enhancement
  below 0.85 Nyquist, 10-frame hangover, silence reset) sets the cutoff per
  frame. Each physical channel keeps independent state. Synthesis is
  Rust-matching Hann WOLA. No crossover, limiter or Rust stream is involved.
- **Evaluation.** P107 v3 rulers and case slicing on the exposed P88 panel. P107
  comparators are reused from its records. A new `oracle_detector_band` path
  applies the *reference* band energies through the same detector region and
  fill, to separate region error from envelope error.

The registration (`data/runs/envelope-predictor-p108-20261122/registration.json`)
froze pool, exclusions, budget, seeds, code hashes and these rules before
extraction:
1. **Learning:** band-arm validation error falls below step 0.
2. **Transfer:** band-arm missing-region half-ERB RMS error is below the native
   P74 parent in both seeds, with intervals below zero.
3. **Parameterization:** the band arm is below the bin arm in both seeds.

## Results

**Learning.** Validation level error fell from 34.5/34.8 dB at step 0 to
3.25/3.38 dB (band) and from 40.6/40.9 to 4.70/4.76 dB (bin). It was still
decreasing slowly at 16,384 updates, so the endpoint is not shown to be
converged. **Criterion 1: met.**

Recording-equal means over 15 recordings (one recording has no missing region):

| Path | Missing LSD dB | ERB RMS dB | ERB bias dB | Temporal corr | Paired preserved LSD | Clean preserved LSD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Untreated | 29.67 | 35.27 | -32.19 | 0.369 | 7.54 | 0.00 |
| Identity through DSP | 24.49 | 29.96 | -26.23 | 0.421 | 7.45 | 0.57 |
| Historical legacy | 14.75 | 14.65 | -11.60 | 0.608 | 7.48 | 5.03 |
| P74 parent (preserved) | 14.57 | 14.44 | -9.60 | 0.569 | 7.15 | 0.70 |
| P108 bin, seed 1 / 2 | 12.88 / 12.92 | 11.99 / 12.08 | -5.62 / -6.04 | 0.589 / 0.591 | 8.03 / 7.95 | 7.78 / 7.82 |
| **P108 band, seed 1 / 2** | **12.95 / 13.02** | **12.04 / 12.16** | -6.85 / -7.27 | 0.595 / 0.598 | 7.44 / 7.49 | 5.67 / 5.97 |
| Reference envelope, detector region | 10.45 | 9.02 | -5.17 | 0.720 | 7.02 | 0.40 |
| Reference envelope, reference region | 9.86 | 5.12 | -2.21 | 0.848 | 7.25 | 0.00 |

Paired recording contrasts (mean, 95% bootstrap interval):

| Contrast | ERB RMS dB | ERB bias dB | Temporal corr | Missing LSD dB | Paired preserved LSD |
| --- | --- | --- | --- | --- | --- |
| Band 1 - P74 parent | -2.39 [-3.54, -1.32] | +2.75 [+1.21, +4.37] | +0.03 [-0.02, +0.09] | -1.62 [-2.32, -0.95] | +0.29 [+0.05, +0.61] |
| Band 2 - P74 parent | -2.28 [-3.43, -1.16] | +2.33 [+0.73, +3.95] | +0.03 [-0.02, +0.09] | -1.56 [-2.25, -0.87] | +0.34 [+0.08, +0.69] |
| Band 1 - bin 1 | +0.05 [-0.33, +0.44] | -1.24 [-2.23, -0.31] | +0.00 [-0.00, +0.02] | +0.07 [-0.46, +0.63] | -0.60 [-1.11, -0.20] |
| Band 2 - bin 2 | +0.08 [-0.30, +0.45] | -1.23 [-2.06, -0.36] | +0.00 [-0.01, +0.01] | +0.10 [-0.39, +0.62] | -0.46 [-0.81, -0.16] |
| Ref. envelope: detector - reference region | +3.90 [+1.03, +7.39] | -2.96 [-5.88, -0.60] | -0.12 [-0.22, -0.03] | +0.58 [-0.49, +1.98] | -0.23 [-0.33, -0.13] |

**Criterion 2: met** in both seeds, with no adverse recording on ERB RMS. The band
arm also lowers missing LSD by 1.6 dB and shrinks the level deficit by 2.3-2.8 dB.
**Criterion 3: not met.** Band and bin have the same envelope error (intervals
cross zero). The bin arm is less quiet; the band arm is gentler on retained and
clean spectra.

**Clean input fails.** On the 16 clean works, preserved LSD rises to 5.7-6.0 dB
(band) and 7.8 dB (bin), against 0.70 dB for the P74 parent. Every work is
affected (1.3-10.7 dB per work). Mean relative complex error is 0.55% against
0.38%, so the added energy lies in weak high bins. The ported Rust detector is
active on about 99.9% of clean frames, because real music's high band sits more
than 60 dB below the spectral peak. The model never saw lossless input during
training and always adds high band.

**Region estimation** accounts for a large part of the remaining gap. With the
reference envelope, the detector region costs 3.9 dB ERB error against the
reference region. The learned envelope sits 3.0 dB above that detector-region
ceiling.

**Compute.** One frame of the band GRU takes 0.034 ms median and 0.071 ms p99
per channel in eager PyTorch on one CPU thread (2,800 steps, after warmup).
That is about 1/80 of the 2.667 ms hop at 48 kHz; P74 peaked at 2.45 ms. The
fill DSP and Rust/ONNX execution are not yet timed.

## Interpretation

A 219k-parameter causal envelope model trained for about 36 minutes on CPU beats
the preserved P74 parent on the missing-band rulers, at a small fraction of its
compute. The output representation (band versus bin) does not decide the gain.
The clean-input regression is a direct consequence of the training distribution
and the detector, and it blocks any deployment step. It is the registered
question for [P109](envelope-clean-preservation-p109.md).

## Limitations

- These are Python proxy outputs. They include no crossover, limiter or Rust
  stream, while the P74 parent and the other native paths do. Native
  correspondence is required before any comparison is treated as deployed.
- 15 exposed works, 10 s, three codecs; a development recheck. No listener
  ratings.
- One fixed budget, not shown converged. Two seeds.
- The ERB ruler shares the half-ERB band definition with the band arm's output
  (on a different STFT grid and hop). The bin arm reaching the same score is
  partial evidence that this is not the cause of the gain.

## Evidence

`data/runs/envelope-predictor-p108-20261122/`: `registration.json`, `cache/`,
`training/{band,bin}-{1,2}/` (endpoints, histories, logs, hashes),
`evaluation-frozen.json`, `records.jsonl`, `summary.json`, `contrasts.json`. Code:
`training/models/envelope_predictor.py`, `training/evaluation/envelope_fill.py`,
`training/scripts/run_envelope_p108.py`, `training/tests/test_envelope_p108.py`.
After evaluation, `envelope_fill.py` gained a `band_deficit` mode for P109. A
re-render of a P108 case reproduced every recorded score exactly.
