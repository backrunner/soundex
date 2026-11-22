# P109 clean-input preservation for the learned envelope

[P108](learned-envelope-p108.md) showed that a 219k-parameter causal envelope GRU
beats the preserved P74 parent on missing-band envelope error. It also showed
that the GRU adds high-band energy to lossless input: clean preserved LSD was
5.7-6.0 dB against 0.70 dB for P74. The model had never seen lossless input, and
the ported Rust detector is active on about 99.9% of clean frames. P109 tests two
remedies as a 2x2 factorial with the P108 data, model, budget and seeds.

## Design

- **Target.** `level` is the absolute reference band energy, as in P108; the
  fill adds max(predicted - present, 0). `deficit` is the missing energy itself,
  max(reference - input, 10^-3 x reference) per band; the fill adds the
  prediction directly. The -30 dB floor bounds the level change from a correct
  prediction on lossless input at +0.0043 dB.
- **Identity share.** 0, or 25% of batch items with lossless input and the same
  lossless reference. For `deficit`, identity items take the floor as target.
- **Fixed for every cell.** The same 48,000 P108 segments and exclusions,
  EnvelopePredictor('band', 128, 2), 16,384 updates, batch 64, seeds 1 and 2.
  Two CPU threads per run; about 47 minutes each.
- **Evaluation.** P107 rulers on the exposed P88 panel through the P108 fill
  proxy (ported Rust detector, copy-up, Hann WOLA; no crossover or limiter).
  Final endpoints only.

The registration (`data/runs/envelope-clean-preservation-p109-20261122/registration.json`)
froze two rules before training:
1. **Clean preservation:** for each target, identity 0.25 lowers clean-role
   preserved LSD against identity 0 in both seeds.
2. **Advancement:** the primary candidate `deficit`/0.25 advances to the native
   Rust port only if, in both seeds, it beats the P74 parent on missing-region
   ERB RMS (interval below zero), and its clean preserved LSD against P74 is not
   an increase (interval not entirely above zero).

## Results

Recording-equal means. The paired role has 15 recordings with a missing
region; the clean role has 16 works.

| Path | Missing LSD | ERB RMS | ERB bias | Temporal corr | Paired preserved LSD | Clean preserved LSD | Clean complex rel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Untreated | 29.67 | 35.27 | -32.19 | 0.369 | 7.54 | 0.00 | 0.000 |
| P74 parent (native) | 14.57 | 14.44 | -9.60 | 0.569 | 7.15 | 0.70 | 0.004 |
| level / 0 (s1, s2) | 12.87, 12.85 | 11.95, 11.96 | -6.52, -7.01 | 0.593, 0.592 | 7.47, 7.45 | 6.30, 5.88 | 0.008, 0.010 |
| level / 0.25 | 13.00, 13.30 | 12.22, 12.70 | -6.77, -8.47 | 0.597, 0.586 | 7.41, 7.36 | 2.44, 2.42 | 0.002, 0.002 |
| deficit / 0 | 13.12, 13.10 | 12.44, 12.32 | -7.60, -7.00 | 0.588, 0.596 | 7.47, 7.48 | 5.37, 5.30 | 0.005, 0.004 |
| **deficit / 0.25** | **13.36, 13.32** | **12.92, 12.89** | -8.28, -8.64 | 0.575, 0.576 | 7.47, 7.41 | **1.11, 0.56** | **0.000, 0.000** |
| Reference envelope, detector region | 10.45 | 9.02 | -5.17 | 0.720 | 7.02 | 0.40 | 0.001 |

Paired contrasts (mean, 95% recording-bootstrap interval; seed 1 / seed 2):

| Contrast | ERB RMS dB | ERB bias dB | Clean preserved LSD dB |
| --- | --- | --- | --- |
| level: id25 - id0 | +0.27 [-0.01, +0.59] / +0.74 [+0.42, +1.11] | -0.25 / -1.46 | -3.86 [-4.96, -2.95] / -3.46 [-4.54, -2.58] |
| deficit: id25 - id0 | +0.48 [+0.07, +0.91] / +0.56 [-0.00, +1.23] | -0.68 / -1.64 | -4.27 [-5.55, -3.15] / -4.73 [-5.93, -3.67] |
| deficit/id25 - P74 parent | -1.52 [-2.81, -0.30] / -1.55 [-2.83, -0.32] | +1.33 / +0.97 | +0.40 [-0.42, +1.23] / -0.14 [-0.70, +0.37] |
| level/id25 - P74 parent | -2.22 [-3.26, -1.27] / -1.74 [-2.86, -0.69] | +2.84 / +1.13 | +1.74 [+0.89, +2.46] / +1.72 [+0.83, +2.51] |
| deficit/id25 - level/id25 | +0.70 [+0.13, +1.37] / +0.19 [-0.24, +0.61] | -1.51 / -0.17 | -1.33 [-1.77, -0.86] / -1.86 [-2.34, -1.42] |

**Rule 1: met** for both targets in both seeds.
**Rule 2: met** in both seeds. The `deficit`/0.25 cell advances to the native
Rust port.

Against the P74 parent on compressed input, the candidate's paired
preserved-band LSD is +0.33 [+0.11, +0.62] and +0.27 [+0.10, +0.49] dB. This
small retained-band cost remains to be checked natively.

**Reproduction.** The `level`/0 cell repeats the P108 band design, but its batch
RNG stream differs because identity sampling draws a random number even at
share 0. Its ERB RMS is within 0.10/0.20 dB of P108 and its clean LSD within
0.63/0.09 dB. That is consistent, not an exact replay.

## Interpretation

Training on lossless input is what fixes the clean regression: both targets gain
3.5-4.7 dB clean LSD. The deficit target adds a further 1.3-1.9 dB of clean
protection over the level target with identity pairs. Clean protection costs
0.3-0.7 dB of the P108 restoration gain, and the level deficit widens by
1.0-1.6 dB. The candidate keeps about 1.5 dB of envelope-error improvement and
1.2 dB of missing-band LSD improvement over the preserved P74 parent, at about
1/80 of its per-hop compute (P108). It remains 3.9 dB above the reference
envelope on the same detector region. Better region estimation and a less
conservative (still clean-safe) level are the next improvement levers.

## Limitations

- Python proxy without crossover, limiter or Rust stream; the P74 parent is
  native. Native correspondence decides the next step.
- Exposed development panel (15/16 works, 10 s, three codecs); two seeds; one
  budget; no listener ratings.
- The deficit floor and the 25% identity share were set prospectively, not
  tuned. Other values are untested.

## Evidence and preservation

`data/runs/envelope-clean-preservation-p109-20261122/`: `registration.json`,
`cache/`, `training/<target>-id<share>-<seed>/`, `evaluation-frozen.json`,
`records.jsonl`, `summary.json`, `contrasts.json`. The registered candidate
endpoints for both seeds are preserved read-only with a receipt in
`data/preserved/p109-envelope-deficit-id25/`. The P74 parent remains preserved
as the best native-validated path. Code: `training/scripts/run_envelope_p109.py`
and the P108 modules (`band_deficit` fill mode).

## Next step

Port the half-ERB deficit fill (detector region, hop-coherent copy-up, additive
band energy) into `soundex-dsp`, and export the GRU to ONNX with explicit state.
Verify native correspondence against this proxy on the same cases, then time
model plus fill per hop on the realtime bench. Score the native outputs with
the P107 rulers against the native P74 parent, including clean, stereo and
silence cases. Then build a blinded pack.
