# P107 missing-band baseline and parametric oracle headroom

P107 re-scores existing frozen native renders on rulers restricted to the spectra
the codec actually removed, and compares them with reference-assisted ceilings for
a per-frame parametric high band. No model was trained and no weight changes. The
16 P88 works are exposed; this is a development recheck, not independent testing.

## Question

Earlier panels score fixed >=8 kHz bands. For AAC 96 and Vorbis q3 most of that
region is retained, so restoration effects are diluted by preserved spectra. P107
asks: (1) how far current models are from the reference *inside the missing
region*, and (2) how much of that gap a causal per-frame envelope applied to a
copy-up patch of the retained band could remove on the deployment grid.

## Method

- **Panel.** All 64 P88 cases (`data/runs/raw-hf-shape-p88-20261119-v2/cases.json`):
  16 stereo works x {MP3 CBR 64, AAC-LC 96, Vorbis q3} plus clean identity, 10 s
  scored region per case, the P88 sample offsets and native lag of 128 samples.
  Every native path is re-aligned and checked on its retained band before scoring.
- **Paths.** Untreated input; identity graph through native DSP; historical
  legacy; frozen P74 parents; all eight P88 endpoints (control/shape x two
  schedules x two parents); blind Rust DSP patch (`EnhancementMode::Spectral`,
  model gain, HEAD `e99cd9e`, binary SHA-256 `628f0bce...`); and five oracles.
- **Missing region** (`training/evaluation/missing_band.py`, `soundex-missing-band-v2`).
  Per clip and channel on a 2048/512 calibrated Hann STFT: bins >=4 kHz whose
  per-bin *median over frames* of the per-frame input/reference dB ratio is below
  -10 dB, excluding bins whose reference is 100 dB below the clip's loudest bin.
  -6/-20 dB and the long-term-mean rule are reported as sensitivity fields.
- **Rulers.** Missing-region LSD; half-ERB band-envelope RMS error and signed
  bias in dB (active frames > -90 dB, active bands > -100 dB re unit PCM mean
  square); energy-weighted correlation of per-band log envelopes over time;
  preserved-region LSD and relative complex error. A missing prediction scores
  as a constant envelope (correlation 0); an absent missing region is `null`.
- **Oracles** (`training/evaluation/envelope_oracle.py`). Offline Python on the
  Rust grid (FFT 256, hop 128, periodic Hann WOLA). The same median rule locates
  the top missing region; only masked bins in it are edited. `oracle_patch_level`:
  copy-up of 4 kHz..cutoff, one per-frame level. `oracle_patch_erb`: copy-up,
  per-frame half-ERB envelope. `oracle_noise_erb`: seeded noise with that
  envelope. `oracle_patch_binmag`: reference per-bin magnitude with copy-up phase.
  `oracle_complex`: reference complex bins (pipeline ceiling). Oracles bypass the
  Rust crossover and limiter and use the reference; **they are not deployable**.
- **Aggregation.** Cases are averaged per recording, then recordings equally;
  paired 2,000-sample recording bootstrap, seed 71. Recording `94f5eb153286` has no
  missing region under the primary mask (all three codecs). It is `null`,
  excluded from missing-band means with its identity disclosed, and 15
  recordings remain.

Registration with code, binary, panel hashes and all constants was written
before each scoring run.

## Results

Recording-equal means over 15 recordings, three codecs; lower is better except
correlation. "Gap closed" is (untreated - path) / (untreated - `oracle_patch_erb`)
on ERB-envelope RMS error.

| Path | Missing LSD dB | ERB RMS dB | ERB bias dB | Temporal corr | Gap closed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Untreated input | 29.67 | 35.27 | -32.19 | 0.369 | 0.00 |
| Identity through DSP | 24.49 | 29.96 | -26.23 | 0.421 | 0.18 |
| Blind DSP patch | 18.18 | 19.59 | -13.21 | 0.518 | 0.52 |
| Historical legacy | 14.75 | 14.65 | -11.60 | 0.608 | 0.68 |
| P74 parent (20261013) | 14.57 | 14.44 | -9.60 | 0.569 | 0.69 |
| P88 control (852/20261013) | 14.85 | 15.03 | -10.59 | 0.561 | 0.67 |
| P88 shape (852/20261013) | 14.79 | 14.94 | -10.44 | 0.561 | 0.67 |
| Oracle: copy-up, one level | 13.91 | 11.65 | -1.85 | 0.639 | 0.78 |
| Oracle: copy-up, half-ERB envelope | 9.86 | 5.12 | -2.21 | 0.848 | 1.00 |
| Oracle: noise, half-ERB envelope | 8.86 | 5.67 | -3.95 | 0.863 | 0.98 |
| Oracle: reference bin magnitude | 8.63 | 4.88 | -2.69 | 0.861 | 1.01 |
| Oracle: reference complex | 2.61 | 3.65 | -1.54 | 0.932 | 1.05 |

Paired recording contrasts (mean, 95% recording-bootstrap interval):

| Contrast | ERB RMS dB | ERB bias dB | Temporal corr | Missing LSD dB |
| --- | --- | --- | --- | --- |
| Envelope oracle - P74 parent | -9.32 [-12.65, -6.44] | +7.39 [+3.92, +11.30] | +0.274 [+0.189, +0.362] | -4.71 [-6.25, -3.29] |
| One-level - half-ERB oracle | +6.53 [+5.21, +7.84] | +0.36 [-1.63, +2.32] | -0.207 [-0.301, -0.127] | +4.05 [+3.07, +5.33] |
| Noise - copy-up (both half-ERB) | +0.55 [+0.39, +0.72] | -1.73 [-1.94, -1.49] | +0.013 [-0.002, +0.035] | -1.00 [-1.26, -0.74] |
| P88 shape - control (852/20261013) | -0.094 [-0.130, -0.064] | +0.145 [+0.121, +0.172] | -0.000 [-0.001, +0.000] | -0.057 [-0.080, -0.037] |
| P88 control - its P74 parent (852) | +0.596 [+0.348, +0.824] | -0.985 [-1.154, -0.804] | -0.008 [-0.016, -0.001] | +0.275 [+0.109, +0.439] |
| P74 parent - legacy | -0.21 [-1.24, +0.90] | +2.00 [+0.40, +3.57] | -0.032 [-0.067, -0.002] | -0.18 [-0.87, +0.54] |

Per codec, the envelope oracle reaches ERB RMS 3.6 (AAC), 6.6 (MP3) and 5.1 dB
(Vorbis) against 14.7, 14.7 and 13.9 dB for the P74 parent.

Clean identity role (16 works): preserved LSD is 0.70 dB for the P74 parent,
0.57 dB for identity DSP, 2.42 dB for the blind DSP patch and 5.03 dB for legacy.
Oracles edit nothing without a missing region (0 dB).

## Interpretation

1. **Every trained model leaves the missing band too quiet.** Mean ERB bias is
   -9.6 to -11.6 dB; 14 of 15 recordings are below -1 dB for the P74 parent, and
   down to -25.6 dB. These renders preserve the model's own level
   (`HighBandGain::Model`); the deficit belongs to model output after fusion. This
   is the outcome predicted for per-bin magnitude/complex objectives on
   unpredictable texture, which reward the conditional mean.
2. **A per-frame envelope has large measured headroom.** On these rulers, the
   reference half-ERB envelope applied to a copy-up patch lowers ERB error by
   9.3 dB and raises temporal correlation by 0.27 against the best frozen
   model, on all 15 recordings. Its interval is far from zero.
3. **Envelope resolution matters; fine structure matters much less here.** One
   level per frame loses 6.5 dB of ERB error against half-ERB bands. Seeded
   noise and copy-up with the same envelope differ by about 0.5 dB on ERB error
   and 1 dB on LSD. Random-phase frames lose about 2-4 dB in WOLA because the
   spectra are inconsistent (also seen in the synthetic tests).
4. **Recent experiment effects are two orders of magnitude smaller.** P88 shape
   versus control changes ERB error by -0.03 to -0.09 dB. The 32-step
   continuation itself costs +0.6 dB against its P74 parent. The legacy model is
   within ±1 dB of the P74 parent on ERB error.
5. **The blind DSP patch closes about half the gap** but costs 2.4 dB preserved
   LSD on clean input. Its level policy, not the patch, limits it (-13.2 dB bias).

## Preserved candidate

The P74 parent `20261013-magnitude` has the lowest envelope error and smallest
level deficit among trained paths here, with 0.70 dB clean preserved LSD. Its
checkpoint (`5f968388...`), ONNX (`87485e30...`) and the diagnostic context-replay
bridge needed for its four-input graph are preserved read-only, with a receipt,
in `data/preserved/p107-best-p74-20261013-magnitude/`. It was selected after
scoring on this exposed panel. It is the starting point for the next
comparison, not a promoted, released or independently qualified weight.

## Limitations

- The oracles use the lossless reference for the region and envelope. They
  measure representational headroom, not learnability from compressed input.
  The ERB rulers share the oracles' band definition (on a different grid and
  hop). LSD and temporal correlation are partly independent checks.
- Oracles bypass the Rust crossover and limiter. Identity DSP alone moves the
  missing-band rulers (35.3 to 30.0 dB ERB error), so model paths include a DSP
  contribution that the oracles do not.
- The static region cannot repair intermittent per-frame dropouts or in-band
  holes below it. `oracle_complex` therefore keeps 3.65 dB ERB error, mostly
  from MP3 64 (5.3 dB) and Vorbis.
- 15 exposed works, 10 s each, three codecs. No perceptual metric or listener
  rating exists for these paths yet; the pack below awaits ratings. Relative scores do not establish audibility.
- Mask, floors and band width were fixed before the v3 run but are new rulers.
  The -6/-20 dB and mean-rule fields remain in every record.

## Invalid measurements retained

Two earlier runs used the same frozen renders and are kept with `superseded.json`.
`...-20261122` required 90% of all bins to Nyquist to be missing. References at
their noise floor near Nyquist left every oracle path untreated.
`...-20261122-v2` used sqrt-Hann oracle analysis, whose sidelobes leaked low-band
power over the high band, and a long-term-mean mask dominated by the few frames
in which codecs keep the high band (median per-frame ratio -43 dB versus mean -2
to -9 dB at 12-16 kHz). Both were found while validating the oracle pipeline,
before results were interpreted. Neither change was motivated by model rankings.

## Evidence

- `data/runs/missing-band-baseline-p107-20261122-v3/`: `registration.json`,
  `records.jsonl` (1,216 rows), `summary.json`, `contrasts.json`,
  `report-contrasts.json`, blind DSP renders under `spectral/`.
- `data/runs/missing-band-baseline-p107-assets-20261122/`: frozen stream-eval
  binary and its commit.
- Blinded listening pack, ratings outstanding:
  `data/runs/missing-band-listening-p107-20261122/listening/index.html`. It has
  16 trials, one per work, with codecs rotated by seed 107. Each trial holds
  untreated, P74 parent, P88 852 shape, legacy, blind DSP patch, both envelope
  oracles, hidden reference and a 3.5 kHz anchor. The plan was frozen in
  `frozen-plan.json` before any audio was written. The key is in
  `listening-key.json`, outside the listener folder.
- Superseded: `data/runs/missing-band-baseline-p107-20261122{,-v2}/`.
- Code: `training/evaluation/missing_band.py`, `training/evaluation/envelope_oracle.py`,
  `training/scripts/score_missing_band.py`, `training/scripts/build_p107_listening.py`,
  `training/tests/test_missing_band.py`.

## Proposed evaluation changes (not adopted)

Recorded for review; [protocol v2](evaluation-protocol.md) remains in force.

1. Score restoration inside the per-clip missing region (median rule above).
   Keep the >=8 kHz panel for continuity and preserved-band checks.
2. Report signed level bias beside every relative error, so that "quieter and
   therefore closer" is visible.
3. Report untreated input and the envelope oracle as fixed anchors and state the
   gap closed. Anchors put sub-0.1 dB effects in context without a practical
   threshold being fitted to known results.
4. Calibrate rulers against blinded ratings (P88/P89 packs, then a pack that
   includes `oracle_patch_erb`). Set any future non-inferiority margin from that
   calibration before training.
5. Keep clean-input preservation as a separate gate (detector false triggers and
   preserved-band error), not a condition inside each restoration screen.

## Next registered step

A **learned envelope** comparison: a causal model predicts half-ERB per-frame
log-gains (and the region start) from compressed input on the existing STFT,
and drives the copy-up patch. Compared at an equal update budget, under the same
rulers and native rendering, against the frozen P74 parent and a matched per-bin
control. This differs from P5-P7: they factorized envelope and shape *inside*
per-bin output with the existing fusion. Here the output parameterization and
synthesis change. Before training: port the patch and envelope application
into the Rust DSP and confirm that, fed reference envelopes, it reproduces the
`oracle_patch_erb` scores through the native stream. Also measure its per-hop
cost. Consult this report and the P5-P7 entries when registering it.
