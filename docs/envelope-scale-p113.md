# P113 new multi-genre panel and broader envelope training

[P112](envelope-fresh-sources-p112.md) showed that the P107-P111 envelope gains
did not transfer to 16 fresh electronic works. P113 does two things:
1. **Panel.** Reserves a new source-disjoint panel outside electronic music.
2. **Training.** Retrains the frozen envelope recipe with broader exposure and
   a longer budget, judged natively on that panel.

No default, release or listening decision follows.

## New reserved panel

The 221 downloaded, non-electronic, not-yet-approved rows of
`data/catalogs/new-acquisition-plan.jsonl` were verified offline with the
acquisition audit (`training/scripts/reserve_panel_p113.py`). The audit checks
publisher size and MD5 and runs a full decode with quality flags; all 221
passed.

The P88 isolation rule (connected file/PCM/work aliases against every history
file and catalog) and publisher rights evidence left 199 eligible stereo works.
Twenty-four were reserved with genre/author/hash balancing (tag 113):
- **Genres:** rock 5, ambient 4, metal 4, hip hop 4, unlabeled 4, pop 3.
- **Authors:** 19 (Balthasar 4, 88 Diagrams 3; Sciurid also appears in P112).
- **Catalog:** `data/runs/reserved-panel-p113-20261124/evaluation-catalog.jsonl`,
  written before training. About 175 eligible works remain unexposed.

Pairs were prepared as in P87/P88/P112 (96 cases). This is source-disjoint,
not artist-independent. Classical, jazz, blues, folk and country are absent.

## Training change (registered before training)

The frozen P109/P111 recipe is unchanged:
- deficit target with a -30 dB floor and 25% identity pairs;
- `EnvelopePredictor('band', 128, 2)` with 219,297 parameters;
- the deployment gate [0, 6] dB.

Two things change:
- **Data:** every eligible TRAIN row with two segments each (66,149 rows,
  132,298 segments), instead of 16,000 rows.
- **Budget:** 65,536 updates instead of 16,384.

Reserved catalogs, including P112 and this panel, are excluded (18 library
works). Both changes are made together, so their effects are confounded. The
work count is unchanged at 235 works: the extra rows are more crops, codecs and
channel views of the same works, not more music.

The rationale was registered with the run: validation was still falling at
16,384 updates in P108/P109, P84 found source-time coverage helps transfer, and
P112 showed a transfer gap. Seeds 1 and 2, final endpoints only, about 3 hours
each on CPU.

Validation at the end was 10.24/10.22 dB deficit error (degraded) and 2.45/2.30
(identity), still decreasing slowly. Per-genre validation (monitoring only):
ambient 8.1, classical 9.6, unlabeled 9.8, folk 10.1, rock 10.2, electronic
10.6, jazz 11.0, blues 12.2 dB.

## Registered rules

`data/runs/envelope-scale-p113-20261124/registration.json`:
1. **Primary**, on the P113 panel, native, SmoothKnee: in both seeds, ERB RMS
   below the native P74 parent (interval below zero), and clean preserved LSD
   not an increase.
2. **Incremental:** in both seeds, ERB RMS below the gated P111 candidate of
   the same seed.
3. **Reported:** the P112 recheck and the remaining rulers.

## Results on the P113 panel

Paired role: 23 recordings (one undefined under strict pairing). Clean role: 24.

| Path | Missing LSD | ERB RMS | ERB bias | Temporal corr | Paired preserved LSD | Clean preserved LSD | Clean HF error dBFS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Untreated | 36.65 | 41.90 | -38.49 | 0.355 | 7.62 | 0.00 | - |
| P74 parent | 14.89 | 12.81 | -7.33 | 0.574 | 8.26 | 0.81 | -83.9 |
| P111 gated, seed 1 / 2 | 13.56 / 13.90 | 11.01 / 11.49 | -6.54 / -7.30 | 0.617 / 0.614 | 8.11 / 7.95 | 0.89 / 0.84 | -91.6 / -91.9 |
| **P113, seed 1 / 2** | **13.29 / 13.53** | **10.50 / 10.90** | -5.76 / -6.97 | 0.630 / 0.621 | 7.99 / 8.02 | **0.42 / 0.45** | **-93.2 / -93.4** |

P113 minus P74 (seed 1 / seed 2):

| Ruler | Seed 1 | Seed 2 |
| --- | --- | --- |
| ERB RMS | -2.17 [-3.77, -0.58] | -1.82 [-3.54, -0.23] |
| Missing LSD | -1.46 [-2.63, -0.26] | -1.26 [-2.50, -0.06] |
| Temporal corr | +0.06 [+0.02, +0.10] | +0.05 [+0.02, +0.09] |
| Paired preserved LSD | -0.27 [-1.56, +0.57] | -0.24 [-1.44, +0.53] |
| Clean preserved LSD | -0.38 [-1.01, -0.04] | -0.36 [-0.93, -0.04] |
| Clean HF error | -9.2 [-15.3, -3.8] dB | -9.5 [-15.7, -3.9] dB |

**Rule 1: met in both seeds.** P113 restores the missing band better than P74,
and it changes lossless input less, by both relative and absolute measures.

P113 minus P111 (same seed): ERB RMS -0.53 [-1.47, +0.59] and -0.68 [-1.49,
+0.23]; clean preserved LSD -0.46 and -0.39 dB, with upper bounds at 0.00.
**Rule 2: not met.** The broader exposure and longer budget give a consistent
but unresolved additional gain, and a clean-input gain.

**P112 recheck** (exposed, electronic): P113 minus P74 ERB RMS is
-0.54 [-2.00, +1.23] and +0.07 [-1.12, +1.33]; clean preserved LSD is +0.90 and
+0.61 (crossing zero). There is still no resolved advantage on these works.

## Supplementary (not registered)

- **P111 gated minus P74 on the P113 panel:** ERB RMS -1.64 [-3.26, -0.21]
  and -1.14 [-2.72, +0.24]. The original candidate also transfers to these
  non-electronic works. P112's failure therefore looks at least partly
  genre-specific, though development selection cannot be excluded.
- **ERB RMS by genre** (3-5 works each, descriptive only), P74 versus P113
  seed 1 / seed 2:

| Genre | P74 | P113 seed 1 | P113 seed 2 |
| --- | ---: | ---: | ---: |
| Ambient | 21.2 | 14.9 | 14.6 |
| Unlabeled | 13.6 | 9.1 | 9.7 |
| Rock | 10.6 | 8.3 | 8.9 |
| Pop | 10.9 | 9.0 | 10.7 |
| Metal | 9.9 | 10.8 | 9.5 |
| Hip hop | 10.8 | 11.2 | 12.4 |

The gain is concentrated in ambient, unlabeled and rock. Hip hop and (from
P112) electronic favor P74.

## Interpretation

This is the first envelope candidate to meet its registered primary criterion
on fresh, source-disjoint works, on both restoration and clean-input
preservation, at well under 1 ms per hop (P110). The evidence remains limited:
- one 24-work panel with 3-5 works per genre;
- two seeds and no listener ratings;
- no improvement on electronic or hip hop;
- the P113 panel is now exposed.

P113 is preserved as the leading experimental candidate. P74 remains the
production-path reference until listening and broader reserved evaluation
support a change.

## Next steps

- **Listening.** Build a blinded pack from P113 native renders on this panel,
  including clean-input trials.
- **Weak genres.** Hip hop and electronic. The remaining ~175 isolated works
  can supply future reserved panels. Register a genre-aware study (for example
  more electronic/hip hop exposure, or a learned bypass) before scoring any of
  them.
- **Runtime.** Vectorize the GRU, and measure with a realtime-scheduled
  producer.

## Evidence

- `data/runs/reserved-panel-p113-20261124/`: verified candidates, failures,
  reservation and catalog.
- `data/runs/envelope-scale-p113-20261124/`: registration, cache, training,
  artifacts, panel, renders, records, summary, contrasts, `supplementary.json`.
- Preserved: `data/preserved/p113-envelope-scale/`.
- Code: `training/scripts/reserve_panel_p113.py` and
  `training/scripts/run_envelope_scale_p113.py`.
