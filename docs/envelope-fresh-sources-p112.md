# P112 first source-disjoint test of the gated envelope mode

P107-P111 made every design decision on the 16 exposed P88 works. P112 freezes the
[P111](envelope-fill-gate-p111.md) gated candidate (both seeds) and tests it,
together with the preserved native P74 parent, on works that no training,
calibration or evaluation run has touched. No default, release or listening
decision follows.

## Reservation and coverage

The P88 isolation rule ran against every processed/archived manifest and
historical run catalog (24 history files), excluding connected file,
decoded-PCM and work aliases. It left **24 eligible stereo works, all
electronic**. Sixteen were chosen with the P88 genre/author/stable-hash
balancing (tag 112) and written to
`data/runs/envelope-fresh-p112-20261124/evaluation-catalog.jsonl`, so future
training excludes them; eight stay unexposed.

**Coverage gap:** one genre and 11 authors. One author (Mesake) also has a
different work in the P88 panel, so the test is source-disjoint, not
artist-independent. Pairs were prepared exactly as in P87/P88: MP3 64, AAC 96,
Vorbis q3 and clean; alternating 44.1/48 kHz; aligned; 10 s scored. Renders
use the preserved P74 bridge (SmoothKnee) and the P111 stream-eval binary
(`f18bd82`, SmoothKnee). Rules and hashes were frozen before preparation.

## Registered rules

`data/runs/envelope-fresh-p112-20261124/registration.json`:
1. **Restoration:** in both seeds, missing-region ERB RMS below the native P74
   parent, with the paired recording interval below zero.
2. **Clean input:** in both seeds, preserved LSD versus P74 not an increase;
   absolute clean high-band error is reported beside it.
3. **Reported:** the remaining rulers, per-codec means and adverse recordings.

## Results

The paired role has 15 recordings: work `2f3436aa6348` is undefined on the
missing-band rulers under strict per-source pairing. The clean role has 16.

| Path | Missing LSD | ERB RMS | ERB bias | Temporal corr | Paired preserved LSD | Clean preserved LSD | Clean HF error dBFS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Untreated | 40.74 | 45.96 | -41.18 | 0.424 | 7.95 | 0.00 | - |
| P74 parent (native) | 15.20 | 11.98 | -6.73 | 0.784 | 8.83 | 1.06 | -90.9 |
| Gated envelope, seed 1 | 15.58 | 12.57 | -7.60 | 0.785 | 9.18 | 2.33 | -89.6 |
| Gated envelope, seed 2 | 15.16 | 12.09 | -7.51 | 0.801 | 9.00 | 1.72 | -91.4 |

| Gated - P74 (seed 1 / seed 2) | Interval and adverse recordings |
| --- | --- |
| ERB RMS | +0.65 [-0.68, +2.31], 7/15 / +0.15 [-0.86, +1.20], 7/15 |
| Missing LSD | +0.39 [-0.64, +1.63] / -0.04 [-0.86, +0.87] |
| Temporal corr | +0.001 [-0.029, +0.030] / +0.018 [+0.002, +0.033] |
| Paired preserved LSD | +0.35 [+0.12, +0.61] / +0.17 [-0.01, +0.35] |
| Clean preserved LSD | **+1.27 [+0.18, +2.61]** / +0.66 [-0.20, +1.75] |
| Clean HF error | +1.3 [-6.1, +8.3] / -0.5 [-7.2, +4.9] dB |

ERB RMS by codec (P74 / seed 1 / seed 2):

| Codec | P74 | Seed 1 | Seed 2 |
| --- | ---: | ---: | ---: |
| MP3 64 | 14.20 | 13.78 | 13.32 |
| AAC 96 | 12.01 | 13.30 | 11.93 |
| Vorbis q3 | 10.04 | 11.12 | 11.47 |

**Rule 1: failed in both seeds**; the development-panel advantage of about
1.7 dB did not transfer. **Rule 2: failed** for seed 1 (seed 2 passes).
Absolute clean high-band error is level with P74 in both seeds.

## Supplementary diagnostic (not registered)

With the reference envelope, the same copy-up fill reaches ERB RMS 5.32 dB
(reference region) and 5.10 dB (detector region) on these works. That
diagnostic averages all 16 recordings. So the parametric representation still
holds about 7 dB of headroom here, and the learned predictor and P74 sit at the
same distance from it. Records: `supplementary-oracles.jsonl`.

## Interpretation

On fresh works the learned envelope ties P74 at best, while being far cheaper,
and costs more on clean input in one seed. Two explanations cannot be
separated:
- **Development selection:** five design rounds on one exposed panel.
- **Genre shift:** these works are all electronic, with a much louder missing
  band (untreated ERB 46 vs 35 dB), and P74 does far better here than on P88.

The headroom result suggests the predictor's generalization, not the output
representation, now limits quality. P107-P111's gains are development
evidence only. The gated candidate stays experimental, and P74 stays the
preserved native reference.

## Next steps

- **Broader reserved evaluation.** Only eight isolated works remain, all
  electronic. New lossless acquisitions with documented rights in other genres
  are needed before any further transfer claim.
- **Generalization of the predictor.** Before more design rounds on P88:
  longer training (the P108 curve was still falling), more source-time
  coverage, and a cross-genre validation split. Keep the 8 remaining works
  reserved.
- **Listening.** The P111 pack is still outstanding.

## Evidence

`data/runs/envelope-fresh-p112-20261124/`: `registration.json`,
`evaluation-catalog.jsonl`, `inputs/`, `cases.json` (paths relative to the
repository root), `renders/`, `records.jsonl`, `summary.json`,
`contrasts.json`, `supplementary-oracles.jsonl`. Code:
`training/scripts/run_envelope_fresh_p112.py`.
