# P114 envelope-gap diagnosis, electronic panel and hip-hop sourcing

[P113](envelope-scale-p113.md) beat the native P74 parent on a fresh multi-genre
panel, but not on electronic ([P112](envelope-fresh-sources-p112.md)) or hip hop.
P114 asks why, prepares a fresh electronic panel, and surveys hip-hop sources.

It involves no training and makes no improvement claim. The diagnosis uses the
already-exposed P112 and P113 panels.

## Training coverage

The 235 P113 training works are: rock 47, electronic 46, classical 45, ambient
25, unlabeled 24, folk 20, pop 11, jazz 5, blues 5, country 4, **hip hop 2**,
funk 1. Hip hop is a coverage gap; electronic is among the best-covered genres.

## Diagnosis

`training/scripts/diagnose_envelope_p114.py` uses the preserved P113 seed-1
model and the exact Python fill proxy with SmoothKnee. On P112 the gated
variant reproduces the native P113 score (11.42 dB).

Missing-band ERB RMS in dB (recording means):

| Variant | P112 electronic (16) | P113 multi-genre (24) | Ambient (4) | Rock (5) | Hip hop (4) |
| --- | ---: | ---: | ---: | ---: | ---: |
| P113 model, gated | 11.42 | 10.50 | 14.85 | 8.26 | 11.21 |
| P113 model, no gate | 11.22 | 10.32 | 14.66 | 8.00 | 11.06 |
| Model band shape, reference total level per frame | 11.42 (bias +3.8) | 11.71 (bias +4.8) | 12.59 | 10.88 | 11.48 |
| Reference deficits, detector region | 5.22 | 5.09 | 7.86 | 4.26 | 5.29 |
| Reference deficits, detector region, gated | 5.34 | 5.22 | 7.98 | 4.42 | 5.38 |
| Reference envelope, reference region, copy-up | 5.20 | 4.03 | 3.40 | 3.46 | 4.49 |
| Reference envelope, reference region, noise | 5.93 | 4.71 | 4.29 | 4.36 | 5.06 |

Detector cutoff minus the reference missing-region start has a median absolute
error of 850 Hz (P112) and 1,280 Hz (P113).

**Findings:**
- **Gate:** costs about 0.2 dB everywhere.
- **Detector region:** not the bottleneck. With reference deficits it reaches
  about 5.1-5.3 dB on both panels. Ambient is the exception, where the region
  costs about 4.5 dB against the reference region.
- **Copy-up source:** better than shaped noise in every group, electronic
  included.
- **Level:** not the problem. Giving the model the reference total deficit per
  frame leaves its error unchanged on electronic and makes it worse elsewhere,
  flipping the bias from -6 to +4-5 dB. The error lies in how the predicted
  deficit is distributed across bands and frames.
- **The learned-vs-reachable gap is similar across groups:** electronic 6.2 dB,
  multi-genre 5.4 dB, hip hop 5.9 dB. P74's relative strength on electronic
  (12.0 dB vs 21.2 on ambient) explains why P113 does not win there. Electronic
  is not uniquely hard for the envelope model.

**Implication.** The next model change should target per-band, per-frame
deficit prediction for every genre, with more distinct works and
genre-balanced sampling. A level calibration, a gate change or a
genre-specific model are not supported by this evidence.

## Fresh electronic panel

`training/scripts/reserve_panel_p114.py` (built on the factored
`reserve_panel` in `reserve_panel_p113.py`) audited the 652 unapproved
electronic downloads offline: 649 verified and 3 failures retained.

The P88 isolation rule left 597 eligible stereo works. Sixteen were reserved,
one per author:
`data/runs/reserved-panel-p114-20261124/evaluation-catalog.jsonl`. This was
done before any electronic-targeted change was designed, so the panel cannot be
chosen around one.

Several authors (Sciurid, James Lowery, Lee Rosevere) also appear in earlier
panels, so the panel is not artist-independent. 581 eligible electronic works
remain unused (`unreserved-eligible.jsonl`). The P113 run has about 175 more in
other genres.

## Hip-hop sources (candidates only)

A read-only archive.org query (hip-hop/rap subjects, netlabels collection,
FLAC) returned 365 items. Only 37 are CC BY or CC0, from 16 creators, and
several of those are electronic or experimental releases with a hip-hop tag.
One duo (Deadman & Digi Hartatak) accounts for about a third.

The rest are NonCommercial, NoDerivatives, share-alike, public-domain-mark or
unlicensed. `legal/TRAINING_DATA.md` names CC0 and attributable permissive
recordings as suitable and leaves rights questions to maintainer review.

The list is in `data/source-discovery/hiphop-candidates-20261124.json`.
Nothing was downloaded. Its open questions:
- per-creator AI-training terms;
- genre verification;
- the non-BY licence classes;
- overlap with P113 panel authors.

## Evidence

- `data/runs/envelope-diagnosis-p114-20261124/` (`records.jsonl`, `summary.json`).
- `data/runs/reserved-panel-p114-20261124/`.
- `data/source-discovery/hiphop-candidates-20261124.json`.
