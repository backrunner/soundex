# Conservative rights screen for new training and evaluation material

Decided 2026-11-25: music used for new training and evaluation work should carry
no avoidable legal risk. `training/data/rights_screen.py`
(`soundex-rights-screen-v3`) implements that as a mechanical, auditable screen.

It complements, and does not replace, [`legal/TRAINING_DATA.md`](../legal/TRAINING_DATA.md)
and the maintainer rights review required before any weight release. Passing
the screen reduces risk but cannot eliminate it: a CC grant covers only rights
the licensor holds, and undisclosed samples cannot be detected from metadata.

## Rules

A track passes only if all of the following hold.

1. **Licence.** The publisher item licence is CC0 1.0 or CC BY 2.0-4.0
   (including ported versions), and it matches the catalog grant.
2. **No restriction text.** The item and file metadata contain no other
   restriction: NC, ND, SA or "all rights reserved".
3. **No AI or data-use language.** No mention of AI, machine learning, neural,
   dataset, data mining, TDM, or training a model. Any mention excludes the item;
   false positives are accepted.
4. **No third-party indicators** in the title, description, subjects, notes,
   file entry or creator credits:
   - remix, mash-up, bootleg;
   - cover or cover version (v2: cover art or photo is not a cover version);
   - sample(s/d)/sampling, Freesound;
   - feat./featuring/ft.;
   - edit or re-edit, medley, tribute, parody, a cappella, interpolation, flip.

   These apply at item level: an album-level mention excludes every track in
   that album.
5. **Attribution.** An attributable creator is recorded (v3). CC BY requires
   attribution.
6. **No blocked creator.** No creator with a recorded AI-training prohibition
   (Loyalty Freak Music and aliases).
7. **Original file.** The file is the publisher's original upload.

v2 and v3 were changes in meaning, fixed before the screen was used to select
any material. v2 stopped "cover photograph", "cover artwork" and "genres
covered" from counting. v3 added creator-credit and attribution checks after
hip-hop candidates showed credits in the creator field and missing creators.

## Results

| Pool | Pass | Main exclusion reasons |
| --- | ---: | --- |
| `current-approved.jsonl` (360; a later catalog, mostly *not* the training library; report only) | 250 | remix 60, featuring 32, sampling 14, cover 2, medley 1, missing metadata 1 |
| Verified, isolated expansion candidates (P113/P114 audits, 749) | 696 | sample(d/s) 31, share-alike text 15, cover 11, featuring 8, tribute 2 |
| Archive.org CC BY/CC0 netlabel hip-hop tracks (203) | 39 | sample 44, featuring 20, edit 13, Freesound 6, flip 4, remix 3; then missing creators |

Passing expansion works: electronic 539, ambient 82, rock 51, unlabeled 23,
metal 1; 63 authors.

The 39 passing hip-hop tracks are mostly experimental electronic, noise or
spoken word. Only about 5-8 tracks from about 5 artists are genuine hip hop,
two of whom already appear in the P113 test panel.

## Correction: the actual training library is already fully reviewed

The first draft of this record treated `current-approved.jsonl` as the training
data. It is not: the processed training library (catalog `ecf374663d1c...`,
300 works, used by every P-series run) overlaps it in only 52 works.

All 300 library works carry a manual `source_review` with status `accepted`:
- scope: commercial codec-restoration training and Apache-2.0 weight distribution;
- explicit grant, composition, recording and performers/samples bases;
- **no unresolved issues**;
- licences: CC BY 3.0/4.0 or CC0.

That manual review is stronger evidence than this keyword screen. The screen
is not applied to the reviewed library (it misreads review prose such as
"covers published compositions").

## Decisions

- **Training snapshots** use the TRAIN-split library works with an accepted
  review. A new work (the expansion pool) enters training only after a
  maintainer accepts a review record of the same form. The automated screen
  narrows the candidates; it does not accept them.
- **Pending drafts.** `training/scripts/draft_expansion_reviews.py` wrote
  pending drafts for the 671 screened expansion works (160 albums, 56 authors)
  to `data/source-discovery/expansion-review-draft-20261125.jsonl`, with an
  album summary beside it. Accepted rows go to
  `data/catalogs/expansion-review-acceptance.jsonl`. P115 refuses to register
  without them.
- **Hip hop** is not acquired from CC netlabels. Sample-based production makes
  undisclosed third-party material likely, and too few genuine tracks pass.
  The coverage gap is recorded. Closing it needs a source with explicit
  training permission.
- **Reserved evaluation panels** are scored internally and not distributed.
  Screen results: P112 10/16, P113 22/24, P114 13/16 pass. The P115 panel was
  reserved from screen-passing works only.

## Evidence

Candidate metadata is in `data/source-discovery/metadata-hiphop-20261124/`.
Screened hip-hop tracks are in
`data/source-discovery/hiphop-screened-20261125.json`. The screen runs again
deterministically from the catalogs and publisher metadata listed above.

## Official-source review of the expansion pool (2026-11-25)

The maintainer asked that expansion licences be checked to the library
standard. That standard requires the creator's own official page to grant the
licence for the exact work.

`training/scripts/review_expansion_evidence.py` looked up each of the 160
screened albums in the creator's official Bandcamp catalog (or a label Bandcamp
listed in the archive.org label collection) and required:
- a title match of at least 0.85;
- at least 50% track-title overlap;
- an explicit CC BY or CC0 licence on the official album page;
- no AI/usage terms or third-party credits on the album page or the artist
  profile.

Every fetched page is saved with its SHA-256 under
`data/source-discovery/expansion-evidence-20261125/`. The first,
slug-only matching pass is kept as `...-pass1`.

| Outcome | Albums | Works |
| --- | ---: | ---: |
| Verified against the official source | 21 | 107 (Dafuqex 56, Humworks 31, epi-chord 10, Artificial.Music 10) |
| Contradicted (official page: all rights reserved) | 1 | 8 (Daniel Birch & Ben Pegley) |
| Unverified (no official source, or album no longer in the official catalog) | 138 | 556 |

Most unverified works are label releases distributed only through archive.org
(for example Silent Running Scandal, Crna Zemlja), or albums since removed from
the artist's catalog (Illusory's official page now returns 404). The uploader's
identity cannot be independently confirmed, so they are excluded.

The 107 verified works are written as **proposed** records in
`data/source-discovery/expansion-acceptance-proposal-20261125.jsonl`.
Acceptance is the maintainer's decision. Accepted rows (status `accepted`) go to
`data/catalogs/expansion-review-acceptance.jsonl`, which P115 requires.
