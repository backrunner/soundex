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
| Existing approved catalog (`current-approved.jsonl`, 360) | 250 | remix 60, featuring 32, sampling 14, cover 2, medley 1, missing metadata 1 |
| Verified, isolated expansion candidates (P113/P114 audits, 749) | 696 | sample(d/s) 31, share-alike text 15, cover 11, featuring 8, tribute 2 |
| Archive.org CC BY/CC0 netlabel hip-hop tracks (203) | 39 | sample 44, featuring 20, edit 13, Freesound 6, flip 4, remix 3; then missing creators |

Passing expansion works: electronic 539, ambient 82, rock 51, unlabeled 23,
metal 1; 63 authors.

The 39 passing hip-hop tracks are mostly experimental electronic, noise or
spoken word. Only about 5-8 tracks from about 5 artists are genuine hip hop,
two of whom already appear in the P113 test panel.

## Decisions

- **New training snapshots** use only screened material: the 250 passing
  approved works (restricted to their training split) plus passing expansion
  works. `current-approved.jsonl` is not edited. The 110 flagged approved works
  are listed with reasons in the next study's registration. Earlier runs and
  weights that used them are historical; their release remains subject to
  maintainer review.
- **Hip hop** is not acquired from CC netlabels. Sample-based production makes
  undisclosed third-party material likely, and the passing pool is too small
  to close the gap. The hip-hop coverage gap is recorded. Closing it needs a
  source with explicit training permission, such as commissioned or owned
  recordings or a licence that permits model training.
- **Reserved evaluation panels** are scored internally and not distributed.
  Their screen results are reported (P112: 10/16, P113: 22/24, P114: 13/16
  pass). Panels are not altered retrospectively, but future reservations apply
  the screen first.

## Evidence

Candidate metadata is in `data/source-discovery/metadata-hiphop-20261124/`.
Screened hip-hop tracks are in
`data/source-discovery/hiphop-screened-20261125.json`. The screen runs again
deterministically from the catalogs and publisher metadata listed above.
