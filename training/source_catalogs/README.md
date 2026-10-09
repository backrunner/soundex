# Native lossless music sources

The public repository contains source documentation, training tools and configs.
Downloaded originals, processed pairs, acquisition plans, per-file audit catalogs
and training checkpoints remain local. Source declarations do not relicense audio
under the code's Apache-2.0 license.

## Current collection, 2026-10-09

The updated snapshot has **1,407 signal-audited candidate entries** and
**300 distinct source-reviewed music works from 22 creators**. It meets the
first-run policy, with zero blocking gaps and 13 advisory coverage gaps.
Candidate entries are not approved song counts. China contributes 29 works,
Japan 48, Korea 58, Europe 32 and North America 95; 38 have unknown regional
repertoire. Chinese/Japanese/Korean vocal coverage and R&B remain absent.
Pop has 14 works, electronic 56, blues 11 and country 5; the new indie-pop and
blues/Americana works are fusion recordings, not broad genre-market coverage.
The original strict 1,000-work plan remains available for later expansion.
At 20:57 Singapore time on 2026-10-09, all preparation/file audits had passed:
86,086 music pairs plus 20,056 speech pairs. Fresh MPS training had completed
epoch 1 and saved checked weights; the 200-epoch run continues. See
[current training](../../docs/native-training-20261009.md) for split counts,
sampling and the first checkpoint hash. Local run progress is authoritative.

See [the source and coverage review](../../docs/dataset-curation.md) for publisher
links, selected grants, credits and limitations. The separate
[creator-original WAV expansion](../../docs/source-expansion-20261008.md) documents
21 original downloads, including exact source pages and original-file links;
20 distinct music works count after measured float-master gain repairs. Its
short country cue is retained separately. The
[Japanese source expansion](../../docs/japanese-sources-20261009.md) documents
another 42 native WAV/FLAC masters: nine standalone compositions count, while
33 loop/cue/variant files remain supplements. Three new works have first-party
Japanese creator repertoire evidence; six JRPG themes retain unknown region.
The [additional Japanese acquisition](../../docs/japanese-additional-sources-20261009.md)
adds 46 original FLAC masters from three creators, representing 44 distinct works.
Two DC-offset repairs retain their originals and edit receipts. Source-specific
CC BY 4.0 reviews do not approve an eventual checkpoint automatically. Slakh has
since been removed from the upcoming run; its historical experiments remain separate.

The [Slakh replacement acquisition](../../docs/slakh-replacement-sources-20261009.md)
retains 83 original FLAC recordings and adds 55 reviewed works from four creators.
The [free pilot completion batch](../../docs/pilot-300-sources-20261009.md) adds
another 23 works / 79.84 minutes from two creators, without email or subscription.
Selected music now totals 1212.39 minutes; with VCTK speech, 1236.24 source
minutes are available across future train/validation/test splits. Slakh is excluded.

The earlier 360-track signal-audited snapshot (2026-10-08, approximately 24 hours /
10 GB) is a historical acquisition result, not the current source-reviewed count.
Its local index and acquisition receipts are retained outside Git. The initial
743-track selection was superseded after excluding the publisher's current
express training objection and holding unresolved signal flags for review.
No weights were trained by that stopped 743-track run.

## Local acquisition and preparation

Download original WAV/FLAC attachments from the documented publishers. Preserve
first-party source pages, the chosen license, credits, original-file URLs and
publisher checksums where available. Save any optional acquisition catalog under
`data/catalogs/`; a fresh clone does not include a completed acquisition plan.

For a locally prepared source catalog, run from `training/`:

```bash
python data/download_library.py --catalog ../data/catalogs/native_lossless.jsonl \
  --output-dir ../data
python scripts/audit_library.py --catalog ../data/catalog.jsonl \
  --output-dir ../data/reports/music-quality
python data/preprocess_library.py --catalog ../data/reports/music-quality/accepted.jsonl \
  --output-dir ../data/processed/music_library/processed \
  --config configs/diverse_lossless.yaml
```

Catalog format and direct-folder import are documented in
[datasets](../../docs/datasets.md). Catalogs are optional for the generic importer;
there is no dataset/license whitelist or mandatory rights form. Keep raw and
processed paths separate. Full-file decoding, finite values, integrity, stereo/
mono behavior and signal measurements are checked. A lossless container and
spectrum cannot prove the upstream production never used lossy coding; retain
publisher master provenance. Known MP3/AAC/Vorbis exports, including FMA MP3
packages, are not clean targets.

Related versions share one work/split group. Duplicate PCM, alternate mixes,
codec variants and channels do not increase song counts. VCTK speech remains a separate supplement; Slakh is excluded from the upcoming
regional run. Prepare every enabled corpus with the same immutable
recipe and derive speech sampling from the smallest actual music genre after
pair generation; see [workspace/run tooling](../../docs/workspace.md).

## Rights and publication

Preserve individual publisher grants, credits and edit descriptions locally;
source documentation stays public. The
[actual weight-license findings](../../legal/WEIGHT_LICENSE_REVIEW.md) remain
separate from signal quality: historical Slakh experiments retain unresolved MIDI/rendering findings,
and no learned restoration weights are approved for publication.
