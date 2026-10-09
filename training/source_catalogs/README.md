# Native lossless music sources

The public repository contains source documentation, training tools and configs.
Downloaded originals, processed pairs, acquisition plans, per-file audit catalogs
and training checkpoints remain local. Source declarations do not relicense audio
under the code's Apache-2.0 license.

## Current collection, 2026-10-09

The updated acquisition/review snapshot contains **1,329 signal-audited candidate
entries**, but only **222 distinct source-reviewed music works from 16 creators**
count toward the requested approximately 1,000-work collection. Candidate entries are not approved
song counts. China contributes 29 works, Japan 48, Korea 58, North America 49;
38 have unknown regional repertoire. Reviewed European and Chinese/Japanese/Korean
vocal coverage remain absent. Pop has 2 works and electronic 56; no full country
songs are currently counted. The collection is not ready for the regional run.

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
CC BY 4.0 reviews do not settle the complete corpus's separate Slakh release issue.

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
codec variants and channels do not increase song counts. Slakh and VCTK speech
remain separate supplements. Prepare every enabled corpus with the same immutable
recipe and derive speech sampling from the smallest actual music genre after
pair generation; see [workspace/run tooling](../../docs/workspace.md).

## Rights and publication

Preserve individual publisher grants, credits and edit descriptions locally;
source documentation stays public. The
[actual weight-license findings](../../legal/WEIGHT_LICENSE_REVIEW.md) remain
separate from signal quality: the Slakh MIDI/rendering review is unresolved,
and no learned restoration weights are approved for publication.
