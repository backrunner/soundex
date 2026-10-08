# Native lossless music catalog

[`native_lossless.jsonl`](native_lossless.jsonl) currently indexes **360 distinct
quality-audited music tracks**, approximately **24.0 hours / 10.0 GB**,
as of 2026-10-08. Expansion toward approximately **1,000** distinct tracks is in
progress. This is a verified snapshot, not a claim that the 1,000-track collection
has already completed. The repository stores the index, not the audio.

The initial 743-track acquisition selection was superseded: 430 recordings by
Rrrrrose/Loyalty Freak Music aliases, including Monplaisir and Komiku, were
excluded after finding the publisher's current express AI-training prohibition.
A further 41 of the remaining old masters have unresolved signal-quality flags
and are held for review. See [the source review](../../legal/TRAINING_DATA.md).
No weights were trained by the stopped 743-track run.

Every indexed original passed complete-file checksums, full mono/stereo decode,
finite-frame checks and the signal audit. Publisher MD5/size receipts are verified
where supplied. The index retains original rate/channels, decoded-PCM SHA-256,
RMS, DC, full-scale samples/runs, silence, sampled spectral metrics and individual
source/credit/license information. These checks describe the available master;
they cannot prove every stage of its previous production avoided lossy codecs.
Known lossy sources, including FMA MP3 packages, are excluded. Narrow bandwidth
alone is not a rejection criterion.

Only unique music works at least 30 seconds long at >=44.1 kHz count here. Exact
decoded copies, aliases, alternate mixes, crops and codec/channel variants do not
increase the count. Slakh and VCTK speech are separate supplements. This selection
policy does not impose a genre, license or duration whitelist on the generic importer.

| Primary publisher-derived style | Tracks |
| --- | ---: |
| ambient | 44 |
| blues | 14 |
| country | 6 |
| electronic | 81 |
| folk | 1 |
| jazz | 39 |
| metal | 2 |
| pop | 93 |
| rock | 69 |
| unlabeled | 11 |

Country and several other styles remain scarce. Soundtrack use and instrument
names alone do not establish a genre; uncertain styles remain unlabeled. This
snapshot does not establish comprehensive artist or genre coverage.

## Download and preprocess

From `training/`:

```bash
python data/download_library.py --catalog source_catalogs/native_lossless.jsonl \
  --output-dir ../data
python scripts/audit_library.py --catalog ../data/catalog.jsonl \
  --output-dir ../data/reports/music-quality
python data/preprocess_library.py --catalog ../data/reports/music-quality/accepted.jsonl \
  --output-dir ../data/processed/music_library/processed \
  --config configs/diverse_lossless.yaml
```

The downloader verifies complete original bytes and decoded PCM before publishing
files, and reuses verified originals on rerun. Source-quality flags are held for
review, not declared recording defects: intentional distortion may warrant a
separate review. Prepare all enabled corpora with the same immutable recipe.
Derive speech's sampling probability from the smallest actual retained training
music genre after pair generation; see [datasets](../../docs/datasets.md) and
[workspace/run tooling](../../docs/workspace.md).

## Licenses and provenance

Recordings retain their individual publisher grants; Apache-2.0 does not relicense
third-party audio. Preserve credits, source links, licenses and change descriptions.
The index preserves source declarations and verified acquired versions. Official
trained-weight publication uses the separate [artifact review](../../legal/TRAINING_DATA.md).
