# Native lossless music catalog

**Superseded selection:** the initial 743-track acquisition index below is being
replaced after full signal-quality review and the discovery of a current express
AI-training prohibition for the Rrrrrose/Loyalty Freak Music aliases. Do not use
those entries in the new official run. See [the source review](../../legal/TRAINING_DATA.md).
The expanded, audited replacement targets approximately 1,000 distinct tracks.

[`native_lossless.jsonl`](native_lossless.jsonl) indexes **743 distinct music
tracks**, approximately **39.4 hours / 13.5 GB**, acquired and verified on
2026-10-08. The repository contains the index, not the audio.

Each entry records an original WAV/FLAC URL, publisher page, recording credit,
publisher-declared license, complete-file SHA-256, decoded-PCM SHA-256, original
sample rate/channel count, and a related-work split group. These originals are
44.1, 48 or 96 kHz stereo. Mono Slakh and VCTK speech remain separate supplements
and do not contribute to the 743-track count. FMA MP3 packages are excluded.

The count excludes 25 short cues or duplicate works/decoded recordings. Counted
tracks are at least 30 seconds long; codec variants and channel roles do not add
tracks. Multiple project names belonging to Rrrrrose Azerty share an author/work
identity for grouping. This is a collection policy, not a new minimum-duration
requirement for the generic training importer.

| Primary publisher-derived style | Tracks |
| --- | ---: |
| Rock | 168 |
| Pop | 119 |
| Folk | 105 |
| Ambient | 90 |
| Electronic | 72 |
| Experimental | 39 |
| Jazz | 39 |
| Blues | 20 |
| Hip hop | 17 |
| Metal | 8 |
| Country | 6 |
| Unlabeled or mixed | 60 |

Several styles remain scarce, especially country. Soundtrack use and instrument
names alone do not establish a genre; unspecified styles remain unlabeled.
The total does not establish comprehensive genre or artist coverage. Related-work
hash partitions currently assign 608 tracks to training, 82 to validation and 53
to test; actual retained training counts are measured after pair generation.

## Download and preprocess

From `training/`:

```bash
python data/download_library.py \
  --catalog source_catalogs/native_lossless.jsonl \
  --output-dir ../data/native-lossless

python data/preprocess_library.py \
  --catalog ../data/native-lossless/catalog.jsonl \
  --output-dir ../data/native-lossless/processed \
  --config configs/diverse_lossless.yaml
```

The downloader verifies complete original bytes and decoded audio before
publishing each file. Verified existing files are reused on rerun. It writes a
local `catalog.jsonl` with relative paths for preprocessing. Configure the music,
Slakh and speech paths for the run and prepare all three with the same recipe.
Derive speech's sampling mass from the smallest actual training music genre
after preprocessing, as described in [datasets](../../docs/datasets.md).

## Licenses and provenance

Audio retains the publisher's individual CC0 or CC BY grant recorded in each
entry. SoundEx's Apache-2.0 license does not relicense this audio. Retain the
recording credits, source links, licenses and change descriptions when using or
redistributing recordings. The index preserves declarations and checksums;
lossless encoding does not by itself prove the entire prior mastering history.
Official trained-weight publication uses the separate
[artifact review](../../legal/TRAINING_DATA.md).
