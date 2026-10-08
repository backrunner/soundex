# Audio sources for application-ready weights

Reviewed 2026-10-08 against publisher pages. These are source candidates, not a
blanket certification of every recording or a claim that data has been downloaded.
SoundEx needs only clean audio: source → codec compression/decode → aligned
clean/degraded pair. Source separation labels and multitrack stems are unnecessary.

## Source survey

| Source | Publisher grant / acquisition | SoundEx use |
|--------|-------------------------------|-------------|
| [Open Goldberg Variations](https://kimikoishizaka.bandcamp.com/album/j-s-bach-open-goldberg-variations-bwv-988-piano) | Artist states all tracks are CC0; offers lossless/24-bit 96 kHz downloads | Strong small piano seed/validation source; insufficient genre/instrument coverage on its own |
| [Slakh2100-redux](https://www.slakh.com/) / [official archive](https://zenodo.org/records/4599666) | Publisher states CC BY 4.0; full archive about 104 GB, 145 hours of synthesized mixes | Existing mix-only preprocessor; synthetic diversity supplement, not a substitute for stereo real recordings |
| [MusicNet](https://zenodo.org/records/5120004) | 330 classical recordings, PCM WAV archive about 11.1 GB; publisher describes CC and public-domain performances and gives per-recording provenance | Promising real-music candidate; review original recording grants individually when preparing a release; do not assume one label grants all recordings |
| [Musopen](https://musopen.org/music/) / [terms](https://musopen.org/tos/) | Per-recording public-domain/CC labels; not a single permissive license for the whole site | Prefer lossless recordings with clear reusable grants; retain exact source and recording credits |
| [FMA](https://github.com/mdeff/fma) | Audio follows the artist's chosen license; metadata CC BY 4.0; available audio packages are MP3 | Review per-track rights for publication; already-compressed audio is not a preferred high-band clean target; seek original lossless masters |
| [FSD50K](https://zenodo.org/records/4060432) | Clip licenses include CC0, BY, BY-NC and Sampling+; publisher requests contact for commercial use | Possible instrument/transient supplement after appropriate source review/contact; not approved as a whole or enabled by default |

MUSDB18-HQ is removed from downloadable/trainable source selection. Its
[publisher](https://sigsep.github.io/datasets/musdb.html) limits use to academic
purposes. Legacy MedleyDB utilities are not part of the default training route.

## Quality and split requirements

Prefer original lossless PCM WAV/FLAC at >=44.1 kHz. Sample rate and file extension
alone do not prove effective high-band content: inspect bandwidth, clipping,
silence, noise, prior lossy coding and resampling history. Retain prior lossy/low-rate material where useful, but do not label it as a full-band
clean master. For music restoration, prioritize diverse stereo
recordings, genres, instruments and transients; keep synthetic/single-piano data
as a supplement and report coverage honestly.

Group alternate versions and related recordings before splitting. Do not put codec
variants or crops of the same recording in different splits. Keep official test
partitions when importing a dataset; review split groups when converting it to a
catalog. A single work/album cannot establish cross-work generalization.

## Audio directory or optional catalog

Training has no dataset/license whitelist or mandatory rights form. Point the
importer at a folder of WAV/FLAC/AIFF/OGG/MP3 audio. It scans recursively, records
original format/rate/channel information and hashes, and deduplicates byte-identical
copies. Prefer lossless sources for clean restoration targets. Low-rate sources
are accepted but cannot demonstrate full-band restoration.

```bash
cd training
python data/preprocess_library.py --data-root /data/audio-library \
  --output-dir /data/music-library/processed --config configs/music_library.yaml
MUSIC_LIBRARY_PATH=/data/music-library python train.py --config configs/music_library.yaml
```

Keep raw audio and processed output in separate directories. For source credits,
related recordings or publisher split adaptation, optionally supply a JSONL
catalog instead of `--data-root`. Only `id` and `path` are required:

```json
{"id":"recording-001","path":"audio/recording-001.flac","author":"performer or rights holder","source_url":"https://publisher.example/recording-001","license":"actual declared license or permission","split_group":"related-work-or-recording-group"}
```

Paths are relative to the catalog. `license`, `license_url`, `author`, `source_url`,
`split_group`, `audio_sha256`, `evidence_path` and `evidence_sha256` are optional.
Supplied hashes must match their files. Omitted split groups default to the audio
checksum; group alternate versions explicitly to avoid leakage. No missing license
is inferred from a folder name or site. Run with `--catalog /data/catalog.jsonl`.

The importer generates aligned MP3 CBR/VBR, AAC-LC and Vorbis pairs at 44.1/48 kHz.
Rows bind source metadata and catalog checksum to the source checksum and recipe.
`source-credits.json` provides a starting point for release credits.

The library profile's validation quota is explicit (256 rows). Adjust it to the
actual source coverage; use sufficient recordings for each split. To mix Slakh
and audio libraries, set both paths, positive source ratios and exact per-source
quotas. Bound the epoch budget so one large corpus does not dominate. Add stereo,
mono, diverse instruments, transients, voices, ambience and dynamics; report the
coverage rather than inferring generalization from one album or genre.

For publication, complete the [source review](../legal/TRAINING_DATA.md),
[model card](../models/MODEL_CARD.template.md) and artifact-bound maintainer review.
That review concerns the actual grants and published artifact; it is not a training
ingestion gate. No new trained weights or large audio archives have been downloaded
or distributed by this survey.
