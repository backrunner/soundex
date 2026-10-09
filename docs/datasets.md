# Audio sources for application-ready weights

Reviewed 2026-10-08 against publisher pages. These are source candidates, not a
blanket certification of every recording or a claim that data has been downloaded.
SoundEx needs only clean audio: source → codec compression/decode → aligned
clean/degraded pair. Source separation labels and multitrack stems are unnecessary.

## Source survey

| Source | Publisher grant / acquisition | SoundEx use |
|--------|-------------------------------|-------------|
| [Open Goldberg Variations](https://kimikoishizaka.bandcamp.com/album/j-s-bach-open-goldberg-variations-bwv-988-piano) | Artist states all tracks are CC0; offers lossless/24-bit 96 kHz downloads | Strong small piano seed/validation source; insufficient genre/instrument coverage on its own |
| [Slakh2100-redux](https://www.slakh.com/) / [official archive](https://zenodo.org/records/4599666) | Publisher states CC BY 4.0; underlying Lakh composition/transcription authority remains unresolved | Removed from the current regional run on 2026-10-09. Historical experiments retain an [incomplete commercial weight review](../legal/WEIGHT_LICENSE_REVIEW.md). |
| [MusicNet](https://zenodo.org/records/5120004) | 330 classical recordings, PCM WAV archive about 11.1 GB; publisher describes CC and public-domain performances and gives per-recording provenance | Promising real-music candidate; review original recording grants individually when preparing a release; do not assume one label grants all recordings |
| [Musopen](https://musopen.org/music/) / [terms](https://musopen.org/tos/) | Per-recording public-domain/CC labels; not a single permissive license for the whole site | Prefer lossless recordings with clear reusable grants; retain exact source and recording credits |
| [FMA](https://github.com/mdeff/fma) | Available audio packages are MP3 | Excluded from clean-target training; converting MP3 to WAV/FLAC does not supply a lossless original |
| [VCTK 0.92](https://datashare.ed.ac.uk/items/30e7453c-9ea8-48b4-8e18-f96d0dc62928/full) | University of Edinburgh provides 48 kHz PCM recordings in FLAC under CC BY 4.0, with original recording/text provenance | Speech supplement; credit Yamagishi, Veaux and MacDonald. Group by speaker and use one microphone version |
| [WaivOps POP-ROK](https://github.com/patchbanks/WaivOps-POP-ROK) | Publisher explicitly offers AI training use under CC BY 4.0; native 44.1 kHz stereo WAV drums | Rhythm/transient supplement, including pop/rock/country styles; drum loops do not establish full-song genre coverage |
| [FSD50K](https://zenodo.org/records/4060432) | Clip licenses include CC0, BY, BY-NC and Sampling+; publisher requests contact for commercial use | Possible instrument/transient supplement after appropriate source review/contact; not approved as a whole or enabled by default |

MUSDB18-HQ is removed from downloadable/trainable source selection. Its
[publisher](https://sigsep.github.io/datasets/musdb.html) limits use to academic
purposes. Legacy MedleyDB utilities are not part of the default training route.

## Quality and split requirements

Use original lossless PCM WAV/FLAC masters, preferably at >=44.1 kHz. Sample rate and file extension
alone do not prove effective high-band content: inspect bandwidth, clipping,
silence, noise, prior lossy coding and resampling history. MP3/AAC/Vorbis sources
are excluded from clean-target synthesis. A PCM container cannot establish that
the upstream recording never passed through lossy compression; retain publisher
master provenance separately. Low-rate lossless audio is technically accepted,
but is not full-band restoration evidence. For music restoration, prioritize diverse stereo
recordings, genres, instruments and transients; keep synthetic/single-piano data
as a supplement and report coverage honestly.

Group alternate versions and related recordings before splitting. Do not put codec
variants or crops of the same recording in different splits. Keep official test
partitions when importing a dataset; review split groups when converting it to a
catalog. A single work/album cannot establish cross-work generalization.

## Audio directory or optional catalog

The [native lossless source documentation](../training/source_catalogs/README.md)
provides publisher references, dated acquisition counts and local preparation
instructions for the ongoing 1,000-track expansion. Download plans and per-file
indexes stay local alongside audio; a fresh clone contains source documents,
not an acquired catalog. Counts exclude speech, Slakh, duplicate works and short
cues. Each original recording retains its publisher's license.

Training has no dataset/license whitelist or mandatory rights form. Point the
importer at a folder of lossless WAV/FLAC/AIFF/W64/CAF audio. It scans recursively, records
original format/rate/channel information and hashes, and deduplicates byte-identical
copies. It checks the actual container and PCM subtype, not just extensions.
Low-rate lossless sources
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
actual source coverage; use sufficient recordings for each split. For any explicitly selected supplemental source, set its path, positive source
ratios and exact per-source quotas. Bound the epoch budget so one large corpus does not dominate. Add stereo,
mono, diverse instruments, transients, voices, ambience and dynamics; report the
coverage rather than inferring generalization from one album or genre.

`configs/diverse_lossless.yaml` enables distinct real-music and speech paths; Slakh is disabled.
Prepare speech with `preprocess_library.py --corpus speech_library` and that
profile; catalog `split_group` should identify the speaker. Set all source ratios
and quotas for the acquired data. A positive configured source missing from either
training or validation raises an error instead of silently changing the mix.

With `data.sampling.recording_balance: true` (enabled in the diverse profile),
each recording receives equal mass within its source, preserving relative codec
and channel-role weights. Mono, stereo and silent-side stereo recordings therefore
do not gain or lose recording mass through role expansion. Recompute weights after
subsampling. For genre counts, count unique training recordings; do not count
codec variants, channels or crops as additional music. Supply a single primary
`genre` label in optional catalog metadata and retain the publisher's original
multi-label tags separately. Unknown genres remain `unlabeled`.

`data.sampling_weights.speech_mix_ratios` derives speech probability equal to the
smallest labeled music genre's final draw probability, including normalization
after adding speech. No synthetic corpus is added by default. An explicitly requested synthetic budget
is separate from music genre counts. Inspect the resulting report before freezing a run: a genre
with one recording is an acquisition gap, not adequate coverage. Validation adds
genre strata and reports speech separately through its corpus stratum.

After preprocessing the acquired music, print its actual counts and ratios:

```bash
cd training
python -m data.sampling_weights /data/music-library/processed/music_library-RECIPE/manifest.jsonl
```

Copy the reported `train_ratios` into the run configuration before starting
training. The profile's initial ratios are placeholders for this calculation.

Legacy profiles without `recording_balance` retain their original sampler weights
for checkpoint reproducibility. Changing the policy changes the saved data config;
exact resume rejects a different config. Start the new diverse run from random
initialization rather than resuming historical MUSDB weights.

For publication, complete the [source review](../legal/TRAINING_DATA.md),
[model card](../models/MODEL_CARD.template.md) and artifact-bound maintainer review.
That review concerns the actual grants and published artifact; it is not a training
ingestion gate. Acquired audio stays local and is not redistributed with the
repository. Each original retains its publisher's license; the code's Apache-2.0
license does not replace those grants.

The current [regional acquisition review](dataset-curation.md) distinguishes downloaded originals, source review and actual repertoire coverage. Free sources have priority; paid libraries require separate approval.
