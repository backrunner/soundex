# Acquiring clean audio

SoundEx creates aligned compression-degradation pairs from full rendered mixes or
audio recordings. No stems are needed. See the [source survey](../../docs/datasets.md)
and [training source policy](../../legal/TRAINING_DATA.md).

## Legacy Slakh2100-redux (explicit opt-in)

Use the [official archive](https://zenodo.org/records/4599666) and verify its source
license. The archive is about 104 GB even when only mixes are extracted; allow
space for archive, lossless mixes and codec-generated pairs. Retained pair size
scales with recipe segments, codec/rate/channel strata and source duration.

```bash
cd training
python scripts/download_datasets.py --datasets slakh2100 \
  --data-root /data --cache-dir /scratch/soundex-cache --preprocess
```

There is no default dataset: `--datasets` is required for acquisition.
The current regional run excludes Slakh and uses real music plus VCTK speech.
MUSDB is no longer selectable. BabySlakh is
16 kHz and suitable only for pipeline smoke tests, not full-bandwidth targets.
MedleyDB and Slakh utilities remain for historical data inspection; they are not
enabled by the current training profiles.

## Real recordings

Obtain lossless originals whose recording/composition grants support application
use and public weight release. Follow the [directory / optional catalog instructions](../../docs/datasets.md).
FMA's MP3 archives and 16 kHz transcription repacks do not supply clean full-band
masters. Public-domain scores alone do not establish recording rights.

Use `configs/music_library.yaml`, set actual roots and validation quotas, then
run the generic importer. Keep source audio and private permission documents out
of Git; retain their hashes, evidence and required credits. Check original sample
rate and effective bandwidth before making pairs.

## Training hardware

For Python/MPS/CUDA and Titan Xp constraints, see [environment.md](environment.md).
A recipe-bound dataset must be regenerated after changing its recipe; immutable
versions are never merged or silently reused with different source catalog.
