# Plan 003: Build A Split-Safe Representative Training Dataset Protocol

> **Executor instructions**: Follow each step and verification gate. Do not run
> destructive preprocessing against an existing dataset tree. Stop on any listed
> condition. Update `plans/README.md` when complete.
>
> **Drift check (run first)**: run
> `rtk shasum -a 256 training/data/audio_prep.py training/data/dataset.py training/data/preprocess_musdb.py training/data/preprocess_slakh.py training/data/preprocess_medleydb.py training/configs/default.yaml`.
> Expected hashes are `98d776ede81fd4100bbacfa1b3ce99be120225ea87e973b56c5fb3e42ff7ef93`,
> `982f5ea64876683577d6c82e352edb8ef19093c148b1384e2b007ace47a4bc87`,
> `34c51d02cf0725e71bd392718cb245ba479f6365a9749a12202e21f53e251d63`,
> `a9fdf2beb86ccf26d82f9bb6f2345fdf19de407d8b418feb4ecb4b60b60270a5`,
> `9587ca94f118ca4291832fbfe922b1618632279ea4761f6d306f9acababfee36`, and
> `5c9d9a19ac729d95708348223cfb873277c15ea5e6a6ed02c43fbf1491195360`.
> Git metadata was absent when this plan was written; stop on semantic drift.

## Status

- **Priority**: P1
- **Effort**: L
- **Risk**: MED
- **Depends on**: none
- **Category**: tests
- **Planned at**: workspace snapshot without Git metadata, 2026-07-28

## Why This Matters

The current preprocessors preserve track-level splits in their first run, but
they write directly into reusable `train/test` trees. Re-running with a different
ratio or fewer segments leaves stale files and can put one track in both splits.
The training loop also uses each corpus's test data as validation, leaving no
untouched final test set. Finally, only mono 44.1 kHz CBR MP3 is generated even
though deployment claims MP3, AAC, OGG/Vorbis, stereo, and 48 kHz support.

## Current State

- `training/train.py:116-117` maps `processed/test` to validation.
- `preprocess_slakh.py:40-47` maps both official validation and test to one
  `test` directory.
- `preprocess_medleydb.py:80-85` uses a configurable two-way split.
- `audio_prep.py:139-151` writes directly into an existing output directory and
  never reconciles stale files.
- `dataset.py:39` recursively loads every matching file left in that tree.
- `audio_prep.py:27-29` downmixes stereo before degradation, `:58-93` supports
  only libmp3lame, and `:16-18` fixes codec rates and 44.1 kHz in constants.
- `audio_prep.py:92-93` aligns clean/degraded only by truncating length. The
  comment at `:143` claims an encoder-delay guard, but no offset is estimated.
  Phase-supervised training requires sample alignment.
- `training/configs/default.yaml:24-27` contains sample rate and segment duration,
  while `:52-53` contains bitrate/split keys that do not drive preprocessing.
- The repository requires full-track mixtures only; stems must never enter the
  dataset. Preserve that legal and product constraint.

## Commands You Will Need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Python tests | From `training/`: `rtk python3 -m pytest -q tests/test_data_protocol.py` | all pass |
| Full Python | From `training/`: `rtk python3 -m pytest -q` | all pass |
| Lint | `rtk ruff check training` | exit 0 |
| Format | `rtk ruff format --check training` | exit 0 |
| Environment | From `training/`: `rtk python3 scripts/check_env.py` | required tools reported present |

## Scope

**In scope**:

- `training/data/audio_prep.py`
- `training/data/dataset.py`
- `training/data/preprocess_musdb.py`
- `training/data/preprocess_slakh.py`
- `training/data/preprocess_medleydb.py`
- `training/configs/default.yaml`
- `training/configs/titan_xp.yaml`
- A new validated data-config/manifest module under `training/data/`
- `training/tests/test_data_protocol.py` and small generated fixtures
- `training/requirements.txt`
- Relevant training/data documentation and model-card fields

**Out of scope**:

- Downloading or committing actual third-party audio.
- Using stems or changing dataset license tiers.
- Training the production checkpoint.
- Defining model quality metrics, handled by plan 005.
- Deleting an operator's existing processed tree.

## Git Workflow

- Preferred branch: `train/versioned-data-protocol`.
- Suggested commits:
  `train(data): add versioned split manifests`, then
  `train(data): cover deployment codec matrix`.
- Never delete or rewrite the operator's existing dataset in place.

## Steps

### Step 1: Define One Authoritative Data Recipe

Move preprocessing settings into a validated schema consumed by every
preprocessor. At minimum include:

- schema/recipe version and random seed;
- supported sample rates, initially 44,100 and 48,000 Hz;
- segment duration/count policy;
- codec, encoder implementation, bitrate/quality mode, CBR/VBR, and weight;
- channel sampling policy;
- split policy per corpus; and
- encoder alignment tolerance.

Remove or reject dead keys. The resolved recipe, not comments or CLI defaults,
must be serialized into every processed dataset manifest and later checkpoint.
CLI flags may override it only if the resolved values are recorded.

**Verify**: a unit test loads both YAML profiles, asserts every data key is
consumed, and rejects an unknown key with its full path in the error.

### Step 2: Preserve Three Disjoint Track-Level Splits

Use `train`, `validation`, and `test` directories and manifest roles.

- MUSDB18-HQ: reserve official `test` for final test; split official `train`
  deterministically by track ID into train/validation.
- Slakh2100: preserve official train/validation/test exactly. For flat layouts,
  use a stable three-way hash of canonical track ID.
- MedleyDB: use a stable three-way track hash. Duplicate V1/V2 identities must
  always land in the same split.

Generate a canonical `track_id` independent of bitrate, codec, segment, or file
path. Assert that the three track-ID sets are pairwise disjoint before publishing
the processed dataset.

**Verify**: synthetic manifests containing multiple codecs/bitrates per track
show zero track-ID intersection across splits and deterministic assignment across
two runs.

### Step 3: Make Generation Atomic And Versioned

Write into a fresh staging directory named from the recipe hash. Emit all audio
and a manifest there, validate it, then atomically rename/publish the directory.
Never merge into an existing version. If the target exists and its manifest hash
matches, allow an explicit skip; otherwise fail with instructions to choose a new
output version. Do not automatically remove old datasets.

Each manifest row must include source corpus/version, canonical track ID, split,
source mix path or stable source ID, source checksum when available, sample rate,
channel role, codec/encoder/options, clean/degraded paths, start sample, length,
alignment offset, measured cutoff, and pair checksums.

**Verify**: rerunning with a changed split ratio or smaller segment count creates
a distinct version and cannot leave stale rows/files in the new version.

### Step 4: Verify Codec Delay And Pair Alignment

After encode/decode, estimate delay using a robust low-band cross-correlation or
known encoder timing metadata, apply the offset, and crop clean/degraded to the
same aligned interval. Reject pairs whose residual alignment exceeds a configured
tolerance. Test impulses, broadband noise, and tones; do not rely only on equal
length. Record the applied offset in the manifest.

Also verify finite samples, non-silent clean content, peak/rms bounds, exact sample
rate, and pair completeness before writing.

**Verify**: FFmpeg integration tests for every enabled codec recover an injected
impulse within one sample after alignment. Unit tests run with a deterministic
fake encoder when FFmpeg is unavailable.

### Step 5: Match The Deployment Degradation Matrix

Add a weighted, bounded matrix covering at least:

- libmp3lame CBR and VBR across representative low/mid/high quality;
- AAC-LC using an encoder available in the supported Docker image;
- Vorbis for the advertised OGG path;
- 44.1 and 48 kHz; and
- stereo coding artifacts.

Preserve stereo through encode/decode. Since the current model processes one
channel at a time, create training examples from left, right, mid, and side using
a documented weighted policy instead of downmixing every source to mono. Keep a
held-out codec setting or encoder implementation for generalization evaluation.
Bound each stratum so one synthetic degradation cannot dominate real data.

**Verify**: manifest summary tests assert configured proportions within sampling
tolerance and prove every promised deployment stratum exists in validation/test.

### Step 6: Load From The Manifest, Not Recursive Globs

Refactor `SoundExDataset` to consume a validated manifest and split name. Verify
file checksums/pair metadata in an optional audit mode. Return degradation metadata
including codec, bitrate/quality, sample rate, track ID, and cutoff/missing-band
mask inputs needed by plan 005. Do not infer identity from filenames.

Validation and test sample positions must be fixed in the manifest. Training may
random-crop within a track segment only when the chosen crop and random seed are
reproducible.

**Verify**: a stray `*_degraded.wav` outside the manifest is ignored, a missing
listed pair fails, and the same manifest yields identical validation/test batches
across workers and runs.

## Test Plan

- Use generated short PCM fixtures; do not commit third-party audio.
- Test split stability/disjointness, duplicate identities, atomic publication,
  stale-file resistance, pair completeness, checksums, codec delay, channel
  policy, sample rates, and source/codec quotas.
- Test interrupted staging generation leaves no published partial dataset.
- Test full-track mix discovery continues excluding all stem/raw trees.
- Test manifest loading with DataLoader worker counts 0 and 2.

## Done Criteria

- [ ] Every published dataset has a versioned recipe and row manifest.
- [ ] Train/validation/test track IDs are pairwise disjoint for every corpus.
- [ ] Existing output trees are never merged, cleared, or silently reused.
- [ ] Equal length is not treated as proof of sample alignment.
- [ ] MP3, AAC-LC, Vorbis, 44.1/48 kHz, and stereo-derived examples are covered
      or an unsupported advertised case is removed from product claims.
- [ ] Validation/test positions and composition are deterministic.
- [ ] The loader consumes only manifest rows and returns degradation metadata.
- [ ] Python tests and Ruff checks pass.
- [ ] No third-party audio or stems are added to the repository.
- [ ] Only in-scope files and `plans/README.md` changed.

## STOP Conditions

- A codec encoder is not legally redistributable or unavailable in the supported
  container. Report it and remove that stratum from claims rather than silently
  substituting another encoder.
- An upstream corpus's official split/license terms conflict with this plan.
- A requested action would delete or mutate an existing operator dataset tree.
- Alignment cannot be established within tolerance for an enabled codec.
- Supporting 48 kHz requires a different model contract than plan 001/004. Stop
  and coordinate the artifact metadata before generating release data.

## Maintenance Notes

- Recipe and manifest schema changes require explicit version bumps and migration
  or regeneration, never best-effort parsing.
- Release model cards should be generated from manifest summaries, not manually
  transcribed paths or remembered settings.
- Keep held-out test identities inaccessible to per-epoch training logs.

