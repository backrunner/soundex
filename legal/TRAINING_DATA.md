# Training data and released weights

## Audio diversity

SoundEx learns from audio, not source-separation labels. Mixes, music, instruments,
voices, ambience and transient sounds can all contribute useful patterns. Expand
coverage across genres, dynamics, bandwidth, recording conditions and stereo
images. Clean targets for codec-restoration training use original lossless masters.
Lossy source libraries, including the FMA MP3 packages, are excluded. Transcoding
MP3/AAC/Vorbis to a PCM container does not create a lossless master. Review upstream
master provenance in addition to checking the actual local encoding.

The 2026-10-08 source review found an express AI-training prohibition in
[Loyalty Freak Music's current FAQ](https://loyaltyfreakmusic.com/faq/).
The new official training selection therefore excludes that creator's material,
including Monplaisir and Komiku aliases. This is a source-selection decision to
avoid disputed permissions; it does not conclude that historical CC0 grants were
revoked. Earlier local catalogs are historical acquisition receipts, not the
current approved training selection. No weights were trained by the superseded
743-track run before it was stopped.

MUSDB18 / MUSDB18-HQ have been removed from download/preprocessing/training
selection because their [publisher](https://sigsep.github.io/datasets/musdb.html)
limits use to academic purposes. Historical local experiments remain readable;
they are not new official Apache weight releases.

## Ingestion and technical checks

There is no training dataset or license whitelist, mandatory per-recording rights
form, or license-based resume guard. The generic importer accepts an audio folder
or an optional JSONL catalog with just `id` and `path`. Optional license, author,
source URL, permission evidence and related-recording groups are retained as
metadata. Missing information is left unknown, not replaced with an assumed grant.

Necessary technical checks remain: readable/nonempty mono/stereo audio, finite
samples, checksum integrity, codec delay alignment, recipe consistency and
separate train/validation/test groups. Exact file copies are deduplicated.
Optional supplied checksums must match their files. Actual target encoding must
be lossless PCM/FLAC; this is a technical quality check, not a license whitelist.
Original sample rate/channel
information is recorded; low-rate material is accepted. Shared preprocessing
creates MP3/AAC/Vorbis degradation at 44.1/48 kHz after assigning groups to splits.

Prefer known publisher partitions and group alternate versions before generating
crops or codec variants. The same recording must not appear in both training and
held-out evaluation. Upsampling does not restore missing original bandwidth.
See [sources and usage](../docs/datasets.md).

## Publication under Apache-2.0

Official weights must support use and redistribution in other open-source
applications, including commercial deployments, under Apache-2.0. Training
flexibility does not grant rights in third-party audio or parent weights.

Maintainers review actual source grants, initialization/resume/teacher provenance,
composition and recording rights, credits and intended weight distribution when
preparing a release. Unknown or incompatible rights must be resolved before
publishing that artifact; changing a license label or fine-tuning does not supply
missing permission. A public-domain composition does not establish rights in its
recording. CC grants cover only rights the licensor holds.

Useful candidates include CC0 recordings, attributable permissive recordings,
and owned or separately authorized audio. These are suggestions,
not a fixed training allowlist. Individual grants and acquired versions matter.
Slakh's dataset grant does not by itself resolve the underlying web-scraped Lakh
MIDI composition/transcription rights. Slakh was removed from the upcoming regional
training mix on 2026-10-09;
its weight-publication rights review remains incomplete. See the concrete
[existing-weight and source findings](WEIGHT_LICENSE_REVIEW.md).

Record source/manifest/checkpoint hashes and retain optional source metadata for
release preparation. A completed model card and maintainer rights review identify
the exact published artifact and applicable credits. Audio retains its upstream
terms and is not distributed with the model. The publication workflow adds no
extra conditions to the Apache-2.0 license granted to downstream users.

See [licensing](LICENSING.md), [model card](../models/MODEL_CARD.template.md) and
[release checklist](MODEL_RELEASE_CHECKLIST.md).

The requested regional acquisition uses an optional
[source-and-coverage review](../docs/dataset-curation.md) before selecting its new
official run. Free sources have priority; paid libraries need separate approval.
This profile does not change generic ingestion or supply missing publisher rights.
