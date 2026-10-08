# Model weight licensing

Official SoundEx weights released with a completed model card are licensed under
**Apache License 2.0**, the same license as the source code. The full, authoritative
license text is [LICENSE](LICENSE). This document explains its scope; it adds no
extra conditions to Apache-2.0.

The grant permits use in other open-source projects, integration into applications,
redistribution with software, modification, fine-tuning and commercial use, subject
to Apache-2.0. Preserve the license and applicable attribution notices, and identify
modifications as required by that license. Linking a separately licensed model does
not by itself change the application's license; compatibility of a combined
redistribution still depends on the application's terms.

A weight release identifies its artifact SHA-256, `Apache-2.0` license, training
provenance and applicable credits in its model card and accompanying NOTICE.
Training audio and third-party weights retain their own terms and are not covered
by SoundEx's grant.

## Unreleased experiments and historical policy

Local experiments, externally obtained checkpoints and files without an official
release/model card are not granted a new weight license by this document. They
must not be presented as approved Apache-2.0 SoundEx weights.

The earlier custom SoundEx Model Weights License 1.0 and Tier A/B/C release policy
are retired for new official releases. No learned weights were publicly released
under that policy. This change does not relicense third-party materials or expand
any upstream grant. The existing MUSDB-trained candidate remains an unpublished
research experiment and is excluded from the application-ready release route.

Maintainer approval of training provenance and release evidence is a publishing
workflow, not an additional restriction on downstream Apache-2.0 users. See
[licensing overview](legal/LICENSING.md) and [training data policy](legal/TRAINING_DATA.md).
The [current factual audit](legal/WEIGHT_LICENSE_REVIEW.md) records unreleased
experiments and unresolved source rights; it is not a release approval.
