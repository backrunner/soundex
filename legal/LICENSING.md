# SoundEx licensing

## Scope

| Material | License and scope |
| --- | --- |
| Source, documentation, demos and synthetic test models | [Apache-2.0](../LICENSE) |
| Official weights released with a completed model card | [Apache-2.0](../LICENSE), as described in the [weight license](../MODEL_WEIGHTS_LICENSE.md) |
| Local experiments and external checkpoints | Their actual provenance and permissions; not automatically official Apache weights |
| Third-party audio and dependencies | Their upstream terms; audio is not distributed with this repository. See [third-party notices](../THIRD_PARTY.md). |

## Use in other projects

Official Apache-2.0 weights may be integrated into other open-source projects,
distributed with applications, modified, fine-tuned and used commercially, subject
to the standard license-copy, notice-preservation and modification-notice requirements.
There is no additional academic-use restriction. Combined distributions must also
satisfy the compatibility requirements of their other licenses.

## Training access and the release process

MUSDB18-HQ has been removed from the download, preprocessing and training entry
points. Training has no dataset/license allowlist and does not require a rights
field on every imported item. Audio directories may be imported directly to improve
coverage of genres, instruments, transients, spatial presentation and recording
conditions. Source, attribution and license metadata may be retained without
blocking training startup.

Before formally releasing weights, maintainers review actual dataset permissions
and initialization/resume provenance for application integration, redistribution
and commercial use. Existing MUSDB experiments remain unpublished research records;
editing this documentation does not turn them into official weights. Permissive
training ingestion does not extend upstream rights. Unconfirmed or incompatible
permissions must be resolved before release.

The [training-data policy](TRAINING_DATA.md), [source survey](../docs/datasets.md)
and [release checklist](MODEL_RELEASE_CHECKLIST.md) explain this distinction.
Automated checks verify records against files; maintainers verify the actual rights.
The [weight provenance review](WEIGHT_LICENSE_REVIEW.md) records reviewed sources,
hashes and limitations. No restoration weights have yet completed formal release
qualification. The native regional training route excludes MUSDB and Slakh, uses
reviewed real music and VCTK, and starts from fresh initialization. Historical Slakh
rights gaps apply to experiments containing it, not as an extra check on new runs
that exclude it. Each released artifact still needs its own complete provenance.

## License boundaries

Unreleased experiments and third-party materials are outside the official weight
grant. Training-data approval and release checks are maintainer publishing
procedures, not additional downstream license terms. A project notice cannot extend
upstream rights. This policy does not determine that every trained model is a
copyright derivative of its training audio.

The full English [Apache License](../LICENSE) controls the license grant, warranty
disclaimer and limitation of liability. This overview is explanatory.
