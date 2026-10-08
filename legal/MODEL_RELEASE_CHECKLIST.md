# Official weight release checklist

Official releases use Apache-2.0 and support application integration, redistribution
and commercial use. This is a maintainer workflow, not extra end-user license terms.
Synthetic identity test graphs already use the software license and are not learned
restoration releases.

## 1. Rights and training provenance

- [ ] Publication grants for training/validation sources satisfy [TRAINING_DATA.md](TRAINING_DATA.md).
- [ ] Source and parent/teacher grants support application use, redistribution and commercial use; resolve missing or incompatible permissions before release.
- [ ] Record random initialization or every approved resume/parent checkpoint.
- [ ] Record config, software commit, audio/catalog/manifest hashes and split groups.
- [ ] Verify source grants, composition/recording rights, attribution and modifications.
- [ ] Resolve the [documented Slakh MIDI/rendering source gaps](WEIGHT_LICENSE_REVIEW.md)
      for the actual included tracks; a dataset license label alone is insufficient.
- [ ] Inspect silence and unrelated held-out input outputs for retained training works;
      account for any upstream protected material in the actual artifact.
- [ ] Complete [DATA_RIGHTS_REVIEW.template.json](DATA_RIGHTS_REVIEW.template.json)
      with a real reviewer/date and the exact artifact/checkpoint/manifest bindings.

## 2. Model and evidence

- [ ] Complete [MODEL_CARD.template.md](../models/MODEL_CARD.template.md); license is `Apache-2.0`.
- [ ] ONNX checker, metadata/tensor contract and artifact-bound PyTorch/ORT/Rust
      [parity budgets](../docs/parity.md) pass.
- [ ] Immutable validation membership selects the checkpoint; test tracks remain held out.
- [ ] Rust offline and both arbitrary-chunk evaluations include degraded baselines,
      per-stratum high/low-band, perceptual/stereo/gate/clipping metrics and 95% CIs.
- [ ] Canonical release gates, ViSQOLAudio and randomized, level-matched blinded listening pass.
- [ ] The exact artifact passes 30-minute mono/stereo performance and continuity
      evidence on named Apple Silicon and x86_64 reference CPUs.
- [ ] From `training/`, run
      `python -m evaluation.model_card ARTIFACT CARD REPORT APPLE_PERF_JSON X86_PERF_JSON`.
      Licensing/provenance and quality/runtime checks must all pass.

## 3. Package and publish

- [ ] Include weight artifact, full LICENSE, applicable NOTICE, model card and rights review.
- [ ] Record artifact SHA-256 and download links; host metadata says `apache-2.0`.
- [ ] Keep dataset audio, private grant documents and optimizer checkpoints out of Git.
- [ ] Publish only the approved artifact; acknowledge that the model synthesizes
      plausible content and cannot guarantee recovery of an original recording.
