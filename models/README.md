# Models directory

This directory holds **exported inference artifacts** (for example ONNX models) and their **Model Cards**.

## License reminder

| Artifact type | License |
|---------------|---------|
| Empty tree / this README / templates | Apache-2.0 with the rest of the Software |
| Learned `*.onnx`, `*.pth`, checkpoints, etc. | **SoundEx Model Weights License 1.0** — see [`../MODEL_WEIGHTS_LICENSE.md`](../MODEL_WEIGHTS_LICENSE.md) |
| Dataset audio | **Not stored here** |

No pretrained weights are shipped until a release adds them **with** a filled Model Card.

## Adding a release

1. Follow [`../legal/MODEL_RELEASE_CHECKLIST.md`](../legal/MODEL_RELEASE_CHECKLIST.md).  
2. Copy [`MODEL_CARD.template.md`](MODEL_CARD.template.md) to e.g. `soundex-v1.model-card.md`.  
3. Declare the correct **Weight Tier** from the training data matrix.  
4. Record the exporter-printed SHA-256 and verify the complete `soundex.*` ONNX metadata contract.
5. Run the held-out Rust-stream evaluation and complete the blinded listening record.
6. Verify the artifact, passing JSON report, and filled card with
   `python -m evaluation.model_card ARTIFACT CARD REPORT` from `training/`.
7. Do not label weight files as Apache-2.0.

## Current availability

No qualified restoration weights are published in this repository. The bundled
[synthetic fixtures](../tests/fixtures/README.md) let you run the demos immediately
and use the software's Apache-2.0 license; they do not restore missing frequencies.

As of 2026-10-08, the local 806,276-parameter candidate has an approximately
3.16 MiB diagnostic ONNX graph and a 12.95 MiB training/resume checkpoint. These
sizes are small enough for distribution, but the latest audited export fails the
strict CPU-ORT silence parity gate (max `1.221e-4`, mean `2.079e-5`; both must be
below `1e-5`). Diagnostic exports are not release artifacts. The current MUSDB
training provenance also requires Tier C under the project's weight policy.

Future weight releases need a completed model card, qualified export and the
quality/runtime gates listed in the release checklist. Training data, optimizer
checkpoints and failed exports are excluded from Git.
