# Models directory

This directory holds **exported inference artifacts** (for example ONNX models) and their **Model Cards**.

## License reminder

| Artifact type | License |
|---------------|---------|
| Empty tree / this README / templates | Apache-2.0 with the rest of the Software |
| `*.onnx`, `*.pth`, checkpoints, etc. | **SoundEx Model Weights License 1.0** — see [`../MODEL_WEIGHTS_LICENSE.md`](../MODEL_WEIGHTS_LICENSE.md) |
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
