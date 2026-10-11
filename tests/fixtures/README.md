# Synthetic ONNX fixtures

These graphs contain no learned parameters or dataset audio. They are generated
by [generate_identity.py](generate_identity.py) and licensed as SoundEx software
under [Apache-2.0](../../LICENSE). They are stored directly in Git (no LFS download).

| File | Purpose |
|------|---------|
| identity.onnx | Legacy 1024/512 identity contract |
| low-latency-identity.onnx | Default 256/128 identity contract; runnable demos |
| temporal-probe.onnx | Synthetic 9-frame causal contract and history/reset tests |
| nonfinite-output.onnx | Deliberately invalid output for rejection/fallback tests |

An identity graph copies features; it does not reconstruct missing high-band
content. Processing still exercises the surrounding DSP. The invalid-output
fixture must never be used for normal audio processing. The temporal probe mixes
10% oldest and 90% current features solely to reveal history/order mistakes; it is
not a learned restoration model or a listening demo.

Regenerate from the repository root with the training dependencies installed:

```bash
python tests/fixtures/generate_identity.py
```

Learned restoration weights have separate terms and release gates; see
[models/README.md](../../models/README.md).

`stateful-probe.onnx` is a schema-2.0 analytic leaky-memory graph used to expose
state order, physical-channel isolation, gap/reset/reload and allocation bugs.
Its synthetic `temporal_kind=gru` exercises the new interface, not a learned GRU.
It requires `HighBandGain::Model` and returns aligned dry audio during the first
64 observed hops. No training data or learned restoration weights are included.
