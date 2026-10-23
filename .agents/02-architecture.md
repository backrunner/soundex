# SoundEx architecture

## Boundaries

`soundex-dsp` owns FFT/windowing, spectral analysis, bandwidth estimation, gain,
phase/fusion, overlap-add and limiting. It must not depend on file codecs or the
training framework. `soundex-core` owns model contracts, ONNX inference, stream state,
processing configuration, worker/callback adaptation and error handling. `soundex-cli`
owns decoding, user options and file output.

Python training separates source preparation, model/loss definitions, evaluation and
registered execution. Frozen experiment source is independent of later main-branch
edits. Shared historical modules remain when current imports or reproducible controls
need them; old filenames alone do not justify deletion.

## Streaming data flow

```text
host callback -> preallocated input queue -> worker
worker: causal frontend -> model -> DSP fusion/synthesis -> output queue
host callback: aligned output or smooth fixed-timeline fallback
```

The callback does no inference, allocation or blocking. Each physical channel owns
its causal history; silence and bypass advance state. Reset, dropped-frame recovery
and model replacement clear history. Enhancement warmup returns aligned dry audio
without creating a hidden extra output delay. The worker's fixed buffer and compute
cost are reported separately.

Offline processing uses the same relevant model/DSP semantics, with explicit startup,
flush and tail behavior. It does not certify callback scheduling or queue recovery.

## Model and experiment contracts

Production models validate metadata, tensor shape, sample rates and supported
contracts before processing. Causal experimental models export explicit state and
input/output timelines. Experimental size exemptions are explicit and do not migrate
the production contract. Diagnostic fusion/condition options stay in diagnostic
constructors and tools, not ordinary product controls.

Native frontend observations and actual Rust synthesis are the deployment reference
for parity. A differentiable proxy must state its differences and pass the registered
correspondence checks. Synthesis equality does not establish perceptual restoration.

## Features and errors

`ort-backend` is the default inference feature; no-ORT configurations remain useful
for DSP and contract checks. `embedded-model` is an explicit feature, not a claim
that a qualified restoration model is bundled. See Cargo manifests for exact feature
and dependency definitions.

Library errors are typed; CLI errors add actionable context. Initialization/contract
failure happens before callback processing. Runtime failures retain a stable timeline
and smooth fallback rather than blocking the callback or injecting discontinuities.

See [library integration](../docs/library.md), [real-time standard](../docs/realtime-standard.md),
[parity](../docs/parity.md) and [model status](../models/README.md). The root
[AGENTS.md](../AGENTS.md) is authoritative over historical API sketches.
