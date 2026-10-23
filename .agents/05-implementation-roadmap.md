# SoundEx implementation and qualification roadmap

This is a status-based roadmap. Early week-by-week estimates and unchecked API
sketches have been retired; they are not completion evidence. The root
[AGENTS.md](../AGENTS.md) defines acceptance.

| Area | Implemented work | Remaining acceptance |
| --- | --- | --- |
| Engineering foundation | Rust workspace, Python training, CI/checks, Apache source license | Keep current checks and dependency/platform evidence accurate |
| DSP | Causal analysis/synthesis, detection, fusion, gain and limiting | Restoration tradeoffs, broader stereo/continuity and real-device validation |
| Inference/integration | ONNX contracts, CPU inference, worker adapter, CLI and demos | Strict continuous deadlines and artifact-specific deployment qualification |
| Training/research | Native-lossless route and completed registered comparisons through P38 | A model that meets the spectral, fidelity and generalization screen |
| Diagnosis | P39 shared-gradient and balanced-order timing measurements | Isolate encoder conditioning and CPU-time versus scheduling stalls |
| Release | Source/docs and synthetic inference fixtures | Qualified restoration weights, complete artifact rights/model card and release checks |

## Next experiment gate

Read the [failed-experiment registry](../docs/failed-experiments.md) and record why a
new hypothesis is not a repeat. Keep the small model until a specific mechanism is
verified. P39 does not justify dropping phase/complex loss: output gradients align
while shared-parameter gradients conflict. First attribute per-band/example and
encoder update behavior under the same frozen endpoints.

A subsequent isolated training control needs fixed initialization, source-time
coverage, both seeds, full matched budgets and common scoring after endpoint freeze.
Do not choose an intermediate checkpoint to rescue a failed comparison. Separate
raw-model recovery, actual DSP output, identity preservation and untouched input.

## Real-time gate

Instrument wall time and per-thread CPU time before assigning ORT-stage outliers to
inference work or scheduling. Retain balanced order, declared background load, every
failure, per-hop deadlines and paced lateness. Then validate actual callback/queue
fallback and recovery. Model-plus-DSP <5 ms does not remove the fixed 256-frame buffer
or the stricter 2.667 ms hop throughput requirement at 48 kHz.

## Release gate

Source availability and runnable demos are not restoration-weight qualification.
Independent recording/source coverage, listening, stereo/continuity, runtime tails
and the [model release checklist](../legal/MODEL_RELEASE_CHECKLIST.md) remain necessary.
Keep local training data, checkpoints and reports ignored; publish provenance and
formal reviewed artifacts only. Update the failure registry and research status with
every completed unsuccessful validation.
