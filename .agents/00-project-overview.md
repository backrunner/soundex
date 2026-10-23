# SoundEx project overview

## Goal and current status

Restore lossy compressed music by estimating missing/damaged high-frequency spectra
and combining the neural prediction with DSP. The reference is paired original
lossless audio. Spectral shape, magnitude, energy envelopes and real transients come
first; phase, waveform, preserved bands, stereo and continuity constrain fidelity.
Merely raising treble or reducing enhancement is not success.

The Rust library, CLI, streaming adapter and runnable demos exist. Bundled synthetic
identity graphs exercise inference; qualified restoration weights are not published.
P38 training is complete and unqualified. P39 provides gradient and timing diagnostics,
not new production weights. See [research](../docs/research.md) and the mandatory
[failure registry](../docs/failed-experiments.md) before designing another experiment.

## Technology and repository map

- Rust workspace: `crates/soundex-core`, `crates/soundex-dsp`, `crates/soundex-cli`.
- CPU ONNX Runtime through `ort` for deployment; PyTorch for training and export.
- `training/`: separate data, models, evaluation, configurations, scripts and tests.
- `models/`: published synthetic fixtures, model metadata and release status.
- `docs/` and `legal/`: maintained English engineering, research and provenance records.
- `data/`: ignored local originals, prepared pairs, registrations, frozen execution
  source, checkpoints, diagnostics and cleanup receipts. It is not disposable merely
  because it is ignored. See [workspace layout](../docs/workspace.md).

## Delivery boundaries

Source and documentation use Apache-2.0. Formal model releases require the documented
model card and artifact/provenance review; local experiments are not automatically
released weights. Original training audio is not committed. Submit source/attribution
documentation instead. Commit as BackRunner <dev@backrunner.top> in logical
Conventional Commits.

The root [AGENTS.md](../AGENTS.md) contains the binding engineering instructions.
These notes are maintained design guidance, not evidence that every proposed feature
or release gate has passed. Older planning sketches remain in Git history.
