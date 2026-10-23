# SoundEx dependencies and environments

Use the repository manifests and lock/constraint files as authoritative versions.
This page explains responsibilities rather than duplicating stale complete manifests.

## Rust

The workspace uses Rust edition 2021 and requires Rust 1.88 or later as recorded in
`Cargo.toml`. `ort` provides optional CPU ONNX inference; `ndarray` supplies tensor
views. `rustfft`/`realfft` support spectral processing, with DSP helpers in `dasp`.
`rtrb` supplies the core streaming queues. On macOS, `qos-threads` and the explicitly
configured `audio_thread_priority` path support scheduling diagnostics outside the
callback. Scheduling policy changes require actual timing evidence.

CLI decoding uses `symphonia`; WAV output uses `hound`; `rubato` supports resampling.
`clap` handles options and `indicatif` progress. Libraries use `thiserror`; the CLI
uses `anyhow`. Logging uses `log`/`env_logger`. Tests and benchmarks use Criterion,
approximate comparisons and deterministic random stimuli as appropriate. Serialization
and evidence hashing use `serde`, `serde_json` and `sha2` where declared.

The current ORT manifest requests 2.0.0-rc.12 with explicit features; check
`Cargo.lock` for resolved versions. Default and no-ORT builds have separate coverage.
Do not introduce ONNX-runtime scheduling or global system changes through the audio
callback.

## Python and codecs

The recorded training stack is Python 3.12.x with PyTorch 2.12.1. See
[environment setup](../training/scripts/environment.md), `training/requirements.txt`
and `training/constraints-cpu-py312.txt` for installation/constraints. Install the
intended official PyTorch CPU/CUDA wheel first; do not assume plain PyPI resolves the
correct GPU runtime. Torchaudio is optional and not used by the current training code.

Current dependencies include NumPy, soundfile, PyYAML, TensorBoard, tqdm, ONNX,
ONNX Runtime, onnxscript, Ruff and pytest. FFmpeg provides actual codec synthesis;
codec binaries and their versions belong in experiment evidence. A process-local
fallback FFmpeg does not repair or certify the system installation.

P36/P38 executed on single-thread CPU. MPS availability or an unrelated Metal service
failure must not be confused with the actual registered device. CPU success does not
validate MPS/CUDA. Before migrating hardware, verify the environment and preserve
the registered numerical, data and source contracts.

## Data, containers and minimum resources

Mount or configure external data paths explicitly. Keep originals, prepared pairs,
registrations and checkpoints out of container images and Git. No generic Docker or
GPU-memory estimate can certify a particular experiment: measure its actual RAM,
VRAM, disk and runtime behavior. Keep source/provenance documentation with the run.

Do not install research-model code or weights merely because a paper was cited;
check their actual license and provenance first. The current original implementation
and declared dependency notices are documented in [THIRD_PARTY.md](../THIRD_PARTY.md).
