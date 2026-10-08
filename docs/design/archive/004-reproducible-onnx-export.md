# Plan 004: Make Checkpoint-To-ONNX-To-Rust Export Reproducible

> **Executor instructions**: Execute in order and run every verification. This
> plan creates a breaking artifact schema; do not preserve silent compatibility
> with unverifiable checkpoints. Update `docs/design/archive/README.md` when complete.
>
> **Drift check (run first)**: run
> `rtk shasum -a 256 training/train.py training/export_onnx.py training/requirements.txt crates/soundex-core/src/inference.rs crates/soundex-core/src/processor.rs crates/soundex-core/tests/processor_test.rs .github/workflows/ci.yml`.
> Expected hashes are `2c7f9d7437f8f81e8199fd1c107b796c24cf31f1eaa5fac19d04e74f95d37f8c`,
> `c233561a97aee3af1dd95898ea4b30d1ae52094d51548c62111d97a348bffef0`,
> `c4fbaba2bf5237b4444ffed36e561580c2f8f84e904859fa7b59f03686ac36ac`,
> `86aa8599583cf3da054ca51532e7845476b2564a3b8ea320d439dd57cc2bcf87`,
> `5d58c3447f4d971bd2da8070e836004c28d55de77ecfd93d5027405c3c4b2c96`,
> `95f633dde245393b3e2038d35efa97af7e3d69ba0318207c57f9d1b8581912ec`, and
> `8d2a3842beaad1074d013c5c704a15e16bb118da3657a1bb718745778b6b15a4`.
> Git metadata was absent when this plan was written; stop on semantic drift.

## Status

- **Priority**: P1
- **Effort**: L
- **Risk**: MED
- **Depends on**: `docs/design/archive/001-streaming-model-contract.md`
- **Category**: tests
- **Planned at**: workspace snapshot without Git metadata, 2026-07-28

## Why This Matters

Checkpoints currently omit their effective configuration, while export requires a
separate YAML and independently chosen frequency width. ONNX declares time and
frequency dynamic even though Rust only supports one time frame, and Rust skips
frequency validation when the axis is dynamic. Python ORT comparison is optional
and uses an `assert`, while CI never installs PyTorch/ONNX or loads an exported
generator in Rust. The conversion path exists, but it is not yet a dependable
runtime capability.

## Historical Pre-Plan State

- `training/train.py:429-438` saves weights and optimizer/scheduler state, but no
  schema version, resolved config, feature contract, data manifest, or provenance.
- `training/export_onnx.py:21-23` accepts a separate config and arbitrary
  `freq_bins`.
- `export_onnx.py:66-69` marks batch, frames, and frequency dynamic.
- `export_onnx.py:94-111` treats ORT as optional, checks one random input, and
  uses a Python `assert` that optimization mode may disable.
- `inference.rs:121-130` validates only rank-4 float32. `processor.rs:216-223`
  cannot validate a dynamic frequency axis.
- Rust tests load only `tests/fixtures/identity.onnx`, not the generator.
- CI's Python job uses Python 3.10, installs only Ruff, and compile-checks source.
- The current target artifact contract is dynamic batch `B in {1,2}` and
  fixed `[B,2,1,513]` for the default FFT. Only axes actually consumed by Rust
  may be dynamic.

## Commands You Will Need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Python tests | From `training/`: `rtk python3 -m pytest -q tests/test_export.py` | all pass |
| Export smoke | From `training/`: `rtk python3 -m pytest -q tests/test_export.py` | generated temporary checkpoint export passes |
| ONNX checker | Covered by `tests/test_export.py` on its temporary artifact | checker exits without error |
| Rust tests | `rtk cargo test --workspace --all-targets` | all pass |
| Rust lint | `rtk cargo clippy --workspace --all-targets -- -D warnings` | no warnings |

## Scope

**In scope**:

- `training/train.py`
- `training/export_onnx.py`
- A new checkpoint/artifact schema module under `training/`
- `training/tests/test_export.py`
- `training/requirements.txt` and a reproducible CPU constraints/lock file
- `crates/soundex-core/src/inference.rs`
- `crates/soundex-core/src/processor.rs`
- `crates/soundex-core/src/error.rs`
- A Rust model-contract/parity test utility under `crates/soundex-core/`
- `.github/workflows/ci.yml`
- Model artifact and release documentation

**Out of scope**:

- Training release weights.
- Quantization until FP32 parity is proven.
- Dynamic time/frequency axes.
- A second inference backend.
- Quality and latency thresholds, handled by plans 005 and 006.

## Git Workflow

- Preferred branch: `test/checkpoint-onnx-rust-contract`.
- Suggested commits:
  `train(model): version checkpoint artifact metadata`, then
  `test(core): verify exported generator parity`.
- Do not commit generated smoke checkpoints or large ONNX artifacts.

## Steps

### Step 1: Version The Checkpoint Schema

Create a typed checkpoint builder/loader. Save at minimum:

- checkpoint schema and model architecture versions;
- epoch/global step and generator/discriminator states;
- optimizer/scheduler/scaler states when applicable;
- fully resolved training/model/audio/data configuration;
- feature schema: dB formula/floor, phase units, Hann periodicity, FFT, hop,
  supported sample rates, tensor channel meanings, context size, causality;
- dataset recipe/manifest hashes and effective source counts;
- Python/PyTorch/FFmpeg versions and source Git SHA when available; and
- Python, NumPy, Torch CPU/CUDA, DataLoader sampler RNG states.

Loading must reject unknown major schema versions and old schema-0 checkpoints
unless an explicit inspection-only migration tool is used. Export must never ask
an external YAML to redefine model/audio semantics; an optional config may only
be compared and rejected on mismatch.

**Verify**: round-trip tests preserve every field and reject a deliberately
mismatched FFT, sample rate, context, and architecture version.

### Step 2: Export The Exact Runtime Contract

Export FP32 ONNX with dynamic batch only and fixed channels, time, and frequency:
`[batch, 2, 1, 513]`. Write ONNX custom metadata for schema/model versions,
sample-rate set, FFT/hop/window, feature scale/floor, channel semantics,
stateless/causal flag, source checkpoint hash, and data/config hashes.

Keep one self-contained ONNX file and opset 17 unless a tested runtime matrix
requires another version. Replace the one-million parameter lower bound with a
two-million maximum and retain the `<8 MiB` FP32 artifact gate.

**Verify**: `onnx.checker`, metadata assertions, exact static shape assertions,
artifact size, and SHA-256 generation all pass.

### Step 3: Make Python ORT Validation Mandatory

Require ONNX Runtime; an ImportError is a failed export. Compare PyTorch and ORT
on deterministic inputs covering batch 1 and 2, silence, near-full-scale sine,
random finite spectra, dB floor values, and phase near `-pi`/`pi`. Use explicit
exceptions, not `assert`. Check shape, finiteness, max absolute error `<1e-5`, and
mean error. A partial output file must be removed or left in a clearly named
failed staging location, never published as success.

**Verify**: test an injected output perturbation and prove export exits nonzero
without publishing the final artifact.

### Step 4: Validate The Complete Contract In Rust

At session load, validate:

- one input and one output with expected names or metadata-defined names;
- float32 rank 4;
- channel dimension 2, time dimension 1, frequency dimension 513;
- batch dynamic or fixed to supported values;
- output shape contract; and
- every required SoundEx metadata field against `SoundExConfig`.

Reject unsupported sample rates, FFT/hop/window/features before the first audio
frame. Use the existing typed error pattern and activate the existing sample-rate
mismatch error or replace it with a richer contract error.

**Verify**: Rust tests reject models with wrong channel/time/frequency metadata
and accept the generated production-architecture smoke model.

### Step 5: Add Cross-Language Golden Parity

Generate a deterministic production-architecture checkpoint in a temporary CI
directory, export it, and write fixed input plus PyTorch/ORT golden output in a
simple binary or NumPy format. Invoke a Rust parity utility that runs the ONNX and
compares every element within `1e-5`. Also run one real STFT feature tensor through
the utility to catch feature ordering/layout errors.

Do not check the 5 MiB smoke artifact into Git. Generate it in CI and pass its
path to Rust. Keep the tiny identity fixture for separate pipeline tests.

**Verify**: CI fails when tensor channel order is intentionally swapped, then
passes after reverting the injection.

### Step 6: Pin And Test The Export Environment In CI

Add a Python 3.12 CPU job with tested exact versions/constraints for PyTorch,
ONNX, ONNX Runtime, ONNX Script, NumPy, and pytest. Run Python unit tests, smoke
checkpoint export, checker, mandatory ORT parity, and Rust parity. Keep the
CUDA Docker environment separate from the CPU export gate, but use compatible
versions.

**Verify**: a clean CI-equivalent environment completes all commands without
network access after dependency installation and publishes no model artifact.

## Test Plan

- Test checkpoint round-trip, old/unknown schema, config mismatch, metadata
  completeness, fixed dimensions, dynamic batch 1/2, numerical parity, failure
  cleanup, Rust metadata rejection, and real STFT tensor ordering.
- Use deterministic seeds and tolerances; never compare only one random input.
- Run tests with Python optimization enabled once to prove no safety check relies
  on `assert`.

## Done Criteria

- [ ] A checkpoint alone contains everything needed to reproduce its ONNX
      feature/model contract.
- [ ] Export has no independent semantic `freq_bins` or config override.
- [ ] ONNX has dynamic batch only, required metadata, opset 17, FP32 size <8 MiB,
      and parameter count <2 million.
- [ ] Missing ORT or numerical mismatch makes export fail explicitly.
- [ ] Python, ONNX Runtime, and Rust outputs agree within `1e-5` on all fixtures.
- [ ] Rust rejects every mismatched model/config before processing audio.
- [ ] Python 3.12 CI runs the complete export/parity path.
- [ ] Rust fmt, clippy, and all-target tests pass.
- [ ] Only in-scope files and `docs/design/archive/README.md` changed.

## STOP Conditions

- Plan 001 has not fixed the final tensor contract.
- ONNX Runtime or `ort` cannot expose required custom metadata. Stop and propose
  a cryptographically bound sidecar schema instead of silently omitting fields.
- The production generator requires an unsupported/custom ONNX operator.
- FP32 parity exceeds `1e-5` after deterministic, equivalent preprocessing.
- Supporting dynamic frequency/time is requested without a matching Rust state
  and test contract.

## Maintenance Notes

- Increment artifact schema whenever tensor semantics, framing, or metadata
  requirements change.
- Keep export and Rust load rejection tests paired; accepting more models is an
  API change, not a convenience tweak.
- Quantized artifacts require their own schema suffix, tolerance, quality, and
  latency gates after FP32 is stable.
