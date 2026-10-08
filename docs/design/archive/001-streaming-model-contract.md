# Plan 001: Make The Generator Stateless And Streaming-Equivalent

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving on. If a
> STOP condition occurs, stop and report it instead of improvising. When done,
> update this plan's status row in `docs/design/archive/README.md`.
>
> **Drift check (run first)**: run
> `rtk shasum -a 256 training/models/generator.py training/train.py training/models/losses.py training/configs/default.yaml training/configs/titan_xp.yaml crates/soundex-core/src/inference.rs`.
> The first four expected hashes are respectively
> `e7d46371a14b41ef89c8c24b22f5ef224849b9017c26b3634ee17ad2141f4f61`,
> `2c7f9d7437f8f81e8199fd1c107b796c24cf31f1eaa5fac19d04e74f95d37f8c`,
> `ad8fe32a5c07295af79f009136e752a644338f0c07aa07cf1953363a27e03202`, and
> `5c9d9a19ac729d95708348223cfb873277c15ea5e6a6ed02c43fbf1491195360`.
> If any differ, compare the Current state excerpts with the live code and stop
> on a semantic mismatch. Git metadata was absent when this plan was written.

## Status

- **Priority**: P1
- **Effort**: L
- **Risk**: HIGH
- **Depends on**: none
- **Category**: bug
- **Planned at**: workspace snapshot without Git metadata, 2026-07-28

## Why This Matters

The model is trained with eight adjacent STFT frames and symmetric temporal
convolutions, but Rust always sends one isolated frame. Training can use future
information that does not exist online, while deployment inserts zero padding at
every time convolution. The current validation score therefore says little about
streaming quality. The lowest-latency contract is a stateless model that consumes
exactly one causal STFT frame and performs convolution only along frequency.

## Historical Pre-Plan State

- `training/configs/default.yaml:15` and `training/configs/titan_xp.yaml:17` set
  `context_frames: 8`.
- `training/train.py:176-182` randomly selects an eight-frame context.
- `training/models/generator.py:61`, `:84`, `:120`, and `:185` use symmetric
  `3x3` kernels, so output frame `t` depends on both earlier and later frames.
- `training/models/generator.py:133-139` maps 513 bins to 1026 bins, then
  `:222-225` bilinearly resizes magnitude and raw wrapped phase to 1025 bins.
  Interpolating angles near `-pi` and `pi` through zero is mathematically wrong.
- `crates/soundex-core/src/inference.rs:60-66` accepts only `[1, 2, 1, F]`.
- The pre-plan default deployment tensor was `[B, 2, 1, 1025]`. The later
  low-latency contract migration changed the production width to 513 while
  retaining the same magnitude/phase semantics and fixed single-frame time axis.
- Python conventions require typed function signatures and Ruff formatting.
  Match `training/models/generator.py`. Tests should use `pytest` under a new
  `training/tests/` package.

## Commands You Will Need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Python tests | From `training/`: `rtk python3 -m pytest -q` | all tests pass |
| Python lint | `rtk ruff check training` | exit 0, no findings |
| Python format | `rtk ruff format --check training` | exit 0 |
| Rust tests | `rtk cargo test --workspace --all-targets` | all tests pass |

## Scope

**In scope**:

- `training/models/generator.py`
- `training/train.py`
- `training/models/losses.py` only where tensor shape semantics must change
- `training/configs/default.yaml`
- `training/configs/titan_xp.yaml`
- `training/tests/test_generator_streaming.py` (create)
- `training/tests/test_spectral_features.py` (create if needed)
- `training/requirements.txt` only to add the test dependency
- Model-contract documentation in `.agents/03-model-design.md`

**Out of scope**:

- Dataset split and codec changes, handled by plan 003.
- ONNX metadata and Rust parity, handled by plan 004.
- Rust DSP crossover behavior, handled by plan 002.
- Training release metrics and checkpoint selection, handled by plan 005.

## Git Workflow

- Preferred branch after original Git metadata is restored:
  `fix/stateless-streaming-model`.
- Use conventional commits, for example
  `fix(model): align training with single-frame inference`.
- Do not initialize a new repository, push, or open a PR without instruction.

## Steps

### Step 1: Add Failing Streaming-Contract Tests

Create `training/tests/test_generator_streaming.py` with deterministic seeds and
small channel widths for fast CPU execution. Cover:

1. Input and output are `[B, 2, 1, 513]` for production width.
2. For a tensor with several time positions, processing all positions together
   equals concatenating independently processed `T=1` positions within `1e-6`.
3. Changing preceding or following frames cannot change the selected frame.
4. Odd frequency sizes return the exact input frequency size without bilinear
   interpolation of the output phase channel.
5. A phase sequence crossing `-pi`/`pi` never goes through a linear angle resize.

Tests 2 and 3 must fail on the current implementation.

**Verify**: from `training/`, `rtk python3 -m pytest -q tests/test_generator_streaming.py`
must fail specifically on temporal dependence before the implementation change.

### Step 2: Remove Temporal Mixing From Every Generator Block

Change frequency-processing kernels from scalar `3` to `(1, 3)`, including:

- encoder convolutions;
- depthwise convolutions;
- inverted-residual depthwise convolutions;
- decoder convolutions; and
- fusion convolutions.

Keep frequency stride `(1, 2)`. Replace spatial `F.interpolate` shape repair with
deterministic frequency-only crop or zero-pad helpers. The helper must never mix
the time axis. The final 514-bin transpose-convolution output should be cropped
deterministically to 513 bins, not interpolated. Do not interpolate raw phase
angles anywhere in the output path.

Changing `(3, 3)` to `(1, 3)` will likely reduce the model below one million
parameters. That is acceptable: retain only the `< 2,000,000` upper bound. Do not
inflate the network merely to satisfy an arbitrary lower bound.

**Verify**: from `training/`,
`rtk python3 -m pytest -q tests/test_generator_streaming.py` passes, and a test
prints/asserts `model.count_parameters() < 2_000_000`.

### Step 3: Train On The Deployment Shape

Set `context_frames: 1` in both configs. Replace random multi-frame selection with
a helper that selects one aligned frame for training and a deterministic frame
from a validation manifest for validation. For training, random frame selection
is allowed; for validation, it is not. Keep the STFT feature scale exactly
`20 * log10(abs(X).clamp_min(1e-10))` with periodic Hann and `center=False`, because
that matches the Rust analyzer.

Add a configuration assertion that the stateless generator only accepts
`context_frames == 1`. Old eight-frame checkpoints must be treated as schema 0
and rejected by later export rather than silently reused.

**Verify**: from `training/`, `rtk python3 -m pytest -q` passes and includes a test
that `context_frames: 8` is rejected with a clear error.

### Step 4: Document The Breaking Contract

Update `.agents/03-model-design.md` so it no longer claims training may use an
arbitrary temporal context. State that time is fixed to one frame, kernels are
frequency-only, and temporal history exists only in the causal STFT ring buffer.
Document that old checkpoints require retraining.

**Verify**: `rtk rg -n 'context_frames: 8|1 frame.*or N frames' training/configs .agents/03-model-design.md`
returns no stale production-contract claim.

## Test Plan

- Use deterministic CPU tests with production frequency width and reduced model
  channels where speed matters.
- Assert frame independence numerically, not by inspecting module definitions.
- Cover batch sizes 1 and 2, `T=1`, odd width 513, and a non-production odd width
  to exercise crop/pad logic.
- Cover invalid ranks/channels and non-finite values at the boundary helper.
- Match Rust's periodic Hann/dB convention with a fixed sine-wave golden feature.

## Done Criteria

- [ ] Every generator convolution has temporal kernel and stride equal to one.
- [ ] No output phase angle passes through `F.interpolate` or ordinary linear
      interpolation.
- [ ] Production configs use `context_frames: 1` and other values fail early.
- [ ] Stateless multi-frame and independent-frame outputs match within `1e-6`.
- [ ] Generator parameter count is below two million.
- [ ] `rtk python3 -m pytest -q` passes.
- [ ] Ruff check and format check pass.
- [ ] `rtk cargo test --workspace --all-targets` still passes.
- [ ] Only in-scope files and `docs/design/archive/README.md` changed.

## STOP Conditions

- The current code no longer matches the cited shape or convolution behavior.
- Quality requirements explicitly require learned temporal context. That would
  require a different stateful causal model and a revised Rust/ONNX contract.
- Frequency-only kernels exceed the quality loss allowed by an existing, measured
  release baseline. Report the baseline instead of restoring future context.
- A proposed fix requires ordinary interpolation of raw phase angles.

## Maintenance Notes

- Any future temporal model must expose explicit causal state tensors or a fixed
  past-only context, then update Python, ONNX, Rust, and chunk-equivalence tests in
  the same change.
- Reviewers should search exported ONNX nodes and confirm no resize operates on
  the phase output.
- This plan intentionally favors a short, stateless inference call over temporal
  quality because that matches the stated low-latency streaming objective.
