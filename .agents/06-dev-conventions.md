# SoundEx development conventions

## Module size and responsibility

Rust files should not exceed 300 lines including comments/tests; split larger files
by responsibility. A module tree above roughly 1,000 lines should be split into peer
modules or a crate. Move long tests into separate test modules. Each file has one
clear responsibility and a module-level `//!` explanation. A `mod.rs` exporting more
than 10 public types is a signal to review its boundaries.

Python files should not exceed 400 lines. Separate configuration, data, models,
training and evaluation. A file contains one class or a closely related function set.
Use typed function signatures, snake_case modules/functions/config keys, PascalCase
classes and SCREAMING_SNAKE constants. YAML configuration and explicit CLI overrides
must be recorded with seeds and hashes for reproducibility.

## Formatting, naming and errors

Use default rustfmt and Clippy; document justified lint exemptions and consider
pedantic findings. CI requires format checks and warnings-free configured lint.
Rust crates use kebab-case, modules/functions snake_case, types/traits PascalCase,
constants SCREAMING_SNAKE, type parameters short PascalCase and lifetimes short lowercase.
Use `thiserror` for libraries and `anyhow` for the CLI. Do not expose `unwrap()` or
`expect()` failure paths in public APIs; tests are exempt. Prefer typed `?` propagation.

Public Rust items need `///` documentation covering behavior, parameters, returns,
errors and useful examples. Cite sources for complex algorithms. Avoid `unsafe`;
necessary performance-specific unsafe code is isolated, documents invariants with
`SAFETY:` and needs appropriate Miri coverage.

Python uses Ruff formatting/lint with 100-character lines. Run shell tools through
`rtk` or `rtk proxy`. Keep all maintained documentation and new code comments in English.

## Dependencies and tests

Explain new dependencies; prefer maintained pure-Rust implementations where suitable.
This mixed library/CLI workspace commits `Cargo.lock`; do not apply an obsolete
library-only ignore example to the workspace. Published library dependency bounds
remain in their manifests.

Use known-answer DSP tests, round trips, empty/short/silent/full-scale inputs, model
contract/golden comparisons and meaningful performance checks. The original STFT
round-trip target was SNR >120 dB on its specified numerical path; document precision,
boundaries and the actual registered tolerance rather than applying it blindly to
all deployment paths. Rust unit/integration/doc tests and Criterion benchmarks serve
different purposes. Python checks cover data alignment/shapes/ranges, model geometry,
short training execution, ONNX checking and runtime parity.

For new loss math, also verify correct-reference gradients, silence/weak bands,
finite differences and native correspondence. A smoke test is not full training or
quality evidence. Retain failures; never rerun until a benchmark passes.

Older M1 estimates (<50 μs FFT1024, <10 μs detection, <5 ms inference, <6 ms full
pipeline, <30 s for a four-minute file) were design estimates, not current acceptance.
The [real-time standard](../docs/realtime-standard.md) and root [AGENTS.md](../AGENTS.md)
now require model-plus-DSP <5 ms and sustained hop deadlines, with fixed buffer,
maximum/p99, RSS/CPU, lateness and actual stream behavior reported separately.

## Git and versions

Keep main buildable. Use focused feature/fix/training branches when appropriate;
PR titles follow the commit convention and describe one coherent outcome. Required
CI and API documentation accompany changes. Follow SemVer; minor versions may break
APIs during 0.x. Initial project version is 0.1.0. See [commit rules](07-commit-convention.md).
Checkpoints remain ignored; model distribution follows actual artifact size and
repository attributes, not an assumed LFS rule. See [workspace policy](../docs/workspace.md).
