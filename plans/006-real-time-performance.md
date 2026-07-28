# Plan 006: Enforce The Real-Model Latency Budget And Harden The Hot Path

> **Executor instructions**: Finish dependency plans first. Establish benchmarks
> before optimizing, preserve quality gates after every change, and stop if the
> target cannot be met without a model/quality decision. Update `plans/README.md`
> when complete.
>
> **Drift check (run first)**: run
> `rtk shasum -a 256 crates/soundex-core/src/inference.rs crates/soundex-core/src/channel.rs crates/soundex-core/src/processor.rs crates/soundex-core/src/stream.rs crates/soundex-dsp/src/stft.rs crates/soundex-core/benches/core_bench.rs training/models/generator.py training/configs/default.yaml .agents/01-requirements.md`.
> Expected pre-plan hashes for the first eight source/config files are
> `86aa8599583cf3da054ca51532e7845476b2564a3b8ea320d439dd57cc2bcf87`,
> `79af24f3769f8023836bd08e42ca9525765102aba352f9215c2eb153c30832f9`,
> `5d58c3447f4d971bd2da8070e836004c28d55de77ecfd93d5027405c3c4b2c96`,
> `7f5ff49cd2142112a3cc72687f69548cf9129330b34fb4c4a848cd3024b84352`,
> `a4208b5aff3f60a309a94b76ebdc7c9408a7499e5039d96e8ef2fbbb3302f577`,
> `38ca75dcfa7979bdf6e6796f2fcc7fc953a599fa76f25a2716cf26866cf61304`,
> `e7d46371a14b41ef89c8c24b22f5ef224849b9017c26b3634ee17ad2141f4f61`, and
> `5c9d9a19ac729d95708348223cfb873277c15ea5e6a6ed02c43fbf1491195360`.
> Dependency plans will intentionally change them; update this plan against the
> final contract before execution. Git metadata was absent during planning.

## Status

- **Priority**: P1
- **Effort**: L
- **Risk**: HIGH
- **Execution status**: BLOCKED - the repository has only the dynamic-batch
  identity fixture, no trained artifact with passing plan-005 evidence, and the
  available Apple M4 host cannot supply the required x86_64 reference result.
  The selected 1024/512 contract still requires retraining and quality evaluation.
- **Depends on**: `plans/001-streaming-model-contract.md`,
  `plans/002-correct-rust-stream.md`, `plans/004-reproducible-onnx-export.md`,
  `plans/005-deployment-evaluation.md`
- **Category**: perf
- **Planned at**: workspace snapshot without Git metadata, 2026-07-28

## Why This Matters

The selected 1024/512 stream has 512 samples of algorithmic delay: about 11.6 ms
at 44.1 kHz or 10.7 ms at 48 kHz, before device buffers and inference. Hot-path
Rust allocations, stereo batching, explicit ORT settings, and fail-closed evidence
checking are implemented, but they do not establish product performance without
a trained quality-passing artifact and sustained measurements. No current
evidence supports the `<5 ms` real-model compute target or audio-callback reliability.

## Current State

- `.agents/01-requirements.md` now separates algorithmic delay, steady-state p99,
  stress reliability, RTF, hot-path allocation, cold start, and RSS.
- Default 1024/512 delays output by 512 samples and passes the isolated
  algorithmic-delay gate; direct STFT round-trip and impulse/tail tests cover it.
- The warmed Rust path has an allocator test requiring zero SoundEx-controlled
  allocations; ORT C/C++ allocation remains covered only by the pending RSS and
  stress evidence.
- Mono and stereo active channels use one ORT call per hop through a dynamic-batch
  artifact. ORT thread counts, execution mode, and output preallocation are fixed.
- The optimized benchmark rejects identity or sub-1 MB artifacts, records raw
  mono/stereo samples at 44.1/48 kHz, and checks duration, p99, deadlines, RTF,
  session runs, non-finite output, and RSS.
- No artifact trained on 1024/512 has passed plan-005 quality gates, no 30-minute
  run exists, and no x86_64 reference result is available. Causal Rust quality
  evaluation and listening evidence remain mandatory.

## Performance Contract Chosen By This Plan

Use explicit, non-contradictory release gates:

- algorithmic library latency at most 512 samples per channel;
- steady-state real-model stereo p99 processing time below 5 ms on each named
  reference CPU, after warm-up and with enhancement forced on;
- zero missed hop deadlines in a 30-minute stress run;
- no source-controlled heap allocation in `process_frame` after construction;
- model construction/loading occurs off the audio thread and is reported
  separately as cold-start latency; and
- sustained RTF below 1.0 is mandatory. Offline throughput is reported, but the
  inconsistent `>100x` requirement is removed unless separately backed by target
  hardware and a different workload.

If the operator requires a different algorithmic or percentile budget, update
this section and its tests before implementation, not after seeing results.

## Commands You Will Need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Correctness | `rtk cargo test --workspace --all-targets` | all tests pass |
| Benchmark | `SOUNDEX_BENCH_MODEL=<production-architecture.onnx> rtk cargo bench -p soundex-core --bench core_bench` | mono/stereo percentile report |
| DSP bench | `rtk cargo bench -p soundex-dsp --bench dsp_bench` | report generated |
| Lint | `rtk cargo clippy --workspace --all-targets -- -D warnings` | no warnings |
| Quality | From `training/`: `rtk python3 evaluate.py --report-only <candidate-report>` | all release quality gates pass |

## Scope

**In scope**:

- `crates/soundex-dsp/src/stft.rs` and focused buffer/window helpers
- `crates/soundex-core/src/inference.rs`
- `crates/soundex-core/src/channel.rs`
- `crates/soundex-core/src/stream.rs`
- `crates/soundex-core/src/processor.rs`
- `crates/soundex-core/src/config.rs`
- Core/DSP benchmarks and performance regression tests
- `training/models/generator.py` and audio configs only for measured architecture
  or FFT/hop changes that require retraining
- Export metadata affected by FFT/hop/model variants
- `.agents/01-requirements.md`, README, and benchmark protocol documentation

**Out of scope**:

- Weakening quality gates to obtain a latency pass.
- Loading models on an audio callback.
- Claiming mobile/iOS/Android performance without measurements on those devices.
- Adding a second inference backend.
- Quantization before FP32 correctness and quality baselines pass.

## Git Workflow

- Preferred branch: `perf/real-time-generator-path`.
- Keep measurement, buffer reuse, stereo batching, and architecture changes in
  separate conventional commits with before/after reports.
- Example: `perf(core): reuse per-hop spectral buffers`.

## Steps

### Step 1: Benchmark The Production Architecture

Use plan 004 to generate a deterministic, untrained production-architecture ONNX
so compute/operators match a trained model. Benchmark forced-enhancement mono and
stereo paths after warm-up. Record cold model load, first inference, steady p50,
p95, p99, maximum, RTF, RSS, and deadline misses. Pin/document ORT thread counts,
CPU power mode, OS, architecture, and model hash. Benchmark at 44.1 and 48 kHz.

Add stage timers for deinterleave, STFT, gate, tensor preparation, ORT, crossover,
iSTFT, and interleave. Do not use the identity graph for a release latency claim.

**Verify**: the report clearly names the generator artifact and fails if the
identity fixture is supplied to the production benchmark target.

### Step 2: Evaluate Lower-Latency STFT Contracts

Delay is `fft_size - hop_size`. The implementation selects 1024/512 and retains
2048/512 only as an explicit legacy artifact contract. Before release, compare
the selected pair against controlled alternatives using separately trained or
controlled-ablation models, including MAC, p99, high-band release metrics,
transient quality, gate accuracy, and reconstruction. Never load an old
checkpoint under new FFT semantics.

Use the existing overlap-weight normalization rather than assuming a window/hop
pair is COLA. Add impulse and round-trip SNR tests for the selected pair.

**Verify**: selected configuration meets <=512-sample delay, STFT round-trip SNR
>120 dB in steady state, and every plan-005 quality gate.

### Step 3: Remove Source-Controlled Hot-Path Allocations

At processor construction, allocate and retain:

- FFT plans, input/output/scratch and complex spectra;
- log-magnitude/phase/generated/blended buffers;
- channel deinterleave/interleave buffers;
- input/output tensors and process-info storage; and
- overlap/ring storage.

Use a ring index instead of shifting `fft_size` samples each hop. Change DSP APIs
to fill caller-provided slices. Add a counting allocator test around a warmed
`process_frame` with an identity/test engine and require zero allocations from
SoundEx-controlled Rust code. Separately document/measure any unavoidable ORT
allocation.

**Verify**: allocation test passes, outputs remain within existing numerical
tolerance, and stage benchmark shows no regression outside measurement noise.

### Step 4: Batch Stereo In One ONNX Run

Extend `InferenceEngine` to accept batch 1 or 2 as established by plan 004.
Analyze/gate each channel, build one batch for active channels, call ORT once, and
scatter results without changing channel state/order. Share gate transition timing
from plan 002 while retaining per-channel spectral state. Add mono, both-active,
one-active, antiphase, and asymmetric-transient tests.

**Verify**: stereo output matches the pre-batching reference within `1e-5` and the
benchmark records exactly one `Session::run` per stereo hop when either/both
channels require enhancement.

### Step 5: Configure ORT For Predictable Tail Latency

Benchmark explicit intra/inter-op thread settings and graph optimizations on each
reference CPU. Choose settings by p99/deadline misses, not mean. Use preallocated
input/output binding if supported by the pinned `ort` API. If ORT still performs
unbounded work or blocking allocation on the callback, introduce a bounded
real-time-safe worker design only with a documented additional latency term and
underflow behavior. Never silently drop or reuse stale frames.

**Verify**: a 30-minute forced-enhancement stress test records zero missed
deadlines, bounded queue depth if applicable, no non-finite output, and the exact
declared latency.

### Step 6: Reduce Model Compute Only If Needed

If steps 2 through 5 do not meet stereo p99 <5 ms, profile and make one controlled
model change at a time: narrower channels, fewer bottleneck blocks, reduced
frequency width from the selected FFT, operator fusion, then FP16/INT8 where the
target runtime supports it. Re-export/retrain and run the entire plan-005 quality
suite after each change. Do not optimize against random weights alone.

Quantized models need distinct artifact metadata, parity tolerance, model size,
quality, and platform support. Stop if no candidate satisfies both latency and
quality; report the Pareto measurements for an explicit product decision.

**Verify**: final artifact meets size, quality, parity, algorithmic delay, p99,
stress, and RTF gates on all named reference CPUs.

### Step 7: Publish Reproducible Performance Evidence

Update requirements and README to distinguish algorithmic delay, callback
processing time, cold model load, device-buffer latency, and offline throughput.
Add `latency_samples_per_channel()` examples. Publish benchmark commands, hardware,
model hash, percentile samples, ORT settings, and date. Never cite the identity
benchmark as model latency.

**Verify**: a documentation test or script confirms reported model/hash/config
matches the benchmark JSON and release model card.

## Test Plan

- Correctness: existing Rust suite plus selected STFT round trip, impulse,
  arbitrary chunks, stereo batch parity, gate state, and reset/flush.
- Allocation: warmed process call with a counting allocator and stage-local
  counters.
- Performance: production architecture, forced gate, mono/stereo, 44.1/48 kHz,
  cold/warm, p50/p95/p99/max, 30-minute stress.
- Quality: rerun every plan-005 gate after FFT/model/precision changes.
- Platforms: at least one named Apple Silicon CPU and one named x86_64 CPU before
  making the cross-platform `<5 ms` claim.

## Done Criteria

- [x] Selected configuration has <=512 samples algorithmic library delay.
- [ ] Real production-architecture stereo p99 is <5 ms on every named reference
      CPU, with enhancement forced on.
- [ ] A 30-minute stress run has zero deadline misses.
- [x] Warming `process_frame` performs zero SoundEx-controlled heap allocations.
- [x] Stereo enhancement uses one batched ORT run per hop.
- [x] ORT thread and output-preallocation settings are explicit and reproducible.
- [ ] Every plan-005 quality gate passes after the final performance changes.
- [x] Rust fmt, clippy, tests, and no-default-feature check pass.
- [x] Performance docs separate algorithmic, compute, cold-start, and device delay.
- [ ] Only in-scope files and `plans/README.md` changed.

## STOP Conditions

- Dependency plans are incomplete or the artifact/evaluation contract drifted.
- No tested FFT/hop candidate meets both <=512-sample delay and quality gates.
- Stereo p99 <5 ms cannot be reached without failing quality/parity.
- Meeting the target requires an unsupported target-specific operator or silently
  different output.
- A worker-thread design would drop frames, hide deadline misses, or add latency
  not exposed by the API.
- Reference hardware is unavailable. Do not substitute identity or mean-only
  numbers for the required evidence.

## Maintenance Notes

- Store raw benchmark samples and model hashes; summary numbers alone are not
  reproducible evidence.
- Re-run performance gates after ORT, compiler, CPU target, model, FFT, or thread
  configuration changes.
- A future mobile claim needs on-device p99/stress/quality evidence and may need a
  separate artifact; desktop results do not transfer automatically.
