# Plan 002: Make The Rust Stream State Machine And Spectral Blend Correct

> **Executor instructions**: Follow every step and verification gate. Stop on
> any listed condition instead of widening scope. Update `docs/design/archive/README.md` when
> complete.
>
> **Drift check (run first)**: run
> `rtk shasum -a 256 crates/soundex-core/src/channel.rs crates/soundex-core/src/processor.rs crates/soundex-core/src/stream.rs crates/soundex-core/src/analysis.rs crates/soundex-dsp/src/crossover.rs crates/soundex-dsp/src/phase.rs crates/soundex-core/tests/processor_test.rs`.
> Expected hashes for the first, second, third, sixth, and seventh files are
> `79af24f3769f8023836bd08e42ca9525765102aba352f9215c2eb153c30832f9`,
> `5d58c3447f4d971bd2da8070e836004c28d55de77ecfd93d5027405c3c4b2c96`,
> `7f5ff49cd2142112a3cc72687f69548cf9129330b34fb4c4a848cd3024b84352`,
> `95970bb2ab701e6659b5a3223ecd5fcbaf49253ce1d1324ce587cd0c6efa449b`, and
> `95f633dde245393b3e2038d35efa97af7e3d69ba0318207c57f9d1b8581912ec`.
> Git metadata was absent when this plan was written; stop on semantic drift.

## Status

- **Priority**: P1
- **Effort**: L
- **Risk**: HIGH
- **Depends on**: none
- **Category**: bug
- **Planned at**: workspace snapshot without Git metadata, 2026-07-28

## Why This Matters

When enhancement is active, the current code retains low-band magnitude but
replaces almost all low-band phase with model output. It also applies gain only
above the nominal cutoff although crossover starts below it, and it switches
between wet and dry output in one hop. The streaming API cannot flush its tail,
silently drops input after an error, and can desynchronize stereo channels if the
second inference fails. These are correctness problems before they are quality
or performance problems.

## Current State

- `crates/soundex-dsp/src/phase.rs:40-44` initializes from predicted phase and
  only edits the transition, leaving predicted phase below the transition.
- `crates/soundex-core/src/channel.rs:122-150` blends magnitude but uses the phase
  result for the entire spectrum.
- `crates/soundex-core/src/channel.rs:134` applies generated gain from
  `cutoff_bin`, while `crossover.rs:37-39` starts blending half a transition width
  below that bin.
- `channel.rs:82` selects a whole wet or dry hop, so hangover release has no ramp.
- `stream.rs:50-55` drops samples when full without returning consumed length.
- `processor.rs:115-127` flushes only inside `process_buffer`; streaming callers
  cannot retrieve the final `fft_size - hop_size` samples.
- `channel.rs:53-80` mutates input/detector/synthesis state before all fallible
  operations finish. `processor.rs:75-84` advances channels sequentially and may
  partially write stereo output before a later failure.
- `analysis.rs:84-95` uses forward-looking frames, unlike the zero-history causal
  streaming path.
- Rust error handling uses `SoundExError` and `Result`; match `processor.rs`.

## Commands You Will Need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Unit/integration | `rtk cargo test --workspace --all-targets` | all tests pass |
| Lint | `rtk cargo clippy --workspace --all-targets -- -D warnings` | no warnings |
| Format | `rtk cargo fmt --all -- --check` | exit 0 |
| Model-free check | `rtk cargo check -p soundex-core --no-default-features --all-targets` | exit 0 |

## Scope

**In scope**:

- `crates/soundex-dsp/src/crossover.rs`
- `crates/soundex-dsp/src/phase.rs`
- `crates/soundex-dsp/src/stft.rs` only for shared frame iteration if required
- `crates/soundex-core/src/channel.rs`
- `crates/soundex-core/src/stream.rs`
- `crates/soundex-core/src/processor.rs`
- `crates/soundex-core/src/analysis.rs`
- `crates/soundex-core/src/error.rs`
- `crates/soundex-core/tests/processor_test.rs`
- Focused new Rust test modules under these crates
- Public streaming documentation in `README.md`

**Out of scope**:

- Generator architecture and checkpoint schema.
- Heap-allocation optimization and stereo batching, handled by plan 006.
- A joint-stereo neural model.
- Changing FFT/hop defaults before the latency study in plan 006.

## Git Workflow

- Preferred branch: `fix/stream-state-and-crossover`.
- Suggested commits:
  `fix(dsp): preserve low-band phase during crossover`, then
  `fix(core): make streaming flush and failures explicit`.
- Do not initialize Git or push without operator instruction.

## Steps

### Step 1: Lock In Spectral Crossover Regressions

Add pure DSP tests where original phase is zero and predicted phase is near `pi`:

1. Every bin with crossover weight zero retains original magnitude and phase.
2. Every bin with weight one uses generated magnitude and phase.
3. Transition phase uses complex-domain interpolation and remains continuous
   across `-pi`/`pi`.
4. Applying a non-unity generated gain produces no discontinuity at the nominal
   cutoff or transition start.

**Verify**: `rtk cargo test -p soundex-dsp crossover` and
`rtk cargo test -p soundex-dsp phase` fail on the current low-band phase behavior,
then pass after step 2.

### Step 2: Blend A Complex Spectrum With One Weight Definition

Refactor crossover so magnitude gain and phase use the same per-bin weights.
Initialize output phase from `original_phase`, copy predicted phase only where
the generated weight is one, and interpolate unit complex values in the
transition. Apply generated gain throughout every bin whose generated weight is
nonzero, not only from `cutoff_bin` onward. Validate all slice lengths and return
a typed error instead of truncating through `zip`.

Keep the original spectrum exactly unchanged below the transition. This is a
bit-exact invariant, not an approximate quality goal.

**Verify**: `rtk cargo test -p soundex-dsp` passes, including the four new tests.

### Step 3: Add A Gate Mix Ramp

Replace whole-hop wet/dry selection with an explicit gate state and a short
sample-domain equal-power or raised-cosine ramp. Attack may start immediately but
must ramp; release must remain wet long enough to drain the overlap-add tail and
then ramp to dry. Share the decision across both stereo channels so one channel
cannot switch a hop earlier than the other. Per-channel bandwidth can still be
reported.

Add impulse and sine tests that bound the maximum sample-to-sample discontinuity
at both transitions. Do not choose the bound by inspecting the new output; derive
it from the dry signal and ramp length in the test.

**Verify**: `rtk cargo test -p soundex-core --test processor_test` passes the new
gate-transition tests.

### Step 4: Expose A Complete Streaming State Machine

Add public APIs with explicit consumed/produced semantics:

- `latency_samples_per_channel() -> usize` returning `fft_size - hop_size` for
  this STFT implementation;
- `process_chunk` accepting arbitrary complete interleaved sample frames and
  returning how many input/output samples were consumed/produced; and
- `flush` or `finalize` that emits the delayed tail exactly once.

Keep `process_frame` as the fixed-hop convenience API. Change `StreamBuffer::push`
to report consumed length or return an overflow error; it must never silently
drop input. Calling process after finalize must fail until `reset`.

**Verify**: integration tests split one signal into many chunk patterns (1 sample,
prime lengths, exact hop, and larger than hop). Concatenated output plus flush
must match `process_buffer` within `1e-6` and have the same length.

### Step 5: Define Failure Atomicity

Do not let a failed inference silently continue from partially advanced state.
Use one of these explicit semantics and document it:

1. stage all channel output/state and commit only after every channel succeeds;
   or
2. mark the processor poisoned after a processing error and reject all calls
   until `reset`.

The second option is acceptable if ORT/session state cannot be cheaply cloned.
Output buffers must remain unchanged on error. Add a test-only failing inference
fixture or injectable engine abstraction to fail the second stereo channel.

**Verify**: the test proves no partial output is written and the documented
reset/poison behavior occurs.

### Step 6: Share Causal Framing With Dry-Run Analysis

Extract one causal frame iterator/state helper used by `analyze_buffer`,
`process_buffer`, and streaming. It must share zero-history startup, hop alignment,
tail padding, and detector updates. Add short-input tests for lengths 1,
`hop_size - 1`, one hop, and `fft_size - 1`.

Do not redesign the bandwidth detector in this step. Record its real-codec
calibration as part of plan 005.

**Verify**: dry-run frame decisions equal the decisions captured while processing
the same buffer with an identity model.

## Test Plan

- Extend `processor_test.rs`, using its model fixture path pattern.
- Add non-identity/failing fixtures only if generated deterministically and small;
  otherwise add an injectable test engine without exposing it publicly.
- Cover mono/stereo, empty input, short input, arbitrary chunks, flush once,
  repeated flush, reset, error recovery, gate attack/release, and phase wrap.
- Assert exact low-band preservation in the spectral unit test.
- Assert stream/offline waveform and decision equivalence.

## Done Criteria

- [ ] Zero-weight crossover bins preserve original magnitude and phase exactly.
- [ ] Generated gain is continuous across the full crossover region.
- [ ] Gate transitions use a tested ramp and stereo channels share gate timing.
- [ ] No public buffer API silently drops samples.
- [ ] Arbitrary chunking plus flush matches offline output within `1e-6`.
- [ ] `latency_samples_per_channel()` reports the tested delay.
- [ ] Processing failure cannot partially advance stereo state or output.
- [ ] Dry-run and processing share causal frame decisions.
- [ ] Rust fmt, clippy, default tests, and no-default-feature check all pass.
- [ ] Only in-scope files and `docs/design/archive/README.md` changed.

## STOP Conditions

- The chosen failure semantics require cloning or rolling back an ORT session.
  Use the poisoned-state option instead; stop if even that cannot be made clear.
- A gate-ramp design changes total latency beyond the value reported by the API.
- Fixing phase requires changing model output representation; that belongs in
  plans 001 and 004 and must be coordinated there.
- Existing consumers rely on silent sample dropping or repeated flush. Report the
  compatibility issue instead of preserving incorrect behavior silently.

## Maintenance Notes

- Review every future DSP change against the exact low-band passthrough invariant.
- Any FFT/hop change must update latency tests and causal framing together.
- Stereo coherence beyond shared gate/gain remains a model/data evaluation topic;
  plan 005 must measure it before release.
