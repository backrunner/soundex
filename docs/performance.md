# Production performance evidence

Performance claims are artifact-bound release evidence. The identity ONNX model
is only a correctness fixture and the benchmark rejects it, even if it is copied
under another path. Models smaller than 1 MB are also rejected as trivial for the
non-trivial production architecture, and FP32 artifacts at or above
8 MiB are outside the release contract.

## Measurements

The benchmark runs forced enhancement for mono and stereo at 44.1 and 48 kHz.
Each JSON report contains:

- ONNX path, size, and SHA-256;
- schema-4 FFT size, hop size, and frequency-bin protocol bound to ONNX metadata;
- CPU, architecture, OS, host label, power mode, and Rust version;
- explicit ORT intra/inter-op threads, execution mode, graph optimization, and
  output preallocation state;
- model construction and first-frame time;
- raw per-hop nanosecond samples plus p50/p95/p99/max;
- algorithmic latency, processing/audio duration, RTF, deadline misses,
  non-finite output count, session-run count, and RSS.

The release checker derives durations, RTF, throughput, deadline misses, session
runs, and percentiles from the raw hop samples. The raw hop count must cover at
least the configured 30 minutes of audio, the measured wall clock must cover the
configured stress duration, and peak process RSS must remain at or below 50 MiB.

`process_frame` is separately covered by an allocation test after warm-up. That
test observes the Rust global allocator and requires zero allocations on both
forced-enhancement and bypass paths. ONNX Runtime's C/C++ allocator is outside
that counter, so the long stress run and process RSS remain mandatory.

## Release run

Run from the repository root. Relative model/report paths are resolved from that
root. The default duration is 1800 seconds for each of the four cases.

```bash
SOUNDEX_BENCH_MODEL=models/soundex-v1.onnx \
SOUNDEX_BENCH_REPORT=models/soundex-v1.performance-apple.json \
SOUNDEX_BENCH_HARDWARE_LABEL='Apple M4 reference' \
SOUNDEX_BENCH_POWER_MODE='AC power, Low Power Mode off' \
cargo bench -p soundex-core --bench core_bench
```

Repeat on the named x86_64 reference CPU with the identical ONNX SHA-256 and a
different output path. A short engineering run must opt into diagnostic mode:

```bash
SOUNDEX_BENCH_MODEL=models/soundex-v1.onnx \
SOUNDEX_BENCH_REPORT=target/soundex-performance-diagnostic.json \
SOUNDEX_BENCH_DIAGNOSTIC=1 \
SOUNDEX_BENCH_SECONDS_PER_CASE=5 \
cargo bench -p soundex-core --bench core_bench
```

Diagnostic reports are written but deliberately fail the release gate and the
process exits with status 2.

## Verification and binding

From `training/`, validate raw samples and derived fields, case coverage, memory,
artifact hashes, and the two required hardware classes:

```bash
python -m evaluation.performance \
  ../models/soundex-v1.onnx \
  ../models/soundex-v1.performance-apple.json \
  ../models/soundex-v1.performance-x86.json
```

The release model-card checker takes the same two reports and requires their
SHA-256 values in `Real-model runtime evidence`. It also requires the independent
Plan 005 quality report and listening record.

The default 256-point FFT / 128-sample hop has 128 samples per channel of
algorithmic delay: about 2.90 ms at 44.1 kHz or 2.67 ms at 48 kHz. It meets the
isolated algorithmic gate, but no release claim is valid until a model trained on
this exact contract passes quality evaluation, stereo p99, 30-minute stress, RSS,
and both required hardware classes. Device and driver buffers are integration
latency and must be measured separately; they are not included in either
algorithmic delay or `process_frame` execution time.

## Added output latency budget

The preferred added software output latency is strictly below 8 ms; the hard redline is
strictly below 10 ms. Schema-4 reports include `serial_latency_p99_ms` and
`serial_latency_max_ms`: algorithmic delay plus processing time. The maximum also
includes the first frame after construction. This is a conservative component
budget, not a measured device end-to-end latency. The redline is a mandatory gate;
`preferred_target_met` separately records the 8 ms target from the measured maximum.
Schema-3 reports used a 5 ms preference and are not reinterpreted under schema 4.

An integration must also measure the enable/bypass output time difference, with
any additional queueing or block conversion included. Feed 128-sample callbacks
directly when possible: another buffered hop adds 2.67–2.90 ms. Model loading
belongs outside the audio callback. A matching low-latency artifact must be
trained/evaluated on 256/128; renaming an old artifact or changing only its metadata
is not a valid migration.

## Nonblocking real-time output

`RealtimeProcessor` moves the synchronous pipeline to an independent worker.
Audio-side queues use [`rtrb`](https://docs.rs/rtrb/0.3.5/rtrb/) because its bounded
SPSC push/pop operations return immediately without allocating or locking.
Queue packets own fixed arrays; dropping a stale packet frees no heap storage.
The audio callback inspects at most two queued results at a hop boundary and
only accepts the exact presentation timestamp. Backlog is discarded instead of
increasing latency. Gaps reset the worker DSP and invalidate its first output
until causal history is restored. Model error or panic stops enhancement while
dry output continues; shutdown/join belongs to a control thread.

The fixed delay is 128 STFT + 128 handoff = 256 samples: 5.805 ms at 44.1 kHz,
5.333 ms at 48 kHz. The conservative added-delay budget is this fixed delay plus
callback execution and any extra host buffering. Worker inference time is not
added to this budget: a late worker result is rejected. The fixed component fits
the 8 ms software target; final acceptance also counts callback and host buffering.
Both wet and fallback audio retain the same sample clock;
no frames are inserted, repeated, shortened, or removed on a miss. Source NaN/Inf
is replaced by a decaying held sample, predictions are screened before mixing,
and final limiting covers both paths. On failure, the last enhancement correction
fades to zero over 128 samples; recovery requires two healthy hops before a fade
back in. Stereo channels share the fade position. These are transition-safety
checks, not proof that an unvalidated model sounds artifact-free.

Run the paced diagnostic (four cases, 60 wall-clock seconds each by default):

```bash
SOUNDEX_BENCH_MODEL=models/soundex-v1.onnx \
SOUNDEX_REALTIME_REPORT=target/realtime-paced.json \
SOUNDEX_REALTIME_SECONDS=60 \
cargo bench -p soundex-core --bench realtime_bench
```

The report preserves raw callback times, presentation misses, queue saturation,
stale/invalid results, exact output counts, peak and non-finite samples, and the
RMS of waveform differences at hop boundaries versus inside hops. It separately
records producer scheduling lateness; the synthetic paced producer is not an
audio device. The report is always diagnostic and does not replace artifact
parity, listening, the 30-minute wall-clock release stress, or two-hardware gates.
Compare physical-device latency against an untreated path outside the adapter.
Its internal dry fallback deliberately retains the same 256-sample delay as
enhancement, so switching wet/dry inside the adapter cannot measure added delay.

Measure process peak RSS and CPU with an external process monitor. Prefer
hop-aligned 128-frame callbacks. Smaller blocks shorten the available computation
time between completing an input hop and its presentation deadline; larger blocks
can force dry fallback inside one callback even if nominal CPU capacity is
sufficient. Callback size must be included in integration availability tests.

For real-music diagnostics, additionally set `SOUNDEX_REALTIME_PCM_DIR` to a
directory containing `44100-1.f32`, `44100-2.f32`, `48000-1.f32`, and
`48000-2.f32`. These are interleaved little-endian float32 PCM streams at the
named rate/channel count and must cover the configured duration plus one hop.
The benchmark streams them through a small input buffer outside callback timing
and records their hashes. Retain source-track and degradation commands alongside
the report. Omitting this option uses the synthetic signal generator.

Set `SOUNDEX_REALTIME_CASE=44100:2` to repeat only that rate/channel case in a
fresh process; supported values are `44100:1`, `44100:2`, `48000:1`, `48000:2`.
Omit it for all four cases. Reports identify selected-case runs explicitly.
Raw arrays are serialized directly to temporary case files and the combined
report is streamed and atomically replaced after each completed case. The
harness retains one case's trace vectors, without expanding every sample into
a JSON value or retaining four expanded case trees. This limits collector RAM
growth during long tests; its remaining trace/report overhead still belongs in
measured process RSS. Keep CPU utilization units explicit: one-core percentage
differs from percentage of the machine's total CPU capacity.

The default paced producer is an ordinary thread. On macOS, opt into
`SOUNDEX_REALTIME_PRODUCER_RT=1` to request the same 128-frame audio time-constraint
scheduling for that simulated host thread. Promotion/demotion occur outside the
callback timing, and reports retain request/acceptance flags. Compare the two
modes explicitly: an ordinary producer can itself pause and then deliver a burst
of callbacks, which also consumes worker presentation deadlines. The optional
mode does not test a Core Audio device or join its workgroup, and cannot establish
physical-device xrun or latency evidence.

Deterministic failure, delay, saturation, arbitrary-block, and no-allocation tests
run with the workspace suite. An additional accelerated 30-minute stereo signal
checks repeated outages and invalid predictions:

```bash
cargo test --release -p soundex-core thirty_minutes_preserve -- --ignored --nocapture
```

This last test advances 30 minutes of sample time faster than wall time. It
checks signal/state continuity, and does not establish scheduling stability over
30 minutes on an operating system or a physical audio device.
