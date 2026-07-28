# Production performance evidence

Performance claims are artifact-bound release evidence. The identity ONNX model
is only a correctness fixture and the benchmark rejects it, even if it is copied
under another path. Models smaller than 1 MB are also rejected as trivial for the
declared 1-2M parameter production architecture, and FP32 artifacts at or above
8 MiB are outside the release contract.

## Measurements

The benchmark runs forced enhancement for mono and stereo at 44.1 and 48 kHz.
Each JSON report contains:

- ONNX path, size, and SHA-256;
- schema-2 FFT size, hop size, and frequency-bin protocol bound to ONNX metadata;
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

The default 1024-point FFT / 512-sample hop has 512 samples per channel of
algorithmic delay: about 11.6 ms at 44.1 kHz or 10.7 ms at 48 kHz. It meets the
isolated algorithmic gate, but no release claim is valid until a model trained on
this exact contract passes quality evaluation, stereo p99, 30-minute stress, RSS,
and both required hardware classes. Device and driver buffers are integration
latency and must be measured separately; they are not included in either
algorithmic delay or `process_frame` execution time.
