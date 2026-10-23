# P40: CPU time versus elapsed-time attribution

The opt-in diagnostic replay now records calling-thread CPU time alongside the
existing frontend/ORT/DSP elapsed stages, enclosing service wall time and paced
wake/completion lateness. It does not change inference, DSP, production callbacks
or scheduling policy. The timed loop preallocates the new observation buffers.

## Measurement contract

`SOUNDEX_DIAGNOSTIC_THREAD_CPU=1` enables `CLOCK_THREAD_CPUTIME_ID` on macOS/Linux.
The clock measures CPU time charged to the calling thread rather than all process
threads. See [Apple's implementation](https://github.com/apple-oss-distributions/Libc/blob/main/gen/clock_gettime.c)
and the [Linux clock documentation](https://www.man7.org/linux/man-pages/man2/clock_gettime.2.html).
The replay configures ORT for one intra-op thread, one inter-op thread and sequential
execution; this still does not make the clock a measure of all process activity.
Unsupported platforms fail explicitly when the probe is requested.

The enclosing wall interval includes the CPU-clock reads; its difference from CPU
time includes descheduling, blocking, timestamp overhead and clock-resolution error.
It is **not** a scheduler-event counter and cannot identify which process or kernel
operation caused a delay. Existing stage measurements retain their original scope.
All 1,000 empty-probe calibration samples are retained per process; estimated clock
cost is never subtracted to manufacture a faster result. Schema version 2 names
these separate timing arrays explicitly; probe-off runs retain empty CPU arrays.

Unit tests verify that sleeping increases elapsed time without charging the same
amount to thread CPU, and busy work advances the CPU clock. These OS-clock tests
run natively and are excluded from Miri; no Miri/FFI emulation evidence is claimed.
The existing `libc` workspace dependency is now available to the diagnostic binary;
no new package/version or audio-callback FFI path is introduced.

## Fixed on/off protocol

All four frozen P38 endpoints are tested in separate processes, with two rounds
reversing endpoint and probe order: off/on, then on/off. Each process uses the same
48 kHz stereo multitone, 100 warmup hops and 10,000 measured hops, paced at hop 128.
USER_INITIATED QoS is requested; time-constraint scheduling stays off. Every run is
retained, including misses. There is no concurrent SoundEx training or build during
measurement, and no imposed stress load; ordinary OS/application activity and light
documentation work are uncontrolled. This is not an idle-host or real-device test.

The old P39 PCM for each endpoint is the output reference. All source, input,
checkpoint, export, reference PCM and binary hashes are retained. The instrumented
binary and executed runner/source are snapshotted. New timing results cannot erase
P39's three >=5 ms calls or establish that those historical calls were descheduling.
A previous binary's stage wall time alone cannot be retroactively split into CPU
and waiting time.

## Execution and decision

Completed all 16 processes on 2026-10-23: **160,000 measured hops**, zero >=5 ms
stage-service calls in this run, maximum **2.986 ms**, **two hop service overruns**
and **97 paced completion delays**. All four endpoints produce exactly the same PCM
as P39 (maximum absolute difference 0), with finite outputs and unchanged frame counts.
This run does not reproduce P39's >=5 ms service outliers and cannot explain them
retroactively. Continuous deadline compliance remains unqualified.

| Run | Endpoint | CPU probe | Stage p99 (ms) | Stage max (ms) | Hop overruns | Paced late |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | control-16192-20261012 | off | 0.733 | 1.038 | 0 | 0 |
| 2 | control-16192-20261012 | on | 0.747 | 1.194 | 0 | 0 |
| 3 | control-16192-20261013 | off | 0.843 | 2.259 | 0 | 4 |
| 4 | control-16192-20261013 | on | 0.826 | 1.837 | 0 | 6 |
| 5 | fixed_detector-16192-20261012 | off | 0.822 | 2.797 | 1 | 2 |
| 6 | fixed_detector-16192-20261012 | on | 0.824 | 1.389 | 0 | 0 |
| 7 | fixed_detector-16192-20261013 | off | 0.752 | 1.011 | 0 | 0 |
| 8 | fixed_detector-16192-20261013 | on | 0.467 | 1.332 | 0 | 7 |
| 9 | fixed_detector-16192-20261013 | on | 0.729 | 1.599 | 0 | 47 |
| 10 | fixed_detector-16192-20261013 | off | 0.772 | 2.986 | 1 | 3 |
| 11 | fixed_detector-16192-20261012 | on | 0.731 | 1.394 | 0 | 1 |
| 12 | fixed_detector-16192-20261012 | off | 0.774 | 2.210 | 0 | 1 |
| 13 | control-16192-20261013 | on | 0.721 | 1.242 | 0 | 3 |
| 14 | control-16192-20261013 | off | 0.820 | 2.160 | 0 | 8 |
| 15 | control-16192-20261012 | on | 0.813 | 2.441 | 0 | 8 |
| 16 | control-16192-20261012 | off | 0.802 | 1.326 | 0 | 7 |

Probe-off: 80,000 hops, max stage time 2.986 ms, two service overruns, 25 late
completions. Probe-on: 80,000 hops, max stage time 2.441 ms, no service overruns,
72 late completions. Calling-thread CPU max is **0.994 ms**, while the largest
within-service wall-minus-CPU gap is **1.759 ms**. The probe-on/off difference in
late counts is not a causal overhead estimate: two rounds cannot control arbitrary
host activity or time-varying wakeup behavior. Lower measured maxima are not evidence
that adding instrumentation speeds up inference.

Of the 72 instrumented late completions, **55 hops start work after their deadline
has already passed**. This includes propagated backlog, not 55 independently
identified scheduler events. Every late hop consumes at most **0.685 ms of measured
thread CPU**. The largest start delay is **8.521 ms**, followed by 0.425 ms enclosing
service time / 0.424 ms thread CPU, completing 6.280 ms late. Another hop spends
1.840 ms inside the enclosing service interval but only 0.615 ms on CPU. These
observations establish that elapsed delay and active computation differ materially
in this runner. They do not name the responsible OS activity or prove that the
production callback/worker has the same behavior.

Maximum per-process empty-probe p99 is **0.334 microseconds**, with observed empty-
probe maximum **3.917 microseconds**, without subtraction. Peak process RSS is
31.750 MiB with probes off and 34.344 MiB on; the latter includes retained diagnostic
arrays and JSON/report construction, not just streaming model state. Summed process
CPU is 61.95 seconds over 431.77 wall seconds, including initialization, pacing and
report work. No standalone worker-only CPU percentage is inferred.

The next runtime comparison should instrument the existing worker/queue benchmark
under its actual scheduling policy, including time-constraint acceptance, callback
queue depth, fallback and recovery. This diagnostic explicitly kept time-constraint
scheduling off. Do not change production buffering, claim device xruns are resolved,
or substitute these passing compute samples for the retained P39 failures.

Evidence: `data/runs/cpu-attribution-p40-20261023/`, with registration, frozen binary,
source, every raw timing/PCM report, process CPU/RSS receipts and final summary.
Entry point: `training/scripts/bench_cpu_attribution.py`.

The fixed 256-frame buffer remains 5.333 ms at 48 kHz, separate from model-plus-DSP
compute. Callback queues, actual device xruns, long-duration availability and
listening are outside this diagnostic. See [P39 timing](interleaved-timing-audit.md),
[P40 optimizer diagnosis](update-direction-audit.md) and [failure registry](failed-experiments.md).
