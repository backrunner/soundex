# Real-time stream standard

This is SoundEx's engineering policy for continuous playback enhancement. It is
not a universal perceptual latency threshold or a specification for microphone
monitoring/hearing assistance. Software correctness, model quality, and physical
device scheduling are separate acceptance gates.

## Default protocol and budgets

| Item | Standard |
|---|---|
| Host callback | 128 frames, aligned with hop boundaries |
| Sample rate | Preserve native 44.1 or 48 kHz; do not resample solely to meet this policy |
| Model protocol | FFT256, hop128, mono/stereo |
| Adapter delay | Fixed 256 frames/channel, including worker handoff |
| Added software delay target | Strictly <8 ms |
| Added software delay redline | Strictly <10 ms, including integration buffering |
| Audio-side callback p99 | Strictly <100 microseconds |
| Audio-side callback maximum | Strictly <500 microseconds and within the host callback deadline |
| Normal worker miss ratio | <=0.1%, excluding startup padding, independently for each test case |
| Longest normal worker miss sequence | <=20 ms |
| Output integrity | Exact output counts; no NaN/Inf; peak <= configured ceiling |
| Physical device continuity | Zero observed xruns/dropouts in 30-minute wall-clock tests per case |

The 128-frame period is 2.667 ms at 48 kHz and 2.902 ms at 44.1 kHz. The worker
normally gets one such period, but never extends the output deadline. The fixed
adapter delay is 5.333/5.805 ms. A 0.5 ms callback ceiling leaves a software budget
of 5.833/6.305 ms before external adaptation. The 8 ms target leaves room for
integration; the 10 ms redline can accommodate one additional 128-frame buffer
(up to 9.207 ms including the callback allowance at 44.1 kHz), but not two.
This arithmetic motivates our thresholds; 5 ms is retained only as a historical
optimization reference and does not decide acceptance.

Host buffering is not always selectable at exactly 128 frames. Query supported
periods and negotiate with the device, then validate the actual arrangement.
Smaller callback fragments shorten the worker's time after hop completion;
larger blocks submit multiple hops in one callback. Merely splitting a larger
buffer into 128-frame slices does not provide extra wall-clock compute time.
If an integration cannot meet availability and the total added-delay budget,
leave enhancement disabled until its bridge is validated. See
[Windows low-latency audio buffer negotiation](https://learn.microsoft.com/en-us/windows-hardware/drivers/audio/low-latency-audio)
and [Apple audio workgroup deadlines](https://developer.apple.com/documentation/audiotoolbox/understanding-audio-workgroups).

## Failure handling and acceptance

Audio callbacks do not invoke the model, wait on locks, grow queues, log, or
allocate. Every valid callback writes all output samples on the same sample
clock. Late predictions are discarded and dry output has the same delay as wet
output. Shared stereo fades handle fallback and recovery. Model failures may
disable enhancement, but must not stop output or accumulate more delay.

The worker limits describe enhancement availability during normal operation;
they are not permission to lose audio samples. Fault-injection tests deliberately
exceed those limits and must still preserve continuous, finite, bounded output.
The 20 ms burst limit bounds how long enhancement can be unavailable. It does
not add 20 ms to playback delay. Recovery fades and the two-hop healthy hold are
separate audible-behavior checks and require listening validation.

Use `realtime_policy::assess` for component budgets, passing any extra buffering
in sample frames/channel. Record `RealtimeStats::max_consecutive_deadline_misses`
alongside total misses. Retain raw `worker_presentation_missed` bits to independently
recompute miss counts and maximum runs. `realtime_bench` schema 2 records these thresholds and
their assessments; schema 1 predates the burst metric and cannot prove the new
availability requirement. Synchronous `core_bench` schema 4 retains its model
compute certification and uses the 8/10 ms component budget; schema 3 used a
historical 5 ms preference and must not be silently relabeled.

Release requires the same model hash, real music plus silence/transients,
mono/stereo at both rates, at least 30 wall-clock minutes per case, and named
Apple Silicon and x86_64 reference machines. Include CPU, peak RSS, callback raw
timings, availability and maximum burst, producer lateness, physical-device
xruns, and output integrity. Synthetic pacing and accelerated sample time do
not certify a device. Model parity, perceptual quality and listening gates remain
required independently.

Measure the difference against an untreated path outside the adapter. Internal
wet/dry switching retains the same delay and cannot measure added latency.
Keep pre-existing device/driver/network delay separate; buffering introduced by
SoundEx's integration belongs inside the software redline. Platforms that pause
the audio callback itself can still xrun despite a nonblocking model worker.

## Worker scheduling

The inference thread first requests macOS USER_INITIATED QoS through the safe
`qos-threads` API (macOS-only dependency; Rust >=1.85, below the default ORT
backend's >=1.88 requirement). This changes only that dedicated thread and is
not a hard real-time scheduling guarantee. Rejected QoS requests are visible in
`worker_stats().macos_qos_applied`. It then requests audio time-constraint
scheduling for exactly 128 frames at the stream rate through Mozilla's safe
[`audio_thread_priority`](https://github.com/mozilla/audio_thread_priority) API
(macOS-only, default features disabled; no D-Bus or process-wide resource-limit
change). The external library sets a preemptible period/deadline equal to a hop
and computation allowance equal to half a hop. The handle is restored on the
worker before model teardown, including error/panic exits.
`macos_realtime_request_accepted` records acceptance at startup, not a live policy
query: macOS may subsequently demote an over-budget worker. The native audio
integration still needs to join the device workgroup; this library has no device
handle. Other platforms report false and require their own scheduling validation.
See [Apple thread QoS guidance](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/EnergyGuide-iOS/PrioritizeWorkWithQoS.html)
and [Mach soft real-time constraints](https://developer.apple.com/library/archive/documentation/Darwin/Conceptual/KernelProgramming/scheduler/scheduler.html).

The worker sleeps until 500 microseconds before the next expected submission,
then polls in 50-microsecond intervals. Under normal scheduling it spins only
from 150 microseconds before to 50 microseconds after the predicted arrival.
Accepted real-time scheduling disables idle spinning so the computation quota
is spent on inference. The earlier coarse wakeup was selected by the
[2026-10-10 diagnostic](loss-audit-20261010.md); it reduces stereo misses on the
reference host but does not establish all-case/device qualification.
Outside that window it sleeps; after a full idle period it backs off to 1 ms.
The audio callback sends no OS wake signals and takes no locks. Predictions
follow actual submission timestamps and do not add buffering or move deadlines.
Obsolete input is skipped; a gap resets model history and invalidates warmup.
Predictions finishing after presentation are discarded. Fallback/recovery and
the fixed delay remain unchanged.

Worker diagnostics record processed/dropped hops, maximum submission queue wait,
maximum processing wall time, and processing calls exceeding a hop period.
Read them on a control thread. Processing overruns, queue waits and callback
presentation misses are distinct metrics; maxima alone cannot identify the
cause of every miss. Benchmark schema 2 includes these optional worker fields.
