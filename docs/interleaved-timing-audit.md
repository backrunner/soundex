# P39: balanced-order CPU timing audit

Completed 2026-10-23. **The <5 ms compute target is not qualified:** three of 160,000
measured hops took at least 5 ms. Both a native-condition control and a fixed-condition
endpoint had outliers. This does not establish that fixing detector conditions causes
the long tail, and it does not erase the P38 failures.

## Fixed protocol

Reuse all four frozen P38 endpoints, the same Rust frontend → ORT → DSP binary and
48 kHz stereo multitone input. Each endpoint runs four times, once in each order
position. The four Williams orders balance every ordered predecessor pair within
rounds. Each fresh process runs 100 warmup and 10,000 measured hops, paced at
hop 128 (2.667 ms). USER_INITIATED QoS is requested; time-constraint scheduling is
off, fusion is current and detector input is native. No endpoint is selected by score.

The benchmark is serial and no SoundEx training runs concurrently. Background OS
activity, documentation translation/network work and inventory reads are uncontrolled;
this is an order-balanced workstation diagnostic, not an idle-host or real-device
certification. All runs are retained without retrying until they pass. CPU stage
measurements are elapsed wall time and include descheduling; they are not per-thread
CPU-time attribution.

## Every measured run

All times below are milliseconds. Service overruns compare compute with the 2.667 ms
hop period; paced delays compare actual completion with the intended wall-clock
schedule. These measure different failure modes.

| Round | Position | Endpoint | p99 | Max | ≥5 ms | Service overruns | Paced delays |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | control-16192-20261012 | 0.838 | 1.322 | 0 | 0 | 0 |
| 1 | 2 | control-16192-20261013 | 0.856 | 1.267 | 0 | 0 | 0 |
| 1 | 3 | fixed_detector-16192-20261013 | 0.835 | 1.974 | 0 | 0 | 2 |
| 1 | 4 | fixed_detector-16192-20261012 | 0.827 | 1.269 | 0 | 0 | 1 |
| 2 | 1 | control-16192-20261013 | 0.733 | 1.095 | 0 | 0 | 1 |
| 2 | 2 | fixed_detector-16192-20261012 | 0.461 | 9.280 | 2 | 5 | 75 |
| 2 | 3 | control-16192-20261012 | 0.213 | 0.533 | 0 | 0 | 1 |
| 2 | 4 | fixed_detector-16192-20261013 | 0.306 | 0.577 | 0 | 0 | 0 |
| 3 | 1 | fixed_detector-16192-20261012 | 0.249 | 0.451 | 0 | 0 | 0 |
| 3 | 2 | fixed_detector-16192-20261013 | 0.306 | 0.702 | 0 | 0 | 0 |
| 3 | 3 | control-16192-20261013 | 0.426 | 0.862 | 0 | 0 | 3 |
| 3 | 4 | control-16192-20261012 | 0.815 | 1.612 | 0 | 0 | 8 |
| 4 | 1 | fixed_detector-16192-20261013 | 0.846 | 4.573 | 0 | 1 | 21 |
| 4 | 2 | control-16192-20261012 | 0.782 | 1.242 | 0 | 0 | 1 |
| 4 | 3 | fixed_detector-16192-20261012 | 0.818 | 1.540 | 0 | 0 | 1 |
| 4 | 4 | control-16192-20261013 | 0.826 | 5.065 | 1 | 2 | 5 |

Across all rounds: **160,000 measured hops, max 9.280 ms, three ≥5 ms calls, eight
service overruns and 119 paced completion delays**. Maximum paced lateness is
19.255 ms; this can exceed individual service time because wakeup/scheduling delay
is included. Per-run p99 ranges from 0.213 to 0.856 ms. Peak process RSS is
31.641 MiB. Summed process CPU time is 57.76 seconds and wall time is 431.29 seconds;
these include process initialization and paced waiting and are not worker-only CPU
percentages. Every output is finite. Repeated PCM output is identical for each
endpoint (maximum difference 0), so order did not change the audio result.

The three ≥5 ms calls spend 9.253, 5.416 and 5.026 ms inside the measured ORT stage;
one other overrun also has 2.031 ms in the frontend. This localizes the elapsed-time
intervals, not their CPU-versus-preemption cause. Most hops are inexpensive, but
small p99 values cannot establish continuous deadline compliance.

The fixed 256-frame buffer still contributes 5.333 ms at 48 kHz, separately from
compute. No callback, physical device, queue recovery or listening qualification is
claimed. The new run does not change P38's failed quality screen or default weights.

## Evidence and follow-up

Local evidence: `data/runs/interleaved-cost-p39-20261023/`, including registration,
all 16 raw timing/PCM reports, process timing records and summary. Input, checkpoint,
export and binary hashes were checked before and after; the executed script is
retained under `source/`. Entry point: `training/scripts/bench_interleaved.py`.

The next useful timing change is per-thread CPU-time instrumentation alongside wall
time and callback/queue telemetry. It must separate scheduling stalls from actual
inference work under a predeclared background-load protocol. Another uninstrumented
best-case benchmark or a model-size increase cannot resolve that attribution.

See [P38 quality results](detector-condition-study.md), the [objective diagnosis](objective-conflict-audit.md)
and the [failure registry](failed-experiments.md).
