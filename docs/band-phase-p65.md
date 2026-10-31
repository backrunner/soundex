# P65: frozen-magnitude phase calibration and worker observability

Date: 2026-11-01. Status: both 512-update calibrations, fixed fit reviews and the 52,000-hop
cost grid completed. Early learning is measurable; native music gains are not
demonstrated. This is not full training, independent testing or release qualification.

## Changed hypothesis and controls

[P64](band-attribution-p64.md) found that MP3 magnitude restoration increases
amplitude-weighted phase error without a correspondingly large angular change.
It did not establish that a compressed-input-only predictor can recover better
phase. Earlier [P31](phase-coordinates-study.md) changed encoder inputs and output
coordinates jointly under a different, short training setup. P65 instead isolates
a phase correction on fully trained P63 magnitudes, retaining the common objective
and a matched zero-correction control; it is not evidence against P31's entire
architecture family. This study freezes both completed P63 band endpoints and learns a small
phase readout. It does not repeat full backbone training, provide reference phase
as an input, weaken the spectral objective or remove DSP.

The 25,025-parameter head uses compressed magnitude, observed/predicted relative
phase and a previous observed-frame phase carrier, plus the existing detector
condition. The frozen backbone has 171,137 parameters; total size is 196,162.
Rotation is bounded by pi, with the existing smooth high-band support and zero
DC/Nyquist rotation. Exact digital silence contributes no correction. All base
weights and predicted dB values remain fixed. Phase changes can still alter
OLA interference and reanalyzed magnitude: final spectral scores are not assumed
preserved. There is no future context. Each physical channel has independent
state, growing from 12,096 to 12,354 floats. This remains a diagnostic export;
production ABI and default weights are unchanged.

The head observes temporal phase advance and the base model's relative phase
correction; it does not encode absolute cross-bin phase relationships. A feature-
level constant rotation of each bin leaves its correction unchanged. A numerical
property test records that representation boundary, not a real-music impossibility
proof. This is a deliberately narrow correction hypothesis, not an unrestricted
missing-phase synthesis architecture; failure cannot reject all phase models.

Zero-initialized heads reproduce the parent output. A deterministic nonzero
stress head must pass sequential ONNX and real Rust frontend/DSP parity on mono
and silent-left stereo at both sample rates, and have a finite nonzero native
PCM response, before any optimizer updates. This proves deployment influence,
not useful reconstruction. Frozen zero controls receive the same fit reviews.

## Registered budget and exposure

Both origins (20261012 and 20261013) receive exactly 512 Adam updates at 1e-4.
Both endpoints must finish and freeze their hashes before post-training raw or
native reviews. This is an early trainability calibration: it cannot reject a
capacity, prove convergence or replace adequate full training.

The first 2,048 optimization draws of P63 are reused unchanged: 10,240 paired
crops per origin, 80% music and 20% speech, 182 sampled recordings. Unique raw-MR
supervised support is 3,561.220 seconds; direct-analysis support is 3,599.405
seconds, excluding warmup/overlap. Sampling includes nine codec conditions and
mono/left/right/mid/side views. Manifest genre and region counts are retained in
registration; unknown labels and unequal coverage remain. These are TRAIN-role
crops, not a new generalization test or physical stereo assembled across times.

The unchanged P63 common objective supervises original-reference direct spectra
and raw OLA multiresolution reconstruction. Paired and identity roles remain
balanced, with the same 80-hop history and frame-89 supervision. DSP is not a
teacher or differentiable training path. Native Rust review is a separate check
of actual band fusion, phase smoothing, synthesis and limiting. The first eight
TRAIN draws provide shared before/after raw fit diagnostics; native review uses
the established every-fourth-draw subset. No held-out material is scored.

## Execution evidence and limitations

Completed evidence: `data/runs/band-phase-p65-20261101-v2/`. Registration freezes
parent hashes, schedule, resources, software versions and executable source.
Source catalogs and registered input hashes are checked before/after execution.
The failed first attempt, `data/runs/band-phase-p65-20261101/`, retains its log,
source and `launch-failure.json`: the snapshot omitted root `parity_policy.json`,
so module import failed before training or quality scoring. Version 2 copies
root configuration resources as well as Python. No threshold or budget changed.

### Completion-gate recovery

After both endpoints and all six native quality reports finished, the final hash
check rejected an inherited path, `target/release/soundex-context-replay`. Building
the new observability executable had changed that file. The actual quality calls
used the explicitly registered frozen executable under P63 `assets/`, not this
unused build alias. Its bytes still match the original alias hash
`5750fdb17984f74f3958e925b6e831e3ec81033ac798f1466b670a6fb5a29eba`.

`post-run-hash-audit.json` checks all 91,491 entries: this duplicate path is the
only mismatch. `completion-recovery.json` records unchanged frozen base tensors,
two full endpoints, six verified native artifact reports and **40 bit-identical**
zero-control outputs between before/after reviews. The original registration and
`failure-1793469464068320000.json` remain unchanged. Assessment and timing monitors
also retained their refusal records; no timing case had run at that point.
Completion-only recovery then permitted the first assessment and first fixed cost
grid, without retraining, rerendering, restoring the build file to conceal drift,
or changing a threshold. Future preparation canonicalizes only this proven,
byte-identical unused alias to its frozen asset; non-equivalent aliases fail closed.

## Fixed endpoint results

The eight raw fit batches cover 32 music and eight speech recordings, with paired
and identity roles. Native review covers the registered two-batch subset: eight
music and two speech recordings, with both roles. These are TRAIN-role fit results.
`assessment.json` retains source-equal summaries and every adverse case. The
supplemental `matched-raw-native.json` selects exactly the native case indices
from the existing raw records; it does not rescore audio or change the population.

Changes versus the corresponding zero-correction parent on the **same eight music
recordings**, with lower error preferred:

| Path / metric | Seed 20261012 | Seed 20261013 |
| --- | --- | --- |
| Raw MR | -0.003118% | -0.001090% |
| Raw HF PCM error RMS | -0.033545% | -0.024216% |
| Native MR | +0.000661% | +0.000735% |
| Native HF PCM error RMS | +0.001220% | +0.001460% |
| Native temporal MR dynamics | +0.003548% | +0.002103% |

Native music HF PCM error rises by only `5.26745e-8` / `6.31358e-8` RMS;
these very small differences do not establish audible harm. They also do not
show restoration improvement. Native music MR worsens on 7/8 and 4/8 sources.
The largest MR counterexample in both origins is Andy G. Cohen's *Crunch Me
Beautiful*, MP3 64 kbps, case 4 / role 1, row `08561aa6340587794a8c7e1a`, crop
start 62,411: MR increases `6.11544e-5` / `2.96831e-5`. All other cases remain.

On the wider 32-recording raw music fit panel, MR changes -0.001277% / +0.001828%,
while HF PCM RMS improves 0.023497% / 0.017039%. Raw clean-music HF PCM error grows
from `3.17679e-6` to `3.80750e-6` and `3.73504e-6` to `4.19660e-6` RMS
(+19.85% / +12.36%). These error levels are -109.96 to -108.39 dBFS and
-108.55 to -107.54 dBFS; large relative increases from small absolute errors are
not listening evidence. Raw identity MR also rises 1.16% / 0.54%. Native clean-
music MR instead changes -0.001411% / -0.001513%, with three adverse cases each.

Raw paired speech MR improves 0.015708% / 0.010197%; the two-source native subset
improves 0.002008% / 0.001061%. Native speech identity is bit-identical. The speech
subset is too small for generalization or broad speech-quality claims. Raw speech
identity regressions and the second origin's native speech dynamics regression
remain in the assessment.

### Learning and history

Both 512-update histories are finite, with frozen bases unchanged. Mean common
fit objective falls `0.2427868079 -> 0.2427787744` and
`0.2423049267 -> 0.2423002962` (-0.003309% / -0.001911%). Direct original-reference
loss also falls slightly. All measured frames preserve the parent's predicted dB
exactly. Mean per-batch rotation RMS over supervised bins is approximately
0.000911 / 0.000903 radians.

Repeating the current carrier in place of the phase head's previous carrier
changes phase by about `6.53e-5` / `7.59e-5` radians RMS and worsens mean common
fit objective by `3.59e-7` / `4.99e-7`. Actual history is better on 7/8 fit batches
in each origin. This establishes a small prediction/fit dependence on that
history, not a native history-ablation quality win or convergence. The existing
backbone history is untouched in this ablation.

Endpoint SHA-256 values:

| Origin | Checkpoint | ONNX |
| --- | --- | --- |
| 20261012 | `5adc24a69a768ef1ddcb1415a843c6d78f2b35a06301cfe53a710711a8d69bc7` | `831122cff50fac30d17b5efbb20915f7744f7068429623f55d00445d6bb5fd98` |
| 20261013 | `eeb25886487bb04aca0b918a93f9faf50897de6f0f518e0e049b7d2de09308af` | `227845c81d07d00ebb3d8cfe60371a4933c09c664848c24a2b7882912e5e12b4` |

## Worker observation changes

Production worker statistics now include all-idle-sleep count, maximum overshoot
and approximate p99 overshoot. Submission-to-start queue wait and processing
histograms already existed. Overshoot includes sleep-call/clock overhead and idle
periods without pending audio; it is not per-hop latency or proof of an OS cause.
The paced diagnostic now records arrival-to-start lateness without requiring the
optional thread-CPU probe. Callback operations, buffering and fallback are unchanged.

The functional replay check in
`data/runs/worker-observability-p65-20261101-functional/` runs a frozen debug binary
with CPU probing disabled. All 100 paced hops have start-lateness observations;
the unpaced case has none. PCM is bit-identical between those cases. The retained
runtime JSON is a functional check concurrent with training, not timing evidence.
Rust worker tests cover idle observations before audio work, histogram accounting,
fixed presentation timelines, recovery and channel-state resets.

P63/P64 timing failures remain valid. No concurrent training/build measurements
are used for timing qualification. A separate fixed grid is registered in
`data/runs/band-phase-cost-p65-20261101/`: both zero/trained origins, mono/stereo,
44.1/48 kHz unpaced (16 cases x 3,000 scored hops), followed by four 48 kHz stereo
paced cases (1,000 scored hops each). Every case excludes its first 100 hops.
The new release binary and stimuli are frozen. It waited for both training and
fit assessment to finish, then froze all four exported model hashes before
measuring 52,000 hops. There is no retry-to-pass. CPU/RSS and paced-start delays
are recorded, with the fixed 256-frame output buffer separate from computation.
This is a diagnostic frontend/model/DSP screen, not actual callback/transport or
physical-device qualification. Worker observations do not resolve availability
by themselves.

### Fixed cost-grid results

All 20 cases and 52,000 scored hops completed. Both phase graphs are 898,993 bytes.
The new release binary is `6c627674997e849d925cbfa24ac602b9d043e3c1c43523403134064b5cc9b100`,
built from the Rust changes in `ce08b0d`; build provenance is retained. It differs
from the quality replay binary and historical timing binary. No speedup over
historical builds or machines is inferred.

| Observation | Zero controls | Trained phase graphs |
| --- | --- | --- |
| Maximum model plus frontend/DSP compute | 1.061792 ms | 3.026458 ms |
| Highest case p99 compute | 0.757095 ms | 0.691190 ms |
| Calls at or above 5 ms | 0 | 0 |
| Calls at or above the hop period | 0 | 1 |
| Paced late completions, 1,000 scored hops per origin | 0 / 0 | 0 / 0 |
| Maximum whole Rust process RSS | 30.00 MiB | 30.16 MiB |

The service overrun is seed 20261013 phase, 44.1 kHz mono, unpaced:
3.026458 ms exceeds the 2.902494 ms hop period. It remains a failed strict per-hop
criterion despite staying below the separate 5 ms ceiling. Paced phase maximum
start lateness is 0.933394 / 0.608330 ms, including sleep, backlog and loop costs;
this does not identify an OS cause. Paced process user+system CPU is approximately
13.1% / 11.4% of one core averaged over total process wall time, including startup
and warmup. It is not an isolated inference CPU counter. The fixed production
adapter's 256 frames (5.333 ms at 48 kHz) are separate from compute.

No real callback/transport integration or physical-device qualification of the
experimental graph is claimed. P63/P64 failures remain recorded; zero misses in
these short synthetic paced cases do not erase them.

## Decision and validation

Early learning and native influence are demonstrated, but native music restoration
has not improved. Do not promote this branch or simply increase its budget.
Before further training, isolate where its correction lies relative to native
fusion/phase handling, and whether the restricted carrier features can represent
the required correction. [P66](phase-transport-p66.md) now completes the native
stage/support audit and a frozen-weight support constraint, without new training
or native quality gains. Keep the frozen magnitude and zero controls. Changing
phase representation or adding a verified deployment-aware objective requires a
new registered hypothesis; the original lossless reference must remain the target.
This calibration cannot reject a phase architecture family or establish a capacity
ceiling, and no independent/listening/device claims follow.

Local validation passes 37 targeted Python checks, including seven
phase/aggregation/representation/alias tests.
Rust realtime tests pass 14 checks, with the existing 30-minute stress case not run.
Core all-target checking, Clippy with warnings denied, Ruff, formatting and English
Markdown/link checks pass. Default weights remain unchanged. Failed preflights,
completion-gate recovery, refusal logs, trained endpoints and raw runtime evidence
are retained in ignored experiment directories.
