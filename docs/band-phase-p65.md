# P65: frozen-magnitude phase calibration and worker observability

Date: 2026-11-01. Status: fixed-budget calibration registered and running;
not full training, independent testing or release qualification.

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

Active evidence: `data/runs/band-phase-p65-20261101-v2/`. Registration freezes
parent hashes, schedule, resources, software versions and executable source.
Source catalogs and registered input hashes are checked before/after execution.
The failed first attempt, `data/runs/band-phase-p65-20261101/`, retains its log,
source and `launch-failure.json`: the snapshot omitted root `parity_policy.json`,
so module import failed before training or quality scoring. Version 2 copies
root configuration resources as well as Python. No threshold or budget changed.

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
The new release binary and stimuli are frozen. It waits for both training and
fit assessment to finish, then freezes all four exported model hashes before
measuring 52,000 hops. There is no retry-to-pass. CPU/RSS and paced-start delays
are recorded, with the fixed 256-frame output buffer separate from computation.
This is a diagnostic frontend/model/DSP screen, not actual callback/transport or
physical-device qualification. Worker observations do not resolve availability
by themselves.

## Next decision

Review both fixed endpoints together, separating raw and native paired/identity
MR, high-band PCM, temporal detail and per-case regressions. A gain on these fit
crops permits further controlled study only. Reject neither a model family from
512 updates nor a spectral candidate solely from a small waveform regression.
Require history-use diagnostics, adequate training, independent source coverage,
blind listening and native worker/device evidence before production promotion.
