# P62: band-local causal TCN feasibility

Date: 2026-10-26. Status: prototype implemented; compute screen passed; paced
continuity not qualified; **zero training updates and no restoration-quality claim**.
Default weights remain unchanged.

Follow-up: [P63](band-training-p63.md) registers matched fresh training and adds
a fixed CPU/wake audit. That audit records an additional 24.131125 ms stage
wall-time outlier; the original measurements below are retained, not a claim
that all subsequent runs stay below 5 ms.

## Why this experiment

The [P61 architecture review](architecture-evaluation-p61.md) proposes delaying
global frequency compression. Earlier capacity and critic failures do not test
this representation. The experiment retains P57 observations, complex residual
coordinates, generation budget, high-band adapter, direct/MR loss and Rust DSP.
It changes frequency representation and temporal organization. It does not
reintroduce the retired P59-P61 critic or change loss weights.

Diffusion and Transformers are not excluded model families:

- [AudioSR's official implementation](https://github.com/haoheliu/versatile_audio_super_resolution)
  documents a default of 50 DDIM steps. Repeated denoising and its synthesis path
  are substantially harder to budget for a 128-frame CPU streaming hop. This is
  an engineering inference, not a measured AudioSR latency or a claim that every
  distilled diffusion system must fail. No AudioSR code or weights are imported.
- [Apollo](https://arxiv.org/html/2409.08514v1) separates frequency-band modeling
  from temporal modeling. Frequency-axis attention does not read future time;
  a Transformer can also use causal temporal masks and bounded caches. Our
  [official source review](implementation-review-20261026.md) does not qualify
  Apollo's complete graph for this project's clock. A small attention mixer
  remains a possible controlled replacement if the band representation helps.
- TCN gives a straightforward single-pass, bounded-history starting point.
  Choosing it first does not establish optimal quality or prove that a
  Transformer would exceed 5 ms. Compare actual graphs, not architecture names
  or parameter counts as latency proxies.

## Implemented representation

`training/models/band_tcn.py` adds an independent `BandTCN` prototype:

1. Split each observed magnitude FFT into eight contiguous, exhaustive bands.
   Short-FFT boundaries are bins 0/8/16/24/40/56/80/104/129. Long-FFT boundaries
   align at the same normalized frequencies and include their own Nyquist bins.
   These are normalized-frequency partitions; physical edges scale with rate.
   Existing physical-rate/detector conditions remain visible to every encoder.
2. Independently encode local short/1024/2048 magnitudes, log RMS and conditions
   into 24 features per band. No phase-input or teacher/DSP feature is added.
3. Six blocks mix the current frame's eight bands with an 8-to-16-to-8 MLP,
   then apply a shared depthwise causal temporal block independently to each
   band. Dilations remain 1/2/4/8/16/32; the temporal field is 64 feature steps.
4. Keep explicit history per band and physical channel. A shared readout maps
   the concatenated 192 band features to width 96 after temporal processing.
   Current/temporal readouts feed the unchanged 192-dimensional output heads.
   This is not yet a band-local decoder: global compression remains at readout.

| Fresh diagnostic graph | Parameters | Float state/channel | ONNX bytes |
| --- | ---: | ---: | ---: |
| Global P57 structure | 480,809 | 6,048 | 1,967,652 |
| Band TCN, 8 x 24 | 171,137 | 12,096 | 786,658 |

Both are freshly initialized with seed 62 and deterministic nonzero output-head
weights for stress testing. They are not historical checkpoints or trained
candidates. Different parameter counts and temporal sharing mean this is **not
a capacity-matched quality comparison**. More band operations make the smaller
graph slower here, illustrating why weight count cannot replace measurement.

## Numerical and native verification

Ten new Python tests cover exhaustive FFT-bin assignment, zero-start identity,
nonzero residual/history sensitivity, one-hop and irregular-chunk equivalence,
future perturbations, reset, physical-channel isolation, silence/weak gradients,
the existing original-reference objective's finite differences, ONNX sequential
state parity and actual Rust frontend/ORT/PCM integration at 44.1/48 kHz.
Repeated-current-frame sensitivity demonstrates a working history path in the
stress fixture; it is not evidence that training learns useful history.

Band ONNX parity over 320 hops gives maximum Cartesian error 1.26441e-7 and
state error 4.50015e-6. Native prediction comparison on Rust observations gives
maximum Cartesian error 4.26496e-6; silent physical-channel output is exactly
zero and mono/non-silent stereo outputs agree within the existing tolerances.
Quiet/bypassed frames still advance model state. No production artifact ABI
or device callback behavior changes.

Retained preflight failures:

- Float32 exact prefix equality failed with maximum 7.62939e-6 dB/phase-coordinate
  difference after changing future rows. Double precision gives exact prefix
  equality; float32 uses the existing 2e-5 absolute/relative numerical tolerance.
  No model arithmetic or future access was added to make this pass.
- The initial parameter finite difference incorrectly recomputed detached
  warmup and then used fixture conditions inconsistent with `probe_inputs`.
  Holding warmup fixed and deriving identical conditions reproduces the common
  objective exactly and passes the original derivative tolerance. Loss is unchanged.
- First complete suite: 819 passed / 2 failed. The old Rust diagnostic loader
  allocated `width * 63 = 6,048`, but the graph required 12,096 state values.
  The corrected loader checks matching float32 dynamic-channel/static-state
  input/output shapes and retains the existing bounded allocation limit. The
  production loader is unchanged. Original rejected graphs/stderr remain under
  `data/runs/band-tcn-p62-preflight-20261026/`.

Completed local checks: Python **821 passed**, 254 warnings, 68.78 s; Rust
**132 passed / 1 ignored**; workspace Clippy, no-default-feature check, Ruff
and formatting pass. Existing exporter/MPS warnings are not suppressed.

## Registered runtime evidence

`training/scripts/bench_band_tcn.py` freezes source/resources, the release Rust
binary, graphs and timing schedule before measurement. Each graph receives
3,000 scored hops for each 44.1/48 kHz x mono/stereo combination, then 1,000
paced 48 kHz stereo hops: 13,000 per graph, **26,000 total**. Each run excludes
100 warmup hops. Deterministic multitone input is repeated without resetting
history; all hops execute inference. No training/tests/build run concurrently
with timing. Host load is retained; this is an ordinary desktop, not an isolated
benchmark machine. Platform: Apple M5 Max, macOS 27.0.1, CPU ORT, one thread.

| Graph / case | Mean compute ms | p99 ms | Maximum ms | Paced late hops | Native peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global, 48 kHz stereo unpaced | 0.148921 | 0.173127 | 0.194417 | n/a | 31,408,128 B |
| Band, 48 kHz stereo unpaced | 0.237344 | 0.264754 | 0.325542 | n/a | 30,785,536 B |
| Global, 48 kHz stereo paced | 0.654241 | 0.859057 | 1.191667 | 0/1,000 | 31,244,288 B |
| Band, 48 kHz stereo paced | 0.916808 | 1.110338 | 1.218834 | 2/1,000 | 30,277,632 B |

Compute includes causal frontend, ONNX inference and DSP. All measured calls
remain below 5 ms and below their hop period; virtual compute backlog is zero.
The global maximum across all cases is 1.191667 ms; band maximum is 1.218834 ms.
Band peak native RSS across all cases is 30,785,536 B (29.359 MiB).
The paced band process consumes 1.00 s user + 0.01 s system CPU over 2.98 s
wall time, about 33.9% of one core including startup/warmup/scheduling. This is
process CPU accounting, not inference-only CPU. Fresh Python/ORT-only RSS
snapshots are separately retained and are not the native peak.

**Paced continuity fails:** band absolute hops 646 and 647 complete 3.622951
and 1.442367 ms late; their compute is only 0.638292 and 0.485041 ms. Finish
lateness plus the hop period minus compute leaves approximately 5.651326 and
3.623993 ms outside measured compute since the respective release times.
This can include wake/scheduling delay and post-compute bookkeeping/preemption;
direct wake-time/thread-CPU instrumentation was disabled in this cost protocol.
It does not isolate late starts, prove an OS root cause or qualify real-device
reliability. Do not rerun until these failures
disappear or merge this result with the global control's zero lateness.

Fixed buffering is still **256 frames = 5.333 ms at 48 kHz / 5.805 ms at 44.1 kHz**,
separate from compute, with no new lookahead. Device/driver latency and worker
availability remain additional deployment concerns. A short synthetic replay
does not establish continuous listening without clicks or underruns.

## Evidence and next gate

Local root: `data/runs/band-tcn-p62-20261026/`. Registration, timing registration,
frozen source/binary, both stress fixtures, all traces, native PCM, time/RSS logs,
summary and receipt remain ignored. All **549 artifact hashes** were verified.

- Band ONNX: `e67f1a30be507a940f04c4907bcf9eff7078a97ce668c434ec75beab207ee6e3`.
- Global ONNX: `0c8cbcaf863e80bd7cd2def0c0ffd01bfc3a90cd12e6d9ba0432fbce54411cb8`.
- Summary: `c38250e2b52eec9323fd101f186b9e746de84eb2abc174847955f19d572199fc`.

Compute feasibility permits a **training experiment**, not production promotion.
Before launch, register fresh controls and band candidates, both seeds, common
source/crop exposure, sufficient equal update budgets, fixed endpoints and the
unchanged original-reference objective. Representation and capacity must be
reported separately; historical P57 remains a reference, not an equal-budget
fresh architecture control. Freeze every endpoint before uniform raw/native
spectral, identity, dynamics and source-equal rechecks. Reserve unseen recordings
through the evaluation catalogs before any independent test. Investigate paced
wake delays and real worker recovery separately; neither failed scheduling nor
a promising training loss can be hidden by mean compute or mean spectral gains.
