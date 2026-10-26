# P63: matched fresh band-TCN training and deadline attribution

Date: 2026-10-26. Status: training running; no completed endpoint or quality
comparison yet. Default weights remain unchanged. This follows
[P62 feasibility](band-tcn-p62.md), whose numerical/native checks and remote CI
completed successfully while paced continuity remained unqualified.

## Question and fixed comparison

Retaining local frequency state before global readout has not been tested by
the previous global-width expansion or critic studies. Test that representation
using fresh models and unchanged original-reference supervision:

| Arm | Parameters | Float state per physical channel | Purpose |
| --- | ---: | ---: | --- |
| Global width 96 | 480,809 | 6,048 | Fresh control for the recent global architecture |
| Global width 46 | 170,859 | 2,898 | Approximately parameter-matched capacity control |
| Eight-band TCN, token width 24 | 171,137 | 12,096 | Local frequency features/history before width-96 readout |

The smaller global control differs from the band graph by 278 parameters
(about 0.16%). Matching weight count does not match compute, state size, head
width or parameter allocation. Report both controls; do not claim a perfectly
isolated causal effect of frequency tokens or an architecture-family ceiling.

Two fixed seeds, 20261012 and 20261013, give **six trajectories**. All begin at
the same identity function with zero residual heads, without historical weights
or copied Adam state. Each receives **16,384 updates** and four paired/identity
batches per update. Execution is serial in registered seed/arm order. All six
endpoints must finish and pass signature/step/hash checks before comparative
evaluation; no intermediate winner or budget shortening is permitted.

Fresh training uses Adam (0.9/0.999, epsilon 1e-8), gradient norm clipping at 5,
and one fixed update-index schedule: 256 updates ramp from 1e-5 to 1e-4, then
cosine decay to 1e-5 at update 16,384. Unlike P57's tiny-rate continuation, these
models must learn features from scratch. This budget is comparable to earlier
full fresh causal studies and quadruples scheduled crop exposure relative to
the 4,096-update continuation. It is not proof of convergence or optimal tuning
for each architecture. Report late training trends after all endpoints finish.

Loss stays unchanged: direct original-reference spectra plus raw-OLA
multiresolution spectra, `log1p` reduction, and equal paired/clean-identity means.
Magnitude/shape, energy and reference-following dynamics remain primary, with
complex/retained-band fidelity. There is no critic, DSP-generated teacher,
phase-input intervention or retrospective threshold. Eighty hops of actual
history are detached; supervised support starts at frame 89. No warmup updates
frozen statistics or enters the supervised-region counts.

## Source isolation and scheduled coverage

Use the existing approved native-lossless paired manifests and the P54
optimization recording role. Exclude all current evaluation catalogs and the
fixed review recordings through connected source-file, decoded-PCM and work
aliases **before sampling**. This removes 96 prior-role speech utterances
sharing reserved `vctk:<speaker>` work identifiers. They are not independently
new speech just because their utterance/file identifiers differ. This new
isolation does not retroactively make historical speech reviews independent.

The resulting schedule contains **157 music recordings and 25 speech recordings**,
not the entire downloaded library. Every trajectory receives the same newly
drawn 327,680 paired crops plus their clean identity views. The previous
diagnostic 4:1 music/speech mixture is fixed for this comparison; it is not a
new product-wide corpus-ratio or genre-balance policy.

Scheduled unique original-source MR supervision, merging overlaps/codecs/rates/
channel views and removing warmup, is **27,627.408 s (460.457 minutes)**. Direct
analysis support is 27,700.135 s. These are scheduled intervals, not already
trained exposure, six times as much independent data, or complete song durations.
`progress.json` reports completed paired crops per running trajectory. Checkpoints
and the immutable schedule allow exact consumed-prefix reconstruction.

Coverage includes nine MP3/AAC/Vorbis settings, mono/left/right/mid/side views,
and metadata for ambient, blues, classical, country, electronic, folk, hip-hop,
jazz, pop and rock. There are 30,056 unlabeled music draws and 28,383 music draws
with unknown region. China, Japan, Korea, Europe and North America all appear,
but regional metadata does not establish equal coverage or listening quality.
Physical stereo playback is reviewed separately; independently drawn channel
views are not assembled into fictitious stereo training pairs. Speech coverage
is reduced by the conservative speaker-alias exclusion and must be reported
separately from music. The remaining speech represents only three speakers
(p233/p267/p340); it cannot establish broad speaker generalization.

## Preflight, checkpointing and assessment

Before the full budget, run two genuine Adam updates for every arm/seed on the
same first eight TRAIN batches. Export those nonzero diagnostic models and
check actual Rust predictions/PCM. Discard every preflight model/optimizer and
restart fresh for the full budget. These twelve diagnostic updates neither
select a rate nor qualify quality. Nonfinite or contract failure stops the run
with its artifacts retained.

Save persistent Adam state and model/history hashes after the first update,
every 32 updates and at the endpoint. Resume validates registration, packages,
source/data hashes, reservations and checkpoint receipts. The execution source
is a committed Git archive, including required non-Python resources; later
workspace edits cannot silently alter the running experiment.

After the endpoint barrier, the shared P57 review pipeline is called with an
explicit P63 checkpoint loader/exporter. Its default P57 behavior is unchanged.
Review original-reference raw and native DSP spectra/fidelity, then all trained
endpoints' native latency/CPU/RSS and the complete P55 stereo recheck. Controls
include untreated input in the review, identity-through-DSP, historical weights
and both frozen P57 full-model graphs. P55 is exposed and remains a **recheck**.
Listening materials are not listener ratings. No fresh independent-test or
actual-device qualification is claimed.

The registered criterion for continuing this band direction requires paired
music native MR improvement over both fresh controls in both seeds, using
source-equal aggregation. Report HF bands, energy, dynamics, clean identity,
absolute PCM errors and every adverse source alongside that criterion. A mean
MR improvement alone cannot promote weights, and failure cannot establish the
best possible result for every band/TCN architecture.

## Fixed deadline-attribution follow-up

Before training, `audit_band_deadlines.py` completed eight registered 1,000-hop
paced runs, alternating CPU/wake probes off/on and reversing the order in the
second round. It reused P62's exact binary, nonzero graphs and 48 kHz stereo
stimulus. No SoundEx training/tests/build ran concurrently. Ordinary host activity
was uncontrolled; the probe is not a production scheduling change.

| Graph | Probe | Late completions, rounds 1 / 2 | Maximum stage wall time, ms |
| --- | --- | --- | ---: |
| Global | off | 0 / 0 | 1.158625 |
| Global | on | 0 / 2 | 4.103584 |
| Band | off | 0 / 0 | 1.776250 |
| Band | on | 15 / 3 | 24.131125 |

The worst band hop uses **1.108583 ms thread CPU** inside **24.136583 ms enclosing
wall time**, leaving 23.028 ms without measured thread CPU progress. It finishes
22.074229 ms late and generates a backlog across subsequent hops. Another band
outlier combines 3.895358 ms late wake with 4.687834 ms stage wall time. The
global outlier uses 0.566417 ms thread CPU inside 4.106750 ms enclosing wall time.
These measurements support substantial off-CPU delay; they do not identify the
OS subsystem, exclude probe-induced scheduling changes, or prove normal runtime
always has only these causes. Probe calibration maxima are 0.000625-0.001500 ms;
no overhead is subtracted from observed failures. Output PCM is exactly unchanged.

**Real-time acceptance remains failed**, including one measured call above 5 ms.
The original P62 failures are retained and are not replaced by the zero-lateness
probe-off runs. Continue quality training separately; before deployment, verify
worker scheduling and fixed-timeline fallback/recovery under missed deadlines,
including real-device runs. Fixed 256-frame buffering remains separate.

## Evidence and retained launch failure

- First registration: `data/runs/band-training-p63-20261026/`, source `d574172`.
  The launcher stopped before hashing data or executing an update because
  `Path` ordering and whole-string ordering differ for a directory prefix and
  its `-v2` sibling. The catalog set did not change. Its registration, schedule,
  traceback and supervisor log are retained.
- Corrected registration: `data/runs/band-training-p63-20261026-v2/`, source
  `3037c00`. The comparison now ignores order while still rejecting additions
  or removals, with an explicit regression test. Budgets, seeds, crop-generation
  seed and all numerical model/loss behavior are unchanged.
- Deadline evidence: `data/runs/band-deadlines-p63-20261026/`, including all eight
  raw traces, CPU/wake outliers, output PCM and hashed receipt.
- Full local regression: **829 Python tests passed**, 259 warnings, 68.04 s;
  Ruff and formatting pass. Tests cover actual Adam resume/feature changes,
  nonzero export reuse, endpoint barriers, source-alias closure and reservation
  ordering. Registration is not a completed training result.
- Both registrations have byte-identical schedules, SHA-256
  `25a56a7de4b273708911dc534ea73c70aba493712f51a6868386c6250422e106`.
  The corrected registration binds **91,236** source/data/resource hashes.
- Corrected launch: all 91,236 hashes verified; all six two-update preflights
  and their nonzero native export/PCM checks completed. Formal training then
  started from a fresh global seed-20261012 model and saved its first checkpoint.
  The band arms remain queued in the registered serial order. The supervisor
  and `caffeinate` processes were verified live; this confirms launch and
  checkpointing, not completion or generalization. All six preflight receipts
  were independently verified after launch.

To resume the corrected experiment, use its frozen source and interpreter:

```sh
cd data/runs/band-training-p63-20261026-v2/source/training
rtk proxy /Volumes/BRData/projects/soundex/training/.venv/bin/python -u \
  -m scripts.run_band_training run \
  --output /Volumes/BRData/projects/soundex/data/runs/band-training-p63-20261026-v2
```

The exclusive run lock rejects duplicate execution. Check the saved supervisor
process and progress first; do not launch a second copy of an active job.
