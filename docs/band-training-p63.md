# P63: matched fresh band-TCN training and deadline attribution

Date: 2026-10-26; final automated review checked 2026-10-31. All six training
endpoints, short reviews, the original cost grid and the complete long-form
recheck are finished and receipt-verified. Storage-only recovery completed on
2026-10-30 at 05:07:59 UTC. Band TCN meets the registered spectral criterion for
further study, with fidelity tradeoffs and failed real-time criteria. Default
weights remain unchanged. This follows [P62 feasibility](band-tcn-p62.md).

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


## Completed long-form recheck (verified 2026-10-31)

The recovered review completed all **160 case groups / 1,600 model renders**:
20 music sources, two excerpts, three compressed codecs and clean identity. There
are 120 paired and 40 identity cases per path, and 3,360 records including raw,
native and untreated paths. These are previously exposed P55 recordings, not new
independent tests. All original-reference comparisons below average within each
source before equally averaging the 20 sources; they do not count codecs or
excerpts as independent recordings.

| Relative reduction in native paired error | Seed 20261012 | Seed 20261013 |
| --- | ---: | ---: |
| MR versus fresh global width 96 | 1.76% | 1.88% |
| MR versus fresh global width 46 | 2.86% | 2.84% |
| MR versus historical legacy | 2.67% | 2.93% |
| MR versus untreated compressed input | 6.31% | 6.55% |
| HF above 8 kHz MR versus matched global | 3.64% | 3.62% |
| HF above 8 kHz MR versus untreated | 7.94% | 8.25% |

Native paired MR is 0.48645490 / 0.48518625 for band, 0.49515721 /
0.49446078 for wide global, and 0.50079350 / 0.49938264 for matched global.
Legacy is 0.49981362, identity-through-DSP 0.51721536 and untreated 0.51921873.
Band also improves HF log-shape and envelope against matched global by
7.84% / 7.72% and 5.69% / 6.13%, respectively. Raw model-output MR improves
3.50% / 3.54% against the matched raw controls; the native table confirms that
part of this spectral gain survives actual DSP. Raw and native values remain
separate; neither alone establishes listening quality.

The registered native music MR criterion for **further study** is met against
both fresh controls in both seeds. Source-level MR improves on 20/20 recordings
against wide global in each seed, and 20/20 and 19/20 against matched global.
It improves on 19/20 and 18/20 against legacy. This is evidence for the current
band representation under the registered budget, not an architecture-family
ceiling, convergence proof or automatic promotion.

Fidelity tradeoffs remain. Against matched global, native HF PCM error RMS
increases 1.31% / 1.58%, full PCM error RMS increases 0.24% / 0.29%, and temporal
error increases 0.30% / 0.39%. HF PCM error RMS is 0.00888867 / 0.00891259 versus
0.00877412 / 0.00877418 for matched controls. Against untreated input, HF PCM
error RMS increases 1.88% / 2.16%, despite better HF spectral shape. Low-band
PCM RMS changes versus matched controls are +0.0016% / +0.0019%; stereo side
error RMS changes are +0.11% / +0.13%. These calibrated absolute PCM errors and
all individual cases are retained; neither small relative changes nor magnitude
improvement prove inaudibility or phase restoration.

Clean identity MR is 0.01503561 / 0.01499772, better than matched controls
(0.01580121 / 0.01608939) and legacy (0.06663333), but worse than identity DSP
(0.01416304). The shared limiter's reference conflict is not removed from scores.
The original lossless reference remains the target.

Codec-stratified native MR versus matched controls is -5.57% / -5.50% for MP3,
-1.41% / -1.18% for Vorbis, and **+0.142% / -0.085% for AAC**. The pooled gain
must not be described as uniform codec superiority. Retained source MR exceptions:

- `Gurdonark - Cave` (`08a1849f...`): seed 20261013 versus matched global,
  delta +0.00051835.
- `reaching` (`f18318fa...`): both seeds versus legacy,
  deltas +0.00462181 / +0.00124583.
- `Dust Time Gravity` (`1f4ac506...`): seed 20261013 versus legacy,
  delta +0.00003411.
- Source `d2c0b7d6...`: seed 20261013 versus untreated,
  delta +0.00013069 (versus identity DSP +0.00013087).

All metric/source deltas, including absolute PCM and clean identity, are retained
in `assessment-20261031.json`; its inputs are SHA-bound to the immutable summary,
strata, completion and cost reports. Full per-case values remain in
`longform-recheck/summary.json`. Existing paired-source bootstrap legacy MR delta
intervals are [-0.01823017, -0.00908916] / [-0.01959489, -0.01034899], descriptive
marginal 95% intervals without multiplicity adjustment. Do not infer universal
superiority from them or retrofit acceptance thresholds to the counterexamples.

Coverage is folk 1, metal 2, pop 2, rock 3, ambient 4, electronic 4 and unlabeled 4;
all 20 region labels are unknown. This recheck cannot establish China/Japan/Korea
coverage, broad speech generalization or artist independence. Twenty blinded
listening trials were generated at `longform-recheck/listening/index.html`;
no listener ratings or actual-device evidence were obtained.

Runtime results were **not rerun**: all 78,000 original measured hops and their
failures remain intact, including the 12.734208 ms band call and 26 / 17 paced
late completions. The 5 ms compute and per-hop deadline criteria remain failed.
The earlier separate 24.131125 ms audit outlier also remains retained. Completion
of this review is not real-time or release qualification.

Both root and recovery completion receipts verify; the long-form receipt binds
1,889 reports/receipts/listening files. Recovery reverified all 91,236 original
inputs and all 3,387 protected files before writing completion. The original
StorageFull failure and external backup remain. No new recovery failures occurred.
The next work should isolate the measured codec/fidelity tradeoff and scheduling
failures, with fixed controls and preserved endpoints, rather than repeat the
same training or scale parameters without a changed hypothesis.

## Completed training and partial review (2026-10-29)

All six trajectories completed 16,384 updates (98,304 total). All six endpoint
files match `endpoints.json`. The supervisor exited; the old
`reviewing-frozen-endpoints` progress marker does not mean a live evaluation.
No whole-run completion receipt exists.

The completed short native review contains 32 paired music clips from 25 source
IDs. Averaging within source, then equally across sources, gives original-reference
multiresolution magnitude error (lower is better):

| Path | Seed 20261012 | Seed 20261013 |
| --- | ---: | ---: |
| Band TCN | 0.35402546 | 0.35261912 |
| Global width 96 | 0.35968938 | 0.35841440 |
| Global width 46 | 0.36032694 | 0.35983993 |
| Historical `legacy` control | 0.36693787 | same frozen control |
| Identity through DSP | 0.37029337 | same frozen control |
| Untreated compressed input | 0.36731059 | same input |

Band MR decreases by 1.57% / 1.62% against the wide controls and 1.75% / 2.01%
against the parameter-matched controls. Against legacy it decreases by 3.52% /
3.90%. This is a positive short-review result, not the complete registered review
or independent testing. Raw paired-music MR clip means are 0.39457177 /
0.39489323 for band, versus 0.40094266 / 0.40144154 for wide global and
0.40329260 / 0.40269174 for matched global; those raw means are clip-weighted,
not the source-equal native table above.

Relative to matched global, band's native HF PCM error RMS increases by 0.62% /
0.84%, temporal MR error by 0.96% / 0.92%, and full PCM error RMS by 0.16% /
0.21%. Preserved-band RMS is effectively unchanged. MR improves on 19 / 18
of 25 sources against matched global, worsens on 5 / 6, and ties on one in each
seed. Worst matched-control MR deltas are +0.00798944 / +0.00180644 on
`music_library:kcc-original-13300270-59099f52e73fb106`. Against legacy, 21 / 22
sources improve, 3 / 2 worsen and one ties. All individual scores remain in the
native reports; these percentages do not establish audibility or overall quality.

All 30 fixed cost runs completed, totaling 78,000 measured hops. On 48 kHz stereo
paced band runs, p99 stage wall time is 1.68246 / 1.59030 ms, maximum 12.73421 /
1.78775 ms, calls above 5 ms number 3 / 0, and late completions number 26 / 17
out of 1,000 scored hops per seed. Peak process RSS is 28.94 MiB for each paced
band run. Global controls also have late completions and >5 ms outliers. The
compute and deadline criteria remain failed; low typical latency cannot erase
these failures. Fixed buffering remains 256 frames (5.333 ms at 48 kHz), excluded
from compute. These are synthetic host measurements, not real-device evidence.

The long-form P55 recheck completed nine of 160 case groups before native rendering
of `20261013-band/01-0-aac-lc-cbr-96` failed with `StorageFull` at
2026-10-28 19:09:55 UTC. At inspection the external volume has only 32 MiB free.
This is an infrastructure failure, not a model-quality rejection. Preserve
`failure-1793214595230516000.json`, the failed render, all successful render receipts,
source snapshots, training inputs, endpoints and existing timing evidence. The
run occupies about 2.2 GiB, including 1.7 GiB of partial long-form evidence; it
alone does not explain the full 1.8 TiB volume.

Recovery requires sufficient storage, verified preservation of existing artifacts,
and an explicit continuation of only the incomplete long-form review. Do not
rerun training or replace the completed cost grid with fresh timings. The failed
render must be archived rather than silently overwritten; the renderer deliberately
rejects incomplete populated directories. Do not extrapolate nine case groups to
the complete 20-source P55 recheck. No cleanup or recovery was performed during
this status inspection.

Evidence below the corrected run root: `endpoints.json`, `raw-summary.json`,
`native-review/summary.json` and each hash-bound `report.json`, `cost-summary.json`,
`longform-recheck/progress.json`, and the retained failure JSON/log. No default
weights were replaced, no new training was launched, and listening is unverified.

## Storage-only recovery (2026-10-30)

This is a continuation of the same frozen review, not a new quality experiment.
No model, loss, case, scoring, selection or timing protocol changed. The external
volume had recovered to about 12 GiB free but remained too small for a conservative
remaining-output allowance; the internal volume had about 373 GiB free.

Before relocation, the exclusive experiment lock confirmed no active owner.
The 671-file long-form tree (1,780,034,756 logical bytes) was inventoried and copied
to the internal volume. Every relative path, byte size and SHA-256 matched before
the logical path was switched. The original external tree remains at
`longform-recheck-before-recovery-20261030/`. The original `longform-recheck/` path
is now a symlink to
`/Users/orchiliao/Library/Application Support/SoundEx/experiments/band-training-p63-20261026-v2/longform-recheck`.
This preserves absolute paths embedded in the existing scoring receipts. No
training data, failed evidence or cache was deleted; **reclaimed bytes: zero**.

All 95 successful model/case renders are retained. The incomplete render (empty
`trace.json` and its StorageFull stderr) was moved to
`longform-recheck/failed-renders/storagefull-1793214595230516000/20261013-band/01-0-aac-lc-cbr-96/`,
and is also intact in the external backup. Only that infrastructure-failed render
and subsequent missing renders may execute; successful renders are receipt-checked
and reused. The completed 78,000-hop timing grid must not be repeated or replaced.

`recovery-20261030/` contains the migration inventory, protected-file manifest,
registered continuation script and hash, supervisor, progress and retained errors.
The manifest pins 3,387 existing checkpoint/export/raw/native/timing/log files.
The continuation uses the original frozen Python implementation, verifies all
91,236 original input/source/resource hashes, registered package versions, catalog
reservations, all six full-budget endpoint signatures and the existing render
receipts before scoring. It rechecks original inputs and protected evidence before
writing completion receipts. At least 64 GiB free on the destination is required
at review entry. It resumes scoring in the original case order, then generates
all-endpoint blinded materials; listening ratings remain pending.

Launch PID 71250 and caffeinate PID 71251 are recorded in
`recovery-20261030/supervisor.json`; consult that file and current process state
rather than assuming a recorded PID is still live. Progress during rendering is
in `longform-recheck/progress.json`; startup integrity verification is in
`recovery-20261030/progress.json`. Completion still requires the final receipts.

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

The original training launcher also reruns finalization and timing, so do not use
it to resume this partially completed review. After checking the recovery supervisor
and retained failures, the storage-only continuation is:

```sh
rtk proxy /Volumes/BRData/projects/soundex/training/.venv/bin/python -u \
  /Volumes/BRData/projects/soundex/data/runs/band-training-p63-20261026-v2/recovery-20261030/resume.py
```

The exclusive experiment lock rejects duplicate execution. New failures stop with
a retained traceback; investigate them before another launch. Do not automatically
retry incomplete populated render directories or incomplete blinded bundles.
