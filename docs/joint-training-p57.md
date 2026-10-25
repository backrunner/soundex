# P57: matched joint training of the existing small model

## Hypothesis and fixed comparison

[P56](trainability-audit-p56.md) found omitted reconstruction directions in frozen
features after complete P54 adapter training. Decoder-only unfreezing changed
little; removing identity protection or increasing learning rate introduced
additional errors. Tiny one-step mean improvements, some near float32 resolution,
do not establish useful quality or a capacity ceiling.

P57 tests whether **cumulative joint adaptation of the existing features and
reconstruction heads** transfers better than adapter-only continuation. No new
parameters, inference operators, context, lookahead or buffering are introduced.
Both scopes use the same 480,809-parameter causal model. Both completed P54
common-loss seeds participate; neither is selected by P55 ranking.

| Setting | Adapter control | Full reconstruction |
| --- | --- | --- |
| Starting weights | Same P54 common endpoint for each seed | Same |
| Trainable parameters | 49,794 | 480,809 |
| Adam state | Fresh, then persistent | Same |
| Learning rate | Fixed 1e-7 | Same |
| Betas / epsilon / clipping | .9/.999 / 1e-8 / norm five | Same |
| Budget | 4096 actual updates per seed | Same |
| Update sampling | Four batches, each four music + one speech with clean identities | Same |
| Objective | Existing common direct spectrum + raw MR, equal paired/identity risk | Same |

The full scope reconnects the adapter to live features. Its budget head receives
original-reference reconstruction gradients, with no DSP/budget-teacher auxiliary.
Phase, envelope and true temporal-change terms remain. No loss weight, hard
threshold, gradient veto or post-Adam displacement multiplier changes. Fixed
evaluation mode prevents training-statistic drift; real warmup is outside
supervision and detached in backpropagation.

Fresh Adam is matched in both scopes because P56 observed a material optimizer-
history effect. The old 1e-7 rate remains; this is not another rate search or a
claim that identity protection can be discarded. The 4096-update budget matches
P54's continuation and tests cumulative effects with broader exposure, not
convergence or architecture superiority. No intermediate winner or budget cut.

## Data and reservation

Reuse the entire registered P54 optimization schedule: 81,920 paired crops per
trajectory, 278 recordings and 287.52 unique MR-supervised minutes after merging
source intervals and removing warmup/overlap. This is scheduled exposure until
the corresponding updates finish. Four trajectories do not multiply unique
audio duration. Source-time, codec, physical sample rate and mono/channel-view
assignments remain fixed. These independent views are not stereo training.

Before launch, verify every selected clean/degraded file checksum and exclude
all registered `evaluation-catalog.jsonl` file/PCM/work aliases using transitive
bridges through processed manifests and the approved catalog. P55's 20 recordings
remain excluded. The registered TRAIN calibration and historical review sets
are not used for optimizer updates. Dataset acquisition, licensing and defaults
do not change.

## Completion, review and runtime

Entry point: `training/scripts/run_joint_training.py`. Local evidence:
`data/runs/joint-training-p57-20261025/`. Registration binds sources, snapshots,
inputs, parents, sampling, review controls, loss code and listening template.
Checkpoints preserve exact moments, parameter scope, history and signature;
parameter/statistic isolation and resumed-step equality are tested.

Finish all four endpoints before comparative review, freeze their hashes, then:

1. Recheck all endpoints and both starts on 32 historical TRAIN music/speech
   batches with the common raw scores and retained per-example errors.
2. Export and verify causal ONNX/state parity. Render the eight evenly spaced
   native batches through the frozen Rust DSP, also retaining historical and
   identity-DSP controls. This adds speech rechecks absent from P55's music set.
3. Measure all four endpoints serially: 3000 hops for each 44.1/48 kHz mono/stereo
   case and 1000 paced 48 kHz stereo hops, totaling 52,000 measured hops. Record
   model+DSP compute, CPU/RSS, service and paced failures; no retry-to-pass.
4. Freeze all eight graph hashes before the complete 160-case P55 stereo recheck:
   four new endpoints, two exact parents, historical weights and identity DSP.
   Use unchanged original references, source-equal scores, strata and source
   bootstrap. Generate a new blinded comparison pack without invented ratings.

P55 recordings are now exposed: this is explicitly a **recheck**, not a new
independent test, even though training remains isolated. The reused renderer
retains its P55 metric/listening protocol; P57 registration and status identify
the new candidate set and recheck scope. Every adverse source/band/codec/identity
result remains visible. No automatic best-checkpoint selection or promotion.

The fixed 256-frame production buffer remains 5.333 ms at 48 kHz, separate from
the <5 ms model+DSP compute target and the 2.667 ms hop service period. Unchanged
graph size is not measured real-time qualification; actual-device and human
listening evidence remain separate requirements.

## Completed results

All four trajectories completed **4096 actual updates each**, with finite update
telemetry and verified final checkpoint hashes. All registered raw/native reviews,
52,000 diagnostic timing hops and the 160-case P55 stereo recheck completed.
The long-form recheck contains 2,720 score records; the completion receipts and
summary/endpoints hashes verify. No default weight changed. This is a completed
experiment, not release qualification or a new independent test.

The following numbers give each of the 20 reused source recordings equal weight.
Lower error is better. Both seeds remain reported; no winner is selected.

| Native paired metric | Seed 12 adapter | Seed 12 full | Seed 13 adapter | Seed 13 full | Historical |
| --- | ---: | ---: | ---: | ---: | ---: |
| MR | 0.51186562 | 0.50867788 | 0.51092117 | 0.50766732 | 0.49981362 |
| HF8k MR | 0.57018818 | 0.56568336 | 0.56885821 | 0.56425484 | 0.55207529 |
| Log-magnitude error | 0.34405320 | 0.33922517 | 0.34307263 | 0.33826258 | 0.31940622 |
| Reference dynamics error | 0.26451699 | 0.26490707 | 0.26460560 | 0.26521475 | 0.26879708 |

Compared with the matched adapter controls, full training reduces native MR by
**0.6228% / 0.6369%**, HF8k MR by **0.7901% / 0.8092%**, and log-magnitude error
by 1.4033% / 1.4020%. MR improves on 16/20 and 14/20 sources, leaving four and
six adverse sources respectively. Compared with the exact P54 parents, full
training reduces MR by 0.6441% / 0.6448%; continuing only the adapter reduces it
by about 0.0215% / 0.0080%. This supports cumulative trainability as a useful
mechanism under this protocol, not a universal architecture or capacity verdict.

Full endpoints still have **1.7735% / 1.5713% higher MR** and **2.4649% / 2.2061%
higher HF8k MR** than historical weights. Their native MR differences versus
historical weights have descriptive marginal source-bootstrap 95% intervals
[0.001509, 0.015729] and [0.000811, 0.014294]. They remain worse on 12/20 and
13/20 sources. These exposed-source intervals are not multiplicity-adjusted
confirmatory evidence or a perceptual rating. Against untreated audio, full
endpoints reduce MR by 2.0301% / 2.2248% and HF8k MR by 2.5517% / 2.7978%.

### Fidelity and counterexamples

Full training increases native reference dynamics error by 0.1475% / 0.2302%
relative to adapter controls, and paired full-band PCM RMS error by 0.0288% /
0.0322%. A small waveform increase does not alone reject spectral progress,
but the temporal and clean-input tradeoffs must remain visible.

On clean identities, native MR rises 5.6649% / 7.2446%. Calibrated full-band PCM
error RMS changes from 0.000838123 to 0.000841751 for seed 12, and 0.000842899 to
0.000848820 for seed 13: increases of **3.629e-6 / 5.922e-6 PCM**, or 0.4330% /
0.7025%. Native HF8k error RMS rises 2.4502% / 3.3705%; below-8-kHz RMS changes
only 0.0020% / 0.0045%. These are errors against original references, not output
loudness or audible-severity thresholds. Neither large relative weak-band errors
nor small absolute changes replace listening evidence.

The historical TRAIN raw recheck also favors joint training on paired spectra:
music MR falls 0.9694% / 0.7619% and speech MR falls 0.5047% / 0.5863% versus
adapter controls. Raw music identity error increases, and phase-sensitive high-band
PCM tradeoffs remain. Full per-case errors are retained in `raw-review/`,
`native-review/` and `longform-recheck/summary.json`; frequency/codec/rate strata
remain in `longform-recheck/strata.json`. These are rechecks, not new validation
recordings. Blinded materials exist; human ratings remain pending.

### Runtime and validation

Across **52,000 measured hops**, maximum frontend + model + DSP compute is
**2.238792 ms**, with zero >=5 ms calls and zero service durations exceeding the
hop period. Paced 48 kHz stereo p99 is 0.807682-0.904797 ms across the four models.
Process peak RSS is **31,490,048 bytes** (about 30.0 MiB).

One paced completion is late by **0.468495 ms**, on seed 13 full. Thus compute
passes this diagnostic but continuous timing is not failure-free; do not erase
the event, attribute it to scheduling without evidence, or retry until it passes.
The fixed 256-frame buffer remains 5.333 ms at 48 kHz and is excluded from compute.
This host simulation is not actual callback/device certification.

All prelaunch checks passed: 773 tests, Ruff on 349 Python files, 135 English
Markdown documents, and CI 40168191632. The real-size preflight models retained
480,809 parameters and 1,972,748 ONNX bytes, with 320-hop state/reset/silence parity.
No inference-size increase was needed for the observed spectral gain.

### Decision

Joint training makes more progress than unchanged adapter continuation, but has
not surpassed historical spectral restoration and adds preservation/temporal
tradeoffs. Do not promote, remove identity protection or expand capacity on this
result. The next diagnosis should decompose the accumulated joint updates by
frequency band, source and training stage, separating useful high-band recovery
from clean-input and transient damage. Use TRAIN-only decisions and a matched
full-training control before changing constraints or objectives. Inspect the
paced-lateness event separately from quality. Any further P55 scoring remains
a recheck; fresh generalization and human/device evidence are still absent.
