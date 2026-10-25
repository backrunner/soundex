# P60: observed-band context and structured critic negatives

Date: 2026-10-26. This follows [P59](texture-calibration-p59.md), whose classifier
failed phase/time controls although paired feature loss penalized them. The
changed hypothesis separates missing cross-band conditioning from limited
negative exposure. Longer unchanged training would not answer that question.

## Fixed protocol before training

Four arms form a 2x2 factorial, repeated against both frozen P57 generators.
Every critic receives 1,024 updates, identical initialization, fresh Adam at
1e-4, betas 0.9/0.999, epsilon 1e-8 and gradient clipping at 5. All eight
endpoints finish and freeze before scoring. No generator update, intermediate
winner, budget reduction, new dataset or default-weight change.

| Arm | Observed full-band summaries | Structured negatives |
| --- | --- | --- |
| high_prediction | Off | Off |
| context_prediction | On | Off |
| high_structured | Off | On |
| context_structured | On | On |

All arms allocate 3,970 parameters. The first convolution retains the original
three inputs and adds eight context channels. Added weights start at zero;
initial predictions match the original 2,818-parameter critic within numerical
tolerance. Off arms supply zeros, so effective active capacity differs
intentionally. This controls allocated size, not effective expressive capacity.
The original critic stays intact.

Simply unmasking low frequencies would leave them outside the high-band local
receptive field. Instead, eight observed log-magnitude averages in physical
3 kHz bands from 0 to 24 kHz are broadcast to every candidate frequency at the
same analysis time. This is compressed full-band context, not full spectral
detail. Normalization uses detached observed audio only. Noncentered 512/1024
analysis windows stay inside supervised support. No batch statistics are used;
symmetric training convolutions are never deployed.

Structured arms use total fake weight one: prediction 1/2, phase-scrambled
reference 1/4 and shifted-HF reference 1/4. The phase seed changes per update;
time shifts cycle 1/8, 1/6 and 1/3 of the excerpt. A synthetic transform whose
PCM difference does not exceed reference-roundtrip numerical error is omitted
per example and weights renormalized. This prevents identical silent references
being introduced as synthetic errors; it is not an audibility gate. Control
arms repeat prediction in all three slots with the same total fake weight and
forward-pass count. DC/Nyquist and frequencies below 8 kHz remain unchanged by
synthetic phase/time transforms. Circular FFT boundaries remain diagnostic.

## Sources, isolation and controls

Reuse the exact P59 source/work-isolated 256-draw fit bank, including codec,
physical channel view, warmup and supervision: 156 music recordings, 49 speech
recordings and 554.812 unique supervised seconds. Four cycles do not multiply
independent duration. Every update retains four music crops and one speech crop.
Calibration reuses 45 music and 16 speech recordings with 36.823 supervised
seconds. Original P59 genre, region and stereo coverage limitations remain.

Verify the parent completion receipt, all input/source hashes and all 512 cached
fixed-generator PCM receipts before fitting. Recheck current reserved evaluation
catalogs and connected work aliases. Critic fit/calibration remain disjoint;
P59 calibration is exposed, so this is a recheck, not an independent test of
critics or historically trained restoration generators.

After all eight critics freeze, score the original 16 fit and 16 calibration
draws for initial random and trained features. Retain every source/corpus and
all P59 prediction, observation, energy-matched, phase, quarter-shift and
reference-roundtrip controls. Add a 3/8 time shift never used in fitting.
Quarter-shift is also excluded from training fractions. Recognition of these
constructed errors cannot establish real restoration quality.

The P59 prerequisite remains per seed and corpus: ordinary prediction logistic
loss improves over initialization and the constant classifier; each required
control has a positive source-mean margin and strict source majority beyond
roundtrip error. Add the same requirement for the untrained 3/8 shift. An arm
must pass for both generators and both corpora. No score-selected thresholds
or loss coefficients; passing does not establish qualified restoration weights.

Report paired feature distances separately from classifier margins. Normalize
each initial/trained critic's feature distances by its own source-equal fit
prediction mean per corpus, without calibration, to expose simple rescaling.
Retain unnormalized values and calibrated absolute PCM band errors. Shift
observed conditioning in time while preserving samples/RMS to probe dependence;
a response alone is not proof of beneficial conditioning.

## Implementation and external-method boundary

Original implementation: `training/models/context_texture.py` and
`training/scripts/calibrate_texture_context.py`. Tests cover initial equivalence
and allocated capacity, physical/time support, a constructive low-band path to
distant high-band logits, context gradients, reference stationarity, silence,
weak signals, detached targets/observations, finite differences and preserved
FFT magnitudes under time shift.

[AERO](https://arxiv.org/abs/2211.12232) and
[Apollo](https://arxiv.org/abs/2409.08514) motivate examining spectral features
and band relationships; see the [official implementation review](implementation-review-20261026.md).
They do not establish that our small frozen-output critic or constructed
negatives improve SoundEx. No external code, weights or audio are imported.
This is an independent mechanism study, not a reproduction or SOTA claim.

Registration, snapshots, optimizer state every 64 updates, initial/final critics,
source reports and hashes remain under ignored
`data/runs/texture-context-p60-20261026/`. Generator tensors are checked unchanged.
No new inference operator, buffer or runtime qualification is claimed. A later
generator comparison still needs matched budgets, original spectral/fidelity
objectives, native streaming and listening evidence.

## Completed result: all four arms fail the prerequisite

All eight critics completed 1,024 updates each (8,192 total), froze before
scoring, and produced 17,920 review records. Generator updates remain zero.
All four arms fail for both generator origins and both corpora. This is a
completed mechanism study, not a new restoration-quality result.

Each entry below counts sources where the original reference receives a higher
classifier logit than the candidate. Music columns list seed 20261012 / 20261013;
speech counts are identical across both seeds. Roundtrip margins are the
numerical control; these are classifier rankings, not listening votes.

| Arm | Prediction, music / 45 | Prediction, speech / 16 | Phase control, music / 45 | Phase control, speech / 16 | Quarter-shift, music / 45 | Quarter-shift, speech / 16 |
| --- | --- | ---: | --- | ---: | --- | ---: |
| high_prediction | 42 / 41 | 16 | 0 / 0 | 0 | 4 / 4 | 1 |
| context_prediction | 42 / 41 | 16 | 0 / 0 | 0 | 8 / 8 | 3 |
| high_structured | 3 / 4 | 0 | 45 / 45 | 16 | 43 / 43 | 15 |
| context_structured | 3 / 4 | 0 | 45 / 45 | 16 | 45 / 45 | 15 |

The structured-negative intervention reverses the classifier's preference for
phase controls and substantially improves time-shift recognition. For the 3/8
shift never used in fitting, high_structured recognizes 41/45 music and 14/16
speech sources; context_structured recognizes 42/45 and 15/16, in both seeds.
Prediction-only arms recognize only 7/45 and 2/16 without summaries, or 9/45
and 4/16 with summaries. This demonstrates transfer across the tested shift
fractions, not general restoration or independent-source generalization.

However, structured arms reverse the original prediction ranking: only 3/45
or 4/45 music sources and no speech sources prefer the reference. Ordinary
prediction logistic loss fails the registered improvement condition. The
energy-matched prediction control also regresses to 7/45 or 9/45 music sources
and 2/16 speech sources, versus 36-38/45 and 14/16 for prediction-only arms.
The context_structured prediction mean margins are -0.0003330 / -0.0003323
for music and -0.0007484 / -0.0007201 for speech. Correctly recognizing the
synthetic controls does not offset this retained failure.

Adding observed-band summaries alone improves a minority of time-shift cases
but never fixes phase ranking. Learned added-weight norms are 0.312-0.327 for
context arms and exactly zero for off arms. This confirms updates reached the
new parameters; it does not prove useful musical conditioning or a capacity
ceiling. Both origins show the same qualitative conflict.

### Paired features and interpretation

Paired feature losses still penalize phase/time controls and remain zero at the
reference. After each critic's own fit-only feature normalization, the learned
distances remain close to random-feature controls. For example, seed-12 music
phase/time distances are 2.5009 / 3.5319 at initialization and 2.4871 / 3.5455
for context_structured; speech values change from 7.1313 / 10.0424 to
7.5122 / 10.6574. These are ratios to fit prediction distance, not audio-error
improvements. They provide no demonstrated restoration advantage over random
features. Original absolute feature values and PCM band RMS remain in every
source-level review record; no audibility threshold is introduced.

The matched intervention shows a tradeoff between recognizing synthetic errors
and distinguishing the actual fixed-generator distribution at this budget.
It supports investigating the combined classification objective, but does not
prove that one particular gradient conflict, model capacity or convergence
limit caused it. Synthetic controls are not a complete substitute for actual
restoration errors. No failed arm is promoted, and no generator training follows.

A concrete next diagnostic is to separate prediction, phase and time loss
contributions and their gradients on the registered fit material. If they compete,
test separate task outputs or paired-reference error features under a fresh
matched protocol, with random features retained. Do not merely retune negative
mixture weights on this exposed calibration, repeat unchanged longer training,
or claim every learned-feature objective is ineffective. That diagnostic and
any subsequent generator comparison have not been launched.

### Verification and retained evidence

All 804 local Python tests passed (239 warnings, 71.71 seconds); Ruff lint/format
passed for 361 files. Both jobs in the
[implementation CI](https://github.com/backrunner/soundex/actions/runs/40220002252)
passed. After completion, all 4,788 registered input/source bindings, the root
completion receipt and all 16 initial/final critic checkpoints were reverified.
The eight endpoints and every failed source/control remain in the registered
ignored run directory. No original audio, checkpoint or raw report is published.
