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

## Execution status

All 804 local Python tests pass (239 warnings, 71.71 seconds); Ruff lint/format
passes for 361 files. The protocol is registered and the eight-critic run has
started from its frozen source snapshot. No outcome is inferred from P59 or
the hypothesis. All eight endpoints and failed controls will remain recorded.
