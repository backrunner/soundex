# P61: critic objective and paired-feature gradient audit

Date: 2026-10-26. Follow-up to [P60](texture-context-p60.md), alongside the
[architecture/evaluation review](architecture-evaluation-p61.md).

## Registered diagnostic

Use every P60 arm, both fixed restoration origins, and initial/trained critics.
Inspect the 16 original P59 fit draws only. No calibration or reserved evaluation
recording is rescored. Verify parent completion, input/source bindings, all critic
checkpoints and current reserved-source aliases before registration. Retain exact
coverage in `data/runs/critic-objectives-p61-20261026/registration.json`.

For each draw, reproduce the phase seed and time fraction from its last P60
training cycle. Decompose real-plus-fake prediction/phase/time tasks and the
exact per-example mixture, including roundtrip exclusions. Known-answer tests
verify equality of both loss and parameter gradients against P60. Report
per-corpus and whole-batch gradient norms/cosines. Batch-gradient geometry is not
an independent per-recording quality estimate.

For every critic, inspect each task's derivative along negative actual-arm
total gradient and copied fresh-Adam displacement. For trained critics also use
the frozen endpoint's saved moments and step clock. Copies preserve original
1e-4 learning rate, betas, epsilon, clipping and float32 displacement. A test
matches the copied persistent update exactly to a multi-parameter Adam step.
No critic tensor, optimizer state or restoration parameter is updated in place.
Local derivatives do not reconstruct the entire previous training trajectory.

Separately, on each paired PCM example, compare frozen paired-feature gradients
with the existing original-reference multiresolution spectral objective. Apply
a fixed negative feature-gradient probe whose RMS is 1% of the current reference
error RMS; report both actual feature and MR changes and absolute PCM errors.
Reference equality is stationary, including silence and weak signals. Compare
random and trained critics independently of their classifier preference.

The probe operates on unconstrained raw PCM, not generator parameters, DSP output,
or the full direct-plus-MR objective. The fixed step is not a selected training
weight or acceptance threshold. It can reveal a helpful local direction without
showing that our restoration network can realize it or that audio sounds better.

## Completed results

The audit finished 768 batch/corpus gradient records and 1,280 paired PCM probes.
Actual exposure is 80 paired crops from **58 music and 14 speech recordings**,
37.097 unique MR-supervised seconds. The same crops are reused across critics,
not independent replications. All nine inherited codec configurations and all
five channel views are present. The selected music lacks labeled jazz, hiphop,
pop and metal; inherited regional labels are metadata, not a new provenance
audit. No fresh evaluation source or long-form native output is assessed.

### Classification objectives compete

The table shows trained-critic whole-batch geometry across 16 fixed fit draws.
Values separated by a slash correspond to the two frozen generator origins.
The last column counts positive derivatives of the ordinary prediction task
along the actual copied Adam displacement using saved endpoint moments.

| Arm | Mean prediction/phase gradient cosine | Phase-opposed draws / 16 | Prediction-uphill saved-Adam directions / 16 |
| --- | --- | --- | --- |
| high_prediction | -0.1283 / -0.1275 | 9 / 9 | 0 / 0 |
| context_prediction | -0.5163 / -0.5178 | 16 / 16 | 0 / 0 |
| high_structured | -0.3380 / -0.3382 | 14 / 14 | 15 / 15 |
| context_structured | -0.2805 / -0.2778 | 12 / 12 | 15 / 15 |

With fresh moments, structured arms have prediction-uphill directions on 7/16
(high) and 6/16 (context), in both origins. Optimizer history therefore matters;
gradient cosine alone does not describe the actual local update. Time-task mean
cosines remain positive (approximately 0.34-0.64), so not every synthetic task
conflicts equally. The retained corpus records separate speech/music behavior.

This is evidence for a local objective/momentum conflict in the current classifier
recipe. It is not a full causal reconstruction of P60, a finite-step audio result,
or proof of an architecture/capacity limit. The prediction-only arms are controls,
not qualified restoration candidates.

### Learned features do not beat random features in this probe

The next table reports the percentage change in source-equal mean **raw PCM MR
error after the fixed analytic probe**, not a trained model improvement. More
negative indicates greater local reduction. Initial critics are functionally
equivalent across arms and are shown once per origin.

| Features | Music 20261012 | Music 20261013 | Speech 20261012 | Speech 20261013 |
| --- | ---: | ---: | ---: | ---: |
| Random initial | -2.4844% | -2.4616% | -1.8355% | -1.8166% |
| high_prediction trained | -2.1661% | -2.1417% | -1.6980% | -1.6790% |
| context_prediction trained | -2.1664% | -2.1424% | -1.6905% | -1.6717% |
| high_structured trained | -2.4029% | -2.3818% | -1.8004% | -1.7823% |
| context_structured trained | -2.3755% | -2.3543% | -1.7933% | -1.7751% |

Every learned arm is weaker than its own random-feature control on these mean
reductions, for both origins and corpora. Learned features beat random features
on only 3-6/58 individual music sources, depending on arm/origin, and 0/14 speech
sources. This is a separate measurement from classifier recognition, not a
deduction from the failed P60 ranking gate.

Counterexamples remain: each probe improves MR on 51/58 music sources and all
14 speech sources; seven music sources do not improve. On five music sources,
the finite step even increases the feature objective despite following its local
negative gradient. The fixed step is not a universal safe learning rate and is
not retuned after seeing those cases. Mean music PCM error RMS is about 0.01677
before the probe and 0.01675-0.01676 afterward; speech is about 0.00291 before
and 0.00290 afterward. Per-source calibrated RMS and actual step RMS remain in
the raw report. Neither mean absolute error nor this gradient construction
establishes audibility, deployable improvement, or correct stereo behavior.

## Decision

Retire the current tiny frozen-output classifier-training recipe as the next
generator auxiliary: additional unchanged training and mixture-weight tuning
are unsupported by P59-P61. Preserve its code/evidence as controls. This does not
reject all adversarial training or paired feature objectives. In particular,
published systems jointly train different generators and discriminators; our
experiment does not reproduce their training procedure.

Prioritize the bounded band-token causal architecture comparison described in
the [architecture review](architecture-evaluation-p61.md), under unchanged
original-reference objectives and matched controls. New architecture prototyping
and generator training have not started. No default weight changes, new native
timing or human listening result are claimed.

All 811 local Python tests passed (239 warnings, 71.71 seconds), Ruff checks
passed for 364 files, and both jobs in
[implementation CI](https://github.com/backrunner/soundex/actions/runs/40229084516)
passed. After completion, all 4,813 registered bindings and the root summary
receipt were reverified. All critic tensors remain unchanged; both critic and
generator update counts are zero.
