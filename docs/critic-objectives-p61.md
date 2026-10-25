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

## Status

The registered audit is running. All 811 local Python tests passed (239 warnings,
71.71 seconds). No new restoration or critic training is launched. Results will
retain every arm, origin, corpus and adverse local direction.
