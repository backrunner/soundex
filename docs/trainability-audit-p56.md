# P56: trained-adapter trainability and identity-risk attribution

## Changed question

[P54](spectral-training-p54.md) completed all four 4096-update continuations, and
[P55](longform-review-p55.md) found only 0.016-0.054% additional paired MR benefit
over their parents. Auraloss does not close the historical spectral gap. Do not
repeat the same continuation or infer a small-model capacity ceiling.

P39-P43 diagnosed older shared-encoder gradients; P48-P53 tested frozen-feature
adapters, finite steps and added preservation penalties. None tested the fully
continued P54 adapter with matched trainability scopes, identity ablation and
actual Adam-history controls. This new diagnostic isolates those mechanisms
without adding parameters or using P55 recordings to choose a training setting.

## Fixed controls and data

Entry point: `training/scripts/audit_adapter_trainability.py`. Evidence:
`data/runs/trainability-audit-p56-20261025-v2/`. Both common-loss P54 final seeds
are included, regardless of their P55 ranking. The already unsuccessful auraloss
change is not combined with another intervention.

Use four evenly spaced draws from each of the registered P54 optimization and
TRAIN calibration schedules. Each draw contains four music and one speech crop
plus their clean identities. Original references, physical rates, codec/view
assignments, source intervals and real warmup stay fixed. These are reused TRAIN
recordings, not fresh generalization evidence. Independent channel views do not
constitute simultaneous stereo training. The registration retains actual source
coverage and unique supervised time for this bounded probe.

Optimization and calibration are checked for recording/source and connected work
alias isolation. Every local `data/runs/*/evaluation-catalog.jsonl` reservation is
excluded through file hashes, decoded PCM hashes and transitive work aliases,
including bridges in the approved catalog and processed manifests. Selected
clean/degraded PCM checksums, both endpoints, schedule, detector, executed Python
sources and source snapshots are hash-bound before measurement.

## Matched trainability and objective controls

All scopes start with exactly the same forward output, budget and causal state;
the adapter, nonlinear decoder and backbone retain their existing tensor sizes.

| Scope | Trainable tensors |
| --- | --- |
| Adapter | Original linear adapter only |
| Decoder | Adapter plus existing nonlinear decoder |
| Full | All existing small-model tensors, including causal features and budget head |

The full scope connects the adapter to its live features instead of detaching
them. This changes derivatives only, not the starting forward function. No new
lookahead, inference branch or parameter is introduced. Warmup is detached in
gradients and recomputed for finite-step measurements; evaluation mode prevents
statistics updates. The full scope trains budget predictions through reference
reconstruction only; no auxiliary DSP/budget teacher is added.

The objective is unchanged common direct-spectrum plus raw multiresolution loss,
including original-reference amplitude, complex, envelope and real dynamics.
Record paired-direct, paired-rendered, identity-direct and identity-rendered
gradients separately. Keep the paired coefficient at 0.5 and test identity
coefficients 0.5 and zero. Zero is a causal ablation, **not** permission to remove
fidelity from production or accept new damage.

Each proposal takes exactly one actual Adam step from the frozen endpoint,
averaging four optimization batches, at each fixed rate 1e-7, 1e-6 and 1e-5.
Fresh moments are matched across all three scopes; an additional adapter control
continues the exact stored P54 moments. Betas .9/.999, epsilon 1e-8 and clip five
are unchanged. No post-step multiplier, veto, projection or line search applies.
This yields 48 proposals across two seeds; all are retained, not just improvements.

Freeze every proposal hash before calibration scoring. Report pooled and
per-batch gradient geometry, parameter-group contributions, actual displacement
and each component's inner product with that displacement. Then recompute all
four fit and four calibration batches, retaining paired/identity direct loss,
MR, calibrated PCM RMS and amplitude/complex/envelope/dynamics errors per batch.
Same-batch descent does not establish recording-disjoint transfer. Gradient norm,
first-order agreement and one-step fit cannot establish convergence, capacity,
causal-history utility or listening quality.

## Validation and interpretation boundaries

Tests verify exact starting forward/state parity, unchanged parameter count,
scope isolation, nonzero adapter-to-feature gradients, finite differences of
the connected feature path, unchanged common-objective value, identity ablation
without changing the paired coefficient, actual Adam agreement, checkpoint
creation and transitive reservation rejection. Full warmup is deliberately not
differentiated; finite differences test a fixed initial state, while actual
step measurements recompute the real warmup.

This is a raw-output optimization diagnosis. It does not replace Rust rendering,
native timing, longer matched training or reference-based blind listening. No
automatic candidate selection, further training or default promotion follows.
Subsequent training must state its changed hypothesis and keep common controls,
all fidelity counterexamples, fixed budgets and freeze-before-review barriers.

The first attempt failed while saving its first proposal because the destination
directory had not been created. It produced no frozen proposal set or calibration
scores. Its registration, exact source snapshot and traceback are preserved in
`data/runs/trainability-audit-p56-20261025/`. V2 adds explicit directory creation
and a real checkpoint-write regression test; data, scopes, rates and scoring are
unchanged. This operational failure is not a model-quality result.

## Completed results and decision

All **48 proposals** and both starts completed both TRAIN roles. Input, source,
endpoint and summary hashes verify; no default weight changed. The fit role has
20 distinct recordings and 9.206 unique MR-supervised seconds; calibration has
19 recordings and 9.167 seconds. This small support is a mechanism probe, not
library-scale exposure or a generalization conclusion.

The existing model has 480,809 parameters. Trainable counts are 49,794 for the
adapter, 93,348 for adapter plus decoder, and 480,809 for full reconstruction.
All starting forward predictions match exactly. The new full scope reconnects
adapter-feature gradients while keeping the existing causal forward graph.

On pooled fit gradients, identity/paired norm ratios are 2.026 and 2.042 for the
adapter. Their cosine is +0.0225 for seed 12 and -0.1345 for seed 13: a larger
identity norm is not evidence that every direction conflicts. Full-scope ratios
fall to 1.722 / 1.370. The feature-group paired norms are 0.241 / 0.357, while the
existing nonlinear decoder contributes only 0.0169 / 0.0149. These local derivatives
indicate useful omitted directions, not guaranteed restoration after training.

At the existing rate 1e-7, with fresh matched Adam and full identity protection:

| Scope | Fit paired MR change, seed 12 / 13 | TRAIN transfer MR change, seed 12 / 13 |
| --- | ---: | ---: |
| Adapter | -0.000130% / -0.000108% | +0.000299% / +0.000217% |
| Adapter + decoder | -0.000151% / -0.000129% | +0.000286% / +0.000207% |
| Full reconstruction | -0.000749% / -0.000899% | -0.0000205% / -0.0000219% |

The full-scope transfer differences are only about -6.71e-8 / -7.08e-8 in raw
MR units, close to float32 resolution. They are **not meaningful quality gains**.
Seed 12 improves only one of four transfer batches; seed 13 improves three.
Paired amplitude still worsens slightly at this rate, and seed-12 paired risk
also rises, despite lower mixed risk and lower identity error. Retain these
counterexamples rather than treating the mean MR sign as qualification.

Across the fixed three-rate diagnostic, full-scope transfer MR decreases for both
seeds, whereas merely unfreezing the decoder retains most adapter-only behavior.
At 1e-5 with full identity weighting, full-scope transfer MR falls 0.00102% /
0.00516%, but identity PCM RMS rises 6.876e-6 / 6.596e-6 and mixed risk worsens.
Removing identity yields larger spectral changes but additional protection
tradeoffs; at 1e-5 even paired-only adapter proposals worsen transfer MR despite
improving fit MR. Neither a larger rate nor deleting identity is a justified fix.
Absolute PCM changes are diagnostics, not audible-severity thresholds.

Continued Adam moments also matter: at 1e-7 the adapter's fit MR step is several
times smaller than its fresh-moment counterpart, and its transfer behavior differs.
Do not compare fresh full-model Adam against a continued adapter and attribute
all differences to architecture. Gradient norms alone do not determine updates.

The resulting [P57 protocol](joint-training-p57.md) tests cumulative training of
existing features with fresh Adam matched in **both** scopes, both seeds, the
same original-reference objective and identity protection, the existing 1e-7
rate and equal 4096-update budgets. It does not promote this one-step diagnostic,
increase inference capacity, tune on P55 or remove fidelity. Meaningful changes
must survive complete training, raw/native rechecks and timing; the hypothesis
remains unproven until those results exist.

Validation at completion: 770 Python tests pass (233 existing warnings), Ruff
passes all 347 Python files, and 134 English Markdown documents pass links/index
checks. The first-attempt operational failure remains preserved above.
