# P39: objective gradients and shared representation

Completed 2026-10-23. The frozen small models show a strong **parameter-space conflict**
between magnitude reconstruction and complex fidelity. However, their gradients in
high-band output magnitude and phase coordinates are aligned. The evidence therefore
does **not** justify deleting phase/complex supervision or claiming the targets are
intrinsically incompatible. It points to shared representation and gradient
conditioning as the next mechanism to test.

## Scope and verification

Use both P38 `control` endpoints, each trained for 16,192 steps with compressed risk,
identity weight 1 and native conditions. These control tensors match the corresponding
P36 compressed endpoints. Select 16 evenly spaced registered **training** draws per
seed before computing gradients. Each draw retains the original four-music/one-speech
sampling and paired/identity roles, real warmup and supervised support. There are
32 diagnostic batches, not another training trajectory or an independent evaluation.

Decompose the actual objective into direct amplitude/complex/envelope/dynamics,
MR amplitude/log-shape/envelope/dynamics, identity and budget contributions. For
`log1p(loss)`, use detached `1 + loss` denominators to obtain the exact local gradient.
The sum must match autograd on the actual total, with the existing assertion
`rtol=2e-4, atol=2e-7`. Maximum observed element errors are `8.94e-8` and `2.01e-7`;
all assertions pass. These decomposed values are derivative carriers, not a new
objective or comparable loss totals.

No optimizer step, model selection, threshold change or DSP change occurs. Inputs,
source dependencies and checkpoint hashes are checked. The endpoint remains fixed.
The initial whole-parameter audit, the output-coordinate extension and the parameter-
group extension are all retained. Later extensions add attribution on the same
predetermined draws; they do not choose a better seed or erase a failed run.

## Results across the fixed draws

Cells list seed 20261012 / 20261013. Cosines are medians across 16 draws, not proof
of a universal conflict across the training distribution.

| Comparison | Median cosine | Negative draws |
| --- | ---: | ---: |
| Parameter gradients: direct amplitude / direct complex | −0.926 / −0.912 | 14/16 / 15/16 |
| Parameter gradients: direct complex / MR amplitude | −0.925 / −0.878 | 14/16 / 15/16 |
| Parameter gradients: direct dynamics / MR amplitude | +0.882 / +0.902 | 0/16 / 0/16 |
| Parameter gradients: MR amplitude / identity | −0.408 / −0.577 | 15/16 / 15/16 |
| Output ≥8 kHz dB gradients: direct amplitude / direct complex | +0.949 / +0.956 | 0/16 / 0/16 |
| Output ≥8 kHz phase gradients: direct complex / MR amplitude | +0.662 / +0.660 | 0/16 / 0/16 |
| Output ≥8 kHz phase gradients: MR amplitude / MR dynamics | +0.822 / +0.837 | 0/16 / 0/16 |

Direct amplitude's high-band phase-gradient norm is about `5.4e-11`, consistent with
magnitude invariance up to numerical roundoff. MR amplitude already supplies a phase
learning path through overlap-add and reanalysis: its median high-band phase-gradient
norm is `6.51e-4 / 6.74e-4`, versus `1.23e-4 / 1.26e-4` for the weighted direct-complex
term. Thus the current objective is not missing phase gradients altogether. Magnitude
uses dB coordinates and phase uses radians; comparing their norm ratio would not be
an invariant importance measure.

The parameter-group audit concentrates most gradient energy in the shared encoder:

| Group | Direct amplitude/complex cosine | Amplitude squared-norm share | Complex squared-norm share |
| --- | ---: | ---: | ---: |
| Encoder | −0.945 / −0.919 | 90.28% / 91.46% | 85.95% / 88.94% |
| Temporal branch | −0.942 / −0.938 | 0.89% / 0.61% | 1.00% / 0.64% |
| Decoder | −0.634 / −0.735 | 0.60% / 0.60% | 0.92% / 0.79% |
| Budget head | −0.927 / −0.920 | 8.05% / 7.09% | 12.24% / 9.50% |

Each cell is a separate median, so shares need not sum exactly to 100%. These are
parameter-coordinate-dependent local gradients, not architecture-independent capacity
scores. Positive output-coordinate cosines over paired high bands coexist with
negative shared-parameter cosines; one cannot infer from the latter alone that phase
must be sacrificed to improve amplitude. Full-band aggregation, sample coupling and
the learned Jacobian still require separation.

## Decision

Keep native conditioning, the original lossless target and current production
weights. Do not launch a larger version of the rejected fixed-detector route, remove
complex fidelity, or claim a quality improvement from this audit. The next small-model
control should first separate per-example/band contributions and encoder update
conditioning, with actual common-objective directional checks. Only a measured
mechanism should motivate an isolated representation or optimization change.

P20 and P31 already tested alternative heads/phase coordinates without qualification;
a new study must explain what changes beyond those attempts. Any new training still
requires matched budgets, both seeds, raw/native spectral assessment, preservation
and continuity checks, plus separate real-time qualification. These 32 batches cannot
establish generalization, convergence or audible improvement.

## Evidence

- `data/runs/objective-conflicts-p39-20261023/`: initial decomposition.
- `data/runs/objective-conflicts-p39-coordinates-20261023/`: added output dB cosines.
- `data/runs/objective-conflicts-p39-groups-20261023/`: encoder/temporal/decoder/budget attribution.
- Each directory retains registration, per-draw results, summary and executed source.
- Entry point: `training/scripts/audit_objective_conflicts.py`.

See the [balanced timing audit](interleaved-timing-audit.md) and [failure registry](failed-experiments.md).
