# P40: reversible update-direction audit

Completed 2026-10-23. The original Adam update can lower total loss while increasing
complex-spectrum and identity errors on the same training batch. A direction that
balances amplitude and complex gradients improves both on every small-radius probe,
but still fails joint preservation/dynamics checks on half the batches. This is a
local optimization diagnosis, not a trained model or a deployment quality gain.

## Registered scope

Reuse the two frozen P38 control checkpoints and precisely the P39 training draws:
16 evenly spaced schedule indices per seed, four music crops and one speech crop
plus their clean identities, real history, and original supervised support. No new
recordings, validation selection, DSP change or saved weight update occurs. Inputs,
source dependencies and endpoints are hashed. Parameters are restored exactly after
every probe. The stored optimizer is deep-copied independently for every draw, so
these are isolated experiments from a common endpoint, not 16 successive updates.

The actual objective remains the compressed P38 objective, including original
identity weighting and budget risk. The Adam probe loads its recorded moments,
step count and hyperparameters and applies the existing gradient clip of 5. It is
therefore more informative than assuming an SGD gradient is the deployed training
update. The median actual update L2 norm is 0.008057 / 0.007543 for seeds
20261012 / 20261013.

Five directions are compared at fixed parameter L2 radii 0.001 and 0.01: negative
total gradient; recorded Adam update; total gradient with the encoder component
removed; encoder-only amplitude/complex bisector; and full-parameter bisector.
The Adam update is additionally measured at its own recorded next-step magnitude.
The negative 0.001 radius supplies a central slope diagnostic. Radii and directions
were registered before execution, with no line search or best-radius selection.

A bisector is the negative normalized sum of the two individually unit-normalized
P39 local component gradients. This retains the exact local log1p denominators.
It diagnoses whether simultaneous descent exists; it is not a replacement objective
or a training recipe that protects every other term. Raw component metrics keep
their original units and definitions; unlike-objective totals are never ranked.

## Actual Adam response

On all 32 batches, the hypothetical next Adam step lowers total loss, direct
amplitude error and paired MR error. Yet complex error increases on **26/32**, and
identity MR increases on **27/32**. Dynamic error improves on 9/16 and 6/16.

| Metric at actual next-step magnitude | Seed 20261012 median change | Seed 20261013 median change |
| --- | ---: | ---: |
| Total compressed objective | −0.0992% | −0.0776% |
| Paired direct complex error | +0.0665% | +0.0505% |
| Paired MR dynamics | −0.00134% | +0.00368% |
| Identity MR | +8.135% | +3.559% |

Identity MR is very small here: median base values are 1.7221e-5 / 5.8593e-5;
median absolute changes are 1.2617e-6 / 4.5537e-7. These are dimensionless spectral
metrics, **not PCM dBFS or an audibility rating**. Percentages do not establish
clicks or audible severity. No calibrated native PCM or listening gain is claimed.

## Equal-radius comparison and retained counterexamples

Counts below are improvements among the 32 batches, pooling both seeds for counts
only. The five checks are direct amplitude, direct complex, paired MR, MR dynamics
and identity MR; every raw per-draw value remains in the evidence.

| Direction, radius 0.001 | Amplitude | Complex | Paired MR | Dynamics | Identity MR | All five |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Total-gradient SGD | 30 | 5 | 30 | 14 | 4 | 2 |
| Stored Adam | 32 | 6 | 32 | 15 | 5 | 4 |
| Total gradient without encoder | 30 | 5 | 30 | 9 | 5 | 2 |
| Encoder bisector | 31 | 32 | 30 | 30 | 16 | 15 |
| Full bisector | 32 | 32 | 30 | 30 | 17 | 16 |

The full bisector changes direct amplitude by a median −0.02665% / −0.02859% and
complex error by −0.01062% / −0.01568%. These are small local changes on the batch
used to construct the direction, not generalization or restored music quality.
Removing encoder updates alone does not resolve the conflict. Encoder-only and
full bisectors still sacrifice other objectives; no direction passes all five on
all batches. At radius 0.01, the full bisector improves complex error on 31/32 and
all five on 15/32; the encoder bisector passes all five on 14/32. The other three
directions retain only 2/32, 4/32 and 2/32 joint improvements, respectively. No
larger-radius failure is discarded in favor of the smaller-radius result.

Individual paired-example gradients above 16 kHz also conflict: 66/80 and 67/80
have negative encoder amplitude/complex cosine, with medians −0.965 / −0.975.
This rules out *only* cross-example averaging as an explanation, not all coupling
between bins, time samples, magnitude and phase. Counts include repeated training
draw content and are not counts of independent recordings.

## History-aware derivative verification

The first run recomputes warmup after perturbing parameters. Those forward changes
are valid local responses of the full warmed model, but their central slopes are
**not exact finite differences of the training graph**: training deliberately
stops gradients through warmup history. Its median symmetric derivative discrepancy
is roughly 2%, with much larger individual discrepancies; it must not be labeled
an exact derivative pass.

A separately registered follow-up freezes the original detached history for all
perturbations, matching the training graph. Across 640 direction/metric comparisons
per seed, all derivative signs agree. Median symmetric discrepancies fall to
0.0161% / 0.0179%, and the 95th percentiles to 0.723% / 0.438%. The worst float32
case remains 32.3% (seed 20261012, draw 2158, non-encoder direction, complex metric:
autograd 0.00195168 versus central difference 0.00381470). This counterexample is
retained; finite precision and a finite perturbation radius must be checked before
calling it a loss bug. Freezing history is a derivative control, not deployment
behavior and not a justification for shortening causal context.

## Precision follow-up for retained numerical counterexamples

A post-hoc float64 follow-up replays seed 20261012 draws 2158 and 14032 and seed
20261013 draw 4317 with the same fixed-history protocol and radii. These cases were
chosen to diagnose the largest float32 discrepancies, not to select a model or
claim independent validation. The original float32 reports remain unchanged.

For the worst original case, float64 autograd is **0.00195184238** and the central
slope is **0.00195232046**, a symmetric discrepancy of about **0.0122%**, compared
with 32.3% in float32. This supports cancellation/rounding as the source of that
particular mismatch. Across all 120 float64 direction/metric checks, the maximum
symmetric discrepancy is 0.440%; finite-radius curvature and numerical precision
remain distinct from an exact symbolic derivative. No loss term, reference,
production precision or checkpoint was changed to obtain these checks.

## Decision and evidence

Do not launch an unchanged Adam continuation, scale the same model, remove phase
supervision, or treat the bisector as a qualified training method. The next optimizer
control needs explicit identity/dynamics protection and transfer checks on different
training draws before a matched small-model training study. Native reconstruction,
low-band/stereo/continuity and real-time checks remain required for any candidate.

- `data/runs/update-directions-p40-20261023/`: original warmed-model responses,
  exact checkpoint/optimizer provenance, per-example gradients and all perturbations.
- `data/runs/update-directions-p40-fixed-history-20261023/`: separately registered
  derivative control, preserving the first run unchanged.
- `data/runs/update-directions-p40-precision-20261023/`: registered post-hoc float64
  numerical cases, executed source and every result.
- `training/scripts/audit_update_directions.py`: reproducible entry point; pass
  `--fixed-history` only for the training-graph derivative control.

See [P39 attribution](objective-conflict-audit.md), [failure registry](failed-experiments.md)
and [CPU-time attribution](cpu-time-attribution.md).

Local verification: 52 objective/context-deployment Python tests pass, including
real ONNX/Rust causal-state, reset, silence and physical-channel checks. Existing
legacy ONNX export warnings remain. Rust workspace tests pass, including two new
CPU-clock tests; the existing 30-minute stress test remains explicitly ignored.
Ruff, Rust formatting/Clippy and English-documentation checks are separate from
model-quality or real-device acceptance.
