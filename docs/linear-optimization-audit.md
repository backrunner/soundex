# P8 Optimization Process Audit and Energy Scale Comparison

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

2026-10-12. Building upon [P8 Final Results](linear-reconstruction-study.md). This round does not relax the loss function, does not remove weak-band material, and does not alter group acceptance criteria. All results are diagnostic for magnitude models; no production weights were promoted.

## Design and Reproducibility Scope

In the first phase, we replay the first 128 steps of the two original seeds' raw draws, using the same hyperparameters: learning rate = 1e-4, 1e-5, 1e-6; AdamW optimizer with weight decay = 1e-4; gradient clipping = 5; original 2048-step cosine decay schedule and 80-hop warmup remain unchanged.

At the 1/8/32/128-step intervals, we re-evaluate all existing development paired and identity segments. No test recordings are loaded during this process.

Additionally, we select the first two music samples and the first speech sample from the original `training_probe_specs` registration order, along with the identity version of the first music sample—totaling four segments. Each group is fit for exactly 256 steps, starting from identical initializations and using a fixed learning rate. This deliberately increases the proportion of speech and identity data, testing whether fixed samples can be learned under controlled conditions. However, this does not constitute a full overfitting test, nor does it prove capacity limits or generalization performance across the full dataset.

In the first phase, we also record gradients before cropping for each module, as well as the cosine angle between magnitude and reference dynamic gradients. For each Adam update, we separately apply updates to only the energy head, only the spectral shape decoder, and to all parameters combined. We then scale these updates down to 0.1, 0.01, and 0.001 times their original magnitude. This is for update attribution analysis and is not used to select final weights. Forward passes restore the initial state before each interpolation, with input and reference signals held constant.

In the second phase, based solely on evidence from the first phase, we modify only the energy residual unit while keeping all other settings fixed. We compare the 1e-4 and 1e-5 cases. This is a post-hoc mechanism intervention, not a new independent generalization experiment. No threshold scanning or selection of a specific recording is used to determine the unit.

## First Update Shows Severe Oscillation

Both initial development seed models started with a development error of 0.494611. Results from the first step of the original energy coordinate:

| Metric | 20261012 | 20261013 |
| --- | ---: | ---: |
| Initial batch loss | 0.339015 | 0.271275 |
| First step loss (lr=1e-4) | 79.370338 | 43.766762 |
| Energy head update only | 80.589989 | 44.400188 |
| Spectral shape decoder update only | 0.472519 | 0.355488 |
| First step development score | 18.625044 | 15.952812 |
| Energy head gradient norm (before clipping) | 26110.893 | 3752.096 |
| Spectral shape decoder gradient norm (before clipping) | 264.294 | 113.763 |
| Cosine of magnitude and dynamic gradient | 0.9559 | 0.4378 |

Both heads were initialized to zero, so a zero backbone gradient at the first step is expected behavior and not indicative of a gradient collapse. The gradients for both objectives were generally aligned in the first batch; there is no evidence that the first-step failure resulted from mutual cancellation. Gradient clipping does not equate to restricting actual output changes—especially in Adam, where per-parameter normalization alters the update scale.

The original energy coordinate is defined as:
`scale * smooth_abs(input_rms / scale + delta)`
The derivative of `delta` is approximately proportional to the full spectrum `scale`, and relative changes in weak bands can be amplified by thousands of times.
For example, in the first seed, a segment at 16 kHz and above had a reference power relative to the full spectrum of -108.31 dB. After the first step, the predicted power rose to -26.94 dB—this is a relative spectral power, not a dBFS or audible output.
Since both the first step and the entire development set degraded, the issue arose at the very beginning of training and cannot be attributed solely to later overfitting.

## Single-Factor Intervention on Input Band Scale

In the experiment `ConditionedEnergyEnvelope`, the initialization tensor, 807,398 parameters, features, spectral shape head, and initial output remain unchanged. Let `r = input_rms / scale`, and we modify the expression to:

`scale * smooth_abs(r + sqrt(r² + 1e-8) * delta)`.

Here, 1e-8 is the square of the existing relative magnitude floor (1e-4); only compressed input statistics are used, and no lossless reference is read.
Despite this, weak-band gradients remain non-zero, and the output upper bound remains at `exp(2) * scale`. There is no hard clean decision or bypass gating.
This change alters the learning difficulty and does not guarantee that missing high-frequency content can be recovered within the same budget; actual recovery must still be verified.

The energy head gradient norm drops to **4.031 / 3.263**, while the spectral shape head gradient remains unchanged.
At step 1, lr=1e-4, the development score drops to **0.544080 / 0.547443**, a significant reduction in overshoot, though still above the initial 0.494611.

After 128 steps, the common development score (lower is better):

| Energy Unit / lr | 20261012 | 20261013 |
| --- | ---: | ---: |
| Full spectrum / 1e-4 | 6.351267 | 9.121172 |
| Full spectrum / 1e-5 | 1.685696 | 1.512088 |
| Full spectrum / 1e-6 | 0.641335 | 0.582026 |
| Input band / 1e-4 | 0.526563 | 0.521826 |
| Input band / 1e-5 | 0.508017 | 0.507573 |

**No configuration exceeds the initial development score.** The improved optimization stability cannot be interpreted as successful reverse compression. Simply reducing the global learning rate while slowing all branches does not resolve the scale mismatch across frequency bands.

## Fixed Small Sample Can Learn, But Still Fails Identity Fidelity

For the four fixed segments, the average initial loss is 0.443272. After 256 steps:

| Energy Unit / lr | 20261012 | 20261013 |
| --- | ---: | ---: |
| Full spectrum / 1e-4 | 0.432235 | 0.574012 |
| Full spectrum / 1e-5 | 0.326860 | 0.359503 |
| Full spectrum / 1e-6 | 0.347671 | 0.344388 |
| Input band / 1e-4 | 0.272663 | 0.271027 |
| Input band / 1e-5 | 0.334590 | 0.334214 |

The input band / 1e-4 case shows a relative error reduction of about 38% compared to the initial average. Errors for all three paired segments decrease, but the identity segment increases from near zero to **0.03515 / 0.03932**. Therefore, this cannot be considered a full success.
This check confirms that the current model possesses some learning capability, but it does not prove complete fitting, sufficient capacity, or acceptable generalization.

After the scale adjustment, applying the spectral shape update alone raises the initial error from 0.3390/0.2713 to approximately 0.4725/0.3555. This head continues to use the full-spectrum unit to rewrite bin seeds, followed by per-band normalization. Future work should independently verify the scale of spectral shape updates, the generative range of missing frequency bins, and the gradient source of identity degradation—especially to avoid confusion when both loss and expansion are modified.

## Delivery and Boundaries

All 20 short training/fittings have been completed: 12 runs in the original coordinate, 8 in the input-band coordinate.
Local reports are stored in `data/runs/linear-p8-20261012/{optimization-audit,conditioning-audit}`.
The parent configuration, sampling, and script SHA are preserved. For the second phase, the Python source code hash is also saved.
Copies of the original scripts used in both phases, along with checksums, are retained in their respective report directories.
The summary includes SHA hashes for each group's result files, enabling per-item verification. Raw audio, diagnostic reports, and experimental outputs are not pushed.

New test coverage includes coordinate sensitivity, missing band generation, finite differences, identical initialization, forward function consistency, silence finite gradient, channel independence, and update interpolation without modifying input.
Loss remains unchanged; P8 mathematical tests remain applicable. 388 local Python tests passed, and ruff checks passed.
No new ONNX export, no new Rust DSP with expanded structure, no blind listening, and no full real-time benchmark were conducted.
Parameter count alone does not substitute for performance validation of new operations.
No small-sample fitting model was exported, and no default weights were replaced.

```sh
PYTHONPATH=training training/.venv/bin/python training/scripts/audit_linear_optimization.py \
  --study data/runs/linear-p8-20261012/study \
  --output data/runs/linear-p8-20261012/optimization-audit-recheck

PYTHONPATH=training training/.venv/bin/python training/scripts/audit_linear_optimization.py \
  --study data/runs/linear-p8-20261012/study \
  --output data/runs/linear-p8-20261012/conditioning-audit-recheck \
  --coordinate input_relative --rates 0.0001 0.00001
```

The output directory must be absent to prevent overwriting prior evidence. Small-sample results and development data are used solely for this mechanism diagnosis. The next formal training will still require frozen selection rules, full group validation, and actual DSP comparison.
