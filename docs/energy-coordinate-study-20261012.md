# P7: Parametrization of Energy in Missing Bands

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

Building upon [P6 Comparison](factorized-envelope-study-20261012.md), this round modifies only the output coordinates of band-level energy, evaluating whether multiplicative updates relative to input energy are obstructing learning under severe band suppression. The following rules are registered prior to formal training.

## Hypothesis and Isolation Probe

The P6 band-level output is defined as:
`s * exp(clamp(log(r/s) + d, -30, 2))`
where `s` is the RMS of the full spectrum input,
`r` is the smoothed RMS of the band, and
`d = 4 * energy_head(latent)`.

When `r/s = 1e-8` and no clipping occurs, the derivative of `d` with respect to the output is non-zero only in `r`.
A zero-gradient deficiency was already resolved in P6, though non-zero gradients may still correspond to poor optimization conditions.

A candidate alternative is proposed:
`s * min(sqrt((r/s + d)^2 + 1e-16), exp(2))`
This directly adds or suppresses energy within the full-spectrum RMS unit. The absolute value smoothing ensures non-negativity and avoids undefined gradients at zero; the upper bound remains consistent with P6.
The zero-head initialization preserves an approximately identity output, with missing bands exhibiting only a numerical floor of approximately 1e-8.
The number of parameters in the energy head, per-band shape normalization, backbone, and input features remains identical.
This remains an additive residual in input RMS units and does not claim to achieve anchor-free absolute envelope prediction.

A fixed, uniformly distributed target spectral shape is used, with training on a single energy bias.
Training parameters: lr=1e-4, AdamW, clip=5, 128 updates.
The target RMS for severely missing bands is set to 0.01.
- Multiplicative head loss: **1.94196 → 1.94196**
- Output RMS: **8.16e-10 → 8.60e-10**
- Additive head loss: **1.94195 → 0.25063**
- Output RMS: **1.15e-9 → 0.00418**

This is a double-precision, single-parameter diagnostic. The initial energy difference in output arises solely from numerical flooring.
The new coordinate system also amplifies initial gradients significantly, so gradient clipping is retained, and non-finite values are monitored during actual training.
These results do not prove that the backbone has learned sufficiently, that convergence to real music has occurred, or that generalization across recordings is achieved.

Reproduction entry: `training/scripts/diagnose_energy_coordinates.py`
Local result: `data/runs/energy-p7-20261012/coordinate-probe.json`

## Formal Comparison

| Group | Energy Output | Other Components |
| --- | --- | --- |
| factorized | Log-gain relative to input RMS | Shared initialization, features, shape head, loss |
| rms_residual | Additive residual in full-spectrum RMS units | Same as above |

Two seeds: `20261012` and `20261013`. Each group is trained from scratch for **2048 updates**, batch size 4, totaling four runs.
All initial parameters are tensor-wise identical across seeds, with a total of **807,398 parameters**. No P6 weights are loaded.
Optimizer: AdamW, lr=1e-4, cosine decay to 0.1×, weight decay=1e-4, clip=5.
A new fixed sampling scheme is used: `seed + 50000`, with 8192 samples per seed, including 1024 identity control points.
Source and recording isolation protocols are preserved, as verified in P6.
Speech sampling ratios and identity substitution methods remain unchanged.

All groups maintain the same 1024/2048 context window, 80-hop warmup exclusion zone, and P6's energy/spectral shape/dynamic losses.
A common scoring metric is used, with validation every 512 steps; initial weights are selectable.
Music and speech weights, as well as identity weights, are unchanged.
Model structure tests cover identical initialization, independent energy/shape heads, zero/weak input gradients, finite-difference checks, causality, chunking, channel isolation, and MPS/CPU correspondence.
Inherited losses do not alter weights or numerical floors.

## Interpretation and Synthesis Validation

The comparison of interest is `rms_residual - factorized`, with bootstrap pairing at the per-recording level.
The shared loss only serves for mode selection and does not drive individual promotion.
Requirements include improved energy recovery and concurrent enhancement of high-frequency spectral shape, with review of flux, low-band preservation, identity, speech, and compressed input.
When results cross zero or show seed-level inconsistency, uncertainty is retained.
Even if the best results lie at the budget boundary, convergence cannot be claimed.

After training, a common scoring is automatically executed, followed by ONNX export of the top candidate from the first seed, and per-hop CPU cost and independent RSS measurements.
The diagnostic model remains constrained to ≤2M parameters and <8 MiB.
The benchmark is not run concurrently with this training session.

Subsequently, actual Rust DSP validation is performed on frozen P3 phase outputs:
- Original P3 phase
- Identity DSP
- Both energy heads paired with P3 phase
- Candidate paired with input phase

A total of 89 segments × 5 paths × 2 seeds = **890** independent playback instances are evaluated.
Common start and clipping windows are identical to P6.
Dynamic HF, fixed above 4 kHz, and five-band raw/output metrics on the same FFT grid are reported to avoid misattributing differences in output to synthesis artifacts.
The native/proxy maximum error threshold remains at **2e-5**.

This is a reuse of the development set—no independent test set is read, and no re-scoring or threshold adjustment is performed.
Default weights are not replaced.
True streaming integration, full latency, stability, and blind listening have not been evaluated in this magnitude of experiment.

## Execution and Outputs

```sh
PYTHONPATH=training training/.venv/bin/python training/scripts/run_energy_study.py \
  --parent data/runs/factor-p6b-20261012/study/config.json \
  --output data/runs/energy-p7-20261012/study \
  --cache data/runs/factor-p6-20261012/study/cache

PYTHONPATH=training training/.venv/bin/python training/scripts/analyze_envelope_study.py \
  --study data/runs/energy-p7-20261012/study --benchmark

PYTHONPATH=training training/.venv/bin/python training/scripts/replay_factorized_study.py \
  --study data/runs/energy-p7-20261012/study \
  --phase-study data/runs/causal-p3-20261012/study \
  --replay-binary target/debug/soundex-prediction-replay
```

Formal execution runs from the frozen source code after submission, logging launch, pipeline stages, and logs; checkpoints preserve the optimizer and RNG.
Only cache is shared via input hash lookup; no weights are reused across experiments.
Original data, checkpoints, and reports are retained in local directories and ignored.

## Formal Results

All four groups completed 2048 updates, with best weights occurring at the final step—this does not imply convergence.
A total of 890 Rust DSP re-plays were completed, with maximum proxy error of **4.54e-6**, and no promotion of default weights.

Per-seed performance on music development segments (same window, 5-band, fixed FFT grid) averaged as follows:

| Path / Seed | Output HF SC ↓ | Output Flux ↓ | Output Energy Error (dB) ↓ |
| --- | ---: | ---: | ---: |
| factorized / 12 | 0.6964 | 0.8072 | 6.8698 |
| rms_residual / 12 | 3.7355 | 2.7455 | 5.4420 |
| factorized / 13 | 0.6189 | 0.7636 | 7.0865 |
| rms_residual / 13 | 2.7273 | 2.3186 | 5.8591 |

Per-recording bootstrap analysis shows that degradation in SC and flux is statistically significant (above zero) in 95% of cases across both seeds.
The improvement in energy error shows a cross-zero interval—indicating no net gain.
The decline in scoring does not constitute overall recovery benefit.

First-seed candidate ONNX: **3,277,624 B**, 807,398 parameters.
CPU frontend + inference (dual channel):
- 36kHz: **44.1k, 0.22883 ms**
- 48kHz: **48k, 0.22050 ms**
Independent NumPy/ORT RSS peak increment: **20.8125 MiB**.

These metrics are consistent with the earlier diagnostic range and do not represent full deployment latency or memory footprint.
360 local tests and corresponding remote CI passes were successfully completed.

Subsequent audit of [loss and weak high-frequency misfill](energy-objective-audit.md) reveals that the degradation in original SC is primarily driven by energy misfill—indicating that SC cannot be treated as a sole proxy for normalized spectral shape.
The new approach was not adopted.

The exact seeds are `20261012/20261013`; the new fixed sampling schedule uses `seed+50000`. This preserves the registered distinction between initialization and sampling randomness.
