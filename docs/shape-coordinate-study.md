# P9: Spectral Shape Update Scale and Accurate Identity Initialization

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

2026-10-13 (Singapore time), following [Linear Optimization Audit](linear-optimization-audit.md). The goal remains high-frequency reconstruction, not using direct pass-through as a substitute for restoration. This round does not modify the P8 loss, group tolerance, or material inputs; no model expansion or replacement of the default model occurs.

## Pre-training Mechanism Diagnostics

In the previous iteration, after correcting only the energy update scale, the spectral decoder still wrote each frequency bin using the full-spectrum RMS, then normalized the in-band spectral shape. In this round, we replay 128 steps using the original P8 draws, two seeds, and lr=1e-4, and repeat the training of the same four training fragments for 256 steps. We change the spectral update unit from full-spectrum RMS to input band RMS (with an existing 1e-4 full-spectrum relative floor), resulting in the following:

| Shared Development Score | 20261012 | 20261013 |
| --- | ---: | ---: |
| Initial | 0.494611 | 0.494611 |
| Energy unit corrected only, step 128 | 0.526563 | 0.521826 |
| Spectral unit corrected too, step 128 | 0.495014 | 0.493785 |

The short training shows a significant reduction in degradation, though the difference is small. Neither seed surpassed the initial value, so we cannot claim a general restoration improvement. The fixed small-sample final error is 0.289683 / 0.292236, which is actually higher than the 0.272663 / 0.271027 from energy-only correction. Therefore, we cannot judge the new coordinate as better across all dimensions based solely on development mean.

Subsequently, we split the paired/identity gradients from the original batch using CPU float64, and applied a single AdamW update to each group:

| Identity First Fragment Diagnostics | 20261012 | 20261013 |
| --- | ---: | ---: |
| Initial loss | 2.974e-8 | 6.206e-8 |
| Spectral gradient norm (original spectral unit) | 7.097 | 21.355 |
| Spectral gradient norm (band-level unit) | 0.00222 | 0.00588 |
| Identity loss after updating with original spectral unit gradient | 0.41674 | 0.47160 |
| Identity loss after updating with band-level unit gradient | 0.04913 | 0.04778 |

This indicates that the model's positive seed leads to initialization that approximates direct pass-through rather than exact pass-through, combined with the first-step overshoot introduced by Adam update scaling. The issue persists in double precision, and cannot be attributed solely to float32 rounding. The initial identity gradient is a small fraction of the paired gradient, and is not the sole cause of degradation in the paired training. Removing identity samples does not resolve this issue.

## Separate Evaluation of Two Changes

Let `s` be the full-spectrum RMS of the current compressed input, `r` be the RMS of the input band, and `d = 2*tanh(decoder)`.

1. **Spectral Unit Update**: Change `magnitude + s*d + s*1e-8` to
   `magnitude + sqrt(r² + (s*1e-4)²)*d + s*1e-8`, followed by smooth absolute value and in-band RMS normalization.
   All frequency bins in a given band share the same unit; empty bins can still generate structured output. Reference statistics or recording ID are not used.

2. **Exact Initialization**: Change the normalized output to
   `max(0, magnitude + (restored - restored_zero_heads))`.
   `restored_zero_heads` is the parsed mapping of the same input under zero output heads, without introducing a second neural network forward pass. When both heads are zero, the difference is strictly zero, and the output equals the compressed magnitude. Positive seeds and their generated gradients are preserved.

After exact initialization, the initial loss, gradients, and loss after one step of training for the identity samples are all zero. However, paired updates still raise the identity error to 0.02392 / 0.03050: this change only eliminates damage caused by initial bias, and **does not guarantee fidelity preservation in learned restoration**. Subsequent full validation of identity remains necessary, and a zero initial loss cannot serve as a valid proof of qualification.

The spectral head retains bounded tanh, so changing the unit also alters the reachable spectral range under finite parameter constraints; capacity equivalence cannot be assumed. Positive seeds allow missing frequency bins to produce non-zero gradients, and synthetic tests confirm the ability to generate structured high-frequency content. This does not prove that all real-world missing details can be expressed or restored. The energy branch retains its existing bounds, but the RMS of the output after zero-mapping residual correction is not strictly equal to the RMS before the energy head correction. Final evaluation still directly measures final magnitude and DSP output.

## Formal Comparison Protocol (Pre-Scoreing Fixed)

| Group | Energy Unit | Spectral Unit | Exact Initialization |
| --- | --- | --- | --- |
| energy_conditioned | Input band | Full-spectrum | No |
| band_conditioned | Input band | Input band | No |
| anchored_band | Input band | Input band | Yes |

Each group uses two seeds from 20261012 and 20261013, trained from scratch for 2048 steps, batch size 4. Within each seed, the initial parameter tensors match across groups, with 807,398 parameters per group. The only difference in the anchored_band group is the current variable. All share the same P8 linear amplitude + 0.25 signed dynamics loss, 80-hop warmup, past-only features, AdamW lr=1e-4, weight decay=1e-4, clip=5, and cosine decay to 0.1×.

For each seed, 8192 draws are generated using `seed+70000`, with 1024 assigned to identity, and all three groups share the same sampling. The music-to-voice ratio remains unchanged. Validation occurs every 512 steps, and the initial checkpoint can be selected. The same paired/identity and music/voice weighting rules are used, with the optimal checkpoint frozen before paired and grouped checks. A win at step 0 indicates no learned restoration weights were acquired.

Three controls are fixed: `band_conditioned` vs. `energy_conditioned`; `anchored_band` vs. `band_conditioned`; `anchored_band` vs. `energy_conditioned`. For each candidate relative to the two controls, 13 group checks are run, including relative improvements in high-frequency SC/energy compared to the original compressed input. The absolute error of the original identity is also preserved in each group report. These are research-level checks, not audible error benchmarks or release criteria. Deterioration in mean, group, or any control cannot prove restoration success.

Development recordings have been used in multiple rounds for diagnostics and cannot be claimed as independent tests or free from overfitting. Default weights remain unchanged.

## DSP, Cost, and Reproducibility

After formal training, the following is automatically executed:

- Full group, original spectrum, and context ablation evaluations; the selected model is not rewritten in subsequent reports.
- Export the ONNX model for the first seed in the anchored_band group, verifying dynamic single/dual-channel ORT correspondence, parameter count, and file budget; per-hop CPU inference and independent process RSS are verified. The benchmark does not overlap with the formal training.
- Six paths are evaluated: P3 phase, candidate input phase, P3 original model, and identity DSP. For each seed, 89 paired and 89 corresponding clean identity fragments are processed for **2,136 native DSP replays**. Paired and identity results are separately aggregated, preserving raw and re-analyzed five-band and full waveform metrics.
  Clean identity outputs may still be modified by phase and DSP, so exact magnitude initialization cannot be equated to final product identity.
- Proxy and native output error requirements are ≤2e-5, with checks for finite values and time alignment. Offline prediction replay does not prove real-time device stability; formal integration, continuous streaming validation, and blind listening are still required before release.

Local full Python test **394 cases passed**, with ruff checks passing. Two-step MPS smoke training has verified that all three groups can run. The best initial checkpoint and the non-zero step 2 checkpoint each completed 24 native replays (12 paired, 12 identity), with maximum proxy/native error of 6.71e-7. The second step was used only for non-zero path validation and did not alter the final model selection. The candidate smoke-selected step 0 ONNX is 3,284,671 bytes (3.13 MiB); for four configurations (44.1/48 kHz, mono/stereo), the frontend + inference P99 is 0.276 ms, with observed maximum 1.885 ms, and zero exceedances of hop budget. Additional peak RSS is 21.05 MiB.
This represents the CPU diagnostic cost for the initial checkpoint, excluding detector, DSP, adapter, and device overhead. Full results and non-zero training checkpoints still require revalidation. ONNX for non-zero random output heads has been separately tested.

After code submission, a frozen worktree is created; stage state, logs, configurations, sampling, and weights are preserved in `data/runs/shape-p9-20261012`. Audio and experimental weights are not pushed. The pipeline halts and records the reason upon stage failure.

```sh
PYTHONPATH=training training/.venv/bin/python training/scripts/run_shape_study.py \
  --parent data/runs/linear-p8-20261012/study/config.json \
  --output data/runs/shape-p9-20261012/study \
  --cache data/runs/factor-p6-20261012/study/cache
```

This document records the diagnostics and formal protocol; full training and quality results will be verified and reported after completion.

## Final Results

All six training runs and 2,136 native DSP replays have been completed. Checkpoint and evaluation file hashes have been verified.
- Both seeds in the `energy_conditioned` group selected step 0 (0.494611).
- `band_conditioned` achieved the best shared score of 0.491648 / 0.492389, and `anchored_band` achieved 0.491649 / 0.492390, selecting 2048 and 1536 steps respectively.
- Relative to the initial mean, improvements are only about 0.45%–0.60%, and exact initialization does not yield discernible final quality gains.

Music high-frequency energy error relative to the original compressed input increased by 0.099 / 0.195 dB; the original identity music magnitude error increased by approximately 0.0353 / 0.0357 (mean across recordings). Both control groups failed at least one group check. Actual DSP identity music spectral and transient errors also degraded, and thus **no promotion to default weights** occurred.
Some five-band energy metrics improved, but this does not equate to overall high-frequency or waveform improvement; the metrics differ in band, window length, and aggregation method. Full results are preserved, and no favorable metrics were selected.

Post-training, the ONNX model for the first seed is 3,291,199 bytes (3.14 MiB), with 807,398 parameters.
Frontend + inference peak P99 is 0.318 ms, with observed maximum 1.310 ms; independent RSS increase is 21.25 MiB.
Maximum proxy/native error across 2,136 replays is 4.54e-6. These performance and consistency metrics do not prove audio quality improvement.
Next round will be [P10: Final Product Supervision and Identity Fidelity](rendered-restoration-study.md).

The bounded decoder coordinate remains `d=2*tanh(decoder)`. The next historical study is [P10 rendered-output supervision](rendered-restoration-study.md).
