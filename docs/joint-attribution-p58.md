# P58: where joint training gains and fidelity losses occur

Date: 2026-10-26. Status: completed descriptive attribution, not qualified weights.
This follows [P57](joint-training-p57.md) and informs the
[implementation review and texture prototype](implementation-review-20261026.md).

## Scope and reproducibility

Entry point: `training/scripts/audit_joint_attribution.py`. Evidence:
`data/runs/joint-attribution-p58-20261026-v2/`.

The audit consumes all 2,720 previously frozen P57 long-form records. Both seeds
are compared with their matched adapter, starting parent and historical model,
separately for paired/identity and raw/native output. Every case and recording
delta is retained. Cases must match source, codec, rate, physical channel count
and scored support; sources receive equal weight after averaging their cases.
Existing MR components and frequency boundaries are unchanged. This is arithmetic
decomposition of exposed P55 results, not another independent test or metric search.

Input checks bind the P57 completion receipt, endpoint and cost summaries,
long-form completion receipt, score summary and four checkpoint hashes. This is
not a new recursive verification or rendering of every audio artifact. Output
registration and summary have a completion receipt. Stage means raw OLA versus
actual Rust output, not training epoch: intermediate generator checkpoints are
not available for a retrospective optimization-stage attribution.

## The highest band carries both benefit and damage

All changes below compare full training with the matched adapter control.
The percentages are signed contributions to the existing weighted MR change,
not percentages of perceived quality or audible distortion.

| Native paired music | Seed 20261012 | Seed 20261013 |
| --- | ---: | ---: |
| MR change | -0.003187741 | -0.003253858 |
| MR relative change | -0.6228% | -0.6369% |
| Contribution from 16+ kHz to MR reduction | 69.93% | 72.94% |
| Reference dynamics change | +0.000390076 | +0.000609150 |
| Contribution from 16+ kHz to dynamics increase | 103.86% | 98.97% |
| Sources with better / worse MR | 16 / 4 | 14 / 6 |
| Sources with better MR but worse dynamics | 13 / 20 | 13 / 20 |

The contribution can exceed 100% because improvements elsewhere cancel part of
the high-band regression. Dynamics measures error in reference magnitude changes;
it is not a human transient rating. At 8-12 and 12-16 kHz, raw paired dynamics
improves in both seeds. Above 16 kHz it worsens before DSP already.

Native 16+ kHz paired PCM error RMS changes from 0.002968333 to 0.003004177
and from 0.002969936 to 0.003012976. These are means of per-source calibrated
band RMS values, not pooled whole-corpus RMS. The corresponding reference mean
is 0.003163114. The spectrum improves while complex waveform error increases;
band energy or magnitude similarity cannot stand in for correct phase/detail.

For clean identity audio, 16+ kHz contributes **97.36% / 92.17%** of the native
MR increase. Its PCM error RMS grows from 0.000132175 to 0.000140535 and from
0.000140908 to 0.000151383, absolute increments 8.361e-6 / 1.047e-5. These
increments are not an audibility threshold. Native identity MR worsens on
14 / 13 sources; all source-level exceptions remain in the report.

## DSP is involved, but the model already contains the tradeoff

The native MR gain is 77.54% / 79.98% of the corresponding raw gain. This ratio
compares differences between two models, not a general DSP efficiency score.
Native identity MR growth is 53.05% / 57.43% of its raw counterpart. DSP therefore
attenuates both useful changes and some damage on this set. The raw paired
high-band dynamics regression and raw clean-spectrum drift rule out attributing
the entire problem to postprocessing. Nonlinear stage differences alone cannot
identify the responsible limiter, mask, phase blend or learned tensor.

The one P57 paced failure remains in this report: seed 13 full, 48 kHz stereo,
completion late by 0.468495 ms. No timing was rerun to remove it, and no scheduler
cause is inferred from aggregate timings. The 2.238792 ms maximum compute and
fixed 256-frame buffer remain separate from callback/device qualification.

## Retained failed audit and numerical correction

The first directory, `data/runs/joint-attribution-p58-20261026/`, stopped while
checking reconstruction of near-zero score differences. It applied an absolute
2e-7 tolerance to the subtraction of two independently rounded float32 scores.
Opposite endpoint rounding can exceed that bound despite each score being valid.
The failed registration, source and failure record are retained.

The corrected audit verifies each original score with an operation-scale
float32 rounding bound before subtracting endpoints, then reports the residual
of the difference explicitly. Tests cover cancellation, corrupted scores,
duplicate/mismatched cases and unequal case counts per source. No audio score,
quality gate, source exclusion or model selection rule changes.

## Decision

Do not repeat adapter-only continuation, suppress all 16+ kHz output, remove
identity protection, or enlarge the same network from this evidence. The next
mechanism to test is paired high-frequency spectrotemporal feature supervision,
with a small critic used only during training. The research/prototype document
defines the reference-correct behavior and the remaining matched-training gates.
This attribution establishes neither texture realism nor the cause of the phase
error; those remain hypotheses to test. Default weights and DSP are unchanged.
