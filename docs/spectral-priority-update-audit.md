# P43: spectrum-led update control

## Scope and changed intervention

[P42](pooled-update-audit.md) exposed a concrete mismatch: its equally normalized
amplitude/complex bisector could improve high-band PCM error while worsening
magnitude spectra. P43 makes the direction explicitly spectrum-led and retains
the original clean targets and finite-step fidelity checks.

This is a **post-hoc mechanism control on the same P42 training roles**. The
direction was chosen after inspecting P42, so these are reused diagnostic draws,
not independent transfer evidence. The two frozen checkpoints, 32 cases, four
fit batches, separate step-guard and transfer batches, all four radii and both
step policies stay fixed. Every baseline metric and calibrated PCM value must
match P42 exactly before its case can be recorded.

The unguarded direction is the negative normalized sum of unit-normalized pooled
**raw direct-amplitude and MR gradients**. The guarded variant projects it against
amplitude, MR, MR dynamics, identity MR, identity risk and budget gradients.
Complex error is no longer part of the bisector or a first-order projection
constraint. It remains measured and protected by the unchanged strict/0.1%
finite-step policies, together with the original total. Phase supervision was
not deleted, and no unrestricted phase regression is silently accepted.

This is a direction-design intervention, not a new scalar loss. Relative to P42,
it changes both the primary gradient carriers and the projection guard set; it
cannot attribute a gain to either change alone. Comparing its guarded and
unguarded variants isolates the effect of its six guards within this design.
Neither restores missing information exactly or establishes model capacity limits.

The same numerical allowance, explicit no-op policy, accepted-only/all-case
reporting and calibrated full-band/>=8 kHz raw PCM definitions apply as in P42.
Transfer metrics never select a radius. No weights are saved or promoted, and
no actual Rust/DSP streaming or listening claim follows from these probes.

## Evidence and verification

Evidence is retained under `data/runs/spectral-priority-p43-20261023/`, including
registration, historical/current source hashes, executed snapshots, 32 case
reports, all rejected radii and summaries. Entry point:
`training/scripts/audit_spectral_priority.py`; the common summary tool is
`training/scripts/summarize_pooled_updates.py`. P42 snapshots remain immutable
after making the common runner's direction builder injectable.

## Completed results

All 32 cases completed on 2026-10-23; every baseline matched P42 exactly. At the
same radius 0.001, the guarded spectrum-led direction improves transfer amplitude
on **28/32** and MR on **29/32**, versus **13/32** and **14/32** for P42's guarded
amplitude/complex bisector. This is a positive local spectrum-priority signal on
reused training draws, not proof of multi-step convergence or independent quality.

| Direction | Radius | Amplitude improves | Complex improves | MR improves | Dynamics improves | Identity MR improves |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Spectrum | 0.01 | 27 | 0 | 28 | 5 | 0 |
| Spectrum | 0.003 | 27 | 0 | 28 | 7 | 0 |
| Spectrum | 0.001 | 27 | 0 | 28 | 7 | 0 |
| Spectrum | 0.0003 | 27 | 0 | 28 | 7 | 0 |
| Spectrum guarded | 0.01 | 28 | 1 | 29 | 10 | 5 |
| Spectrum guarded | 0.003 | 28 | 1 | 29 | 12 | 6 |
| Spectrum guarded | 0.001 | 28 | 1 | 29 | 12 | 6 |
| Spectrum guarded | 0.0003 | 28 | 1 | 29 | 12 | 6 |

No row improves all five metrics together. At radius 0.001, the guarded direction's
median transfer changes are:

| Metric | Seed 20261012 | Seed 20261013 |
| --- | ---: | ---: |
| Direct amplitude | −0.02050% | −0.01745% |
| MR spectrum | −0.00856% | −0.01293% |
| Direct complex | +0.01793% | +0.01395% |
| MR dynamics | +0.00244% | +0.00167% |
| Identity MR | +1.19351% | +0.76787% |

The unguarded spectrum direction gives larger spectral gains but worse identity,
complex and dynamics tradeoffs. The guards reduce that damage without eliminating
it; the small single-step medians cannot be extrapolated over thousands of updates.

## Selected steps and absolute fidelity evidence

The strict policy accepts **0/32** updates for either direction. That is a stalled
diagnostic policy, not successful preservation through restoration. The 0.1%
policy accepts zero unguarded updates and **8/32** guarded updates (6/16 and 2/16).
Their transfer amplitude and MR improve on **8/8**, dynamics on 6/8 and identity
MR on 2/8; complex error increases on 8/8. The other 24 cases are explicit no-ops.
Accepted-only medians are selection-dependent and do not replace the fixed-radius
table or establish broad generalization.

At radius 0.001, the 0.1% policy's main remaining blocker is separate-guard identity
MR: 10/16 and 15/16 cases exceed its relative budget. At radius 0.0003, the counts
remain 10/16 and 14/16. A purely relative identity budget near an almost-exact
baseline can reject small absolute changes. That warrants calibrated assessment,
not quietly loosening a threshold around these known cases.

Among the eight accepted updates, mean raw full-band identity PCM error RMS
increases on all eight; maximum increases are **1.890e-7 / 7.827e-7** by seed.
Mean paired high-band PCM error RMS improves on 5/8 and worsens on 3/8. These are
calibrated error statistics, not the waveform of the update or a perceptual rating.

Retained counterexamples illustrate the actual scale:

- Seed 20261012, fit anchor 10626, transfer 7927, radius 0.001: transfer amplitude
  decreases 0.02946% and MR 0.01044%, but identity MR increases 3.2283%. Mean raw
  identity error RMS changes from 1.03145e-5 to 1.05036e-5, corresponding to
  −99.7310 to −99.5733 dBFS.
- Seed 20261013, fit anchor 5566, transfer 2869, radius 0.01: transfer amplitude
  decreases 0.11220%, MR 0.11222% and dynamics 0.02404%; identity MR increases
  7.5752%. Mean identity error RMS changes from 1.33349e-5 to 1.41176e-5,
  or −97.5002 to −97.0048 dBFS. Complex error increases 0.001675%.

Those dBFS values are levels of mean per-example error RMS, not per-recording
maxima or proof of audibility. Both spectral gains and fidelity regressions remain
in the evidence. No case-specific acceptance threshold is introduced.

## Interpretation and next gate

P43 supports investigating a spectrum-led small-model training method. P42's equal
geometric balancing was a diagnostic device and did not faithfully express the
user's spectrum-first priority. The difference is already visible in raw predictions
before deployed DSP, so changing DSP alone cannot explain this particular result.
P43 does not establish that the loss implementation is wrong or the architecture
has reached a capacity limit.

Before a sustained trajectory, define preservation budgets using calibrated
absolute errors alongside relative spectra, on an independently fixed calibration
protocol. Do not derive a pass threshold from the two examples above. The bounded
0.1% policy is still diagnostic, not a justified optimizer acceptance rule: its
24/32 no-ops and accepted transfer regressions must remain visible.

Then compare a spectrum-led small-model method with the unchanged optimizer under
fixed exposure, complete matched budgets and frozen endpoint selection. Report
music/speech and original/raw/native-DSP paths separately; preserve low-band,
stereo, transient and continuity checks. No current result qualifies weights or
demonstrates accumulated training gains. Default weights, runtime and DSP remain
unchanged; real-time scheduling issues remain open.

Local verification: **677 tests passed**, including known-answer PCM/Parseval
checks, role isolation, no-op accounting and an opposed-complex-gradient direction
test. Ruff and all 121 English documentation/link checks pass. The full suite uses
the documented process-local functional FFmpeg; existing ONNX/tracing/MPS warnings
remain. No new native performance benchmark was run.
