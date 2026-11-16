# P83: finite Adam updates for reconstructed phase relations

Completed 2026-11-17: all 16 trajectories, 256 updates, 800 fresh native roles and
320 correspondence checks. Reconstructed circular supervision learns and changes
the output, but does not establish a stable incremental spectral improvement over
matched old training. Old continuation itself improves fit risk while its transfer
spectral means rise. This is an unresolved optimization and transfer diagnostic,
not an architecture rejection or a qualified new weight.

## Changed hypothesis and fixed design

[P82](realized-phase-relations-p82.md) establishes different direct/reconstructed relational derivatives, but its
head-only directions do not establish whole-model or persistent-Adam behavior.
The [failure registry](failed-experiments.md) was consulted before this run. P83
keeps the old spectral objective for every parameter and adds circular relations
only to `phase_head.*`. It compares the same original-reference auxiliary before
synthesis, after raw overlap-add, and after DSP fusion. This is not another
unchanged P80 direct-only two-update retry, a capacity comparison or an experiment
that substitutes DSP output for the original reference.

Every origin has 216,802 trainable parameters. Origins are both P74 magnitude and
phase checkpoints for seeds 20261012 and 20261013. Four arms start independently
from each frozen checkpoint and its persistent FP32 Adam state:

| Arm | Objective beyond the unchanged joint spectral objective |
| --- | --- |
| Old | None; matched continued-training control |
| Direct | Circular relations in predicted complex features |
| Raw | Circular relations reanalyzed after raw overlap-add |
| Fused | Circular relations reanalyzed after model plus DSP |

The auxiliary is `0.1*log1p(mean(instantaneous, frequency, time))`. All three
domains share P82's original lossless Hann256/hop128 reference, frames 89--253,
availability, reference support and floors. Original direct/raw/fused magnitude
losses, clipping and paired/identity weighting remain unchanged. Wrapped loss is
not a candidate in this fixed follow-up; P82's speech conflicts remain recorded.
Different domain gradient norms are disclosed, not retrospectively normalized.

Each of the 16 trajectories completes exactly 16 updates with four draws per
update: **256 actual optimizer updates**. Alternating the same two four-draw groups
gives eight passes through the 40-recording fit bank. This reduces the inherited
first-moment coefficient from `0.9^2 = 0.81` to `0.9^16 = 0.185302`, while the
second-moment coefficient remains `0.999^16 = 0.984119`. It tests accumulation
beyond P80's first pair of steps; it does not establish convergence or an adequate
full-training budget. The stored base/hidden learning rate is `1e-5`, betas are
`(0.9, 0.999)` and global clipping uses canonical norm 5. All arms retain P77's
same gain-64 physical readout step and equivalent `W=64U` training coordinates.
There is no new scale selection, momentum reset, line search or early stopping.

All four old arms must reproduce their P80 step-2 canonical model, coordinate
model and coordinate optimizer exactly. That intermediate is a numerical control,
not a selected quality checkpoint. All 16 final checkpoint hashes and all 16
ONNX graph hashes are frozen before any fit/output/transfer scoring.

## Evaluation and scope

Fit contains 32 music and eight speech recordings, with 18.724499 seconds of
unique MR-supervised support. Transfer contains 16 music and four speech
recordings, with 9.283955 seconds. Repetition does not increase unique duration.
The sets are source/work-separated within this diagnostic but were historically
eligible for training. This is a development recheck, not unseen generalization.
Speech is 20% of draws. Channel views are individually aligned mono, not an
artificially joined stereo pair or physical stereo validation.

Raw/proxy scores cover every fit and transfer draw. Native scores cover one fit
draw (five recordings) and all four transfer draws (20 recordings). The run
produces 800 fresh native roles and reuses 250 hash-bound parent/identity-DSP
roles after exact input, rate and timeline checks. Each endpoint/draw traces the
first paired music, paired speech, identity music and identity speech roles:
320 PCM and analytic phase-risk correspondence checks. The historical scores
must reproduce exactly; protocol-v2 metrics are additive. Recorded render timing
is not a new callback, deadline or device qualification.

The fixed primary comparison is raw versus matched old native paired transfer
at all four origins. Direct/fused, placement controls, frozen parent, untreated
and identity DSP remain separate comparators. Historical legacy is not rescored.
Spectral MR/HF convergence, shape, energy, envelope dynamics, within-band dynamics,
coherent and calibrated absolute errors remain visible, separately for music,
speech and clean identity. No phase-sign or machine-epsilon gate is applied.
All fit endpoints use the same common risk and each separate relation ruler;
differently weighted training totals are never compared as quality scores.

The initial registration used bootstrap seed 83 while the common protocol
specifies 71. Before endpoint freezing or any scoring, a supplemental registration
fixed a complete seed-71 reaggregation. Both sets of intervals are retained;
paired means and sources must agree exactly. This administrative discrepancy is
not repaired by choosing whichever interval is favorable. Neither seed proves
numerical resolution, audible relevance or independent generalization.

## Actual learning and fit preservation

All four step-2 model/optimizer controls are bitwise exact. All 192 auxiliary-arm
updates have negative auxiliary-gradient dot actual Adam displacement. No step is
clipped: canonical gradient norms range from 0.846514 to 2.562499. The old-gradient
dot displacement is negative on 252/256 updates. Seed-20261013 phase step 2 is
positive in every arm, including old, from `5.871057e-6` to `5.938165e-6`. These
are directional diagnostics, not measured finite loss changes. Update telemetry
uses training-coordinate displacement; the summary separately records canonical
base/head endpoint differences.

Against matched old endpoints, paired-fit own relation risk falls at all origins:
direct **0.025304--0.027683%**, raw **0.011558--0.017921%**, fused
**0.010963--0.015162%**. Paired common risk also falls slightly in every new arm.
Neither a disconnected loss nor a universally erased auxiliary Adam step explains
the small output effects.

Every new arm has a higher clean-fit common-risk mean than old. Raw clean relation
risk rises by 0.040368%, 0.281161%, 2.665923% and 3.636122%, respectively. Absolute
increments are `6.096417e-9`, `3.084043e-8`, `5.428922e-8` and `5.618045e-8`
in dimensionless relation-risk units; the last three seed-71 intervals are above
zero. These are not PCM dBFS or audible-error estimates. Lower mixed/paired risk
does not guarantee lower clean risk; no clean example is removed.

## Native incremental results

Values are percent changes of source-equal means relative to each origin's
**matched 16-step old control**, on the 20-recording paired transfer panel.
Negative is lower error. Both MR and HF-SC 95% intervals cross zero for **all 12
new endpoints**, under both bootstrap seeds. No epsilon or coherent-sign gate
turns these unresolved spectral results into acceptance or rejection.

| P74 origin | Arm | MR change | HF SC change | Coherent error change |
| --- | --- | ---: | ---: | ---: |
| Seed 12 magnitude | Direct | -0.00041907% | +0.00035411% | +0.00114965% |
| Seed 12 magnitude | Raw | +0.00000280% | +0.00028338% | -0.00025067% |
| Seed 12 magnitude | Fused | +0.00002923% | +0.00017260% | -0.00032254% |
| Seed 12 phase | Direct | -0.00052495% | +0.00036444% | +0.00098251% |
| Seed 12 phase | Raw | -0.00008217% | +0.00020587% | -0.00040309% |
| Seed 12 phase | Fused | -0.00000945% | +0.00012764% | -0.00043738% |
| Seed 13 magnitude | Direct | -0.00020003% | -0.00002968% | +0.00061581% |
| Seed 13 magnitude | Raw | -0.00021468% | +0.00006188% | -0.00046768% |
| Seed 13 magnitude | Fused | -0.00023049% | +0.00002317% | -0.00059500% |
| Seed 13 phase | Direct | -0.00017371% | +0.00015377% | +0.00071542% |
| Seed 13 phase | Raw | -0.00010596% | +0.00007554% | -0.00036094% |
| Seed 13 phase | Fused | -0.00007826% | +0.00000752% | -0.00047537% |

Direct coherent intervals increase at all four origins. Raw/fused coherent means
decrease at all four, but each has only one interval entirely below zero. Raw
versus direct has a coherent interval below zero at every origin. This supports a
placement effect on this ruler, without the missing spectral/perceptual evidence.

Raw model-only MR decreases at all origins, with intervals below zero for the two
seed-13 origins. Yet model-only HF-SC means also rise at all four. The HF-SC problem
already exists before DSP; it cannot be assigned solely to DSP erasing an otherwise
complete high-band gain. Proxy/native primary directions remain unresolved.

For raw versus old, native squared-shape and envelope changes have opposite mean
signs across seeds. Envelope-dynamics means decrease at all origins, whereas
within-band spectral-dynamics means increase at all four. Music MR means decrease
at all four; speech MR rises at seed 12 and falls at seed 13. Speech's envelope
dynamics worsens in mean at every origin despite slightly lower within-band
dynamics. Clean-native primary intervals remain unresolved for raw; its coherent
mean decreases at seed 12 and increases at seed 13. All preservation, absolute,
band and per-recording fields remain in both full assessments.

## Continuation versus parent and untreated input

The incremental table must not hide the larger shared continuation effect. Old
means the 16-step control; parent means the frozen fully trained P74 start.

| Origin | Native transfer MR vs parent | HF SC vs parent | Coherent error vs parent |
| --- | ---: | ---: | ---: |
| Seed 12 magnitude | +0.042517% | +0.110365% | +0.965106% |
| Seed 12 phase | +0.034281% | +0.080494% | +0.923115% |
| Seed 13 magnitude | +0.084778% | +0.129666% | +0.443055% |
| Seed 13 phase | +0.072042% | +0.139432% | +0.392088% |

All old paired-fit common-risk intervals decrease. Transfer MR/HF-SC intervals
remain unresolved, while coherent intervals increase at every origin. The new
auxiliaries' effects are much smaller than this shared change. This does not mean
every candidate loses to historical legacy, which is absent from the experiment.
It does not establish overfitting or convergence: finite crop exposure, copied
optimizer, gain-64 readout and changing base are not isolated by this comparison.

Against untreated compressed input, raw-arm native all-source MR means improve
1.42--2.07%, while HF-SC means worsen 19.45--19.87%; these primary intervals all
cross zero. Shape means improve 20.68--21.47%, with two intervals below zero, but
HF PCM RMS errors rise 3.41--3.52%, with intervals above zero at all origins.
Music-only MR means worsen 1.60--2.20% and HF-SC means worsen 27.73--28.41%, with
unresolved primary intervals. Four speech recordings instead show MR/HF-SC
intervals below zero at all origins, alongside higher HF PCM error. This reused,
short panel does not establish a universal music/speech or genre conclusion.

Against identity DSP, the same raw-arm native MR means improve 6.67--7.29% and
HF-SC means improve 5.41--5.74%. Identity-DSP improvement cannot replace untreated
and original-reference tests. These comparators expose a remaining spectral and
fidelity tradeoff, not a hidden overall restoration win.

## Counterexamples and correspondence

The worst raw-versus-old HF-SC increment at every origin is music case `10/1`,
source `51019ced2a92e4b7096c9eeb257296c3b6d772b3836a7d48abd2fd533ff298fe`.
Absolute SC increments are `2.084634e-5`, `1.323618e-5`, `9.440272e-6` and
`1.393934e-5`. At seed-12 magnitude, HF PCM RMS error changes from
`1.337603847e-4` to `1.337613521e-4`; calibrated multiresolution reference HF
power is `9.309215e-9`. A difference of RMS errors is not the RMS of the waveform
difference. These weak-band values neither prove audible severity nor justify
ignoring the spectral counterexample. Every adverse source and exact score is
retained, not just this largest case.

All 320 correspondence checks pass existing PCM and analytic phase-risk bounds.
Maximum PCM discrepancy is `4.172325e-7`, maximum RMS `2.413305e-8`. Circular risk
difference peaks at `4.294814e-5` and wrapped at `3.371983e-5`; every individual
case/component satisfies its own bound. Maximum bounds are `0.002326233` and
`0.001480926`, not general score-error tolerances. There are 112 zero-bound and
zero-discrepancy cases and no supported exact-zero prediction. Traces cover five
music and five speech recordings in paired/identity views, not all transfer
sources or physical stereo. All 250 reused control-role scores reproduce after
exact input, rate and timeline checks.

The review contains 5,970 output records, 1,600 fit-risk records, 1,680 output
comparisons and 216 fit comparisons. Seed 71 versus seed 83 preserves all
**104,256 paired metric records**, source sets and counterexamples exactly.
There are 135 changed descriptive interpretation dictionaries elsewhere in the
grid, reflecting interval sensitivity to the fixed 2,000 bootstrap resamples.
All 12 main native spectral comparisons remain unresolved under both seeds;
neither seed is selected according to outcome.

## Decision and next diagnostic

P83 is complete and unqualified. Reconstructed circular relations are connected
and more favorable than direct circular on the coherent ruler, but no stable
incremental native spectral gain is established. This is not an automatic phase
veto, a disconnected derivative or gradient-clipping failure. It is not a capacity
ceiling or a reason to discard the architecture.

Do not extend the same short-bank phase recipe merely because it learns its own
loss. First isolate the shared continuation effect: compare unchanged-objective
canonical versus gain-64 updates from the same frozen parents/moments, and
distinguish repeated intervals from fresh training-only crop exposure without
changing the budget or touching reserved evaluation sources. This differs from
selecting a circular-loss coefficient or a larger model. Keep original-reference
magnitude/shape, absolute weak-band, music/speech, identity and native rulers fixed.
Broader training needs a new exposure/convergence rationale and a complete matched
budget before scoring.

Music magnitude/energy restoration and fit-to-output transfer remain the primary
targets. Retain phase placement as mechanism evidence, without treating it as the
principal missing spectral fix. Defaults, graph structure, causal history and
fixed buffering are unchanged. There is no new timing, device, independent-test
or blind-listening qualification. No evidence or training source was deleted.

## Evidence and validation

- `data/runs/realized-relation-updates-p83-controls-20261117/`: frozen source and
  log for 58 passing focused tests, with warnings treated as errors.
- `data/runs/realized-relation-updates-p83-20261117/`: registration, source,
  trajectories, endpoints, exports, native roles and registered seed-83 assessment.
- `data/runs/realized-relation-updates-p83-summary-20261117/`: prospectively
  registered seed-71 descriptive review and complete source counterexamples.

Controls cover original-reference relation gradients, silence/weak material,
finite differences, coordinate-equivalent Adam with each new auxiliary, fixed
exposure, unchanged score arithmetic and strict v2 aggregation. Original audio,
checkpoints, failed history and frozen evidence remain local and ignored. Source
and report changes are English; no training data or experimental weight is
published.
