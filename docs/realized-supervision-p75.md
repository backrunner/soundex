# P75: realized-spectrum supervision research and frozen gradient audit

Date: 2026-11-13. This work investigates the next step after
[P74](phase-temporal-p74.md). It does not change training weights, the inference
graph, DSP policy, default artifacts or acceptance thresholds. Research conclusions
and local derivative diagnostics are not restoration-quality claims.

Follow-up: [P76](phase-routing-p76.md) completes routed finite Adam and native
transfer checks. Its incremental output changes are tiny and mixed, so the
registered advancement gate fails without establishing a phase-family limit.

## What the current loss actually supervises

The current P67/P72/P74 objective is the equally balanced paired/identity mean of
`log1p(direct) + log1p(raw MR) + log1p(fused MR)`. All references remain the
aligned original lossless recording, including loud references affected by the
deployment limiter. The three paths have different meanings:

| Path | Representation and supervision | Limitation relevant to P74 |
| --- | --- | --- |
| `direct_spectrum_objective.py` | Predicted 256-point frame spectra against the clean frame spectra; amplitude 1, complex 0.1, envelope 0.1, dynamics 0.25 | Complex agreement is measured before overlap-add and DSP. Arbitrary predicted frame spectra need not be the spectra realized by synthesis. |
| `multires_spectrum.py` | Raw overlap-add audio, reanalyzed at 256/1024/2048 points; magnitude, log shape, envelope and signed magnitude dynamics | Already supervises realized spectral magnitudes, but does not directly compare complex spectra after synthesis. |
| `fused_spectrum.py` | Same multiresolution magnitude terms after native-compatible band fusion, synthesis, limiting and gating | Constrains actual output magnitudes against the original; explicit realized complex agreement is absent. Phase still receives indirect gradients through interference. |

Therefore, “we forgot to train on the synthesized spectrum” and “there is no phase
loss” are both incorrect. The narrower hypothesis is that the existing direct
complex term may not supply a useful fidelity constraint for the final output,
while two realized magnitude terms dominate the remaining tradeoff. Relative
coefficient values alone do not establish gradient dominance.

This is not the first complex/consistency experiment. Historical
[V4](optimization-20261011.md) already combined complex reconstruction and
consistency in another model and obtained mixed energy/quality results.
[P10](rendered-restoration-study.md) used a different magnitude model and an
identity-DSP target. [P67](fused-phase-p67.md) established the present original-PCM
fused magnitude proxy. [P68](phase-gradients-p68.md) audited a frozen-magnitude
phase head. The changed question here is about fully trained P74 joint models,
realized high-band complex gradients, and their relationship to the unchanged
spectrum-first objective. None of those older studies establishes this answer.

## Research findings and relevance

Sources were checked on 2026-11-13. Their reported benchmarks have different data,
latency definitions and model budgets; they are evidence for mechanisms, not a
ranking of systems under SoundEx's requirements.

| Primary source | Useful mechanism | Boundary for SoundEx |
| --- | --- | --- |
| [Differentiable Consistency Constraints](https://arxiv.org/abs/1811.08521) | Differentiate through synthesis and reanalysis so supervision sees realizable spectra. | Speech enhancement/separation. Mixture-sum consistency is not a bandwidth-extension target. SoundEx already does reanalysis for magnitudes; adding a consistency penalty alone is not a new recovery target. |
| [AP-BWE](https://github.com/yxlu-0102/AP-BWE) and its [training implementation](https://github.com/yxlu-0102/AP-BWE/blob/main/train/train_48k.py) | Joint amplitude/phase streams; magnitude, wrapped phase, complex and self-consistency losses, plus adversarial/feature terms. Validation reanalyzes generated audio. | Speech bandwidth extension. The [generator](https://github.com/yxlu-0102/AP-BWE/blob/main/models/model.py) uses symmetric temporal convolution padding, so its throughput claim is not proof of causal hop deadlines. Its self-consistency term compares prediction with reanalysis, not directly with the original reference. |
| [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) and its [loss implementation](https://github.com/Rikorose/DeepFilterNet/blob/main/DeepFilterNet/df/loss.py) | A hybrid learned/DSP system; optional multiresolution magnitude and complex terms after waveform reconstruction. | Full-band speech denoising, not music inpainting. [Defaults](https://github.com/Rikorose/DeepFilterNet/blob/main/DeepFilterNet/df/config.py) use FFT960/hop480 at 48 kHz. A noise-removal filter cannot be assumed to generate absent high-frequency detail. |
| [AERO](https://arxiv.org/abs/2211.12232) | Complex spectral prediction with reconstruction and adversarial/feature supervision, evaluated on speech and music. | Relevant representation evidence, but not validation of our causal CPU streaming budget or codec degradation distribution. |
| [AEROMamba with PAQM, 2026](https://arxiv.org/abs/2608.02918) | Includes heavily compressed MP3 music and differentiable auditory supervision, rather than treating ideal low-pass upsampling as the entire problem. | Its efficiency table measures 10-second excerpts on RTX 3090, with about 21M parameters. It does not establish per-hop CPU latency. The paper discloses MUSDB test-set model selection; its reported gains are not an unbiased independent-test guarantee for our use. |

The latest PAQM work is a useful second direction, not a reason to remove paired
spectral targets here. Its codec variant retains adversarial/feature losses while
replacing spectral reconstruction. That combination has not been tested in our
causal model. Our [P59 failures](texture-calibration-p59.md) remain relevant to
classifier calibration but do not disprove all perceptual/feature supervision.
`auraloss==0.4.0` is already pinned and used as an independent ruler; changing a
library name is not a mechanism for resolving the amplitude/phase tradeoff.

No external weights, datasets or source code were imported. AP-BWE's current
repository declares MIT for code and [weights](https://github.com/yxlu-0102/AP-BWE/blob/main/weights_LICENSE.txt);
that declaration alone is not a training-data provenance audit. The PAQM
implementation has conflicting visible notices: its [README](https://github.com/bvm810/torchpaqm)
says MIT, while [LICENSE](https://github.com/bvm810/torchpaqm/blob/main/LICENSE)
contains GPLv3. Do not copy it into this Apache-2.0 project without resolving the
licensing/provenance route. This observation does not assert that merely using a
training tool automatically licenses its output weights under the tool's license.

## Why simply increasing complex loss can also fail

For one bin with predicted magnitude `r`, reference magnitude `a` and phase
error `delta`, complex squared error is
`r^2 + a^2 - 2*r*a*cos(delta)`. Holding phase fixed, the optimal nonnegative
magnitude is `a * max(cos(delta), 0)`. Large unresolved phase error can therefore
make attenuation reduce complex error. That would conflict with high-frequency
restoration, even when the complex score improves.

This algebra motivates retaining independent magnitude, shape, band energy and
transient targets, and reporting coherent error separately. It does not prove
that P74 is at a conditional-mean optimum or that missing phase is unlearnable.
A phase-only auxiliary is also not automatically safe: changing phase changes
overlap-add interference and therefore realized magnitudes. Output-domain controls
and actual native transfer remain necessary.

Self-consistency is different again: minimizing the distance between predicted
and reanalyzed spectra can make an incorrect signal self-consistent. It cannot
replace the original-reference restoration loss. There is no justification for
requiring arbitrary model spectra to be perfectly consistent before deployment
or for discarding small, acceptable numerical errors.

## Frozen diagnostic protocol

`evaluation/realized_complex.py` adds a diagnostic ruler only. It reuses the
existing MR resolutions, unpadded windows, Parseval PCM scaling, reference power
floors and supervised support. The three bands are 8–12, 12–16 and 16 kHz to
Nyquist. Equal band/resolution smooth relative norms are recorded with calibrated
absolute magnitude-error, complex-error, phase-interaction and reference powers.
The complex-power decomposition is descriptive, not a phase-only causal effect.

`scripts/audit_realized_supervision.py` registers all four frozen P74 endpoints
and the same first eight TRAIN draws: 32 endpoint/draw probes. Every draw and role
remains; this is a finite fit diagnostic, not new held-out testing. Gradients are
reported over all parameters and separately for the base and appended phase head.
The existing direct complex contribution includes its exact `log1p(direct)`
chain-rule scale. Other terms include raw/fused magnitude MR and realized complex
error, with paired and identity contributions visible.

The hypothetical direction is fixed before probing:
`gradient(old) + 0.1 * gradient(log1p(realized fused HF complex))`.
The coefficient copies the existing direct complex coefficient as a diagnostic
reference, not as an optimal scale. No weight search, optimizer update, rerendered
validation, altered DSP or new timing measurement occurs. Parameter-gradient
cosines are local and coordinate dependent; they do not predict Adam displacement
or quality after training.

Fourteen numerical/renderer tests pass, including original-reference zero and
stationarity, silence/weak gradients, phase-inverted equal-magnitude controls,
calibrated sine power, feature finite differences, warmup exclusion, limiter
conflict and full-model double-precision gradient decomposition. These tests do
not substitute for native transfer or a meaningful restoration comparison.

## Completed diagnostic and retained numerical failure

The first FP32 attempt, `data/runs/realized-supervision-audit-p75-20261113/`,
retains its registration, frozen source, 19 completed probes and `failure.json`.
At seed 20261013 magnitude / draw 3, separately accumulated gradients versus a
joint backward differed at one of 216,802 entries by 9.173527e-7. The existing
`rtol=2e-4, atol=2e-7` comparison rejected it. No optimizer step occurred and no
model-quality conclusion was taken from those partial results.

The separate `data/runs/realized-supervision-audit-p75-20261113-fp64/` registration
binds the original failed records, lifts the same frozen tensors exactly to FP64
and retains the original tolerance. All 32 probes finish, including that case;
maximum gradient-decomposition error is **5.552e-15**. This supports floating-point
accumulation as the cause of the numerical rejection. It is a precision correction
for the diagnostic, not an inference precision change or a relaxed criterion.
Input/source hashes verify before and after; every endpoint remains unchanged,
and the complete receipt verifies. No failed timing grid was rerun.

The fixed eight draws contain 32 music and eight speech crop specifications,
each with paired and identity roles. Repeating them across endpoints creates no
additional source coverage. The following are Euclidean **parameter-gradient**
cosines over those 32 endpoint/draw observations, not independent audio trials.
Negative cosine means local descent for one scalar increases the other to first
order; it is not a finite Adam or perceptual result.

| Whole-model gradient comparison | Median cosine | Negative cases |
| --- | ---: | ---: |
| Existing direct complex contribution versus realized fused HF complex | +0.990040 | 0/32 |
| Existing fused magnitude MR versus realized fused HF complex | -0.451736 | 19/32 |
| Old total versus realized fused HF complex | +0.477465 | 9/32 |
| Old + fixed 0.1 auxiliary versus realized fused HF complex | +0.595804 | 9/32 |
| Old total versus fused magnitude MR | +0.964563 | 3/32 |
| Old + fixed 0.1 auxiliary versus fused magnitude MR | +0.948674 | 3/32 |
| Old total versus realized HF magnitude | +0.973474 | 3/32 |
| Old + fixed 0.1 auxiliary versus realized HF magnitude | +0.974061 | 4/32 |

This **weakens the initial hypothesis that pre/post-synthesis placement alone is
the missing signal**: the existing direct complex gradient is already strongly
aligned with the realized one in every whole-model probe (range 0.967460–0.997879).
Simply adding a realized term mainly changes an existing tradeoff. The unchanged
total conflicts with realized complex error on nine probes; the fixed auxiliary
does not remove those nine conflicts. Adding it cannot be called an improvement.

The extra adverse HF-magnitude direction is seed 20261013 magnitude / draw 3:
cosine changes +0.075758 to -0.048339. Its fused-MR cosine stays positive but drops
from +0.151905 to +0.028503. Other negative MR directions already existed in the
old total and must not be incorrectly attributed entirely to the auxiliary:
seed 13 magnitude draw 0, and seed 13 phase draws 0/3. Draw 6 in every endpoint
retains a strongly negative old/realized-complex direction (about -0.98).
All sample identities, per-example roles, absolute band powers and remaining
counterexamples are saved in `probes/` and `assessment.json`.

The appended phase-head group has a different pattern. Existing direct versus
realized complex median cosine is +0.545632, with 5/32 negative. The old total
versus realized complex has 7/32 negative; the fixed auxiliary reduces that to
3/32 within this group. Both old and augmented head directions align with fused
MR and realized HF magnitude on all 32 probes. This motivates testing gradient
routing to the phase head while keeping the backbone's old spectrum-first
objective. It does **not** show that the resulting whole-model Adam step or
overlap-add magnitudes will improve. All four endpoints remain controls; no seed
or recording was selected on these observations.

The auxiliary whole-model gradient norm is 2.14–24.56% of the old norm across
probes, despite its nominal 0.1 coefficient. Its scale is therefore not a fixed
percentage of the actual update. A longer run or a larger coefficient is not a
supported conclusion. The current full-model additive recipe is unqualified;
the phase-head routing alternative is an untested next diagnostic.

## Real-time investigation

The P74 trained grid used `ContextHeadStream` directly. Every paced trace records
QoS requested/applied, with time-constraint scheduling and thread-CPU sampling
disabled. Its loop sleeps until the nominal hop arrival. It does not exercise
the production callback queue, predictive worker wait, fallback/recovery or a
physical audio device. The production worker already has predictive waiting and
an optional real-time request; proposing to add those as if absent would repeat
implemented work.

The retained seed-12 phase sequence starts 3.880961 ms late after an on-time hop,
computes for 1.047334 ms, and finishes 2.263961 ms late. The next hop is also late
before recovery. Both seed-13 magnitude exceptions combine delayed starts with
2.173125 / 2.449333 ms compute. Stage wall time can include descheduling, and
there is no per-thread CPU trace here. These facts do not identify an OS root
cause, prove a worker fix, or establish audible clicks. All four failures remain.

The next runtime experiment should use the actual callback/worker path and the
same frozen model/data hashes, recording submission, queue wait, processing,
presentation misses, consecutive fallback, recovery and output continuity.
Compare scheduling policies under a frozen order/budget with no retry-to-pass.
Keep the 256-frame buffer and output clock fixed; preserve dry alignment and
reset history on gaps. Only after that software check should a device-level
integration assess [Apple audio workgroups](https://developer.apple.com/documentation/audiotoolbox/understanding-audio-workgroups)
and the existing 30-minute continuity requirement. Faster whole-file GPU inference
or a successful QoS call cannot satisfy those checks.

## Next decision gates

1. Keep the completed numerical correction and all conflicting gradients recorded.
   The audit does not justify a full-model additive loss rollout; do not infer
   useful learning solely from a new nonzero gradient.
2. Verify a candidate original-reference realized-complex auxiliary against native
   output on TRAIN material, then inspect finite matched Adam displacements with
   the old objective as control and auxiliary gradients routed only to the phase
   head as the changed hypothesis. Continue to train the backbone under the old
   joint objective: this differs from P69's frozen magnitude training. Include a
   direct-complex reweighting control if claiming a benefit from loss location,
   since whole-model direct/realized gradients are already closely aligned.
   Keep every spectral ruler and reject an apparent coherent win obtained by
   suppressing recovered magnitudes. Do not adapt coefficients to these eight draws.
   Concretely, add auxiliary derivatives only to `phase_head.*` before the common
   clipping/Adam operation; retain the old objective's derivatives for every
   parameter. Merely detaching predicted magnitudes is not equivalent when the
   head also consumes base features. The next probe must use copied persistent
   Adam moments and unchanged controls, not interpret Euclidean cosine as Adam.
3. Only after those prerequisites, register an equal-budget small-model comparison
   changing the objective alone. Keep architecture, data, seeds, exposure and all
   controls fixed, and freeze every endpoint before scoring. The FP64 audit is not
   a proposal to train or deploy in FP64. Do not launch another unchanged P74 run.
4. If the candidate still cannot jointly improve spectra and fidelity, evaluate
   auditory/perceptual auxiliaries as a separate factor with paired targets,
   calibrated corruption controls, explicit licensing and blind listening. Keep
   small causal TCN as the control; architecture replacement requires a concrete
   causal CPU feasibility test, not literature throughput extrapolation.

The real-time work is independent of the objective work. No model replacement,
new full training or deployment occurred here. A favorable diagnostic alone would
not establish release qualification.
