# P47: output-band coupling and auxiliary-budget attribution

## Question and changed control

[P46](reference-preservation-pilot.md) finishes with spectrum gains but few accepted
updates and individual preservation failures. This audit asks whether low-band
changes arise from the low-band output coefficients themselves or from high-band
changes passing through overlap-add synthesis. It also isolates how often the
auxiliary budget is the only failed recorded acceptance check.

[P19](coverage-failure-analysis.md) already measured adjacent-band interference
in an older magnitude/energy head with deployed DSP. That result cannot establish
the cause of low-band changes in the current complex residual head and raw-OLA
objective. [P28](small-mechanism-study.md) already tested detaching the auxiliary
gradient from shared features without an overall recovery win. P47 therefore
does not repeat that training or claim that removing the auxiliary alone solves
restoration. It examines current frozen outputs, the actual acceptance records,
and the relation between a learned scale and reconstructed PCM.

## Frozen diagnostic design

Local evidence: `data/runs/output-coupling-p47-20261024/`; entry point:
`training/scripts/audit_output_coupling.py`. The registration binds both P38
starts, all twelve P46 endpoints, both matched P45 Adam endpoints, their original
reviews, update logs and executed source snapshot. Every candidate is evaluated;
no endpoint is chosen after this audit's scores.

All 32 common P44 batches are reused: 160 paired crops and their clean-identity
counterparts, with 128 music and 32 speech examples per role. This is a post-hoc
TRAIN mechanism check, not independent generalization. Existing source-time,
codec, genre, region, channel-view and unique-duration limits from
[P44](pcm-fidelity-calibration.md) remain. Independent channel views do not prove
simultaneous stereo behavior.

Three output paths use the same model predictions and original clean targets:

1. The full frozen candidate, required to reproduce every saved raw-review PCM,
   MR and direct-spectrum score exactly.
2. Replace only candidate bins below 4 kHz with corresponding frozen-start bins.
3. Replace candidate bins below 8 kHz with corresponding frozen-start bins.

The latter two isolate changes relative to the frozen start; they do not replace
those bands with clean targets or prove that the start is correct. They use no
target, role label or track identity. Hard bin boundaries are attribution tools,
not a deployable crossover proposal. Their spectral, original-reference PCM and
all three existing preservation rulers are reported without threshold tuning.
The original target remains unclipped, and no reconstruction gain is attributed
to an unchanged band merely passing a ruler.

The complex coefficient delta is also partitioned into below 4 kHz, 4-8 kHz and
above 8 kHz. Each component is synthesized in float64 on identical supervised
support. The report retains linear additivity error, the difference from actual
float32 rendering, individual low-band PCM contributions and cross terms.
Projection shares use `dot(component, total) / dot(total, total)` and are pooled
with total-change energy; they may be negative or exceed one when components
cancel. They are not independent energy percentages or percentages of restoration
error. The long-FFT below-4-kHz measurement is offline, not an inference filter.

## Auxiliary controls

For every recorded P46 proposal at both radii, replay acceptance after removing
only the budget metric violation. Retain spectrum progress, complex/dynamics and
every per-example PCM check. Count added attempts once even if both recorded
radii qualify. Unrecorded radii are unknown. This does not simulate subsequent
training after a different accepted update; all observations stay on P46's actual
trajectories.

In the current residual-scope head, the generated correction is proportional to
the positive interpolated budget times bounded decoder coordinates. Doubling the
budget and halving those coordinates leaves the complex correction unchanged in
real arithmetic, while its auxiliary loss can change. Reversible output hooks
perform that control, including real warmup. The report measures float32 PCM
differences and auxiliary scores; hooks are removed and weights remain unchanged.
This establishes a representation ambiguity, not that arbitrary weight changes
can realize the same transformation or that budget supervision has no benefit.

## Verification and decision boundary

Known-answer checks cover physical-rate band partitions, exact no-ops, additive
OLA, a low-bin-only correction, leakage from a localized high-bin correction,
budget-only versus overlapping gate failures, and reversible coordinate hooks
on nonzero model output, including exception cleanup. Existing spectrum, budget
and PCM-ruler tests are run alongside these checks.

This audit adds no training loss, optimizer, production model or DSP behavior.
It does not qualify weights, audibility, runtime deadlines or a new architecture.
Results must guide a distinct, bounded next experiment rather than a longer
unchanged training run or retrospective relaxation of failed checks.

## Completed main audit

All fourteen candidate/control endpoints completed all 32 batches. Full-path raw
PCM, MR and direct-spectrum arrays exactly reproduced their frozen reviews.
All 354 current bound hashes, 243 inherited hashes and fourteen case-report
hashes were verified. No model was trained or selected.

### Low-band changes mostly come from low-band coefficients

Across the eight changed P46 endpoints, the energy-pooled projection share of
below-4-kHz coefficient changes in the resulting music low-band PCM delta is
99.932-99.983%. High-band leakage exists, but it does not dominate this particular
change. This attribution is to the update, not to total model error, and does not
contradict the older P19 adjacent-band failure on another head.

Restoring only the frozen start below 4 kHz retains **99.50-99.74%** of each
changed candidate's music MR improvement. Its per-example low4k ruler failures
fall to zero at that candidate's original 16/20-bit ruler. Identity failures
remain. Negative MR changes below mean improvement relative to the original start.

| Seed | Arm | Full music MR change | With original low4k | Full failures: identity / low4k | After: identity / low4k |
| --- | --- | ---: | ---: | ---: | ---: |
| 20261012 | policy_pcm16 | -0.06319% | -0.06294% | 12 / 11 | 1 / 0 |
| 20261012 | policy_pcm20 | -0.00661% | -0.00660% | 14 / 11 | 2 / 0 |
| 20261012 | projected_pcm16 | -0.06161% | -0.06138% | 12 / 11 | 1 / 0 |
| 20261012 | projected_pcm20 | -0.00883% | -0.00880% | 16 / 15 | 4 / 0 |
| 20261013 | policy_pcm16 | -0.06801% | -0.06767% | 16 / 13 | 3 / 0 |
| 20261013 | policy_pcm20 | -0.00591% | -0.00589% | 14 / 14 | 7 / 0 |
| 20261013 | projected_pcm16 | -0.07304% | -0.07271% | 15 / 13 | 3 / 0 |
| 20261013 | projected_pcm20 | -0.01098% | -0.01094% | 16 / 13 | 7 / 0 |

The four 24-bit no-op endpoints remain exactly unchanged in all three paths;
their zero failures do not count as restoration. At 16 bits, music identity RMS
growth falls from 3.837-8.908% to 0.601-0.705%, an absolute PCM RMS increase of
6.178e-8 to 1.140e-7 after the intervention. The baseline identity error remains;
these small increments are not a listening result. Speech MR gains remain on
the four 16-bit controls. The 20-bit projected second-seed speech regression is
retained, not hidden by the music mean.

Keeping only changes above 8 kHz does not eliminate all identity failures either.
The 16-bit variants still worsen music complex error by about 0.062-0.115% and
MR temporal error by about 0.0037-0.0178%. Restoring low-band coefficients does
not resolve high-band phase/detail fidelity.

Matched Adam controls retain 99.70% / 99.82% of their MR gain when low4k updates
are removed, but retain 16 / 7 identity failures at the 16-bit ruler, and the
first seed retains three paired-low4k failures. The second Adam seed originally
improved mean identity RMS by 9.143%; restoring its starting low band instead
increases that error by 0.458%. Low-band edits can be beneficial, so this is not
evidence for a universal ban on low-band restoration.

Float64 decomposition additivity error is at most 1.39e-17 PCM. The discrepancy
between decomposed synthesis and differences of two actual float32 renders is
at most 4.78e-7 PCM. Raw counterfactual scoring uses actual float32 synthesis;
the decomposition is not claimed exact at sub-float32 error scales.

### The auxiliary is an extra obstruction, not the only one

Removing only the budget check would admit a recorded proposal on 81 instead
of 30 attempts across the 1,536 actual attempts: 51 additional attempts. At
16 bits, policy counts become 18 / 14 instead of 6 / 5; projected counts become
17 / 21 instead of 6 / 5. At 20 bits they become 4 / 2 and 3 / 2 instead of
2 / 2 each; all 24-bit arms remain at zero. These are conditional decisions on
fixed logged proposals, not effective updates of a counterfactual trained model.

Under the reversible coordinate control, the four 16-bit endpoints' mean
auxiliary losses increase by factors 2.319 / 2.697 for policy and 2.318 / 2.695
for projection. The corresponding maximum PCM difference is 2.384e-7, and the
largest per-example RMS difference is below 8e-9 PCM (about -162 dBFS). The
same audio correction can therefore receive substantially different auxiliary
scores depending on internal coordinates. This supports treating the teacher
as a representation regularizer, not a calibrated output-error safety metric.
It does not prove that removing it improves optimization or that its entire
gradient is harmful; P28's unsuccessful detachment control remains relevant.


## Supplemental input-band control

A separately registered post-hoc extension runs the same fixed boundaries with
observed compressed-input coefficients below 4 or 8 kHz, instead of coefficients
from a second frozen model. It includes both starts, all twelve P46 endpoints
and both Adam controls, each on all 32 batches. Evidence:
`data/runs/input-band-p47-20261024/`; entry point:
`training/scripts/audit_input_band_control.py`. Full-path scores reproduce exactly;
all sixteen case-report hashes and the completion receipt were verified.

Its acceptance-ruler anchor is explicitly the untreated input, different from
the main audit's model-start anchor. Those violation counts must not be compared
as though the anchors were identical. Candidate/start spectral comparisons apply
the same transform to both. No new threshold, cutoff search or model selection is
performed, and every raw result remains available.

The input-low4k transformation needs no second inference model. For the two
16-bit policy endpoints, it changes music identity RMS from 1.079e-5 / 1.761e-5
to 3.844e-6 / 4.795e-6 PCM. Music MR remains 1.794% / 1.907% below untreated
input, versus 1.799% / 1.911% for the original full outputs. The similarly
transformed starts already achieve 1.732% / 1.840%, so the new gain remains only
0.06294% / 0.06766% relative to the matched transformed starts. Projected endpoints
show the same pattern: MR reductions of 1.792% / 1.912% against untreated input,
and identity RMS 3.828e-6 / 4.821e-6.

However, music paired low4k error increases when the original input replaces a
useful learned correction. For policy, low4k RMS changes from 0.008718287 /
0.008716739 to 0.008746300 / 0.008746264. Full paired PCM RMS also increases from
0.01197747 / 0.01195141 to 0.01199914 / 0.01197545. The tradeoff is small relative
to magnitude-spectrum recovery but real. Input bypass therefore cannot be called
an unconditional fidelity fix. The old P15 result that hard band closures can
hurt paired recovery remains relevant.

Furthermore, even the frozen starts have music direct-complex error about
1.62-1.74% above untreated input; the four changed 16-bit models are about
1.68-1.86% above it. Frequency restrictions retain this problem. A baseline's
inherited magnitude gain does not certify phase recovery, nor does passing a
relative preservation check certify the baseline's absolute fidelity.

## Resulting next experiment

The evidence favors isolating **where a learned update can act**, before adding
capacity or repeating whole-model guarded updates. A bounded next control should
retain the frozen causal feature path and existing low-band reconstruction, then
train a small independent high-band residual adapter against original clean
spectra. A matched unrestricted adapter is needed to distinguish output support
from simply freezing most parameters. This is a proposed experiment, not an
implemented or started training run.

The implementation should share the frozen feature computation rather than run
two complete models. Only the added adapter should receive updates, so a nominal
high-band head cannot alter low bins indirectly through a changing shared trunk
or generation scale. Test a fixed smooth support transition, raw OLA spill,
reference-following temporal errors, silence and identity, then actual Rust
streaming. Do not infer high-band recovery from unchanged low bins, and retain
low-band reconstruction improvements demonstrated by the input-control contrast.

Keep paired spectrum supervision primary. Treat the old latent-budget teacher
as a separate regularizer/control, not an output-quality acceptance certificate;
measure actual generated corrections, original-reference phase/PCM and individual
preservation failures. The remaining complex and identity failures require those
controls even when spectrum gains improve. Freezing features also limits what an
adapter can learn, so a negative short diagnostic would not establish a model
family's capacity ceiling.

Any subsequent pilot needs common starts, matched effective exposure and budgets,
all endpoints finished before scoring, and explicit disclosure that these clips
have already been used for diagnosis. No default weights or native DSP changed.
The <5 ms compute requirement and prior streaming deadline failures remain open.

Validation for this work: **706 tests passed**, 223 existing warnings; the focused
synthesis/budget/reference suite passed 64 tests. No new loss was added. Numerical
controls establish the attribution machinery, not listening or release quality.
