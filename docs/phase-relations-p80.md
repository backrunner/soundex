# P80: explicit phase-relation supervision

## Changed hypothesis and research boundary

[P79](parameter-outputs-p79.md) locates tiny coherent gains in the phase readout,
without consistent spectral improvement. P80 asks whether original-reference
phase relations supply different derivative information from complex distance.
It does not repeat a larger readout step, reset Adam, change architecture or
extend an unchanged training run. P39–P46's direction/projection failures and
P75–P79's spectral/runtime failures remain recorded.

The [AP-BWE paper, Section III-B](https://arxiv.org/html/2401.06387v2) explicitly
supervises instantaneous phase, frequency differences (group delay) and time
differences (instantaneous angular frequency) using wrapped angular distance,
alongside amplitude and complex losses. It is a speech low-pass bandwidth-extension
study; its whole-file efficiency cannot establish our causal music-codec per-hop
budget. This independently implemented diagnostic imports no upstream code or
weights and makes no new licensing claim about external datasets.

SoundEx already has legacy `models/losses.py::phase_gradient_terms`, using circular
frequency/time differences, plus its instantaneous phase term. The active P74
`fused_spectrum.phase_objective` instead combines direct amplitude/complex/envelope/
dynamics and raw/fused magnitude MR. P68/P74 add phase observations, not these
explicit relation targets. P80 tests this wiring difference on the current small
models; it is not the first phase loss in the project.

## Fixed objective and controls

`models/phase_relations.py` defines `soundex-phase-relations-p80-v1`. Compare the
complete old objective and the existing realized-complex routed control against
two relation auxiliaries. All old derivatives remain; additional derivatives go
only to `phase_head.*` before common canonical clipping and persistent Adam.

- Target original paired FFT256 phase (input phase for the clean identity role),
  never a DSP target, clipped reference or the model's own reanalyzed spectrum.
- Limit the auxiliary to >=8 kHz interior bins where input-driven native support
  is available. Use reference relative amplitude and the legacy -60 dB relative /
  -120 dB absolute feature-magnitude floors. These are FFT feature coordinates,
  not PCM dBFS. Reference weights are detached and independent of predictions.
- For instantaneous error `d`, compare circular `(1-cos(d))/2` and wrapped
  `abs(d-2*pi*round(d/(2*pi)))/pi`, each bounded by one. Apply the same kernels to
  adjacent-frequency and current-minus-previous-frame errors against the reference.
  Edge weights use the minimum endpoint weight; both frames must be supervised.
- Average each component within each example using its own weight mass, then
  give the three components equal weight. An empty support contributes zero phase
  risk; existing direct/raw/fused magnitude terms still penalize invented energy.
  The auxiliary is fixed at `0.1*log1p(relation_mean)` with balanced paired/identity
  risk and the existing 80:20 music/speech draws. No gradient-scale matching or
  transfer-driven coefficient tuning is claimed.

Circular distance has a vanishing derivative near opposite phases. Wrapped L1
retains a finite one-sided slope there, but is nondifferentiable at exact antipodes
and zero. PyTorch's zero subgradient at zero and round-to-even branch at exact
antipodes are explicit implementation conventions, not unique physical directions.
Finite-difference checks exclude cusps; separate controls verify both one-sided
slopes, periodicity and reference stationarity. This mathematical difference is
not evidence that the real model's errors concentrate near antipodes.

Numerical controls cover correct reference, silence, weak supported/unsupported
energy, generation despite unsupported phase, legacy-kernel equality with the
declared half scale, warmup/preserved-bin/channel exclusion, prefix-only temporal
support, actual head routing and parameter finite differences. No inference layer,
lookahead, buffering, DSP policy or deployed default changes.

## Preregistered finite experiment

Entry point: `training/scripts/audit_phase_relations.py`. Planned evidence root:
`data/runs/phase-relations-p80-20261116/`. Inherit frozen P77 inputs, references,
source/work partition and all four P74 16,384-update parents (two seeds and both
temporal observations, 216,802 parameters each). Every input and source is hashed.

1. Complete 32 read-only FP64 probes: four parents times eight fit draws. Keep
   canonical head-gradient norms/cosines for old, fused MR, fused HF magnitude,
   realized complex, circular and wrapped targets, plus proposed summed directions.
   Retain per-example support and weighted fraction within 1% of an antipode as a
   descriptive statistic, not a quality threshold. Model tensors remain unchanged.
2. Run all four arms at all origins for exactly two updates, four draws per update:
   16 new relation updates and 16 control replay updates. Keep P77's gain64 physical
   readout learning rate and coordinate transformation fixed, not another gain
   increase. Both controls must reproduce canonical/coordinate model and Adam
   tensors bitwise. Freeze all 16 endpoints before scoring any changed output.
3. Export and freeze all eight relation graphs, then score all 12 raw/proxy draws
   and five native draws: 400 new native mono roles. Reuse old/realized/parent,
   untreated and identity-DSP controls only from verified P77 evidence. Native
   proxy max `2^-14` / RMS `2e-5` correspondence bounds remain unchanged.
4. Report unchanged FP64 spectral/fidelity metrics, aggregation by recording before
   bootstrap, music/speech and paired/identity separately. Reuse P77's advancement
   rule: every origin must reduce native paired-transfer MR and independent HF SC
   beyond `8*float32_epsilon*abs(old_mean)`, and reduce realized complex, with no
   missing/undefined scores. Shape, energy, dynamics, low bands, identity and all
   counterexamples remain visible. Passing is necessary, not quality qualification.

The fit bank has 40 recordings (32 music/eight speech), 18.724499 unique supervised
seconds after warmup/overlap; transfer has 20 previously exposed recordings and
9.283955 seconds. They cover nine codec settings and independent channel views,
not physical stereo. Reused transfer is a recheck, not unseen generalization.
Two updates cannot establish convergence, an architecture ceiling, adequate
training, audible benefit or release qualification. No timing benchmark is rerun.

## Descriptive common-fit follow-up

After primary completion, `training/scripts/review_phase_relation_fit.py` compares
all 16 frozen endpoints and all four parents on the same eight fit draws. It
reports both relation definitions and the unchanged common risk in FP64, using
actual FP32 predictions lifted exactly and original references. This distinguishes
a lower fit objective from improved final spectra; totals from different objective
definitions are never compared against each other. The review freezes source and
input hashes before its forward passes, performs no optimizer update or extra
native/held-out render, and cannot modify the registered transfer gates. Evidence:
`data/runs/phase-relations-p80-fit-review-20261116/`.

## Retained preparation failure

The first numerical suite passed 21 checks and failed two weak-fixture assertions:
at PCM scale `1e-8`, both auxiliaries correctly gave zero risk because no reference
bin passed the existing absolute phase floor. The test incorrectly required a
nonzero phase gradient. Evidence and exact initial source remain in
`data/runs/phase-relations-p80-controls-20261116/`. No optimizer update occurred.
The correction preserves the floor, explicitly checks unsupported weak samples,
adds a supported `1e-5` sample and verifies that invented magnitude is penalized
even when phase support is empty. This is a fixture correction, not a relaxed
quality gate. All 31 targeted relation/routing/coordinate checks now pass.

## Completed gradient and optimizer checks

All 32 frozen FP64 probes and all 32 diagnostic optimizer steps complete. Both
old and realized-complex controls reproduce P77 canonical models, coordinate
models and coordinate Adam states bitwise at all four origins. All 16 endpoints
and eight new graph hashes are frozen before scoring. No global clipping is
active (largest canonical gradient norm 2.56215, below the unchanged limit 5).

The median auxiliary head-gradient norm divided by the old head-gradient norm,
within each origin's eight draws, is:

| Origin | Realized complex | Circular relation | Wrapped relation |
| --- | ---: | ---: | ---: |
| 20261012 magnitude | 0.146 | 0.364 | 1.132 |
| 20261012 phase | 0.142 | 0.404 | 0.863 |
| 20261013 magnitude | 0.084 | 0.391 | 1.049 |
| 20261013 phase | 0.098 | 0.446 | 0.962 |

Equal nominal coefficients are not equal gradient scales. The wrapped derivative
can also be larger near zero; these differences cannot be attributed exclusively
to antipodal errors. In each origin, 61/80 fit role views have instantaneous support
and 59/80 have frequency/time edge support. Weighted mass within 1% of an antipode
is only 0.52–0.93% when pooled by support across views/components/origins separately.
This does not support a widespread 180-degree saturation explanation. These views
reuse 40 recordings; pooling is descriptive, not an uncertainty estimate.

Every proposed head-gradient sum has a nonnegative dot product with both fused MR
and fused HF magnitude gradients in all 32 probes. Conflict with realized complex
remains in 13/32 circular and 8/32 wrapped sums, compared with 8/32 for the existing
realized-complex auxiliary. Euclidean head directions do not describe all base
motion or the eventual Adam step. The actual auxiliary directional derivative
is negative in every one of the 16 relation updates, with magnitudes
`1.667e-7`–`4.396e-7`. This confirms a local descent contribution, not final quality.

## Completed common-fit review

All 20 states (16 endpoints plus four original parents) and eight fit draws finish,
producing 1,600 per-role records. The unchanged common risk is independently
reconstructed and checked against the original function on every forward. Compared
with matched old endpoints, both relation recipes lower their own phase risk and
the common risk at every origin under the fixed balanced paired/identity mean.

| Fixed-fit change versus matched old | Circular | Wrapped |
| --- | ---: | ---: |
| Own relation mean, relative percent | -0.000952 to -0.001040% | -0.002048 to -0.002732% |
| Unchanged common risk, absolute | -6.679e-8 to -1.012e-7 | -8.179e-8 to -1.089e-7 |

Each column uses its own fixed definition; the percentages are not a cross-loss
ranking. Frequency-relation decreases are larger than time-relation decreases.
The circular recipe also raises clean-identity common risk in three of four
origins; the balanced decrease does not erase those exceptions. These tiny fit
effects establish response to the objective, not convergence or useful restoration.
There is no reason here to call the loss disconnected or reset all optimizer
moments. Full per-component/role/corpus/recording results remain in the fit-review
`assessment.json` and `records.json`; its verified receipt binds 456 files.

## Completed native spectral review

All 400 new native roles complete; all 40 native/proxy checks pass. Maximum PCM
discrepancies are `2.980232e-7` peak and `2.404784e-8` RMS. The main receipt binds
2,122 files and 5,900 score records including retained controls. All source/input
and endpoint/graph hashes verify. These facts establish a valid measurement and
matched implementation, not a successful quality recipe.

All eight registered joint-transfer gates fail. The table shows candidate minus
matched old, paired native means over 20 exposed transfer recordings. Negative
means less error. These are not legacy-relative changes or unseen test scores.

| Origin / relation | MR delta | HF SC delta | Realized complex delta |
| --- | ---: | ---: | ---: |
| 20261012 magnitude / circular | -4.359e-8 | +2.047e-8 | +1.177e-7 |
| 20261012 magnitude / wrapped | -3.634e-8 | -1.436e-7 | +3.260e-9 |
| 20261012 phase / circular | -5.486e-8 | +5.375e-8 | +9.460e-8 |
| 20261012 phase / wrapped | -5.675e-8 | -1.383e-7 | +3.589e-8 |
| 20261013 magnitude / circular | -1.539e-8 | +2.913e-8 | +7.204e-8 |
| 20261013 magnitude / wrapped | +5.929e-9 | -2.199e-7 | +3.655e-8 |
| 20261013 phase / circular | -1.806e-8 | +7.672e-8 | +7.753e-8 |
| 20261013 phase / wrapped | +7.898e-9 | -4.661e-8 | +3.696e-8 |

Every MR/HF SC interval spans zero and every mean change is below its numerical
ruler (MR `2.791e-7`–`2.807e-7`, HF SC `4.881e-7`–`4.900e-7`). Circular complex
regression intervals exclude zero at all four origins; wrapped complex intervals
span zero. All complex changes are also small relative to the corresponding
`2.945e-7`–`2.982e-7` ruler, which is displayed but is not its sign gate or an
audibility threshold. No failure establishes audible damage or an architecture
ceiling, and no tiny favorable mean establishes restoration gain.

Circular native HF shape improves numerically at all four origins; wrapped shape
improves at three and worsens at seed-13 phase. Energy envelopes and dynamics
mostly decrease, while hop-boundary difference RMS rises at all eight endpoints
(`1.561e-9`–`8.068e-9`). Low-band RMS increases `5.846e-12`–`4.149e-11` and HF PCM
RMS changes range from `-4.983e-10` to `+4.582e-10`. Keep these calibrated absolute
quantities beside relative errors; their small size does not prove audible quality.
HF prediction power slightly increases at every endpoint; reduced enhancement
does not explain all of the small spectral changes, nor does more power prove
better spectral reconstruction.

On the same transfer population, circular realized-complex error already rises
before DSP at all four origins. Wrapped raw complex means instead fall slightly,
but proxy/native means rise. Stage behavior is therefore recipe dependent. The
increments are tiny and near implementation discrepancy scales: this does not
justify removing DSP or claiming it universally blocks useful reconstruction.

Keep domain and clean-identity conflicts visible. Wrapped HF SC decreases in all
four music strata but increases in all four speech strata. Circular music MR also
decreases at all four origins; its speech MR worsens in both seed-12 origins and
improves in both seed-13 origins. Identity spectral results remain mixed. All four
identity-speech native differences are exactly zero for both recipes on these
four recordings; this does not prove general clean-speech preservation.

Concrete retained paired-transfer counterexamples include:

- `83d9806afed95724ed33230a2aefd7c361406aeeb1f642ac87dc1f23dcb50669`:
  seed-12 magnitude circular complex delta `+7.247869e-7`; wrapped `+5.395849e-7`.
- `88ea9784b1a6e849599d20fd09276e6836297aff6185709127708a2ee4df2b35`:
  seed-13 magnitude circular complex delta `+7.564885e-7`; wrapped `+5.108705e-7`.
- `363a3164861dca776d124a7786ba452e65085466014079484bff680047e36b8f`:
  seed-13 magnitude wrapped MR delta `+2.166159e-7`.

Every counterexample, undefined value, interval and untreated/identity-DSP
comparison remains in the main `assessment.json` and `records.json`. The historical
legacy model is not rescored here; P74's separately reported legacy-relative
progress is unchanged. Do not describe this incremental failure as all newer
models losing to legacy.

## Decision and next evidence

The auxiliary is mathematically connected, takes descent steps and lowers fixed-fit
risk, yet does not give a reliable combined spectral/coherent gain in this bounded
native recheck. The tested direct phase-relation recipe is not advanced to full
training or default weights. Two steps do not reject the family or establish
overfitting; the result specifically fails the registered prerequisite. Do not
increase coefficients, reset moments, select the best origin, or extend the same
recipe merely because its own fit loss fell.

The next distinct question is whether phase relations measured **after actual
overlap-add/fusion** align with the desired reconstructed spectra, versus these
direct FFT256 feature relations. Keep the same kernel/support policy and direct
lossless magnitude targets as explicit controls; do not conflate a new measurement
location with another coefficient change. P75/P76 compared marginal complex
distance, not this relational derivative. Before another finite proposal, require
reference/weak-bin/wrap and native-correspondence checks, matched raw/fused relation
derivatives and retained component/domain conflicts. This is a proposed diagnostic,
not a new claim that synthesis supervision is absent or that more training will
solve the problem.

The inference graph structure, parameter count, causal state, fixed 256-frame
buffer and DSP remain unchanged. No new compute, CPU/RSS or device measurement is
made; P79 timing numbers cannot qualify these endpoints. Physical stereo music,
long device runs and blind listening remain outstanding. Source/docs are committed,
all failed/source/checkpoint evidence is retained, and no cleanup removes data.
Ruff, all 31 targeted tests and the English/link/index documentation check pass.
