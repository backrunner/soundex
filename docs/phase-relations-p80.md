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

## Retained preparation failure and status

The first numerical suite passed 21 checks and failed two weak-fixture assertions:
at PCM scale `1e-8`, both auxiliaries correctly gave zero risk because no reference
bin passed the existing absolute phase floor. The test incorrectly required a
nonzero phase gradient. Evidence and exact initial source remain in
`data/runs/phase-relations-p80-controls-20261116/`. No optimizer update occurred.
The correction preserves the floor, explicitly checks unsupported weak samples,
adds a supported `1e-5` sample and verifies that invented magnitude is penalized
even when phase support is empty. This is a fixture correction, not a relaxed
quality gate. The corrected tests pass; registered results are pending.
