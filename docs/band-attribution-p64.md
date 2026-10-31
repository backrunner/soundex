# P64: codec, complex-error and deadline attribution

Date: 2026-11-01. Status: completed diagnostic; no new trained model or release
qualification. Default weights and DSP remain unchanged. This follows the
completed [P63 comparison](band-training-p63.md), whose spectral criterion for
further study passed while fidelity and real-time limitations remained.

## Changed question and fixed evidence

[P58](joint-attribution-p58.md) attributed another architecture's MR components;
[P30](phase-swap-audit.md) tested reference-assisted phase swaps. Neither identifies
which codecs drive the new band architecture's tradeoff, or whether increased
phase-interaction error comes from changing magnitude or changing phase angles.
P64 adds exact accounting on existing outputs; it does not repeat training, swap
in reference phase, search thresholds or rerun timing to obtain a pass.

The main audit uses all 160 exposed P55 case groups, six P63 endpoints plus legacy
and identity-DSP controls, and raw/native/untreated audio: **2,720 paths**. It
verifies 4,177 input/source/receipt hashes before and after processing, including
all consumed PCM. Original per-case metadata and scoring support are preserved.
Scores average cases within each source, then sources equally. These are reused
music recordings, not independent testing, new speech coverage or listening.

The follow-up factorization uses all 160 case groups, both seeds and raw/native
band-versus-matched comparisons: **640 matched pairs**. It was specified after the
interaction increase was observed and is explicitly a follow-up diagnosis, not a
preregistered candidate-selection experiment. Both audits preserve every source.

## AAC and MP3 have different tradeoffs

Relative native paired changes versus each seed's matched global control:

| Codec | MR, seed 12 / 13 | HF PCM error RMS, seed 12 / 13 | Dynamics, seed 12 / 13 |
| --- | --- | --- | --- |
| AAC-LC 96 kbps | +0.142% / -0.085% | -0.307% / -0.154% | -0.396% / -0.259% |
| MP3 64 kbps | -5.569% / -5.505% | +3.341% / +3.870% | +1.257% / +1.408% |
| Vorbis q3 | -1.415% / -1.179% | +0.111% / +0.117% | -0.151% / -0.208% |

AAC seed 12's native MR delta is +0.00064817; the 16+ kHz contribution is
+0.00087573, partly offset by other bands. The same regression exists before DSP
(raw MR +0.263%). AAC MR worsens on 10/20 and 8/20 sources across seeds; there
is no uniform codec benefit. Its HF PCM error nevertheless improves.

MP3 carries most of the aggregate spectral benefit and the HF PCM/dynamics cost.
Its 12-16 kHz contribution to native MR is -0.02369100 / -0.02310770, within total
deltas -0.03870944 / -0.03808171. MR worsens on one source in each seed. Vorbis
MR worsens on three / two sources. All source and case deltas remain in
`score-attribution.json`; no exceptions were removed or used to set gates.

## Magnitude and phase interact

Use a normalized rectangular Fourier transform of each aligned ten-second
excerpt, with one-sided Parseval factors and physical-channel mean power:

```text
|Y - T|^2 = (|Y| - |T|)^2 + |Y||T| |unit(Y) - unit(T)|^2
```

The second term is an amplitude-weighted phase interaction, not a pure phase
quality metric. Zero reference bins contribute zero interaction; no weak-bin
threshold is introduced. The diagnostic includes Nyquist and reports calibrated
PCM power/RMS, not unnormalized FFT dBFS. It does not replace the registered
multiresolution or reference-following transient objectives.

MP3 native HF power deltas versus matched global, in calibrated PCM squared:

| Component | Seed 20261012 | Seed 20261013 |
| --- | ---: | ---: |
| Magnitude-error power | -2.16120943e-5 | -2.20568988e-5 |
| Phase-interaction power | +3.69287458e-5 | +4.05713583e-5 |
| Total complex-error power | +1.53166515e-5 | +1.85144595e-5 |

Magnitude gets closer while interaction increases by more, so complex error
increases. The 8-16 kHz bands carry most of this PCM regression. Raw MP3 HF
complex-error increases are +3.02525707e-5 / +3.44209234e-5; native DSP reduces
those increases to the table's values. This rules out blaming the entire
regression on DSP, without establishing which nonlinear DSP component causes
any individual difference.

A symmetric two-factor decomposition further separates the interaction change.
Writing C/B for candidate/control magnitudes and D for squared unit-phase distance:

```text
Delta interaction = |T| (C-B) (D_C+D_B)/2
                  + |T| (C+B) (D_C-D_B)/2
```

For native MP3, the magnitude factor contributes **96.19% / 96.52%** of the
interaction increase, and the angular-distance factor contributes the remaining
3.81% / 3.48%. Before DSP the magnitude factor contributes 78.15% / 78.76%.
Thus the evidence primarily shows stronger magnitudes in bins whose phase is
still imperfect; it does not show that a new large angular error alone explains
the regression. These are signed algebraic allocations, not causal intervention
percentages or audible-distortion shares. Unit(0)=0 is only a numerical convention;
phase quality at zero magnitude is undefined.

AAC has the opposite power tradeoff: native magnitude-error deltas are
+6.73546688e-7 / +4.11281159e-7, interaction deltas -1.52752097e-6 /
-8.36202681e-7, and total error improves. Different rectangular-power and MR
weightings can disagree; do not replace spectral selection with this diagnostic
ruler. Weak-band reference/error PCM RMS and every exception remain in the reports.

## Existing deadline failures include outside-stage delay

No new runtime benchmark ran. The audit uses the original six paced traces and
matches their 1,000 scored hops each. The first **100 benchmark hops** are excluded,
separately from the model's 80-hop history warmup.

| Endpoint | Late hops | Late with stage service below 2.667 ms | Of those, previous hop on time | Calls above 5 ms |
| --- | ---: | ---: | ---: | ---: |
| Seed 12 band | 26 | 20 | 3 | 3 |
| Seed 13 band | 17 | 17 | 4 | 0 |
| Seed 12 global | 21 | 19 | 4 | 1 |
| Seed 13 global | 28 | 24 | 4 | 2 |
| Seed 12 matched | 24 | 23 | 7 | 1 |
| Seed 13 matched | 21 | 19 | 5 | 1 |

For a late hop, arrival-to-finish elapsed minus measured stages equals
`finish_lateness + hop_period - stage_service`. This includes queued work,
sleep/wakeup delay and unmeasured loop overhead. It is not a direct wake-latency
or CPU measurement. Seed 13 band hop 335 has 1.367083 ms stage service, an on-time
preceding hop, and 12.429472 ms outside the measured stages. The stored test uses
`thread::sleep` pacing in a diagnostic loop; it is not an actual callback/worker
certification. Exact OS cause cannot be recovered from these probe-off traces.

The 12.734208 ms stage outlier and all late hops remain failures. The earlier
24.131125 ms CPU/wake audit is also retained. Smaller compute alone cannot resolve
every recorded deadline miss; actual worker pacing and fixed-timeline fallback
need a separate, bounded, registered validation with wake/service/queue telemetry.
No diagnostic arithmetic subtracts scheduling overhead from acceptance.

## Decision and next controlled work

Keep the small band model as the experimental spectral candidate. Do not enlarge
it, uniformly raise the HF loss, suppress restored magnitudes, or remove DSP based
on this audit. The existing spectral gain remains the primary objective.

The next quality experiment should isolate a **magnitude-preserving phase
correction** on the frozen band endpoint, starting with optimization-role data and
a zero-correction control. The changed hypothesis is whether a dedicated phase
correction can reduce interaction cost while retaining the already learned band
magnitude gains. This differs from [P31](phase-coordinates-study.md)'s simultaneous
coordinate/feature intervention and P30's nondeployable reference swaps. No such
new training has started in P64. Before launching it, require zero-initialization
parity, causality/physical-channel state tests, correct-reference and weak-energy
gradients, common warmup/support, and real native phase-policy correspondence.
A training-only spectral rotation is insufficient if Rust fusion removes it.
Freeze the controls/budget and all endpoint selection rules before rechecks;
keep the magnitude target and identity/stereo safeguards unchanged.

The runtime work is separate: instrument wake and queue timing independently of
the optional CPU clock probe, then test the actual worker with the same fixed
stimulus and all outcomes retained. Do not interpret a sleep-pacing improvement
as real-device certification. No fresh dataset or threshold is selected here.

## Evidence, invalid attempts and validation

- Main completed audit: `data/runs/band-attribution-p64-20261101-v3/`.
- Follow-up: `data/runs/band-phase-factors-p64-20261101/`.
- Code: `training/scripts/audit_band_attribution.py`,
  `training/scripts/audit_band_phase_factors.py`, and
  `training/evaluation/band_attribution.py`.
- Initial `band-attribution-p64-20261101/` rejected unnormalized `training/../data`
  receipt keys before registration/scoring. The corrected launcher resolves only
  the experiment root, preserving its logical symlink child paths. Failure and
  source snapshot retained; regression covers the symlink boundary.
- `band-attribution-p64-20261101-v2/` stopped on a numerical-identity assertion at
  `01-0-mp3-cbr-64`, native legacy, 16+ kHz. Sequential axis-0 reduction produced
  residual 2.41192632e-18 against bound 2.34053237e-18. Full-array pairwise summation
  fixes reduction roundoff; the original 128-epsilon operation-scale tolerance
  is unchanged. This is an invalid numerical diagnostic, not a model rejection.
- Both completed receipts verify. `source-snapshot.json` maps registered Python
  hashes to preserved execution sources, including the pre-format main audit.
  The existing P63 endpoint and evaluation files were not rewritten.
- Targeted tests cover gain versus quadrature, silence and weak energy, exact
  Parseval power, odd/even FFT endpoints, stereo isolation, long interleaved sums,
  relative-root/symlink receipts, fixed benchmark exclusion and late fast calls.
  Authored documentation and failure records are maintained in English.

Validation: **31 targeted tests passed** (five existing SciPy deprecation
warnings); Ruff lint/format and all **144 Markdown documentation checks** pass.
No Rust runtime change or new device/performance validation was made.
