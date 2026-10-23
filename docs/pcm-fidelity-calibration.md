# P44: absolute PCM fidelity calibration

## Registered question

[P43](spectral-priority-update-audit.md) improved spectrum-led local directions,
but its relative identity checks were ill-conditioned near zero. Before sustained
training, P44 separates frozen-model identity damage from feature/OLA numerical
roundtrip error and gives preservation metrics an absolute PCM scale.

This is a new measurement question, not another attempt to pass P43 by reducing
step sizes. No model update, tolerance selection or release decision is made by
this calibration. The two P38 small control endpoints remain frozen.

## Controls and isolation

- Select 32 common TRAIN batches by schedule geometry, before measuring either
  endpoint. Exclude the entire source support of P39/P41, P42/P43 and P38
  development/shifted crops across rates, codecs and channel views. Newly selected
  intervals are also mutually disjoint. Recordings can recur and the draws were
  seen in P38 training; this is not independent testing.
- Compare the frozen model, compressed passthrough and clean-feature OLA against
  the original clean PCM on identical supervised support. Promote stored clean
  features to float64 for a separate synthesis control; this does not remove the
  earlier float32 feature-analysis error.
- Apply rounding-only 16-, 20- and 24-bit precision lattices to the original target
  as measurement rulers. Their pointwise half-step bounds are exactly `2^-bits`.
  Do not clip, dither, replace the target or infer that these errors are inaudible.
  All three rulers are reported; none is selected from observed model failures.
- Retain every per-example full-band, below-8-kHz and above-8-kHz PCM RMS, peak,
  error first-difference and hop-boundary error-difference statistic. Report the
  unchanged MR spectral and dynamics objectives alongside these absolute values.
  Music and speech, paired and identity examples must remain distinguishable.

The fixed-anchor diagnostic helper separately bounds total error growth and the
actual output-displacement waveform. A small difference between two error RMS
values does not bound their waveform difference. Future cumulative training must
keep the initial anchor fixed; allowing the same increment against each previous
step would accumulate an unbounded allowance. This helper alone is not a complete
optimizer policy or proof of stereo, continuity, perception or restoration.

Entry point: `training/scripts/calibrate_pcm_fidelity.py`. Local evidence is under
`data/runs/pcm-fidelity-p44-20261023/`; registration binds selected draws, source
snapshots, original inputs and endpoint hashes before measurement. Raw source
audio, checkpoints and per-example reports remain ignored.

## Completed results

All 32 batches completed for both endpoints on 2026-10-23. They contain 160 paired
crops plus corresponding identities: 128 music and 32 speech clips, 132 distinct
recordings, **74.076 seconds** of unique MR supervision and 74.968 seconds of
direct-analysis support. Both models see the same material; these counts must not
be doubled. Left/right/mid/side/mono views and all nine codec recipes are present.
China/Japan/Korea/Europe/North America are represented, but 19 music clips have
unknown region, 11 have unlabeled genre, and funk is absent. Independent views do
not establish simultaneous stereo preservation.

| Identity metric | Seed 20261012 | Seed 20261013 |
| --- | ---: | ---: |
| Frozen-model mean PCM error RMS | 8.3471e-6 | 1.3008e-5 |
| Frozen-model median PCM error RMS | 3.2376e-6 | 2.4292e-6 |
| Frozen-model maximum per-example PCM error RMS | 7.9386e-5 | 2.5593e-4 |
| Clean-feature float32 mean PCM error RMS | 1.9488e-8 | 1.9488e-8 |
| Float64-synthesis mean PCM error RMS | 1.6644e-8 | 1.6644e-8 |
| Frozen-model mean identity MR | 2.7852e-4 | 5.2515e-4 |
| Clean-feature float32 mean identity MR | 1.0582e-8 | 1.0582e-8 |

Model mean identity error is about **428 / 667 times** the float32 roundtrip mean.
Upgrading synthesis precision alone does not explain or remove it. The identity
input is already clean, and model-minus-passthrough measurements confirm that
these changes largely come from the model, including the low band.

The common 16/20/24-bit precision rulers have mean PCM error RMS
6.9838e-6 / 4.3726e-7 / 2.4902e-8 and maximum per-example RMS
8.9086e-6 / 5.5646e-7 / 3.6598e-8. Their mathematical half-step bounds are
1.5259e-5 / 9.5367e-7 / 5.9605e-8; every pointwise rounding error satisfies its
bound. Some original samples already lie on a lattice and rounding adds no error.
The rulers' MR means are 2.3480e-3 / 5.0711e-5 / 2.5535e-7. Different spectral
placement can therefore produce very different relative MR at a small PCM level;
neither the rulers nor a small RMS certify inaudibility.

Counterexamples remain in the per-example reports:

- Seed 20261012, draw 8223, example 0, a music mid view: identity RMS 7.9386e-5
  (−82.005 dBFS), below-8-kHz RMS 7.5486e-5 and peak error 3.9756e-4. Its
  clean-feature roundtrip RMS is only 1.9274e-8.
- Seed 20261013, draw 6198, example 1, a classical side view: identity RMS 2.5593e-4
  (−71.838 dBFS), below-8-kHz RMS 2.4662e-4 and peak error 1.7026e-3. Its
  roundtrip RMS is 1.6073e-8. This independent side view is not a stereo playback
  or perceptual result.

Music mean identity RMS is 1.0277e-5 / 1.6169e-5, versus speech 6.2921e-7 /
3.6515e-7. These group means do not license ignoring individual failures.

## Decision

Do not waive identity regressions as floating-point noise, or choose an absolute
tolerance to pass a known example. [P45](anchored-spectrum-pilot.md) tests all three
fixed precision scales as separate experimental arms, with cumulative constraints
against the frozen starting model, per-example peaks and low-band protection.
It reports the original spectrum/phase/dynamics objectives as well. It does not
claim these are validated perceptual thresholds or inherit baseline damage as an
acceptable release condition. Default weights and runtime remain unchanged.

All **74 bound inputs/sources** were verified after calibration. Known-answer
tests cover quantization bounds, Parseval scaling, band partition, boundary-error
differences, source-time isolation and cumulative/phase-cancellation counterexamples.
The completed reports, analysis and source snapshots are retained with an integrity
receipt. No raw audio or experimental checkpoint is included in Git.
