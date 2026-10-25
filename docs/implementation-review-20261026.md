# External implementations and a reference-anchored texture prototype

Date: 2026-10-26. Scope: primary papers and official implementation inspection,
followed by an original training-only prototype. This is not a SOTA ranking,
reproduction of a published system, or a qualified restoration model.

The immediate evidence is [P58](joint-attribution-p58.md): joint training gains
and added fidelity/dynamics errors concentrate above 16 kHz. The errors already
exist in raw output. Existing P6 factorization, P31 phase-input/coordinate changes,
P39-P53 gradient/preservation interventions and P54 auraloss comparisons do not
justify repeating those mechanisms unchanged. See [failed experiments](failed-experiments.md).

## What the official implementations actually do

| Implementation | Relevant mechanism and task | SoundEx implication |
| --- | --- | --- |
| [AERO paper](https://arxiv.org/abs/2211.12232), [official model](https://github.com/slp-rl/aero/blob/6054bd918a099447f8e149f7e8bd55676e641b68/src/models/aero.py) | Speech/music sample-rate bandwidth extension; complex full-band encoder-decoder, spectral reconstruction and adversarial/feature losses. | Complex and learned texture supervision are useful references. Ideal resampling is not MP3/AAC/Vorbis restoration. The [STFT](https://github.com/slp-rl/aero/blob/6054bd918a099447f8e149f7e8bd55676e641b68/src/models/spec.py) is centered and [modules](https://github.com/slp-rl/aero/blob/6054bd918a099447f8e149f7e8bd55676e641b68/src/models/modules.py) include bidirectional LSTM and symmetric temporal convolution; the unmodified implementation is not our causal hop worker. |
| [Apollo official model](https://github.com/JusperLee/Apollo/blob/e84bcacc59d5455f05d86a5c97dd4aeb3c14dbb6/look2hear/models/apollo.py), [loss implementation](https://github.com/JusperLee/Apollo/blob/e84bcacc59d5455f05d86a5c97dd4aeb3c14dbb6/look2hear/losses/gan_losses.py) | Directly relevant compressed-music restoration; band communication plus temporal processing, multi-resolution spectral reconstruction, discriminator and feature matching. | Training-side learned features are a distinct untested route in the recent P-series. Band attention being noncausal along frequency is not a temporal lookahead problem by itself; the supplied temporal ICB defaults to symmetric convolution, and centered STFT also needs adaptation. Published throughput does not certify our CPU hop service. |
| [AP-BWE official model](https://github.com/yxlu-0102/AP-BWE/blob/751710f22404c27e5bcc983248f8b856a04b8422/models/model.py), [training](https://github.com/yxlu-0102/AP-BWE/blob/751710f22404c27e5bcc983248f8b856a04b8422/train/train_48k.py) | Speech bandwidth extension; parallel amplitude/phase prediction, wrapped phase and its time/frequency differences, complex and consistency losses, multiple discriminators and feature matching. | Keep phase/time fidelity separate from magnitude gains. Speech results do not establish music quality. We do not copy phase-loss weights or treat them as validated on weak musical bands. The released network is not a measured causal SoundEx worker. |
| [DeepFilterNet3 official model](https://github.com/Rikorose/DeepFilterNet/blob/d375b2d8309e0935d165700c91da9de862a99c31/DeepFilterNet/df/deepfilternet3.py), [losses](https://github.com/Rikorose/DeepFilterNet/blob/d375b2d8309e0935d165700c91da9de862a99c31/DeepFilterNet/df/loss.py) | Full-band speech denoising; ERB envelope gains, multi-frame complex filtering, compressed magnitude/complex supervision, explicit lookahead configuration. | Good engineering reference for a model plus DSP and state contract. A gain-only mask cannot generate energy from an identically absent band. Filter ordering, lookahead and preset configuration must be inspected rather than inferred from a real-time label. Its denoising task does not certify musical inpainting. |

Two additional research routes were screened:

- [AEROMamba](https://arxiv.org/html/2411.07364v1) replaces AERO temporal modules
  with state-space blocks for music super-resolution and reports GPU efficiency
  improvements. This paper-level review does not verify its entire streaming
  state/normalization path or a CPU 128-frame deadline. Its recurrent component
  alone is insufficient to claim an end-to-end causal deployment.
- [Differentiable harmonic-plus-noise music BWE](https://mathieulagrange.github.io/ddspMusicBandwidthExtension/)
  explicitly predicts DSP synthesis parameters. Model/DSP cooperation can thus be
  a valid restoration design; the decisive issue is supervision against original
  clean audio and faithful synthesis, not a blanket ban on learning a DSP-related
  representation. Harmonic/noise modeling still needs polyphonic, percussion,
  stereo and codec-specific verification in our setting.

## Provenance and licensing boundary

Twenty selected source/license/config files were archived locally with repository
revision and SHA-256 in `data/research/implementation-review-20261026/sources.json`.
The four pinned revisions above are the inspected versions; upstream changes
are not silently included. No external audio, checkpoint, teacher output,
dependency or source implementation was added to SoundEx training or releases.

[AERO](https://github.com/slp-rl/aero/blob/6054bd918a099447f8e149f7e8bd55676e641b68/LICENSE)
and [AP-BWE code](https://github.com/yxlu-0102/AP-BWE/blob/751710f22404c27e5bcc983248f8b856a04b8422/LICENSE)
declare MIT; AP-BWE has a [separate weight notice](https://github.com/yxlu-0102/AP-BWE/blob/751710f22404c27e5bcc983248f8b856a04b8422/weights_LICENSE.txt).
[DeepFilterNet](https://github.com/Rikorose/DeepFilterNet/blob/d375b2d8309e0935d165700c91da9de862a99c31/LICENSE)
offers MIT/Apache-2.0 alternatives. [Apollo](https://github.com/JusperLee/Apollo/blob/e84bcacc59d5455f05d86a5c97dd4aeb3c14dbb6/LICENSE)
declares CC BY-SA 4.0: it is a method reference, not code or weights vendored into
our Apache-2.0 artifacts. Repository notices are not a new audit of all underlying
datasets or weight rights. Our prototype is independently implemented.

## Implemented prototype: paired learned spectral features

`training/models/spectral_texture.py` contains a small conditional critic with
two analysis resolutions, 512 and 1024 samples. It operates on calibrated
Hann/Parseval magnitude spectra of already supervised PCM, with noncentered
windows fully inside that support. A detached observed-input scale and fixed
absolute floor keep reference information out of normalization. Frequency
conditioning uses physical Hz; the candidate/observation texture images cover
8+ kHz, without a special cutoff fitted to P58 counterexamples. This still does
not guarantee preserved low-band PCM after OLA or DSP.

Each resolution uses two small smooth convolution layers and a classifier head,
without batch statistics or dropout. The critic is trained to distinguish
original references from frozen predictions on paired TRAIN material. Its
training has no inference role. During generator supervision, critic parameters
and reference features are frozen and the model receives squared distance to
the **paired original reference features**. Clean identity examples remain
supervised against their own original PCM.

The generator does **not** receive an unconditional realism reward. Standard
adversarial generator loss generally has a nonzero gradient even when its output
exactly equals the clean reference. Paired feature distance instead has zero
value and zero gradient there, including silence. This adaptation may sacrifice
some adversarial texture benefit; a critic trained against frozen outputs may
also learn an unhelpful shortcut. Neither issue is resolved by unit tests.

This is an auxiliary prototype, not a replacement for direct clean spectrum,
multi-resolution magnitude/temporal reconstruction, complex fidelity, identity
protection or actual Rust validation. No generator/DSP/export behavior changes.
No new restoration weights are claimed from training the critic.

## Verification and next experiment boundary

Known-answer tests cover exact-reference zero loss/gradient, silence and weak
signals, detached reference/input/critic paths, finite differences, batch/rate
isolation, physical frequency support and finite critic updates. The runnable
`scripts/preflight_spectral_texture.py` uses the two frozen P57 full generators
and the already registered P56 optimization/calibration crops, rechecking reserved
source aliases. It fixes eight critic updates per generator and freezes both
critic endpoints before calibration. Generator tensors must remain exactly
unchanged; feature-loss gradients must reach the generator and remain finite.
Eight critic steps only check integration, not useful learned texture or
convergence. P56 crops are reused TRAIN material, not a fresh test.

Completed evidence: `data/runs/spectral-texture-preflight-20261026/`. Both
2,818-parameter critics finish all eight updates; both endpoints are frozen before
the four calibration batches per generator. All sixteen fit/calibration gradient
checks are finite and nonzero, with generator gradient norms from 1.47e-6 to
1.96e-4. Both generator state dictionaries remain exactly unchanged. This range
is not a selected auxiliary weight or evidence that the gradients improve quality.
Registrations retain hashes, source snapshots, actual P56 crop lists and coverage;
critic checkpoints are local ignored artifacts. No new restoration weight is
produced, no human listening occurs, and no deployment timing is rerun.

Local regression: all **787 Python tests pass** (239 warnings, 66.27 seconds),
including 14 new numerical/attribution tests. Ruff lint and formatting pass for
355 Python files; English documentation/link/index checks pass for 137 Markdown
files. The full suite uses the scoped FFmpeg-full path documented in the earlier
environment failure record. These are local checks, not remote CI or new-device
evidence. All 789 preflight input/source bindings and both new completion receipts
were verified after execution.

Before a generator comparison, establish critic learning on broader TRAIN
coverage, source-disjoint calibration, noise/temporal-order controls and
music/speech gradient contribution under a fixed training-only scale rule.
Then register equal-budget control versus added paired feature supervision for
both small-model seeds, fresh matched Adam, unchanged source schedule and primary
losses. No holdout-based weight tuning or intermediate winner selection. Freeze
every endpoint before raw/native review; retain identity, weak-band PCM,
temporal, phase and source-level counterexamples. Record complete source-time
coverage, not just the eight-step preflight crop pool.

The generator architecture remains unchanged, so the critic introduces no
deployed operators or buffering. New generator weights would still require
ONNX parity, measured model-plus-DSP compute below 5 ms, 2.667 ms hop throughput,
paced continuity checks, separate fixed-buffer reporting and listening/device
evidence. No runtime qualification can be inherited solely from parameter count.
