# SoundEx engineering conventions

Use `rtk` or `rtk proxy` for shell commands. Local usage guide: `~/.codex/RTK.md`.

## Project goal

SoundEx restores lossy compressed audio. The model should inpaint missing or damaged
high-frequency spectra, reconstructing their shape, detail, energy and transients.
DSP provides band fusion, loudness control, phase handling and artifact suppression
so that the final output approaches the paired original lossless recording.
Raising treble alone is insufficient. Reducing enhancement or leaving compressed
input unchanged is not success. Estimating missing information does not guarantee
an exact inverse of lossy compression.

## Training and validation

- Synthesize lossy inputs from native lossless music with documented rights and
  speech mixed at the agreed sampling ratio. Keep pairs strictly aligned. Handle
  mono, stereo, silence and low-energy material correctly.
- Design loss against the original lossless reference. Missing-band spectral
  shape/magnitude, energy envelopes and transient detail are primary targets;
  phase at meaningful bins and full reconstructed waveforms constrain fidelity.
- Supervision and candidate assessment prioritize spectral agreement: jointly
  inspect multiresolution magnitude, high-band shape, band energy and real temporal
  variation before phase, waveform, preserved bands and listening. A small waveform
  regression alone does not reject a spectral candidate. Matching band energy does
  not establish spectral shape, phase or sound quality. Distinguish training
  analysis windows from deployment lookahead.
- The baseline is not ground truth. Judge gains against paired lossless references
  and untreated compressed input. Audit historical training budgets, data leakage,
  development-set selection bias, DSP policy and cross-recording generalization.
  Unequal historical budgets cannot establish architecture superiority. Reused
  held-out recordings are rechecks, not proof against overfitting. Include an
  identity model through actual DSP to isolate limiting/synthesis effects. Report
  conflicts between safe limiting and the original reference; never substitute a
  limited reference and describe the result as original-audio restoration.
- New losses require correct-reference behavior, silence/weak-energy gradients,
  finite differences and deployment correspondence checks. Do not tune thresholds
  to tracks or validation counterexamples. Separate representational feasibility
  and capacity ceilings from generalization.
- Constrain preserved bands, crossover seams and adjacent-frame variation errors
  to protect low frequencies, stereo imaging and continuity. Temporal consistency
  must follow the reference's real changes, not smooth away transients.
- Report model-only and model-plus-DSP reconstruction errors separately. State how
  differentiable training proxies differ from Rust deployment and verify them
  through actual Rust streaming.
- Verify a direct supervision path to original lossless high-band spectra.
  Distinguish direct reconstruction, reconstruction through DSP and spectral
  consistency auxiliaries. Identity DSP is a control/fidelity diagnostic, not a
  substitute for paired high-band targets. Better direct predictions do not
  replace final-output validation.
- Compare losses with fixed versions, weights, scales, splits and sampling ratios.
  Do not claim improvement from totals with different objectives. Added noise can
  reduce weak-bin LSD; also assess high-band spectral convergence, energy,
  transients, low-band fidelity, stereo and blind listening. Loss, SI-SDR or
  spectral coverage alone is insufficient.
- For very weak reference bands, report relative shape and calibrated absolute PCM
  RMS/dBFS errors together. Unnormalized FFT power is not PCM dBFS. Retain every
  counterexample. Large relative scores do not establish audible severity; small
  absolute errors do not establish restoration gains or listening quality. Never
  set acceptance thresholds retrospectively around known failures.
- Compare the model, DSP and combined paths on the same material. Candidates without
  overall quality evidence remain experimental and do not replace default weights.
  Development validation is not independent testing or release qualification.
- Discriminator classification and synthetic-corruption controls diagnose that
  classifier, not overall restoration or the usefulness of paired feature loss.
  Preserve registered failures without generalizing them to every GAN/feature
  architecture. Compare learned and random features independently; unconstrained
  PCM-gradient probes do not replace matched generator and native-output evidence.
- Freeze candidate-selection rules and model hashes before held-out scoring. Label
  reused diagnostic recordings as held-out rechecks. Audit training isolation by
  recording/source and report genre, region and channel coverage gaps. Do not join
  left/right excerpts from different source times into stereo. Do not extrapolate
  local means to all music.
- Reserve registered evaluation sources before scoring. Future training and
  calibration must exclude their file hashes, decoded PCM hashes and connected
  work aliases, including local `data/runs/*/evaluation-catalog.jsonl` catalogs.
  A longer crop or different codec from an exposed recording is a recheck, not a
  new source-disjoint test. Keep per-recording aggregation and declared coverage
  gaps separate from artist independence and listening evidence.
- Separate model history, DSP-state warmup and supervised regions. Warmup must not
  enter supervision or update frozen statistics. Rescore experiments with different
  loss weights under one common objective; inspect raw and deployed errors.
- Catalog track counts are not actual training coverage. Report sampled recordings,
  codecs, channel views, source intervals and unique supervised duration after
  removing warmup/overlap. A finite diagnostic crop bank is not full-library training.
- New temporal branches need evidence of learning and history dependence. Use
  common warmup/crops and retain repeated-current-frame ablations. Nonzero parameters
  and a slightly lower total loss alone do not show history use or audio improvement.
- Prefer papers and official implementations for architecture research. Distinguish
  music restoration, speech enhancement and ideal low-pass upsampling. Compare
  representation and capacity separately. Short single-seed fine-tuning cannot
  establish an architecture family's ceiling.
- A 512-step calibration diagnoses early optimization only. It cannot reject
  capacity or replace full training. Give controls and all capacities the same
  preregistered adequate budget, finish every trajectory, freeze endpoints, then
  review together. Reaching the step budget does not prove convergence. Do not
  select intermediate scores or shorten controls after observing results.

## Real-time behavior and delivery

- Follow `docs/realtime-standard.md`: no inference, allocation or blocking in the
  callback; maintain a fixed timeline and smooth failure fallback/recovery. Measure
  added output delay, processing time, CPU/RSS and deadline misses.
- Experimental capacity is screened by latency, without parameter/file-size caps.
  Model-plus-DSP compute must remain below 5 ms, excluding fixed buffering as
  confirmed by the user, while sustaining continuous throughput and each hop's
  deadline (48 kHz/hop 128: 2.667 ms). Report compute and full output delay separately;
  disclose the fixed 256-frame buffer. Production artifacts retain the current
  <=2M / <8 MiB contract until an explicit production-contract migration. Experimental
  exports may explicitly bypass only the size gate. Whole-file RTF or GPU throughput
  cannot replace per-hop CPU, lookahead/buffering and streaming availability checks.
- Causal context reads only past/current frames. Each physical channel has independent
  history; silence and bypass still advance it. Train with real warmup and declare
  export input/output timelines. Reset, dropped-frame recovery and model changes
  clear history. Return aligned dry audio when history is insufficient. Enhancement
  warmup is not additional output delay.
- Preserve real transients and stable continuous output. Finite values, limiting,
  block-size consistency, silence and channel isolation are basic checks, not
  substitutes for real-device validation or listening.
- Source, documentation and formally released weights follow the project's
  Apache-2.0 policy. Keep original training audio, pairs, local checkpoints and
  experiment reports in ignored directories. Commit source documentation, not
  original training material.
- Commit as `BackRunner <dev@backrunner.top>` using Conventional Commits, grouped
  by responsibility.

## Experiment memory, documentation and cleanup

- Before proposing or launching a repeat, consult `docs/failed-experiments.md` and
  the linked detailed report. Record the changed hypothesis/control and why existing
  evidence does not already answer it. A larger budget alone requires a documented
  convergence or exposure rationale, not repeated hope.
- Every unsuccessful validation must remain recorded: exact endpoint/budget and
  reference, failed criterion, counterexamples, evidence paths, scope limitations
  and a concrete condition for revisiting. Distinguish quality failures, runtime
  failures, invalid measurements and inconclusive diagnostics. Never discard failed
  runs or retry until a benchmark passes. Update the registry in the same work item.
- Maintain authored project documentation in English, including this file and
  `.agents/`. Preserve exact source titles, attribution names, literal paths and
  immutable historical evidence where translation would alter provenance.
- Inventory cleanup candidates before deletion. Remove only confirmed regenerable
  caches or unused build intermediates after checking active users and references.
  Preserve training sources, manifests/licenses, checkpoint hashes, frozen binaries,
  source snapshots and failed-validation evidence. Record reclaimed bytes and verify
  retained input integrity afterward. Ignored does not mean disposable.
