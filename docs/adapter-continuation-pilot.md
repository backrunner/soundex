# P50: cumulative small high-band adapter updates

## Hypothesis and fixed protocol

[P49](adapter-scale-calibration.md) establishes that the original first Adam
displacement overshoots, and that fixed 0.001 scaling is more conservative than
the largest fit-admissible displacement. Its reset probes cannot establish
whether spectral gains accumulate or whether Adam history helps after the first
step. P50 tests that missing mechanism rather than repeating the original rate,
changing the loss, increasing model capacity or relaxing preservation rulers.

- Use both original P38 starts and P48's high-support adapter. The base feature
  path, original decoder and budget head remain frozen. Only the 49,794 adapter
  parameters can train; total model size stays 480,809 parameters.
- Run two matched arms per start for 128 actual updates: persistent Adam moments,
  and fresh Adam at every step. Both retain the cumulative adapter parameters.
  Each update computes the existing Adam proposal at 1e-4, then applies exactly
  0.001 of its displacement. Keep betas, epsilon, clipping and the original loss.
  This is explicit displacement scaling, not a claim that a different optimizer
  trajectory is identical to P49's independent reset probes.
- Four equally weighted batches per update, each with four music crops and one
  speech crop plus their clean identity counterparts. Keep true causal warmup
  and the existing direct/MR supervision region; no new DSP supervision or loss.
- Partition the existing P45 finite crop bank by connected `track_id` and
  `source_checksum` aliases, with a deterministic digest modulo five allocating
  review components. Select recording-balanced batches from the resulting roles.
  All 512 fit batches are globally disjoint from all 32 review batches by both
  identifiers. Preserve the bank's source-time exclusions and original references.
- The sources were used in historical training/diagnostics. This is a new
  pilot-level separation within reused TRAIN, not an independent generalization
  test. Do not compare its scores directly with P49's different batch population.
- Fix all schedules and 128-step endpoints before training. No guard-score access,
  adaptive step choice, veto, early stopping or intermediate model selection.
  Keep before/after same-batch scores, actual parameter movement and the Adam
  directional derivative, including increases and numerical-scale changes.
- Finish all four trajectories and freeze hashes before review. Evaluate every
  start and endpoint on 32 review batches and 32 evenly spaced fit batches.
  Compare original-reference spectra, temporal/complex errors, clean identity,
  calibrated absolute PCM and all 16/20/24-bit rulers. Tiny waveform regressions
  alone do not reject a spectrum-first candidate; tiny absolute changes do not
  establish restoration either.

This is a bounded optimizer/accumulation pilot. The 128-step budget matches the
preceding adapter mechanism pilot and is not sufficient evidence of convergence,
an architecture ceiling or full-library training. No automatic long-run extension
or default-weight replacement follows from completion.

## Native validation and evidence

After the complete endpoint barrier, export all six start/endpoint models and an
identity-through-real-DSP control. Reuse the tested P48 ABI/export implementation;
its inherited study metadata identifies that implementation, while P50's sidecars,
checkpoint hashes and run registration identify this experiment. Preserve
non-deployable experimental markers and sequential state/output parity checks.

Replay eight evenly spaced review batches through the frozen Rust binary for
560 paired/identity mono-view roles, with original lossless targets and untreated
controls. This is not simultaneous stereo music or listening qualification.
For each of four endpoints, measure 3,000 hops at each 44.1/48 kHz mono/stereo
configuration and 1,000 paced 48 kHz stereo hops. Retain every timing result;
do not repeat until it passes. Run timing without concurrent project training,
tests or builds. Report model-plus-DSP compute separately from pacing, CPU/RSS
and the fixed 256-frame buffer (5.333 ms at 48 kHz). Historical runtime failures
remain open until new evidence actually addresses them.

Entry point: `training/scripts/run_adapter_continuation.py`.
Role construction and updates: `training/evaluation/adapter_continuation.py`.
Planned local run: `data/runs/adapter-continuation-p50-20261024/`.
Original audio, checkpoints, source snapshots and reports remain ignored local
artifacts. Keep registration, schedule, optimizer state, checkpoint/receipt hashes
and all failures; resumption must reproduce uninterrupted optimizer state.

## Registered exposure

The common fit schedule has 2,560 paired crops (2,048 music / 512 speech)
from 351 recordings. Its unique MR-supervised support is 986.623 seconds
(16.444 minutes), excluding warmup and overlapping intervals across codecs and
channel views. The 160 paired review crops (128 music / 32 speech) cover
83 recordings and 71.706 seconds of unique MR support. Identity counterparts
are additional roles, not additional source audio. Reusing the schedule across
four trajectories does not multiply independent duration. Both global recording
and source-checksum intersections are zero.

All nine existing codec recipes appear in both roles. Fit left/mid/mono/right/side
counts are 413/788/572/405/382; review counts are 24/52/32/31/21. These are mono
views, not joint stereo evaluation. Fit music covers ten named genre labels plus
223 unlabeled crops; review covers ten named labels plus nine unlabeled crops.
Review has no hiphop, and its three funk crops have no funk-labeled fit counterpart.
The finite bank and sampling do not establish genre balance.

For China/Japan/Korea/Europe/North America/unknown source metadata, fit counts are
204/351/442/190/639/222 and review counts 12/14/20/20/39/23. These labels are not
verified national musical styles. Per-recording IDs, source intervals, codec,
channel and metadata counts are retained in the registration. Base-model history
still includes reused sources; this is pilot-level isolation only.

## Status

Implementation and preregistration checks pass. Focused checks cover
source aliases, fixed speech ratio, exact P49 first-step correspondence, persistent
moments, interruption/resumption and actual frozen-review receipts. The full suite
passes 725 tests; Ruff and the English documentation checks pass. Four fixed-budget
trajectories have been launched under `caffeinate`, with atomic checkpoints and
optimizer state. Results remain pending; process liveness is not completion or
quality evidence. Completed results and any failures will be recorded here after
execution.
