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
Completed local run: `data/runs/adapter-continuation-p50-20261024/`.
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

## Completed training and raw review

All four trajectories finish 128 actual updates without no-ops. Same-batch total
decreases on 124/128 and 127/128 persistent updates, and 128/128 updates for both
reset controls. Persistent Adam has two / one non-descending local directions;
reset has none. Reset therefore gives more consistent local descent, but those
counts are not endpoint quality evidence. All six start/endpoint hashes were
frozen before scoring 32 fit and 32 globally separated review batches per model.

The following relative error changes use each seed's own start, the same review
crops and unchanged objectives. Negative means improvement. Raw music has 128
paired crops; native music has the registered 32-crop subset. Their means must
not be subtracted to attribute DSP effects because the populations differ.

| Seed / optimizer history | Raw music amplitude | Raw music MR | Native music MR | Raw common total change |
| --- | --- | --- | --- | --- |
| 12 / persistent | -0.021910% | -0.007647% | -0.009041% | -1.7383e-5 |
| 12 / reset | -0.025898% | -0.009077% | -0.010882% | -1.9095e-5 |
| 13 / persistent | -0.008340% | -0.003046% | -0.003736% | -5.6340e-6 |
| 13 / reset | -0.012481% | -0.004684% | -0.005983% | -8.0126e-6 |

All four endpoints also improve mean fit total and fit music MR on the fixed fit
recheck. Review total improves beyond the numerical floor in 28/32 batches for
each seed-12 arm, and 25/32 for each seed-13 arm; the remaining batches worsen.
Paired amplitude improves in 31/32, 32/32, 30/32 and 30/32 batches respectively;
paired MR improves in 32/32, 31/32, 30/32 and 30/32. Near-numerical changes are
retained separately in the raw report. This is reproducible small spectral
progress on this reused-TRAIN population, not a convergence/generalization claim.

Raw speech MR also decreases by 0.005065%, 0.005864%, 0.001839% and 0.002712%
in the same row order. Music envelope error decreases 0.002308-0.003940%.
However, music complex error rises 0.006269-0.010763%, and MR dynamics rises
0.001795-0.002043%. Direct-frame dynamics changes are mixed across starts.
Mean paired music HF PCM RMS error increases by 3.03e-8 to 3.26e-8. Spectrum
agreement and waveform agreement have not improved together, even though the
unchanged scalar objective decreases.

### Preservation and calibrated absolute errors

All four endpoints pass the 16-bit identity and paired-low4k reference rulers.
All paired-low4k review cases also pass at 20 and 24 bits. Identity violations
on the 160-example review role remain:

| Seed / optimizer history | 20-bit identity failures / 160 | 24-bit identity failures / 160 |
| --- | --- | --- |
| 12 / persistent | 49 | 121 |
| 12 / reset | 54 | 107 |
| 13 / persistent | 16 | 85 |
| 13 / reset | 33 | 98 |

The fit recheck also retains failures: 20-bit identity counts are 48/61/25/37
and 24-bit counts 115/111/105/112 in the same row order. Both seed-13 arms have
one 24-bit paired-low4k peak failure on fit draw 112, example 1, recording
`music_library:1453597560-f8f4d240ebfc8e68`. The persistent arm's peak-error
increase is 7.404e-8 PCM against the fixed 5.960e-8 allowance, while its RMS
error increase is 4.382e-10. Keep this small absolute counterexample; the review
pool's low-band pass is not an all-data claim, nor is this evidence of audible
low-band damage. All fit/review 16-bit checks and 20-bit low4k checks pass.

Unlike P49's independent first steps, cumulative training no longer preserves
every 20-bit identity case. Do not reinterpret that earlier pass as a persistent
training guarantee. These are existing diagnostic rulers, not audible-damage
thresholds; all per-example counters and original references remain unchanged.

Mean raw music identity full-PCM RMS error is 7.538e-6, 7.544e-6, 8.803e-6 and
8.817e-6 in table order. Its increase is 6.672e-8, 7.286e-8, 1.282e-8 and
2.709e-8, respectively. Levels are approximately -102.46 to -101.09 dBFS.
Mean identity HF PCM RMS is 2.368e-6 to 2.726e-6 (-112.51 to -111.29 dBFS),
increasing by 6.61e-8 to 2.91e-7. Speech identity full-PCM RMS is 4.655e-7 to
7.922e-7 (-126.64 to -122.02 dBFS); HF RMS is 1.323e-7 to 2.121e-7
(-137.57 to -133.47 dBFS). Levels use `20*log10(mean clip RMS)`, not pooled
recording RMS or FFT power. Speech identity MR rises 79-143% from a very small
base; that relative score does not establish comparable audible severity.

Reset gives slightly more spectral gain and generally more identity drift than
persistent Adam. Neither dominates every preservation/temporal component. Do not
select a release winner from the small mean advantage or reject all spectral
progress solely because a waveform metric increases by a tiny absolute amount.

## Actual Rust output and timing

All seven exports pass the inherited sequential output/state parity checks, and
all 560 native paired/identity roles finish with finite aligned output. Native
music MR improves by 0.003736-0.010882% versus the matching start, and native
speech MR by 0.004046-0.009720%. The spectral direction survives real DSP, but
native music MR dynamics still rises 0.000993-0.001571% versus the start.

Against untreated compressed audio on the native subset, endpoint music MR is
0.553-0.978% lower and speech MR 4.273-5.478% lower. These include the historical
base model's benefit; they are not gains attributable solely to P50. Music MR
dynamics is still 1.425-1.479% worse than untreated, and music HF PCM RMS error is
2.490-2.525% worse. This is not an overall restoration or listening pass.

The identity model through real DSP has mean music identity full-PCM RMS error
0.00286440 (-50.859 dBFS), versus 0.00286491-0.00286523 for the four endpoints.
The similar control error shows substantial existing DSP/safety processing error
relative to the original reference; it must not all be attributed to training.
Raw adapter-induced identity drift nevertheless remains separately measurable.
References are not limited to make the DSP appear more faithful. Native speech
identity RMS is 5.82e-8 to 8.17e-8; its large relative weak-band changes likewise
require these absolute values for interpretation.

Across 20 final timing runs, 52,000 measured hops (54,000 including warmup):

- Maximum model-plus-DSP compute: **1.965292 ms**; largest per-run p99:
  **1.312835 ms**. Zero >=5 ms calls and zero compute service overruns.
- Two paced late hops, both in seed-12 reset; maximum lateness **3.753121 ms**.
  This remains a continuous-scheduling failure, even with compute below 5 ms.
  This sequential timing run does not show that resetting Adam causes lateness.
  Other runs and historical failures are not erased by passing compute results.
- Rust process peak RSS: **29.59-30.00 MiB**. Paced process user+system CPU is
  0.58-0.62 seconds over 2.95 seconds wall time, about 19.7-21.0% of one core.
  These whole-process numbers include loading/warmup, not callback-only CPU.
- Each endpoint has 480,809 parameters and a 1,972,748-byte ONNX file
  (about 1.88 MiB). Isolated ORT warmed RSS increment is about 13.31-13.50 MiB;
  that separate Python/ORT measurement is not the Rust process peak.
- The unchanged fixed buffer is 256 frames: **5.333 ms at 48 kHz** and
  **5.805 ms at 44.1 kHz**, excluded from compute. These synthetic tests do not
  certify physical-device callbacks, recovery or listening quality.

## Decision, failures and next diagnostic

The scaled high-support adapter makes actual accumulated spectral progress and
lowers the common objective on both starts. Thus the previous overshoot diagnosis
was actionable. Persistent Adam momentum is not required for this small gain;
reset improves spectra slightly more but generally increases identity drift.
P49's first-step preservation does not survive accumulation at the tighter rulers.
The completed pilot does not justify blanket rejection of the architecture,
threshold relaxation, promotion or an unchanged long training run.

The next targeted question is whether the frozen feature representation drives
the adapter similarly on clean and compressed inputs. Inspect paired/identity
adapter-feature responses and band contributions at these frozen endpoints,
with fixed source-separated fit/review roles and the zero-adapter controls.
Any proposed separation or preservation mechanism must retain actual paired
spectral gains, not simply suppress the adapter. This differs from P39's shared
encoder gradient audit: the encoder is now frozen, steps are calibrated and the
remaining failure accumulates inside the isolated high-band adapter. No such
follow-up or new training is launched by this report. Worker scheduling needs
its own measured fix; an optimizer change is not a scheduling remedy.

The full suite passes 725 tests; Ruff, English documentation checks and remote CI
pass for the implementation. Focused tests cover source aliases, fixed mixture,
exact P49 first-step correspondence, persistent moments, interrupted optimizer
resumption and actual frozen-review receipts. The completed run verifies all
766 registered inputs, all six checkpoint hashes, raw-review receipts and native
artifact hashes. A completed-run re-entry verifies receipts without retraining.
`assessment.json`, `cost-summary.json`, `native-review/summary.json` and the
hash-bound `interpretation.json` retain the results and every counterexample.
No execution failure occurred in the registered run; identity/temporal and paced
scheduling failures remain recorded. No default or qualified release weight changes.
