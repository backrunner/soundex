# Current native-lossless training run

This run uses **300 distinct reviewed music works from 22 creators** and a
separate VCTK speech supplement. Unique source music totals **1,212.39 minutes**;
speech adds **23.85 minutes**. These are source durations across training,
validation and test, not multiplied codec/channel examples.

The current job is `regional-native-300-no-slakh-20261009-181934`. Its training
implementation is frozen at commit `abe0e497eca5d72643c18f2929f1c7527c7143b6`;
later documentation commits do not change the frozen code. Initialization is
random, without a historical checkpoint or teacher. MUSDB and Slakh are disabled.

The original 200 epochs are complete. A subsequent [loss audit and 8-epoch continuation](loss-audit-20261010.md)
imports only its native epoch-168 generator with explicit lineage; it remains an
experimental comparison and does not replace that baseline. The audit also identifies
and fixes the archived-source Git metadata mismatch without rewriting original checkpoints.

## Data and model contract

- Recipe SHA-256:
  `7fda9e58ab583617f3514f9dd9cfcf87f29871ae0587e91ca29ac0ddaf6b29d4`.
- Native references are creator-published lossless PCM masters. Source grants,
  contributor credits, source pages, edits and recording hashes stay bound to
  the selected catalog; source documentation is public, audio and catalogs local.
- Source/work groups determine disjoint training, validation and test membership.
  Mono input is supported; stereo channel roles do not multiply recording draw mass.
- Pairs cover 44.1/48 kHz, MP3 CBR/VBR, AAC-LC and Vorbis. AAC 160 kbps is held
  out from training. Silent channel-role segments are skipped instead of creating
  invalid targets; this does not discard a complete source recording.
- Speech probability matches the smallest labeled genre's actual training
  recording share after preparation, rather than a fixed 20% estimate.
- Model/audio profile: [diverse_lossless.yaml](../training/configs/diverse_lossless.yaml),
  FFT 256 / hop 128, magnitude and phase streams, batch size 16, 2,048 draws per
  epoch, 200 epochs, 50 epochs of generator warm-up before adversarial updates.

The [300-work curation policy](../training/configs/curation_300.yaml) has zero
blocking gaps and 13 advisory coverage gaps. R&B and Chinese/Japanese/Korean
vocal music remain acquisition gaps. The source pool supports the first pilot;
it is not complete genre or regional coverage. See
[curation](dataset-curation.md) and [the latest sources](pilot-300-sources-20261009.md).

## Live status and artifacts

Training completed at **01:00 Singapore time on 2026-10-10**: 200 epochs and
25,600 optimizer steps on MPS, with an 806,276-parameter generator initialized
randomly. The selected checkpoint is epoch 168, with validation total 11.6241
and high-band error 10.8692. The final epoch reports 11.7189 / 10.9424.
The first [deployment diagnostic](native-validation-20261010.md) checks this
selected artifact; training completion does not establish release qualification.

| Corpus | Training recordings / pairs | Validation recordings / pairs | Test recordings / pairs | Total pairs |
| --- | --- | --- | --- | --- |
| Music | 253 / 71,327 | 26 / 8,308 | 21 / 6,451 | 86,086 |
| Speech | 193 / 13,896 | 59 / 4,720 | 18 / 1,440 | 20,056 |

The **106,142 pairs** retain disjoint work groups and exclude the held-out codec
from training. Speech is **1/254 = 0.3937008%** of training draws, equal to the
smallest training genre (funk, one recording); music is 253/254. This describes
sampling probability, not speech's share of stored pairs or audio duration.

The historical first epoch reported generator loss 70.9548, validation loss 66.6289 and
high-band validation loss 62.8678. These initial objective values do not establish
restoration improvement. Both schema-1.2 checkpoints passed configuration,
recipe/manifest hash and finite-generator-state checks. The epoch-1 best candidate
is **11,292,919 bytes (10.77 MiB)**, including training state; its SHA-256 is
`30826811486ea1e1dab8bcdc8c4a78c7e74e67826a10ee387b80d31c77ad3367`.
A local audit snapshot is retained under `data/evidence/native-training-20261009/`;
the completed run's best/final checkpoints are separate from this first-epoch snapshot. This is not
the size of a qualified inference export.

On the maintainer's local workspace, read the actual run files:

```bash
cat data/reports/current-training-job.json
cat data/reports/current-training-dataset.json
cat data/reports/current-new-weight-verification.json
cat data/runs/regional-native-300-no-slakh-20261009-181934/training-run/progress.json
tail -n 10 data/runs/regional-native-300-no-slakh-20261009-181934/training-run/preprocess-music.log
tail -n 10 data/runs/regional-native-300-no-slakh-20261009-181934/training-run/training.log
```

The runner requires completed preparation and immutable manifest/file audits
before optimization. Those checks passed for this run.
The runner writes its source receipt, exact config, curation report, genre
sampling report and stage logs under the job's `training-run/` directory.
`checkpoints/final-resume.pth` is updated atomically after each completed epoch;
`checkpoints/best-validation.pth` is the selected export candidate.

Training checkpoints and raw/processed audio stay local and ignored. No new
restoration artifact is approved for publication merely because training runs.
ONNX export, numerical parity, real-music quality/listening, continuity, runtime
latency/resources and exact-artifact rights review precede an official
Apache-2.0 release. See [evaluation](../training/evaluation/README.md),
[performance](performance.md) and [model status](../models/README.md).

## Workspace cleanup

Current music originals, speech, active staging/published data and the current
run are retained. Slakh audio/pairs and unused historical generated data are
removed with per-file inventories, freeing **7,128,404,337 bytes (6.64 GiB)**.
The cleanup inventory covers 2,364 targets and 24,286 files. Historical manifests, recipe/metrics and
unique audit checkpoints remain historical evidence. Inactive runs are kept in
`data/archive/runs/`, separate from the current job; 20 old run/pointer entries
were relocated. The best/final historical checkpoints remain audit-only. All 300
selected original audio hashes and their source-review evidence were checked again
after cleanup. The source-survey compatibility
alias stays available for immutable source references.

Cleanup receipts are local at `data/reports/workspace-cleanup-20261009/`.
See [workspace layout](workspace.md) for the active and historical directory roles.
