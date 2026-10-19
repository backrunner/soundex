# Workspace layout

Keep source code, optional source indexes and documentation in Git. Local audio,
generated pairs and experiment state live under the ignored `data/` directory:

```text
crates/                    Rust runtime, DSP and CLI
docs/design/archive/       Historical design plans and audit baselines
models/                    Release model artifacts and their provenance
training/
  configs/                 Reproducible model/data/training profiles
  data/                    Import, audit, degradation and sampling code
  models/                  Python model definitions
  evaluation/              Evaluation implementation
  scripts/                 Executable workflow tools
  source_catalogs/         Public source documentation; acquisition indexes stay local
  tests/                   Training and export contract tests
  .venv/                   Local Python environment (ignored)
  .tools/bin/              Local FFmpeg symlink (ignored)
data/                      Local artifacts, all ignored
  raw/{music,speech}/
  processed/{music_library,speech_library}/
  source-discovery/         Candidate metadata; not approved training data
  catalogs/                Local input and final audited selection catalogs
  evidence/                Publisher/acquisition receipts and historical surveys
  reports/                 Quality, inventory and cleanup receipts
  maintenance/             Local deletion inventories and retention checks
  research/                Read-only upstream source snapshots; not vendored code
  runs/<run-id>/           Frozen config, source, logs and checkpoints
  archive/runs/            Inactive historical runs and unique audit checkpoints
```

`training/outputs` is a local compatibility symlink to `data/runs`, preserving
references in historical immutable manifests. New runs should use `data/` paths.
Retained survey receipts are historical evidence, not an approved source catalog.
Only the current audited catalog selects training inputs.
The completed [P28 small-model study](small-mechanism-study.md) lives under
`data/runs/small-mechanisms-p28-20261019/`, with committed frozen source,
`study/progress.json`, `pipeline.log` and `supervisor.json` recording successful exit.
`completion-audit-20261019.json` records the final artifact integrity check.
Its separate discarded preflight is under `data/runs/small-mechanisms-p28-preflight-20261019/`.
The [P27 mechanism audit](head-collapse-audit.md) remains under
`data/runs/head-collapse-p27-20261019-v2/`. The completed P26 large-model route is
retired from further training; `data/runs/broad-budget-p26-20261017-v2/` retains
the frozen source, complete checkpoints, exports and review evidence. Earlier runs are indexed in
[research history](research.md) and [native-lossless training](native-training-20261009.md). Retired Slakh audio/pairs
have been removed; their historical manifest and recipe receipts remain under
`data/evidence/retired-corpora/`. Historical generated audio, duplicate periodic
checkpoints and unused tool/cache copies are removed with deletion receipts.

The earlier [V5 experiments](goal-refinement-20261011.md) use
`data/runs/{calibrated,circular}-continuation-20261011/`; their frozen source,
resolved config, best/final checkpoints and logs remain local. Export/parity,
development and recording-held-out comparisons, metric diagnostics, listening WAVs
and resource receipts are under `data/reports/goal-refinement-20261011/`.
The listening score sheet is unfilled; generated listening files are not listening evidence.

The [V6 deployment-aware continuation](deployment-refinement-20261011.md)
uses `data/runs/deployment-continuation-20261011/` and
`data/reports/deployment-refinement-20261011/`. These contain the frozen source,
resolved configuration, local checkpoints/export, validation and stream comparisons,
proxy diagnostics and resource receipts. Training audio and experimental weights
remain outside Git.

The [architecture study](architecture-research-20261011.md) keeps a frozen source
under `data/runs/architecture-study-20261011/source/`, matched configs/checkpoints
under `experiments/{polar-small,shape-small,shape-large}/`, and evidence under
`data/reports/architecture-study-20261011/`. The matched V6 control is reused with
a fresh CPU validation/export/parity check. `progress.json` tracks training/export;
`delivery-progress.json` tracks the subsequent Rust quality and cost checks.
The [completed study report](architecture-study-20261011.md) records the selected
hash, quality tradeoffs and measured budgets. Local blind-listening materials are
under `data/reports/architecture-study-20261011/listening/`; their score sheet is
unfilled and does not establish a listening pass.

Run the optional source audit before generating codec pairs:

```bash
cd training
python scripts/audit_library.py --catalog ../data/catalogs/candidates.jsonl \
  --output-dir ../data/reports/music-quality
```

The tool fully verifies/decode-checks each recording, reports RMS, DC, peak,
full-scale runs, silence and sampled spectral power, and writes `accepted.jsonl`.
Flagged recordings are held for review; intentional distortion is not automatically
a recording defect. Narrow bandwidth alone is not grounds for rejection: spectral
statistics cannot prove lossless mastering history. Retain original publisher
files and acquisition evidence, and exclude known lossy transcodes separately.

Before cleanup, stop the exact superseded process, check open files and references,
record the deletion inventory, and verify retained paths afterward. Do not remove
raw audio, checkpoints or mixed `target/` trees just because they are ignored.

For a fresh run with the matching prepared VCTK speech supplement:

```bash
cd training
PATH="$PWD/.tools/bin:$PATH" python scripts/run_native_training.py \
  --catalog ../data/catalogs/training-music.jsonl \
  --run-dir ../data/runs/native-300-YYYYMMDD-HHMMSS \
  --processed-root ../data/processed \
  --config configs/diverse_lossless.yaml --minimum-tracks 300 \
  --curation-policy configs/curation_300.yaml \
  --evidence-root ../data/catalogs
```

Slakh is disabled in the current profile; existing directories do not enable it.
Use a profile with the exact recipe of the enabled supplements. The runner
verifies their immutable manifests, freezes committed code and the final music
catalog, prepares music pairs, derives speech's sampling probability from the
smallest retained music genre, then starts training from random initialization.
The run's `progress.json` and stage logs distinguish preparation from training.

During acquisition, `scripts/wait_for_native_training.py` can watch growing
`acquired.jsonl` receipts alongside an audited base catalog. It requires the exact
source commit, verifies prepared supplements before waiting and starts the above
pipeline only once the distinct music count reaches the configured minimum. A
finished or stopped collection below target fails visibly; it never reduces the
target or silently starts with fewer songs. Keep the queue's source/config snapshot
and its `progress.json` under `data/runs/<job-id>/`.

For the requested China/Japan/Korea/Europe/North America run, use the optional
[regional curation profile](dataset-curation.md). The first run uses a 300-work
minimum and basic regional/creator balance;
fine-grained genre and regional-vocal gaps remain advisory. Source review and
quality selection are required before starting. The count-only queue
has been superseded. The selected catalog now meets its 300-work target and
required checks; fine-grained coverage gaps remain acquisition work.

## Source cleanup (2026-10-14)

Removed the retired Slakh/MedleyDB importers, legacy dataset downloader, mix-only
pruner, their obsolete storage guide and importer-only tests. Git history at
`723c028` retains the old files. Current folder/catalog ingestion, mono/stereo
handling, recipe compatibility, licensing records and all runtime models remain.
Docker now accepts prepared libraries, copies the current training dependencies,
and rejects removed acquisition options explicitly.

The root READMEs show current status and usage. Earlier results are indexed in
`docs/research.md`; continuation instructions live in `training/EXPERIMENTS.md`.
Historical model/loss and study modules remain because they are shared by current
experiments or supply reproducible controls. Acquisition queue helpers also still
serve source auditing and retries. Historical dataset identifiers in checkpoint
and recipe readers remain for compatibility; they do not acquire old corpora.

Local deletion receipts are under `data/reports/workspace-cleanup-20261014/`.
P19 uses its frozen source from `b2b8a2d`, its existing Python environment and
registered parent inputs. Resume it using that pipeline/source, not edited main.
Audio, source-rights evidence, frozen runs/checkpoints and mixed Rust build trees
were preserved. Only regenerable caches outside frozen runs are eligible here.

Cleanup validation: 468 local Python tests passed (including six container shell
cases), Ruff lint/format, shell syntax and changed-document links passed. A staged
copy matching Docker's COPY instructions imported train/export/evaluation modules
successfully; a full CUDA container build/run was not performed. Removed nine
regenerable cache directories (2,865,011 bytes); active pipeline and child PIDs
remained 5235 / 5994, and the wide training arm started after cleanup.

## Feature-cache cleanup (2026-10-19)

After P26 and the P27 audit completed, removed **146,013 regenerable NPZ feature
cache files**, 143.76 GiB of logical data / 144.26 GiB of allocated blocks. Observed
volume free space increased by about 144.24 GiB during deletion; other processes
can also affect the volume counter.

| Cache | Files | Logical bytes removed |
|---|---:|---:|
| `factor-p6-20261012/study/cache` | 86,902 | 91,870,013,536 |
| `causal-p2-20261012/cache` | 32,857 | 34,735,368,976 |
| `causal-p3-20261012/study/cache` | 9,434 | 9,973,322,912 |
| `causal-p4-20261012/study/cache` | 8,410 | 8,890,782,880 |
| `envelope-p5-20261012/study/cache` | 8,410 | 8,890,782,880 |

Paths above are relative to `data/runs/`. This cleanup explicitly includes these
derived caches inside historical run directories; it does not remove their
immutable source, configuration, reports or checkpoints. A scan of 1,030
registration/configuration/completion documents found no individual NPZ filename
references. The old cache directory paths remain valid and empty; loaders can
reconstruct their contents from retained PCM, crop specifications and Rust detectors.

Deletion inventory, pre-delete inode/size/mtime checks, completion record and
post-cleanup retention checks are under `data/maintenance/cache-cleanup-20261019/`.
Raw audio, prepared pairs, rights evidence, manifests, every P26 checkpoint and
ONNX export, frozen sources and review receipts remain local. No raw data or
experimental weights are committed. Shared historical model/loss modules remain
because current imports and reproducible controls use them; deleting them would
break those workflows. The READMEs now point to current decisions rather than
repeating a growing list of retired experiments.

Post-cleanup verification passed: all **82 P26 completion receipts and 2,555
referenced file hashes**, registered inputs and frozen sources remain intact.
Rebuilding a five-pair diagnostic batch with an empty disk cache reproduced the
entire prior inspection JSON exactly and left the read-only cache empty.
Local validation: **548 Python tests passed**, Ruff lint/format and changed-document
links passed. No new listening, physical-device or model-quality qualification is claimed.
