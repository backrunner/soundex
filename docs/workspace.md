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
  runs/<run-id>/           Frozen config, source, logs and checkpoints
  archive/runs/            Inactive historical runs and unique audit checkpoints
```

`training/outputs` is a local compatibility symlink to `data/runs`, preserving
references in historical immutable manifests. New runs should use `data/` paths.
Retained survey receipts are historical evidence, not an approved source catalog.
Only the current audited catalog selects training inputs.
The current run and artifact locations are documented in
[native-lossless training](native-training-20261009.md). Retired Slakh audio/pairs
have been removed; their historical manifest and recipe receipts remain under
`data/evidence/retired-corpora/`. Historical generated audio, duplicate periodic
checkpoints and unused tool/cache copies are removed with deletion receipts.

The earlier [V5 experiments](goal-refinement-20261011.md) use
`data/runs/{calibrated,circular}-continuation-20261011/`; their frozen source,
resolved config, best/final checkpoints and logs remain local. Export/parity,
development and recording-held-out comparisons, metric diagnostics, listening WAVs
and resource receipts are under `data/reports/goal-refinement-20261011/`.
The listening score sheet is unfilled; generated listening files are not listening evidence.

The latest [V6 deployment-aware continuation](deployment-refinement-20261011.md)
uses `data/runs/deployment-continuation-20261011/` and
`data/reports/deployment-refinement-20261011/`. These contain the frozen source,
resolved configuration, local checkpoints/export, validation and stream comparisons,
proxy diagnostics and resource receipts. Training audio and experimental weights
remain outside Git.

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
