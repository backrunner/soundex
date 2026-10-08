# Workspace layout

Keep source code, optional source indexes and documentation in Git. Local audio,
generated pairs and experiment state live under the ignored `data/` directory:

```text
crates/                    Rust runtime, DSP and CLI
models/                    Release model artifacts and their provenance
training/
  configs/                 Reproducible model/data/training profiles
  data/                    Import, audit, degradation and sampling code
  models/                  Python model definitions
  evaluation/              Evaluation implementation
  scripts/                 Executable workflow tools
  source_catalogs/         Portable original-audio indexes, without audio bytes
  tests/                   Training and export contract tests
  .venv/                   Local Python environment (ignored)
  .tools/bin/              Local FFmpeg symlink (ignored)
data/                      Local artifacts, all ignored
  raw/{music,speech,slakh2100}/
  processed/{music_library,speech_library,slakh2100}/
  source-discovery/         Candidate metadata; not approved training data
  reports/                 Quality, inventory and cleanup receipts
  runs/<run-id>/           Frozen config, source, logs and checkpoints
```

`training/outputs` is a local compatibility symlink to `data/runs`, preserving
references in historical immutable manifests. New runs should use `data/` paths.
Retained survey receipts are historical evidence, not an approved source catalog.
Only the current audited catalog selects training inputs.

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
