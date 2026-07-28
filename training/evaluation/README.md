# Deployment evaluation protocol

Release evaluation uses immutable `test` manifest rows only. Every row is compared
as clean reference, degraded baseline, and enhanced output. Enhancement always runs
through `soundex-stream-eval`, which uses the production Rust gate, causal STFT,
crossover, limiter, latency compensation, and flush behavior. Python never
substitutes a waveform implementation for release scoring.

## Reproducible sequence

From `training/`, export the checkpoint selected by fixed validation metrics, then
generate artifact-bound cross-runtime evidence:

```bash
python export_onnx.py \
  --checkpoint checkpoints/best-validation.pth \
  --output ../models/soundex-v1.onnx

python -m evaluation.parity \
  --checkpoint checkpoints/best-validation.pth \
  --model ../models/soundex-v1.onnx \
  --output ../models/soundex-v1.parity.json
```

Run held-out evaluation with an installed ViSQOLAudio command. The command template
must accept `{reference}` and `{degraded}` WAV paths and print JSON containing
`moslqo`/`mos`/`score`, or a named `MOS-LQO: value` line.

```bash
python evaluate.py ../models/soundex-v1.onnx \
  /datasets/musdb18_hq/processed/<version>/manifest.jsonl \
  /datasets/slakh2100/processed/<version>/manifest.jsonl \
  /datasets/medleydb/processed/<version>/manifest.jsonl \
  --output-directory ../models/soundex-v1-evaluation \
  --config configs/default.yaml \
  --parity-evidence ../models/soundex-v1.parity.json \
  --visqol-command 'visqol --reference_file {reference} --degraded_file {degraded} --use_speech_mode=false' \
  --listening-record ../models/soundex-v1-listening.md \
  --audit-files
```

The runner writes `evaluation-report.json` and `evaluation-report.md`. It exits 2
when any release gate fails. `--allow-failed-gates` is for diagnostic reports only;
it does not turn failures into a pass. If ViSQOLAudio is absent, unsupported, or
returns an unparseable score, the report records that state and the perceptual gate
fails. This environment does not silently replace ViSQOL with another metric.

## Metrics and uncertainty

Reports contain full/high/low-band LSD, spectral convergence, recovered high-band
energy error, SI-SDR, low-band enhanced-vs-degraded preservation, ViSQOLAudio,
stereo correlation/inter-channel phase/image width, gate call/false-bypass,
clipping/finiteness, and offline-vs-two-chunk-pattern maximum error. Aggregates are
reported overall and by corpus, codec, quality, sample rate, and channel role.
Evaluation report schema 2 computes deterministic 95% confidence intervals by
resampling whole `track_id` clusters, so correlated segments, codecs, sample rates,
and channel roles from one source track are not treated as independent evidence.

The versioned thresholds are in `release_gates.v1.yaml`. Synthetic good/bad reports
are CI tests only and must not be presented as model-quality evidence.

## Release binding

After the objective gates pass and the listening record is complete, verify the
filled model card against the actual files:

```bash
python -m evaluation.model_card \
  ../models/soundex-v1.onnx \
  ../models/soundex-v1.model-card.md \
  ../models/soundex-v1-evaluation/evaluation-report.json \
  ../models/soundex-v1.performance-apple.json \
  ../models/soundex-v1.performance-x86.json
```

The checker rejects instructional or blank fields, failed gates, pending listening,
diagnostic/short performance runs, missing Apple Silicon or x86_64 evidence, and
any mismatch among the ONNX hash, evaluation/performance reports, and model card.
For release, it ignores self-asserted `passed` flags: row and stereo aggregates are
recomputed with the canonical repository gate config, then every gate is rerun.
Parity evidence, the listening record, and test manifests must still exist at their
recorded paths and match their SHA-256 values; evaluated row IDs must equal the
bound manifests' test-row IDs. Alternate gate configs remain diagnostic only.
