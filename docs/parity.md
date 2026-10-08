# FP32 export and runtime parity

SoundEx compares the same frozen learned parameters in PyTorch, Python CPU ORT
and the Rust ORT path. This verifies deployment equivalence, not restoration
quality. Non-finite values and tensor/metadata mismatches remain fatal.

## Why the original 1e-5 gate was replaced

The original gate pooled magnitude in dB and phase in radians into a single
absolute max/mean comparison. These channels have different units. At an
absolute dB value of about 200, one float32 ULP is approximately `1.526e-5`,
already larger than that gate. Equivalent phase angles near the wrap boundary
also need circular comparison.

PyTorch documents that floating-point evaluation order, platforms and backends
can produce different results even for mathematically identical operations:
[numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html).
ORT performs graph fusion and layout transformations:
[graph optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html).
Neither document supplies an audio acceptance threshold; the budgets below are
SoundEx engineering choices, checked against signal-domain error.

## Version 2 budgets

The exporter and Rust validation binary use the same
[policy file](../training/parity_policy.json). Every case must satisfy every
budget, using float64 only for measurements of float32 outputs:

| Metric | Limit |
|--------|-------|
| Magnitude error, max / mean | 0.001 / 0.0001 dB |
| Circular phase error, max / mean | 0.001 / 0.0001 radians |
| Complex spectrum relative RMS, worst independent spectrum | 0.0001 |
| Normalized real inverse FFT sample-error upper bound | 0.00002 |

A 0.001 dB amplitude change corresponds to about 0.0115% linear amplitude;
0.001 radians is about 0.0573 degrees. The complex relative RMS guard limits
aggregate signal error to a ratio of 1e-4 (80 dB), including magnitude/phase
interactions. Each batch item is checked independently. The inverse FFT guard
uses the triangle inequality: weighted complex-bin error sum divided by FFT
size, counting DC/Nyquist once and interior bins twice. This bounds every sample
of that inverse FFT; it is below one normalized 16-bit PCM quantization step.

These guards prevent a tiny dB error on a huge spectrum from producing a large
linear sample error, and prevent one strong phase-corrupted bin from hiding in
a mean. The bound is for the windowed frame before crossover, overlap-add,
loudness and limiting. Deployed quality, stream continuity and chunk-equivalence
are separate checks. No universal inaudibility guarantee is inferred.

The 13 deterministic cases cover silence, sine, random spectra, dB floor, phase
wrap, tones, quiet audio, noise and a center-window impulse at both 44.1/48 kHz;
batches 1 and 2 are represented. Reports use `pytorch-ort-rust-v2`, schema 2,
include the exact policy and its canonical JSON SHA-256, and record per-case Python/Rust metrics.
Release evaluation rejects old evidence, altered budgets, missing/duplicate
cases and out-of-budget metrics. Raw pooled errors remain diagnostic fields.

## Measured audit, 2026-10-08

[Full numerical record](parity-audit-2026-10-08.json) binds the checkpoint and
artifact hashes, CPU backend versions, all 13 cases and four ORT optimization
levels. Measurements used Apple Silicon CPU, PyTorch 2.12.1 and ORT 1.28.0,
with one intra/inter-op thread. A float64 PyTorch reference uses the same stored
parameters; it helps diagnose rounding and is not an independent audio truth.

For the default ORT optimization level:

| Worst result across cases | Measured |
|--------------------------|----------|
| Magnitude max / mean | 1.3733e-4 / 4.2050e-5 dB |
| Circular phase max / mean | 2.1029e-4 / 4.1229e-6 radians |
| Complex relative RMS | 5.0187e-5 |
| Inverse FFT sample-error upper bound | 7.1693e-6 |
| Actual inverse FFT peak sample difference | 2.6672e-6 |
| Lowest inverse FFT comparison SNR | 92.79 dB |

All cases pass version 2, including the artifact-bound Rust comparison. The old
strict gate failed on normal backend rounding; disabling ORT optimizations did
not remove the discrepancy. This confirms an inappropriate numerical gate for
this artifact, not a guarantee of its learned audio quality. The learned
candidate has not been audited on an x86 release reference machine.

Reproduce with your own authorized checkpoint from the repository root:

```bash
cd training
python scripts/audit_parity_numerics.py --checkpoint /path/to/checkpoint.pth \
  --output ../target/numerical-audit.json
python export_onnx.py --checkpoint /path/to/checkpoint.pth \
  --output ../target/candidate.onnx
python -m evaluation.parity --checkpoint /path/to/checkpoint.pth \
  --model ../target/candidate.onnx --output ../target/candidate.parity.json
```

The diagnostic audit never distributes checkpoint weights or dataset audio.
