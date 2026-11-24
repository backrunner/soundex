# P110 native Rust envelope mode: correspondence, quality and runtime

P110 ports the registered [P109](envelope-clean-preservation-p109.md)
deficit/identity candidate into the Rust stream and checks three things:
whether native output matches the Python proxy, whether native quality holds
against the native P74 parent, and whether it runs in real time. No default,
release or listening decision follows.

## Implementation (commit `a9296a3`)

- **`EnhancementMode::Envelope`** (opt-in). A native causal GRU
  (`crates/soundex-core/src/envelope.rs`) loads an experimental `.sxenv`
  artifact: magic, JSON header, SHA-256-checked f32 payload, in PyTorch tensor
  order. It steps on every hop for every physical channel, including bypass and
  silence. Enhancement is suppressed for the declared 32 warmup hops, and reset
  clears state.
- **Fill** (`crates/soundex-dsp/src/envelope.rs`). Half-ERB layout,
  current-frame features, and a hop-coherent copy-up above the existing
  detector's cutoff with deficit or level energy. All buffers are preallocated.
- **Unchanged.** No ONNX engine is used, and the production ONNX contract,
  default mode and weights are unchanged. `training/export_envelope.py` writes
  artifacts.
- **Tests.** Native/PyTorch GRU parity on a seeded probe fixture (max error
  < 2e-5), artifact tamper and geometry rejection, warmup identical to the dry
  path, silence, stereo isolation, reset determinism, and zero allocations in
  both the worker hop and the realtime callback. The full workspace passed
  (144 tests in a clean worktree).
- **Bench.** The realtime bench's 1 MB "trained model" heuristic rejected the
  0.88 MB artifact. Envelope mode now requires a valid `.sxenv` with at least
  100k parameters instead (commit `7a4de9a`, bench only).

## Registered rules

Registered in `data/runs/envelope-native-p110-20261124-v2/registration.json`
before rendering:
1. **Correspondence:** native recording means within 0.1 dB of the proxy;
   larger deviations are attributed before reading quality.
2. **Native quality:** in both seeds, missing-region ERB RMS below the native
   P74 parent (interval below zero), and clean preserved LSD not an increase.
3. **Runtime:** realtime bench at 44.1/48 kHz, mono/stereo; no retries.

## Invalid first measurement (v1)

Rule 1 failed in v1: ERB RMS differed by -0.46/-0.51 dB and clean LSD by
+0.93/+0.96 dB. On loud clean files, the native-minus-dry error was broadband
and largest below 4 kHz, outside the fill. Envelope renders peaked at 0.868
against 0.940 for identity-DSP and P74 renders of the same files. Cause:
`soundex-stream-eval` defaults to the legacy limiter, while
`soundex-context-replay` forces SmoothKnee, and context-replay produced every
P74, identity-DSP and P107 comparator render. v1 is retained with
`superseded.json`. v2 renders with SmoothKnee and compares against the proxy
passed through the same memoryless SmoothKnee curve.

## Results (v2)

**Rule 1: met.** Native and limited-proxy recording means agree to 0.000 dB on
every ruler in both roles and both seeds. Waveform error-to-signal is
-130 dB median and -87 dB worst across all 64 cases: float32 rounding.

Recording-equal means (15 paired recordings; 16 clean works):

| Path | Missing LSD | ERB RMS | ERB bias | Temporal corr | Paired preserved LSD | Clean preserved LSD | Clean complex rel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Untreated | 29.67 | 35.27 | -32.19 | 0.369 | 7.54 | 0.00 | 0.000 |
| Identity through DSP | 24.49 | 29.96 | -26.23 | 0.421 | 7.45 | 0.57 | 0.004 |
| P74 parent (native) | 14.57 | 14.44 | -9.60 | 0.569 | 7.15 | 0.70 | 0.004 |
| **Envelope native, seed 1** | **13.11** | **12.57** | -7.73 | 0.578 | 7.41 | 1.61 | 0.004 |
| **Envelope native, seed 2** | **13.05** | **12.53** | -8.08 | 0.577 | 7.35 | 1.08 | 0.004 |
| Unlimited proxy, seed 1 / 2 | 13.36 / 13.32 | 12.92 / 12.89 | -8.28 / -8.64 | 0.575 / 0.576 | 7.47 / 7.41 | 1.11 / 0.56 | 0.000 |

Native envelope minus native P74 parent (95% recording-bootstrap interval):

| Seed | ERB RMS dB | Missing LSD dB | ERB bias dB | Paired preserved LSD | Clean preserved LSD |
| --- | --- | --- | --- | --- | --- |
| 1 | -1.87 [-3.05, -0.79] | -1.46 [-2.16, -0.81] | +1.87 [+0.29, +3.59] | +0.27 [+0.04, +0.57] | +0.90 [+0.34, +1.57] |
| 2 | -1.91 [-3.05, -0.83] | -1.52 [-2.21, -0.84] | +1.53 [-0.18, +3.23] | +0.20 [+0.03, +0.44] | +0.38 [+0.15, +0.68] |

Per codec, ERB RMS falls from 14.72 to 12.57/12.08 dB (AAC 96), 14.73 to
13.33/13.19 dB (MP3 64) and 13.85 to 11.80/12.31 dB (Vorbis q3).

**Rule 2: restoration met, clean preservation failed.** Both seeds improve
missing-band envelope and LSD over the native P74 parent. On clean input,
preserved LSD increases in both seeds, with intervals entirely above zero; 15 of
16 works are adverse. Relative complex error is equal to P74 (0.004), so the
cost lies in weak high bins. The SmoothKnee limiter adds about +0.5 dB of it
against the unlimited proxy: added high band pushes loud clean material into
the knee. The ported detector stays active on most clean frames (P108).

**Rule 3: compute met, availability failed.** Settings: realtime bench, 60 s per
case, SmoothKnee, worker realtime requested, producer realtime not requested.

| Case | Worker max / p99 | Processing overruns | Callback p99 / max | Late producer hops | Worker misses (burst) | 8 ms target |
| --- | --- | ---: | --- | ---: | --- | --- |
| 44.1 kHz mono | 76 / <=75 us | 0 | 5.3 us / 33.1 ms | 970 | 9.01% (394.7 ms) | no |
| 44.1 kHz stereo | 488 / <=475 us | 0 | 6.7 us / 0.26 ms | 288 | 2.89% (14.5 ms) | yes |
| 48 kHz mono | 249 / <=249 us | 0 | 5.5 us / 0.06 ms | 3 | 0.04% (10.7 ms) | yes |
| 48 kHz stereo | 525 / <=475 us | 0 | 9.3 us / 9.4 ms | 101 | 0.66% (72.0 ms) | no |

Model plus fill stays at least 5x below the hop period (P74 peaked at 2.45 ms
compute). The misses come with late producer hops, worker queue waits up to
2.4 ms and multi-millisecond callback stalls: host scheduling of the simulated
producer, not processing time. The GRU's inner products are not yet
SIMD-friendly (sequential f32 sums), so compute headroom remains. Availability
is a recorded runtime failure; no rerun was made to obtain a pass.

## Interpretation

The native port is exact, so the P108-P109 proxy results now stand for the Rust
stream. A 219k-parameter, sub-millisecond model beats the preserved P74 parent
by about 1.9 dB missing-band envelope error and 1.5 dB LSD in every codec. Its
remaining failure is clean-input preservation: through the real limiter it adds
measurable weak high band to lossless input. The candidate stays experimental,
and the P74 parent stays the preserved native reference for clean preservation.

## Next steps

- **Clean preservation.** A registered study of the bypass decision is the
  direct lever: the existing detector rarely bypasses real music. Candidates are
  a learned missing-band gate trained on identity pairs and a limiter-aware fill
  cap. Thresholds must be prospective, not fitted to these works.
- **Runtime.** Vectorize the GRU inner products, then measure availability with
  a realtime-scheduled producer as a separately registered configuration.
- **Listening.** Build a blinded pack from the v2 native renders (P74,
  envelope, untreated, reference).

## Evidence

- `data/runs/envelope-native-p110-20261124-v2/`: `registration.json`,
  `artifacts/*.sxenv`, `renders/`, `records.jsonl`, `summary.json`,
  `contrasts.json`, `correspondence.json`, `runtime/realtime-report.json`.
- Invalid v1: `data/runs/envelope-native-p110-20261124/` with `superseded.json`.
- Binary: `data/runs/envelope-native-p110-assets-20261124/` (commit `a9296a3`).
- Code: `training/scripts/run_envelope_native_p110.py`.
