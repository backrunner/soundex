# Stage 1: Oracle, Explicit State Backbone, and Few-Shot Learning

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

2026-10-11, implementation baseline `64bcbbc`. This is the P0/P1 experiment record for the [Architecture Reset Plan](architecture-reset-20261011.md). The P2 unified training has since been initiated on October 12. Two novel backbones and six few-shot training runs have been completed; P2 full corpus, old backbone control, and dual-seed budgeted training remain pending. No new recovery weights have been released, nor has the default strategy been replaced.

## 1. First, Verify That DSP Does Not Corrupt the Correct Answer

A new `soundex-oracle` module has been introduced, which only replaces model output while preserving the production Rust input detection, gain, phase fusion, cross-band, OLA, gate, and limiter. Eight sets of comparisons are observed: clean, clean magnitude, clean phase, clean complex × MatchEdge / Model gain. The "clean" version serves only as an offline reference and does not participate in input detection or gating—it is not deployable.

The old `shape-small/best.onnx` is fixed:
`bdef9d1269ea4b8b1e8e5e04bd6d25032b817f5401e13c51e2c064d721503782`.
The same 89-line development selection is applied, aggregated by recording and bootstrap; data already used is not considered new independent test.

Below are the actual Rust outputs for the full piece, not training loss. HFSC/energy/flux values should be minimized. Lower is better.

| Clean complex oracle | Gain | HFSC | High-Frequency Absolute Energy Error (dB) | Flux Error | SI-SDR dB |
|---|---|---:|---:|---:|---:|
| Music                | MatchEdge | 0.61503 | 2.33333 | 0.91253 | 27.20916 |
| Music                | Model | 0.38059 | 0.77544 | 0.70378 | 27.87846 |
| Speech               | MatchEdge | 0.44475 | 2.13039 | 0.52911 | 29.71703 |
| Speech               | Model | 0.32582 | 0.90755 | 0.41828 | 29.89717 |

For music, when only clean magnitude is retained, and compressed input phase is preserved with Model gain, HFSC drops to 0.61204, energy error rises to 2.48035 dB. The correct magnitude remains insufficient—phase and actual reconstruction path must be optimized jointly.
The non-zero Oracle value is as expected: low-frequency content preserved comes from the compressed input, and detectors, cross-band, and gating remain active.
These results cannot be used as a proxy for new model quality, nor should Model gain be applied to all legacy models.

**Decision: The new backbone adopts Model gain + Model phase. P2 control must adopt the same strategy and target.**
The old default remains MatchEdge. Learned gain is retained for future independent ablation.

## 2. Complex Backbone Has Been Realized

Input consists of magnitude-scaled compressed Re/Im, log magnitude, and native sample rate. After shared encoding, it passes through a true time-domain backbone. Output is additive complex residuals, allowing insertion of high-frequency content near zero; DC and Nyquist imaginary components are fixed to zero.
No external model source code, weights, audio, or new dependencies have been introduced. Full interface details are in the [State Contract](causal-state-contract.md).

| Prototype | Parameters | Untrained ONNX Bytes | Per Channel State (floats) |
|---|---:|---:|---:|
| Two-layer GRU, width 192 | 643458 | 2590696 | 384 |
| Six-layer TCN buffer, width 144 | 639906 | 2592496 | 9072 |

Both backbones total ~2.47 MiB, below the ≤2M / <8M byte budget. This is not the simplified paper-based structure from the prior document.
No long windows, no GAN, no learnable gain—this phase focuses on isolating backbone feasibility.

The first 64-hop history is advanced and output-aligned in dry mode. Silence/bypass updates all physical channel states; reset, sequence gaps, and reload clear the history. ORT prewarm does not count as real audio history.
A new counter distinguishes "output received on time" from "still in state preheat": the latter may produce normal output packets without enhancement.

Validation coverage includes frame-by-frame, sequence, and irregular chunk alignment; no future dependencies; batch and independent channel support; phase equivalence modulo 2π; silence gradient bounded; 64-hop dry start; reset/reload; real worker frame loss equals fresh instance; and callback/fixed hop Rust heap zeroing. C/C++ internal allocation is measured separately for runtime cost.

For 96-hop × 44.1/48 kHz × mono/stereo × two backbones, including silent-left and 75-hop reset:
PyTorch/ORT/Rust passed pre-registration Cartesian `atol=1e-5, rtol=1e-4`; state `atol=2e-5, rtol=2e-5`. Maximum Cartesian absolute difference: GRU 1.573e-5, TCN 3.563e-5—absolute differences must be interpreted with relative tolerance. Actual Rust full segment and irregular chunk differences are zero.
The exporter runs a state and spectral validation for 80-hop × two sample rates × two batches before release; failures are retained in original files.

## 3. Six Few-Shot Diagnoses—Not a Generalization Benchmark

Three existing **train** clips are fixed: two Chris Zabriskie music pieces (44.1/48 kHz), one VCTK speech (48 kHz), each 24,576 samples. High-frequency activity is selected, and source and full file checksums are logged.
This is not a genre coverage set; balanced across three samples only for gradient and representation diagnostics—**it does not alter the future full corpus sampling ratio**.
The full corpus must still follow music 253/254, speech 1/254.

Each arm is randomly initialized, seed `20261011`, AdamW lr `3e-4`, 256 updates. 64-hop no-gradient burn-in + 128-hop supervised (≈0.34–0.37 seconds). Uses existing **V5** weights and scale, FFT256/H128, steady-state waveform, GAN=0, gain set to Model; V7 MatchEdge loss is not silently replaced with another target.
Actual cutoff trajectories are extracted from Rust for each damaged input, not directly using manifest cutoffs during training.

The differentiable proxy includes band fusion, phase, and OLA; no sample gate or limiter is included. The V5 spectral convergence term uses time-averaged masking. Actual Rust audio is separately scored and cannot be considered equivalent to the full deployment chain.
Ideal low-pass is used only for offline diagnosis of damage generation and does not enter real-time processing. Three tasks are trained separately, not mixed.

Pre-registration thresholds: final total loss ≤ 60% of initial; non-identity must have raw HFSC ≤ 80% of initial, and gradients must be finite. This is a learnability filter, not a quality promotion threshold.

| Task | Backbone | Loss Initial → Final | Raw HFSC Initial → Final |
|---|---|---:|---:|
| clean identity | GRU | 1.20234 → 0.00897 | 0.32530 → 0.00259 |
| clean identity | TCN | 2.05912 → 0.00889 | 0.63808 → 0.00251 |
| Ideal 8 kHz Lowpass | GRU | 2.90070 → 1.45915 | 0.78836 → 0.41865 |
| Ideal 8 kHz Lowpass | TCN | 3.26906 → 1.53075 | 0.94414 → 0.44895 |
| Real MP3 64k | GRU | 2.50053 → 0.97551 | 0.75666 → 0.23964 |
| Real MP3 64k | TCN | 2.90624 → 0.99925 | 0.92424 → 0.25213 |

All six arms pass the threshold. Actual CPU time for training/export/scoring per arm is ~23–36 seconds. No background training remains after completion.
This confirms that representation and gradients can be learned effectively, but does not imply convergence after 256 updates, nor that any one model generalizes superiorly.

Actual Rust evaluation of two known MP3 music pieces:
Original input flux: 0.92782 / 0.99935
GRU: 0.83654 / 0.64398
TCN: 0.85893 / 0.65998
GRU SI-SDR relative to input: -0.00043 / -0.02901 dB (music), -0.30248 dB (speech).
The ideal lowpass music still loses 1.9–4.0 dB SI-SDR, and the first piece shows degradation.
**We cannot claim full recovery or surpassing baseline performance based on this.**
Identity Rust output SI-SDR is ~88.7–98.1 dB; the delta relative to a 120 dB upper limit on original audio should not be interpreted as equivalent perceptual degradation in a typical lossy input.
No blind listening was conducted during this phase.

## 4. Initial Runtime Cost and Failure Logs

M5 Max, ORT single-thread, FFT256/H128, Model gain, untrained weights. Initial core per configuration took 3 seconds for diagnosis—fails the official 30-minute stress test threshold: GRU stereo p99 0.128–0.222 ms; TCN 0.260–0.743 ms.
GRU peak RSS: 36.03 MiB, TCN: 38.16 MiB.
TCN at 48 kHz exhibits 14 instances of hop-period spikes (maximum 10.501 ms), which cannot be dismissed by citing only p99. Original failure logs are preserved.

First run with 30 seconds per configuration, producer at normal priority, worker with audio time constraint:
Both backbones achieve worker p99 upper bound <1.5 ms, but GRU had a miss ratio >0.1%, and both backbones had callback max exceeding 500 us in some configurations.
Limited value, length, and boundary checks passed, but this does not equate to actual absence of pops.
Subsequently, only the simulated producer was set to audio time constraint, and sequential retesting was performed, with process CPU/RSS recorded.
This setup is used to isolate scheduling effects and does not replace physical sound card callbacks—nor does it erase the first failure.

Complete retest values are listed in the table below. Output latency is always 256 samples (44.1kHz: 5.805 ms, 48kHz: 5.333 ms); any 64-hop preheat must be reported as temporarily inactive and not extended to playback latency.

## 5. Reproducibility and Next-Phase Thresholds

All local evidence is preserved in the `data/reports/architecture-reset-20261011/` directory:
`oracle.json`, `probe/probe.json`, `probe/{gru,tcn}/cost.json`, `probe/{gru,tcn}/realtime.json`, `probe/rt-repeat-resource.json`, `tiny/report.json`.
Code and source documentation are submitted; original audio, experiment reports, and six tiny weights are not pushed.

To run from the training directory:
`PYTHONPATH=. .venv/bin/python scripts/probe_causal_core.py --help`
or `scripts/fit_causal_tiny.py --help`.
First compile release versions of `soundex-state-parity`, `soundex-stream-eval`, `soundex-gain-data`; use `soundex-oracle` for the oracle.
These scripts are probes and few-shot diagnostics—**they are not the full P2 training entry point**.

P2 must still unify the target between old and new backbones, warmup/supervised regions, sampling, optimization search, and 4096 updates ×2 seeds per arm, validate proxy gate/ reconstruction differences, and compare actual Rust output against compressed direct-through and legacy model baselines.
Only after this can longer history, texture, and learnable DSP ablation be considered.
Existing development data must not be upgraded to new independent tests; new testing, blind listening, long-term state, physical hardware, and x86_64 validation remain incomplete.

Local check results: 285 Python tests passed; Rust workspace tests passed (one 30-minute test ignored by default); Clippy `-D warnings`, no ORT backend build, and Python lint/format all passed. Remote CI records separately.

## Controlled Retest Results

Each configuration at 30 seconds, audio time constraint requests from producer/worker are fully accepted; no parallel training or compilation.
Synthetic signals include tone/noise/silence/transient—these are not audio quality benchmarks.
CPU is measured as user+sys/wall time from `/usr/bin/time -l`, normalized per one CPU core at 100%, including simulated producer, worker, ORT, and reporting overhead; RSS is peak memory of the independent process.

| Backbone | Hz / ch | Worker p99 Upper Bound (ms) | callback max us | Software Overhead (ms) | CPU % (Single Core) | RSS MiB | State Preheat Hops |
|---|---|---:|---:|---:|---:|---:|---:|
| gru | 44100 / 1 | 0.225 | 8.583 | 5.814 | 5.48 | 31.91 | 64 |
| gru | 44100 / 2 | 0.550 | 14.250 | 5.819 | 14.09 | 31.72 | 64 |
| gru | 48000 / 1 | 0.350 | 21.125 | 5.354 | 8.59 | 31.75 | 64 |
| gru | 48000 / 2 | 0.500 | 13.417 | 5.347 | 11.26 | 31.42 | 64 |
| tcn | 44100 / 1 | 0.475 | 12.666 | 5.818 | 11.49 | 33.30 | 64 |
| tcn | 44100 / 2 | 0.875 | 20.167 | 5.825 | 23.18 | 33.61 | 64 |
| tcn | 48000 / 1 | 0.550 | 23.916 | 5.357 | 13.39 | 33.41 | 64 |
| tcn | 48000 / 2 | 0.900 | 21.250 | 5.355 | 25.50 | 33.61 | 64 |

All eight configurations achieve zero deadline misses, queue overflows, or worker processing overruns. State preheat occurs only in the first 64 hops.
This represents a short-term feasibility result. The initial failure of a normal producer schedule remains valid and cannot be extrapolated to all host threads, physical devices, long-term state, or post-training weights.
P1 cost profile is acceptable for next training design, but formal real-time thresholds remain unmet.

The subsequent [P2 matched training](causal-training-20261012.md) started on 2026-10-12; seed 20261012 belongs to that subsequent study. this P0/P1 report does not claim its later full-library qualification. Each tiny-set diagnostic excerpt contains 24576 samples.
