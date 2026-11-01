# P68: phase gradients and cross-frequency observations

## Decision and scope

The frozen P67 audit completes all 64 endpoint/batch probes without changing any
weight. Native output support substantially improves agreement between the common
and fused objectives in phase-head parameter space. The residual problem is not
explained by widespread activation saturation or uniformly opposing objectives.
Paired/identity and source-specific conflicts remain. This does not prove that
the current objective is optimal, that Adam follows the raw gradient, or that more
steps would improve audio.

An optional adjacent-frequency phase observation and an equal-width repeated-input
control are implemented. Both preserve the frozen magnitude branch, original
lossless supervision, existing state timeline and zero lookahead. They are
untrained feasibility fixtures, not new restoration weights. Default weights and
the production model contract remain unchanged.

## Registered question and evidence

P39 studied the earlier full-model shared encoder. P65 tested temporal phase
observations, P66 identified output discarded by DSP, and P67 isolated output
support and fused supervision. P68 asks a narrower question at all eight frozen
P67 512-update endpoints: where do their phase derivatives go, and does the
current observation omit cross-frequency phase relations? It does not repeat
P39, extend the same calibration budget, or declare a phase-architecture ceiling.

The audit uses the first eight registered TRAIN draws at each endpoint: 32 music
and eight speech examples, each paired and identity. These are reused training
examples, not 64 independent recordings or a held-out test. Raw reports retain
every draw and source. Frozen evidence roots are:

- `data/runs/phase-gradients-p68-20261101/`: registration, frozen source, 64
  per-probe JSON reports, `assessment.json` and verified `complete.json`.
- `data/runs/cross-phase-feasibility-p68-20261101/`: separate source snapshot,
  nonzero fixtures, exports, native traces and the registered timing grid.
- `data/runs/phase-test-diagnostic-p68-20261101/assessment.json`: initial
  numerical causality-test failure and its frozen-parent/derivative controls.

The gradient experiment changes no objective or weights. It differentiates each
existing log-transformed direct, raw-MR and fused-MR term; paired/identity risk
weights remain one half each. Corpus/role vectors retain their existing sampling
weights rather than renormalizing each subgroup. It also differentiates the
output phase coordinates and checks their vector-Jacobian product back to the
phase parameters. The OLA guard frame remains included.

Maximum absolute reconstruction discrepancies across all probes are
`4.502e-11` for the three loss terms, `3.676e-11` for corpus/role decomposition,
and `7.276e-12` for the output-phase chain. All pass the pre-existing numerical
policy (`rtol=2e-4`, `atol=2e-7`), and every endpoint tensor remains bit-identical.

## What the gradients show

Each cell below is the median of eight batch cosines. Negative counts are shown
in parentheses; they diagnose local gradient disagreement, not audio quality.

| Origin / objective | Broad: common vs fused | Native support: common vs fused |
| --- | --- | --- |
| 20261012 / common | 0.3716 (1/8) | 0.8327 (0/8) |
| 20261012 / fused | 0.3317 (2/8) | 0.8257 (0/8) |
| 20261013 / common | 0.3376 (1/8) | 0.8805 (0/8) |
| 20261013 / fused | 0.3207 (1/8) | 0.8718 (0/8) |

Raw-MR versus fused-MR output-phase cosines in available HF bins have medians
0.876–0.887 and no negative batch in any endpoint. Native parameter-space
raw/fused medians are 0.900–0.933. Direct/fused parameter agreement is weaker:
native medians 0.220–0.305, with one or two negative batches per endpoint.
There is no evidence here for discarding the direct original-reference term.

For the native heads, median available-HF shares of hypothetical output-coordinate
gradient power are only 15.24–18.35% for direct and 7.89–8.37% for raw MR, versus
100% for fused MR. **These are not wasted parameter updates:** unavailable
coordinates are fixed by the model support and their gradients do not propagate
to the head. The verified parameter chain distinguishes this from P66's earlier
discarded output corrections. Do not use squared-coordinate shares as loss weights
or compare phase-radian norms against magnitude-dB norms.

Paired versus identity music parameter gradients remain mixed. Broad-head median
cosines range from -0.511 to -0.178; native-head medians range from -0.017 to
0.064, with 3–5 of eight batches negative. Native paired-music/paired-speech
medians range from 0.003 to 0.127. Some speech gradients are exactly zero because
the input support is inactive; their cosine is undefined and retained as null,
not counted as agreement or disagreement. The weighted native identity-music
gradient median norm is approximately `2.8e-5`–`3.0e-5`, below paired music's
`6.3e-5`–`8.5e-5`; this does not justify removing identity protection.

No hidden activation exceeds absolute 0.99. The largest absolute output tanh
value is 0.005118, far from saturation. Thus simply increasing the tanh range,
changing weights from pooled norms, or enlarging the head is not the next
supported intervention. These local observations do not prove convergence.

## Representation change and research context

The historical phase readout sees magnitude, carrier-relative phase and same-bin
temporal phase differences. Its correction is invariant to independent constant
phase rotations at each frequency bin. That is a concrete missing observation,
not proof that the entire model family cannot learn useful restoration.

The optional `adjacent` route adds two current-frame features. With
`z[k] = log1p(relative_magnitude[k]) * exp(i * input_phase[k])`, they are the real
and imaginary parts of `z[k] * conj(z[k-1]) / (1 + abs(z[k])*abs(z[k-1]))`.
The first-bin predecessor is zero. This avoids phase unwrapping and division by
weak magnitudes. It preserves invariance to a common phase rotation while
allowing sensitivity to relative phase between frequency bins. The alternative
`temporal_repeat` repeats the existing two temporal features, adding equal width
without new information. Repetition may still affect optimization conditioning.

Both routes have 204,418 total parameters, including 33,281 trainable phase
parameters: 8,256 more than P65/P67. State remains 12,354 values per physical
channel, with the existing 80-hop enhancement warmup. There is no extra lookahead,
history, output buffering, DSP teacher, imported model weight or external code.
Historical `temporal` remains the default and keeps its old export signature;
new route names enter the export cache signature and metadata.

The [AP-BWE paper](https://arxiv.org/html/2401.06387v2) studies parallel amplitude
and phase prediction with interaction between branches, including instantaneous
phase, group-delay and instantaneous-angular-frequency losses. Its
[official implementation](https://github.com/yxlu-0102/AP-BWE) provides a relevant
speech bandwidth-extension comparison. This motivates checking explicit phase
relations; it does not validate SoundEx's independently implemented feature,
music codec restoration, causality or a 5 ms per-hop compute limit. Whole-file
CPU/GPU throughput cannot establish our streaming deadline. No new phase loss is
introduced in P68.

## Validation and retained failures

The targeted 18 tests pass: zero-head equality, frozen magnitudes, state/chunk
equivalence, exact future/other-channel derivative exclusion, common-phase
invariance, newly observable cross-frequency rotations, silence/weak gradients
and gradient decomposition. Existing reference/finite-difference tests are retained.

The first test run had two failures because a new float32 causality assertion
required bitwise equality. Both the frozen parent and extended model differed by
`1.1920929e-7` on past frames after a future perturbation; the other channel was
exactly unchanged and future/other-channel input derivatives were exactly zero.
The corrected test uses the existing chunk-equivalence tolerance (`2e-5`) and
adds those exact derivative checks. This is a numerical test correction, not a
relaxed audio-quality or real-time gate. The evidence remains in the diagnostic
root above.

An initial command invoked the benchmark file without its package import root
and failed with `ModuleNotFoundError: evaluation` before preparation. Corrected
preparation uses `python -m scripts.bench_cross_phase`; execution uses the frozen
source with an explicit `PYTHONPATH`. No graph, optimizer update or timing case
was produced by the failed invocation.

The native feasibility screen registers four deterministic nonzero fixtures
(two origins times two observations), 44.1/48 kHz mono/stereo parity, and twenty
timing cases: 52,000 measured hops after warmup. Every timing is retained without
retry-to-pass. The binary is the frozen P66 runtime with current fusion, native
conditioning and worker observations. This is synthetic streaming evidence,
not device qualification, listening evidence or a trained-model quality score.

All four exports and sixteen native mono/stereo replays pass. The largest complex
prediction discrepancy is `7.689e-6` under the existing parity policy; silent
physical channels remain exactly zero. Adjacent-feature ONNX graphs are 941,407
bytes; equal-width repeated-feature graphs are 936,947 bytes.

All twenty timing cases complete, with no >=5 ms computation, no hop service
overrun, and no late completion among the 4,000 paced measured hops. Maximum
frontend + model + DSP computation is **1.464625 ms**; individual-case p99 values
range from 0.208044 to 1.259983 ms. This passes this fixture screen, not a universal
deadline guarantee; P63/P65 and earlier timing failures remain recorded.

In paced runs, process peak RSS is 31,490,048–31,637,504 bytes (30.03–30.17 MiB).
User plus system CPU time is 0.95–1.16 seconds during 2.97–2.98 seconds wall time,
roughly 32–39% of one core including process setup and warmup. Separate fresh
ORT-only processes increase warmed RSS by 14,876,672–15,155,200 bytes; those are
allocator/session snapshots, not the full Rust process peak. Fixed output
buffering remains 256 frames: 5.333 ms at 48 kHz or 5.805 ms at 44.1 kHz, excluded
from the user's compute limit. End-to-end device output latency is not measured.

## Next evidence required

Native correspondence and this compute screen permit registering matched training
of the new observation and repeated-input control under the same P67 fused
objective, seeds, data exposure and adequate fixed budget. Keep original-reference
spectra primary, review phase/PCM/dynamics/identity and actual Rust output, and
freeze all endpoints before comparison. Do not select early checkpoints, repeat
the unchanged P67 calibration, remove identity protection or claim improvement
from an untrained nonzero fixture. Native scheduling failures remain a separate
requirement even if all computation stays below 5 ms.
