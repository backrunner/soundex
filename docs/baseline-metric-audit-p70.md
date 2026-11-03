# P70: baseline identity, metric calibration and frozen feature use

Completed 2026-11-04. No optimizer updates, replacement weights, new native
renders or timing measurements. Default artifacts remain unchanged.

## Finding

The premise that every newer model loses to the historical baseline is incorrect.
P69's adjacent-feature head loses to its **matched trained repeated-input control**,
while both seeds already improve native spectral error over historical
**shape-small**. Recent reporting emphasized the incremental comparison without
making the legacy comparison equally visible. This report corrects that ambiguity.

Independent spectral rulers reproduce the important distinctions. However, log
spectral distance and energy agreement have demonstrated blind spots. Neither
the historical model nor a favorable spectral mean establishes overall quality.
Recent phase-only work also freezes the model's predicted magnitude: that is a
concrete restriction on the restoration objective, not evidence that the metric
has reached a floor or that larger architectures cannot work.

## Changed question and frozen protocol

Consulted the [failure registry](failed-experiments.md), the historical
[baseline audit](baseline-generalization-20261012.md), and completed
[P68](phase-gradients-p68.md) / [P69](cross-phase-training-p69.md) evidence.
P54 studied short reused crops; P68 measured gradients before the full P69 budget.
P70 instead checks comparator identities, independent rulers on all twenty fixed
first-excerpt MP3 sources, and feature use at all four completed P69 endpoints.
It does not repeat training or retrospectively select a checkpoint or metric.

Local frozen evidence: `data/runs/baseline-metric-audit-p70-20261104/`.

- `registration.json` binds the prior receipts, sources, endpoints and executed
  source snapshot. The metric panel is every first 10-second MP3 CBR 64 kbps
  excerpt from the twenty exposed P55 sources, preserving physical channels.
- `metrics/*.json` retains every source and counterexample: eight saved native
  paths, untreated input and five analytical controls, totaling 280 scored paths.
- `features/*/*.json` retains all four endpoints and the first eight registered
  TRAIN draws per endpoint, totaling 32 frozen probes. These are fit diagnostics.
- `assessment.json`, `complete.json`, `summary.json`, `summary-source.py` and
  `summary-receipt.json` preserve results, derivation source and hashes.
- All completion/summary receipts verify. No audit execution failure or endpoint
  mutation occurred. Three analytic-control tests pass; all records are retained.

Sources are reused diagnostic recordings, not a new independent test. This narrow
MP3 calibration does not replace P69's complete multi-codec long-form review.

## Which baseline, and how simple?

| Comparator | Parameters | Meaning |
| --- | ---: | --- |
| Historical `legacy` / shape-small | 905,734 | Dual-stream frequency U-Net and spectral refiner; no temporal history |
| P63 zero-phase band parent | 171,137 | Frozen band-TCN magnitude model used by P69 |
| P69 repeated-input control | 204,418 | Same-width trained phase branch, repeated temporal observation |
| P69 adjacent-input candidate | 204,418 | Same budget and width, additional adjacent-frequency observation |
| Untreated / identity DSP | No restoration model | Compressed input / actual native DSP fidelity control |

The original shape-small ONNX SHA-256 is
`bdef9d1269ea4b8b1e8e5e04bd6d25032b817f5401e13c51e2c064d721503782`.
Its local `data/reports/architecture-study-20261011/shape-small/audit.json`
records the parameter count. P69's legacy wrapper metadata binds that same original
hash. The original graph contains 56 Conv, eight ConvTranspose and three MatMul
nodes, including amplitude/phase streams and a spectral refiner. It is framewise
in time but has substantial frequency modeling; it is not a trivial linear model.
Its parameter count is about 4.43 times P69's total, whose trainable phase branch
has only 33,281 parameters.

The historical chain inherits the original epoch-168 checkpoint and several
continuations, as recorded in the baseline audit. Comparing that chain to a fresh
small branch does not establish architecture superiority under equal budgets.
Parameter count and training history are plausible contributors, not a causal
ablation proving why legacy works.

### Existing complete P69 native results

These are source-equal means from the full twenty-source long-form recheck, not
the new MP3-only calibration. Lower is better; MR is the fixed multiresolution
spectral composite. PCM error is calibrated linear full-scale RMS.

| Path | MR | HF >=8 kHz MR | HF PCM RMS | Spectral dynamics error |
| --- | ---: | ---: | ---: | ---: |
| Untreated | 0.519219 | 0.580496 | 0.008725 | 0.264055 |
| Identity DSP | 0.517215 | 0.577754 | 0.008720 | 0.264003 |
| Legacy | 0.499814 | 0.552075 | 0.008959 | 0.268797 |
| P63 parent, seed 20261012 | 0.486455 | 0.534405 | 0.008889 | 0.266366 |
| P69 adjacent, seed 20261012 | 0.485695 | 0.533329 | 0.008899 | 0.266402 |
| P63 parent, seed 20261013 | 0.485186 | 0.532625 | 0.008913 | 0.266604 |
| P69 adjacent, seed 20261013 | 0.484544 | 0.531715 | 0.008923 | 0.266643 |

P69 adjacent reduces MR **2.824682% / 3.055135% versus legacy**. Codec-stratified
reductions are 2.704665% / 3.077300% for MP3, 1.777709% / 2.039865% for AAC and
4.398613% / 4.323074% for Vorbis. Most improvement is inherited from P63.
The incremental parent-relative gain is only 0.156115% / 0.132448%; against the
matched trained repeated-input head, MR is **0.004151% / 0.000784% worse**.
The source-isolated review also loses to that matched control. This remains the
registered P69 failure, regardless of its legacy-relative gain.

HF PCM and dynamics remain worse than untreated input. Means hide source
counterexamples; spectral gains do not establish listening quality or justify
default promotion. See P69 for every endpoint and source-specific failure.

## Is the evaluator rewarding the wrong thing?

All 160 saved native paths and twenty untreated scores reproduce exactly:
maximum absolute difference **0.0** across saved fields. The existing diagnostic
alignment is retained; no new offset is optimized. Its 128-sample alignment and
the deployment adapter's fixed 256-frame buffering are distinct quantities.
Reproduction checks consistency, not independent correctness.

For an independent check, NumPy computes HF spectral convergence (SC) and
log-magnitude RMSE using a fixed 2048/512 Hann analysis, >=8 kHz support and an
absolute `1e-7` per-bin amplitude floor. It also computes whole-excerpt Parseval
HF PCM RMS and energy error. This implementation does not call the training
spectral objective or inherit its band weights. These are diagnostic distances,
not perceptual estimators.

| MP3 calibration path | Existing MR | Independent HF SC | Independent log RMSE, dB | HF PCM RMS |
| --- | ---: | ---: | ---: | ---: |
| Untreated | 0.728854 | 0.813766 | 41.521853 | 0.010143 |
| Legacy | 0.677706 | 0.775950 | 16.921398 | 0.010541 |
| Adjacent, seed 20261012 | 0.661730 | 0.750254 | 16.994053 | 0.010551 |
| Repeated, seed 20261012 | 0.661673 | 0.750193 | 16.993313 | 0.010552 |
| Adjacent, seed 20261013 | 0.660389 | 0.749344 | 16.648155 | 0.010572 |
| Repeated, seed 20261013 | 0.660376 | 0.749337 | 16.647546 | 0.010572 |

Independent SC agrees that newer models beat legacy. Both independent SC and log
distance retain the small adjacent-versus-repeated disadvantage in both seeds.
Pure log distance does not favor the newer model over legacy in both seeds.
There is no observed evaluator implementation error reversing those comparisons;
this is not a proof that every metric is correct or sufficient for perception.

### Known-answer controls expose blind spots

Controls keep compressed audio below 8 kHz and manipulate the whole-excerpt
Fourier coefficients above it. They are noncausal, reference-informed probes,
not deployable models, feasible upper bounds or achievable optima.

- Reference-versus-reference errors are zero. Correct HF and half-correct HF
  improve MR, independent SC, log distance and HF PCM in **20/20** cases.
  Half-correct HF has half the untreated HF PCM error within registered tolerance;
  fully correct HF has near-zero error. Residual short-window SC after whole-FFT
  HF correction reflects leakage from remaining low-band error into analysis
  windows, not an exact-reference failure.
- Reference HF magnitudes with randomized phases retain whole-band energy but
  worsen HF PCM in **20/20**. Nevertheless, independent log distance improves
  versus untreated in **18/20**, with mean 16.640889 dB versus 41.521853 dB.
  Correct energy or better log distance alone can therefore reward an unsuitable
  reconstruction. This is an analytic fidelity counterexample, not a listening
  experiment proving perceptual severity.
- The existing MR rejects that scrambled control in 18/20; independent SC rejects
  it in 19/20. Both exceptions and all muted-HF scores remain recorded. No ruler
  is promoted as universally correct and no acceptance threshold is retuned.

The existing MR is already spectrum-first: >=8 kHz accounts for 3/4.25, or
70.59%, of its band weight; 0–4 kHz accounts for 5.88%. There is no waveform
distance term in this primary score. Low-frequency dominance or waveform-first
selection therefore does not explain the current ranking. Its `log_shape` term
is floored log-magnitude distance, not gain-invariant shape or a perceptual score.
Auraloss is useful but shares spectral assumptions and is not an independent
auditory oracle. Energy, magnitude shape, real dynamics, phase and listening
must remain separate evidence.

## Why recent changes have so little incremental effect

P65–P69 freeze predicted frame magnitudes and train phase only. In the direct
objective, amplitude, envelope and dynamics terms cannot change with phase when
magnitudes are fixed; the complex-spectrum term can. The weights are respectively
1, 0.1, 0.25 and 0.1 for amplitude, envelope, dynamics and complex error.
Phase can still change reconstructed waveform spectra through overlap-add, so
the restriction does not imply literally zero final spectral improvement.

P70 zeros only the added first-layer input columns `645:903` in copied frozen
heads, preserving every other tensor. Diagnostics exclude history warmup and
retain all four endpoints and eight draws each. Removing the adjacent features
changes output phase RMS by only 0.000331 / 0.000259 rad, compared with original
rotation RMS 0.032496 / 0.030494 rad: **1.018% / 0.848%**. These are RMS ratios,
not explained variance or universal feature importance. Added features have
nonzero hidden contributions; maximum output tanh is 0.170 / 0.188, providing no
evidence of saturation. Paired-music fused MR changes by about `1e-7` absolute
under removal, with mixed signs. The extra observation is present but weakly used
on these fit draws. This does not identify whether initialization, optimization
or parameterization causes the weak use.

Replacing predicted quantities with original-reference values on the same
available HF support gives the following reductions from the trained adjacent
head, averaged over paired-music fit examples:

| Reference-informed intervention | Fused MR reduction, seed 20261012 | Seed 20261013 |
| --- | ---: | ---: |
| Phase only | 6.0901% | 6.6717% |
| Magnitude only | 10.7552% | 10.3313% |
| Both | 31.5437% | 31.2903% |

These use model/raw reconstruction and the previously checked differentiable
fusion proxy, **not newly rendered Rust intervention outputs**. They demonstrate
score headroom and magnitude relevance; they are not learnable ceilings or
predicted training gains. Interactions are nonadditive. Original tensors remain
unchanged. Neither more unchanged phase-only training nor immediate scaling is
justified by these results alone.

## Leakage and generalization scope

The connected file-hash, decoded-PCM and work-alias isolation check passes for
all 446 distinct inherited TRAIN recordings against the twenty reserved P55
sources, including approved-catalog aliases. This is manifest coverage, not the
actual P69 sampling count (182 recordings), and not a complete reconstruction of
every historical exposure. No overlap is found within the audited scope.
Repeated development selection and historical budget differences remain.
The P55 panel has unknown regional labels and has been repeatedly reviewed;
neither artist independence nor general music coverage can be claimed. There is
no basis here to diagnose legacy overfitting, or to rule it out categorically.

## Industry evidence and next decision

[Apollo](https://arxiv.org/html/2409.08514v1) combines frequency-band and temporal
modeling with complex reconstruction, and reports SI-SNR, SDR and ViSQOL rather
than a single spectral score. Its 20 ms analysis / 10 ms hop and one-second RTF
measurement do not establish our per-hop CPU or buffering contract. This supports
investigating the reconstruction representation, not importing its architecture
or claiming it meets SoundEx latency. The
[official implementation](https://github.com/JusperLee/Apollo) is a research
reference; no data, weights or implementation were imported.

The music [DDSP bandwidth-extension study](https://link.springer.com/article/10.1186/s13636-023-00315-5)
evaluates LSD together with MUSHRA and shows a compact approach outperforming a
more complex ResNet. Its ideal bandwidth-extension setting differs from real
codec restoration. It supports testing model inductive bias and perception,
not assuming that greater complexity must win or that its result transfers here.

[ViSQOL's official guidance](https://github.com/google/visqol) requires 48 kHz for
audio mode and downmixes multichannel input. It could complement our spectral
panel, but cannot validate stereo preservation and has not been run in P70.
Human blinded ratings remain missing. Neither a new metric nor listening should
be introduced selectively after seeing which candidate wins.

Next, register a small matched joint amplitude/phase trainability experiment
only after checking gradient paths, feature utilization and output representation.
Retain the frozen parent and matched phase-only control, direct original-HF,
raw and fused reconstruction objectives, original references, and all preserved
band/identity/continuity checks. Validate actual Rust output and keep the <5 ms
model-plus-DSP compute target plus per-hop deadlines. Freeze the metric panel and
perceptual protocol before candidate scoring; use additional unexposed sources
for a genuine independent test. Do not switch metrics to obtain a win,
infer convergence from a fixed step count, or restart unchanged phase training.
P70 identifies a justified next hypothesis, not a completed new model or an
overall-quality improvement.
