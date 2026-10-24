# P51: clean/compressed adapter response separation

## Question and distinction from prior work

[P50](adapter-continuation-pilot.md) accumulates small paired spectral gains, but
also clean-input drift. The encoder is frozen and the finite step is calibrated.
P51 asks whether that encoder's affine features allow selective attenuation of
clean-input adapter responses while retaining the existing compressed-input
responses. This is narrower than learning a new restoration model.

P39 found shared-encoder gradient conflicts; P45/P46 projected update directions
using reference-preservation constraints. Neither answers this frozen-feature,
post-training response question. P51 includes an equally suppressed scalar
control: lower clean response alone is not evidence of better separation.

## Fixed fit-only intervention

- Use both P38 starts and all four frozen P50 persistent/reset endpoints. Keep
  the causal encoder, decoder, budget, high-frequency support, DSP and ABI intact.
- Collect actual affine-head inputs with the existing forward path and real
  80-frame causal warmup. Only direct-supervision frames 89..255 enter statistics;
  frames 80..88 and warmup do not. The OLA guard frame is a separate concept.
- Use every one of P50's 512 fit batches, with unchanged four-music/one-speech
  mixture and aligned identity counterparts. Reuse its globally recording/source-
  disjoint review role and source-time exclusions. This remains reused historical
  TRAIN, not an independent generalization test.
- Append the intercept to each latent feature vector. Accumulate float64 paired
  and clean second moments with equal frame/clip weights. Normalize each coordinate
  by its pooled RMS, with a fixed 1e-12 squared floor. Do not use review statistics.
- Let `V` be the original head in normalized feature coordinates `u`. Solve once
  for `V'` minimizing `E_pair||(V'-V)u||^2 + E_clean||V'u||^2 + 1e-6||V'-V||^2`.
  The identity penalty is fixed at one; no rank, penalty, checkpoint or spectrum
  score is selected using the review pool. The ridge retains the original head
  in unsupported directions. The original clean spectrum is not a solver target.
- Save the separated coefficients in deployment float32. Independently scale the
  original head by `sqrt(clean_power_separated / clean_power_original)` using
  only the same fit moments. This scalar control matches fit clean **logit** power,
  not PCM error, tanh output or restoration quality. Record the actual float32
  matching residual and normal-equation residual.
- Freeze all eight derived checkpoint hashes and fit statistics before any new
  review score or review-feature statistic. These are analytic diagnostic edits,
  not eight additional gradient-trained models. No optimizer state is fabricated.

When clean and compressed features are identical, this method must reduce to
roughly half-scale suppression; a synthetic regression checks that it cannot
claim selective preservation in that case. A complementary separable-feature
control verifies that it can preserve paired response better than scalar scaling.
Neither control establishes the limit of nonlinear architectures or longer training.

## Assessment and deployment

Recheck every derived head on P50's fixed 32 fit and 32 review batches against
its parent, the zero-adapter start and its matched scalar control. Reuse P50
control scores only after verifying unchanged input/source/checkpoint/report hashes.
Report original-reference magnitude/MR, envelope, complex, temporal, identity,
absolute PCM and all 16/20/24-bit rulers. Small absolute regressions alone do not
erase spectral gains, and suppressed enhancement alone is not recovery.

Record aligned paired/identity feature cosines and mean-square contrast, including
per-clip values. Cosine similarity is not a classification or information-theoretic
ceiling. Report per-band adapter response after tanh/support masks in input-normalized
units, without the input-dependent amplitude units, phase rotation or OLA. These
diagnostic response powers are explicitly **not PCM dBFS**; actual reconstruction
errors are measured separately in calibrated PCM units.

Export all eight derived heads with the existing experimental P48 ABI implementation
and verify nonzero sequential/state parity. Replay the same eight evenly spaced
review batches through the frozen Rust DSP: 640 new mono paired/identity roles.
Retain the seven hash-verified P50 native controls, including identity-through-DSP,
on identical source clips. This does not establish joint-stereo listening quality.

Each new head receives four 3,000-hop 44.1/48 kHz mono/stereo timing runs plus
1,000 paced 48 kHz stereo hops. Keep all failures; no timing retries to pass.
Compute, pacing, CPU/RSS and the unchanged 256-frame buffer are separate facts.
No model/worker/callback scheduling fix or release promotion is implied.

## Coverage, evidence and status

Source coverage is exactly P50's registered fit/review schedule: 351 fit recordings
and 986.623 unique MR-supervised seconds; 83 review recordings and 71.706 seconds.
The feature statistic itself uses direct frames 89..255 rather than claiming its
frames are extra independent audio. Fixed fit/review metadata gaps, the existing
20% speech diagnostic mixture and lack of simultaneous stereo evaluation remain.
See [P50 coverage](adapter-continuation-pilot.md#registered-exposure) for details.

Entry point: `training/scripts/audit_adapter_response.py`; solver and extraction:
`training/evaluation/adapter_response.py`. Completed local run:
`data/runs/adapter-response-p51-20261024/`. Audio, moments, diagnostic coefficients,
source snapshots and raw reports remain ignored local artifacts.

## Completed feature and reconstruction results

Both 512-batch feature collections, all eight frozen derived heads, their raw
reviews and 640 new native roles finish. The average aligned latent cosine is
0.8230 / 0.8083 on fit and 0.8174 / 0.8070 on review for seeds 12 / 13. Individual
clip means range from negative values to approximately one. These features are
neither identical everywhere nor evidence of a general classifier limit.

The four scalar factors, in seed-12 persistent/reset then seed-13 persistent/reset
order, are 0.355917, 0.370397, 0.296321 and 0.323049. After float32 conversion,
separated/scalar fit clean logit energy ratios differ from one by less than 4e-9.
The separated head's paired-logit distortion is 31.50-37.63% of scalar distortion:
it discards substantially less of the parent's compressed-input response at the
same fit clean logit power. This is a property of the proxy, not proof that the
retained response is correct audio.

The normalized systems have condition numbers 1.81e7 / 2.77e7. Float32 saved
coefficients have relative normal-equation residuals 7.15e-7 to 9.91e-7. The
calculation uses float64 and a fixed anchored ridge; no numerical threshold or
regularizer was changed after examining results. Deployment parity is checked
separately rather than assuming a small matrix residual implies audio parity.

### Original-reference raw music

The following relative MR changes use each seed's zero-adapter start. Negative
means improvement. Each raw role contains 128 paired music and 32 paired speech
crops, with clean counterparts. "Gain retained" compares the separated head's
music MR improvement with its P50 parent's improvement on the exact same clips.
"Added RMS reduced" concerns the **mean music identity RMS increase over the
start**, not total error or every individual recording.

| Parent | Separated music MR | Scalar music MR | Parent MR gain retained | Added music identity RMS reduced |
| --- | --- | --- | --- | --- |
| 12 / persistent | -0.006022% | -0.002651% | 78.75% | 54.75% |
| 12 / reset | -0.006889% | -0.003273% | 75.89% | 56.98% |
| 13 / persistent | -0.002541% | -0.000857% | 83.44% | 79.45% |
| 13 / reset | -0.003627% | -0.001447% | 77.43% | 70.94% |

All separated heads retain more music and speech MR benefit than their scalar
controls, but lose some music benefit relative to their P50 parents. Raw music
amplitude improves 0.006373-0.018872% versus the starts. Speech MR improves
0.001896-0.005305%. Complex music error remains 0.005475-0.009247% above the
start, although below the corresponding parent; music MR dynamics remains
0.001551-0.001822% above the start. All common totals are lower than the start
and matched scalar control, but higher than the P50 parent. This is a better
spectral/preservation tradeoff in several respects, not an all-metric winner.

Mean raw music identity full-PCM RMS is 7.5012e-6, 7.5024e-6, 8.7924e-6 and
8.7976e-6 in table order, about -102.50 to -101.11 dBFS. Increases over the
starts are 3.0193e-8, 3.1345e-8, 2.6343e-9 and 7.8709e-9. Identity HF RMS is
2.3113e-6 to 2.5530e-6 (-112.72 to -111.86 dBFS). Raw speech identity full RMS
is 4.5423e-7 to 7.8340e-7 (-126.85 to -122.12 dBFS), with small improvements
over the starts; its HF RMS is 8.8505e-8 to 1.6655e-7 (-141.06 to -135.57 dBFS).
Levels use `20*log10(mean clip RMS)`, not FFT power or pooled-recording RMS.
Small absolute errors do not establish audible restoration or erase counterexamples.

All eight derived heads pass the 16-bit reference checks and every fit/review
paired-low4k check at 20 bits. All four separated heads also pass paired-low4k
at 24 bits. The seed-13 reset scalar control retains one 24-bit fit-low4k peak
failure (draw 112, paired example 1, kazunocobit's "Rainy night and Dazzling RGB",
row `7f43cc47d3b2e74ec9bb17b9`, crop start 35,296): original-reference peak error increases
from 0.06152213694 to 0.06152221171, a 7.4768e-8 PCM change against the fixed
5.9605e-8 allowance. Its RMS error increases from 0.01101515948 to 0.01101516005
(5.7131e-10 PCM). This small absolute counterexample remains recorded without
threshold relaxation or an audibility claim. Identity failures persist; the
table uses 160 review identity examples per head:

| Parent | P50 20-bit / 24-bit | Separated 20-bit / 24-bit | Scalar 20-bit / 24-bit |
| --- | --- | --- | --- |
| 12 / persistent | 49 / 121 | 30 / 74 | 18 / 87 |
| 12 / reset | 54 / 107 | 32 / 71 | 24 / 82 |
| 13 / persistent | 16 / 85 | 5 / 39 | 0 / 31 |
| 13 / reset | 33 / 98 | 10 / 54 | 6 / 54 |

Separated-head fit identity failures at 20/24 bits are 24/80, 29/76, 5/55 and
14/62, also out of 160. All scalar and separated per-example reports remain.
No threshold is relaxed. Fewer failures than the parent is not a preservation
pass, and the scalar control is better on some fidelity measures.

### Why the proxy is not an equal-fidelity control

On review, separated/scalar music clean logit power ratios are 0.917-0.945;
speech ratios are 0.210-0.267. However, after tanh and frequency support, the
music clean response powers in individual high bands are **1.20-2.42 times**
the scalar control. Compressed music responses are roughly 4.94-8.59 times the
scalar control in those bands. These are normalized coefficient powers, not PCM
dBFS or proof that more generated energy is better.

The affine fit does exploit different clean/compressed responses, but equal total
logit power does not match frequency placement, physical input units, OLA or
original-reference PCM error. In particular, separated heads still have greater
mean music identity RMS error than their scalar controls. Consequently the result
does not prove superiority at identical audible or PCM preservation. This metric
mismatch is a concrete reason to align the next constraint with physical-band and
original-reference evidence, rather than declare the latent representation unusable.

## Native audio, compute and resources

All eight exports pass the existing sequential/state parity checks and produce
finite aligned Rust output. The unchanged ABI has 480,809 parameters and a
1,972,748-byte ONNX graph per head (about 1.88 MiB). Native music MR improvement
versus the same start is 0.006415%, 0.007392%, 0.002851% and 0.004165% for the
four separated heads. These retain roughly 68-76% of their parents' native gains
and improve MR 0.001856-0.003530% relative to scalar controls. Native speech MR
improves 0.004054-0.009141% versus the starts. Raw/native populations differ:
native has 32 music / eight speech crops per paired or identity role; compare
each candidate with its own same-population controls.

Against untreated native compressed music, the separated heads have 0.552-0.975%
lower MR, including the historical base model's benefit. MR dynamics is still
1.425-1.478% worse and HF PCM RMS error 2.490-2.525% worse. The retained
identity-through-DSP control continues to expose substantial existing safety/DSP
error on music; original references are not limited to disguise that error.
No listening or overall restoration pass follows from these spectral gains.

Across all 40 new timing runs and 104,000 measured hops (108,000 with warmup):

- Maximum compute **2.307583 ms**, largest per-run p99 **1.285475 ms**; zero
  >=5 ms calls and zero compute service overruns.
- **Five paced late hops**, maximum lateness **1.152206 ms**. One occurs in a
  separated head and four in scalar controls. Sequential run order does not
  establish a causal performance difference between the methods; historical
  scheduling failures remain open and no retry-to-pass was performed.
- Rust peak RSS **29.56-30.03 MiB**. Paced whole-process CPU/wall ratios range
  **8.8-28.4% of one core**, including loading and warmup. Isolated ORT warmed
  RSS increment is **13.34-13.58 MiB**, a distinct process measurement.
- Fixed buffering remains **256 frames**, or **5.333 ms at 48 kHz** and
  **5.805 ms at 44.1 kHz**, excluded from compute. Synthetic pacing is not
  physical-device callback, recovery, stereo-image or listening certification.

## Decision and retained evidence

The frozen features support a useful degree of response separation: the measured
tradeoff is better than simply scaling all adapter coefficients under the chosen
fit-logit matching rule. There is no basis here for claiming that clean/compressed
information is wholly absent, or that more parameters are the immediate remedy.
There is also no all-metric dominance: the separated head sacrifices some parent
spectral benefit and does not match the scalar control's actual preservation.

The next bounded training comparison should align this separation mechanism with
physical high-band residual units and original-reference reconstruction, retain
the unchanged spectral supervision, and include the existing small-step and
suppression controls. It must test whether gains can grow without renewed identity
drift; merely optimizing clean logit energy is insufficient. Any new differentiable
objective needs reference-correct, weak/silent-gradient and finite-difference checks.
No follow-up training, default-weight replacement or release promotion is launched
by this diagnostic. Worker scheduling remains a separate unresolved task.

The full suite passes 730 tests; Ruff, English documentation checks and remote CI
pass for the implementation. Tests cover identical/separable features, singular
coordinates, zero penalty, basis scaling, real warmup exclusion, hook cleanup,
candidate barriers and actual review receipt resumption. The registered run has
810 bound inputs and 4,882 completed artifact hashes, including all new candidates
and new/reused native artifacts. Completed-run re-entry verifies receipts without
refitting. `assessment.json`, `parent-responses.json`, `review-feature-geometry.json`,
`cost-summary.json`, `native-controls.json` and hash-bound `interpretation.json`
retain detailed comparisons. No execution failure occurred; remaining identity,
temporal, proxy-matching and paced scheduling failures are preserved.
