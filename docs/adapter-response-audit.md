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
`training/evaluation/adapter_response.py`. Planned local run:
`data/runs/adapter-response-p51-20261024/`. Audio, moments, diagnostic coefficients,
source snapshots and raw reports remain ignored local artifacts.

Status: registered and running. The full Python suite passes 730 tests; Ruff and
English documentation checks pass. Tests cover identical and
separable features, singular coordinates, zero penalty, basis scaling, warmup
exclusion, hook cleanup, real candidate barriers and review receipt resumption.
Results and any failed criteria will be recorded after the full frozen review.
