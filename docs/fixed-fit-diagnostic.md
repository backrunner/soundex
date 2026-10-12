# P11: separate synthesis limits from shared-network fitting

This is a training-only fitting diagnostic, not a restoration release candidate.
The registered study fits the same five paired clips and five clean-identity
counterparts using free amplitude coefficients, free complex coefficients, or the
unchanged P10 shared network. Free coefficients are independently optimized for
each example using its reference. They are not predictors, cannot generalize, and
cannot be shipped as models. A small-set network fit also cannot establish quality
on unseen recordings.

## Frozen protocol

- Select the first four distinct music genres by sorted row ID from P10's existing
  train-domain probes, then the first speech probe. Selection never reads outcomes.
  The selected genres are classical, electronic, ambient and country, from Korea,
  North America and Europe, plus VCTK speech. This small subset does not represent
  the entire source library or regional coverage.
- Keep each original crop and its paired clean-identity counterpart. Do not assemble
  independently cropped channels into stereo. No development/test examples are fitted.
- All paths share the input-only detector, blend, smooth limiter, 80-hop warmup and
  fully supported scoring window beginning at hop 89. References remain original
  clean audio for restoration and the existing DSP output for identity preservation.
  Original-clean identity errors remain separately visible.
- Reuse P10's per-example rendered spectral/dynamics plus 0.25 waveform objective.
  Average paired and identity risks equally; average the five examples equally
  within each group. These totals are **not comparable to full-library P10 totals**,
  which use different corpus weights.
- Run amplitude and complex coefficients at learning rates `1e-4` and `1e-3`.
  Run the shared 807,398-parameter anchored network at the same two rates and seeds
  20261012/20261013. Free coefficients start at the same deterministic zero vector,
  so repeating seeds for them would be duplicate optimization, not replication.
- Eight runs, 256 full-batch steps each, Adam without decay, gradient norm capped
  at 5. Free amplitudes use projection onto nonnegative output amplitudes; both free
  representations express updates in input full-spectrum RMS units. The network
  retains P10's input-band-relative update coordinates. This intentionally separates
  attainable direct fitting from the current parameterization, not equal-capacity
  architectural performance.
- Select the minimum fixed-set objective at step zero, every 64 steps and the last
  step. Record the full measured curve and per-example losses. This is fitting-set
  selection, never held-out checkpoint selection. No claim of globally optimal
  free spectra follows from finite Adam runs.
- Replay the input baseline and all eight selected outputs through actual Rust DSP
  (90 mono-role clip replays). Require finite output and proxy/native sample error
  <=2e-5. Report raw and rendered high-band spectral convergence, energy, flux and
  low-band errors against the original references. Parity is not a quality gate.

## Implementation safeguards

`training/models/fit_diagnostic.py` contains free spectrum coordinates and a cached
objective. Finite-difference tests cover amplitude and complex gradients; tests
also cover silent/weak input, exact zero-coordinate identity, missing-bin complex
generation gradients, and equality to P10 per-example objectives.

The first two-step integration run exposed nonzero identity updates caused by
Cartesian/polar round-trip noise. Both free representations now subtract their
zero-coordinate mapping, preserving the same baseline exactly. The zero-complex
angle guard handles only exact zeros, preserving generation gradients in weak
bins. The first run also showed overshoot at `1e-2`, so the formal learning-rate
range was lowered to `1e-4`/`1e-3` before the registered 256-step experiment. The
original smoke evidence and exact sources are retained, not silently overwritten.

The corrected two-step integration run showed paired-only fitting gains at the
lower rate and zero identity damage for free coefficients. That is an integration
check; conclusions depend on the registered full run and native replays.

The full Python suite passed 416 tests before the final zero-complex edge guard;
the subsequent focused suite passed all 10 tests including the new missing-bin
case. Ruff lint and formatting checks passed. No production inference, DSP,
buffering, latency policy, published model or dataset licensing was changed.

## Evidence and interpretation

Local ignored artifacts:

- `data/runs/fit-p11-20261013/smoke/`: first integration run and archived source.
- `data/runs/fit-p11-20261013/smoke-corrected/`: corrected two-step integration run.
- `data/runs/fit-p11-20261013/source-commit.json` and `source/`: frozen formal source.
- `data/runs/fit-p11-20261013/study/config.json`: registered cases, source/binary/
  manifest hashes, objective and optimizers, written before fitting.
- Per-run `history.json`, `best.pth`, `native.json`, `complete.json`, plus final
  `summary.json`. Training audio and target-fitted coefficients remain local.

```sh
PYTHONPATH=training training/.venv/bin/python training/scripts/run_fit_diagnostic.py \
  --parent data/runs/rendered-p10-20261013/study/config.json \
  --output data/runs/fit-p11-20261013/study \
  --replay-binary target/debug/soundex-prediction-replay \
  --onnx data/runs/causal-p3-20261012/study/control-20261012/best.onnx
```

If direct amplitudes improve substantially, fixed phase and DSP alone cannot
explain the network's failure on these clips. Additional gains from free complex
spectra support investigating phase-aware reconstruction, but do not establish
that a small causal network can predict those coefficients. If the shared network
fits this subset, global representational impossibility becomes less plausible;
full-library optimization and generalization remain unresolved. If it cannot,
inspect the input-relative head and optimization before merely adding parameters.
A loss reduction must also be checked for per-example fidelity tradeoffs.


## Completed results

All eight registered runs and all 90 native replays completed. Initial score is
0.30649558. All comparisons below use the same fixed-set score; none are held-out
quality results. Best checkpoints are step 256 except complex `1e-3` (step 192).

| Representation | Learning rate | Seed | Best score | Reduction vs initial | Identity preservation error |
|---|---:|---:|---:|---:|---:|
| amplitude | 0.0001 | deterministic | 0.179141 | 41.55% | 0.000000 |
| amplitude | 0.001 | deterministic | 0.165654 | 45.95% | 0.000000 |
| complex | 0.0001 | deterministic | 0.146024 | 52.36% | 0.000000 |
| complex | 0.001 | deterministic | 0.151636 | 50.53% | 0.000000 |
| network | 0.0001 | 20261012 | 0.272360 | 11.14% | 0.005700 |
| network | 0.001 | 20261012 | 0.233003 | 23.98% | 0.004223 |
| network | 0.0001 | 20261013 | 0.271464 | 11.43% | 0.004311 |
| network | 0.001 | 20261013 | 0.234346 | 23.54% | 0.005094 |

Every selected run improves native high-band spectral convergence on each of the
five compressed clips. The maximum proxy/native sample discrepancy is 1.699e-6.
For the shared networks at `1e-3`, mean high-band SC changes versus input DSP are
-0.172717 / -0.170991; high-band energy errors change by -7.528955 / -7.637991 dB.
These are tiny fitted-set means, with large gains concentrated in particular
clips, not estimates of library-wide performance. Native identity high-band SC
still worsens by +0.004493 / +0.005534, and low-band errors have small regressions.
Thus even successful small-set fitting does not satisfy preservation acceptance.

Free identity coefficients remain exactly unchanged because they are independent
variables with a zero-loss starting point. This is expected and does not prove an
input-conditioned predictor can distinguish all clean from compressed material.

The findings narrow the next step:

1. Keeping input phase still permits about 42–46% fitting-score improvement through
   the current DSP. DSP restrictions or fixed phase alone cannot explain P10's
   near-zero gains on these cases.
2. Complex fitting adds room (about 51–52% total improvement), but the experiment
   does not show that a lightweight causal network can predict those complex
   coefficients on new audio.
3. All four shared-network runs learn useful corrections (11–24% score improvement),
   so a universal inability to fit these targets is disproved. This does not prove
   adequate capacity for all music. Higher learning rate helps this small set;
   simply copying it to full-library training is not justified.
4. Prioritize how corrections are learned as dataset diversity grows: a controlled
   subset-size/optimization study, keeping objective, sampling, initialization and
   evaluation fixed within each comparison, followed by a separate head-coordinate
   intervention. A synthesis-aware amplitude teacher is another testable training
   route; its target-fitted advantage must not be counted as student generalization.
   Phase-aware architecture remains a separate experiment, not a simultaneous change.

No new full-library continuation was started from these fitted weights. The model
size, runtime and default weights remain unchanged. No blind listening, independent
quality test or realtime deployment qualification is claimed. Local
`results-audit.json` records per-clip native metric changes and verified selected
checkpoint/native-report hashes.
