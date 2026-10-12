# P10: supervise the rendered restoration, preserve clean input

Status: the audit, pilot and all six 2048-step runs are complete. Export/cost and
2,136 best-checkpoint native DSP replays also completed. Every selected checkpoint
is step zero; those native replays therefore assess the initial function, not
trained restoration. No candidate qualifies for promotion.
The default weights and runtime DSP are unchanged.

## What the P9 diagnosis actually found

`training/scripts/audit_restoration_conflict.py` inspected 436 cases and 80
paired/identity gradient pairs from the frozen P9 anchored checkpoints. The
training probes and reused development recordings are diagnostic, not new tests.
The following are recording-balanced music means; the two values are seeds
20261012 / 20261013:

| Diagnostic | Seed 12 | Seed 13 |
|---|---:|---:|
| Raw reconstruction loss change vs input | -0.010293 | -0.009170 |
| Applied-region normalized squared error change | -0.007863 | -0.006143 |
| Discarded-region normalized squared error change | -0.003557 | -0.003424 |
| Rendered spectral error, input through DSP | 0.576460 | 0.576460 |
| Rendered spectral error, candidate with input phase | 0.585766 | 0.588117 |
| Clean-identity spectral error, input through DSP | 0.016621 | 0.016621 |
| Clean-identity spectral error, candidate with input phase | 0.055148 | 0.057517 |
| Negative paired/identity gradient cosines | 18 / 40 | 16 / 40 |

Applied/discarded terms partition squared error with the analyzer blend mask and
activity flag; they are not the smoothed norm reported as raw loss. Raw error
improves even where the DSP uses the correction. Rendered error regresses on the
training probes too. Discarded corrections and generalization alone therefore
cannot explain the failure. Gradient conflict exists in some samples, but the
median cosine is positive for both seeds; it is not a universal explanation.

An oracle using clean amplitudes with input phase reaches 0.416155 rendered
spectral error, but waveform relative RMS worsens from 0.085954 to 0.087036.
The clean-complex oracle reaches 0.234078 spectral error and 0.079641 waveform
relative RMS. These are target-informed diagnostics, never inference paths or
achievable quality promises. They motivate supervision of actual synthesis and
separate phase investigation rather than rewarding raw per-frame magnitude alone.

Local evidence: `data/runs/shape-p9-20261012/restoration-conflict.json`; its exact
executed script is retained beside it as `restoration-conflict.py`.

## Implemented objective and matched experiment

All arms use the same P9 anchored band model, 807,398 parameters, initialization,
past-only context, draws, AdamW schedule and 80-hop warmup. All primary DSP
comparisons use input phase. P3 phase is an additional diagnostic for the candidate.
Identity labels select training losses only and never enter inference.

| Arm | Optimized target |
|---|---|
| `raw_control` | Existing P8 reference-normalized amplitude and signed dynamics |
| `rendered_control` | Rendered restoration; identity preservation weight 1 |
| `rendered_balanced` | Same rendered objective; identity preservation weight 7 |

For paired compressed examples, the rendered objective is the P8 band-normalized
spectral/dynamics objective after DSP synthesis and FFT reanalysis, plus 0.25 times
a smooth relative waveform error against the original clean waveform. Low-band
fidelity and reference-relative temporal differences remain supervised. The old
raw loss is reported separately and is not added to this optimized sum.

For identity examples, the preservation target is the same input passed through
the existing DSP, including its smooth limiter. This asks the model to add no
extra damage, rather than compensate for the limiter. Original-clean spectral and
waveform errors are still reported independently; the DSP output is never relabeled
as the original reference. Thus zero preservation loss does not mean perfect audio.

The fixed training draws contain 1/8 identity examples. Weight 7 gives equal
expected paired/identity risk; normalization is the fixed expected weight
`0.875 + 0.125 * identity_weight`, not a per-batch renormalization. All three arms
are selected with the common weight-7 objective, including step zero. New totals
cannot be compared directly with P9 totals. Validation uses the existing music /
speech weighting and 0.875 paired / 0.125 identity weighting; applying the same
objective gives the intended equal expected risks.

CPU FFT/DSP operations remain autograd-connected to MPS predictions. This changes
training, not inference compute or buffering. Reanalysis adds no extra left pad;
scoring starts at hop 89, after common warmup and support margins, and excludes
the unsupported tail. Finite differences, CPU/MPS gradient agreement, silent /
weak signals, exact identity preservation and nonzero ONNX exports are tested.

## Pilot results and decision

The pilot uses seed 20261012, all 89 development clips plus identity counterparts,
three fresh 128-step runs and validation every 64 steps. Its cosine schedule ends
at 128 steps; this is not the first 128 steps of a 2048-step schedule. Source hashes
and exact Python sources are retained under each run. These are clip-averaged
music metrics, unlike the recording-balanced audit above:

| Common proxy metric, lower is better | Initial | Raw control at 128 | Rendered control at 128 | Balanced at 128 |
|---|---:|---:|---:|---:|
| Common selection score | 0.293735 | 0.350684 | 0.312146 | 0.298539 |
| Paired music rendered spectral/dynamics | 0.567836 | 0.610447 | 0.573867 | 0.566986 |
| Identity music extra DSP damage | 0 | 0.071730 | 0.030940 | 0.010496 |
| Identity music original-clean spectral/dynamics | 0.014653 | 0.084712 | 0.043965 | 0.024370 |

Balanced supervision reduces extra identity damage by about 85% against the raw
control and gives a small paired-music spectral improvement (~0.15%) over initial.
However, its common score is still worse than initial, and all selected checkpoints
are step zero. This is not a restoration-quality win. Balanced score decreases
from 0.302546 at step 64 to 0.298539 at 128, while identity damage also decreases;
this supports a bounded two-seed longer experiment, not unlimited continuation or
relaxed acceptance. No thresholds were tuned per recording.

The two-step smoke completed 24 actual Rust DSP replays, with maximum proxy/native
sample error 6.706e-7. This proves numerical correspondence on those cases, not
quality. Its benchmark overlapped another process, so its timing is not used as
clean performance evidence. The formal pipeline runs cost measurement serially.
The 128-step latest checkpoints completed 1,068 native replays, maximum sample
error 2.6674e-6. This nonzero diagnostic never changes best-checkpoint selection.
Against input through DSP, recording-balanced paired music high-band SC changes
by -0.004680 (95% CI [-0.011655, +0.000356]), flux by -0.002452 and energy error by
-0.152373 dB; all three intervals include zero. Identity music SC worsens by
+0.011109 (CI [+0.004185, +0.020268]), flux by +0.015226 and energy error by
+0.044863 dB. Paired speech energy error also worsens by +0.005633 dB. These
remaining failures are retained as explicit gates, not hidden by the mean score.

Validation: the full Python suite passed 401 tests before adding six weak/silent
cases; the subsequent focused objective/export suite passed all 19 tests, including
those six. Ruff lint/format checks passed. No Rust implementation was changed.

## Evidence and reproducibility

Local artifacts (ignored; no raw audio or experimental weights are pushed):

- `data/runs/rendered-p10-20261013/smoke/`: two-step integration and native replay.
- `data/runs/rendered-p10-20261013/pilot/`: 128-step results and native latest replay.
- `data/runs/rendered-p10-20261013/study/`: registered two-seed 2048-step study.
- `data/runs/rendered-p10-20261013/pipeline.json` and `pipeline.log`: stage and log.

```sh
PYTHONPATH=training training/.venv/bin/python training/scripts/run_rendered_study.py \
  --parent data/runs/shape-p9-20261012/study/config.json \
  --output data/runs/rendered-p10-20261013/study \
  --cache data/runs/factor-p6-20261012/study/cache
PYTHONPATH=training training/.venv/bin/python training/scripts/analyze_envelope_study.py \
  --study data/runs/rendered-p10-20261013/study --benchmark
PYTHONPATH=training training/.venv/bin/python training/scripts/replay_factorized_study.py \
  --study data/runs/rendered-p10-20261013/study \
  --phase-study data/runs/causal-p3-20261012/study \
  --replay-binary target/debug/soundex-prediction-replay
```

No independent test, scored blind listening, production integration, device audio
or full-chain realtime qualification has been completed for P10. Repeated use of
development recordings limits generalization claims. Native replay retains the
original clean references, high-band energy/flux and low-band fidelity checks.
A lower composite score cannot override persistent identity, energy, stereo or
continuity failures. If longer training does not improve actual restoration,
inspect consistency/phase representation rather than treating bypass as success.


## Why step zero keeps winning (completed P10 diagnosis)

The initial function is intentionally identity: both output heads initialize to
zero, and the anchored mapping subtracts the zero-head transformation. It retains
the compressed input amplitude and input phase, then passes through DSP. It does
not reconstruct missing information. “Best” means lowest registered development
score among measured checkpoints, not best listening quality or a successful
restoration model.

All final scores below use the same registered weight-7 objective. The score
reduces algebraically to `0.5 * R + 0.5 * I`, where `R` is paired rendered
spectral/dynamics plus 0.25 waveform error against original clean, and `I` is
identity preservation relative to the existing DSP path. Each term uses the
existing 253:1 corpus weighting. The table decomposes final-minus-initial score:

| Seed / arm | Paired contribution change | Identity contribution change | Final score |
|---|---:|---:|---:|
| 12 raw | +0.01570134 | +0.03437806 | 0.34381483 |
| 12 rendered | -0.00051758 | +0.01786946 | 0.31108732 |
| 12 balanced | -0.00001985 | +0.00184385 | 0.29555944 |
| 13 raw | +0.01691302 | +0.03599266 | 0.34664111 |
| 13 rendered | -0.00062049 | +0.01667053 | 0.30978547 |
| 13 balanced | +0.00011800 | +0.00173082 | 0.29558426 |

Initial score is 0.29373544. For ordinary rendered supervision, identity harm
outweighs paired benefit by about 35× / 27× in this selection score. Reweighting
identity from 7 to 1 still does not reverse the development ranking. Algebraic
break-even weights are only about 0.203 / 0.261 for these two final checkpoints;
these are post-hoc sensitivity diagnostics, not proposed acceptance thresholds.
Changing a score cannot remove the measured damage.

A separate evaluation of all six **latest** checkpoints on the existing 40
train-domain probes distinguishes some mechanisms. These are training-domain
recordings/crops, not the full training objective over every sampled draw:

| Music rendered spectral/dynamics | Initial | Rendered seed 12 | Rendered seed 13 | Balanced seed 12 | Balanced seed 13 |
|---|---:|---:|---:|---:|---:|
| Train-domain probes | 0.572802 | 0.565293 | 0.565239 | 0.572899 | 0.572943 |
| Reused development | 0.567836 | 0.566820 | 0.566617 | 0.567796 | 0.568073 |

Ordinary rendered supervision improves its own weight-1 train-probe score from
0.522832 to 0.518950 / 0.518496. This shows effective learning, while much smaller
development restoration gains indicate a generalization gap. The balanced arms
fail to improve even the train-domain common score: initial 0.298761 versus
0.300368 / 0.300135. That cannot be explained solely as successful fitting followed
by held-out failure. Reducing edits is currently easier than learning corrections
that restore compressed clips while preserving clean clips.

Additional structural facts, not yet isolated causes:

- The predictor sees amplitude/context magnitudes and detector conditions, but
  not input phase, and its primary synthesis path cannot change phase. The oracle
  audit shows substantial differences between amplitude and complex restoration.
- Input-relative energy units are `sqrt(input_band_rms^2 + (scale * 1e-4)^2)`.
  Near-empty bands receive much smaller update units than surviving bands. This
  avoids the earlier overshoot but may impede generation. Missing-bin gradients
  are tested nonzero; this is an optimization-conditioning concern, not proof of
  a hard inability to represent the target.
- Raw per-frame improvement previously regressed after synthesis even on training
  probes. Direct rendered supervision addresses this mismatch, but it has not
  solved generalization or clean-input damage.

Therefore further identical long training is not justified by P10. The next
useful diagnostic is a fixed training-clip fit comparing (a) freely optimized
amplitudes through the same DSP, (b) freely optimized complex spectra, and (c) the
network, each with its paired identity case. This separates synthesis/phase limits
from head conditioning and optimization before changing capacity. Then test an
input-conditioned missing-band generation branch separately from preservation;
validate on development recordings without curve-specific thresholds. Capacity
and loss-weight changes alone are not established solutions.

Local evidence: `initial-ranking-audit.py` and `initial-ranking-audit.json` under
`data/runs/rendered-p10-20261013/`. The audit uses frozen P10 source, records latest
checkpoint hashes and does not change checkpoint selection. Full latest-checkpoint
native replay is a separate diagnostic, because the registered best checkpoints
are all initial functions.


The latest-checkpoint native diagnostic subsequently completed all 2,136 replays
with maximum proxy/native sample error 4.538e-6. Against identity DSP, balanced
music high-band SC changes by -0.001068 (seed 12, CI [-0.002751, +0.000146]) and
+0.000724 (seed 13, CI [-0.000465, +0.002645]); neither establishes a consistent
spectral gain. Identity music SC worsens by +0.002945 / +0.002627, with both 95%
intervals strictly positive. Paired high-band energy error improves by
-0.077763 / -0.071004 dB, but only the second interval excludes zero. Thus
restoration and preservation failures remain in actual DSP output, not only the
selection scalar. Evidence: `study/dsp-latest-review.json` and
`study/dsp-latest-comparisons.json`; no best-checkpoint selection was changed.
