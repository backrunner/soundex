# P18: Low-Learning-Rate Continual Training and Cross-Recording Recheck

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

## Pre-Execution Protocol

Building upon P17: lr=1e-4 short-range continual training on two seeds, which improved training objectives primarily by reducing identity degradation; native paired high-frequency metrics showed inconsistent improvements. This round evaluates whether a longer fixed-budget continuation can yield genuine recovery benefits.

- Continuation from the lr=1e-4, 512-step state of both seeds to a total of 2,048 steps, adding 1,536 additional steps per seed.
- Learning rate fixed at 1e-4; momentum, count, and random state restored, using P16 sampling from step 513 to 2,048.
- Model, loss, auxiliary coefficient 0.1365237430457469, clip=5, gain, codec, and warmup remain unchanged.
- Continuation on 40 recordings × 8 positions, with 4 music + 1 speech and corresponding identity per batch—this is the diagnostic ratio.
- Evaluation at steps 512 / 768 / 1,024 / 1,280 / 1,536 / 1,792 / 2,048, checking all 320 training positions.
- No early selection of checkpoints based on training or development scores. Both fixed endpoints must complete and be hashed before development scoring begins.
- Comparison across seven fixed conditions: initial identity output via DSP, two P17 starting points, two P16 baseline learning-rate endpoints, and two P18 endpoints. P18 is the continuation after a 256-step reduction in learning rate, not a restart from zero with low learning rate.
- Recheck of 40 identical-recording protected windows and all 89 cross-recording development segments; the development set consists of 33 music segments (26 recordings) and 56 speech segments (54 recordings), all previously used data and thus not independent test.
- 7 conditions × 89 segments × paired/identity = 1,246 native Rust DSP re-plays.
  For each, raw predicted spectra and actual DSP output metrics are saved individually, preserving original lossless references.

- Common loss maintained at 80% music / 20% speech; separate reporting of per-segment and per-recording averages is provided to avoid overrepresentation by multi-segment recordings.

## Interpretation Rules

A reduction in training error does not equate to successful compression recovery. Only when both seeds meet the following numerical screening at all fixed endpoints is further testing on new recordings and blind listening justified; this screening itself is not a release criterion.

- For music development, both the main and paired recovery errors must be below their respective P17 starting points and initial identity DSP baseline; identity degradation must not exceed the starting point.
- Native paired music high-frequency spectral convergence and transient errors must be below both the P17 starting point and initial DSP.
- Native paired music high-frequency energy absolute error and low-frequency spectral convergence must not degrade relative to the starting point.
- These conditions are evaluated both at the per-segment and per-recording level; a tolerance of 1e-6 applies only to numerical comparisons and cannot be interpreted as an auditory threshold.

The P16 baseline learning-rate endpoint serves as an auxiliary reference and is not considered successful if it exceeds a single failed model. Both speech, same-recording protected windows, per-segment anomalies, and raw vs. deployed metrics are displayed—no averaging is permitted to mask degradation.
If recovery is limited to reduced identity damage with inconsistent high-frequency cross-recording recovery, no additional training budget is automatically allocated. Instead, a contrast study of music coverage and output representability is initiated. Weighting remains under experimental status.

## Execution and Boundaries

- `training/scripts/run_consistency_followup.py`: pre-registration, checkpoint resumption, endpoint freezing, and full development re-play.
- `training/scripts/analyze_consistency_followup.py`: validates all inputs, source code, checkpoints, completion proofs, and score bindings according to registration rules, then aggregates results.
- Experiment directory: `data/runs/consistency-followup-p18-20261013/`.
- `pipeline.json` and `study/progress.json` record status; `pipeline.log` logs progress.

Native re-play remains a frozen prediction passed through Rust DSP and cannot be used as a proxy for candidate ONNX hop-by-hop performance validation.
This round does not modify the inference graph or DSP, does not alter default weights, and does not push audio or local experiment outputs.
New recovery tests include dropout and Adam validation to verify consistency of interrupted parameters, momentum, and update history at each step; other tests cover immutability, completion product drift prevention, and the failure of identity-only improvements to pass recovery screening.

## Completion Status

Launched at 22:55 (Singapore time) on 2026-10-13, backend queue initiated with source code snapshot
`4258462bf14d858762bf785225752d48f4e727ea`; pipeline PID at startup was `58162`.
The queue completed both seed training, endpoint freezing, full 1,246 native re-plays, and registration rule analysis by 23:09:39 (Singapore time) on 2026-10-13.
On 2026-10-14, a re-execution of analysis using frozen source code confirmed all inputs, source code, completion proofs, checkpoints, and score bindings passed validation.
Both seeds failed the recovery screening.

During execution, 460 local Python tests and Ruff lint/format checks passed. CI `38417618821` (code) and CI `38417737298` (launch doc) both succeeded.
This round only performed revalidation of outputs and updated documentation—no retraining or weight replacement occurred.

## Training Improvements Do Not Translate to Cross-Recording Recovery

Lower common main error is better. Development set results use the fixed 80% music / 20% speech weighting.

| Metric | 20261012：512 → 2,048 | 20261013：512 → 2,048 |
| --- | ---: | ---: |
| Full training library main error | 0.279703 → 0.273955 | 0.279112 → 0.272857 |
| Full training library composite target | 0.294147 → 0.287927 | 0.293456 → 0.286820 |
| Development main error | 0.296702 → 0.357384 | 0.309035 → 0.390484 |
| Music development paired recovery | 0.606519 → 0.672024 | 0.621027 → 0.710225 |
| Music development identity degradation | 0.046274 → 0.133471 | 0.062608 → 0.178525 |

All registered training checkpoints show a gradual decrease in main error, ending with reductions of 2.05% / 2.24%;
development main error, however, increased by 20.45% / 26.36%. The initial identity DSP development main error was 0.271475, paired recovery 0.588365—both endpoints still fall short of the baseline.
P16 baseline learning-rate endpoint: 0.362854 / 0.426925; surpassing these does not indicate achievement of recovery goals.

Per-recording average, music development main error increased from 0.320941 / 0.330513 to 0.369912 / 0.393549—degradation cannot be dismissed as a statistical artifact from a few segments.
Same-recording protected window music paired error slightly decreased, but identity degradation increased, resulting in a net gain and loss respectively.
Speech development main error decreased to 0.175929 / 0.174919 (initial 0.180645), and cannot be extrapolated to music gains.

This result supports the notion that "a small learning rate improved optimization behavior during this training phase," while exposing persistent cross-recording generalization and severe tail errors.
It does not support the claim that "continuing training will eventually surpass the initial." No new capacity ceiling evidence is observed.

## Native Development Replays and Tail Errors

Native/proxy maximum absolute difference: **1.022220e-5**, below the predefined 2e-5 validation threshold.
This value is a numerical consistency indicator and does not imply audible artifacts, pops, or device-level acceptance.

For all 33 development music segments, native paired output performance must be minimized:

| Metric | Initial Identity DSP | 20261012 Endpoint | 20261013 Endpoint |
| --- | ---: | ---: | ---: |
| High-frequency spectral convergence error | 0.548761 | 0.696447 | 0.773183 |
| High-frequency transient error | 0.717164 | 0.809499 | 0.861455 |
| High-frequency energy absolute error (dB) | 8.480117 | 7.497984 | 7.266781 |

While high-frequency energy error decreased, spectral shape and transient performance worsened; the same trend holds when averaged per recording.
Both seeds failed native high-frequency spectral shape and transient screening—this failure is not masked by a 1e-6 tolerance.
The second seed's low-frequency error increased by only ~2.8e-6 relative to the starting point, which still triggered the numerical screening and cannot be interpreted as perceptible.
No blind listening or new candidate ONNX real-time benchmark was conducted.

Relative to the baseline, 20/33 and 21/33 music segments improved in paired error, while 13/33 and 12/33 degraded.
Identity degradation occurred in 27/33 segments across both seeds. Thus, not all segments deteriorated—only a few severe failures offset the small improvements in most segments, accompanied by broader identity degradation.
A single blues segment, `9d95e7f035f878fa1a6ab421`, accounts for 77.81% / 77.82% of the net error increase in paired recovery.
This is a net error increase proportion, not the percentage of total errors attributable to this segment.
It is also a known counterexample from P14/P15; no re-intervention in frequency band was performed, and cannot be definitively linked to identical mechanisms.
All counterexamples and original scores are preserved—no track is removed, and no post-hoc screening relaxation is applied.

## Coverage Audit and Next Steps

After verifying the frozen manifest, current diagnostic training music includes only 32 recordings: ambient (7), electronic (7), classical (6), rock (4), pop (3), unlabeled (2), country/folk/jazz (1 each).
Blues and funk are absent.
Actual training music available: 253 recordings, including blues (7), funk (1), hip-hop (2).
Validation on 26 recordings and test on 21 recordings remain isolated.
Currently only 8 speech recordings are included in training; 193 available speech segments are used.
Counts are per-recording, not per synthetic pair—totaling tens of thousands of paired rows does not equate to independent song count.
Audit saved in root experiment directory as `coverage-audit.json`, including configuration hash and fixed manifest reconciliation.

The absence of blues and the presence of the most severe failure in style-matching is a testable coverage assumption and does not prove it is the sole cause.
Per the predefined protocol, this round ends without automatic continuation of the same configuration.
Next step: prioritize registration of music recording coverage comparison, using more recordings from the existing training split, while maintaining model, loss, low learning rate, and diagnostic mixing rules.
Update budgets and exposure per recording are controlled; development and test counterexamples are not included in training, and no style-specific gating is introduced.
If expanding coverage fails to stabilize reduction in native spectral shape/transient and identity degradation, a further output representability analysis will be conducted.
This coverage comparison has now been registered and launched as [P19](music-coverage-study.md), with default weights retained.
