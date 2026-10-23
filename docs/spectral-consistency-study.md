# P16: Reference Perception-Aided Synthetic Consistency Constraint

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

## Objective and Fixed Protocol

P15 confirmed the failure mechanism of source-band correction under OLA influence, where adjacent weak bands are affected.
This round compares the original training objective against one augmented with a reference perception-based synthetic consistency term; no frequency band is directly disabled.

- Two groups: `control` / `consistency`, using the same 40 audio segments from P13, each with 8 positions.
- Identical pairing/identity, 4 music + 1 speech batch, gain=0, codec, sample order, and protection window.
- Identical 807,398-parameter model and input phase, Adam 1e-3, no decay, gradient clipping 5.
- Seeds: 20261012 / 20261013; each group runs 2,048 steps per seed, with fixed reporting at 256 / 2,048.
- Two checkpoint hashes are registered prior to development scoring; no checkpoint is selected based on development score, no seed change, or weight adjustment.
- Evaluation uses only the original P13 loss and pristine, unclipped reference; the auxiliary loss is excluded from joint scoring.
- A new model is created, with the same code snapshot frozen; weights remain in experimental state.

## Auxiliary Term

Let `P` be the complex STFT re-analysis after OLA, and `Zp` be the predicted spectrum after existing band mixing.
`Zr` is the reference spectrum under the same mixing procedure: for paired cases, it uses clean magnitude and input phase; for identity cases, it uses true input magnitude and input phase. The warmup phase retains input values.

The auxiliary error is defined as `(P - I)(Zp - Zr)`, smoothed with RMS normalization over five frequency bands, and split equally between pairing and identity. This constraint only enforces the reference coordinates that are representable within the current input phase architecture. It does not pretend to reconstruct a full clean complex spectrum, nor does it modify the original, unclipped reference in the main objective.

It does not require frame-wise invariance or simple smoothing: when prediction matches the reference, the auxiliary term and its gradient vanish, including in cases with real transients. The synthetic inconsistency in the reference itself is penalized.
This term is used solely during training; the model structure, inference graph, and DSP execution path incur no additional computation or buffering.
It may introduce adverse trade-offs, and thus must be validated through actual training and original metric recheck.

## Weight Calibration Rules (Fixed Prior to Training)

Only the P13 training positions are read; development samples are not used for coefficient calibration.
For each seed, take the first 64 batches of the sampling table, exactly covering all 320 training positions once.
Using the initial model, compute gradients for the main objective and auxiliary term, without parameter updates.

`lambda = 0.1 * sqrt(sum(||grad(main)||²) / sum(||grad(aux)||²))`

The fixed value 0.1 indicates that, on the initial training batches, the RMS of the auxiliary term gradient is 10% of the main term gradient. This does not guarantee a consistent ratio across subsequent steps. After calibration, the coefficient is fixed, and the original gradient values, input table, source code, and output hashes are recorded. The training script rejects any calibration file derived from different source code, position list, or seed.

## Validation and Status

The auxiliary term has been verified: zero error and zero gradient when reference is correct, finite gradients for silence and low-energy content, finite difference checks, transient reference behavior is not penalized by smoothing; the complex synthesis operator matches numerically with the original OLA/re-analysis.

Training-side calibration is complete: initial RMS of main objective gradient is 1.349659, auxiliary term is 0.988589, resulting in a fixed coefficient `lambda = 0.1365237430457469`. The coefficient was not determined using development samples.
Four training and validation runs were completed on 2026-10-13 22:05 (Singapore time). Source code snapshot is
`000c450041e29b6fa44be4168e0b9b6f717186de`. Experiment root directory: `data/runs/consistency-p16-20261013/`.
The queue proceeds in order: training-side calibration, two-seed double training, fixed checkpoint scoring and native playback, joint metric comparison and aggregation, completing 300 formal native replays.

## Final Results and Progress Assessment

The joint development score is lower is better. The initial identity model, processed through the same DSP, achieves a score of **0.271475**:

| Seed | Steps | Control | Consistency Auxiliary Term |
| --- | ---: | ---: | ---: |
| 20261012 | 256 | 0.305075 | 0.305742 |
| 20261012 | 2,048 | 0.408393 | 0.362854 |
| 20261013 | 256 | 0.415565 | 0.318891 |
| 20261013 | 2,048 | 0.394683 | 0.426925 |

Relative improvement over control: +11.15% / -8.17%. Neither seed exceeded the initial value.
We cannot claim effectiveness of the auxiliary term based on the first seed alone, nor can we select the 256-step point as official.
A recheck script validates all checkpoints, scores, sampling, and completion evidence; both seeds' full training positions and development scores are exactly reproduced as in P13 multiposition.

Additional audit of frozen checkpoints: for each seed, the first 64 batches of the sampling table exactly cover all 320 training positions once, preserving 4 music + 1 speech and corresponding identity. No optimization was performed; development scores were not used to reselect points. Report saved as `checkpoint-audit.json`, alongside script, script and configuration hashes, checkpoint hashes, and per-batch results.
Recomputed main term averages match the per-position training scores (difference <1e-6).

| Consistency Group | Steps | Full Training Position Main Term | Auxiliary Term | Main + λ × Auxiliary |
| --- | ---: | ---: | ---: | ---: |
| 20261012 | 256 | 0.290827 | 0.116055 | 0.306671 |
| 20261012 | 2,048 | 0.272333 | 0.101363 | 0.286171 |
| 20261013 | 256 | 0.289273 | 0.111606 | 0.304510 |
| 20261013 | 2,048 | 0.335158 | 0.149575 | 0.355578 |

The second seed shows degradation in both main and auxiliary terms, and the sum, indicating a degradation in optimization state—not merely overfitting, nor a case of "sacrificing main term for better auxiliary term." This suggests a degradation between the two saved points, though no definitive conclusion about divergence or learning rate is drawn.
The first seed shows improvement in training, but degradation in development, indicating a generalization issue also exists.

The final auxiliary gradient-to-main gradient RMS ratio per batch is 12.87% / 12.31%, with cosine median values of 0.265 / 0.485; 12 / 21 batches show backward gradients. These are diagnostic values at the frozen checkpoint, not along the historical update trajectory, and do not prove absence of local conflicts.
There is no evidence that the final failure resulted from the auxiliary gradient overwhelming the main term overall.

Final development set results: 21/33 music segments and 10/33 speech segments show improvement over control. The first seed's net improvement is approximately 97.0% attributable to three severe counterexamples identified in P14/P15. The second seed improves some of these counterexamples, but introduces a new severe degradation in ambient segment `b925b66fc8abcf285a7277df`.
Among the 254/256 music training positions in the second seed, performance is worse than control—indicating issues beyond just development counterexamples. These are post-hoc diagnoses and cannot be used to filter tracks or implement band-specific gating.

Native playback and proxy maximum absolute difference is **1.020730e-6**. The fixed development subset (4 music + 1 speech and corresponding identity) does not represent all development recordings. The relative change in high-frequency spectral convergence error from initial is: control -0.004476 / -0.004637, consistency -0.001880 / +0.002253; the high-frequency transient error relative to initial is +0.005492 / +0.007183.
Local energy improvements do not translate into stable improvements in spectral shape, transients, or identity. This is a frozen feature playback via Rust DSP, not a full real-time ONNX performance validation; blind listening was not conducted.

### Follow-up Validation

The following short-term comparisons have been completed as part of [P17](consistency-learning-rate-study.md): both seeds with smaller learning rates show improvement in full training set main and combined objectives, primarily due to reduced identity damage. High-frequency spectral shape and transient performance remain unchanged. This supports further investigation with smaller updates, though it does not prove successful cross-recording recovery.

Long training with P16 configuration is paused. No direct scaling is proposed. Instead, we begin with two fixed 256-step checkpoints (including Adam state) from each seed, and replicate short-term comparisons: preserving data order, auxiliary coefficient, and all other settings. Compare lr=1e-3 vs lr=1e-4, each running 256 additional steps, with evaluation at every 64 steps across the full training position library. This is a mechanism diagnostic and must be registered prior to execution; no point selection is based on development results, and no claim is made for a new independent generalization test.

We must verify whether real updates simultaneously improve the main term, auxiliary term, and identity—not just immediate small-batch loss.

If smaller updates fail to stabilize main term improvement, further investigation should focus on output head update scale and target coupling, not automatic step extension. If training stabilizes but cross-recording degradation persists, we should expand music training coverage, and then conduct single-factor comparisons on synthesizable complex or waveform residual representations. Current evidence does not support a parameter count deficiency.

Future winners must be validated on uninvolved recordings, with actual high-frequency recovery, identity, transient, and low-frequency fidelity, and must undergo blind listening. Any structural or DSP change requires revalidation of streaming latency, continuity, and model budget.

We must monitor whether both seeds improve across recordings, and preserve identity, transient, and low-frequency fidelity.
We cannot consider a return to initial identity output, loss reduction, or improvement in three counterexamples as success.
Repetition of development data is limited; final validation must include uninvolved recordings, blind listening, and actual deployment.

## Scripts

- `training/scripts/calibrate_consistency.py`: Fixed training-side gradient calibration.
- `training/scripts/run_subset_study.py --sizes 40 --crop-study --crop-positions 8`: Shared training process.
  The experimental group additionally passes `--consistency-calibration <calibration.json>`.
- `training/scripts/analyze_consistency_study.py`: Validates sampling and output hashes for both groups, comparing only original metrics.

Implementation validation: 451 local Python tests passed, Ruff lint/format passed; P16 remote CI `38407254788` and launch status CI `38407487492` both succeeded.
Status is documented in the experiment root directory's `pipeline.json`, logs in `pipeline.log`, and joint metrics in `study/consistency-comparisons.json`. No default weights were replaced.

The auxiliary residual is `(P-I)(Zp-Zr)`. Its coefficient was calibrated on the first 64 registered batches per seed as `lambda = 0.1 * sqrt(sum(||grad(main)||^2) / sum(||grad(aux)||^2))`, then frozen without development data. Relative to the corresponding control the final results improved by 11.15% for one seed and regressed by 8.17% for the other; neither exceeded initialization.
