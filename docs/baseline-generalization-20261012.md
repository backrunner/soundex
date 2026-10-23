# Baseline Generalization and DSP Attribution Recheck

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

2026-10-12. Conclusion: It is not valid to treat the old `shape-small` as a restoration target, and there is insufficient evidence to assert that it has overfit. What is currently confirmed is that the old baseline, the new model, and the application pipeline each have issues. The current comparisons mix training budgets, high-frequency gain strategies, and clipping distortion. Restoration should be judged against lossless references, with compressed inputs and identity DSP used to isolate newly introduced artifacts.

## Is the Baseline Overfit?

The old `shape-small` derives from the original epoch-168 model and subsequent continuation; its inherited training budget differs from the fresh P2 runs. The exact historical initialization chain is recorded in the linked study and provenance reports. It is not under budget parity with the P2 model, which was trained from scratch with 4096 updates. Therefore, the performance gap between the two cannot be used to conclude that the TCN/GRU architecture is inferior. The internal baseline of P2 under the same training budget serves as the architectural control group.

In the final 8-epoch log of `shape-small`, the training loss ranged from 1.299 to 1.393, and the validation loss from 1.342 to 1.388. There is no clear trend of training loss decreasing while validation loss increases. The distribution of random training samples differs from that of fixed validation samples. Thus, this curve neither proves overfitting nor rules out early-stage overfitting.

Cross-checks for leakage across track, source checksum, and `work_group/split_group` isolation have found no evidence of cross-contamination between training, validation, and test sets. However, the development set has been repeatedly reused for structural evaluation, loss analysis, and candidate selection, and historical test data has been used multiple times for diagnostics. As such, it cannot be considered a truly independent and fresh test set. Generalization to unseen sources has not been established. A more accurate conclusion is: **There is a risk of generalization and selection bias, and it remains insufficient to attribute the issues of the old baseline to overfitting; degradation in the application pipeline has been directly observed.**

## Actual Rust Comparison on the Same Input

A frozen model, Rust binary, gain strategy, and post-processing script were fully paired with sample lists for complete end-to-end comparison (2–4 second audio segments, not full tracks). Evaluation started from the 72nd hop, with the final 256 samples removed. All samples are at 44.1/48 kHz native sample rates.

- **Training probe set**: 40 recordings, 32 music, 8 speech; used only for diagnosis, not for weight selection.
- **Development recheck**: 89 entries / 80 recordings; 33 music entries / 26 recordings.
- **Historical held-out recheck**: 78 entries / 39 recordings; one training-encountered encoding and one preserved AAC160 encoding per recording; 42 music entries / 21 recordings. The recordings themselves were not part of the training set.
- **Identity DSP**: Real Rust output with `low-latency-identity.onnx`, model gain, and no modification to the input. This is used to measure the impact of synthesis, gating, and safety clipping, and differs from spectral DSP operations that generate new high-frequency content.

Below are the music entries, averaged equally; lower error in high-frequency spectral shape, transient, and energy is better, while higher SI-SDR relative to input is better.

| Historical Held-Out Path | High-Frequency Spectral Shape Error | High-Frequency Transient Error | High-Frequency Energy Error (dB) | SI-SDR Relative to Input (dB) |
| --- | ---: | ---: | ---: | ---: |
| Compressed Input         | 0.74064 | 0.89945 | 4.01805 | 0 |
| Identity DSP             | 0.74182 | 0.92589 | 3.70158 | −0.73523 |
| Old shape-small / MatchEdge | 0.76051 | 1.13067 | 2.92218 | −1.16909 |
| Same Old Weights / Model Gain | 0.70575 | 0.97294 | 4.04358 | −1.08951 |
| P2 Same Budget Old Backbone / Model Gain | 0.85224 | 1.09783 | 3.83411 | −1.18089 |
| P2 TCN / Model gain | 0.77669 | 0.98618 | 3.23143 | −1.01688 |

The old MatchEdge reduces high-frequency energy error but degrades spectral shape, transient, and waveform fidelity. Switching to the same weights with Model Gain improves spectral fidelity, but energy error does not improve. This demonstrates that no single metric can definitively rank quality.
The Identity DSP has a baseline SI-SDR degradation of −0.735 dB, with a 95% bootstrap interval across 21 recordings of [−1.469, −0.108] dB. For the old MatchEdge, the interval is [−2.095, −0.378] dB; for TCN, it is [−1.919, −0.264] dB. These intervals are uncorrected for multiple comparisons and do not constitute independent experimental validation.
It is not valid to attribute the entire difference to network error. Similarly, the dB difference cannot be interpreted as a linear proportion of energy degradation.

| Domain / Path            | Input High-Frequency Spectral Shape | Output High-Frequency Spectral Shape | SI-SDR Relative to Input (dB) |
| --- | ---: | ---: | ---: |
| Training Probe / Old MatchEdge | 0.73900 | 0.70265 | −0.19256 |
| Development / Old MatchEdge | 0.84266 | 0.89908 | −0.96549 |
| Historical Held-Out / Old MatchEdge | 0.74064 | 0.76051 | −1.16909 |
| Training Probe / TCN     | 0.73900 | 0.65589 | −0.08722 |
| Development / TCN        | 0.84266 | 1.31585 | −0.37492 |
| Historical Held-Out / TCN | 0.74064 | 0.77669 | −1.01688 |

Both models exhibit differences in performance between training-domain and unseen recordings. However, the material, encoding, and loudness distributions are not perfectly aligned, and the inherent distortion of the Identity DSP further complicates interpretation. This table cannot be taken as evidence that the model has memorized the training set.

In the historical held-out music set, comparing seen vs. unseen AAC160 encodings, input high-frequency spectral shape was 0.70456 (seen) / 0.77671 (unseen); for old MatchEdge: 0.68436 (seen) / 0.83666 (unseen); for same weights with Model Gain: 0.65967 (seen) / 0.75183 (unseen); for TCN: 0.68440 (seen) / 0.86899 (unseen). The partial advantage of the old MatchEdge does not transfer to the preserved AAC160 encoding. When the same weights are switched to Model Gain, the result changes—demonstrating that the application strategy is a variable. Each layer has only 21 entries, and conditions such as sample rate, time crop, and duration are not strictly identical. This is not a complete, causal experiment that isolates encoding change.

Speech results are recorded separately: historical held-out input high-frequency spectral shape was 0.50819; for old MatchEdge: 0.54600; for TCN: 0.48288; corresponding SI-SDR differences: −0.11139 dB (old MatchEdge), −0.07285 dB (TCN). Speech and music are not mixed in average—this prevents masking degradation in music.

## Limiting and Additional Generation Localization

The current sample-wise limiter begins compression at 0.855 and caps at 0.95. It is not merely a transparent safety fallback: legitimate peaks are also altered, and nonlinearity introduces new spectral content. The following comparison uses Rust output from an identity model and compares it against per-sample parsed limiter formulas, using the same steady-state window.

| Diagnostic Fragment       | Reference High-Frequency RMS (dBFS) | Input | Identity DSP | P2 TCN |
| --- | ---: | ---: | ---: | ---: |
| Weak High-Frequency AAC192 | −81.99 | −81.70 | −72.74 | −59.02 |
| Weak High-Frequency Vorbis q3 | −77.23 | −77.39 | −77.39 | −66.66 |

For the AAC fragment, 0.0901% of steady-state samples exceed 0.855. The Identity DSP raises the high-frequency power level by approximately 8.97 dB. The maximum difference between Rust identity output and parsed limiter output is 2.38e−7, RMS difference is 3.90e−8—indicating that the DSP distortion in this fragment is primarily due to limiting. For Vorbis, the peak is only 0.7243 and does not trigger limiting. The identity path is nearly transparent, while the TCN introduces an additional ~10.73 dB of high-frequency content—indicating a genuine problem of model-generated content. In the historical held-out set, the worst Identity SI-SDR entry shows a maximum difference of only 3.58e−7. This is a post-hoc diagnosis and does not represent all materials, nor has blind listening been performed. Therefore, it cannot be used to assert perceptibility.

Correction to the prior baseline report attribution: the values reported were outputs of the model **plus DSP**. The additional high-frequency content in AAC cannot be fully attributed to the network. Both types of defects coexist, and a uniform reduction in enhancement cannot resolve the root cause.

## Meaning of Loss and Subsequent Control

Numerical testing does not imply that the target is fully reasonable. The current training objective reproduces the deployed limiter, so it is not a single issue of "training not simulating deployment." Even clean→clean paths passing through limiting can introduce errors. When reference peaks exceed the output ceiling, full reconstruction becomes impossible. The `prelimit_excess` term penalizes peaks that were originally valid, introducing a trade-off between safety and waveform/weak high-frequency restoration. It is incorrect to claim that the loss is exactly zero at lossless reference points. The P3-added excess power term is zero at correct reference points, but only for that specific term.

The [P3](weak-band-study-20261012.md) experiment has been initiated with the original deployment pipeline frozen. It maintains three equally budgeted groups and two seed isolations, with weak high-frequency supervision and identity training held in place. The experiment definition is not altered during execution. Even if P3 improves, it does not imply that the limiter issue has been resolved.

The next round of DSP comparison must be conducted with a separate freeze: under constraints of transparent intervals, smooth peak control, and safe ceiling, verify identity, legitimate large peaks, overload, transients, stereo, block switching, and fault recovery. The loss proxy and parity must also be updated.
It is not valid to remove the limiter to improve scores, nor to claim that a lossless reference pre-limited constitutes "restoration of original audio." Subsequently, cross-validate the model, two gain strategies, and DSP. Report the original prediction and final waveform, and retest real-time overhead.

Generalization must still include independent sources and missing styles. The current training set contains 253 music tracks, 26 validation tracks, and 21 test tracks. Validation lacks country, pop, hip-hop, funk; test lacks Chinese, country, hip-hop, jazz. Until these gaps are closed, no claims of generalization or absence of overfitting can be made. No weight promotion is authorized.

## Traceable Evidence

Local directory `data/reports/baseline-audit-20261012/` is ignored: `registration.json`, `summary.json`, per-path/domain JSONs, `identity-registration.json`, `identity-summary.json`, `limiter-audit.json`, and the generation scripts `audit.py`, `identity_audit.py`, `limiter_audit.py`.
Each input is individually verified via checksum, and intervals are clustered by `track_id` (2000 instances). Materials and reports are not pushed.

- Old weights SHA-256: `bdef9d1269ea4b8b1e8e5e04bd6d25032b817f5401e13c51e2c064d721503782`
- P2 TCN: `cb02a6f15055e4423e6403cf3bd4b18b50007209452502c760d09007fc2958d9`
- Rust scorer: `07f9373faaed7d448396025e076c5466a9bf10ede5a98797dd1c693f395faac4`

`independent_test=false`, `listening_performed=false`. No model promotion occurred in this round, and no CPU benchmark during concurrent training was used to claim real-time qualification.

This steady-state window differs from the earlier 32768-sample center-crop report;
do not combine the reports to calculate gains. Historical held-out SI-SDR changes
are −1.169 dB for old MatchEdge and −1.017 dB for TCN, with the confidence intervals
reported above.

The historical initialization chain specifically includes 8 continuation epochs, V4 epoch 4 and this stage's selected epoch 4 after the original epoch-168 model; these are not matched to P2's fresh budget.
