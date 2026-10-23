# P2 First Seed Recheck: Progress Made, Not Yet Fully Outperformed by Old Weights

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

2026-10-12, mid-checkpoint.
First seed of the [Training Scheme](causal-training-20261012.md)
20261012: All three groups completed 4096 steps, exported, and evaluated with real Rust scores; the second seed is still running.
Below, we freeze the results of the first seed and do not claim any conclusions for the second seed based on it. Default weights remain unchanged.

Future updates: All six groups are now complete. See the [full status](causal-training-20261012.md).
The [Baseline and DSP Attribution Recheck](baseline-generalization-20261012.md) further identifies that limiters can also introduce additional high-frequency content. Below, raw values are preserved, but any error incurred when applying DSP to the model cannot be attributed solely to the network.

## Training Objectives and Budget

| Backbone | Best Validation Step | Unified Validation Loss | Parameter Count | ONNX MiB |
| --- | ---: | ---: | ---: | ---: |
| Legacy Backbone (same budget) | 4096 | 2.662400 | 905,734 | 3.550 |
| GRU | 2048 | 2.485908 | 643,458 | 2.471 |
| TCN | 4096 | 2.462378 | 639,906 | 2.479 |

The GRU and TCN achieved relative reductions in loss of 6.63% and 7.51%, respectively.
At 4096 steps, GRU loss was 2.4883, slightly worse than at 2048 steps, so this does not justify blindly extending training.
For TCN, the loss improved from 2.4769 to 2.4624 from 2048 to 4096 steps, but the improvement in the second half was marginal.
Both models are smaller in size, but training time is influenced by initial cache construction and I/O ordering, so the recorded total training seconds cannot be used as a proxy for inference performance.

The loss improvements must also be broken down: the music raw-low-complex metric dropped from 1.0004 in the legacy backbone to 0.00122 (GRU) and 0.00149 (TCN).
This metric carries a weight of 0.1, contributing approximately 56% and 50% respectively to the total difference in music loss between old and new models.
This helps maintain low-frequency content in the original model, though during runtime, the model naturally fuses low-frequency observations.
Therefore, the full difference in loss cannot be equated to an overall gain in audio quality.

## Actual Rust Comparison on Fixed Segments

Fixed 89 development validation segments: 33 music segments / 26 recordings, 56 speech segments / 54 recordings.
Original sampling rate: 44.1/48 kHz; 32768-sample center crop; common steady-state interval: `[72*128, 32768-256)`.
All values below are per-segment mean for music; full per-segment metrics are preserved in JSON and cannot be extrapolated to full song performance.
All comparisons use the same batch of inputs, with no mixing of different windowing strategies from prior reports.
No blending of old reports with different windows.

| Path | High-Frequency Spectral Shape Error ↓ | High-Frequency Transient Error ↓ | High-Frequency Energy Error (dB) ↓ | SI-SDR dB ↑ |
| --- | ---: | ---: | ---: | ---: |
| Compressed Input | 0.69958 | 0.82139 | 7.49576 | 27.77139 |
| Original shape-small / MatchEdge | 0.80451 | 1.27979 | 4.20721 | 26.75789 |
| Original shape-small / Model Gain | 0.77570 | 1.15518 | 6.00342 | 26.93062 |
| P2 Legacy Backbone / Model Gain | 1.38157 | 1.41816 | 7.30113 | 26.99665 |
| P2 GRU / Model gain | 1.11588 | 1.22104 | 6.24508 | 27.35952 |
| P2 TCN / Model gain | 1.15316 | 1.20281 | 5.97236 | 27.39247 |
| Pure DSP Spectral | 2.07465 | 2.65664 | 6.73168 | 27.49107 |

Spectral shape error refers to high-frequency spectral convergence; transient error refers to normalized error against a reference forward spectral flux.
The new backbone outperforms the legacy backbone retrained under the same budget; TCN achieves better SI-SDR than the original shape-small, though its high-frequency spectral shape and energy errors do not fully dominate.
Relative to compressed input, both new backbones show a mean reduction in music SI-SDR of approximately 0.41 dB and 0.38 dB, respectively.
For 24/33 and 25/33 music segments, the new backbones achieve lower spectral shape errors than input, but a few counterexamples significantly affect the mean.
These counterexamples are not removed, and we do not replace comprehensive evaluation with "improvement in most segments."

### Recalculated by Recording

We discovered that the training entry `actual_score` did not pass `cluster_field`, resulting in an automatic row bootstrap instead of the previously documented per-recording clustering.
This round recalculates **track_id clustering, 4000 bootstrap samples**, and maintains paired differences with all encoding and channel roles sampled together within each recording.
The per-segment mean remains unchanged.
The currently running training code is frozen and not modified during execution; new statistics are written to independent `*-clustered.json` and `comparison.json`, and old reports are not overwritten.
The next round should correct the default clustering setting in the entry.

TCN vs. P2 Legacy Backbone (music, 95% recording cluster interval):

| Metric | Difference | 95% CI |
| --- | ---: | --- |
| High-Frequency Spectral Shape | −0.2284 | [−0.5616, −0.0108] |
| High-Frequency Transient | −0.2154 | [−0.5408, −0.0250] |
| High-Frequency Energy Error (dB) | −1.3288 | [−2.6696, −0.0153] |
| SI-SDR dB | +0.3958 | [+0.0340, +1.0314] |

Relative to original shape-small / MatchEdge, TCN shows an increase in high-frequency energy error of 1.7651 dB [+0.6828, +2.8451], and an increase in SI-SDR of 0.6346 dB [+0.1237, +1.4881].
This represents a trade-off, not a comprehensive victory. The intervals are derived from development data used in selection and a single training seed; they have not been corrected for multi-metric comparison and cannot be considered independent or definitive evidence of significance.

## Counterexamples and Gaps in the Objective Function

The two worst-performing music segments come from the same piece: `Ballad Of Ishmael`, in AAC192 and Vorbis q3 versions.
In these cases, the reference target already had extremely low energy in the high-frequency region, and the compressed input barely altered it.
The model, however, continued to add high-frequency content.
Using the same steady-state window, and computing high-frequency RMS via single-sided Parseval and Hann window power normalization:

| Segment | Lossless Reference (dBFS) | Compressed Input (dBFS) | P2 Legacy Backbone | GRU | TCN |
| --- | ---: | ---: | ---: | ---: | ---: |
| AAC192 | −81.99 | −81.70 | −57.24 | −60.37 | −59.02 |
| Vorbis q3 | −77.23 | −77.39 | −61.48 | −66.19 | −66.66 |

Thus, the relative spectral error is indeed amplified by a weak denominator, and there is actual excess energy introduced.
We cannot simply allow counterexamples due to small denominators, nor can we conclude the presence of audible noise without blind listening.
After excluding the `Ballad Of Ishmael` segment, TCN's high-frequency spectral shape mean drops to 0.6852, compared to 0.6777 for the original MatchEdge.
This is a diagnostic indicator of sensitivity; the official result retains this recording.

Currently, the relative loss uses a lower bound of 1% of total spectral energy and a reference power threshold to prevent gradient explosion in silence; this also weakens constraints on very weak high-frequency content.
Multi-resolution weighting with average cutoff and short cropping may affect alignment between proxy and actual scoring.
These are documented optimization directions with evidence, but causal ablation has not been completed, so all errors cannot be attributed to this factor.
Future work should, without compromising the recovery of genuinely missing high frequencies, incorporate:
- native low-bandwidth / near-identity training baselines,
- perceptual constraints on excess high-frequency energy,
- input-conditioned generation confidence,
and retain limited gradients in low-energy scenarios.
We do not advocate for uniform suppression, enhancement, or hard clipping as a means to achieve recovery goals.

Speech results also show trade-offs: TCN achieves a high-frequency spectral shape of 0.6792, an improvement over input (0.7367), but still worse than the legacy backbone's 0.6393; energy error of 5.4596 dB is also worse than the legacy backbone's 4.6399 dB.
Music and speech are reported separately; we do not average 56 speech segments to mask 33 music counterexamples.

## Evidence and Next Steps

Original and newly generated reports are stored in the directory: `data/reports/causal-p2-review-20261012/`, including:
`comparison.json`, `*-clustered.json`, `weak-band-audit.json`, and the local scripts that generated them: `compare.py` and `energy.py`.
All reports are tied to config, Rust binary, and model hash; reloaded PCM files are checksum-verified.
The training directory remains at: `data/runs/causal-p2-20261012/`.

SHA-256 hash for original shape-small:
`bdef9d1269ea4b8b1e8e5e04bd6d25032b817f5401e13c51e2c064d721503782`

First seed ONNX SHA-256 hashes:

- Legacy Backbone: `0f24a65a6d40cbd2132204f41fffe652f26f307ed77e06619c6a5be7586a5355`
- GRU: `243bb9e5c5e612c389f9dbbd70ca66314d4afca1e11da61f6cc654be23687235`
- TCN: `cb02a6f15055e4423e6403cf3bd4b18b50007209452502c760d09007fc2958d9`

We will first complete the second seed before assessing reproducibility.
Historical ablation, full music and true stereo rechecks, and blind listening remain preserved.
This round did not interrupt background training to measure CPU real-time cost, nor did it use concurrency-induced latency as a proxy for deployment performance.
Per-hop CPU / RSS / callback utilization for the new weights remains to be independently verified; ONNX size compliance does not substitute for this requirement.
`independent_test=false`, `listening_performed=false`; no model promotion.

The exact shared steady-state interval is `[72*128,32768-256)`. The reported positive intervals are [+0.6828,+2.8451] and [+0.1237,+1.4881], respectively, with direction interpreted according to their metric.
