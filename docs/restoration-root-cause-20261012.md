# Recovery Gain Stagnation: Attribution Auditing and Experiment Sequence Adjustment

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

2026-10-12, baseline `b0dc7d2`. This round of training is paused; no additional training is being added. Only frozen weights and existing data are being evaluated.

Conclusion is not "all modifications are useless": existing engineering, operational cost, and clipping improvements have been achieved. However, P3/P4 have not yet produced a complete, publishable recovery quality gain. Adding energy penalty terms along the current path lacks justification.

Future progress: [P5 magnitude-based discrimination experiments](envelope-study-20261012.md) have been completed across six training runs, following this audit sequence, and further decomposed the per-frame band intensity and intra-band spectral shape errors. The audit evidence and original plan remain preserved as documented.

## New Evidence

The first seed of frozen P3 control / P4 balanced was evaluated, using 32 training music probes and 33 development music snippets (26 development recordings), without introducing new held-out data or new weights. All inputs were drawn from original registration configuration and cached sources. Fixed input detection, model gain, and smooth-knee DSP were maintained. Only the magnitude, phase, or full complex spectrum of the network was replaced.

Reference information was used solely for offline localization; the oracle is not a deployable model performance.

The table below shows the mean values over development music snippets, aligned with the original time axis and actual output reports. Outside of differentiable proxies, on the first two training and first two development snippets, per-intervention replay was performed via Rust replay, resulting in 40 total replays across both models. The maximum waveform difference in the stable region was **6.04e-7**, supporting the proxy's accuracy, though this does not constitute a full deployment-level consistency proof for all audio.

| P4 Balanced Prediction Replacement | High-Frequency Spectral Shape SC ↓ | High-Frequency Flux Error ↓ | High-Frequency Energy Error (dB) ↓ | ΔSI-SDR dB ↑ |
| --- | ---: | ---: | ---: | ---: |
| Original Model                    | 1.28360 | 1.21830 | 5.92832 | -0.33256 |
| Correct Magnitude + Model Phase  | 0.32872 | 0.57243 | 1.03228 | -0.32974 |
| Model Magnitude + Correct Phase | 1.15632 | 1.36888 | 5.81142 | -0.17381 |
| Correct Magnitude + Correct Phase | 0.15362 | 0.28950 | 0.26452 | +0.46533 |
| Correct Complex Spectrum (constrained by current output head boundary) | 0.15410 | 0.29055 | 0.27208 | +0.46424 |

Corresponding spectral consistency (SC) values for P3 control are **1.18844 / 0.34058 / 1.08119 / 0.15362**, showing consistent trends.

It is not valid to compute independent contribution percentages by subtracting individual error terms: these interventions interact, and swapping magnitude/phase can result in spectral outputs that fail adjacent-frame synthesis consistency. However, the following can be confirmed:

1. **Magnitude and spectral shape prediction are initially deficient.** Providing only correct magnitude leads to significant improvements in spectral shape and transient fidelity. Providing only correct phase retains most spectral shape errors. Referring to this as a "phase" or "limiter" distortion is insufficient—this reflects a deeper issue in phase or texture generation.
2. **Correct magnitude alone does not imply waveform recovery.** When correct magnitude is paired with model phase, ΔSI-SDR shows almost no improvement. Only when both correct magnitude and correct phase are provided does a clear waveform benefit emerge. Band power loss cannot independently determine these fine-grained details.
3. **The current synthesis chain and residual output range allow for significantly better results.** When the clean residual is constrained to ±2× input RMS spectrum per component, the oracle retains most of its benefit. Output range is not the primary bottleneck for this probe. This does not prove that a shared network with width=144 can learn this mapping, nor does it rule out limitations in latent representations, optimization, or generalization.

## Why Adding Energy Loss Easily Becomes a Trade-Off

For the first seed of frozen P4 balanced, two local gradients on the decoder parameters were computed:
`0.25 * output_band_energy` and `output_waveform + 0.2 * raw_high_complex`.

Among 32 training probes, **22** had negative cosine similarity, with a median of **-0.3713**; among 33 development snippets, **20** were negative, with a median of **-0.0762**.

This represents evidence of gradient conflict in the current weights and sample neighborhood. It does not equate to the actual update direction of AdamW, nor does it prove that the two objectives necessarily conflict at the correct solution. It explains why "increasing energy recovery" tends to amplify existing amplitude and phase errors, while "penalizing excess energy" simultaneously weakens valid recovery. A total loss can be well-behaved and gradients can be correct, but the local optimization direction may not lead to the desired recovery. The next step should be to first improve the predictability of spectral shape representation, then determine how missing components—phase or texture—can be generated, rather than continuing to add closely related power terms.

Another sensitive probe: when input is globally inverted, the target magnitude spectrum should remain unchanged. The current model's predicted high-frequency magnitude deviation relative to the target high-frequency magnitude norm has a median of **0.1646** for P3 and **0.1608** for P4 (development music). This indicates the model is sensitive to this physically equivalent transformation. This is not a perceptible distortion ratio, and it has not yet been proven to be the primary cause of generalization degradation. The current training sampling only introduces gain perturbations and does not explicitly include polarity control; this should be tested under small-scale, controlled conditions.

## Training Task Definition Is Not Uniform

- Network input consists of only short-window log magnitude, compressed Re/Im, and sample rate. The cutoff frequency, gate, and band masking from the online detector are used for synthesis and loss computation, but are not explicitly passed as network conditions. The network can only indirectly infer the synthesis region.
- The training proxy uses per-hop online cutoff synthesis, with multi-resolution supervision using time-averaged values; the final HF metrics use a static cutoff derived from pre-processing of the entire 2–4 second snippet, 8192-point analysis. The absolute difference in mean cutoff values between the two approaches has a median of **1016 Hz**, with a maximum of **10869 Hz**. This is not an audio alignment bug or a guaranteed error in the metric, but the three measures evaluate different frequency regions and cannot be treated as a single objective.
- The new P4 power term covers a fixed 8 kHz and above; it does not cover damage starting below 8 kHz in certain codecs. The base term still covers dynamic detection regions, so it does not imply the model lacks supervision in any frequency band.
- Reference energy floor in relative loss, denominator in actual SC, time-differenced positive/negative variations, and the definition of positive flux are all different. It is essential to observe these under fixed frequency bands and dynamic regions simultaneously, preserving absolute dBFS of weak HF and relative error, and not arbitrarily adjusting the floor or removing counterexamples for scoring purposes.

The current detector identifies effective bandwidth, and cannot be used to prove content loss simply by a single energy boundary. Natural low-band music and actual high-frequency loss must be distinguished in both training tasks and reporting. Identity control is essential, but adding a batch of identity samples has not yet been shown to resolve the issue of generation quantity estimation.

## Data and Budget: Quantity Cannot Replace Effective Coverage

Re-examining the actual list: 253 music works, 22 artist IDs. The first five artist IDs account for **149 tracks (58.9%)**. A total of **71,327 music training pairs** have been prepared, not just 253 short clips. These entries include variants such as codec, sample rate, and channel role, and cannot be treated as equivalent in number or as independent music patterns.

Of the 253 tracks, **171** have less than 1e-4 total 16 kHz power. This does not prove pseudo-lossless behavior, nor does it imply that no high-frequency detail is recoverable per codec; natural weak high-frequency content must be preserved as a correct non-generation control. Effective high-frequency, harmonic, and transient coverage must be statistically analyzed and tied to specific damage types before adjusting internal music sampling. The music/voice ratio of 253/254, 1/254, remains unchanged; any deviation must be formally registered.

For the first seed P2, 4096 steps resulted in 16,384 training snippets and 14,286 distinct row entries. When including repetitions and different codec views, total supervision time was **145.2 minutes**. For P4, each 1024-step group has a baseline supervision time of **36.35 minutes**, with 12.5% allocated to identity controls. This is not equivalent to independent audio duration or a complete data traversal ratio. It cannot be concluded that the architecture has reached its performance ceiling based on a few rounds of short continuation training.

## Correction of Recent Experiment Sequence

Prior work demonstrated correct execution via numerical, exported, and real-time checks. These checks do not prove that the learning task has been designed well. P3/P4 have continuously focused on weak-band power, peak, and identity constraints, forming a local patching loop. The [original architecture reset plan](architecture-reset-20261011.md), which included long-window features, band-specific conditioning, fidelity/generation separation, and texture training, has not been implemented or sufficiently validated by the current short-window TCN approach. Therefore, the negative results from these continuation runs should not be extended to the claim that "better representations and structures are also ineffective."

Evaluation must also correct overinterpretation: a degradation of SI-SDR by **0.003 dB** in the prior round is not sufficient to reject the model outright without blind listening. The failure to advance P4 is not due to a mechanical requirement for every metric to improve. The lack of overall recovery gain—unimproved spectral shape/transient fidelity, zero-crossing average energy gain, and persistent development counterexamples—justifies the non-promotion. Future work should establish a primary metric, layered absolute error bounds, and fidelity tolerance ranges in advance, followed by fixed perceptual evaluation. Post-hoc standard adjustments are prohibited.

## Next Minimum Discrimination Experiment Will Not Be Auto-Triggered

1. **First, verify whether magnitude envelope generalization is possible.** On the same small budget and same data split, compare the current short-window condition against the past-only 1024/2048 resolution condition. Isolate the spectral shape and envelope tasks. The past long-window features should be treated as available input features only, not as a requirement for future prediction. Winning this test should precede full complex output comparison.
2. **Then, investigate the source of target conflicts.** Fix the winning representation and separately evaluate reliable harmonic amplitude and phase, versus uncertain high-frequency textures. Compare the actual synthesized output—spectral shape, transient, and waveform—not just total band power. Do not directly enable GANs, expand model size, or conduct long training to cover unlocalized issues.
3. **Synchronize the establishment of an effective data and evaluation loop.** Statistically track effective high-frequency duration, native narrowband controls, sources, and transient coverage. Freeze the primary evaluation region and tolerance range. Existing development and historical held-out data remain marked as reused. A new set of tests, isolated by work and source, will be prepared before candidate promotion and binding to a SHA for long-term real-time validation.

No changes were made to the default model, DSP, loss, or training sampling. No new training was initiated. Diagnostic scripts and per-fragment evidence are stored locally at:
`data/reports/root-cause-20261012/`
including: `audit.py`, `registration.json`, `diagnostics.json`, `bounds.py`, `bounds.json`. The first seed, short snippets, and post hoc diagnostic boundaries are preserved.

The original residual range was `±2 * input_RMS_spectrum`; this bound concerns representation and does not establish generalization.
