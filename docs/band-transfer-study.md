# P15: Band-Specific Intervention with Frozen Weights

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

## Conclusion

The error amplification observed in severe counterexamples from P14 can be further localized to crosstalk between adjacent frequency bands.
Two blues-type counterexamples are primarily driven by corrections in the 12–16 kHz band causing output changes above 16 kHz;
a third, unlabeled counterexample is mainly influenced by corrections in the 4–8 kHz band affecting the 8–12 kHz range.
This is *not* a problem that can be resolved simply by disabling generation above 16 kHz.

## Protocol and Verification

We froze the step-2048 weights from the P13 multi-location group, retaining all 320 training positions and 89 development snippets,
resulting in 818 paired cases. For each pair, we performed 5 frequency band interventions:
- `only`: retain only one specific frequency band's correction
- `without`: remove the correction of one specific frequency band
All other conditions were held constant.

The final error was computed using the original DSP pipeline and the P13 objective function, without altering the reference audio or evaluation weights.

We also decomposed the complex spectral residuals after mixing, by source frequency band, and then re-analyzed them using the same OLA (Overlap-Add) and processing chain to generate a residual RMS matrix mapping from source frequency band to output frequency band. This matrix was computed prior to clipping and gating; the power of individual components cannot be directly summed due to potential phase cancellation. The resulting `only` and `without` outcomes constitute a separate, direct line of evidence.

Component-summed error:
- Waveform maximum: `1.862645149230957e-8`
- Complex spectrum maximum: `1.6858739115832577e-7`

For the three worst-performing music snippets from P14, we selected the dominant source bands and performed real-time Rust playback for both `only` and `without` conditions:
- 24 runs completed successfully
- Maximum error between proxy and native waveforms: `3.5762786865234375e-7`
This remains a mechanism-level diagnosis, not an independent quality assessment or a model validation for deployment.

Source code was frozen at commit `1af287b3c25b9a0957c2d9e08daa468d9035e56c`, and the report hash was verified to match.
Runtime directory: `data/runs/band-transfer-p15-20261013/`
All generated outputs were ignored and not pushed to audio storage.

## Direct Intervention Results

Severe blues counterexample `9d95e7f035f878fa1a6ab421`:

| Seed | Identity: All Corrections | Only 12–16 kHz Correction | Remove 12–16 kHz Correction |
| --- | ---: | ---: | ---: |
| 20261012 | 2.273816 | 2.272545 | 0.043032 |
| 20261013 | 2.087431 | 2.088101 | 0.025600 |

The residual RMS in the 16 kHz+ output region induced by the 12–16 kHz correction is approximately
52% and 79% of the RMS produced directly by a 16 kHz+ correction—this is a component-wise RMS ratio, not a directly additive error proportion.

However, disabling the 12–16 kHz correction also results in a loss of recovery benefit:

| Music Training Position / Seed | Original Pairwise Recovery Loss | Recovery Loss with 12–16 kHz Correction Disabled |
| --- | ---: | ---: |
| 20261012 | 0.572272 | 0.578455 |
| 20261013 | 0.572836 | 0.578190 |

While the average performance on development music improves slightly, it still remains above the baseline model. This rule was discovered only after observing the development counterexamples.
It cannot be used as a production toggle or treated as a direct improvement to the model. No modifications to DSP or weights were made in this round.

## Next Steps

Initiate [P16: Reference Perceptual Synthesis Consistency Benchmark](spectral-consistency-study.md):
During training, enforce consistency between predicted and reference outputs in synthesis, preserving the true temporal dynamics of the reference.
Continue evaluating gains using existing metrics: pairwise recovery, identity, transient, low-frequency, and real DSP performance.
This is a pending auxiliary constraint and has not yet been proven to yield effective fixes.
