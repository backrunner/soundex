# Restoration evaluation protocol v1

Frozen for prospective P72 comparisons on 2026-11-04. P71 calibrates this protocol
on previously exposed P70 material; it does not retroactively change P69's result.
The implementation is `training/evaluation/restoration_panel.py`, version
`soundex-restoration-panel-v1`. Historical metrics remain reported alongside it.

## References and comparison paths

The original aligned lossless PCM is the reference. Use the same recordings,
source intervals, physical channels, codec settings, gain and DSP policy for
untreated input, identity DSP, historical legacy, matched small control and larger
candidate. Keep raw reconstruction and native model-plus-DSP output separate.
Clean identity roles are separate from compressed-input restoration roles.

Freeze endpoints, hashes, the candidate rule and the complete evaluation panel
before scoring. Finish every registered seed and capacity. A reused recording,
different codec or longer excerpt is a development recheck. New independent
evaluation requires source reservations and connected file/PCM/work-alias exclusion
before training. Never choose clips, offsets or metrics to favor an endpoint.

## Spectral and fidelity panel

| Dimension | Ruler and interpretation |
| --- | --- |
| Historical spectral objective | Existing fixed MR, per-band components and auraloss; preserve continuity with prior work |
| High-band magnitude agreement | Independent spectral convergence at FFT sizes 512, 2048 and 8192, hop size one quarter of the window |
| Shape independent of gain | Hellinger distance between normalized frame power spectra; reference frame-energy weighting, separate from energy error |
| Energy envelope | L2 distance between frame band RMS envelopes, relative to reference envelope norm and in absolute PCM RMS units |
| Real temporal changes | Error between adjacent-frame envelope differences, normalized by reference envelope norm and also in absolute units; no target smoothing |
| Coherent reconstruction | Complex spectral distance, HF PCM error and existing waveform/boundary rulers; conflicts remain visible |
| Weak bands and preservation | Calibrated absolute errors beside relative metrics; historical low-band, identity and crossover checks retained |
| Stereo | Original simultaneous physical channels and separate mid/side rulers; never assemble stereo from unrelated channel excerpts |
| Perception | Blinded reference-based listening; optional separately pinned perceptual estimator, never substituted for listener results |

The independent panel reports >=8 kHz and 8–12, 12–16, 16–Nyquist bands separately.
Each resolution uses a periodic Hann window and one-sided normalization
`sqrt(sides / (FFT_size * sum(window**2)))`; band frame norms therefore have
window-energy-normalized PCM RMS units. Arithmetic means over the three
resolutions are reported, with every resolution retained. These offline analysis
windows do not add model lookahead or runtime buffering.

For shape, normalize each magnitude vector by its L2 norm and compute half the
squared distance between normalized vectors. A completely missing predicted
frame receives shape distance one if its reference has energy. This ruler can
recognize shape while ignoring gain, so it must be paired with energy metrics.
The reference frame-energy weights intentionally emphasize energetic events.

Numerically absent reference norms yield `null` relative scores, not artificial
zero errors: denominator limits are `1e-12` for norms and `1e-24` for power.
Absolute errors are always retained. These are numerical limits, not audibility
thresholds. A source with an undefined excerpt is excluded from that relative
metric's mean with its identity disclosed; its absolute error remains included.
No weak-band exemption is fitted to known failures.

## Aggregation, uncertainty and prospective decisions

Pair candidate/control by recording and exact case first. Reject missing or
duplicate case coverage. Average each recording's cases before taking an equal
recording mean; channels, codecs and excerpts are not independent observations.
Report paired mean differences with 2,000 recording-cluster bootstrap resamples,
fixed seed 71 and percentile 95% intervals. Retain every adverse source and
per-case metric, and separately report codec and paired/identity strata.

Intervals describe resampling this panel. They do not establish artist
independence, correct development-set selection bias, supply multiple-comparison
adjustment or prove generalization to all music. A mixed interval is inconclusive
for directional confidence on that ruler; do not relabel a tiny mean as robust.

For P72, a useful capacity direction requires lower source-equal **native MR and
HF spectral convergence in both seeds versus the matched small full-training
control**. Report interval support and shape, envelope, dynamics, absolute weak
errors, identity, preserved bands and stereo together. Conflicting rulers mean a
tradeoff, not an overall quality win. A small waveform regression alone does not
erase a spectral gain; conversely spectral improvement does not certify sound
quality. This is an experimental continuation criterion, not a new release gate.
Default promotion still needs broader quality, listening and runtime evidence.

## Calibration and perception

Before candidate scoring, check reference stationarity, silence/weak material,
gain-only changes, changed frequency with matched energy, shifted transients,
stereo polarity, correct/half-correct/muted high bands and energy-matched phase
scrambling. Retain conflicting behavior rather than adjusting weights until every
negative receives the desired rank. [P71](evaluation-capacity-p71.md) records
the first complete calibration and its limitations.

Perceptual metric performance depends on application domain, as examined in the
[audio quality metric review](https://arxiv.org/abs/2110.11438).
[ViSQOL's official guidance](https://github.com/google/visqol) requires 48 kHz in
audio mode and downmixes multichannel material. It may complement this panel after
pinning its executable/model and preprocessing, but cannot establish stereo
fidelity. No ViSQOL executable or ratings were available for P71; missing results
remain explicitly missing. Do not fabricate MOS from spectral distance.

Use the existing [listening protocol](../training/evaluation/LISTENING_PROTOCOL.md)
and blinded pack with shown/hidden references, untreated input, identical common
gain per trial and no model labels. Freeze randomized conditions before ratings;
retain all seeds, exclusions and listener/device details. Material preparation is
not completed listening evidence.
