# Restoration evaluation protocol v2

Revised on 2026-11-16 after [P81's metric audit](evaluation-audit-p81.md).
The P71 measurement definitions and P72 rules frozen on 2026-11-04 remain
reproducible in `training/evaluation/restoration_panel.py`, version
`soundex-restoration-panel-v1`. The additive implementation is
`training/evaluation/restoration_evidence.py`, version
`soundex-restoration-evidence-v2`. New evaluations use its extended panel and
strict pairing. Historical numerical scores, selection rules and decisions are
not rewritten; retrospective interpretation must be labeled as such.

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
| Shape independent of gain | Squared Hellinger distance between normalized frame power spectra; reference frame-energy weighting, separate from energy error |
| Energy envelope | L2 distance between frame band RMS envelopes, relative to reference envelope norm and in absolute PCM RMS units |
| Real temporal changes | Both adjacent-frame envelope errors and signed bin-magnitude change errors; relative and absolute units, without target smoothing |
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
squared distance between normalized vectors: **squared Hellinger**, not its square
root. The historical `*_shape` fields retain their numbers and names.
A completely missing predicted frame receives shape distance one if its reference
has energy. This ruler can
recognize shape while ignoring gain, so it must be paired with energy metrics.
The reference frame-energy weights intentionally emphasize energetic events.

V2 adds `*_spectral_dynamics`: the norm of
`diff(abs(candidate) - abs(reference), frame_axis)`, divided by the complete
reference magnitude norm. Its absolute companion is the RMS of those calibrated
bin changes. This detects frequency redistribution inside a constant-energy band
that `*_dynamics` (band envelope only) can miss. The existing training MR dynamics
already measures signed bin-magnitude changes at its own resolutions; no training
loss or weight changes here. Both diagnostics are differences per analysis hop,
not derivatives per second. Report each resolution's hop duration and scores;
the aggregate is a fixed multiscale diagnostic, not a physical temporal rate.

Periodic Hann windows have zero weight at the first sample, and complete-window
analysis can leave an unused tail. V2 reports that coverage and complete-support
`full_pcm_error_rms`. Preserve the existing whole-PCM low/HF, boundary and peak
errors as well. Zero STFT error does not establish zero error at every sample.
No padding, alignment search or target modification is introduced.

Numerically absent reference norms yield `null` relative scores, not artificial
zero errors: denominator limits are `1e-12` for norms and `1e-24` for power.
Absolute errors are always retained. These are numerical limits, not audibility
thresholds. A source with an undefined excerpt is excluded from that relative
metric's mean with its identity disclosed; its absolute error remains included.
No weak-band exemption is fitted to known failures.

## Aggregation, uncertainty and prospective decisions

Pair candidate/control by recording and exact case first. Reject missing or
duplicate case coverage. V2 also rejects differing metric schemas and every
nonfinite score, including a nonfinite value opposite an undefined relative
value. Never take the intersection of available metric fields: that can silently
remove a failed/missing ruler. Explicitly stratify physical mono/stereo before
mid/side comparisons, or explicitly declare undefined optional fields.
Average each recording's cases before taking an equal
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

### Prospective interpretation after P81

Keep three questions separate: whether the measurement is valid, which dimensions
change on this panel, and whether the result is useful enough to advance or ship.
`describe_comparison` reports MR/HF SC interval directions separately from shape,
energy, dynamics, coherent error and preserved-band/boundary errors. Below-zero,
above-zero, crossing-zero, exactly unchanged, missing, undefined, invalid and
single-source evidence are distinct states. Directional labels are descriptive;
they are not multiplicity-adjusted hypothesis tests or practical effect thresholds.

Do not infer a metric error bound from `8 * float32_epsilon * abs(control_mean)`.
The P77--P80 rule used that heuristic, not an analytic bound on normalized metric
error or a calibrated audibility threshold. FP64 scoring can resolve actual FP32
PCM changes smaller than that ruler. Establish numerical resolution with analytic
answers, score reproducibility and independently registered native repeatability
controls when necessary. Proxy/native discrepancy concerns cross-implementation
correspondence; it is not automatically uncertainty in native-versus-native scores.
Bootstrap intervals concern recording composition, not floating-point error.

Primary spectral progress is not erased by a small coherent/waveform increase.
Report that increase and its absolute PCM scale as a tradeoff; do not silently
declare it safe. Likewise a coherent improvement that worsens restored spectra
is not a restoration win. No universal phase-sign veto, newly fitted scalar
weighting or retrospective noninferiority margin is introduced. Any future
practical preservation budget requires its own prospective engineering/listening
rationale, independent of known candidate counterexamples.

Two-update derivative diagnostics need correct-reference/weak-signal tests,
connected gradients, matched finite optimizer controls, target learning and native
correspondence. They do not have to demonstrate a full-training effect, and cannot
reject an architecture's capacity. A changed recipe can proceed to a separately
registered adequate-budget comparison when its mechanism and feasibility checks
justify it; finish every matched trajectory before evaluating quality. A review
with uncertain directions is not a reason to repeat unchanged tiny updates.
The P72 full-training comparison remains the historical example above, not an
automatic gate imported into every diagnostic. Default promotion remains separate.

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
