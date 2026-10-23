# Model and DSP Responsibility Boundaries: Survey and P36 Audit

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

Date: 2026-10-22. Scope: Low-latency, lossy audio restoration; speech denoising, ideal low-pass super-resolution. Instrument synthesis provides only transferable mechanisms and does not constitute performance validation for the same task. This document combines academic papers, official code, and local frozen experiments; no third-party weights or training data were imported, and no new deployment or training was initiated.

## Decision and Strategy Adjustment

The model should learn the spectral, fine details, and temporal dynamics of original lossless audio, and should not be forced to learn to compensate for any post-processing attenuation, phase bias, or clipping. User concerns about responsibility boundaries are valid. However, the statement "no DSP should enter training" is overly broad: analysis/synthesis transforms themselves are DSP; differentiable, correct synthesis enables loss evaluation of truly playable outputs; parametric synthesizers or filters can also be intentionally chosen as model architecture.

A more appropriate constraint for SoundEx is: **maintain direct, lossless supervision with fixed and accurate analysis/synthesis during training; protective post-processing should not become a recovery target or teacher; first verify that fusion does not systematically distort correct predictions before deciding whether to optimize through it.**

This adjusts the priority order proposed in the previous [dynamic diagnosis](risk-temporal-diagnosis.md): temporary suspension of introducing a full deployment chain with dynamic loss. First, audit and reduce conflicts between fusion and lossless recovery. Fixing DSP parameters does not isolate gradients; as long as backpropagation passes through `DSP(model(x))`, the model may still learn inverse compensation.

## Literature and Official Implementations

| Work | Model and DSP Division | Relevance and Boundary Implications |
|---|---|---|
| [AudioSR，2023](https://arxiv.org/html/2309.07314v1) | Diffusion model estimates high-resolution mel, vocoder synthesis, then separately replaces low-frequency mel and waveform STFT content | Post-processing protection of existing information has precedent; the paper relies on the assumption of identical low-frequency content and uses full roll-off, which cannot be directly adopted for lossy encoding recovery or causal detection |
| [AERO，ICASSP 2023](https://arxiv.org/abs/2211.12232) | Directly predicts full complex spectrum, outputs via iSTFT, with training in frequency, time, and perceptual domains | Demonstrates that full-spectrum recovery is a valid path; low/high-frequency stitching is not essential; its resampling task and full-segment normalization cannot directly satisfy our streaming contract |
| [Apollo，2024](https://arxiv.org/html/2409.08514v1) | Targets encoding damage, models per-band, directly reconstructs complex spectrum, then iSTFT; jointly trains GAN and reconstruction objectives | More closely aligned with music de-compression than simple speech denoising; draws on band/time modeling and direct complex synthesis, without adopting scale, training data, or real-time claims |
| [RNNoise，2018](https://arxiv.org/abs/1709.08243) | Network estimates band gains, supplemented by traditional pitch filter for harmonic suppression | Neural and DSP fusion works effectively, but the goal is noise suppression—this does not imply that gain filtering is sufficient to recover deleted high-frequency music content |
| [DeepFilterNet，2022](https://arxiv.org/abs/2110.05588) | ERB gain estimation and multi-frame complex filtering form an enhancement model | Filtering is an explicit model structure, not a model that first outputs correct signals and then gets arbitrarily rewritten; speech tasks, future-frame options, and full-segment RTF cannot replace our hop-by-hop validation |
| [DDSP，ICLR 2020](https://arxiv.org/abs/2001.04643) | Network outputs parameters, harmonic/noise synthesizers as differentiable DSP generate audio, with spectral loss supervising audio | Learning to control DSP itself is not erroneous; however, this defines a different output, not direct recovery of full complex spectrum; strong instrument priors are unproven for complex mixes |
| [Differentiable Consistency Constraints, ICASSP 2019](https://research.google/pubs/differentiable-consistency-constraints-for-improved-deep-speech-enhancement/) | Projects STFT consistency into the network | Requires evaluation of achievable waveform spectra; cannot infer that "a heuristic post-processing must match," nor can mixed source conservation constraints be applied to missing information generation |

Official code further validation:

- [AudioSR postprocessing](https://github.com/haoheliu/versatile_audio_super_resolution/blob/main/audiosr/latent_diffusion/models/ddpm.py)
  First performs low-frequency mel replacement, then replaces input/output STFT low frequencies and adjusts energy ratios; full-segment peak normalization is unsuitable for direct porting into our real-time stream. The `ground truth` comment in the code refers to observed low-pass input, not a recoverable reference during inference.
- [AERO forward](https://github.com/slp-rl/aero/blob/main/src/models/aero.py)
  Converts real/imaginary channels, outputs complex spectrum, then synthesizes; mean/variance computed over full segment features. It supports research into direct spectral reconstruction, but does not prove causal implementation.
- [Apollo implementation](https://github.com/JusperLee/Apollo/blob/main/look2hear/models/apollo.py)
  Outputs real/imaginary subbands, then concatenates and iSTFT. The paper discusses the possibility of causality, but the current implementation defaults to `causal=False`, centered padding, and STFT without explicit centering closure. This default does not satisfy our past/future frame constraints. The paper uses 20 ms window, 10 ms hop; the reported speed is for processing 1 second of audio on GPU, and cannot be converted into evidence for our 128-frame CPU per hop requirement.
- [DDSP official example](https://github.com/magenta/ddsp) explicitly demonstrates differentiable synthesizers with multi-scale spectral loss; the [DeepFilterNet official implementation](https://github.com/Rikorose/DeepFilterNet) includes an optional postfilter.
  "DSP within the structure" and "additional post-processing" are not equivalent in these engineering implementations.

Additional recent work reviewed:
[TVF, 2026-03](https://arxiv.org/html/2603.02794v1) uses a ~1M parameter network controlling 35 IIR filters—conceptual speech denoising, with reported inference latency of 21 ms;
[Adaptive Spectral Loss, 2026-08 preprint](https://arxiv.org/html/2608.30739v1) discusses unreliable phase supervision leading to high-frequency amplitude suppression. This mechanism is worth investigating, but is not the proven root cause of SoundEx, nor constitutes justification for adopting its weights. Its amplitude power compression differs from P36's per-sample `log1p(loss)`.

None of the above constitutes a unified SOTA ranking: tasks, datasets, latency, listening evaluation, and hardware differ. Web pages and default branch code are referenced as of the current access state; no external implementations or weights were incorporated into the repository.

## What Does Current P36 Actually Learn?

Code entry points:
`training/scripts/run_risk_attribution_study.py::training_terms`,
`training/models/direct_spectrum_objective.py`, `multires_spectrum.py`, `budget_objective.py`, `envelope_probe.py`, and `restoration_head.py`.

1. The paired direct target is the original lossless magnitude, complex spectrum, envelope, and dynamics; the multi-resolution (MR) target is the prediction after fixed OLA, compared against the original unclipped lossless reference via multi-resolution spectral comparison. All frequency bins are supervised, and supervision is not canceled due to deployment masking.
2. An additional 0.1x budget term targets the complex residual energy between reference and observation—not the output of harmonic DSP. P36 does not invoke the old `balanced_risk` deployment reconstruction term to optimize the model; the existence of a deprecated function in the same file does not imply current usage.
3. P36 is a new initialization training run; historical deployment loss weights cannot be inherited as evidence of degradation.
4. Model inputs indeed include **sample rate, detection cutoff, and active flags**. Thus, it is not a mapping entirely disconnected from DSP: it can leverage detector statistics. Effective bandwidth is a reasonable degradation condition, but `active` is a strategic decision; harmful dependencies require separate ablation and cannot be inferred solely from input presence.
5. Actual P36 uses Neural, Model gain, Model phase, and smooth-knee limiting. Traditional harmonic texture Hybrid mixing and MatchEdge edge loudness matching are not enabled. It still passes through detection masking, amplitude crossover, unit-circle phase interpolation, OLA, wet/dry gating, and limiting.

Therefore, the claim that "the current primary loss is teaching the model to mimic traditional DSP boosting" does not align with current code. The assertion that "predictions are being adjusted by deployment rules, future deployment loss may induce compensation, and input dependencies on detectors" has clear basis.

## Confirmed Fusion Conflicts

### Magnitude: Correct Predictions Are Pulled Back to Compressed Input

Current per-bin magnitude rule is:
`A_out = (1 − w) A_input + w A_model`.
When `A_model = A_ref`, we have `A_out − A_ref = (1 − w)(A_input − A_ref)`.
Only when `w=1` or input equals reference is the correct magnitude preserved.

Single-frequency point validation using current Python implementation: input magnitude 0.1, target 0.2, `w=0.25`.
When the model correctly outputs 0.2, the final magnitude is only 0.125. To achieve a final output of 0.2, the model must output **0.5**.
The gradient of the final magnitude squared loss with respect to prediction dB is `−8.63e-4`, still pushing the model to increase magnitude. In general, the required prediction is `A_input + (A_ref − A_input)/w`. At small weights, overcompensation is easily induced; when `w=0`, the model has no correction capability.

This is a local mathematical conflict—not evidence that P36 has learned such compensation. The direct objective of P36 is to pull predictions toward 0.2, and no such final loss is currently included. A simple linear blend `X + w(P − X)` still suffers attenuation, and cannot be claimed to resolve simply by changing the interpolation coordinate.

### Phase: PhaseSmoother Is Not a Time-Averaging Filter

`crates/soundex-dsp/src/phase.rs` only interpolates unit vectors between frames, with no inter-frame state. Even when observed magnitude is weak, the phase still participates as a unit vector. This does not guarantee STFT temporal consistency, nor is it a calibrated phase reliability estimate.

For `0 < w < 0.5`, the maximum angular deviation after unit circle mixing is `asin(w/(1−w))`. Numerical scan at `w=0.25` yields **19.4712°**, consistent with the formula.
Even if the model can output arbitrary phase, the final phase at a single bin may still fail to reach the reference phase. This local constraint does not equate to a strict upper bound on OLA waveform performance—validation with real samples across stages is still required.

### How Much Error Remains After Perfect Prediction, Then Real Rust Processing?

For all 89 reused segments, in paired and identity roles, the correct reference spectrum replaces model output, and original input detection, fusion, gating, limiting, and shared supervision intervals are preserved. A total of **178** native predictions were replayed; this uses an undeployable lossless reference to diagnose the processing chain, not to evaluate learning generalization.

Common MR v1, aggregated by recording, with reference always being the original unclipped audio:

| Stage | Music Paired | Speech Paired |
|---|---:|---:|
| After perfect spectrum via original OLA | 1.12e-8 | 6.42e-9 |
| After detection-controlled fusion | 0.219961 | 0.236473 |
| After gating, unclipped | 0.219961 | 0.236481 |
| Final Rust output | **0.220684** | **0.236481** |

For music identity, perfect prediction error after fusion is ~1.20e-8, and after limiting is 0.006973—indicating that protective limiting and the original reference are not always aligned.
For music paired, the native relative RMS to original reference is 0.0123593; for speech, 0.00339318—both are per-recording RMS averages, not evaluated for listening quality or peak clipping.

The retained input region still contains encoding errors, so these residuals cannot be fully attributed to "new DSP artifacts."
Dividing 0.2207 by a candidate score does not represent DSP error contribution—MR is not additive, and causal decomposition is not valid.
This proves: **current fusion is not a mapping that preserves correct recovery results**—there is a conflict between the full, original lossless target and the strategy of preserving observed regions.

Residual magnitude errors in music across 4–8, 8–12, 12–16, and above 16 kHz bands are 0.2150, 0.2257, 0.1104, and 0.0317, respectively.
The issue cannot be explained solely as insufficient high-frequency generation. For encoding damage, "reliable below cutoff" does not hold.

The first proxy using Python input features detected a difference of `2.70e-5` in the 11th paired segment—exceeding the original `2e-5` tolerance—triggering a stop and record. After switching to actual Rust-observed spectrum, full replay maximum difference is **2.98e-7**, unchanged from tolerance. This confirms that proxy validation must include front-end features, not just identical formulas. Failures must be recorded, not passed off as success.

## Recommended Responsibility Boundaries

| Component | Recommendation |
|---|---|
| Model | Estimate original lossless complex spectrum or a well-defined residual; learn spectral shape, detail, envelope, and reference temporal dynamics |
| Fixed STFT/iSTFT/OLA | Retain in training reconstruction chain; supervise the spectrum of a truly realizable waveform, verifying perfect reconstruction and temporal alignment |
| Retained bands/fusion | As a signal strategy with evidence-based justification; not as a lossless teacher; avoid full-spectrum reshaping of already-correctly predicted magnitude/phase |
| Cutoff / active conditions | As a pending item for audit; sample rate preserved; cutoff and active should undergo independent sensitivity and budget-conditioned training comparisons |
| Safe limiting, fault fallback, mode fade-in/fade-out | Independent protection and state management; preserve native validation; do not require the main reconstruction objective to reverse-limit or predict fault states |
| Traditional boosting/peak matching | P36 is not enabled; if future evaluation is conducted, maintain explicit, toggleable independent controls—do not use its output to replace lossless supervision |

Phase is part of the model's reconstruction—cannot be simply handed off to post-processing. Missing-band input phase may lack reference value.
Direct iSTFT is not "the model learning DSP side effects"—it is a fixed physical implementation.
User-adjustable mixing strength, if intentionally reduced, is a product design trade-off; evaluation must fix intensity.
Reversing the model to cancel such controls undermines the meaning of different intensity settings.

## Recommended Experimental Sequence

Step 1 has been completed in [P37 Frozen Fusion Audit](fusion-boundary-audit.md): magnitude remixing shows a small, measurable cost; removing all gating yields no significant gain in paired performance, and cannot replace model learning improvements.
New training has not been initiated; full results, interface decisions, and current environment blockers are detailed in this report.

1. **First, freeze the model and audit DSP.** Retain current control, compare direct model synthesis, only safe processing, and separately retain magnitude/phase fusion diagnostic paths. Add zero-residue, perfect reference, weak magnitude/phase, real transients, cutoff changes, and stereo validation. Bare model full-band output is only a baseline; without validation of continuity and retention of bands, it cannot be considered a default deployment.
2. **Define the recovery interface.** Clarify whether it is full-spectrum output or a clearly defined residual repair. Avoid full-spectrum re-scaling, nor secretly modify the prediction target to the inverse of DSP spectrum. If bands are retained, the trade-offs between residual error and recovery gain must be publicly disclosed. Hard-cut bands may introduce seams or ringing; removing crossover cannot replace validation. All diagnostic interventions must follow causal, unified rules and not be tuned per track.
3. **Then conduct small-model training comparison.** Use P36 compressed and identity 1.0 as control; prioritize comparing original lossless spectrum, dynamic supervision, and reduced detector condition dependency. Isolate one factor at a time, with identical initialization, sample rate, and sufficient budget. No direct "full-limiting chain backprop" target is introduced.
4. **Add constrained deployment only if necessary.** First verify whether gradients still push toward erroneous compensation at correct reference points, and whether they conflict with raw target. Limiting/fault strategies should not backprop gradients to the main recovery path; simple `detach` of the full deployment output loses training signal and cannot be used to claim optimization of final recovery. Any gradient stop or substitution must explain differences from actual deployment. Native final quality must always be verified—promotion cannot be based solely on raw output improvement.

All research continues to satisfy model plus DSP computation <5 ms, and 48 kHz/hop128 with 2.667 ms sustained throughput. A fixed 256-frame buffer is maintained.
Frequency-band shared time modeling can be inspired by Apollo, but external model CPU benchmarking was not run—no claim is made that any referenced architecture meets the requirement.
The existing validation set is for localization, not proof of absence of overfitting; after candidate stability, independent recording and blind listening are still required.

## Local Evidence

`data/runs/dsp-boundary-p36-20261022/` (ignore directory):

- `oracle.py`, `native-observed/registration.json`: all segments, parent experiments, binary, replay contracts, script hashes; execution depends on frozen P36 source code.
- `native-observed/report.json`, `band-summary.json`: all stages, per-segment scoring, PCM absolute error, and band-level summaries—no counterexamples removed.
- `single-bin.json`: algebraic validation of magnitude compensation gradient and phase expressibility range.
- `oracle-initial.py`, `registration.json`, `initial-attempt.json`: evidence of early proxy failure.

Recheck entry: run `oracle.py` from the repository root using `training/.venv/bin/python`, environment set to `PYTHONPATH=data/runs/risk-attribution-p36-20261022/source/training` and `OMP_NUM_THREADS=1`. This is a diagnostic native replay, not a test of ORT model capability, production candidate, device callback stability, or new training results.

For the unit-circle fusion bound, the stated domain is `0<w<0.5`. Perfect music-identity prediction has fusion error about `1.20e-8`, increasing to `0.006973` after limiting. Native paired PCM RMS versus original references averages `0.0123593` for music and `0.00339318` for speech. These quantities do not establish audible severity or a deployable oracle.
