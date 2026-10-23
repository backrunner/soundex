# High-Frequency Spectral Inpainting and Model/DSP Collaboration

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

## Objective and Evaluation Criteria

SoundEx uses the original lossless audio as the recovery reference: the model estimates the missing high-frequency spectral shape, details, energy envelope, and phase, while the DSP performs band fusion, loudness control, artifact suppression, and safe output. The final evaluation target is the audio after full Rust pipeline processing. Increasing high-frequency content or reducing waveform loss alone, without preserving the original compressed input, cannot be considered successful.

Bandwidth extension requires simultaneous handling of spectral shape, tonal characteristics, and time-domain envelope; inappropriate frequency filling may introduce background noise, hum, non-harmonic beats, and transient tailing. Principles and counterexamples are detailed in
[AudioLabs' BWE Tutorial](https://www.audiolabs-erlangen.de/content/resources/aesCodingTutorial/bwe.html).
The encoder/decoder methods described in this tutorial use high-frequency side information; SoundEx performs blind recovery on already-decoded PCM and does not have access to the original high-frequency side information, so it cannot claim exact reversal of compression.

The collaborative workflow between neural network predictions and DSP signal generation is inspired by
[Grumiaux & Lagrange's Music DDSP bandwidth extension research](https://arxiv.org/abs/2311.07363).
The current implementation is neither a reproduction of that model nor any standard codec.

## Objective v3

`training/configs/inpainting_continuation.yaml` is the new experimental configuration. V1/V2 formulas and historical checkpoints maintain readability, and direct comparison of total loss across versions is avoided. No old optimizer is reused for new objectives. The V3 loss terms are defined as follows, with all sample quality comparisons still requiring real-world deployment through the full processing chain:

| Item | Definition and Role | Initial Weight |
|---|---|---:|
| High-frequency dB amplitude | Within masked missing regions, L1 loss between predicted and reference dB after truncating relative to a common baseline | 1 / 20 |
| High-frequency spectral convergence | L2 loss of linear amplitude residuals, normalized against the reference high-frequency amplitude norm of the recording | 1 |
| High-frequency energy envelope | Absolute dB difference of per-frame average power in missing regions relative to reference | 0.5 / 20 |
| High-frequency temporal details | L2 loss of signed magnitude differences between adjacent frames of prediction and reference, normalized against the reference change norm | 0.25 |
| Post-reconstruction high-frequency | After frequency-domain fusion, phase interpolation, causal Hann OLA, and truncation to full support, re-analyze the spectrum and compare high-frequency spectral convergence | 0.5 |
| Effective high-frequency phase | Weighted by reference effective energy using `1 - cos(phase difference)` | 0.05 / 2 |
| Waveform fidelity | L1 loss over fully supported regions, relative to the dry reference error | 0.25 |
| Retained band | L1 loss of low-frequency prediction relative to input in dB | 0.1 / 20 |
| Band transition seam | Residual constraint on frequency variation in transition zones | 0.05 / 20 |

The common baseline for high-frequency dB is `max(reference per-frame peak - 80 dB, -120 dB)`. This prevents trivial gains from simply filling in noise by shifting the numerical baseline; linear spectral, energy, and waveform terms continue to penalize erroneous high-frequency predictions. Relative norms use a reference full-band RMS of 1% and an absolute `1e-4` floor. Energy comparisons use corresponding power floors. Scales are computed per recording and detached, so they cannot be manipulated by changing prediction values to inflate denominators. Absolute baselines are intentionally relaxed near quiet segments to break strict gain invariance, and such segments should be ignored.

The temporal term compares actual reference changes — **it does not require the output high-frequency signal to be constant**. Smearing or misaligning transients will still introduce errors. Training uses 8 causal frames, but each frame is independently fed into the generator. The inference input remains `[batch, 2, 1, 129]` and does not introduce runtime forward-looking or contextual buffering. No GAN is used; BatchNorm statistics are frozen, and affine parameters are trained to reduce interference from short-range initialization variations.

Initial weights were measured on 64 random samples from the native training set, not derived solely from loss values:

| Weighted Term | Average Parameter Gradient (L2) |
|---|---:|
| High-frequency amplitude / spectral convergence / energy / temporal | 1.676 / 3.060 / 0.806 / 0.711 |
| Post-reconstruction high-frequency | 2.618 |
| Phase / waveform | 1.639 / 1.124 |
| Retained band / seam | 0.170 / 0.084 |

This is a local diagnostic under MPS, parent model epoch 168, and frozen BN mode. Gradient directions may cancel, so the sum of these values cannot be interpreted as a "contribution ratio," nor does it guarantee optimal weights throughout training. Loss single-tests cover perfect recovery, missing high frequencies, overfilling, noise floor, transient smearing, phase/OLA artifacts, gain scaling, and limited gradients. True audio quality still requires independent testing and blind listening.

### Deployment Proxy Boundaries

The differentiable reconstruction pipeline includes frequency separation, phase interpolation, OLA, and re-analysis, but **excludes** real-time bandwidth detectors, loudness matching, limiters, wet/dry transitions, and asynchronous fallback. The transition boundary is derived from paired data measurements, while runtime detection is used at inference time. `reconstruction_high_spectral_convergence` is a differentiable proxy error and should not be equated with full Rust DSP error. Candidates must be validated on the same binary and same material, and results must be reported separately for music and speech.

## Model/DSP Comparison Path

`EnhancementMode::Neural` retains the existing default chain: detection → neural magnitude/phase → loudness matching → band/frequency fusion → OLA → limiter/wet/dry transition.

`Spectral` is a standalone blind DSP extension baseline that does not load or invoke the model: it only transposes the upper portion of the retained band, with a gradient fade at patch boundaries, frequency-dependent decay, and phase rotation correction based on analysis frame position. It adjusts generation volume based on donor band spectral flatness and cutoff frequency reduction, and follows input energy. It does not process DC/Nyquist and does not generate details in silence. It is not an accurate harmonic completion algorithm.

`Hybrid` retains model-derived phase and an upper bound of energy in the 750 Hz subband. The DSP provides candidate textures within subbands, with up to 25% power weight based on flatness confidence. Donor candidates are constrained by model subband energy, without increasing the total candidate power in any subband, and without modifying the retained band, transition zones, or Nyquist. This is a verifiable experimental path and has not yet been adopted as the default audio quality solution.

Both new paths reuse the existing STFT and do not introduce new FFT audio buffers or forward-looking delays. Fixed hop API Rust allocation tests, arbitrary chunking, mono channel, and reset tests have been covered. These paths still require listening and real device testing. Continuous output does not imply absence of audio artifacts.

```bash
# Baseline DSP without model
cargo run --locked --release -p soundex-cli -- input.mp3 \
  --enhancement-mode spectral --output target/spectral.wav --bits 24
# Default / combined path with same training weights
cargo run --locked --release -p soundex-cli -- input.mp3 \
  --model /path/to/candidate.onnx --enhancement-mode neural --output target/neural.wav --bits 24
cargo run --locked --release -p soundex-cli -- input.mp3 \
  --model /path/to/candidate.onnx --enhancement-mode hybrid --output target/hybrid.wav --bits 24
```

Experimental results and default solution decisions are documented in [this optimization record](optimization-followup-20261010.md).

A clear selection of validation rows is used to reproduce the three-path comparison:

```bash
cargo build --locked --release -p soundex-core --bin soundex-stream-eval
PYTHONPATH=training python -m evaluation.compare_extension \
  --model /path/to/candidate.onnx \
  --manifest /path/to/music/manifest.jsonl --manifest /path/to/speech/manifest.jsonl \
  --selection /path/to/validation-selection.json \
  --binary target/release/soundex-stream-eval --output target/extension-comparison.json
```

Selected files are `{"split":"validation","row_ids":["..."]}` and must be fixed before candidate scoring. The tool verifies validation members, training leakage, recipe consistency, and audio checksums, maintains identical selection, and outputs paired recording clustering bootstrap, as well as per-utterance/codec/sample rate/channel role/category results. Audio roles are mono; these results do not replace true stereo imaging tests. All reports are explicitly labeled as development diagnostics.

## Objective v4: Complex Reconstruction and Spectral Consistency

V4 retains the high-frequency recovery terms from V3 and introduces supervision of magnitude and phase jointly forming a complex spectrum. Let `C` be the candidate spectrum after band/phase fusion, `R` the lossless reference, `D` the compressed input, `P` the STFT after causal OLA and re-analysis, and `M` the missing region mask. Only overlapping windows with full support and time alignment are compared:

| New Term | Residual | Normalized Reference | Experimental Weight |
|---|---|---|---:|
| Post-reconstruction complex error | `sqrt(M) * (P(C) - P(R))` | Lossless high-frequency complex norm | 0.2 |
| Post-reconstruction retained band pollution | `sqrt(1-M) * (P(C) - P(D))` | Input retained band complex norm | 8.0 |
| High-frequency spectral consistency | `sqrt(M) * (P(C) - C)` | Lossless high-frequency complex norm | 0.1 |

Each residual is computed as per-recording L2 relative error; denominators use the corresponding band norm, 1% of full-band reference RMS, and a maximum of `1e-4`, all detached. The consistency term denominator comes from the lossless reference and cannot be scaled by prediction. The retained band term independently constrains pollution introduced by enhancement. Approaching lossless recovery is still guided by high-frequency, full waveform, and other reference terms, and cannot be achieved by suppressing fill-in to reduce retained band error.

The idea of complex spectral and re-extracted spectral consistency draws from
[AP-BWE paper's spectral training objective](https://arxiv.org/html/2401.06387v2#S3.SS2.SSS1).
This implementation uses SoundEx's causal window, missing region mask, and per-recording scaling, without porting the code, pre-trained weights, GAN discriminator, or performance claims from the paper. The original work focuses on speech tasks and does not imply verified music recovery.

V4 initial gradient audit uses the same 64-sample training set, V3 epoch 8, MPS, and frozen BN: new term gradients (L2) are `0.891 / 0.367 / 0.547`. The initial gradient for the retained band term (weight 1) is only ~0.046, so it is adjusted to 8 prior to training. This is a local-scale diagnostic and not a proof of optimal hyperparameter selection. Single tests cover accurate STFT recovery, phase damage-induced spectral inconsistency and low-frequency leakage, silence, empty missing regions, and limited gradients. High-frequency collapse is not rewarded.

V4 does not include online detection, real-time loudness matching, limiting, wet/dry transitions, or fallback. The gap between training proxy and actual Rust deployment must be measured empirically. The V4 proxy error cannot be equated with full DSP error.

## Cross-Stream Interaction Experiment

`cross_stream_interactions: true` adds bidirectional 1x1 projections between magnitude and phase branches at four encoding layers: the magnitude branch receives phase features and the phase branch receives magnitude features, then both proceed to subsequent encoding, bottleneck, and decoding. Projections are initialized to zero, so when migrating from parent models, initial outputs are preserved exactly. Subsequent training learns the interaction. Constructing zero projections does not consume additional CPU RNG, avoiding changes to training clipping and base random parameters.

This variant is model structure `1.1`, with parameters increasing from 806,276 to 848,900 (+5.29%). The runtime contract remains `[batch, 2, 1, 129]`, with per-frame independence. No time convolution, state, forward-looking, or buffering is added. Actual runtime cost must be re-measured; parameter growth cannot be used to infer latency.

`consistent_continuation.yaml` and `interacting_continuation.yaml` use the same V4 objective, with 6 rounds / 768 updates, frozen BN, no GAN, identical seed/data/sample rate/training budget, isolating the effect of the new objective and additional branch interaction. Sampling maintains the established music/voice ratio, and validation reports are provided separately for each material type.

Interaction migration must be explicitly declared; all other structural, data, or feature contract differences are rejected:

```bash
# First parse the profile's data path and sampling configuration into existing native data configuration.
PYTHONPATH=training python training/train.py --config /path/to/resolved-interacting.yaml \
  --initialize-generator-from /path/to/parent.pth --initialize-zero-interactions
```

Only the parent generator is migrated; new projections must be initialized to zero, and the optimizer/discriminator must be reinitialized. Structural or objective changes cannot resume with the old optimizer. Results are documented in [next optimization record](optimization-20261011.md).

## High-Frequency Level Deployment Comparison

`HighBandGain::MatchEdge` retains the default cutoff edge matching (target high-frequency to edge RMS ratio of -6 dB). Experimental `HighBandGain::Model` directly retains the model-predicted level, still undergoing band, phase, limiting, and wet/dry transitions. The Spectral mode does not use neural candidate level matching. This is a deliberate control option and has not yet been proven to recover more closely to lossless, so it does not replace the default.

CLI uses `--high-band-gain model`, Rust bridge uses `high_band_gain="model"`, and reporting is tied to the selected strategy. Legacy reports can only be read under the existing `match-edge` default. Level control results and limitations are detailed in [next optimization record](optimization-20261011.md).

## Objective v5: Reconstructed Energy and Phase Change

V4 improves on some spectral and transient metrics, but actual deployed high-frequency energy may still be insufficient. V5 adds two complementary constraints: energy measured after causal OLA, and phase change relative to the lossless reference. Historical V1–V4 formulas remain unchanged; when comparing parent model to new candidate, scores are uniformly evaluated using V5.

| Item | Definition | Weighting and Scaling |
|---|---|---:|
| Reconstructed high-frequency energy | Average power in missing regions after full support STFT, relative to lossless reference in absolute dB difference | Averaged with next term, then multiplied by `1 / 20` |
| Reconstructed subband energy | Every 8 frequency bins as a subband, average across frames, then compare absolute dB difference; only count effective subbands | Same as above |
| Phase frequency change | Difference in phase difference between adjacent frequency bins, using `1-cos` circular error | `0.0025 / 2` |
| Phase time change | Difference in phase difference between adjacent frames, using `1-cos` circular error | `0.025 / 2` |

The reference baseline for reconstructed energy is the square of 0.01² of the lossless full-band average power and the square of `1e-4` with a maximum, and is detached. Subbands with fewer than 8 points are padded with zero-weight. Phase change uses the minimum of adjacent effective reference phase weights, and does not penalize actual transient or phase changes present in the lossless reference. V5 increases the weight of the original high-frequency per-frame energy term from 0.5 to 1.0; all other V4 terms are retained.

The phase change constraint concept draws from
[AP-BWE's phase training objective](https://arxiv.org/html/2401.06387v2). Here, only the principle is adopted, and the implementation uses SoundEx's effective high-frequency mask, causal frames, and scaling — no code or weights from the project are introduced. The training proxy does not include online cutoff detection, edge-level matching, limiting, wet/dry transitions, or asynchronous fallback. V5 values cannot replace actual Rust output audio quality and continuity validation.

After auditing on parent model and 64 training set samples, new high-frequency/subband energy gradient (L2) values are 2.196 / 2.045, and phase frequency/time terms are 2.142 / 1.385. The initial frequency term was previously 21.421 and has been reduced prior to training. These are local gradient diagnostics and do not prove optimal hyperparameter selection.

## Circular Phase Input Experiment

`circular_phase_features: true` expands the first layer of the phase branch input from raw phase to `[phase, sin(phase), cos(phase)]`. The original channel is preserved, and new channels have zero-initialized convolution weights; explicit parent weights are migrated and then gradually learned. Raw phase is retained, so the entire network does not guarantee strict `2π` input invariance. Single tests verify gradient availability, initial output migration consistency within tolerance, base RNG invariance, and no cross-frame or future-frame dependencies.

Structure version is `1.2`, with 806,420 parameters — a 144 increase (0.018%). Inference contract remains `[batch, 2, 1, 129]`, with no new forward-looking, context buffering, or runtime state. The V5 baseline structure is used for comparison with the same V4 parent weights, data split, sampling, seed, and 8 rounds / 1,024 updates. Validation and actual cost are measured separately and cannot be inferred from parameter count.

```bash
# First parse data and sampling into existing native data configuration; only migrate generator.
PYTHONPATH=training python training/train.py --config /path/to/resolved-circular.yaml \
  --initialize-generator-from /path/to/parent.pth --initialize-phase-features
```

`--initialize-phase-features` allows only the first layer to expand from 1 to 3 channels; all other configurations and parameter shapes must be compatible. New channels must be initialized to zero, and the optimizer must be reinitialized. This switch cannot be used in conjunction with interaction structure migration.
Experimental decisions and results are documented in [goal refinement and improvement record](goal-refinement-20261011.md).

## V6 Deployment-Aware Continuation (Experimental)

`training/configs/deployment_continuation.yaml` retains the raw-model terms from V5 and applies the default Neural/MatchEdge gain before every reconstructed waveform, complex, energy, and spectral term. The proxy uses the manifest cutoff/rate and Rust's rounded edge-bin geometry, -6 dB target ratio, [0.25, 2] gain target, 0.1 linear smoothing, and gain only where crossover weight is positive. This does not alter the exported model architecture or Rust runtime.

A cropped sequence has no prior stream gain. V6 initializes it from the first frame's **detached** target gain and then follows causal recurrence. This is a steady-state approximation; it is not an exact full-stream gain history. The online cutoff detector, enhancement gate, limiter, wet/dry ramp, and asynchronous fallback remain outside the proxy. Static manifest cutoffs may disagree with the detector. Actual Rust audio remains decisive. An identity generator evaluated with V6 also passes through this DSP proxy and is not the untreated dry baseline.

V6 adds `reconstruction_temporal_weight = 0.25`: signed adjacent-frame magnitude changes after causal OLA and reanalysis, compared to the clean reference. The denominator is the reference change norm, floored by 1% of reference full-band RMS times the square root of the difference-element count and an absolute `1e-4` floor; all scales are detached. Both attacks and decays are supervised. This differs from canonical evaluation's positive spectral flux. It does not reward constant spectra when the reference changes. Raw and reconstructed temporal errors are reported separately. V1–V5 formulas remain unchanged; total losses from different versions must not be directly compared.

Contract details retained from the original design: V4 uses `1x1` cross-stream projections; V4 input/output remains `[batch,2,1,129]`, and V5 also retains `[batch,2,1,129]`. The reconstructed-energy floor is the maximum of full-reference mean power times `0.01²` and the absolute `1e-4²`. The first V6 control uses 8 epochs / 1,024 updates and 896 validation rows; those are historical protocol values, not the later P-series budget.
