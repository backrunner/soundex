# P66: native phase transport and output-support feasibility

Date: 2026-11-01. Status: stage attribution and frozen-weight support checks
complete. Zero optimizer updates. This diagnoses P65's fit/deployment mismatch;
it does not demonstrate improved restoration or qualify a new model.

## Changed question and controls

[P65](band-phase-p65.md) learned very small phase corrections on both fully
trained, frozen P63 band backbones. Its raw gains did not survive native DSP.
[P30](phase-swap-audit.md) examined oracle phase/fusion limits; it did not locate
these learned corrections. P66 therefore uses the exact P65 endpoints, inputs,
paired original references and native cases, without training or reference phase
injection. Neither the loss nor any quality threshold changes.

The panel is the same eight music and two speech TRAIN recordings, paired and
identity roles. It is a fit diagnostic, not independent testing. Mono views do
not establish physical stereo quality; source, genre, region and speaker coverage
are limited. Comparisons aggregate by recording. All individual cases, adverse
metrics and support partitions remain in the local assessment.

## Native stage observation

The offline replay adds an optional `stages` mode. It observes actual fused
spectra and effective magnitude/phase weights, with four independent Rust OLA
synthesizers advanced from the first hop. It captures:

1. Raw predicted spectra through Rust synthesis.
2. The activity/bypass decision.
3. Actual magnitude fusion with the preceding stage's phase.
4. Actual phase fusion through shadow synthesis.
5. Actual wet output after limiting.
6. Actual final output after gating, dry limiting and the output ceiling.

These are ordered counterfactuals, not order-independent causal shares. The
native `PhaseSmoother` mixes unit phasors within each frame; it does not temporally
smooth phase. Observation allocates outside the measured hop, only when requested.
It is forbidden in paced/repeated cost runs, requires consecutive hops from reset,
and rejects fusion/condition ablations. Production callback processing is unchanged.

The study freezes its binary, Rust/Python source and consumed inputs before
execution. All **80 P65 final outputs are bit-identical** to their saved versions.
Five additional trace-disabled controls are bit-identical to traced output.
Twenty identity-model roles isolate DSP fidelity. All 100 roles finish. The
reference region remains samples `89 * 128` through the final sample minus 128.
Raw scores now use actual Rust features, ONNX predictions and Rust OLA; their tiny
differences from P65's Torch raw path are not a changed population or objective.

## Where the gains disappear

Relative error change from each P65 zero head to its trained head on the matched
eight paired music recordings; negative values mean lower error:

| Stage | MR, seed 20261012 | MR, seed 20261013 | HF PCM RMS, seed 20261012 | HF PCM RMS, seed 20261013 |
| --- | --- | --- | --- | --- |
| Raw Rust OLA | -0.003146% | -0.001071% | -0.033545% | -0.024215% |
| Activity/bypass | -0.003146% | -0.001071% | -0.033545% | -0.024215% |
| Magnitude fusion | +0.001956% | +0.003748% | -0.032377% | -0.023952% |
| Phase fusion | +0.000699% | +0.000766% | +0.001226% | +0.001458% |
| Wet limiter / final output | +0.000661% | +0.000735% | +0.001220% | +0.001460% |

Magnitude fusion already changes the MR tradeoff. Phase fusion removes the small
HF waveform benefit. Limiting is not the main cause of this incremental P65 gain
loss. The native counterexample remains *Crunch Me Beautiful*, MP3 64 kbps,
case 4 / role 1, row `08561aa6340587794a8c7e1a`, crop start 62,411; both origins
increase MR there. The previous report retains absolute errors and all exceptions.
These small changes do not establish audible harm or improvement.

To locate corrections, we partition `|predicted_complex - zero_complex|^2`, with
one-sided Parseval scaling by `256^2`. This is windowed-frame difference power,
not unwindowed PCM dBFS, a share of training gradients, or useful recovered audio.
Effective weights are zero during bypass. All predicted magnitudes must match
exactly before phase-only attribution is accepted.

| Pooled HF correction power, >=8 kHz | Seed 20261012 | Seed 20261013 |
| --- | --- | --- |
| In active bins retained from input (native phase weight zero) | 96.98% | 95.92% |
| In crossover bins | 1.60% | 1.94% |
| In fully generated bins | 1.43% | 2.14% |
| Fused correction power / predicted correction power | 1.88% | 2.69% |

These shares pool power across the eight recordings, rather than giving each
recording equal weight. They cannot be extrapolated to all music. Above 16 kHz,
**97.85% / 98.95%** of correction power survives fusion. This contradicts a claim
that DSP discards all missing-band predictions: quality inside the passed band
and the phase representation remain unresolved. Removing DSP wholesale is not
supported by this evidence.

### Identity control and limiting

The identity model through the actual DSP has a separate reference-fidelity
limitation. On the eight clean-music identity roles, pre-limit full PCM error RMS
averages `2.04125e-8`; final RMS averages `0.00326303`, and MR rises from about
`1.59e-8` to `0.02270924`. On compressed music, MR goes from `0.31010056` before
limiting to `0.32939462` afterward. These are source-equal local means.

Thus safe limiting can conflict with the original recording even with identity
predictions. This absolute floor is distinct from the loss of P65's small
incremental gain. The original reference remains unchanged: do not replace it
with a limited target or describe that substitution as original-audio restoration.
This audit does not authorize removing the safety limiter.

## Implemented support-constrained phase head

`BandPhaseCorrection(..., phase_support="native")` is an optional experimental
route; the default broad route preserves historical behavior. The new route uses
only the existing input-side rate, detected cutoff and activity condition. It
zeros the appended correction where the current 1 kHz native crossover has zero
phase weight. Its mask is binary: it does not apply crossover attenuation twice.
All input bins remain available to the head, including preserved low-band cues.
The original-reference direct spectral loss is unchanged, and DSP output is not
a teacher. This is output availability, not a fused reconstruction objective.

Base weights, predicted dB, phase-carrier history, parameter count (196,162), state
(12,354 floats/channel), lookahead and buffering remain unchanged. During the
first 80 hops the diagnostic runtime overrides predictions with observed spectra;
the condition alone is not a warmup clock. The route is specific to this native
diagnostic policy, not arbitrary production phase/crossover configurations.

Both existing P65 endpoints were exported with this constraint, without training:

- 100 P66 traces / **2,270,400 post-warmup bin memberships** match native support
  exactly, including bypass and identity controls.
- All **40 candidate native outputs match P65 bit-for-bit**. Removing discarded
  corrections therefore changes the raw model but not the final audio here.
- Each graph passes 320 sequential Torch/ONNX hops across both rates and channel
  counts, including silence/reset. Graphs are 903,840 bytes each. This is not a
  new latency measurement; historical timing failures remain valid.
- Tests show unavailable outputs contribute zero head gradients, while retained
  input frequencies still contribute gradients to available outputs. Frozen base
  tensors and magnitudes remain unchanged. Cache signatures distinguish support
  modes even when all state tensors match.

| Origin | Native-support ONNX SHA-256 |
| --- | --- |
| 20261012 | `eb4332e421b9479a02a4dadb48f3cd144704f8e450b7aa1cbc547b33052d51db` |
| 20261013 | `bb3ae87f786554766287db6bbfc135a2cb9051012cd9391ac2e8e7e9573965dd` |

The counterfactual passes interface feasibility only. Equal final PCM is not a
restoration improvement, and removing unavailable outputs does not prove that
retraining will allocate useful capacity elsewhere. The P65 carrier head still
lacks static cross-bin phase information.

## Evidence, validation and next decision

Local ignored evidence:

- `data/runs/phase-transport-p66-20261101/`: registration, frozen source/binary,
  100 stage renders, five trace-off controls, `assessment.json`, completion receipt.
- `data/runs/phase-support-p66-20261101/`: registered frozen-weight counterfactual,
  two exports, 40 renders, support-membership report and completion receipt.

Both completion receipts and all registered files verify. No study execution
failed. An initial Parseval unit assertion allowed only absolute `1e-30` error
and rejected a two-ULP dB round trip (`0.019933422158758363` versus
`0.01993342215875837`). Its numerical fixture now declares `64 * float64 epsilon`
relative tolerance as well; this is not an audio acceptance threshold or discarded
counterexample. ONNX emits existing constant-folding/deprecation warnings.
The 33 targeted Python checks pass (phase head, stage observation and direct
spectral objective), including native mono/silent-left stereo at 44.1/48 kHz.
The Rust shadow-synthesis test, core all-target Clippy with warnings denied,
formatting, Ruff and English Markdown/link/index checks pass locally.

Before another training run, compare a support-constrained head and the broad
control under identical original-reference targets, initialization, exposure and
budget. Verify a differentiable actual-fusion reconstruction path, including OLA,
against Rust before using it as an auxiliary: output support alone does not
resolve the magnitude/phase fusion mismatch. Keep direct missing-band supervision
and separate raw/final scores. Register the objective and finish both trajectories
before reviewing endpoints; do not extend P65 unchanged or select intermediate
scores. A short calibration can establish trainability, not convergence or an
architecture ceiling. Default weights remain unchanged.
