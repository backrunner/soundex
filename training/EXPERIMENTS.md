# Historical experiment workflows

These profiles and commands document earlier registered comparisons. Use the
matching frozen source, config and artifacts to reproduce them; do not resume
an old trajectory with edited main-branch code. Results and current P38/P39 status
are indexed in [research history](../docs/research.md).

Model/loss modules used by these workflows are also shared by later studies.
A completed experiment does not make its implementation unused code. Consult the
[failed-experiment registry](../docs/failed-experiments.md) before repeating a route.

## Continuation profiles (v3–v6)

`inpainting_continuation.yaml` introduces objective v3: missing-band spectral
shape, frame energy, reference-relative transient changes and high-band error
after differentiable crossover/phase/OLA reconstruction. Waveform and valid-phase
terms constrain fidelity alongside retained-band and seam preservation. It uses
8 causal training frames, independent-frame inference, frozen BatchNorm statistics
and an explicitly fresh optimizer trajectory. Initial weights were gradient-audited
on 64 training draws; they are experimental. V1/V2 totals are not comparable to V3.
See [the inpainting design and proxy limits](../docs/inpainting-objective.md).

`consistent_continuation.yaml` and `interacting_continuation.yaml` provide a paired
objective-v4 experiment: complex reconstruction, retained-band corruption and
high-band STFT consistency. Only the second enables zero-initialized amplitude/phase
encoder interactions (architecture 1.1). Both keep independent T=1 inference and
six-epoch budgets. A legacy-parent architecture migration explicitly requires
`--initialize-zero-interactions` with `--initialize-generator-from`; changed objectives
or structure cannot directly resume old optimizer state. The data and sample ratios
must be resolved from the existing native run, rather than using example paths/ratios.
See [the objective definitions](../docs/inpainting-objective.md)
and [measured follow-up results](../docs/optimization-20261011.md).

`calibrated_continuation.yaml` and `circular_continuation.yaml` compare objective v5
with and without circular phase input features. V5 supervises reconstructed high-band
and subband energy plus reference-relative phase changes across frequency/time.
Both profiles use eight epochs, unchanged independent T=1 deployment, and frozen BN.
The architecture-1.2 variant expands the first phase convolution to raw/sin/cos
channels, adding 144 parameters; migration explicitly requires
`--initialize-phase-features --initialize-generator-from /path/to/parent.pth`.
It preserves the parent raw channel and initializes the added channels to zero.
Do not combine this migration with `--initialize-zero-interactions`.
Resolve real source paths and audited sampling ratios before running either profile.
See [the V5 methods and deployment evidence](../docs/goal-refinement-20261011.md).

`deployment_continuation.yaml` adds objective v6: default MatchEdge gain before
reconstruction and a reference-relative reconstructed temporal loss. It retains
raw V5 terms, the legacy architecture, eight epochs and frozen BN. The cropped
sequence gain uses a detached first-frame steady-state bootstrap; online cutoff,
limiting and gating still need actual Rust validation. Pass the canonical parent
data/ratios with `--initialize-generator-from`, not `--resume` with a changed loss.
See [the V6 experiment and limitations](../docs/deployment-refinement-20261011.md).

`scripts/prepare_architecture_study.py` derives matched control / polar-small /
shape-small / shape-large configurations from a reviewed parent without changing
its data paths or sampling. The optional `spectral_refiner` (architecture 1.3)
uses 129-bin frame-local global encoding, a zero-initialized bounded correction
head and widths 128/512. Explicit migration requires
`--initialize-spectral-refiner --initialize-generator-from /path/to/parent.pth`.
It adds no temporal context or runtime buffering. Keep the V6 objective and compare
actual Rust audio, model size and runtime costs before promoting a candidate.
See [the research and experiment protocol](../docs/architecture-research-20261011.md).
The [completed eight-epoch comparison](../docs/architecture-study-20261011.md)
separates development loss, Rust output quality and deployment cost; lower loss
alone does not promote any of the new experimental weights.

`scripts/prepare_temporal_study.py` derives matched frame-control / causal
continuations from a frame-refiner parent. The causal arm adds a depthwise temporal
filter and zero-initialized mixing on nine encoded observations; the U-Net processes
only the current frame. Explicit migration requires `--initialize-temporal-refiner`
with `--initialize-generator-from`. Artifact/architecture 1.4 declares input
`[B,2,9,129]`, output `[B,2,1,129]`; the ONNX graph has no hidden recurrent state.
The Rust host retains independent oldest-to-current spectral histories per physical
channel, including bypass/silence, and resets them on gaps/reloads. First eight hops
stay aligned dry while history fills; this enhancement warmup adds no output delay.
Both experiment arms use `sequence_warmup_frames: 8` followed by eight supervised
frames, so CPU validation must pass `warmup_frames=8` too. Keep loss and crop protocol
matched; totals from the earlier eight-frame-only study are not direct comparisons.
See [the causal experiment contract](../docs/causal-context-20261011.md).
`training.trainable_scope: temporal_only` freezes all parent parameters and trains
only the four new temporal tensors. It requires nine-frame context, frozen BN,
no GAN and a warm-started nonzero refiner head; a zero frozen head cannot propagate
learning into history. The default `all` retains normal training. Changing scope
requires a fresh optimizer trajectory; it is bound into the resolved checkpoint
configuration. This supports isolated learning-rate diagnostics, not automatic
quality promotion.


### Long reconstruction objective (v7, experimental)

Objective v7 adds an explicit `reconstruction_warmup_frames` prefix after real model
history. `causal_sequence_frames` still counts supervised frames only. The runner
infers the DSP prefix without gradients and frozen BN, initializes MatchEdge gain
at the Rust reset value of 1, advances it through the prefix, and excludes that
prefix from every supervised term. Gain history is finite and uses manifest cutoff;
this does not reproduce the online detector, gate ramp or limiter.

V7 requires `waveform_region: steady_state`, frozen BN and zero GAN weights.
The new observable terms reanalyse the fully supported OLA region at 2x/4x/8x the
model FFT size: `multires_high_spectral`, `multires_high_energy`,
`multires_high_temporal`, and `multires_low_complex`. They compare against the clean
reference (retained-band complex error compares against input), use detached
reference scales/floors, and penalize deviations from true temporal changes.
At FFT256/hop128 at least 21 supervised frames are needed for two longest-window
frames; the experiment uses 32 supervised + 32 DSP warmup + 8 model warmup frames.
No extra window is added to deployment inference or playback buffering.

`scripts/prepare_reconstruction_study.py` prepares two fixed four-epoch arms
with `--parent <frame-refiner.pth> --output-dir <new-study>`. Both use identical
long sequences and DSP warmup; only multiresolution loss weights differ. Final
checkpoints are rescored with one common CPU objective, so their differently
weighted training totals are not compared directly. See the
[protocol and results](../docs/multires-reconstruction-20261011.md).

### Input-conditioned gain calibration (experimental)

`scripts/fit_gain_calibration.py` fits ten bounded DSP coefficients using the
production Rust channel analyzer via `soundex-gain-data`. It accepts reviewed
music/speech manifests, excludes non-train recordings, chooses regularization on
a recording-disjoint inner split, and freezes an artifact before Rust development
scoring. The neural ONNX stays unchanged. The current regression target improves
while actual high-band recovery worsens; do not enable the fitted JSON by default.
See the [protocol and measured limits](../docs/gain-calibration-20261011.md).

### Phase reconstruction ablation (experimental)

`evaluation.compare_phase` fixes the model, detector, magnitude crossover and
MatchEdge recurrence while comparing `model`, `observed` and `surviving-power`
phase policies through the complete Rust stream. The last policy reduces the
model phase weight by the squared ratio of observed magnitude to the post-gain
candidate, capped at one. It is a heuristic and does not change model magnitude.

From `training/`, after rebuilding the release evaluator:

```bash
.venv/bin/python -m evaluation.compare_phase \
  --model /path/to/model.onnx \
  --manifest /path/to/music/manifest.jsonl \
  --manifest /path/to/speech/manifest.jsonl \
  --selection /path/to/frozen-validation-selection.json \
  --binary ../target/release/soundex-stream-eval \
  --output ../data/reports/phase-comparison.json
```

The bridge explicitly sets and verifies `SOUNDEX_EVAL_PHASE_POLICY`; benchmark
executables accept `SOUNDEX_PHASE_POLICY` and record the loaded policy. Both
experimental policies require Neural mode. The training loss proxy has not been
changed to use this policy; this is a DSP-only study with fixed ONNX weights.
See [results and limitations](../docs/phase-reconstruction-20261011.md).

## Original causal backbone probes

The [P0/P1 study](../docs/causal-prototypes-20261011.md) implements original GRU/TCN
complex-residual models with an [explicit-state 2.0 contract](../docs/causal-state-contract.md).
`export_causal.py` checks sequential ORT parity before publishing local artifacts;
`scripts/probe_causal_core.py` also checks Rust parity and irregular chunking.
`scripts/fit_causal_tiny.py` fits identity, ideal lowpass and real MP3 training clips,
using 64 burn-in hops and the production input detector trajectory. These are
learnability probes, not production weights or full-library training. The six runs
finished; matched-budget fresh training, independent testing and listening remain pending.

Build `soundex-state-parity`, `soundex-stream-eval` and `soundex-gain-data` first.
Run either script with `PYTHONPATH=. .venv/bin/python scripts/<name>.py --help`.
New artifacts require Model gain/phase and 64 observed warmup hops with aligned dry
audio. Benchmarks accept `SOUNDEX_HIGH_BAND_GAIN=model` and record this setting.
State warmup counters are separate from deadline availability; a timely dry packet
does not prove that enhancement was available. No raw audio/checkpoints are published.


## P2 matched training

`run_causal_study.py` trains fresh legacy/GRU/TCN models using the same frozen
paired draws, native-rate crops, 64-hop burn-in and `causal-reconstruction-v1`.
The registered comparison is 4096 updates × batch 4 × two seeds per backbone.
See [methods, numerical validation and limits](../docs/causal-training-20261012.md).

Build `soundex-gain-data`, `soundex-prediction-replay` and `soundex-stream-eval`
before running. From this directory, the historical study can be resumed with its frozen source with:

```sh
PYTHONPATH=. .venv/bin/python scripts/run_causal_study.py \
  --manifest ../data/processed/music_library/processed/music_library-7fda9e58ab583617/manifest.jsonl \
  --manifest ../data/processed/speech_library/processed/speech_library-7fda9e58ab583617/manifest.jsonl \
  --selection ../data/reports/optimization-followup-20261010/inpainting/stream-selection.json \
  --detector ../target/release/soundex-gain-data \
  --stream-binary ../target/release/soundex-stream-eval \
  --output ../data/runs/causal-p2-20261012
```

The file lock rejects duplicate processes. Resume requires identical configuration,
training source hash and binaries. Checkpoints retain optimizer and RNG state;
`export_study.py` understands the separate `soundex-p2-v1` checkpoint format.
Every completed arm exports its validation-selected checkpoint and records actual
Rust output metrics. No checkpoint is automatically promoted or uploaded.


## P3 weak-band and generalization controls

`run_weak_band_study.py` registers three matched TCN continuations per seed:
v1 control, reference-conditioned excess-energy supervision, and the same loss
with 12.5% native-lossless identity controls. All arms select checkpoints using
the same new objective, with separate train-domain probes and post-selection
held-out recording review. See [the protocol and limits](../docs/weak-band-study-20261012.md).
It requires completed P2 parents, freezes scoring binaries, and never promotes a
model automatically. Passing loss/shape tests does not establish generalization.


## Frozen-model limiter ablation

`run_limiter_study.py --study <completed-P3> --output <new-report-dir>
--binary <release-stream-eval> --baseline <old-shape-small.onnx>
--identity <identity-fixture.onnx>` compares the registered P3 controls, both
seeds' excess/identity candidates, the old gain policies, and identity DSP under
legacy and smooth-knee protection. It freezes the binary, checks PCM/model
hashes and reports recording-clustered metrics. The historical held-out review
is not an independent test and never promotes a model.

`models.causal_objective.render(..., limiter_mode="smooth-knee")` is covered by
Rust replay parity. Completed P2/P3 training used the default legacy limiter;
changing their objective after training is not a valid loss comparison. See
[the DSP study and remaining limits](../docs/limiter-study-20261012.md).

## P4 smooth-DSP restoration study

`scripts/run_balanced_study.py` continues completed P3 control parents with three
matched arms: smooth DSP control, reference-aware peaks, and bidirectional band
energy. Two seeds, identical identity-mixed draws, 1024 updates per arm and one
common selection objective isolate the changes. Raw references remain unlimited;
Rust output metrics and historical held-out review remain separate from selection.
See [the frozen protocol and run status](../docs/balanced-restoration-study-20261012.md).
