# SoundEx model design

## Target and representation

Estimate missing/damaged high-frequency structure from compressed audio, supervised
by the paired original lossless spectrum. Preserve useful low-band content, channel
identity and real temporal changes. A larger high-band amplitude is not inherently
better; frequency support, shape, envelope, phase and transient fidelity all matter.

The production integration supports validated ONNX magnitude/phase contracts. Research
includes causal temporal backbones, complex residuals and generation budgets with
explicit history. The current P38 small-model controls have width 96 and 431,015
parameters. They are experimental, not qualified release weights.

Experimental capacity is limited by measured latency/throughput rather than parameter
count alone. Production still retains <=2M / <8 MiB until its contract is explicitly
migrated. Context must be past/current only, independently cached per physical channel.
Training windows and deployment lookahead are different quantities.

## Supervision

Use direct original-spectrum objectives and raw-OLA multiresolution magnitude/shape,
energy and reference-relative dynamics. Meaningful-bin complex/phase fidelity and
full-output preservation constrain the reconstruction. Identity DSP is a diagnostic
control, not the paired high-band target. Do not train through full safety limiting
to learn inverse limiting or to cancel a user intensity setting.

New losses require correct-reference behavior, silence/weak-energy gradients, finite
differences and actual deployment correspondence. Compare different training losses
under a fixed common evaluation. Report calibrated absolute PCM errors alongside
relative weak-band errors, and retain all counterexamples.

## Evidence-driven changes

Read [architecture research](../docs/architecture-research-20261011.md),
[model/DSP boundaries](../docs/dsp-model-boundary-review.md) and the
[failure registry](../docs/failed-experiments.md). Papers and official implementations
inform methods; ideal-low-pass upsampling or speech results do not establish music
codec restoration within a causal CPU budget. Referencing a method does not imply
importing its code, weights, audio or teacher outputs.

P20/P31 alternative heads and phase coordinates did not qualify. P38 fixed detector
conditions regressed music. P39 finds opposed parameter gradients concentrated in
the shared encoder, despite aligned high-band output-coordinate gradients. This
supports further mechanism diagnosis, not deleting phase supervision or scaling an
unchanged rejected route. See [P39](../docs/objective-conflict-audit.md).

## Training, export and acceptance

Use reviewed native-lossless sources, actual codecs and strict pair alignment. Keep
warmup outside supervision, freeze selection rules and complete matched budgets
across all controls/seeds before review. Validate raw predictions and actual Rust
frontend/ORT/DSP output separately. Export metadata declares state and time geometry;
verify ONNX structure, PyTorch/ORT correspondence and streaming parity.

Qualification needs recording/source isolation, genre/region/channel coverage,
identity and untreated-input controls, continuity, stereo, real-time tail performance
and listening. Reused held-out data and tiny fitted crops are explicitly limited.
The [training guide](../training/README.md) contains runnable workflows; old config
examples are not a substitute for a registered experiment.
