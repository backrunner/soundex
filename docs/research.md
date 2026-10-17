# Research status and experiment history

Current experiment: [P24 generation-budget factorial](generation-budget-study.md),
four matched 640-wide configurations, two seeds and full 16,192-step budgets.
All eight training trajectories completed on October 17. Native review stopped on
a weak-bin FFT phase discrepancy between training and Rust observations. A separate
review-only recovery retains the original failure and tolerance while isolating
synthesis parity; full relative/absolute output review is still pending.

Latest diagnostic: [P23 absolute-error and stage audit](capacity-outlier-audit.md),
1,424 measurements plus 128 native candidate/control role replays. The outlier
combines small absolute HF generation with amplified relative-band scores;
absolute preservation ranks differ from spectral risk. No threshold or model changes.

Latest completed full run: [P23 capacity comparison](full-capacity-study.md),
129,536 updates plus 4,450 native role replays. Train loss improved; no comparison passed all quality
checks, with a weak-high-band outlier dominating aggregate music regressions
despite paired proxy improvements on most clips. No new qualified weights.

Latest completed calibration: [P22 latency-first capacity](latency-capacity-study.md),
up to 11.92M parameters, two seeds and 512 updates per arm/seed. CPU cost passed;
identity damage outweighed small restoration gains. No candidate passed all
optimization checks. [P21](capacity-calibration-study.md) also ended with its
calibration rejected, so its longer comparison did not start. No new qualified
weights or held-out quality results; these short runs do not establish capacity limits.

Previous full experiment: [P20 restoration heads](restoration-head-study.md), with a common
clean-complex auxiliary target and matched fresh anchored/amplitude/complex heads.
All six trajectories and 2,314 native replays completed. New heads regress on full
training and development review; gradient/weak-band diagnostics point to optimization
scale and tail failures. No weights were promoted.

Previous experiment: [P19 music coverage](music-coverage-study.md).
All four runs and 1,246 native replays completed; broader coverage reduces some
regression, but final music fidelity still trails initialization. Previous studies below are
completed diagnostics, not recommended production models. Their code remains
where it supplies shared model/loss/evaluation functionality or reproducible
controls. None has qualified new release weights.

For current use, start with the [training guide](../training/README.md). Historical
continuation commands are in [experiment workflows](../training/EXPERIMENTS.md).
Resume registered studies from their frozen source and recorded inputs; editing
main does not change a running experiment's source contract.

P19 [failure analysis](coverage-failure-analysis.md) separates broad small gains from
severe failures on a few recordings, and confirms cross-band synthesis spill with
frozen stage analysis and 24 additional native intervention replays.

## Experiment record

The [original native-lossless run](native-training-20261009.md) uses 300
reviewed music works plus VCTK speech, from fresh initialization with MUSDB and
Slakh disabled. Raw audio, prepared pairs and training checkpoints stay local;
the repository publishes their source documentation and reproducible workflow.
The original 200 epochs completed; [loss auditing and controlled continuation](loss-audit-20261010.md)
address the remaining fidelity issues. Deployment validation still determines release qualification.
The [inpainting objective and neural/DSP paths](inpainting-objective.md) make
high-band reconstruction the main training target; DSP-only and hybrid paths remain experimental.
The [P2 matched training study](causal-training-20261012.md) has completed all six local runs:
fresh legacy, GRU and TCN backbones, 4096 updates each across two seeds, with a
shared reconstruction objective and production DSP validation.
Experimental weights have not replaced the default model. The
[first-seed review](causal-p2-review-20261012.md) finds gains over the matched
fresh control, but no overall win over the previous checkpoint and excess
generation on naturally weak high bands. The [P3 controlled continuation](weak-band-study-20261012.md)
has completed both seeds with excess-energy supervision and identity/generalization controls.
It reduces excess generation but retains restoration tradeoffs. The optional
[smooth limiter](limiter-study-20261012.md) reduces DSP distortion and avoids
recompressing worker output in the callback; the default remains unchanged.
The [P4 matched continuation](balanced-restoration-study-20261012.md) completed six runs against
the smooth DSP. Bidirectional energy supervision improves energy recovery but regresses
other fidelity measures; no candidate is promoted.
The [P5 amplitude representation study](envelope-study-20261012.md) completed six
fresh short/tiled/past-context runs. Past-only features improve one transient measure
but do not establish a consistent amplitude-quality win. Band-envelope/shape diagnostics
guide the next structural experiment; the 3.10 MiB amplitude probe is not a deployable audio model.
The [P6 band-envelope/shape study](factorized-envelope-study-20261012.md) completed six
matched fresh runs and native DSP comparisons. Separating energy and shape reduces some
raw spectral errors but sacrifices energy recovery, without a consistent final-output win.
The 3.13 MiB probe meets the diagnostic compute budget; default weights remain unchanged.
The [P7 energy-coordinate experiment](energy-coordinate-study-20261012.md) completed four
matched runs and 890 native DSP replays. Additive energy prediction improves average energy
recovery but regresses spectral and transient errors; no candidate is promoted.
The [objective audit](energy-objective-audit.md) locates the main regression in excess
energy on naturally weak bands and documents loss-ranking and aggregate-score tradeoffs.
The [P8 reconstruction-objective study](linear-reconstruction-study.md) completed four runs,
but every selected checkpoint is step zero: neither objective beats the initial common score.
The [optimization audit](linear-optimization-audit.md) isolates first-step energy overshoot
and tests input-relative updates without changing loss or model capacity. These are diagnostics;
no new weights are promoted.
The [P9 shape-coordinate study](shape-coordinate-study.md) separates band-relative shape
updates from exact identity initialization, with matched training and clean-identity native DSP
controls. All six runs and 2,136 native replays are complete; small raw-loss gains
still cause identity damage. The [P10 rendered-output study](rendered-restoration-study.md)
now supervises DSP synthesis directly and balances identity preservation. Its
pilot and six 2048-step runs reduce damage but select step zero in every arm.
The completed diagnosis separates a generalization gap from preservation pressure;
no candidate has demonstrated a balanced restoration win. The
[P11 fixed-set fitting diagnostic](fixed-fit-diagnostic.md) separates direct
amplitude/complex fitting from the shared network. All eight runs and 90 native
replays completed: direct fitting has substantial headroom and the existing network
can fit the small set, but identity damage and generalization remain unresolved.
The [P12 subset-scaling study](subset-scaling-study.md) now compares 5/10/40
recordings at equal update and equal per-recording exposure budgets. All six runs
and 330 native replays completed: broader data reduces regression, but longer
fixed-crop fitting worsens development fidelity. No candidate is promoted.
The [P13 crop-coverage study](crop-coverage-study.md) completed four runs and
300 native replays. Eight-position sampling reduces final development regression
versus fixed crops, but still trails initialization; cross-recording music fidelity
and identity damage remain unresolved.
The [P14 loss and synthesis audit](loss-scale-audit.md) covers every P13 training
position and development clip. Weak-band errors persist in native DSP, with severe
cases amplified during overlap-add synthesis. The completed
[P15 frequency interventions](band-transfer-study.md) locate cross-band spill
and show why simply disabling a band loses restoration gains.
[P16](spectral-consistency-study.md) completed four runs and 300 native replays
with a reference-aware consistency auxiliary and unchanged evaluation. Final
development changes disagree across seeds; one seed also regresses on the full
training bank in both objective terms. The [P17 matched continuation](consistency-learning-rate-study.md)
completed four short runs and 60 native replays: a smaller learning rate improves
the training objective in both seeds, mainly by reducing identity damage. Native
paired high-frequency detail does not improve consistently; cross-recording
fidelity remains unresolved.
The [P18 follow-up](consistency-followup-study.md) completed both continuations
to step 2,048 and 1,246 native replays. Training loss improves, but development
music fidelity regresses in both seeds; lower energy error does not bring better
spectral shape or transients. The 32-recording music subset also omits blues,
including the genre of the largest recurring failure. Further identical
continuation is not justified. The [P19 coverage study](music-coverage-study.md)
now compares 32 versus 253 training music recordings with fixed speech, model and
loss, reporting matched updates and matched music exposure separately. Training
and 1,246 native replays completed. Broader coverage reduces some regression,
but neither budget passes the registered restoration screen across both seeds.
Default weights remain unchanged.
The [baseline audit](baseline-generalization-20261012.md) identifies gain-policy tradeoffs
and limiter distortion even with an identity model; baseline overfitting is not established.

The [V4/architecture comparison](optimization-20261011.md) evaluates complex
reconstruction, amplitude/phase interactions and prepared real-time startup.
The [V5 goal review](goal-refinement-20261011.md) tests reconstructed energy
calibration and circular phase features, with recording-held-out rechecks and
explicit data-coverage limits. Experimental weights remain local.
The [V6 deployment-aware continuation](deployment-refinement-20261011.md)
adds edge-gain matching and reconstructed transient supervision to the training proxy.
The [architecture research and capacity study](architecture-research-20261011.md)
reviews recent restoration methods against the streaming budget and tests bounded
global spectral refinement with matched representation and capacity comparisons.
All three new arms completed eight epochs; the [measured results](architecture-study-20261011.md)
show a lower validation loss, but no balanced quality win that warrants promotion.
The [causal follow-up](causal-context-20261011.md) implements eight past frames
with unchanged output delay and two matched eight-epoch runs; it remains experimental.
The [V7 reconstruction study](multires-reconstruction-20261011.md) adds real DSP
warmup and multiresolution waveform supervision. Two four-epoch runs are complete;
measured improvements remain mixed, so neither candidate replaces the baseline.
A [bounded DSP gain calibration study](gain-calibration-20261011.md) fits ten
coefficients from production analyzer observations. Its regression improves but
actual high-band recovery regresses, so the default gain policy remains unchanged.
The [phase reconstruction study](phase-reconstruction-20261011.md) retains
model magnitudes while anchoring phase to surviving input power. Waveform fidelity
and some high-band metrics improve, but music remains short of balanced restoration;
the new DSP policy is opt-in and no weights are promoted.
The [architecture reset review](architecture-reset-20261011.md) broadens the
research to complex spectral generation, causal temporal models, differentiable
DSP and generative training. It proposes fresh backbone experiments with explicit
quality and streaming gates. The [first implementation stage](causal-prototypes-20261011.md)
now includes original GRU/TCN prototypes, an explicit-state Rust/ONNX contract,
production-DSP oracle diagnostics and six completed tiny-set learning probes.
Full-library matched training and qualification remain pending.

