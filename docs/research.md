# Research status and experiment history

Current work: [P57 matched joint training](joint-training-p57.md), following
completed [P56 trainability attribution](trainability-audit-p56.md),
[P54 spectral training](spectral-training-p54.md) and
[P55 long-form source-disjoint review](longform-review-p55.md). All four 4096-step
continuations finish, with actual scheduled exposure of 278 recordings and 287.52
unique supervised minutes per trajectory. P55 scores 20 reserved sources and
160 cases per path. Native paired MR improves 1.39-1.59% over untreated input but
remains 2.23-2.44% worse than historical weights; HF8k MR is 3.05-3.32% worse.
Auraloss does not beat matched common-loss controls. Cleaner identity and lower
waveform errors remain distinct benefits, not proof of superior restoration.

P54's 52,000-hop diagnostic records maximum compute 1.531375 ms, no >=5 ms calls,
no hop service overruns and no paced lateness. The fixed 256-frame buffer and
actual-device qualification remain separate. P53 failures below are retained.
Blinded P55 audio is prepared, but human ratings are pending. No model is promoted.
P56 completed all 48 probes: decoder-only changes remain weak, full trainability
exposes additional directions, but tiny transfer gains and batch counterexamples
preclude quality claims. P57 tests cumulative adapter/full updates with identical
loss, fresh moments and 4096-step budgets. It excludes all reserved P55 aliases
and adds no inference parameters. Training completion and quality remain pending.

Previous completed pilot: [P53 balanced preservation](balanced-preservation-pilot.md).
Six matched 128-update trajectories, twelve raw reports, six exports and 480 new
native roles finish. Fixed corpus-gradient balancing improves clean-speech HF4k
MSE by 12.53% / 11.89% versus P52 physical, but loses paired spectral benefit
against the no-auxiliary controls. Seed-13 physical arms regress on fit and review;
this is not evidence for an architecture ceiling or isolated overfitting. A
post-completion first-update probe shows why gradient matching is limited: actual
Adam displacement differs from control by about 25-26% for physical arms and
44-49% for balanced logit, despite the registered aggregate calibration.

All 740 tests and implementation CI pass. Runtime does not: 78,000 measured hops
include four >=5 ms calls (maximum 7.403 ms), six hop service overruns and 14 paced
late completions (maximum 5.761 ms). CPU-thread probes were off; retain the failures
without attributing them to scheduling or model size. Next inspect the adapter's
post-Adam contributions under fixed fit-only controls and separately instrument
the production worker/queue. No longer unchanged training or default replacement.

Previous completed pilot: [P52 physical preservation](physical-preservation-pilot.md).
Six matched 128-update trajectories, twelve raw reports and 480 new native roles
finish. The physical auxiliary reduces clean-music error but loses paired spectral
benefit versus both controls; seed 13 also regresses against its P50 parent on fit
and review. Raw clean-speech HF4k MSE rises 9.79% versus the seed-13 control. A
post-completion fit audit shows music supplies 99.985-99.991% of the physical term;
batch-gradient RMS is 2.19-2.67 times the logit auxiliary despite pooled-norm
matching. The mathematics is verified; corpus/gradient balancing remains inadequate.
All 736 tests and implementation CI pass. Maximum compute is 1.699 ms across
78,000 hops, with one paced delay (0.447 ms). No default weight changes or longer
unchanged run follows. Next validate a fit-only group/gradient balancing rule.

Latest completed diagnostic: [P51 adapter responses](adapter-response-audit.md).
All eight analytic heads freeze before raw/native review; 640 new native roles
finish. Selective suppression retains 76-83% of P50 raw music MR gains and reduces
55-79% of added mean music identity RMS error. It beats scalar suppression on
spectra under the fit-logit matching rule, but actual PCM preservation is not
matched. Absolute gains remain tiny and identity/temporal counterexamples persist;
one scalar control also retains a 24-bit fit-low4k peak failure. All 104,000 compute
hops stay below 5 ms (max 2.308 ms); five paced delays remain. The 730-test suite
and implementation CI pass. These are analytic edits, not new gradient training
or qualified weights. Next align the separation constraint with physical high-band
residuals and original-reference reconstruction before bounded small-model training.

Previous continuation pilot: [P50 cumulative adapter updates](adapter-continuation-pilot.md).
All four 128-update persistent/reset Adam trajectories, six frozen raw reviews
on each fit/review role and seven native model controls finish. Fixed 0.001
high-support updates lower common loss and music MR on both starts: raw gains
0.0030-0.0091%, native gains 0.0037-0.0109%. Reset gives slightly more spectral
gain but generally more identity drift. All 16-bit rulers pass; identity failures
remain at 20/24 bits, with complex/temporal tradeoffs. Calibrated absolute errors
are reported alongside weak-band relative scores, without changing thresholds.
All 52,000 compute hops are below 5 ms (max 1.965 ms); two paced delays remain.
Fit/review sources are globally disjoint within this pilot but historically reused
TRAIN: 351 fit recordings and 986.623 unique supervised seconds are not full-library
training or independent generalization evidence. The 725-test suite and code CI
pass; inputs, checkpoints and review receipts verify. No weight is promoted.
The frozen high-band response audit follows in P51 above; do not repeat unchanged
long training or infer a capacity ceiling from this short pilot.

Latest completed calibration: [P49 adapter update scale](adapter-scale-calibration.md).
All 64 reset first-step probes and frozen guard reviews complete. Full Adam
overshoots in every probe; fixed 0.001 scaling and fit-only backtracking both
improve all fit totals. Guard total improves/worsens/is near the numerical floor
in 35/16/13 fixed probes versus 18/42/4 backtracked probes. The largest fit-passing
step is not a reliable transfer rule. Fixed high-support updates pass the
16/20-bit reference rulers with tiny spectral gains; 24-bit identity and
complex/temporal counterexamples remain. Absolute PCM checks prevent weak-band
relative scores from being mistaken for audible severity. These are independent
one-step models, not a cumulative training run or generalization evidence.
A bounded continuous-update pilot follows in P50 above. P49 itself claims no
new trained weight, long training or runtime qualification. The receipt-path execution
failure was repaired and reviewed from unchanged frozen proposals; both original
and recovery evidence remain. The complete 721-test Python suite and code CI pass.

Previous intervention: [P48 isolated spectral adapter](spectral-adapter-pilot.md).
All four 128-step runs, six frozen raw reviews and 560 native role replays finish.
High support reduces low-band preservation failures while retaining small music
MR gains (raw 0.120-0.214%; native 0.137-0.242% versus the corresponding start).
Identity, complex and dynamics errors still regress, and the common total
objective worsens. A retained first-step probe finds genuine overshoot: the
gradient direction is locally descending, but the full Adam displacement raises
fit loss about 0.027-0.030. A 0.001-scale probe descends on fit data but does not
consistently improve the disjoint guard. Calibrate update scale before another
matched run; this is not a capacity ceiling or a selected new optimizer. All
52,000 final compute hops are below 5 ms (max 2.480 ms), with nine paced delays.
No production weight is changed or qualified.

Preceding diagnosis: [P47 output coupling](output-coupling-audit.md).
Low-band coefficient changes explain 99.932-99.983% of the energy-pooled low-band
update projection on the changed P46 music endpoints. Reverting those updates
retains 99.50-99.74% of music MR gains while identity and phase failures remain.
An additional input-band control confirms that preserving compressed low bins
also loses useful low-band correction. This motivates P48's small independent
high-band adapter with a frozen shared feature path. P47 itself is diagnostic;
it did not train or promote a new model.

Previous preservation intervention: [P46 original-reference preservation](reference-preservation-pilot.md).
All twelve 128-attempt trajectories and 22 frozen endpoint reviews completed.
Policy and projected methods both accept 6 / 5 updates at 16 bits, 2 / 2 at
20 bits and none at 24 bits. Music MR falls about 0.06-0.07% at 16 bits, but
identity and individual low-band peak regressions persist; added mean-MSE guards
do not improve acceptance. All eight changed candidates fail their registered
ruler on some separate reused-TRAIN review examples. Hashes and full histories
were checked. No candidate is promoted and no follow-up training is launched.
Next proposed diagnostics separate output-band coupling and the budget surrogate
from per-example preservation; this pilot does not establish a capacity ceiling.

Latest measurement: [P44 absolute PCM calibration](pcm-fidelity-calibration.md).
Both small-model starts have mean identity errors hundreds of times larger than
clean-feature roundtrip noise; low-band counterexamples remain. The
[P45 anchored spectrum pilot](anchored-spectrum-pilot.md) completed all eight
trajectories and frozen raw rechecks. The loosest spectrum arm accepts only 2 / 3
of 128 attempts; both stricter arms are exact no-ops. Low-band displacement peak
guards reject corrections that can improve original-reference error, exposing a
preservation-policy mismatch. Stored Adam improves spectra more but retains
fidelity tradeoffs. No candidate is qualified or promoted. Its original preflight found
and fixed a column-rate broadcasting error before any checkpoint was saved.

Previous optimizer diagnosis: [P43 spectrum-led directions](spectral-priority-update-audit.md).
Matched to P42's reused training roles, fixed-radius amplitude/MR improvement
counts rise from 13/32 and 14/32 to 28/32 and 29/32. Fidelity/dynamics tradeoffs
remain: the 0.1% diagnostic policy accepts 8/32 steps, all improving spectra but
all increasing raw identity PCM error. No new trained weights or promotion.

[P42 pooling and step protection](pooled-update-audit.md) completes both seeds
with explicit no-op and calibrated PCM reporting. Its amplitude/complex bisector
improves high-band PCM on many accepted cases while worsening magnitude spectra;
this motivates spectrum-led directions, not a claim that waveform recovery is enough.

Previous optimizer diagnosis: [P41 guarded transfer](guarded-transfer-audit.md).
Simultaneous first-order constraints improve local joint descent from 16/32 to
19/32 at radius 0.001, but different-training-recording transfer reaches only 3/32.
Finite steps can violate locally certified guards; no consistent transfer benefit
or new trained model is established. Multi-batch estimation and finite-step
preservation remain prerequisites to a new matched small-model proposal.

Previous optimizer diagnosis: [P40 reversible update directions](update-direction-audit.md).
Actual Adam lowers total loss on every fixed training batch but often raises complex
and identity errors. A balanced amplitude/complex direction improves the local
tradeoff without passing all checks. No new training or promotion. The separate
[CPU-clock attribution](cpu-time-attribution.md) preserves all old timing failures.

Previous completed diagnostics: [P39 objective gradients](objective-conflict-audit.md)
and [balanced-order CPU timing](interleaved-timing-audit.md). Shared encoder gradients
conflict while high-band output-coordinate gradients mostly align. The >16 kHz
dB-only chain also reverses after the encoder Jacobian, so removing phase supervision
alone does not address the observed conflict. Timing retained three ≥5 ms calls, eight service overruns
and 119 paced delays over 160,000 hops, with max compute 9.280 ms. No new training
or promotion. The [failure registry](failed-experiments.md) is mandatory before a repeat.

Latest completed training: [P38 detector-condition controls](detector-condition-study.md).
All four 16,192-step CPU trajectories and native reviews completed. Fixing explicit
cutoff/active worsened music MR by 0.19–0.56% against matched controls, already present
in raw predictions; neither output path passed. Timing max was 5.411 ms, with two
≥5 ms calls, 11 service overruns and 27 paced delays over 280,000 hops. No promotion
or subsequent training. P36 also used CPU; Metal failure did not block this route.

Latest completed diagnostic: [P37 fusion boundaries](fusion-boundary-audit.md).
Frozen P36 weights, two seeds, eight interventions and 2,848 native replays on reused
validation. Removing magnitude fusion improves spectral/dynamics means slightly;
identity and local-band regressions remain. 120,000 timing hops stayed below 5 ms
with no hop overruns or paced delays. No promotion; subsequent CPU training is P38.

Previous completed training: [P36 factorial attribution](risk-attribution-study.md),
following [P35 broad risk](broad-risk-study.md) and [P34 small risk](risk-balance-study.md).
Both new P36 arms completed two 16,192-step seeds. Compression alone retains spectral
gains, but dynamics/identity gates still fail. The subsequent
[DSP boundary review](dsp-model-boundary-review.md) motivates P37 before further training.

Previous completed experiment: [P33 broad spectrum supervision](broad-spectrum-study.md).
Matched control/mr_add, two seeds and 16,192 steps each, fresh initialization,
full broad schedules and recording-isolated reused validation. All four trajectories
completed; no added-MR music advantage, speech means improved with worse dynamics.
Two compute calls exceeded 5 ms. No promotion or subsequent training.
Companion P32 stage analysis localizes most raw-to-native spectral loss to fusion;
the scheduling diagnostic separately compares the same binary under two policies.

Previous experiment: [P32 spectrum-first multi-resolution supervision](spectrum-priority-study.md).
All arms share architecture and initialization. Six fixed 1,024-step trajectories
and all native reviews completed. Added MR reduced native music spectral error
by 2.02–2.35% versus control on fitted crops, with inconsistent dynamics and higher
PCM error. Primary MR weakened speech results. Generalization remains untested;
91 of 420,000 compute calls exceeded 5 ms. No promotion; P33 follows with broader data.

Previous experiment: [P31 phase input and residual coordinates](phase-coordinates-study.md).
All eight fixed endpoints, 320 native roles and 560,000 timing hops completed.
Amplitude improvements did not establish phase/PCM restoration; Cartesian residuals
remained near zero without decoder saturation. Compute had nine >5 ms outliers and
79 paced completion misses. This remains training-only diagnosis, with no promotion.

Latest audit: [P30 high-frequency amplitude/phase swaps](phase-swap-audit.md).
All 1,320 native DSP roles completed on P29 fitted training crops. Reference phase
reduces PCM error, while reference amplitude with predicted phase increases it.
The existing fusion also limits reference-assisted recovery. These are nondeployable
oracle diagnostics, not new model gains; the subsequent P31 controls are recorded above.

Latest diagnostic: [P29 direct clean-spectrum supervision](direct-spectrum-study.md).
All eight small-model endpoints and 320 native roles completed. Direct supervision
improved HF energy reconstruction and produced larger residuals, but complex,
transient and final PCM fidelity did not improve together. These are fitted training
crops, not independent evaluation. No promotion or automatic continuation.

Latest completed study: [P28 small-model mechanism controls](small-mechanism-study.md),
96-wide, 431K–456K parameters, four configurations × two seeds × 2,048
updates. Training, native review and 560,000 timing hops completed. No intervention
passed the two-seed improvement screen. The diagnostic crops show no GELU collapse,
yet output remains near identity DSP. Compute stayed below 5 ms, with five service
overruns and 32 paced completion misses. No promotion or scale-up.

Current diagnostic: [P27 residual-decoder collapse and upstream review](head-collapse-audit.md).
Fixed training crops localize widespread negative GELU saturation; oracle spectra
still improve the common DSP objective. The 640-wide P26 route is retired from
further scaling. A compact causal neural/DSP route is specified; P28 implements only
the small mechanism controls, while a new DSP excitation branch remains unimplemented.
No quality qualification or default-weight change.

Previous completed study: [P26 broad coverage × residual budget](broad-budget-study.md),
four configurations × two seeds × 16,192 updates. Broad arms remained close to
identity DSP; full native review did not establish overall restoration. All 560,000
timing hops were below 5 ms, but per-hop deadlines still missed. No promotion.

Previous diagnostic: [P25 native inference and coverage audit](context-deployment-audit.md).
All fixed final P24 endpoints undergo native frontend → ORT → DSP review. The
finite bank provides only 482 seconds of unique music supervision, and zeroing
the old budget leaves a local residual. A continuous full-residual budget was
implemented and subsequently trained in P26 without promotion. Complete-chain CPU costs are measured
separately; no default weight or release qualification changes.

Previous completed study: [P24 generation-budget factorial](generation-budget-study.md),
four matched 640-wide configurations, two seeds and full 16,192-step budgets.
All training and review completed on October 17: 9,078 native role replays and 24
exports. Same-observation synthesis parity passes; one original frontend failure
is retained. Budget improves the old weak-band relative failure, but music waveform
error and some identity perturbations increase versus identity DSP. Final timing
shows deadline misses. No candidate is qualified for release.

Previous diagnostic: [P23 absolute-error and stage audit](capacity-outlier-audit.md),
1,424 measurements plus 128 native candidate/control role replays. The outlier
combines small absolute HF generation with amplified relative-band scores;
absolute preservation ranks differ from spectral risk. No threshold or model changes.

Previous completed full run: [P23 capacity comparison](full-capacity-study.md),
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

