# Research status and experiment history

[P111](envelope-fill-gate-p111.md) added a soft fill gate (0-6 dB on
fill/present band energy) to native Envelope mode. It kept a 1.70/1.76 dB
envelope-error advantage over the native P74 parent in every codec. It cut
absolute clean high-band error to -92/-98 dBFS, against -88 dBFS for P74. Clean
preserved LSD versus P74 was +0.68 dB for seed 1 (interval above zero) and
+0.12 dB for seed 2 (crossing zero), so the registered two-seed rule failed. A
20-trial blinded native pack, including clean-input trials, awaits ratings.

[P110](envelope-native-p110.md) ported the P109 candidate into an opt-in
native Rust Envelope mode (`.sxenv` artifact, native GRU, copy-up fill). With
matched SmoothKnee limiting, native output equals the Python proxy to float32
rounding. Against the native P74 parent, missing-band envelope error improves by
1.87/1.91 dB and LSD by 1.46/1.52 dB in both seeds and every codec. Clean preserved
LSD rises by 0.90/0.38 dB (15/16 works adverse). Model plus fill takes at most
0.53 ms per hop; host-scheduling availability failed in 3/4 bench cases.

[P109](envelope-clean-preservation-p109.md) fixed P108's clean-input
regression with a 2x2 factorial over target (absolute level or missing-energy
deficit) and a 25% lossless identity share. Identity pairs lowered clean preserved
LSD by 3.5-4.7 dB for both targets in both seeds. The registered deficit/identity
candidate still beats the native P74 parent on missing-band envelope error
(-1.52 and -1.55 dB, intervals below zero). Its clean LSD is level with P74
(+0.40 and -0.14 dB, intervals crossing zero). It met its rule to advance to a
native Rust port. Offline Python proxy on an exposed panel; not promoted.

[P108](learned-envelope-p108.md) trained a 219k-parameter causal GRU to predict
half-ERB high-band energies that drive a copy-up fill, plus a matched per-bin
control. On the P107 rulers it beat the preserved P74 parent on envelope error
by 2.3-2.4 dB in both seeds, at about 0.034 ms per hop. Band and per-bin outputs
tied. On lossless input it raised preserved LSD to 5.7-6.0 dB, because the model
never saw clean input and the ported Rust detector is active on ~99.9% of clean
frames.

[P107](missing-band-baseline-p107.md) re-scored the frozen P88 native renders
inside each clip's actually-missing region, with no training. Every trained model
leaves that region 9.6-11.6 dB too quiet and reaches 14.4-15.2 dB half-ERB
envelope error, against 35.3 dB untreated. A reference-assisted per-frame
half-ERB envelope on a copy-up patch of the retained band reaches 5.1 dB, lower
on all 15 defined recordings, and raises envelope correlation over time from
0.57 to 0.85. P88 shape versus control moves the same ruler by under 0.1 dB.
The oracle is not deployable; it bounds a parametric output, not learnability.

[P83](realized-relation-updates-p83.md) completes four old/direct/raw/fused arms
at all four P74 origins: 16 updates each, 256 updates total, 800 fresh native roles
and 320 passing correspondence checks. Every auxiliary-arm update descends its
auxiliary direction and paired-fit relation risks fall versus old. Yet all 12
native incremental MR/HF-SC intervals cross zero. Raw/fused coherent means improve
slightly, whereas direct increases; this does not establish restored audio quality.
Old continuation itself lowers paired fit risk while transfer spectral means rise
and coherent intervals increase. Music and speech, clean identity, model-only and
DSP paths expose different tradeoffs. All 104,256 paired metric records reproduce under
the prospectively added protocol seed-71 reaggregation; seed-83 evidence remains.
Fifty-eight focused tests pass. No default promotion or timing qualification.
Next isolate canonical/gain-64 continuation and repeated/fresh training crop
exposure, before any further full-training or capacity claim.

[P82](realized-phase-relations-p82.md) completes 32 frozen direct/raw/fused phase
relation probes and 40 saved native correspondence checks. Common original-PCM
targets isolate loss placement; median direct/raw head-gradient cosine is 0.591
for circular and 0.613 for wrapped. All native metric dictionaries reproduce,
with phase-risk discrepancies inside per-case analytic bounds. A follow-up
replays every probe exactly and distinguishes actual balanced routed gradients
from hypothetical stratum-only updates. Raw circular improves some paired head
directions but introduces three clean-music MR conflicts; full-model conflict
counts remain unchanged from old. Preserve the initial cusp-fixture failure,
all source strata and undefined speech vectors. Sixty-one tests pass. Next use
matched persistent-Adam finite updates and native spectral/preservation controls;
no new weight, full training, timing or quality qualification is claimed.

[P81](evaluation-audit-p81.md) audits metric definitions and interpretation without
new training. The additive panel measures within-band magnitude changes and full
PCM support; strict pairing rejects silently dropped fields and nonfinite scores.
An analytic FP32 ULP control shows that the P77 epsilon heuristic is not a score
error bound. All 144 native reaggregations reproduce P80 exactly; 1,248 comparisons
are described with separate spectral and coherent directions. All eight new
relation endpoints remain unresolved on both primary spectral rulers. Their old
gates remain failures, without becoming architecture or audible-quality verdicts.
Protocol v2 does not promote weights or approve a new budget. Next audit direct
versus reconstructed phase-relation supervision before a changed matched recipe.

[P80](phase-relations-p80.md) completes 32 frozen gradient probes, 16 two-update
endpoints and 400 new native roles. All eight old/realized-complex controls are
bitwise exact, and all 40 native checks pass. Explicit circular/wrapped phase
relations lower their fixed-fit risk and the unchanged common risk, but all eight
native joint-transfer gates fail: spectral changes are tiny/mixed and coherent
means rise. Near-antipodal mass is below 1%, so widespread circular-gradient
saturation is unsupported. Preserve the initial weak-fixture failure and every
counterexample. No full training, default promotion or new timing qualification.
Next distinguish direct feature relations from reconstructed raw/fused relations
under fixed kernels and original-reference spectral priorities.

[P79](parameter-outputs-p79.md) completes 32 frozen base/hidden/readout combinations,
5,860 score records and 57 new native correspondence checks, with zero optimizer
updates. Readout motion explains nearly all tiny coherent gains; MR/shape effects
remain mixed and numerically small. Eight controlled late-producer cases recover
aligned dry output exactly, with compute at most 1.663 ms, but retain 85 presentation
misses and fail all normal availability screens on faulted input. No overall
quality/device qualification. The next distinct prerequisite is original-reference
phase-relation supervision and its derivative/output behavior, auditing the legacy
helper versus the active objective before registering any new training.

[P78](adam-worker-attribution-p78.md) reproduces all sixteen P77 model/Adam
endpoints bitwise and decomposes the first physical update. Readout momentum
supports the auxiliary at all four origins; second-moment normalization does
not cancel the extra step. Some hidden-layer directions conflict, so a global
moment reset is unsupported. Sixteen bounded event traces retain three misses:
a host replay catch-up burst, its continuity-reset consequence, and a compute
window shorter than a nominal hop. One software case fails. The trace reconciles
all final counters; no new audio-quality or physical-device qualification follows.

[P77](phase-step-p77.md) completes the physical phase-step diagnostic: sixteen
two-update proposals, eight bitwise-equivalent model/Adam controls and 400 new
native roles. A larger readout step makes the realized auxiliary's complex mean
decrease at all four origins, but MR/HF SC increase slightly and all four joint
gates fail. Effects remain tiny; this does not reject the architecture or justify
longer unchanged training. A separate opt-in bridge now runs experimental graphs
through the actual bounded callback/worker transport. All 48 host cases execute;
six fail, with 14 presentation misses and one hop-processing overrun. A final
drain-accounting error is preserved and corrected without rerunning timing.
Observed compute peaks at 2.968 ms, but frozen final-metric snapshots have a race
boundary; no device/listening or complete final-call timing qualification follows.

[P76](phase-routing-p76.md) completes twelve two-update proposals with copied Adam,
bitwise canonical controls and 850 native roles. All four realized-phase routes
fail the registered joint-transfer gate, but MR changes against matched old
updates are only about 1e-9 in absolute units; this is inconclusive about audible
quality or architecture. Maximum per-role incremental PCM RMS is 2.655e-8.
Direct reweighting is similarly tiny/mixed. All 17 native correspondence checks
pass. The original speech-isolation preparation failure and a corrected read-only
trace-reader failure remain. Next audit physical phase-update conditioning with
an equivalent control before longer training; do not infer capacity failure or
promote numerical sign changes. Runtime interface migration remains separate.

[P75](realized-supervision-p75.md) completes literature/code research and 32 frozen
P74 TRAIN gradient probes, with zero optimizer updates. Current training already
supervises raw/fused realized magnitudes and direct pre-OLA complex spectra.
Existing direct and realized complex whole-model gradients align closely (median
cosine 0.990040); missing loss placement alone is not the demonstrated cause.
Fused magnitude MR conflicts with realized complex error in 19/32 probes.
A fixed 0.1 full-model auxiliary leaves nine total/complex conflicts and adds one
adverse HF-magnitude direction. Do not launch that recipe as an established fix.
The follow-up P76 above tests phase-head-only routing through native transfer and
matched finite Adam checks; the backbone keeps its old joint objective.
An initial FP32 gradient-summation rejection is preserved; the separate FP64 audit
passes the same tolerances (maximum 5.552e-15). All hashes and frozen weights verify.
No new quality, runtime or listening result. The report distinguishes AP-BWE,
DeepFilterNet, AERO and 2026 PAQM research from causal music CPU requirements and
separates P74's paced replay from production worker/device acceptance.

[P74](phase-temporal-p74.md) implements phase relations inside the small joint
temporal backbone with a same-parameter magnitude-only control. Both have 216,802
parameters and no additional lookahead/output buffer. All four 16,384-update
trajectories and fixed reviews are complete. Endpoint/receipt/native-artifact
and timing-trace checks pass, with finite model/Adam/history and no execution
failure. The phase observation fails the registered matched-control criterion:
long-form native MR -0.040630% / +0.165277%, HF SC +0.005631% / +0.034812%.
The other native music panel also has mixed MR -0.148400% / +0.206354%; speech
MR worsens +0.158238% / +0.008810%. Legacy-relative MR still improves 3.24–3.45%.
All phase TRAIN probes show learned history dependence, so an unused observation
does not explain the lack of consistent improvement. All 52,000 compute hops
stay below 5 ms (max 2.449333 ms), with no service overrun, but four paced
completions are late (maximum 2.263961 ms). No retry or promotion; all counters
are retained. Inspect realized-spectrum supervision with a separate control and
worker wake/headroom before any changed training. Listening, independent-test,
convergence and device evidence remain unestablished.

[P73](capacity-tradeoff-p73.md) completes all frozen P72 magnitude/phase and
source-stage accounting. High-band magnitude gains coexist with increased
phase-interaction power; its symmetric magnitude factor accounts for 94.45% /
93.60% of the increase. This is descriptive algebra, not a causal fraction.
DSP reduces the raw coherent regression, while the other development panel has
mixed source/codec and stage changes. No new weights, timing retry or default
promotion. The retained late replay is distinct from production-worker/device
evidence. Next investigate phase observations inside the small temporal backbone
and realized-spectrum supervision with separate controls, rather than unchanged
scaling or phase-only continuation. Full records and next-stage prerequisites
are in the report.

[P72](band-capacity-training-p72.md) completes four 16,384-update trajectories and
all fixed reviews. Doubling parameters improves long-form native MR 0.919099% /
0.690461% and HF SC 0.789985% / 0.362459% versus matched small controls, with both
paired source-bootstrap intervals below zero. HF shape is mixed and HF PCM
worsens 0.980238% / 0.662302%. The other fixed native music panel has MR +0.002340% /
-0.290006%, so the benefit is not uniform across panels. Legacy-relative long-form
MR improves 4.064560% / 4.103731%. Trained compute peaks at 1.663041 ms across 52,000
hops, but one paced completion is 0.217405 ms late. All integrity checks pass;
no execution failure, rerun or promotion. Inspect frozen phase/shape/source
counterexamples and worker wake behavior before another recipe. Listening,
independent testing, convergence and device qualification remain unestablished.

[P71](evaluation-capacity-p71.md) implements and calibrates a fixed multiresolution
panel with separate shape, energy, dynamics, complex/absolute/stereo rulers and
paired recording-bootstrap intervals. All 280 paths retain historical scores and
counterexamples. Same-depth joint small/double capacity fixtures (204,418/408,023
parameters) pass native parity and 26,000 measured hops, max 0.558/2.243 ms with no
service overruns or paced lateness. These are untrained feasibility results.
The subsequent P72 trained results are above; default weights remain unchanged.

[P70](baseline-metric-audit-p70.md) completes comparator, metric and frozen-feature
audits. Legacy shape-small has 905,734 parameters versus P69's 204,418; it is not
the matched repeated-input control. P69 already improves long-form native MR
2.82–3.06% over legacy. All 160 saved native scores reproduce exactly; independent
HF spectral convergence retains the legacy gain and matched-control failure.
Energy-matched phase scrambling improves pure log distance on 18/20 excerpts
despite worse HF PCM on all twenty. Added-feature removal changes phase RMS by
only 0.85–1.02% of the original rotation RMS on fit probes. Reference substitutions
show remaining magnitude/phase headroom, not achievable optima. The audited alias
check passes, without eliminating historical selection bias. Next inspect small
joint amplitude/phase trainability and a frozen perceptual protocol; do not extend
unchanged phase-only training or promote defaults.

[P69](cross-phase-training-p69.md) completes all four 16,384-update trajectories
and fixed reviews. Adjacent native music MR improves 0.105384% / 0.076662% over
zero parents on the source-isolated panel, but trails matched repeated-input
controls by 0.003600% / 0.000632%. The exposed P55 long-form recheck agrees:
parent-relative gains 0.156115% / 0.132448%, matched-control regressions
0.004151% / 0.000784%, with HF PCM/dynamics tradeoffs. All four endpoints and
completion receipts verify, with no execution failure. The trained 52,000-hop
grid peaks at 1.419083 ms without service overruns or paced lateness. The new
observation fails its registered criterion; inspect frozen feature use/output
expressiveness before any changed recipe, without promotion or convergence claim.

[P68](phase-gradients-p68.md) completes 64 frozen P67 phase-gradient probes.
Native support raises median common/fused parameter-gradient agreement from
0.321–0.372 to 0.826–0.880; paired/identity conflicts remain, with no activation
saturation. New optional adjacent-bin phase features remove a demonstrated
observation invariance, with a same-width repeated-temporal control. All four
204,418-parameter nonzero fixtures pass native parity and 52,000 measured compute
hops (max 1.464625 ms, no service overruns or paced lateness). This is zero-update
feasibility, not a quality gain, convergence result or real-device qualification.
The initial numerical causality-test failure and package-launch failure are
documented. Register adequate matched training before judging the new observation;
keep original-reference spectra, all controls and prior timing failures.

[P67](fused-phase-p67.md) completes all eight 512-update endpoints and 240 native
roles after passing 100 DSP correspondence checks. Native support improves mean
native music MR under both objectives; the combined arm improves 0.002671% /
0.001439% versus zero-phase parents. MR still worsens on 5/8 and 2/8 sources,
and dynamics/HF PCM regress slightly. Both broad/common controls reproduce P65
tensors exactly. Results are tiny TRAIN-fit changes, not convergence or overall
quality. All receipts verify; the original zero-update build-alias failure remains.
Inspect passed-band gradients and phase representation before further training;
no new runtime or release qualification is claimed.

[P66](phase-transport-p66.md) completes 100 native stage traces, five trace-off
controls and 40 frozen-weight support counterfactuals. Most pooled P65 HF phase
correction power (95.92–96.98%) lies in input-retained bins; magnitude fusion
changes the spectral tradeoff and phase fusion removes the small raw HF PCM gain.
Nearly all correction power above 16 kHz survives, so the passed-band predictor
remains an open issue. Limiting creates a separate identity-reference floor.
The new optional native-support phase head matches 2,270,400 observed bin
memberships and leaves all 40 native outputs bit-identical, with no optimizer
updates or quality improvement claimed. Before retraining, verify the fused
reconstruction objective on this branch and register matched broad/support
controls. Default weights and previous timing failures remain unchanged.

[P65](band-phase-p65.md) completed both 512-update frozen-magnitude phase
calibrations and all fixed reviews. Common fit loss and history dependence improve
slightly; matched raw music gains do not become native gains (native MR
+0.000661% / +0.000735%). This is early trainability evidence, not convergence or
a phase-model ceiling. The 52,000-hop cost grid remains below 5 ms but includes
one 3.026458 ms service overrun of a 44.1 kHz hop. Worker wake telemetry and both
execution failures are documented; completion recovery verified unchanged consumed
inputs and frozen binaries without retraining or quality rerenders. Default
weights remain unchanged. Investigate correction support/phase representation
before enlarging this branch's training budget.

[P64](band-attribution-p64.md) completes frozen-output codec, PCM error and
existing-deadline attribution. MP3's closer magnitudes interact with imperfect
phase; symmetric factorization attributes 96.19% / 96.52% of the native
interaction increase to magnitude changes. AAC has a different tradeoff. DSP
attenuates the MP3 PCM regression. All sources and invalid audit attempts remain
retained. No new training or timing benchmark ran. The next bounded hypothesis
is magnitude-preserving phase correction on frozen band endpoints, with native
correspondence and separate worker wake/queue validation before any qualification.

[P63](band-training-p63.md) completed fresh global width-96, global width-46
and eight-band TCN models, two seeds and 16,384 updates per trajectory. The
parameter-matched comparison is 170,859 versus 171,137 weights. All endpoints
and the recovered 160-case / 20-source long-form review are complete. Native
source-equal music MR improves 2.86% / 2.84% over matched controls, 1.76% / 1.88%
over wide controls, and 2.67% / 2.93% over legacy. Raw model-output gains also
survive DSP. This meets the registered spectral criterion for further study,
with HF PCM RMS +1.31% / +1.58%, dynamics +0.30% / +0.39% versus matched controls,
source regressions and mixed AAC results. P55 sources are exposed rechecks; all
20 regional labels are unknown, and no listening/device qualification is claimed.
The unchanged 78,000-hop cost grid fails real-time criteria. Storage-only recovery
preserved original errors, endpoints and timing, verified all 91,236 inputs and
3,387 protected files, and completed on 2026-10-30. Twenty blinded listening trials
are ready, with no ratings. Default weights remain unchanged.

[P62](band-tcn-p62.md) implements an eight-band causal TCN with 171,137
parameters and 12,096 state floats per physical channel. Nonzero ONNX and actual
Rust parity pass. A registered 26,000-hop control/prototype comparison measures
band maximum frontend/model/DSP compute of 1.218834 ms, but two paced band
completions are late by up to 3.622951 ms. The original diagnostic state-size
failure and all timing failures remain retained. Python 821 tests and Rust 132
tests pass locally. This permits a matched training experiment; it is not a
trained quality result, production promotion or real-device qualification.

[P61](critic-objectives-p61.md) completes 768 frozen-critic gradient records and
1,280 raw PCM probes on 72 fit recordings / 37.097 unique supervised seconds.
Structured-negative arms have prediction-uphill saved-Adam directions on 15/16
batches in both origins. Learned features also lose to random-feature controls
in source-equal mean MR probe benefit for every arm/origin/corpus. Retire the
current classifier recipe as the next generator auxiliary; do not infer that
all GANs or feature losses fail. These are local probes, not trained audio gains.
The [architecture/evaluation review](architecture-evaluation-p61.md) documents
the actual 480,809-parameter causal TCN, refreshed primary-source research and
why classifier ranking is not restoration qualification. The band-token
prototype follows in P62 above. No new generator training; at P61 completion all
811 local tests and implementation CI passed.

Latest completed study: [P60 context and negatives](texture-context-p60.md).
All eight 1,024-update critics freeze before 17,920 review records. Structured
negatives fix phase-control ranking on every source and recognize most unseen
shift controls, but reverse ordinary prediction ranking to only 3-4/45 music
and 0/16 speech sources preferring the original reference. All four arms fail
the registered prerequisite. Observed-band summaries alone do not fix phase
recognition; normalized paired feature distances remain close to random controls.
Next diagnose separate loss-component gradients on fit data before proposing
separate task outputs. P59 calibration is an exposed recheck. All 804 local
tests and implementation CI pass; no restoration generator updates or promotion.

Latest completed calibration: [P59 critic learning](texture-calibration-p59.md).
Both 1,024-update critics finish before uniform scoring. Fit includes 205
recordings and 554.812 unique supervised seconds; critic calibration contains
45 music and 16 speech recordings, with connected source/work aliases excluded
from critic fitting. These remain reused TRAIN recordings for the generators.
Ordinary prediction classification improves slightly, but every phase-scrambled
source receives a higher realism score than its reference; most time shifts also
escape. All four seed/corpus groups fail the registered prerequisite.
Paired feature distance still penalizes both controls, so this does not show
that the proposed generator loss rewards noise. No generator updates follow.
The P60 follow-up above isolates observed-band conditioning and structured
negative exposure. At P59 completion, all 793 local tests passed; no
restoration-quality, runtime or default-weight promotion is claimed.

Latest diagnosis: [P58 band/source/stage attribution](joint-attribution-p58.md)
reuses every frozen P57 long-form case. Above 16 kHz supplies 69.93% / 72.94%
of the native paired MR gain and 97.36% / 92.17% of added clean MR error. The
raw model already contains the temporal/fidelity tradeoff; DSP retains 77.54% /
79.98% of raw spectral gain. These are descriptive score contributions, not
audibility or a new independent test. The initial rounding-check failure is
retained separately from the corrected attribution.

The [official implementation review](implementation-review-20261026.md) compares
AERO, Apollo, AP-BWE and DeepFilterNet and adds an original training-only paired
spectral-feature prototype. Both frozen P57 generators complete eight critic
updates and fit/calibration gradient checks; restoration weights remain exact.
This validates integration, not critic learning, improved music or new trained
generator weights. The broader P59 calibration above now fails the intended
false-detail mechanism check; a matched generator comparison remains deferred.
No unchanged long continuation is launched.

Latest completed work: [P57 matched joint training](joint-training-p57.md), after
[P56 trainability attribution](trainability-audit-p56.md),
[P54 spectral training](spectral-training-p54.md) and
[P55 long-form review](longform-review-p55.md). All four P57 trajectories finish
4096 updates, followed by raw/native reviews, timing and the P55 stereo recheck.
Full-model training reduces native paired MR 0.6228% / 0.6369% versus matched
adapter controls and HF8k MR 0.7901% / 0.8092%, without adding parameters.
It still trails historical MR by 1.7735% / 1.5713%, increases clean-identity error
and slightly worsens reference dynamics. No weights are promoted.

P57 timing covers 52,000 hops: max compute 2.238792 ms, zero >=5 ms calls or hop
service overruns, but one paced completion is late by 0.468495 ms. Keep this failure
separate from compute feasibility and retain earlier P53 failures below. Fixed
256-frame buffering, actual-device qualification and human listening remain
separate. P55 recordings are exposed and all new scoring is labeled a recheck.

P58 above attributes cumulative joint-training changes by band, source and output
stage. It motivates the paired texture-supervision prototype, not automatically
scaling capacity or removing identity.
All original-reference, phase and transient targets and source reservations remain.

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

