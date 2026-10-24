# Model artifacts

Official released weights use **Apache-2.0**, permit integration and redistribution
in other open-source applications, and include a completed model card and NOTICE.
See [weight license scope](../MODEL_WEIGHTS_LICENSE.md) and the
[release checklist](../legal/MODEL_RELEASE_CHECKLIST.md).

## Current availability

No qualified restoration weights are published yet. The bundled
[synthetic identity fixtures](../tests/fixtures/README.md) run the demos immediately;
they use Apache-2.0 and do not restore missing frequencies.

The latest [P54 common comparison and training protocol](../docs/spectral-training-p54.md)
compares historical shape-small with current small models under the same raw/native
scoring and DSP. Neither is uniformly better. Auraloss and actual-step calibration
are experimental training changes; they do not introduce a released weight file.
Use [research status](../docs/research.md) for current results; the entries below
describe historical artifact provenance rather than current training status.

The historical MUSDB candidate has 806,276 parameters and an approximately
3.16 MiB ONNX graph. It passes the [numerical parity audit](../docs/parity.md),
but is an unpublished research experiment excluded from the application-ready
release route. Changing this policy does not relicense its training sources.

The non-MUSDB real-source smoke is also unpublished: its Slakh MIDI/composition
rights and complete music-source grants are unresolved. The new 300-work native
music/VCTK run completed 200 fresh MPS training epochs at 01:00 Singapore time
on 2026-10-10. Its selected epoch-168 checkpoint is about 12.94 MiB including
training state; its local diagnostic ONNX is 3.16 MiB and passes 13 cross-runtime
parity cases. The [deployment diagnostic](../docs/native-validation-20261010.md)
does not establish restoration/release qualification; see
[original training](../docs/native-training-20261009.md) and the hash-bound
[current license review](../legal/WEIGHT_LICENSE_REVIEW.md).

The [objective-v2 continuation](../docs/loss-audit-20261010.md) imports only that
native epoch-168 generator, retains identical data/validation membership and starts
a new optimizer trajectory with explicit parent hashes. Its experimental artifacts
remain local and do not replace the baseline without comparative deployment evidence.

The [inpainting objective](../docs/inpainting-objective.md) adds explicit high-band
shape, energy, transient and reconstructed-spectrum supervision. Its new continuation
and [DSP comparisons](../docs/optimization-followup-20261010.md) remain local experiments;
none establishes a released restoration model.

The [V4 paired experiment](../docs/optimization-20261011.md) adds complex
reconstruction/spectral consistency and optional encoder amplitude/phase
interactions. Both six-epoch continuations are complete and pass 13 export
parity cases each. The control ONNX is 3.16 MiB (806,276 parameters); architecture
1.1 is 3.30 MiB (848,900 parameters), with unchanged causal input and buffering.
Actual DSP comparisons and remaining quality tradeoffs keep these weights local
and experimental; the original epoch-168 baseline is retained.

The [V5 goal review](../docs/goal-refinement-20261011.md) compares reconstructed
energy/phase-gradient supervision with optional circular phase features. Architecture
1.2 adds 144 parameters (806,420 total), with unchanged T=1 I/O and no added context
buffer. Candidate choice is frozen on development loss before the recording-held-out
recheck; that recheck reuses historically diagnosed recordings and is not a fresh
independent test. Source data and parent lineage are unchanged. These artifacts remain
local experiments, pending broader quality and listening evidence.


The [V6 deployment-aware experiment](../docs/deployment-refinement-20261011.md)
adds MatchEdge gain and reconstructed temporal supervision in training, retaining
the 806,276-parameter architecture and T=1 runtime contract. The eight-epoch run
selected epoch 4; all 13 export parity cases passed. Its weights remain local and
experimental; original baseline weights are retained.

The [representation/capacity study](../docs/architecture-study-20261011.md)
adds an original bounded global spectral residual branch, with matched polar and
circular gain/shape encodings. Three eight-epoch arms are complete, all passing
13 export parity cases. Small variants have 905,734 parameters (3.55 MiB ONNX);
the selected large variant has 1,399,942 parameters (5.44 MiB). The lower validation
loss does not establish an overall quality win. These are local experimental
weights, with unchanged data/parent lineage and no external pretrained teacher.

The [finite causal context study](../docs/causal-context-20261011.md) adds eight
past observations to the small refiner, retaining a current-frame U-Net and no
lookahead. Both matched arms completed eight epochs; the causal model has 923,526
parameters and a 3.62 MiB local ONNX. Its CPU loss is only 0.05% below the continued
frame control and 0.20% above the parent on matched crops. The 13 fixed cases pass
for the causal model, but broader numerical checks still contain out-of-budget
validation batches; the frame control also fails a fixed case. Both remain local
diagnostic artifacts. Existing Apache release goals and source obligations remain
unchanged; no external weights, teacher outputs or new audio were introduced.
A separate two-epoch pilot freezes every parent tensor and trains only the 17,792
temporal parameters at a higher learning rate. It reduces matched CPU loss by
about 0.013% versus the parent, while the history-ablation effect remains tiny;
this is a development-only diagnostic with no promotion or per-artifact realtime
qualification.

The [V7 reconstruction study](../docs/multires-reconstruction-20261011.md) keeps the
923,526-parameter causal graph and compares two four-epoch runs with real DSP
warmup and longer reconstruction supervision. The multiresolution arm scores
0.35% below its matched control but 1.21% above the parent on the common CPU
objective. Both pass 13 fixed export cases; broader real-frame checks still fail
6/256 control and 4/256 multiresolution batches. Quality gains remain mixed, so
both artifacts stay local and diagnostic, with unchanged source obligations.

The [input-conditioned gain study](../docs/gain-calibration-20261011.md) fits a
455-byte JSON with ten bounded DSP coefficients using only training recordings.
It leaves the 905,734-parameter shape-small ONNX unchanged. Despite lower internal
regression error, actual high-band energy and spectral convergence regress. The
coefficients remain local diagnostics and are not enabled by default.

The [phase reconstruction study](../docs/phase-reconstruction-20261011.md) uses the
same frozen shape-small ONNX with an experimental `surviving-power` phase policy.
No neural weights or fitted parameters were added. Partial quality improvements
remain insufficient for release; the default phase policy is still `model`.


Official releases require reviewed source grants and initialization under
[TRAINING_DATA.md](../legal/TRAINING_DATA.md). Source approval, restoration quality,
listening and real-time runtime/continuity evidence are all required before release.

Copy [MODEL_CARD.template.md](MODEL_CARD.template.md) next to a release artifact,
record its hashes and credits, and run the complete release checker described in
the checklist. Training audio and optimizer checkpoints stay outside Git.
