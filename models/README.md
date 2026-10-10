# Model artifacts

Official released weights use **Apache-2.0**, permit integration and redistribution
in other open-source applications, and include a completed model card and NOTICE.
See [weight license scope](../MODEL_WEIGHTS_LICENSE.md) and the
[release checklist](../legal/MODEL_RELEASE_CHECKLIST.md).

## Current availability

No qualified restoration weights are published yet. The bundled
[synthetic identity fixtures](../tests/fixtures/README.md) run the demos immediately;
they use Apache-2.0 and do not restore missing frequencies.

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
[current training](../docs/native-training-20261009.md) and the hash-bound
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


Official releases require reviewed source grants and initialization under
[TRAINING_DATA.md](../legal/TRAINING_DATA.md). Source approval, restoration quality,
listening and real-time runtime/continuity evidence are all required before release.

Copy [MODEL_CARD.template.md](MODEL_CARD.template.md) next to a release artifact,
record its hashes and credits, and run the complete release checker described in
the checklist. Training audio and optimizer checkpoints stay outside Git.
