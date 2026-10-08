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
rights and complete music-source grants are unresolved. The new large regional
queue has not trained weights yet. See the hash-bound
[current license review](../legal/WEIGHT_LICENSE_REVIEW.md).

Official releases require reviewed source grants and initialization under
[TRAINING_DATA.md](../legal/TRAINING_DATA.md). Source approval, restoration quality,
listening and real-time runtime/continuity evidence are all required before release.

Copy [MODEL_CARD.template.md](MODEL_CARD.template.md) next to a release artifact,
record its hashes and credits, and run the complete release checker described in
the checklist. Training audio and optimizer checkpoints stay outside Git.
