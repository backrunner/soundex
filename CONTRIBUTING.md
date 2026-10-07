# Contributing

SoundEx is experimental. Discuss changes to the model/feature contract or
real-time behavior in an issue before changing compatibility semantics.

By submitting a contribution, you agree to license your original software
contribution under Apache-2.0. No separate CLA is required. Do not include dataset
audio, secrets, training checkpoints or weights without a qualified model card.

## Checks

Run the Rust and Python checks in [README.md](README.md#development).
Use Python 3.12 and the [training setup](training/README.md). Add focused regression
coverage for defects, including non-finite values, sample continuity and model
contract rejection when applicable. CI compiles benchmark targets but performance
and physical-device acceptance are separate, artifact-bound measurements.

Audio callbacks must not allocate, lock, wait for inference or perform model
lifecycle operations. Preserve fixed output length, bounded queues and aligned
fallback. See [real-time standard](docs/realtime-standard.md).

Keep Rust source files around 300 lines or fewer and Python files around 400;
split larger functionality into modules. Put `SPDX-License-Identifier: Apache-2.0`
in new source files. New dependencies need a purpose and license review.

## Commits

Use `<type>(<component>): <description>` with a lowercase imperative description,
no trailing period and a subject of at most 72 characters. Keep commits focused
and buildable. Examples:

```text
fix(core): preserve aligned output after a worker deadline miss
train(model): bind small-window loss to the checkpoint contract
ci(infra): run synthetic model demos on Linux
```

Supported types and components are in [.agents/07-commit-convention.md](.agents/07-commit-convention.md).
Use your own author identity for contributions. Open a pull request describing
behavior, validation and remaining limitations; do not present fixture results
as learned-model quality or device measurements.
