# SoundEx commit conventions

Commit as `BackRunner <dev@backrunner.top>` using Conventional Commits grouped by
responsibility. Format:

```text
<type>(<component>): <description>

[optional body]

[optional footer]
```

Type, component and description are required. Use an imperative, lowercase initial
verb and no final period. Keep the description at most 72 characters and body lines
at most 100. Explain the reason and behavior in the body; link issues or include
`BREAKING CHANGE` when applicable. Each commit should be coherent and buildable.

| Type | Purpose |
| --- | --- |
| feat / fix / perf | Feature, bug fix or measured performance change |
| refactor / style | Behavior-preserving structure or formatting |
| test / bench | Tests or benchmarks |
| docs / train | Documentation or training/model/data work |
| build / ci / chore | Build/dependencies, CI or maintenance |

| Component | Scope |
| --- | --- |
| core / dsp / cli | Corresponding Rust crate |
| train / model / data | Training code, model architecture/artifact or preprocessing |
| infra / deps / agents | Build/CI, dependencies or `.agents` guidance |

Examples:

```text
feat(core): add streaming process_frame API
fix(dsp): correct phase wrap-around in crossover blend
perf(dsp): reuse FFT scratch buffers
train(model): add multi-bitrate MP3 augmentation
docs(agents): align research guidance with current evidence
feat(cli)!: change output format option syntax
```

## Licensing and repository contents

Use Apache-2.0 source notices, either an SPDX identifier or the full standard header
where appropriate. Preserve applicable third-party notices. Contributions follow the
project's Apache-2.0 contribution policy without a separate CLA requirement.

Do not commit original training audio, paired corpora, local checkpoints, experiment
outputs, build products, IDE/OS debris or secrets. Commit provenance/source documents
and formally reviewed release artifacts. Synthetic short test fixtures are distinct
from training audio. Current `.gitignore` and `.gitattributes` are authoritative;
old illustrative patterns are not a reason to ignore this workspace's `Cargo.lock`
or force every small ONNX fixture into LFS.

Code, Markdown and configuration use consistent LF text; binary formats remain
binary. Check staged content and artifact sizes before each logical commit. Do not
rewrite frozen local source or failed-run evidence while preparing a public change.
