# Weight provenance review

Historical source-chain audit updated on 2026-10-11; documentation translated and
status clarified on 2026-10-23. **No restoration model weights have completed formal
rights and release qualification.** Source and demos use Apache-2.0; this does not
automatically license local experiments. Official weights are intended for
Apache-2.0 release, including integration, redistribution, modification and commercial
use in other open-source projects, with no academic-only restriction.

## Artifacts covered by the historical audit

The audit read actual checkpoints, configurations and five bound manifests from
three experiments, verified their SHA-256 consistency, and recorded hashes for
42 checkpoint/diagnostic ONNX files. Training, validation and test membership were
recorded separately. Paired-example counts are not counts of independent works.

| Experiment | Reviewed provenance | Release status |
| --- | --- | --- |
| `local-mps-20261007-125023` | MUSDB18-HQ manifest | Unpublished; excluded from the commercially usable official-weight route |
| `local-mps-256-128-20261007-220130` | MUSDB18-HQ; `parity-qualified.onnx` denotes numerical verification only | Unpublished; numerical parity does not establish rights clearance |
| `local-mps-diverse-lossless-smoke-20261008` | The Crypts! / Revolution Void music, Slakh and VCTK; not the completed later large training run | Unpublished; Slakh underlying rights and complete music-source permissions remained unresolved |
| Native regional lossless run | 300 music works plus VCTK passed paired-manifest/file checks; random initialization, no old weights or teacher; 200 epochs completed at 01:00 on October 10; epoch 168 exported and passed 13 cross-runtime consistency checks | MUSDB and Slakh excluded; final-artifact rights and complete quality acceptance remain unfinished. See [training](../docs/native-training-20261009.md) and [validation](../docs/native-validation-20261010.md). |
| Objective v2, eight-epoch continuation | Only the native epoch-168 generator was inherited; parent SHA-256 `a4fed7e7f4a463c01213d2a0705286c85a63b8e3b34c6681dcd9124cf7530650`; unchanged recipe, manifests and validation membership; no other old weights or teacher | Baseline unchanged; derivation does not remove attribution or release-review obligations. Numerical consistency is not an official Apache-2.0 release. See [loss audit](../docs/loss-audit-20261010.md). |

Subsequent V3 → V4 → V5 initialization still derives only from native epoch 168,
without external teachers or MUSDB/Slakh weights. Both V5 arms completed eight epochs
on 2026-10-11 with frozen source `db738344` and common V4 epoch-4 parent SHA-256
`a152f00c6907221a4af80986364f621c1d884e9df0af22ba001ff419a574ccab`.
The recipe, manifests and attribution obligations are unchanged; exact derived hashes
are in the [goal review](../docs/goal-refinement-20261011.md). This updates provenance,
not permissions or release clearance. Experimental weights remain local.

V6 deployment alignment also starts from V4 epoch 4, without external models or new
data. Frozen source is `e6946b2`; eight-epoch training selected epoch 4, checkpoint
SHA-256 `22c39a78baca386c2b30d34641f02dea7cf83a331f00eefd8390d769f0a8a433`,
ONNX `18ea464cfcc4c978b252f1b6ad04ce5e1fc427dc225b58759b6a8e69d2b57336`.
Recipe and manifest bindings are unchanged. See [V6](../docs/deployment-refinement-20261011.md).
This adds no release authorization.

The three new [representation/capacity arms](../docs/architecture-study-20261011.md)
also start from V4 epoch 4, frozen source `7cfaf784`, with eight epochs and unchanged
data/manifests. The fixed CPU selection rule selected `shape-large`, checkpoint
`8d5795cfe62a572e2221c657997951f6fb3b843911219087584e006b2218a53c`, ONNX
`a9a5096d1cc87eaf3c151ce9d25fc4dc1f4abe33db84b7a94076739c66fa78d3`.
The [paper survey](../docs/architecture-research-20261011.md) informs methods only:
these original implementations import no surveyed-model code, weights, training
audio or teacher outputs. Architecture changes do not remove source obligations
or grant release clearance. All artifacts remain local.

The [finite causal-context study](../docs/causal-context-20261011.md) starts from the
`shape-small` parent `a5166a6f56239a31a48986725188aa19c1c199c565904c76aca0f160a19f393a`.
Both arms completed eight epochs with source `52ca6b93`, unchanged recipe/manifests
and no external code, teacher, weights or audio. Exact derived hashes are in the
study report. Both are local diagnostic artifacts with numerical/quality gaps.
The subsequent two-epoch temporal-parameter probe, source `9fd534d7`, uses the same
parent, updates only new temporal parameters and preserves every parent tensor
bit-for-bit. Sources and obligations are unchanged; no new clearance is claimed.

The [continuous reconstruction/MR study](../docs/multires-reconstruction-20261011.md)
uses the same `shape-small` parent and source `a27550ac`. V7 changes supervision and
DSP warmup without adding audio, external weights or teacher outputs. Its local
artifacts retain the same source obligations and are not automatically release-ready.

[Input-conditioned gain calibration](../docs/gain-calibration-20261011.md) leaves that
neural graph unchanged and fits 10 DSP coefficients from training recordings in the
same native-lossless recipe. Its JSON is unpublished. No new data, external weights
or teacher are introduced; attribution and release review remain necessary.

The old native epoch-168 checkpoint recorded later main-repository source metadata
`dd0d15ab`. Its actual frozen training receipt is `abe0e497`; all 106 archived training
files match that version. Original artifacts/hashes remain unchanged, so review must
use the frozen receipt as well. Later archived runs correct source capture; Objective
v2 uses an isolated worktree bound to `b4ed52c`.

The two MUSDB best-checkpoint hashes are
`167174450fc6591f616a3a5d645623417d06c76d3651623ddf802d573ff52c20` and
`5134cd9b80c5e36cb674dcad1686b77dc1fc39dbac1595d58d230e33407d9d21`.
The non-MUSDB smoke best checkpoint is
`df2704704f7c68b1da3218abd25b5fc8153a71d77d94cd6e7838cc9901b83639`.
The complete local inventory is `data/reports/weight-license-audit-20261008.json`;
audio and weights were not committed. [MUSDB's publisher](https://sigsep.github.io/datasets/musdb.html)
restricts the dataset to academic use. Deleting MUSDB or changing documentation does
not change a checkpoint's training provenance.

Later P-series registrations and hashes are indexed in [research history](../docs/research.md)
and the [failure registry](../docs/failed-experiments.md). Those technical records do
not extend the scope of this historical rights audit or imply formal approval.

## Slakh: distinguish dataset licensing from underlying rights

The [original generation repository](https://github.com/ethman/slakh-generation#license-and-attribution)
labels Slakh CC BY 4.0. The [official archive](https://zenodo.org/records/4599666)
describes synthesis from Lakh MIDI and sampled instruments. The [Lakh maintainer](https://colinraffel.com/projects/lmd/)
explains that the MIDI was scraped from the public web, not transcribed by him, and
has inconsistent internal copyright information without reliable per-track authorship.
This is a specific composition/transcription rights gap; a dataset CC label alone
cannot confirm permission for every third-party work.

The authors' [rendering instructions](https://github.com/ethman/slakh-generation#step-2-installsetup-kontakt-andor-other-synthesis-vsts)
identify Kontakt Komplete 12 (2018). Evidence for the actual instrument/sample
license version is also needed. The Native Instruments [EULA reviewed in the audit](https://www.native-instruments.com/pages/end-user-license-agreement)
was dated 2026-07-01 and included AI-training restrictions for products and audiovisual
content. That text alone does not establish retroactive application to historical
Slakh renders, or equate complete synthesized music with raw instrument samples.
The question is the actual version's terms and the scope covering released mixes.

Following the user's 2026-10-09 decision, the new regional configuration and queue
exclude Slakh and do not require its paired files. Historical manifests, hashes and
initialization records remain audit evidence, separate from the current route.
Cleanup removed unused Slakh original/paired audio, old generated audio and redundant
periodic checkpoints, retaining independent audit checkpoints and deletion/migration
inventories. Removing data does not retroactively clear old weights. Historical
import tooling was retained at that time and later retired during documented cleanup;
no dataset allowlist or hard training-license gate was added.

Weights containing Slakh cannot be described as cleared for an Apache commercial
release until the specific gaps are resolved. Publisher clarification of underlying
composition/rendering rights or a demonstrably licensed subset could address them.
Title recognition, MIDI hashes and public-domain composer names alone do not prove
that a particular transcription is authorized.

## Documented licensing routes

- Original, owned or explicitly first-party-authorized recordings: review composition,
  performance/recording and identifiable sample sources; retain actual versions,
  source pages and audio hashes.
- CC0 / CC BY recordings: commercial training is possible within the actual grant.
  Preserve CC BY authorship, source, license and necessary modification notices.
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en) covers rights
  the licensor owns and cannot supply another party's missing copyright permission.
- VCTK 0.92: the [University of Edinburgh record](https://datashare.ed.ac.uk/items/30e7453c-9ea8-48b4-8e18-f96d0dc62928/full)
  specifies CC BY 4.0 and authors Junichi Yamagishi, Christophe Veaux and Kirsten
  MacDonald. Recordings were downsampled from 96 kHz / 24-bit to 48 kHz / 16-bit for
  machine-learning use; newspaper text has Herald & Times Group permission. This is
  not public-domain speech. Retain recording attribution and source. SoundEx processes
  audio and does not publish reading texts or voice-identity models.

[Creative Commons' AI guidance](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/)
supports observing actual CC conditions during training and subsequent public use,
with traceable attribution. It is not a universal ruling on model-weight copyright.
CC data does not automatically assign CC or Apache licensing to new weights; review
whether a specific artifact retains protected upstream material and applicable terms.

## Formal release conclusion

The Apache-2.0 route requires review of the actual dataset combination,
initialization/parent provenance and final artifact. The historical fact inventory
records `rights_confirmed: false`; it does not supply a maintainer's approval.
Release requires the full LICENSE, applicable NOTICE, data attribution, model card
and exact hashes, plus checks for training-work reproduction on unrelated or silent
inputs. These are maintainer publishing tasks, not additional license restrictions
for downstream Apache-2.0 users. See the [weight license](../MODEL_WEIGHTS_LICENSE.md)
and [release checklist](MODEL_RELEASE_CHECKLIST.md).
