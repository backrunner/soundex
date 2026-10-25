# P55: source-disjoint long-form spectral review and blinded listening

## Purpose and frozen scope

The [P54 metric audit](spectral-training-p54.md#metric-sensitivity-audit) exposed
an unstable near-tie and only 14.75 seconds of scored music. This evaluation tests
the same restoration question on longer, simultaneous stereo excerpts. It changes
neither P54 training nor the interpretation of its already registered comparison.
No model is selected or promoted by a new favorable scalar metric.

Current review: `data/runs/longform-review-p55-20261024-v2/`; original prepared
audio and isolation evidence: `data/runs/longform-review-p55-20261024/`. Entry point:
`training/scripts/run_longform_review.py`. All source audio, paired material,
rendered audio, private listening keys and experimental results remain ignored.

Install optional review dependencies with
`python -m pip install -r training/requirements-evaluation.txt -c training/constraints-cpu-py312.txt`
after the documented PyTorch installation. SciPy 1.16.3 supplies resampling and
the listening anchor. CI installs this extra file; the requirements and constraints
bound to the live P54 training run are unchanged. The local environment already
has this exact SciPy version, so no live training environment was upgraded.

The fixed native model set is historical shape-small, identity DSP, both exact
P50 parents of P54, and all four final P54 continuations (two seeds, common versus
auraloss objective). The same frozen P54 Rust diagnostic binary renders every
model. Its original ModelGain/SmoothKnee pipeline, native detector conditions and
per-channel state remain unchanged. All eight native paths and their raw overlap-add controls have now been scored.

The supervisor waited for verified P54 `complete.json` **and**
`runtime-complete.json`, checked the exact four endpoint labels, and froze all
eight graph hashes in `endpoints.json` before opening the new scoring cases.
This kept long-form evaluation out of P54's registered timing run. The
listening pack was generated after the complete long-form objective review.

Before any model scoring, execution revision 2 added another prepared-audio hash
check after the waiting barrier. The original registration, exact old runner source
and supersession record are retained. Sources, audio bytes, crops, models, metrics
and listening randomization are unchanged. V2 references the verified original
paired files without duplicating 1.53 GB of audio. A regression test simulates PCM
changing during the wait and confirms scoring stops before model review.

## Source isolation and coverage

The local approved catalog contains 360 recordings. Exclusion against 16 retained
processed manifests and non-snapshot run-specific catalogs removes 272 through
file hashes, decoded PCM hashes and transitive work aliases. Even historical
catalog entries without demonstrated optimizer exposure are excluded. All 88
remaining candidates pass the cached publisher-license/original-file checks.

Selection is fixed before model scoring: rare catalog genres first, deterministic
source-hash order, at most four recordings per genre and three per exact author
label. The resulting 20 recordings have 13 author labels and are all stereo:

| Catalog category | Recordings |
| --- | ---: |
| Ambient | 4 |
| Electronic | 4 |
| Rock | 3 |
| Metal | 2 |
| Pop | 2 |
| Folk | 1 |
| Unlabeled | 4 |

The source-grant evidence binds the catalog to publisher metadata, the original
file MD5, a supported CC BY 3.0/4.0 or CC0 declaration, and the actual local SHA256.
Complete masters were decoded again, checked for finite samples and existing
quality flags, and matched by decoded PCM hash. Spectrum is not proof of lossless
mastering history. Source grants do not constitute a new legal certification of
every upstream right or model weight. These sources are evaluation-only.

A supplemental search for all selected file/PCM hashes in retained non-snapshot
run/report/archive JSON and JSONL found only prior acquisition metadata outside
this new run, not previous model scores. `source-isolation.json` records that
audit. This supports a new source/work-disjoint review against the retained
history; it does not prove artist independence or knowledge of unretained history.

All region labels are unknown in this catalog. There is no verified China/Japan/
Korea coverage, and no classical, jazz, country, blues or hiphop category in this
selection. Four unlabeled recordings are not four additional genres. This is a
bounded music pilot, not universal musical or geographic generalization. P54's
separate speech rechecks remain necessary; this set adds no fresh speech evidence.

`evaluation-catalog.jsonl` reserves these recordings locally. Future training or
calibration must exclude their file/PCM/work aliases. After this review, reuse
must be labeled a recheck, not another untouched test.

## Paired audio and timeline

Each recording supplies two source positions, at 25% and 65% of available
13-second support. Each has two seconds of real prior audio, ten scored seconds,
and one second of codec/tail guard. The first is resampled to 44.1 kHz and the
second to 48 kHz before codec degradation. Both physical channels always come
from the same source time. There is no per-condition gain normalization.

Each excerpt has MP3 64 kbit/s, AAC-LC 96 kbit/s, Vorbis q3 and clean-identity
conditions. This yields **160 cases: 120 paired and 40 identity**, or 1,200 paired
codec seconds plus 400 identity seconds per model path. Merging source intervals
confirms **400 independent scored source seconds**, not 1,600 independent seconds.
All 160 cases have zero estimated codec offset and zero residual offset on both
physical channels. The encoder path/version, recipe and every prepared file hash
are retained. The prepared set occupies about 1.53 GB in decimal bytes.

The Rust diagnostic output and raw overlap-add both have a 128-sample timeline
offset; comparison slices explicitly remove it. The production fixed 256-frame
adapter buffer is a separate delivery property. Model history, codec guard audio
and the two-second prelude are excluded from supervision/scoring. These larger
evaluation windows add no model lookahead.

## Metrics and aggregation

The diagnostic primary family is original-reference high-band amplitude,
log-magnitude and reference-following dynamics, with equal 8-12/12-16/16+ kHz
weights and the unchanged 256/1024/2048 analysis resolutions. Retain every band,
full MR, auraloss, absolute reference/error RMS, full waveform error and stereo
mid/side error. Whole-excerpt one-sided Parseval rulers use calibrated PCM RMS;
they are not unnormalized FFT power or audible-damage thresholds.

For every case, collect untreated input, native model-plus-DSP output and raw
model prediction synthesized from that same Rust run's prediction trace. Thus
Python frontend differences cannot masquerade as a model/DSP quality difference.
Successful predictions are retained compactly as float32 NPZ and raw PCM; redundant
verbose frontend/context trace JSON is removed only after the successful render
receipt. Failed evidence is retained. These trace runs are not timing benchmarks.

Average physical channels, then excerpts/codecs within each source, then sources
equally. Also report clip means and genre/codec/rate/channel strata. Paired source
bootstrap intervals use 5,000 replicates and a fixed seed; they are descriptive
marginal intervals, without a multiple-comparison-adjusted superiority claim.
Keep both historical and untreated comparisons. Better clean-input preservation
cannot substitute for paired high-band recovery, and better-than-historical
does not necessarily mean better-than-untreated.

## Listening materials and remaining evidence

Use one predetermined 48 kHz MP3 excerpt per source. The local page includes the
shown reference, randomized hidden reference and 3.5 kHz anchor, untreated input,
identity DSP, historical weights and all four final candidates. Each trial uses
one common playback gain for all conditions; numerical scores retain original
unscaled PCM. Answers remain outside the listener folder. The page records
reference similarity, high-frequency naturalness, no-audible-difference choices,
artifact notes, skip reasons and playback setup, and exports ratings locally.

This is a reference-based blinded study informed by
[ITU-R BS.1534](https://www.itu.int/rec/R-REC-BS.1534-3-201510-I/en), not a claim of
completed standard-compliant testing. Listener eligibility, panel size, exclusions
and uncertainty still need a real listening record. No simulated votes are filled
in. [ViSQOL audio mode](https://github.com/google/visqol) is not installed or scored
by this change; an optional later diagnostic needs its own pinned model/version
and cannot replace stereo listening or become a retrospectively chosen winner.

## Verified preparation status

- All 20 complete masters and 160 paired/identity cases prepared and hash-bound.
- Ten targeted tests pass: transitive isolation, deterministic sampling,
  publisher evidence, recording weighting, raw OLA timeline/channel independence,
  exact-reference spectral behavior, calibrated RMS, the endpoint barrier and
  blinded common-gain export.
- The complete local Python suite passes: 761 tests, 233 warnings, 68.31 seconds.
  Ruff passes for 343 Python files; English documentation/link/index checks pass
  for 133 Markdown files. All live P54 source/input/dependency hashes still match.
- A real frozen Rust synthetic-stereo replay validates native output and raw trace
  scoring over ten seconds. PCM error RMS is approximately 3.32e-9 native and
  3.18e-9 raw. This is transport/alignment evidence, not music restoration quality.
- The listening template passes JavaScript syntax checking. Browser UI validation
  is pending because no browser surface was available; there are no human ratings.
- The complete objective review and blinded audio pack are available. No human
  listening verdict or new weight qualification is claimed.

The bounded coverage and missing listening/device evidence prevent a release
conclusion. An inconclusive result must remain in the experiment registry; do not
change source weighting, substitute tracks or select one seed after seeing scores.

## Evaluation source credits

Publisher declarations and exact original-file entries are bound in the local
registration. These recordings retain their source licenses; they are not
relicensed as Apache-2.0. Local evaluation makes cropped, resampled, codec-degraded,
enhanced and anchor-filtered excerpts, with common playback gain. No artist
endorsement is implied. Audio is not committed or publicly redistributed.

| Original recording | Attribution | Publisher source | Declared license |
| --- | --- | --- | --- |
| anu014_litmus0001_004_weakness_pt_1 | litmus0001 | [Publisher](https://archive.org/details/anu014_litmus0001_east_bay_sunrise) | [CC BY 3.0](http://creativecommons.org/licenses/by/3.0/) |
| _-_Frankenstein | Balthasar | [Publisher](https://archive.org/details/OMR002) | [CC BY 3.0](http://creativecommons.org/licenses/by/3.0/) |
| _-_Dracula_pt_1 | Balthasar | [Publisher](https://archive.org/details/OMR002) | [CC BY 3.0](http://creativecommons.org/licenses/by/3.0/) |
| Poofodeibohr | Pixel Nya | [Publisher](https://archive.org/details/nya5-pixel-nya-hindi-poofodeibohr) | [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/) |
| Hindi | SONiKA | [Publisher](https://archive.org/details/nya5-pixel-nya-hindi-poofodeibohr) | [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/) |
| 04 Instalación Tristeza Pérez 2000 | Intérprete desconocido | [Publisher](https://archive.org/details/XaviCorbera-PetitsFragmentsNocturns) | [CC BY 3.0](http://creativecommons.org/licenses/by/3.0/) |
| 02 The Ambient Cat | Intérprete desconocido | [Publisher](https://archive.org/details/XaviCorbera-PetitsFragmentsNocturns) | [CC BY 3.0](http://creativecommons.org/licenses/by/3.0/) |
| Gurdonark - Cave | Gurdonark and Lee Rosevere | [Publisher](https://archive.org/details/Gurdonark_LeeRosevere_Harmony) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| LR - Anthropomystery | Gurdonark and Lee Rosevere | [Publisher](https://archive.org/details/Gurdonark_LeeRosevere_Harmony) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| 02 Las inéditas ventajas del puerta | Lautaro Barcelo, Pablo Mati­as Vidal, Pablo La Ferrara, German Pasalagua, Juan Bautista Barcellandi | [Publisher](https://archive.org/details/OrquestaDePerros-RolesYOficios) | [CC BY 3.0](http://creativecommons.org/licenses/by/3.0/) |
| 12 Velocidad Roja | Lautaro Barcelo, Pablo Mati­as Vidal, Pablo La Ferrara, German Pasalagua, Juan Bautista Barcellandi | [Publisher](https://archive.org/details/OrquestaDePerros-RolesYOficios) | [CC BY 3.0](http://creativecommons.org/licenses/by/3.0/) |
| 10 En el camino | Lautaro Barcelo, Pablo Mati­as Vidal, Pablo La Ferrara, German Pasalagua, Juan Bautista Barcellandi | [Publisher](https://archive.org/details/OrquestaDePerros-RolesYOficios) | [CC BY 3.0](http://creativecommons.org/licenses/by/3.0/) |
| Purple Fiction | Crystal Dreams | [Publisher](https://archive.org/details/bof075) | [CC BY 3.0](http://creativecommons.org/licenses/by/3.0/) |
| Micrometer | Gurdonark | [Publisher](https://archive.org/details/Weights_And_Measures) | [CC BY 3.0](http://creativecommons.org/licenses/by/3.0/) |
| Dust Time Gravity | The Ghost of an Alien | [Publisher](https://archive.org/details/Dust-Time-Gravity) | [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/) |
| Two and One Half Miles | Gurdonark | [Publisher](https://archive.org/details/Weights_And_Measures) | [CC BY 3.0](http://creativecommons.org/licenses/by/3.0/) |
| Can You Feel feat. Irmina Lastovskaya | Mesake | [Publisher](https://archive.org/details/wave020) | [CC BY 3.0](http://creativecommons.org/licenses/by/3.0/) |
| reaching | raymundo mendoza | [Publisher](https://archive.org/details/unfound53) | [CC BY 3.0](http://creativecommons.org/licenses/by/3.0/) |
| Under the ground feat. Irmina Lastovskaya | Mesake | [Publisher](https://archive.org/details/wave020) | [CC BY 3.0](http://creativecommons.org/licenses/by/3.0/) |
| Coereave - Dirty Thief | James Lowery | [Publisher](https://archive.org/details/volumevii_201401) | [CC BY 3.0](http://creativecommons.org/licenses/by/3.0/) |

## Completed objective results

All eight model/control graphs completed 160 cases each, with native and raw
scores plus untreated controls: 2,720 score records. `complete.json` binds the
frozen endpoints, summary, strata, outputs and listening materials. The following
numbers average physical channels and cases within each source before giving
the 20 sources equal weight. Lower error is better; these are numerical music
outcomes, not listening scores.

| Native paired path | MR | HF8k MR | Auraloss-band diagnostic |
| --- | ---: | ---: | ---: |
| Untreated | 0.51921873 | 0.58049597 | 0.50617539 |
| Identity DSP | 0.51721536 | 0.57775399 | 0.50547090 |
| Historical shape-small | 0.49981362 | 0.55207529 | 0.49503353 |
| Seed 12 common | 0.51197556 | 0.57034170 | 0.50351613 |
| Seed 12 auraloss | 0.51201827 | 0.57039861 | 0.50352087 |
| Seed 13 common | 0.51096214 | 0.56891670 | 0.50321660 |
| Seed 13 auraloss | 0.51098866 | 0.56895465 | 0.50321964 |

All four new endpoints improve MR over untreated by 1.39-1.59%, but remain
2.23-2.44% worse than historical weights; HF8k MR remains 3.05-3.32% worse.
Each has lower MR than historical weights on 8/20 sources and higher error on
12/20. Descriptive paired-source 95% bootstrap intervals for the native MR
difference versus historical weights are entirely positive: common seed 12
[0.003927, 0.020078], auraloss seed 12 [0.003966, 0.020125], common seed 13
[0.003201, 0.018762], auraloss seed 13 [0.003229, 0.018800]. These marginal intervals
are not multiplicity-adjusted proof of universal superiority or inferiority.
The new source/work-disjoint evidence does not support the hypothesis that current
models already surpass the old baseline but only the previous weighting hid it.
It still cannot establish that the baseline never overfits or wins perceptually.

The newer models retain tradeoffs: native reference dynamics error is 1.57-1.60%
lower than historical weights, and clean-identity MR error is 76.23-77.84% lower.
The latter measures reduced damage on already lossless audio, not restored lossy
content. Native paired HF8k PCM error RMS is 0.008735-0.008737, below historical
0.008959 but above untreated 0.008725. Side error RMS is 0.012447 versus historical
0.012502 and untreated 0.012464. Better phase-sensitive PCM error and preservation
do not override the still worse high-frequency spectral match or prove audibility.

Raw current MR is 0.511324-0.512635 and native MR is 0.510962-0.512018: on these
sources DSP slightly improves spectral error. This differs from the older short
TRAIN review, where DSP worsened it. Therefore DSP cannot be designated the sole
bottleneck. Historical raw MR is 0.744517 before its intended fusion, so comparing
only raw outputs would misrepresent the delivered historical baseline.

Auraloss provides no extra common-MR benefit over the matched common-loss control
at either seed, and neither new objective closes the historical gap. Keep all
per-source, codec, rate, frequency-band and identity counterexamples in `summary.json`
and `strata.json`. These exposed recordings remain reserved; future reuse is a
recheck, never an untouched test. The [P56 diagnosis](trainability-audit-p56.md)
uses only the prior TRAIN roles, excluding these reserved aliases.

The 20-trial local blinded pack is complete under the review's `listening/`
directory, including lossless reference, hidden reference, anchor and all registered
conditions. Human ratings and real-device qualification remain pending. No default
weight was replaced and no subjective improvement is claimed.
