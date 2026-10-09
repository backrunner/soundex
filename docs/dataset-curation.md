# Regional music acquisition

Use free sources first. Paid libraries require separate approval of the library,
price and actual grant before purchase. Downloaded audio, signal quality, source
review and regional coverage are separate counts. A thousand files do not establish
a diverse approved corpus.

## Collection targets

[`curation_1000.yaml`](../training/configs/curation_1000.yaml) records initial
targets: 1,000 distinct recordings, 25 creators, 50 recordings each for
China/Japan/Korea and 100 each for Europe/North America. Each region needs three
creators and three genres. Pop, rock, electronic, country, folk, jazz, blues,
hip-hop, R&B and classical have coverage floors. No creator should exceed 10%
or genre 35% of the selected pool.

These are working acquisition targets, not a claim of comprehensive market
coverage. Chinese/Japanese/Korean vocal music has separate ten-song floors in
the corresponding language. Speech, Slakh mixes, effects, short loops and alternate
versions do not fill full-song or regional-vocal counts. Creator aliases share one
canonical ID; related versions share a work group before splitting.

Use primary repertoire/market evidence. A title, nationality or single instrument
does not establish repertoire. Unknown stays unknown; Chinese game BGM does not
automatically count as Mandopop.

## Actual acquisition on 2026-10-08

| Source | Acquired originals | Review and coverage limits |
| --- | --- | --- |
| [Heiyaoyao free pack](https://heiyaoyao.itch.io/heiyaoyaos-music-asset-cottage-free-commercial-music-library) | 37 original WAV files, 44.1/48 kHz | 34 recordings have source-bound reviews, representing 29 distinct compositions. Two game-homage works remain unresolved; one recording has a large DC offset. Per-volume credits include tohka on two collaborations. Instrumental game BGM does not establish mainstream pop or vocal coverage. |
| [KCC expansion](https://gongu.copyright.or.kr/gongu/wrt/wrt/view.do?menuNo=200020&wrtSn=13300271) | 100 original WAV/FLAC files downloaded | 53 orchestral works have linked original-score/composer checks and 4 other original-composer recordings have first-party rights-assignment records. Short instrument effects and carol arrangements do not inflate reviewed regional-song counts. |
| [KCC Travel](https://gongu.copyright.or.kr/gongu/wrt/wrt/view.do?menuNo=200020&wrtSn=13048721) | Stereo 44.1 kHz PCM16 WAV, 174.093 s, no signal flags; CC BY 4.0 | Source review retained. Page duration 258 s disagrees with both actual native and publisher preview, approximately 174 s. Aligned low-band correlation 0.9847 supports the same recording; stale metadata is not a hard audio failure. |
| [Yubatake Kawarayu](https://opengameart.org/content/kawarayu) | Stereo 48 kHz PCM24 FLAC, 292.656 s, no signal flags; CC BY 4.0 | Source review retained. Explicit Japanese enka/folk repertoire; creator nationality unknown. One instrumental track does not establish broad Japanese coverage. |
| [Chris Zabriskie](https://chriszabriskie.com/use/) | 30 previously acquired native FLAC masters | Exact work names match six official album pages. Artist blanket CC BY 4.0 grant and credit instructions retained alongside historical archive license receipts. His official composer/recording-artist page locates his working repertoire in Brooklyn, New York; one artist does not establish US genre breadth. Fifteen other legacy tracks remain outside this reviewed subset. |
| [Andy G. Cohen original masters](https://archive.org/details/andy-g-cohen-2014-2017) | 31 previously acquired native FLAC masters | Collection explicitly credits composition, performance, recording and mastering to the artist and grants CC BY 4.0. Every selected file matches publisher original-file MD5 and embedded creator credit. Collection tags support post-rock; regional origin remains unknown. FMA MP3 copies are not clean targets. |

The 2026-10-08 source-reviewed snapshot contained 149 distinct works: China 29, Japan 1,
Korea 58 (including Travel), North America 30 and unknown region 31. These counts
do not describe the older unreviewed candidate pool. Creator and genre dominance
and remaining regional/genre/vocal gaps still prevent the requested 1,000-work
regional run from starting.

The Chinese pack's [publisher terms](https://www.heiyaoyao.cn/announcements/import-notice/)
offer CC BY 4.0 as an alternative to custom terms. Keep the chosen grant and credits
bound to the acquired version. The itch.io “No AI” content field describes whether
generative AI created the assets; it is not itself a training prohibition.

KCC expansion discovery collects individually linked CC BY 4.0 original WAV/FLAC
candidates across popular, traditional, orchestral and ensemble categories.
Government hosting and category labels do not automatically resolve each work's
rights or establish its musical genre. The older acquisition pool remains useful
but is dominated by electronic music and lacks complete regional evidence.
The count-only queue was superseded before training. Regional vocal music, country
and other underrepresented genres still need more free originals and creators.

Read the work's own definition-list owner and contributor fields with
`data.publisher_metadata.gongu_work_fields`; recommended-work authors are unrelated
to the acquired recording. Reviews may correct `credited_author` and
`reviewed_genre`, while the original publisher/extraction values remain available
in the audit. Use `work_group` for explicitly shared melodies; five arrangements
are not five compositions. `content_kind` keeps licensed effects out of music counts.

The mostly-silent KCC *ROMANCE FOR TRUMPET* recording contains 353.850 seconds of
exact digital-zero tail padding. A lossless derivative removes only that tail,
retains one second after the last nonzero sample and has bit-identical retained
PCM16 samples. The derivative passed a new full decode and signal audit; its review
binds the derivative hash to the original publisher page and edit receipt. Do not
reject sparse music or intentional distortion merely because a flag exists.

Cleanup removed 437 files from this run's previously excluded creator catalog
(6,297,613,410 bytes), following the creator's
[express training objection](https://loyaltyfreakmusic.com/faq/). This corpus choice
does not assert retroactive revocation of historical CC0 grants. The duplicate
910,448,507-byte Chinese ZIP was also removed after every retained WAV matched its
archive member byte for byte. The padding repair replaced an oversized source copy
with its verified derivative, saving another 93,589,797 bytes. Download fragments
open by active collectors and retained baseline originals were protected; all 272
baseline hashes were checked again. Source pages, hashes, edit/deletion receipts
and useful instrument effects remain available locally.

## Readiness check, 2026-10-09

The live regional queue remains in `waiting-for-regional-source-selection`.
The updated acquisition/review snapshot has 1,384 signal-audited candidate entries,
but **277 distinct source-reviewed music works / 20 creators**, against the
requested approximately 1,000 works.
Only retained unique music works fill that target; supplements and variants do not.

| Reviewed repertoire | Distinct works |
| --- | ---: |
| China | 29 |
| Japan | 48 |
| Korea | 58 |
| Europe | 32 |
| North America | 72 |
| Unknown | 38 |

The reviewed genres are ambient 30, classical 55, country 5, electronic 56, folk 29,
funk 2, hip-hop 2, jazz 7, pop 2, rock 59 and unlabeled 30.
Blues and R&B remain absent; country has one creator, and two hip-hop works
from one creator do not establish broad coverage.

No Chinese/Japanese/Korean vocal music is counted. These acquisition gaps remain
even without the profile's
additional creator-share and per-category working thresholds.

Slakh was removed from the upcoming run on 2026-10-09. The retained VCTK speech
supplement has 270 utterances / 20,056 aligned pairs, using recipe
`7fda9e58ab583617`. Native mono input is supported; recording-balanced sampling
avoids multiplying a stereo source's draw mass by its channel roles. The queue
has no training child or checkpoints and will start fresh training after the
regional selection reaches its target. A waiting queue is not a started run.

Raw audio, processed pairs, source-page evidence, download plans and per-file
catalogs stay local and ignored. Only source documentation is published; see
[native sources](../training/source_catalogs/README.md). Public documentation
records a dated observation; the local job's `progress.json` is the live status.

The [Japanese source expansion](japanese-sources-20261009.md) acquired 42 original
WAV/FLAC masters: three Japanese creator compositions and six standalone JRPG
themes add nine reviewed full works. Thirty-three loops/cues/variants remain
supplements. Native mono and Japanese-style music are retained without inventing
Japanese regional or vocal coverage.

The [additional Japanese acquisition](japanese-additional-sources-20261009.md)
retains 46 original FLAC masters from kazunocobit, menogin and HarryArtz, adding
44 distinct works after grouping variants and keeping the short cue separate.
Two valid masters with DC offsets have measured, explicitly credited derivatives;
their originals remain intact. That Japanese-batch snapshot reported 37
remaining gaps. Raw downloads and private email delivery links remain local.

The [native Slakh replacement batch](slakh-replacement-sources-20261009.md) adds 55
reviewed works from free German and US publisher FLAC albums, including vocal
rock, acoustic/anti-folk and country-rock. The updated source/coverage audit
reports 34 gaps. Unique selected music totals 1132.55 minutes; VCTK adds
23.85 minutes, for 1156.40 source minutes across future train/validation/test splits.
Slakh contributes zero minutes to this run.

## Source and artifact review

The [additional creator-original survey](source-expansion-20261008.md) supplies a
separate 21-file free WAV plan with jazz-fusion, pop, funk and electronic material.
Its short country cue is retained but does not inflate full-song coverage. Three
valid float masters with slight over-unity peaks receive measured common stereo
gain derivatives rather than being discarded; originals and edits count once.
Live acquisition/review counts are in `data/reports/oga-original-expansion/`.

Retain primary pages/version receipts and hashes, audio hashes, composition and
recording basis, credits and identifiable performer/sample restrictions.
Do not require a separate AI letter for every CC work: [CC's FAQ](https://creativecommons.org/faq/)
explains that existing grants cover AI technologies when their conditions are met.
An uploader's license badge cannot grant someone else's rights. Concrete conflicts
stay pending; later statements do not automatically revoke historical CC grants.

This run's source assessment covers commercial codec-restoration training and
intended Apache-2.0 weight publication. It is not a legal guarantee or an automatic
copyright decision about weights. Review the exact artifact's provenance and
possible retained source material before release. Audio keeps its upstream terms
and is not bundled with the Apache model.
The exact future checkpoint still requires artifact and full-provenance release
review. Slakh is excluded from the current combination; the [historical Slakh MIDI/rendering review](../legal/WEIGHT_LICENSE_REVIEW.md) remains
incomplete. Source-reviewed music counts do not imply a released-model rights approval.

The generic folder/JSONL importer still accepts arbitrary audio and optional rights
metadata. This profile applies only to the requested regional run and adds no
downstream restriction to Apache-2.0.

## Optional regional audit

```bash
cd training
python scripts/audit_native_curation.py \
  --catalog ../data/reports/regional-pilot/acquired.jsonl \
  --reviews ../data/catalogs/regional-source-reviews.jsonl \
  --policy configs/curation_1000.yaml \
  --output ../data/catalogs/regionally-curated-YYYYMMDD.jsonl
```

Repeat `--catalog` for other audited receipts. Reviews bind recording IDs and
audio/evidence hashes. Missing/changed evidence, unresolved rights, insufficient
diversity or wrong-language vocals stay visible in the report. Incomplete audits
publish a report and no training catalog. Published catalogs are immutable;
use a new output path for each approved snapshot.

Add `--curation-policy configs/curation_1000.yaml` and
`--source-reviews ../data/catalogs/regional-source-reviews.jsonl` to
`scripts/wait_for_native_training.py` to watch acquisition. The queue freezes the
policy and waits for coverage/source review even after downloads finish. It passes
the policy and evidence root to the runner for another check before preprocessing.
Manual runs use `--curation-policy` and `--evidence-root` together. Run receipts
retain the actual policy hash and source review results.

Whole-file decode, native encoding/rate, PCM/work deduplication and signal checks
remain necessary. HTTP gzip is transport compression, removed before verifying
original bytes. Publisher size/checksums are checked when supplied and never
fabricated. Spectrum cannot certify historical absence of lossy transcoding.
