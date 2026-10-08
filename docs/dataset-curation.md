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

The reviewed regional pool currently contains 88 distinct works: China 29, Japan 1
and Korea 58 (including Travel). These counts do not describe the older unreviewed
candidate pool. Creator and genre dominance and remaining regional/genre/vocal gaps
still prevent the requested 1,000-work regional run from starting.

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

## Source and artifact review

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
