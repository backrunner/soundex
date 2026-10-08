# Free original-master expansion, started 2026-10-08

The local 21-file acquisition plan uses creator-published, uncompressed WAV
attachments. It stores publisher byte lengths, exact original-file URLs, credits and original style tags. This is a
download plan, not a statement that all files or a released model are approved.
All **21 originals completed download and full decoding on 2026-10-09**, totaling
1,130,364,100 bytes and 54.93 minutes. The local verified index records their
actual SHA-256, decoded PCM hashes and native measurements. Audio, download plans
and per-file catalogs stay outside Git; this source document remains public.
Local receipts are in `data/reports/oga-original-expansion/`.

## First-party original WAV collection

- **Zane Little: 20 originals under CC0**, including one short country-style cue.
  His [own artist account](https://opengameart.org/users/zane-little-music) and each
  work's own author/license/attachment fields were reviewed. The
  [Electronic Outlaw permission statement](https://opengameart.org/content/electronic-outlaw)
  explicitly allows reuse of his songs. The free WAV is distinct from optional
  paid deluxe stems, loops and jingles, which are not required or acquired.
- **Lennartmusic: one original complete mix**,
  [Beansjam Sad Budi Blues](https://opengameart.org/content/beansjam-sad-budi-blues).
  The creator credits his composition/arrangement and Pocket Piano instrument.
  Select the offered CC0 alternative; do not combine it with the page's other
  alternative licenses. Stems and intro variants are not additional songs.

Zane's creator-provided tags and descriptions cover pop/new wave, electronic,
chiptune, jazz fusion and funk. Original multi-label tags remain in the plan;
primary genre labels are only sampling categories. The
[Barriers description](https://opengameart.org/content/barriers) places his own
musical experience in the US. That supports this artist's North American
instrumental game-music repertoire, not a nationality or mainstream-market claim.
Lennartmusic's region and primary genre remain unknown.

[Miniature Saloon](https://opengameart.org/content/miniature-saloon) has explicit
country/honky-tonk tags, but its native recording is only 33.31 seconds. Keep it
as a useful cue; it does not fill the 1,000 full-song target or country-song quota.
After the three gain repairs below, this batch adds 20 distinct music works and
two creators. The reviewed music pool increases from 149 to **169** distinct works;
the requested 1,000-work regional/genre/vocal collection remains incomplete. Loop versions
and repeated releases of the same composition were excluded from the plan.

## Measured float-master headroom repair

The originals comprise 19 FLOAT and two PCM16 WAV files, 13 at 44.1 kHz and eight
at 48 kHz; 20 are stereo and one is mono. Complete decoding, finite samples,
size, hashes, silence, DC and
peak measurements are verified before a file is published locally. No MP3 export
is used as a clean target. First-party original-file provenance supports the
master assessment; the spectrum alone cannot establish historical codec use.

Three FLOAT originals have valid peaks slightly above 1.0. These float values do
not establish clipping or a damaged file. Preserve
their original bytes and use a uniform stereo attenuation before codec synthesis:

| Work | Gain | New peak | Largest all-frame rounding error |
| --- | ---: | ---: | ---: |
| Drive | -1.02169 dB | 0.95 | 2.98022e-8 |
| Empty Stretch | -1.13069 dB | 0.95 | 2.98020e-8 |
| Post-Adventure Tea Party | -1.01714 dB | 0.95 | 2.98023e-8 |

All three derivatives preserve frames, rate and channel relationships, pass a new
complete decode and have no unresolved signal flags. Error is below the native
float rounding budget of 1.13249e-7. No clipping, time cropping or resampling is
performed. Edit receipts bind original/derivative audio hashes, gain and measured
error; source reviews retain the creator's grant. Original and derivative share
one composition/split group and count once. The reusable
[gain utility](../training/data/master_gain.py) checks hashes and every output
sample, and refuses to replace an existing output.

## Other sources investigated

| Source | Actual finding | Current collection decision |
| --- | --- | --- |
| [BarkTunes](https://barktunes.jp/explanation.php) | Original Japanese pop/instrumental repertoire; limited karaoke WAVs, many MP3-only songs; additional unilateral use-stop terms and collaboration/remix exceptions | Preserve as a candidate. Resolve the chosen grant's scope before counting it as an unrestricted commercial-model source; exclude third-party remixes |
| [John Bartmann](https://www.johnbartmann.com/faq) / [Western Skies](https://johnbartmann.bandcamp.com/album/western-skies) | Country/bluegrass asset packs and lossless Bandcamp offer; current FAQ combines CC BY statements with paid commercial-use restrictions | Preserve the exact conflicting statements; do not automatically approve the whole catalog or presume that historical CC grants were revoked |
| [Scott Buckley](https://www.scottbuckley.com.au/library/donate/) | Free library generally provides MP3; original WAV access is a paid supporter benefit | Do not transcode the free MP3 into clean targets; no payment authorized |
| [Redafs country guitar](https://www.redafs.com/2018/09/chicken-picking-country-guitar.html) | Free CC BY 3.0 MP3; HQ WAV/MP3 is the paid offer | No free original WAV acquired; do not use the MP3 as a master |
| [Heartfelt Media Group](https://www.heartfeltmg.com/) | Own songs offered as CC BY 4.0 FLAC/WAV, with separately identified third-party songs. First-party production notes explicitly describe AI assistance | Hold pending actual production/provider rights and original-master provenance; no assumption that every FLAC is a native lossless production |
| [OtoLogic](https://otologic.jp/free/license) / [country BGM](https://otologic.jp/free/bgm/other01.html) | Clear CC BY 4.0 creator grant, but the inspected country BGM is provided as MP3 | Rights-positive candidate; inspected file does not satisfy the native-lossless target |
| [PeriTune](https://peritune.com/about/) | CC BY 4.0 remains for works through February 2026; newer works use custom terms. Free files are MP3/OGG/M4A; WAV is sold separately | Track the actual version and grant. No paid WAV bought; not all tracks share the same license |

Free sources remain the priority. Paid libraries need separate approval of the
specific material, price and training/weight-distribution grant. Genre/style names
do not replace original regional repertoire or language evidence. Chinese,
Japanese and Korean vocal music and full country songs still require expansion.

## Original source references

Each row selects the creator-offered CC0 alternative. Zane Little is credited on
the first 20 works; Lennartmusic is credited on the last. These publisher links
describe the acquired originals; gain derivatives and the short cue retain their
separate local edit/count receipts.

| Work / source page | Original WAV | Primary sampling genre |
| --- | --- | --- |
| [Miniature Saloon](https://opengameart.org/content/miniature-saloon) | [Publisher file](https://opengameart.org/sites/default/files/miniature_saloon.wav) | country |
| [The Way It Is](https://opengameart.org/content/the-way-it-is) | [Publisher file](https://opengameart.org/sites/default/files/the_way_it_is.wav) | pop |
| [Electronic Outlaw](https://opengameart.org/content/electronic-outlaw) | [Publisher file](https://opengameart.org/sites/default/files/electronic_outlaw.wav) | electronic |
| [Barriers](https://opengameart.org/content/barriers) | [Publisher file](https://opengameart.org/sites/default/files/barriers.wav) | jazz |
| [Detour](https://opengameart.org/content/detour) | [Publisher file](https://opengameart.org/sites/default/files/detour.wav) | jazz |
| [Freeway Fumes](https://opengameart.org/content/freeway-fumes) | [Publisher file](https://opengameart.org/sites/default/files/freeway_fumes.wav) | jazz |
| [Midnight Cruiser](https://opengameart.org/content/midnight-cruiser) | [Publisher file](https://opengameart.org/sites/default/files/midnight_cruiser.wav) | jazz |
| [Empty Stretch](https://opengameart.org/content/empty-stretch) | [Publisher file](https://opengameart.org/sites/default/files/empty_stretch_0.wav) | jazz |
| [Drive](https://opengameart.org/content/drive-0) | [Publisher file](https://opengameart.org/sites/default/files/drive.wav) | jazz |
| [Shortcuts](https://opengameart.org/content/shortcuts) | [Publisher file](https://opengameart.org/sites/default/files/shortcuts.wav) | electronic |
| [Wednesday Night Funk Fusion](https://opengameart.org/content/wednesday-night-funk-fusion) | [Publisher file](https://opengameart.org/sites/default/files/wednesday_night.wav) | funk |
| [Rhythm Factory](https://opengameart.org/content/rhythm-factory) | [Publisher file](https://opengameart.org/sites/default/files/rhythm_factory.wav) | jazz |
| [Synesthesia](https://opengameart.org/content/synesthesia) | [Publisher file](https://opengameart.org/sites/default/files/synesthesia.wav) | electronic |
| [Poker Night](https://opengameart.org/content/poker-night) | [Publisher file](https://opengameart.org/sites/default/files/poker_face.wav) | electronic |
| [Backup Plan](https://opengameart.org/content/backup-plan) | [Publisher file](https://opengameart.org/sites/default/files/backup_plan.wav) | electronic |
| [Sideways City](https://opengameart.org/content/sideways-city) | [Publisher file](https://opengameart.org/sites/default/files/sideways_city.wav) | electronic |
| [The Cool Factor](https://opengameart.org/content/the-cool-factor) | [Publisher file](https://opengameart.org/sites/default/files/the_cool_factor.wav) | electronic |
| [Post Adventure Tea Party](https://opengameart.org/content/post-adventure-tea-party) | [Publisher file](https://opengameart.org/sites/default/files/post-adventure_tea_party.wav) | electronic |
| [Moms Workout Cd](https://opengameart.org/content/moms-workout-cd) | [Publisher file](https://opengameart.org/sites/default/files/moms_workout_cd.wav) | electronic |
| [Space Cadet Training Montage](https://opengameart.org/content/space-cadet-training-montage) | [Publisher file](https://opengameart.org/sites/default/files/space_cadet_training_montage_0.wav) | electronic |
| [Beansjam Sad Budi Blues](https://opengameart.org/content/beansjam-sad-budi-blues) | [Publisher file](https://opengameart.org/sites/default/files/sad%20budi%20blues.wav) | unlabeled |

## Acquire originals from a local plan

Create an optional local plan from the publisher references above; it is not
included in a fresh clone. From `training/`, with that plan:

```bash
python scripts/fetch_archive_masters.py \
  --plan ../data/catalogs/oga_originals_plan.jsonl --root ../data \
  --receipt-dir ../data/reports/oga-original-expansion --workers 3
```

Use acquired receipts for the source/coverage review. Store any gain derivative
in a separate catalog, retain the original SHA-256 and edit receipt, and preserve
the original composition group. Flags trigger an appropriate review or repair;
they are not automatically a declaration that music is defective.

The local verified index intentionally preserves the three originals' over-full-scale
flags. They require their separate gain derivatives for this selection; do not
silently clear those measurements or distribute derivative hashes as though they
were hashes of the publisher download.

These first-party source assessments do not resolve the separate Slakh MIDI
rights gap and do not approve an unreleased trained model. See the
[actual weight-license findings](../legal/WEIGHT_LICENSE_REVIEW.md).
