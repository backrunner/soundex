# Japanese music sources, 2026-10-09

Free creator-published WAV/FLAC masters are preferred. Original audio, acquisition
plans, source-page copies, package licenses and per-recording measurements remain
local and ignored. This document records source references and measured results;
it does not bundle audio or relicense it under Apache-2.0.

## Japanese creator: Mochi no Okome

The [creator's own profile](https://mochi-no-okome.itch.io/) identifies a music
creator working in Japan and describes all works as original. The three work pages
and their downloaded package READMEs consistently grant CC BY 4.0, including
commercial use, modification and redistribution with credit.

| Original work | First-party source | Measured duration | Native master |
| --- | --- | ---: | --- |
| 前へ、すすむぞっ | [Work page](https://mochi-no-okome.itch.io/mae-e-susumuzo-bgm) | 152.034 s | WAV, PCM24, 48 kHz, stereo |
| 新しいあした / 新しい明日 | [Work page](https://mochi-no-okome.itch.io/atarasii-asita) | 209.000 s | WAV, PCM24, 48 kHz, stereo |
| thanks for long time | [Work page](https://mochi-no-okome.itch.io/thanks-for-long-time-bgm) | 301.000 s | WAV, PCM24, 48 kHz, stereo |

These are three distinct instrumental game compositions, totaling 662.034 seconds.
Each source ZIP passed every member's CRC check; retained WAV bytes match the
original members. Full-file decode, finite samples, frame count and source/PCM
SHA-256 checks passed. No sustained full-scale, DC, excessive-silence, very-quiet
or over-unity signal flags were raised. MP3 exports were not extracted.
Package licenses, creator/source-page versions and audio hashes are bound in the
local source reviews. Credit: **Music by Mochi no Okome**, with the source and
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) link; describe later edits.

The Japanese-language creator catalog and its original compositions support
Japanese artist repertoire. They do not establish J-pop, Japanese lyrics or human
singing coverage. Their primary genre remains `unlabeled`: an RPG use case alone
is not evidence for pop or rock. Frequency content varies, including a dark
low-bandwidth ending piece; this alone is not proof of a lossy upstream export.
Publisher master provenance is retained, while historical mastering is not
independently certified by a spectrum or lossless container.

## Japanese-style instrumentals

| Creator and source | Selected original attachment | Grant | Repertoire limit |
| --- | --- | --- | --- |
| Yubatake, [Rainbow Rush](https://opengameart.org/content/rainbow-rush) | [FLAC/OGG/MIDI ZIP](https://opengameart.org/sites/default/files/iabo.zip); retain only FLAC | CC BY 4.0 | Creator describes para-para synth dance; original member is a loop, with a separate title jingle. Both remain supplements. |
| Yubatake, [JRPG Collection](https://opengameart.org/content/jrpg-collection) | [Original FLAC ZIP](https://opengameart.org/sites/default/files/jrpgcollection_flac.zip) | CC BY 4.0 | Game themes and short event cues; supplied MIDI and synthesizer patches support original production provenance. |
| Yubatake, [JRPG Collection 2](https://opengameart.org/content/jrpg-collection-2) | [Original FLAC ZIP](https://opengameart.org/sites/default/files/jrpgcollection2_flac.zip) | CC BY 4.0 | Contains explicit loops and related eerie variants; variants share a work group. |
| Kistol, [Hot Springs Town](https://opengameart.org/content/hot-springs-town) | [Original FLAC](https://opengameart.org/sites/default/files/hot_spring_town.flac) | CC0 | Creator-made town BGM with Japanese/koto tags; described as a looping piece. |

Creator nationality/working region is not established for Yubatake or Kistol.
Japanese-style music remains useful, but JRPG, a Japanese instrument or a title
alone does not put these works into the native Japanese repertoire count.
Keep cues, explicit loops and related versions separate from distinct full-work
counts. Never promote compressed OGG/MP3 siblings to clean references.

The four sources yielded **39 original FLAC files**, in addition to Mochi no
Okome's three WAVs: **42 masters / 421,410,135 bytes / 51.576 minutes** in this
batch. All 42 passed full decode, native format, finite-sample, frame-count and
source/PCM checksum checks, without signal review flags. Whole-archive CRC checks covered every retained ZIP; publisher attachment sizes
were checked where available before deleting the redundant ZIP. Thirty-six JRPG files are native mono PCM24 at 48 kHz; the
other FLACs are stereo, including Kistol's PCM16 44.1 kHz original.

A production-loader/encoder check used actual four-second crops from the new
mono JRPG main theme and a new stereo Mochi composition. MP3 64 kbps, AAC 96 kbps
and Vorbis quality 3 passed at both 44.1 and 48 kHz: 12 cases, channel counts
preserved and residual alignment zero. This is a representative preparation
check, not completed training preprocessing. It uses the managed FFmpeg described
in [workspace tooling](workspace.md), which supplies all configured encoders.

Six standalone JRPG themes count as distinct full works: `JRPG_temple`,
`JRPG_princess`, `JRPG_mainTheme`, `JRPG_battleFinal`, `JRPG_mysticIsle` and
`JRPG_tavern`. Combined with the three Japanese creator compositions, this adds
**9 distinct source-reviewed full works**. The other **33 loop/cue/variant files**
remain local supplements and do not increase this run's full-work target. This
includes the 158.675-second Rainbow Rush loop and the 160-second Hot Springs Town
piece intended for looping. The 38.736-second victory cue also stays a cue;
duration alone does not turn it into a full composition. Eight related eerie
variants share one work group.

The reviewed pool therefore becomes **178 distinct full works / 13 creators**,
with Japan **4**, China 29, Korea 58, North America 49 and unknown repertoire 38.
Six new Japanese-style themes retain unknown region; they do not fill the Japan
quota. Japanese vocal coverage remains zero. Cue and loop grants are preserved,
but this waiting regional queue selects only the nine new full works for eventual
music preparation.

## Singing and electronic candidates not included

- [jaCappella's official terms](https://tomohikonakamura.github.io/jaCappella_corpus/)
  describe 50 Japanese vocal-ensemble songs across ten genre subsets, recorded in
  a studio at 48 kHz, with native Japanese singers and mono WAV parts. The free
  grant prohibits commercial use; commercial licensing is chargeable. No corpus
  audio was downloaded or mixed into this training route. A paid grant needs
  separate confirmation before purchase.
- [PJS's official page](https://sites.google.com/site/shinnosuketakamichi/research-topics/pjs_corpus)
  offers 100 short singing and 100 short speech recordings at 48 kHz under
  CC BY-SA 4.0, explicitly allowing commercial use. These are short utterances,
  not 100 full songs. PJS is a possible vocal supplement; the share-alike
  obligations and intended weight artifact still need an assessment before
  combining it with the planned Apache publication. It is not an academic-only
  dataset, and CC BY-SA does not by itself decide the copyright status of weights.
- R-9's [first-party Neo-Saitama announcement](https://note.com/epxstudio/n/n5a17d92a0194)
  links seven newly composed electronic instrumentals under CC BY 4.0. The linked
  WAV loop ZIP returned HTTP 404 on this check; no WAV was acquired. The creator
  explicitly describes an existing bassline motif and an orchestral-hit source
  for one track, so that recording's third-party material remains unresolved.
  This candidate is not counted as approved material or as a received album.

CC grants can cover AI uses within the granted rights; see
[Creative Commons' AI guidance](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/).
Our source assessment covers commercial codec-restoration training and intended
Apache weight distribution, preserving attribution. It is not automatic approval
of every eventual checkpoint: review the actual artifact for retained source
material and the complete corpus provenance. The separate
[Slakh weight-rights findings](../legal/WEIGHT_LICENSE_REVIEW.md) still apply.

See [regional targets and current readiness](dataset-curation.md). This expansion
adds Japanese instrumentals; regional vocal music and wider Japanese genre and
creator coverage remain incomplete.

The [subsequent same-day acquisition](japanese-additional-sources-20261009.md)
adds creator-published albums by kazunocobit and menogin, plus four original
HarryArtz instrumentals. The counts above describe this earlier batch;
current coverage is reported in the regional readiness document.
