# Additional Japanese music sources, 2026-10-09

This follows the [earlier Japanese acquisition](japanese-sources-20261009.md).
Use creator-published lossless downloads, retain individual credits and source
versions, and keep original audio and private delivery links outside Git.
An audio grant does not relicense the recording under SoundEx's Apache-2.0 license.

## kazunocobit: original electronic album

[Retroscape BGM collection](https://kazunocobit.bandcamp.com/album/retroscape-bgm-collection)
contains 13 remastered compositions created by the publisher in 2025. The creator's
own catalog identifies Osaka, Japan and electronic, electronica, techno, synthwave
and retrowave styles. The matching
[BOOTH release](https://booth.pm/ja/items/8180341) explicitly grants commercial use,
adaptation, remixing and sampling under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The official Bandcamp zero-price route does not require email or an account. Its
FLAC files measure PCM24, 48 kHz, stereo. BOOTH describes a different WAV export
at PCM16, 44.1 kHz; that description is not substituted for the files actually
received from Bandcamp. Compressed previews and MP3 alternatives are not targets.

The album's full published compositions last approximately two minutes each.
Being suitable for repeated playback does not itself make a complete published
composition an extracted short loop. Retain all 13 masters, but group
`Explore(sazanami)` and `Explore(seseragi)` under one composition/split group;
their independence is not established, so they contribute at most one work.
Credit **Music by kazunocobit**, link the album and CC BY 4.0, and identify edits.
Artwork by 桃治いの (2iUno) is separate. Cover files are not extracted;
embedded picture metadata is not a model input. No artwork grant is inferred
from the music license.

## menogin: original computer-created instrumentals

The creator's own catalog identifies Tokyo, Japan and computer-created music.
Each selected album links CC BY 4.0 and offers zero-price FLAC without email:

| Album | Original files | Source |
| --- | ---: | --- |
| music data | 7 | [Album](https://menogin.bandcamp.com/album/music-data) |
| Music Data 2 | 7 | [Album](https://menogin.bandcamp.com/album/music-data-2-2) |
| miscellaneous | 9 | [Album](https://menogin.bandcamp.com/album/miscellaneous) |
| miscellaneous 2 | 6 | [Album](https://menogin.bandcamp.com/album/miscellaneous-2) |

The generic album tag is electronic; more specific first-party track descriptions
are retained rather than inferring genres from titles. `street` is described as
Boom-bap and `crimson` as a hip-hop track. `untitled` and `Jack-o'-Lantern` identify
classical style; this is not evidence of an acoustic orchestral recording.
`祭・fes-beat` identifies Bhangra/folk/dance, which supports folk fusion by a
Japanese artist, rather than Japanese traditional folk. `spaceship` identifies
trance; `agitation` is a hip-hop/DnB/club hybrid. A title such as `countryside`
does not establish country music. The eight-second `nightingale` stays a short
supplement and does not fill the full-work target.

Credit **Music by menogin**, link the applicable album and CC BY 4.0, and describe
edits. Keep the original album and individual track-page license/description
versions with the exact retained audio hashes.

## HarryArtz: four free original instrumentals

The creator's own catalog identifies Kanagawa, Japan. These individual work pages
credit writing and production to HarryArtz, link CC BY 4.0 and expressly allow
commercial and noncommercial use with credit:

| Work | First-party source | Publisher master declaration |
| --- | --- | --- |
| HarryArtz - Beyond Despair | [Track](https://harryartz.bandcamp.com/track/harryartz-beyond-despair) | 24-bit / 48 kHz |
| VIXI³ ÷ II = Final Regret | [Track](https://harryartz.bandcamp.com/track/vixi-ii-final-regret) | 24-bit / 44.1 kHz |
| Chronos | [Track](https://harryartz.bandcamp.com/track/chronos) | 24-bit / 44.1 kHz |
| ΑΠΟΛΛΩΝ | [Track](https://harryartz.bandcamp.com/track/-) | 24-bit / 44.1 kHz |

Use the **Free Download** route, rather than a paid checkout. It emails the
download link and requires joining the author's mailing list; there is no optional
subscription checkbox in this flow. Private delivery URLs and postal fields
remain local. Unsubscribe requires an actual email link or a request to the
publisher; completing an audio download is not evidence of removal from the list.
Credit **HarryArtz — track title**, source URL and CC BY 4.0; identify edits.

The author's two Synthesizer V Mai vocal releases are not part of this selection.
Their voicebank training/rendering permissions need a separate source assessment;
the instrumental grants are not applied to those vocal recordings by analogy.

## Candidates held or excluded

- [IdolSongsJp's dataset license](https://huggingface.co/datasets/imprt/idol-songs-jp/blob/main/LICENSE.md)
  permits free noncommercial academic research and requires permission for
  commercial research. It separately prohibits training on instrumental signals,
  including masters and stems. Permission for vocal ML does not remove the
  overall research restrictions. The paper's license is not the audio grant;
  no corpus audio is included.
- [Keynata Commons](https://carf-coder.github.io/keynata-commons/) offers CC0
  procedural compositions, but its
  [third-party soundfont notice](https://carf-coder.github.io/keynata-commons/LICENSE-3RD-PARTY.md)
  includes GeneralUser GS's explicit uncertainty about some underlying sample
  origins. Treat it as a pending rendered-audio source; CC0 composition/MIDI does
  not resolve every renderer sample, or establish Japanese human vocal repertoire.
- [mastin.channel's source guide](https://mastinchannel.com/free-music-guide/)
  identifies its catalog as AI-generated and permits commercial uses. Providing
  WAV alone does not establish the upstream original output/master encoding or
  the upstream service's sublicensable training rights. Keep it pending until
  those points are established. AI generation alone is not proof of lossy audio.
- kazunocobit's other checked releases require payment, use a noncommercial
  license, or reserve all rights. They are not covered by the selected free
  album's grant and were not purchased.

## Measured result and release scope

The batch contains **46 original FLAC masters / 1,047,380,001 bytes /
91.921 minutes**, plus two explicitly documented signal derivatives. All originals
passed byte integrity and full-file decode checks. Forty-three originals are
PCM24 at 48 kHz and three HarryArtz recordings are PCM24 at 44.1 kHz; all are stereo.
The five source ZIPs passed every member's CRC, and extracted audio bytes match
their original members. Separate artwork files were not extracted.
After rechecking all retained original-member hashes and confirming no process
held the archives open, the five redundant ZIPs were removed, saving 814,434,974
bytes. Original FLAC masters, source evidence and deletion receipts remain local.

Two menogin originals have measured DC offsets: `Jack-o'-Lantern`, approximately
+0.0204/+0.0221 per channel, and `silent invasion`, -0.0366/-0.0347. They are valid
recordings, retained unchanged. Their selected PCM24/48-kHz FLAC derivatives use
an identical 5-Hz second-order Butterworth filter forward/backward on both channels,
with common headroom gains of -1.1793 dB and -0.6715 dB respectively. No frames,
channels or sample rate change. The filter stage alone attenuates 20 Hz by less
than 0.04 dB, independently of that recorded common gain; verification covers
decoded samples against the filter/gain result and PCM24 quantization tolerance.
Both derivatives pass the signal checks. Related originals and derivatives share
the same work/split group and count once; neither is presented as an unmodified
publisher master.

The selected batch adds **44 distinct reviewed works from three creators**:
12 kazunocobit, 28 menogin and four HarryArtz. The second Explore version and
the eight-second cue do not inflate the count. The reviewed collection becomes
**222 works / 16 creators**, including **48 Japanese artist-repertoire works**.
Human Japanese vocal coverage remains absent. The collection still does not
meet the requested 1,000-work regional training target.

The local acquisition report binds official download bytes, range receipts,
whole-ZIP CRC checks where applicable, byte-identical retained members, complete
decoded frames and source/PCM SHA-256. Full-file signal measurements cover peak,
DC, sustained full-scale samples and silence. Lossless containers and spectral
statistics cannot independently certify every earlier mastering step.

Actual four-second excerpts from kazunocobit, menogin, both DC derivatives and
HarryArtz passed 30 codec-pair cases: MP3 CBR 64, AAC-LC CBR 96 and Vorbis VBR Q3
at 44.1/48 kHz. All left/right/mid/side roles were finite and the measured residual
alignment was zero samples. This is preparation evidence, not trained-model quality.

[Bandcamp's current terms](https://bandcamp.com/terms_of_use), effective May 7,
2026, provide a default personal/noncommercial license with an exception for the
identified copyright holder's written permission. The artist-to-platform grant
also withholds permission for training music-generating models without the
artist's express permission. Our source assessment relies on each creator's
separate written CC BY 4.0 grant for codec restoration, not on a download alone
or Bandcamp's platform license. This assessment does not claim permission to
train a music generator; the retained terms and creator grants remain separate
evidence for the actual use and artifact review.

Source-specific CC BY assessments support commercial codec-restoration training
and intended Apache weight distribution within the granted rights, with credits.
The actual checkpoint still needs review for retained source material and the
complete corpus's provenance. The separate unresolved
[Slakh composition/rendering findings](../legal/WEIGHT_LICENSE_REVIEW.md) remain;
these new sources do not settle the rights of the whole future model.

See [current reviewed coverage and training readiness](dataset-curation.md).
