# Native music replacing Slakh, 2026-10-09

The upcoming regional run excludes Slakh from its dataset paths, sampling,
preparation, supplement audit and new queue. The queue will initialize fresh
weights when the reviewed regional collection is ready. Historical Slakh files
and experimental checkpoints remain separate and do not establish permission
for publishing learned weights. Legacy tooling is available only by explicit
selection; a Slakh directory on disk does not enable it.

## Free publisher originals

The following seven albums provide 83 original FLAC recordings. Downloads use the
publishers’ zero-price computer-download path without providing an email,
subscribing, creating an account or purchasing anything. Artwork is not used.
Source pages and individual track pages, credits, chosen grants and audio hashes
remain local. Audio and private delivery URLs are excluded from Git.

| Publisher album | Repertoire and genre evidence | Grant and contributor evidence |
| --- | --- | --- |
| [Victor van Voorn — Rock, Miez, Unfug](https://victorvanvoorn.bandcamp.com/album/rock-miez-unfug) | Germany; five original German vocal rock songs | CC BY 4.0; artist credits composition, lyrics, recording, production, mixing and mastering. |
| [Gerissene Seiten — Sehnen & Benehmen](https://gerisseneseiten.bandcamp.com/album/sehnen-benehmen) | Halle, Germany; 18 acoustic / anti-folk / Holzpunk recordings | Explicit whole-album and lyric CC BY 4.0 statement; ph4nt. recording/mixing credits retained. |
| [Gerissene Seiten — Höhnen & Versöhnen](https://gerisseneseiten.bandcamp.com/album/h-hnen-vers-hnen) | Halle, Germany; 17 acoustic / anti-folk recordings | Explicit whole-album and lyric CC BY 4.0 statement; ISD-Crew and ph4nt. recording/mixing credits retained. |
| [Gerissene Seiten — Hybris](https://gerisseneseiten.bandcamp.com/album/hybris) | Halle, Germany; 15 bedroom-punk recordings | CC BY 4.0; ph4nt. mixing/mastering credits retained. |
| [Victor van Voorn — The Red Dead EP](https://victorvanvoorn.bandcamp.com/album/the-red-dead-ep) | Germany; five original English vocal country / country-rock / folk songs | CC BY 4.0; artist explicitly credits original concept, music, lyrics and recording. |
| [The Grumbles — A Pretty Bad Good](https://thegrumbles.bandcamp.com/album/a-pretty-bad-good) | Colorado Springs, US; eight vocal indie-rock songs | Whole-album CC BY 4.0 statement; band authorship/performance, AvA and named backing vocals, production and engineering credits retained. |
| [Austin Moffa — A Speck of Dust in Space](https://austinmoffa.bandcamp.com/album/a-speck-of-dust-in-space) | Fairfax, US; 15 folk / acoustic / electric-folk / anti-folk songs | Explicit written CC BY 4.0 commercial-use grant chosen. The page also links a BY-SA 4.0 badge; both receipts are retained. Austin Moffa / ROZKOL share one creator ID. |

Region follows the documented artist repertoire, not invented nationality or
market coverage. Country-rock fusion is retained as such; it does not establish
US-country or mainstream-pop breadth. Actual track lyrics support vocal labels;
missing lyric evidence is not filled by assumptions about the album.

## Verified acquisition result

The completed batch retains **83 original FLAC files / 282.38 source minutes**.
After signal and source review, **55 distinct works / 179.07 minutes** enter the selection,
including 54 lyric-evidenced vocal works. Counts do not multiply channels, codecs or versions.

| Album | Native format | Downloaded files | Selected works | Selected minutes |
| --- | --- | ---: | ---: | ---: |
| A Speck of Dust in Space | 44.1 kHz / PCM16 / 2 channels | 15 | 15 | 54.64 |
| Höhnen & Versöhnen | 44.1/48 kHz / PCM24 / mono or stereo | 17 | 2 | 4.60 |
| Hybris | 44.1 kHz / PCM16 / 2 channels | 15 | 15 | 51.95 |
| Sehnen & Benehmen | 44.1/48 kHz / PCM16 / stereo | 18 | 5 | 19.84 |
| A Pretty Bad Good | 44.1 kHz / PCM24 / 2 channels | 8 | 8 | 20.54 |
| The Red Dead EP | 44.1 kHz / PCM24 / 2 channels | 5 | 5 | 14.22 |
| Rock Miez Unfug | 44.1 kHz / PCM24 / 2 channels | 5 | 5 | 13.28 |

28 files retain sustained-full-scale signal flags; one of these also has the
pending underlying-composition finding described below. Valid pending originals
remain outside the clean-target selection. Related versions retain shared work groups.

The updated pool has **277 works / 20 creators / 1132.55 music minutes**.
Adding VCTK speech gives **1156.40 minutes** across future train/validation/test source material.
This snapshot reported 34 gaps against the strict 1,000-work plan. The active
[first-run policy](dataset-curation.md) now targets 300 reviewed works, with
fine-grained coverage gaps reported separately from its required readiness checks.

Three new vocal rock/folk/country samples passed 18 real FFmpeg codec round-trips:
MP3 CBR 64 kbps, AAC-LC 96 kbps and Vorbis q3 at both 44.1 and 48 kHz.
All residual alignment offsets were zero and left/right/mid/side pairs were finite.
This is a bounded preprocessing smoke check, not completed whole-corpus pairing
or model quality/runtime evidence. The unchanged mono contract remains covered
by the focused data-protocol/library tests. One downloaded 48 kHz mono master
is held for full-scale signal flags, rather than rejected for its channel count.
That held mono recording separately passed all six codec/rate round-trips with
finite mono pairs and zero residual offsets; it was not added to training.
Native rates in the actual archives range from 44.1 to 48 kHz; an album-level
download header is not substituted for the decoded per-file metadata.

Seven redundant download ZIPs were removed only after CRC, delivery SHA-256 and
exact retained-member/hash verification; original FLAC, source pages and audit
receipts remain local, saving 2,083,521,367 bytes. No active download fragments or historical originals
were removed.

## Selection and rights limits

Alternate versions of *Ampeln*, *Studenten*, *Sonnenfinsternis* and *Hitler war auch
nur ein Deutscher* share their respective work and split groups. They do not add
independent works or leak into separate train/validation/test partitions.
*Who You Gonna Call (the EA!)* remains outside the reviewed selection: its
identifiable adapted *Call Me Maybe* refrain has no established underlying
composition grant in the retained publisher evidence. This is a pending source
finding, not a legal determination about the recording.

The Red Dead EP publisher expressly identifies five original songs. Inspiration
from game characters alone is not treated as proof of borrowed game soundtrack
music; the retained authorship and track evidence identified no borrowed
recording or composition. This is not forensic verification of every sample.
Game characters and separate artwork are not covered by the chosen audio grant.

A license-badge mismatch alone does not erase a separate explicit written grant.
For Austin Moffa, preserve the affirmative CC BY 4.0 text and its version alongside
the BY-SA badge, following the chosen grant and its attribution requirements.
An uploader cannot grant third-party composition, performer or sample rights.
Concrete conflicts remain pending; a separate AI permission letter is not a
blanket requirement for existing CC grants.

Each completed download is checked for exact byte ranges, whole-file SHA-256,
ZIP member CRC and exact retained original FLAC bytes. Every audio file receives
a full finite decode, native format/rate/channel checks and signal measurements.
Quantized masters with sustained full-scale samples remain flagged outside the
clean-target selection pending review; lowering their gain would hide the
measurement without repairing clipping. Sparse music and artistic distortion
are not automatically corrupt files. Spectrum or a lossless container cannot
prove every upstream production step was lossless; publisher-native provenance
is retained rather than making that stronger claim.

Source grants support this project’s commercial codec-restoration training and
intended Apache-2.0 weight publication within the rights actually granted, with
attribution and edits recorded. Audio keeps its upstream license. The exact
future checkpoint still requires full provenance and retained-source artifact
review before release; this source batch is not an automatic weight approval.

## Other candidates

[HoliznaCC0](https://holiznacc0.bandcamp.com/music) has original rock and guitar-jazz
FLAC candidates with an explicit CC0 dedication; the current free delivery path
requires an email, so these are not counted as acquired. The earlier HarryArtz
email authorization does not extend to another publisher.
[Cullah’s licensing page](https://www.cullah.com/licensing/) currently combines
commercial-service wording with differing CC statements; a concrete version-bound
grant must be resolved before counting it. CC BY-SA itself permits commercial use.
[Axletree’s Hengestesieg EP](https://axletree.bandcamp.com/album/hengestesieg-ep)
offers native lossless delivery for a fee and remains unpurchased. FMA MP3 exports
are not used as clean targets. Paid acquisition needs separate approval.

See [current coverage and readiness](dataset-curation.md) and
[weight release findings](../legal/WEIGHT_LICENSE_REVIEW.md).

The later [free pilot completion batch](pilot-300-sources-20261009.md) adds 23
more reviewed works from two creators, reaching 300 works and satisfying the
first-run blocking checks. The 277-work figures above describe this earlier batch
snapshot; see the current [readiness check](dataset-curation.md) for updated totals.
