# SPDX-License-Identifier: Apache-2.0
"""Native music counts remain independent of encodings, aliases and quality holds."""

from scripts.select_native_catalog import select_recordings


def test_aliases_pcm_copies_and_flagged_masters_do_not_inflate_count() -> None:
    base = {
        "id": "first",
        "audio_sha256": "a",
        "split_group": "composer:work",
        "source_quality": {
            "review_flags": [],
            "audio_sha256": "a",
            "decoded_pcm_sha256": "pcm",
            "duration_seconds": 60,
            "sample_rate": 48000,
        },
    }
    encoding = {**base, "id": "second", "split_group": "different-label"}
    alias = {
        **base,
        "id": "third",
        "source_quality": {**base["source_quality"], "decoded_pcm_sha256": "other"},
    }
    flagged = {
        **base,
        "id": "fourth",
        "split_group": "composer:other",
        "source_quality": {
            **base["source_quality"],
            "decoded_pcm_sha256": "new",
            "review_flags": ["over_full_scale"],
        },
    }
    selected, held = select_recordings([base, encoding, alias, flagged])
    assert [e["id"] for e in selected] == ["first"]
    assert len(held) == 3
