# SPDX-License-Identifier: Apache-2.0
"""Protect recording mass and speech proportions against channel/crop expansion."""

from __future__ import annotations

from collections import defaultdict

import pytest

from data.sampling_weights import recording_weights, speech_mix_ratios


def test_mono_stereo_and_silent_side_have_equal_recording_mass() -> None:
    rows = []
    for track, roles in (
        ("mono", [1.0]),
        ("stereo", [0.2, 0.2, 0.4, 0.2]),
        ("stereo-silent-side", [0.2, 0.2, 0.4]),
    ):
        for codec_weight in (1.0, 2.0):
            for role_weight in roles:
                rows.extend(
                    {"track_id": track, "sampling_weight": codec_weight * role_weight}
                    for _ in range(4)
                )
    mass = defaultdict(float)
    for row, weight in zip(rows, recording_weights(rows), strict=True):
        mass[row["track_id"]] += weight
    assert dict(mass) == pytest.approx({"mono": 1.0, "stereo": 1.0, "stereo-silent-side": 1.0})


@pytest.mark.parametrize("synthetic_ratio", [0.0, 0.2])
def test_speech_matches_smallest_genre_after_final_source_normalization(
    synthetic_ratio: float,
) -> None:
    rows = []
    for genre, count in (("electronic", 5), ("country", 2), ("pop", 3)):
        for track in range(count):
            # Stereo/crops are duplicates for ratio calculation, not additional songs.
            rows.extend(
                {
                    "track_id": f"{genre}-{track}",
                    "split": "train",
                    "source_metadata": {"genre": genre},
                }
                for _ in range(4)
            )
    rows.append({"track_id": "ignored", "split": "test", "source_metadata": {"genre": "rare"}})
    ratios, report = speech_mix_ratios(rows, synthetic_ratio=synthetic_ratio)
    assert sum(ratios.values()) == pytest.approx(1)
    assert report["training_recordings_by_genre"] == {"country": 2, "electronic": 5, "pop": 3}
    assert ratios["speech_library"] == pytest.approx(report["music_genre_draw_ratios"]["country"])
    assert report["smallest_genres"] == ["country"]
    assert ("slakh2100" in ratios) == bool(synthetic_ratio)


@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf")])
def test_invalid_manifest_weights_are_rejected(weight: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        recording_weights([{"track_id": "bad", "sampling_weight": weight}])


def test_speech_ratio_never_invents_a_music_style() -> None:
    with pytest.raises(ValueError, match="labeled real music"):
        speech_mix_ratios([{"track_id": "unknown", "split": "train"}])
