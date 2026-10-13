from collections import Counter

import pytest

from data.coverage_study import PoolView, banks, expanded_anchors, schedule


def anchors(music=253):
    rows, specs = {}, []
    for corpus, count in (("music_library", music), ("speech_library", 8)):
        for i in range(count):
            key = f"{corpus}-{i:03d}"
            rows[key] = {
                "row_id": key,
                "track_id": key,
                "corpus": corpus,
                "split": "train",
                "held_out": False,
                "codec_id": "mp3-cbr-64",
                "sample_rate": 48000,
                "channel_role": "left",
                "source_channels": 2,
                "length_samples": 160000,
            }
            specs.append({"row_id": key, "start": 40000, "gain_db": 0.0})
    return specs, rows


def test_nested_banks_preserve_original_crops_and_handle_mono():
    wide, rows = anchors(33)
    rows["music_library-032"].update(channel_role="mono", source_channels=1)
    narrow = wide[:32] + wide[33:]
    result, selection = banks(narrow, rows)
    assert all(s in result["wide"]["bank"] for s in result["small"]["bank"])
    assert selection[0]["channel_role"] == "mono"
    assert selection[0]["channel_role_mapped"]
    assert all(
        not any(r["supervision_overlaps_reserved_input"])
        for g in result.values()
        for r in g["crop_audit"]
    )
    rows["music_library-032"]["held_out"] = True
    chosen, _ = expanded_anchors(narrow, rows)
    assert len(chosen) == 40


def test_missing_profile_fails_instead_of_silently_switching_codec():
    wide, rows = anchors(33)
    rows["music_library-032"]["codec_id"] = "other"
    with pytest.raises(ValueError, match="no matched profile"):
        expanded_anchors(wide[:32] + wide[33:], rows)


def test_matched_exposure_and_speech_are_exact_and_deterministic():
    wide, rows = anchors()
    small = wide[:32] + wide[253:]
    a = schedule(small, rows, 2048, 13)
    b = schedule(wide, rows, 16192, 13)
    assert schedule(small, rows, 2048, 13) == a
    assert schedule(small, rows, 2048, 12) != a
    for x, y in zip(a, b[:2048], strict=True):
        assert small[x[-1] // 8] == wide[y[-1] // 8]
        assert x[-1] % 8 == y[-1] % 8
    for draws, music in ((a, 32), (b, 253)):
        assert all(len({i // 8 for i in draw[:4]}) == 4 for draw in draws)
        counts = Counter(i // 8 for draw in draws for i in draw[:4])
        assert len(counts) == music and set(counts.values()) == {256}
        crops = Counter(i for draw in draws for i in draw[:4])
        assert len(crops) == music * 8 and set(crops.values()) == {32}
    assert sorted(i for draw in a[:64] for i in draw) == list(range(320))


def test_pool_view_maps_speech_after_expanded_music():
    class Pool:
        def __init__(self):
            self.specs = ["music-a", "music-b", "speech"]
            self.rows = self.specs

        def select(self, indices):
            return indices

    view = PoolView(Pool(), ["music-a", "speech"])
    assert view.select([1, 0]) == [2, 0]
