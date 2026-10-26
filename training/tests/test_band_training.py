"""P63 actual optimizer continuity, complete-endpoint barrier and source aliases."""

import copy
from pathlib import Path

import onnx
import pytest
import torch

from data.causal_study import digest
from evaluation.band_training import (
    ARMS,
    export_model,
    freeze_endpoints,
    fresh,
    learning_rate,
    load,
    pack,
)
from evaluation.reporting import sha256_file
from scripts.run_band_training import assert_catalogs_unchanged, isolated_rows
from scripts.run_joint_training import step
from tests.test_joint_training import items


@pytest.fixture(autouse=True, scope="module")
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("kind", ARMS)
def test_actual_adam_resume_and_feature_learning(kind):
    model = fresh(kind, 12)
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    batch = items()
    first = step(model, batch, optimizer)
    assert first["changed_parameters"] > 0
    saved = copy.deepcopy(pack(model, kind, 12, optimizer, [first], "test"))
    other = load(saved)
    resumed = torch.optim.Adam(other.parameters(), lr=1e-5)
    resumed.load_state_dict(saved["optimizer"])
    optimizer.param_groups[0]["lr"] = resumed.param_groups[0]["lr"] = learning_rate(1, 16384)
    left, right = step(model, batch, optimizer), step(other, batch, resumed)
    assert left == right
    assert all(torch.equal(v, other.state_dict()[k]) for k, v in model.state_dict().items())
    encoder = "backbone.encoders.0.0.weight" if kind == "band" else "backbone.encoder.0.weight"
    assert not torch.equal(model.state_dict()[encoder], before[encoder])
    if kind == "band":
        for key in ("backbone.temporal.0.depthwise.weight", "backbone.mixers.0.0.weight"):
            assert not torch.equal(model.state_dict()[key], before[key])


def test_global_capacity_control_and_indexed_rate():
    counts = {kind: sum(p.numel() for p in fresh(kind, 12).parameters()) for kind in ARMS}
    assert counts == {"global": 480809, "global_matched": 170859, "band": 171137}
    assert learning_rate(0, 16384) == pytest.approx(1e-5)
    assert learning_rate(255, 16384) == pytest.approx(1e-4)
    assert learning_rate(256, 16384) == pytest.approx(1e-4)
    assert learning_rate(16383, 16384) == pytest.approx(1e-5)
    for index, total in ((-1, 16384), (16384, 16384), (256, 257)):
        with pytest.raises(ValueError):
            learning_rate(index, total)


def test_reservation_closure_removes_connected_work_before_sampling():
    base = {"split": "train", "held_out": False, "track_id": "fit"}
    rows = {"a": {**base, "source_checksum": "a"}, "b": {**base, "source_checksum": "b"}}
    universe = [
        *rows.values(),
        {"audio_sha256": "a", "work_id": "same-work"},
        {"decoded_pcm_sha256": "reserved", "work_id": "same-work"},
    ]
    reserved = [{"audio_sha256": "reserved"}]
    assert set(isolated_rows(rows, {"fit"}, universe, reserved)) == {"b"}
    assert not isolated_rows(rows, {"another"}, universe, reserved)


def test_catalog_paths_ignore_order_but_never_added_or_removed_reservations():
    paths = [
        Path("/runs/study/evaluation-catalog.jsonl"),
        Path("/runs/study-v2/evaluation-catalog.jsonl"),
    ]
    stored = list(map(str, sorted(paths)))
    actual = sorted(map(str, paths))
    assert stored != actual  # reproduce the first frozen launcher rejection
    assert_catalogs_unchanged(actual, stored)
    for changed in (actual[:1], [*actual, "/runs/new/evaluation-catalog.jsonl"]):
        with pytest.raises(ValueError, match="reservations changed"):
            assert_catalogs_unchanged(changed, stored)


def test_review_barrier_rejects_partial_steps_wrong_seed_and_hash(tmp_path):
    config = {"arms": ["band"], "seeds": [12], "steps": 2}
    model = fresh("band", 12)
    opt = torch.optim.Adam(model.parameters())
    signature = digest({"config": config, "kind": "band", "seed": 12})
    path = tmp_path / "checkpoint.pth"

    def entry(history, seed=12):
        torch.save(pack(model, "band", seed, opt, history, signature), path)
        return {"12-band": {"path": str(path), "sha256": sha256_file(path)}}

    with pytest.raises(ValueError, match="complete all"):
        freeze_endpoints(config, {}, tmp_path)
    with pytest.raises(ValueError, match="incomplete"):
        freeze_endpoints(config, entry([{}]), tmp_path)
    with pytest.raises(ValueError, match="incomplete"):
        freeze_endpoints(config, entry([{}, {}], 13), tmp_path)
    entries = entry([{}, {}])
    entries["12-band"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash drift"):
        freeze_endpoints(config, entries, tmp_path)
    freeze_endpoints(config, entry([{}, {}]), tmp_path)
    assert (tmp_path / "endpoints.json").exists()


def test_trained_band_export_reuses_hash_and_metadata(tmp_path):
    model = fresh("band", 12)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    step(model, items(), optimizer)
    path = export_model(model, tmp_path / "export", "band")
    meta = {p.key: p.value for p in onnx.load(path).metadata_props}
    assert meta["soundex.study"] == "p63-fresh-band-training"
    assert meta["soundex.architecture"] == "band" and meta["soundex.deployable"] == "false"
    assert export_model(model, path.parent, "band") == Path(path)
