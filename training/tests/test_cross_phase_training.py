"""P69 genuine Adam continuation, fixed bases and final assessment paths."""

import copy
from types import SimpleNamespace

import pytest
import torch
from test_fused_spectrum import batch_for

from evaluation.band_training import fresh, learning_rate
from evaluation.reporting import write_json_report
from scripts.run_consistency_followup import read
from scripts.run_cross_phase_training import ARMS, assess, load, pack, wrap
from scripts.run_fused_phase_pilot import update


@pytest.mark.parametrize("kind", ARMS)
def test_cross_phase_exact_adam_resume_and_new_observation_learning(kind):
    model = wrap(fresh("band", 12), 12, kind)
    frozen = copy.deepcopy(model.base.state_dict())
    before = model.phase_head[0].weight.detach().clone()
    batch = batch_for(0.01)
    batch = {k: v.float() if v.is_floating_point() else v for k, v in batch.items()}
    batch["identity"] = torch.tensor([False, True])
    batch["cutoff"] = torch.full((2, 160), 14000.0)
    batch["y"][0, 1, 89:, 85:100] += 0.1
    items = [(SimpleNamespace(batch=batch), torch.zeros(2, 160, 1538))] * 4
    optimizer = torch.optim.Adam(model.phase_head.parameters(), lr=1e-5)
    first = update(model, items, optimizer, True)
    assert first["changed_parameters"] > 0
    saved = copy.deepcopy(pack(model, kind, 12, optimizer, [first], "test"))
    other = load(saved)
    resumed = torch.optim.Adam(other.phase_head.parameters(), lr=1e-5)
    resumed.load_state_dict(saved["optimizer"])
    optimizer.param_groups[0]["lr"] = resumed.param_groups[0]["lr"] = learning_rate(1, 16384)
    left, right = update(model, items, optimizer, True), update(other, items, resumed, True)
    assert left == right
    assert all(torch.equal(v, other.state_dict()[k]) for k, v in model.state_dict().items())
    assert all(torch.equal(v, frozen[k]) for k, v in model.base.state_dict().items())
    assert not torch.equal(before[:, 645:903], model.phase_head[0].weight[:, 645:903])


def test_matched_initial_tensors_and_zero_parent_checkpoint():
    models = [wrap(fresh("band", 12), 12, kind) for kind in ARMS]
    assert all(torch.equal(v, models[1].state_dict()[k]) for k, v in models[0].state_dict().items())
    base = fresh("band", 12)
    parent = load(
        {
            "format": "soundex-band-training-v1",
            "kind": "band",
            "seed": 12,
            "model": base.state_dict(),
        }
    )
    assert all(torch.equal(v, parent.base.state_dict()[k]) for k, v in base.state_dict().items())
    assert not parent.phase_head[-1].weight.any()
    assert sum(p.numel() for p in parent.parameters() if p.requires_grad) == 33281


def test_assessment_matches_parent_control_and_endpoint_artifact_layout(tmp_path):
    config = {"seeds": [12, 13], "scope": "test", "selection": "test"}
    sample = {
        "row": {
            "source_checksum": "source",
            "source_id": "recording",
            "codec_id": "mp3",
            "corpus": "music_library",
        }
    }
    for seed in config["seeds"]:
        for arm, score in (("parent", 1.0), ("temporal_repeat", 0.9), ("adjacent", 0.8)):
            for folder, key in (("raw-review", "scores"), ("native-review", "native")):
                label = f"{seed}-{arm}"
                path = (
                    tmp_path
                    / folder
                    / (label + ".json" if folder == "raw-review" else label + "/report.json")
                )
                case = {
                    "index": 0,
                    "samples": {"calibration": [sample]},
                    "identity": [False, True],
                    key: {"mr": [score, score]},
                }
                write_json_report({"cases": [case]}, path)
    assess(config, tmp_path)
    report = read(tmp_path / "assessment.json")
    assert len(report["comparisons"]) == 6
    for pair in report["comparisons"].values():
        for path in pair.values():
            assert (
                path["source_equal_summary"]["music_library/paired"]["metrics"]["mr"]["delta"] < 0
            )
