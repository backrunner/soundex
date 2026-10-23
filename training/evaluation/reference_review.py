"""Frozen P46 low-band/identity reference checks on the reused P44 review clips.

SPDX-License-Identifier: Apache-2.0
"""

from collections import Counter
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from evaluation.reference_preservation import pcm_guard
from evaluation.reporting import sha256_file, write_json_report
from models.multires_spectrum import supervised_waves
from models.restoration_head import load_restoration_head, predict_budget
from scripts.run_consistency_followup import read


@torch.no_grad()
def render(model: torch.nn.Module, item: tuple) -> torch.Tensor:
    prediction, _ = predict_budget(model, item[0].batch, item[1])
    return supervised_waves(prediction, item[0].batch)[0]


def review_reference(
    config: dict, endpoints: dict, clips: BoundedEnvelopeClips, output: Path
) -> None:
    destination = output / "reference-review.json"
    if destination.exists():
        old = read(destination)
        if old["endpoints"] != endpoints:
            raise ValueError("reference review endpoint drift")
        return
    source = read(Path(config["calibration"]))
    original_cases = next(iter(source["cases"].values()))
    specs = [sample["spec"] for c in original_cases for sample in c["samples"]["calibration"]]
    pool = LazyPool(specs, clips)
    items = [pool.select(list(range(i * 5, i * 5 + 5))) for i in range(len(original_cases))]
    anchors = {}
    for label, endpoint in config["endpoints"].items():
        model = load_restoration_head(
            torch.load(endpoint["path"], map_location="cpu", weights_only=False)
        ).eval()
        anchors[label] = [render(model, item) for item in items]
    reports = {}
    for label, endpoint in endpoints.items():
        if sha256_file(Path(endpoint["path"])) != endpoint["sha256"]:
            raise ValueError("reference review endpoint changed")
        initial = next(k for k in anchors if label == k or label.startswith(k + "-"))
        model = load_restoration_head(
            torch.load(endpoint["path"], map_location="cpu", weights_only=False)
        ).eval()
        cases = []
        for i, item in enumerate(items):
            wave = render(model, item)
            rulers = {
                str(bits): pcm_guard([wave], [anchors[initial][i]], [item], bits)
                for bits in (16, 20, 24)
            }
            cases.append(
                {
                    "index": original_cases[i]["index"],
                    "samples": original_cases[i]["samples"],
                    "measurements": rulers["16"]["measurements"],
                    "ruler_violations": {
                        bits: report["violations"] for bits, report in rulers.items()
                    },
                }
            )
        reports[label] = {
            "cases": cases,
            "violation_counts": {
                bits: dict(
                    Counter(v["role"] for case in cases for v in case["ruler_violations"][bits])
                )
                for bits in ("16", "20", "24")
            },
        }
        print("reference review", label, "complete", flush=True)
    write_json_report(
        {
            "endpoints": endpoints,
            "cases": reports,
            "scope": "post-endpoint P44 TRAIN recheck; all fixed rulers, no candidate selection",
        },
        destination,
    )
