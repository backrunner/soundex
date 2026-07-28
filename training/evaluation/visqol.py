"""Configurable fail-closed adapter for external ViSQOLAudio scoring."""

from __future__ import annotations

import json
import math
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


class ExternalVisqolAudio:
    """Run a command template containing `{reference}` and `{degraded}`."""

    def __init__(
        self,
        command_template: str,
        *,
        supported_sample_rates: tuple[int, ...] = (48_000,),
        timeout_seconds: float = 120.0,
    ) -> None:
        self.command = shlex.split(command_template)
        if not self.command or not any("{reference}" in item for item in self.command):
            raise ValueError("ViSQOL command must contain {reference}")
        if not any("{degraded}" in item for item in self.command):
            raise ValueError("ViSQOL command must contain {degraded}")
        self.supported_sample_rates = frozenset(supported_sample_rates)
        self.timeout_seconds = timeout_seconds

    def supports(self, sample_rate: int) -> bool:
        return sample_rate in self.supported_sample_rates

    def score(self, reference: np.ndarray, degraded: np.ndarray, sample_rate: int) -> float:
        """Return a finite MOS-LQO or raise with the external scorer's detail."""
        if not self.supports(sample_rate):
            raise ValueError(f"ViSQOLAudio does not support {sample_rate} Hz in this configuration")
        with tempfile.TemporaryDirectory(prefix="soundex-visqol-") as directory:
            root = Path(directory)
            reference_path = root / "reference.wav"
            degraded_path = root / "degraded.wav"
            sf.write(
                reference_path,
                np.asarray(reference, dtype=np.float32),
                sample_rate,
                subtype="FLOAT",
            )
            sf.write(
                degraded_path, np.asarray(degraded, dtype=np.float32), sample_rate, subtype="FLOAT"
            )
            command = [
                item.format(reference=reference_path, degraded=degraded_path)
                for item in self.command
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_seconds,
            )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"ViSQOLAudio failed: {detail}")
        score = _parse_score(result.stdout)
        if not math.isfinite(score):
            raise RuntimeError("ViSQOLAudio returned a non-finite score")
        return score


def _parse_score(output: str) -> float:
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        parsed = None
    score = _find_json_score(parsed)
    if score is not None:
        return score
    matches = re.findall(
        r"(?:MOS(?:-LQO)?|moslqo|score)\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        output,
        flags=re.IGNORECASE,
    )
    if not matches:
        raise RuntimeError("could not parse ViSQOLAudio MOS-LQO output")
    return float(matches[-1])


def _find_json_score(value: Any) -> float | None:
    if isinstance(value, dict):
        for key in ("moslqo", "mos_lqo", "mos", "score"):
            if key in value:
                return float(value[key])
        for child in value.values():
            result = _find_json_score(child)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = _find_json_score(child)
            if result is not None:
                return result
    return None
