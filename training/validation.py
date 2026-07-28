"""Deterministic validation aggregation and best-checkpoint selection."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

VALIDATION_SCHEMA_VERSION = 1
PRIMARY_METRIC = "high_band_magnitude"
TIE_BREAKER_METRIC = "low_band_identity"
_STRATUM_FIELDS = ("corpus", "codec_id", "quality", "sample_rate", "channel_role")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def quality_label(codec_mode: str, codec_setting: float) -> str:
    """Return a stable codec-quality label for validation strata."""
    return f"{codec_mode}:{float(codec_setting):g}"


def aggregate_validation_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate row metrics identically regardless of input or worker order."""
    if not rows:
        raise ValueError("validation requires at least one row")
    ordered = sorted(rows, key=lambda row: str(row["row_id"]))
    row_ids = [str(row["row_id"]) for row in ordered]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("validation row_id values must be unique")

    metric_names = tuple(sorted(_finite_metrics(ordered[0]["metrics"]).keys()))
    if PRIMARY_METRIC not in metric_names or TIE_BREAKER_METRIC not in metric_names:
        raise ValueError(
            f"validation metrics require {PRIMARY_METRIC!r} and {TIE_BREAKER_METRIC!r}"
        )
    normalized: list[dict[str, Any]] = []
    for row in ordered:
        metrics = _finite_metrics(row["metrics"])
        if tuple(sorted(metrics)) != metric_names:
            raise ValueError("validation rows must contain the same metric vector")
        normalized.append({**row, "row_id": str(row["row_id"]), "metrics": metrics})

    strata: dict[str, dict[str, dict[str, Any]]] = {}
    for field_name in _STRATUM_FIELDS:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in normalized:
            if field_name not in row:
                raise ValueError(f"validation row {row['row_id']}: missing {field_name}")
            grouped.setdefault(str(row[field_name]), []).append(row)
        strata[field_name] = {
            name: {
                "row_count": len(group_rows),
                "metrics": _mean_metrics(group_rows, metric_names),
            }
            for name, group_rows in sorted(grouped.items())
        }

    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "row_count": len(normalized),
        "row_ids": row_ids,
        "overall": _mean_metrics(normalized, metric_names),
        "strata": strata,
    }


def should_run_validation(epoch: int, interval_epochs: int) -> bool:
    """Use an absolute epoch schedule so resume cannot shift validation."""
    if epoch < 1:
        raise ValueError("epoch must be positive")
    if interval_epochs < 1:
        raise ValueError("validation interval must be positive")
    return epoch == 1 or epoch % interval_epochs == 0


@dataclass
class BestCheckpointTracker:
    """Select lower high-band loss, then lower low-band identity loss."""

    best_epoch: int | None = None
    best_metrics: dict[str, float] = field(default_factory=dict)

    def consider(self, epoch: int, metrics: dict[str, Any]) -> bool:
        """Record a candidate and return whether it is the new best."""
        if epoch < 1:
            raise ValueError("epoch must be positive")
        candidate = _finite_metrics(metrics)
        for name in (PRIMARY_METRIC, TIE_BREAKER_METRIC):
            if name not in candidate:
                raise ValueError(f"best checkpoint metric {name!r} is missing")
        candidate_key = (candidate[PRIMARY_METRIC], candidate[TIE_BREAKER_METRIC])
        if self.best_epoch is not None:
            best_key = (
                self.best_metrics[PRIMARY_METRIC],
                self.best_metrics[TIE_BREAKER_METRIC],
            )
            if candidate_key >= best_key:
                return False
        self.best_epoch = epoch
        self.best_metrics = dict(sorted(candidate.items()))
        return True

    def state_dict(self) -> dict[str, Any]:
        """Return a weights-only-safe state mapping."""
        return {
            "primary_metric": PRIMARY_METRIC,
            "tie_breaker_metric": TIE_BREAKER_METRIC,
            "best_epoch": self.best_epoch,
            "best_metrics": dict(self.best_metrics),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> BestCheckpointTracker:
        """Restore and validate checkpointed selection state."""
        if state.get("primary_metric") != PRIMARY_METRIC:
            raise ValueError("checkpoint validation primary metric is incompatible")
        if state.get("tie_breaker_metric") != TIE_BREAKER_METRIC:
            raise ValueError("checkpoint validation tie-breaker metric is incompatible")
        best_epoch = state.get("best_epoch")
        metrics = _finite_metrics(state.get("best_metrics", {}))
        if best_epoch is None:
            if metrics:
                raise ValueError("checkpoint has best metrics without a best epoch")
            return cls()
        epoch = int(best_epoch)
        tracker = cls()
        if not tracker.consider(epoch, metrics):
            raise ValueError("invalid best checkpoint state")
        return tracker


def build_validation_state(
    *,
    report: dict[str, Any],
    manifest_sha256: str,
    tracker: BestCheckpointTracker,
    interval_epochs: int,
    epoch: int,
) -> dict[str, Any]:
    """Build the complete validation payload stored in a checkpoint."""
    if _SHA256_RE.fullmatch(manifest_sha256) is None:
        raise ValueError("validation manifest_sha256 must be lowercase SHA-256")
    if int(report.get("schema_version", 0)) != VALIDATION_SCHEMA_VERSION:
        raise ValueError("validation report schema is incompatible")
    if epoch < 1 or interval_epochs < 1:
        raise ValueError("validation epoch and interval must be positive")
    _finite_metrics(report.get("overall", {}))
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "manifest_sha256": manifest_sha256,
        "last_epoch": epoch,
        "report": report,
        "best": tracker.state_dict(),
        "schedule": {"interval_epochs": interval_epochs},
    }


def _finite_metrics(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("metrics must be a mapping")
    metrics = {str(name): float(metric) for name, metric in value.items()}
    if any(not math.isfinite(metric) for metric in metrics.values()):
        raise ValueError("validation metrics must be finite")
    return metrics


def _mean_metrics(rows: list[dict[str, Any]], metric_names: tuple[str, ...]) -> dict[str, float]:
    return {
        name: math.fsum(float(row["metrics"][name]) for row in rows) / len(rows)
        for name in metric_names
    }
