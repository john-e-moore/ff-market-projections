"""Auditable aggregation of eligible source means at player/stat grain."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import pandas as pd

from .contracts import CheckResult


class ConsensusError(ValueError):
    """Source consensus inputs or configuration are invalid."""


@dataclass(frozen=True)
class ConsensusResult:
    consensus_stats: pd.DataFrame
    validation: dict[str, Any]


KEYS = ("run_id", "season", "canonical_player_id", "stat")


def _check(checks: list[CheckResult], name: str, passed: bool, message: str, **details: Any) -> None:
    checks.append(CheckResult(name, passed, message=message, details=details))


def aggregate_source_projections(
    source_projections: pd.DataFrame,
    sources: dict[str, dict[str, Any]],
    aggregation: dict[str, Any],
) -> ConsensusResult:
    """Aggregate only passed source rows, retaining workbook-visible inputs."""

    required = set(KEYS) | {"source", "mean", "sensitivity_low", "sensitivity_high", "quality_status"}
    missing = sorted(required - set(source_projections.columns))
    if missing:
        raise ConsensusError(f"source projections missing required columns: {', '.join(missing)}")
    enabled: dict[str, float] = {}
    for source, config in sources.items():
        if not isinstance(config, dict) or not isinstance(config.get("enabled"), bool):
            raise ConsensusError(f"sources.{source}.enabled must be boolean")
        weight = config.get("weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(float(weight)) or float(weight) < 0:
            raise ConsensusError(f"sources.{source}.weight must be non-negative")
        if config["enabled"] and float(weight) > 0:
            enabled[source] = float(weight)
    if not enabled:
        raise ConsensusError("at least one enabled source must have positive weight")

    frame = source_projections.copy()
    frame["source"] = frame["source"].astype(str)
    unknown = sorted(set(frame["source"]) - set(sources))
    if unknown:
        raise ConsensusError(f"source projections contain unknown source(s): {', '.join(unknown)}")
    duplicate = frame.duplicated(list(KEYS) + ["source"])
    checks: list[CheckResult] = []
    _check(checks, "aggregation.source_projection_keys", not duplicate.any(), "Source projection grain is unique", duplicate_rows=int(duplicate.sum()))
    if duplicate.any():
        raise ConsensusError("source projections contain duplicate player/stat/source rows")

    eligible = frame.loc[frame["quality_status"].eq("passed") & frame["source"].isin(enabled)].copy()
    invalid = eligible[["mean", "sensitivity_low", "sensitivity_high"]].isna().any(axis=1)
    _check(checks, "aggregation.eligible_values", not invalid.any(), "Eligible source means and sensitivity bounds are present", invalid_rows=int(invalid.sum()))
    if invalid.any():
        raise ConsensusError("eligible source projections contain missing numeric values")
    for column in ("mean", "sensitivity_low", "sensitivity_high"):
        if not pd.api.types.is_numeric_dtype(eligible[column]) or not eligible[column].map(math.isfinite).all():
            raise ConsensusError(f"eligible source projection {column} must be finite numeric")

    min_sources = aggregation.get("minimum_sources")
    renormalize = aggregation.get("renormalize_available_source_weights")
    max_absolute = aggregation.get("max_absolute_disagreement")
    max_relative = aggregation.get("max_relative_disagreement")
    if isinstance(min_sources, bool) or not isinstance(min_sources, int) or min_sources <= 0:
        raise ConsensusError("aggregation.minimum_sources must be a positive integer")
    if not isinstance(renormalize, bool):
        raise ConsensusError("aggregation.renormalize_available_source_weights must be boolean")
    for value, name in ((max_absolute, "max_absolute_disagreement"), (max_relative, "max_relative_disagreement")):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            raise ConsensusError(f"aggregation.{name} must be non-negative")
    max_absolute, max_relative = float(max_absolute), float(max_relative)

    rows: list[dict[str, Any]] = []
    for key, _group in frame.groupby(list(KEYS), sort=True, dropna=False):
        key_values = dict(zip(KEYS, key, strict=True))
        available = eligible.loc[(eligible[list(KEYS)] == pd.Series(key_values)).all(axis=1)].drop_duplicates("source")
        available_sources = sorted(available["source"].tolist())
        available_weight_sum = sum(enabled[source] for source in available_sources)
        denominator = available_weight_sum if renormalize else sum(enabled.values())
        row: dict[str, Any] = {
            **key_values, "aggregation_version": "source_consensus_v1", "source_list": "|".join(available_sources),
            "source_count": len(available_sources), "configured_weight_sum": available_weight_sum,
            "effective_weight_sum": available_weight_sum / denominator if denominator else 0.0,
            "sensitivity_label": "model_sensitivity_not_confidence_interval",
        }
        for source in sorted(enabled):
            source_row = available.loc[available["source"].eq(source)]
            row[f"{source}_mean"] = float(source_row.iloc[0]["mean"]) if not source_row.empty else None
            row[f"{source}_sensitivity_low"] = float(source_row.iloc[0]["sensitivity_low"]) if not source_row.empty else None
            row[f"{source}_sensitivity_high"] = float(source_row.iloc[0]["sensitivity_high"]) if not source_row.empty else None
            row[f"{source}_configured_weight"] = enabled[source]
            row[f"{source}_effective_weight"] = enabled[source] / denominator if source in available_sources and denominator else 0.0
        means = available["mean"].astype(float).tolist()
        if len(means) >= min_sources:
            weights = [enabled[source] for source in available_sources]
            row["consensus_mean"] = sum(weight * mean for weight, mean in zip(weights, means, strict=True)) / denominator
            for bound in ("sensitivity_low", "sensitivity_high"):
                values = available[bound].astype(float).tolist()
                row[bound] = sum(weight * value for weight, value in zip(weights, values, strict=True)) / denominator
            row["minimum_source_mean"], row["maximum_source_mean"] = min(means), max(means)
            row["source_range"] = row["maximum_source_mean"] - row["minimum_source_mean"]
            row["relative_disagreement"] = row["source_range"] / max(abs(row["consensus_mean"]), 1e-12)
            row["disagreement_flag"] = row["source_range"] > max_absolute or row["relative_disagreement"] > max_relative
            row["quality_status"], row["exclusion_reason"] = "passed", ""
        else:
            row.update({"consensus_mean": None, "sensitivity_low": None, "sensitivity_high": None, "minimum_source_mean": None, "maximum_source_mean": None, "source_range": None, "relative_disagreement": None, "disagreement_flag": None, "quality_status": "excluded", "exclusion_reason": "minimum_sources_not_met"})
        rows.append(row)

    result = pd.DataFrame.from_records(rows).sort_values(list(KEYS)).reset_index(drop=True) if rows else pd.DataFrame(columns=list(KEYS))
    passed = result.loc[result["quality_status"].eq("passed")] if not result.empty else result
    _check(checks, "aggregation.consensus_keys", not result.duplicated(list(KEYS)).any(), "Consensus grain is unique", rows=int(len(result)))
    _check(checks, "aggregation.eligible_lineage", bool(not passed.empty and passed["source_count"].ge(min_sources).all()), "Every consensus value has eligible source lineage", consensus_rows=int(len(passed)))
    validation = {"status": "passed" if all(check.passed for check in checks) else "failed", "aggregation_version": "source_consensus_v1", "checks": [check.to_dict() for check in checks], "summary": {"input_rows": int(len(frame)), "eligible_source_rows": int(len(eligible)), "consensus_rows": int(len(result)), "published_rows": int(len(passed)), "minimum_sources": min_sources}}
    return ConsensusResult(result, validation)
