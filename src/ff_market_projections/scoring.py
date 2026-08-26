"""Config-driven fantasy scoring with explicit coverage and lineage."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import pandas as pd

from .contracts import CheckResult


class ScoringError(ValueError):
    """Fantasy scoring inputs or configuration are invalid."""


@dataclass(frozen=True)
class ScoringResult:
    fantasy_projections: pd.DataFrame
    validation: dict[str, Any]


PPR_MODES = ("standard", "half_ppr", "three_quarter_ppr", "full_ppr")
SCORE_STATS = (
    "passing_yards", "passing_touchdowns", "passing_interceptions",
    "rushing_yards", "rushing_touchdowns", "receiving_yards",
    "receiving_touchdowns", "fumbles_lost", "two_point_conversions",
)
ALL_COMPONENTS = SCORE_STATS + ("receptions",)


def _check(checks: list[CheckResult], name: str, passed: bool, message: str, **details: Any) -> None:
    checks.append(CheckResult(name, passed, message=message, details=details))


def _validate_config(config: dict[str, Any]) -> tuple[dict[str, float], dict[str, float], dict[str, list[str]], str]:
    policy = config.get("missing_stat_policy")
    if policy not in {"blank_total", "partial_total"}:
        raise ScoringError("scoring.missing_stat_policy must be blank_total or partial_total")
    weights = {stat: config.get(stat) for stat in SCORE_STATS}
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in weights.values()):
        raise ScoringError("scoring component weights must be finite numeric values")
    bonuses = config.get("reception_bonus")
    if not isinstance(bonuses, dict) or set(bonuses) != set(PPR_MODES):
        raise ScoringError("scoring.reception_bonus must define all four PPR modes")
    bonus_values = {mode: bonuses[mode] for mode in PPR_MODES}
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0 for value in bonus_values.values()):
        raise ScoringError("scoring reception bonuses must be finite and non-negative")
    if bonus_values != {"standard": 0.0, "half_ppr": 0.5, "three_quarter_ppr": 0.75, "full_ppr": 1.0}:
        raise ScoringError("scoring reception bonuses must be standard, half, 0.5, 0.75, and 1.0")
    profiles = config.get("required_profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ScoringError("scoring.required_profiles must be a non-empty mapping")
    normalized_profiles: dict[str, list[str]] = {}
    for profile, components in profiles.items():
        if not isinstance(profile, str) or not isinstance(components, list) or not components or len(components) != len(set(components)):
            raise ScoringError("scoring.required_profiles entries must be non-empty and unique")
        if any(component not in ALL_COMPONENTS for component in components):
            raise ScoringError(f"scoring.required_profiles.{profile} contains an unsupported component")
        normalized_profiles[profile] = list(components)
    return {stat: float(value) for stat, value in weights.items()}, {mode: float(value) for mode, value in bonus_values.items()}, normalized_profiles, policy


def score_consensus(consensus_stats: pd.DataFrame, scoring_config: dict[str, Any]) -> ScoringResult:
    """Pivot consensus means and calculate partial and publishable totals."""

    required = {"run_id", "season", "canonical_player_id", "stat", "consensus_mean", "quality_status"}
    missing = sorted(required - set(consensus_stats.columns))
    if missing:
        raise ScoringError(f"consensus stats missing required columns: {', '.join(missing)}")
    weights, bonuses, profiles, policy = _validate_config(scoring_config)
    frame = consensus_stats.copy()
    duplicate = frame.duplicated(["run_id", "season", "canonical_player_id", "stat"])
    if duplicate.any():
        raise ScoringError("consensus stats contain duplicate player/stat rows")
    frame["_value"] = pd.to_numeric(frame["consensus_mean"], errors="coerce").where(frame["quality_status"].eq("passed"))
    rows: list[dict[str, Any]] = []
    group_keys = ["run_id", "season", "canonical_player_id"]
    for key, group in frame.groupby(group_keys, sort=True, dropna=False):
        key_values = dict(zip(group_keys, key, strict=True))
        values = {
            str(stat): float(value)
            for stat, value in group[["stat", "_value"]].itertuples(index=False, name=None)
            if pd.notna(value) and math.isfinite(float(value))
        }
        active_profiles = sorted(
            profile for profile, required_components in profiles.items()
            if any(component in values for component in required_components if component != "receptions")
        )
        required_components = sorted({component for profile in active_profiles for component in profiles[profile]})
        components_used = sorted(values)
        components_missing = sorted(set(ALL_COMPONENTS) - set(values))
        row: dict[str, Any] = {
            **key_values,
            "canonical_player_name": next((value for value in group.get("canonical_player_name", pd.Series(dtype=object)).tolist() if pd.notna(value) and str(value)), ""),
            "scoring_profile": "|".join(active_profiles),
            "components_used": "|".join(components_used),
            "components_missing": "|".join(components_missing),
            "projection_complete": bool(active_profiles and not components_missing),
            "scoring_scope": "market_supported_stats_only",
        }
        for component in ALL_COMPONENTS:
            row[component] = values.get(component)
        available_base = {stat: value for stat, value in values.items() if stat in weights}
        base_points = sum(value * weights[stat] for stat, value in available_base.items())
        receptions = values.get("receptions")
        has_scoring_value = bool(available_base or receptions is not None)
        for mode in PPR_MODES:
            partial = base_points + (receptions * bonuses[mode] if receptions is not None else 0.0) if has_scoring_value else None
            row[f"partial_fpts_{mode}"] = partial
            missing_required = not active_profiles or any(component not in values for component in required_components)
            row[f"fpts_{mode}"] = None if policy == "blank_total" and missing_required else partial
        rows.append(row)
    columns = group_keys + ["canonical_player_name", "scoring_profile", "components_used", "components_missing", "projection_complete", "scoring_scope"] + list(ALL_COMPONENTS)
    columns += [f"partial_fpts_{mode}" for mode in PPR_MODES] + [f"fpts_{mode}" for mode in PPR_MODES]
    result = pd.DataFrame.from_records(rows, columns=columns)
    checks: list[CheckResult] = []
    _check(checks, "scoring.player_grain", not result.duplicated(group_keys).any(), "Fantasy projections are one row per player", rows=len(result))
    missing_total_without_partial = pd.Series(False, index=result.index)
    for mode in PPR_MODES:
        missing_total_without_partial |= result[f"fpts_{mode}"].isna() & result[f"partial_fpts_{mode}"].notna() & result["components_missing"].eq("")
    _check(checks, "scoring.no_missing_to_zero", not bool(missing_total_without_partial.any()), "Missing projections are not converted to zero")
    if not result.empty:
        identity_rows = result["partial_fpts_full_ppr"].notna() & result["partial_fpts_standard"].notna()
        receptions_numeric = pd.to_numeric(result["receptions"], errors="coerce").fillna(0)
        identity = (result.loc[identity_rows, "partial_fpts_full_ppr"] - result.loc[identity_rows, "partial_fpts_standard"] - receptions_numeric.loc[identity_rows]).abs() <= 1e-9
        _check(checks, "scoring.full_ppr_identity", bool(identity.all()), "Full-PPR partial points equal standard plus receptions", failures=int((~identity).sum()))
        full_identity = (result["fpts_full_ppr"].notna() & result["fpts_standard"].notna())
        if full_identity.any():
            identity = (result.loc[full_identity, "fpts_full_ppr"] - result.loc[full_identity, "fpts_standard"] - result.loc[full_identity, "receptions"].fillna(0)).abs() <= 1e-9
            _check(checks, "scoring.full_ppr_publishable_identity", bool(identity.all()), "Publishable full-PPR points equal standard plus receptions", failures=int((~identity).sum()))
        nonnegative = receptions_numeric.ge(0)
        monotone_rows = nonnegative & result["partial_fpts_standard"].notna()
        monotone = result.loc[monotone_rows, "partial_fpts_standard"].le(result.loc[monotone_rows, "partial_fpts_half_ppr"]) & result.loc[monotone_rows, "partial_fpts_half_ppr"].le(result.loc[monotone_rows, "partial_fpts_three_quarter_ppr"]) & result.loc[monotone_rows, "partial_fpts_three_quarter_ppr"].le(result.loc[monotone_rows, "partial_fpts_full_ppr"])
        _check(checks, "scoring.ppr_monotonicity", bool(monotone.all()), "PPR outputs are monotone for nonnegative receptions", failures=int((~monotone).sum()))
    else:
        _check(checks, "scoring.full_ppr_identity", True, "No rows to validate")
        _check(checks, "scoring.ppr_monotonicity", True, "No rows to validate")
    validation = {"status": "passed" if all(check.passed for check in checks) else "failed", "scoring_version": "fantasy_scoring_v1", "checks": [check.to_dict() for check in checks], "summary": {"input_rows": int(len(frame)), "player_rows": int(len(result)), "complete_rows": int(result["projection_complete"].sum()) if not result.empty else 0, "partial_rows": int((~result["projection_complete"]).sum()) if not result.empty else 0, "missing_stat_policy": policy}}
    return ScoringResult(result, validation)
