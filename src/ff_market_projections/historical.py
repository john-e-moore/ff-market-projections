"""Leakage-free preparation of nflverse weekly player history."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .contracts import CheckResult


TARGET_STATS = (
    "passing_yards",
    "passing_touchdowns",
    "rushing_yards",
    "rushing_touchdowns",
    "receiving_yards",
    "receiving_touchdowns",
    "receptions",
)

_COLUMN_ALIASES = {
    "gsis_player_id": ("gsis_player_id", "player_id"),
    "player_name": ("player_display_name", "player_name"),
    "team": ("team", "recent_team"),
    "position": ("position",),
    "passing_attempts": ("passing_attempts", "attempts"),
    "rushing_attempts": ("rushing_attempts", "carries"),
    "targets": ("targets",),
    "passing_yards": ("passing_yards",),
    "passing_touchdowns": ("passing_touchdowns", "passing_tds"),
    "rushing_yards": ("rushing_yards",),
    "rushing_touchdowns": ("rushing_touchdowns", "rushing_tds"),
    "receiving_yards": ("receiving_yards",),
    "receiving_touchdowns": ("receiving_touchdowns", "receiving_tds"),
    "receptions": ("receptions",),
    "age": ("age",),
    "experience": ("experience", "years_exp"),
}

_OPPORTUNITY_BY_STAT = {
    "passing_yards": "passing_attempts",
    "passing_touchdowns": "passing_attempts",
    "rushing_yards": "rushing_attempts",
    "rushing_touchdowns": "rushing_attempts",
    "receiving_yards": "targets",
    "receiving_touchdowns": "targets",
    "receptions": "targets",
}

_PLAUSIBLE_SEASON_MAX = {
    "passing_yards": 8_000,
    "passing_touchdowns": 100,
    "rushing_yards": 4_000,
    "rushing_touchdowns": 50,
    "receiving_yards": 4_000,
    "receiving_touchdowns": 50,
    "receptions": 300,
}


class HistoricalDataError(ValueError):
    """Historical input or output failed a hard validation gate."""

    def __init__(self, check_name: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.check = CheckResult(check_name, False, "error", message, details or {})


@dataclass(frozen=True)
class HistoricalPreparation:
    player_seasons: pd.DataFrame
    backtest_predictions: pd.DataFrame
    validation: dict[str, Any]


def scheduled_games(season: int) -> int:
    """Return the NFL regular-season schedule length for the supported era."""

    if season < 1999:
        raise ValueError("Historical ingestion supports seasons from 1999 onward")
    return 17 if season >= 2021 else 16


def _resolved_column(columns: Iterable[str], canonical: str, *, required: bool = True) -> str | None:
    available = set(columns)
    for candidate in _COLUMN_ALIASES[canonical]:
        if candidate in available:
            return candidate
    if required:
        raise HistoricalDataError(
            "historical.required_fields",
            f"Historical input is missing required field {canonical}; accepted names: {', '.join(_COLUMN_ALIASES[canonical])}",
            {"field": canonical, "accepted_names": list(_COLUMN_ALIASES[canonical])},
        )
    return None


def _nonempty_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _latest_text(group: pd.DataFrame, column: str) -> str | None:
    for value in reversed(group[column].tolist()):
        text = _nonempty_text(value)
        if text is not None:
            return text
    return None


def _ordered_join(group: pd.DataFrame, column: str) -> str | None:
    values: list[str] = []
    for value in group[column]:
        text = _nonempty_text(value)
        if text is not None and text not in values:
            values.append(text)
    return "|".join(values) if values else None


def _finite_nonnegative(series: pd.Series) -> bool:
    values = series.dropna().to_numpy(dtype=float)
    return bool(np.isfinite(values).all() and (values >= 0).all())


def _minimum_opportunity(stat: str, historical_config: dict[str, Any]) -> float:
    key = _OPPORTUNITY_BY_STAT[stat]
    return float(historical_config["prior_opportunity_filters"][key])


def _read_input(path: str | Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, compression="infer", low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise HistoricalDataError("historical.input_readable", f"Could not read historical CSV: {exc}") from exc


def _canonical_weekly(raw: pd.DataFrame, historical_config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, str]]:
    for required in ("season", "season_type", "week"):
        if required not in raw.columns:
            raise HistoricalDataError("historical.required_fields", f"Historical input is missing required field {required}", {"field": required})

    resolved: dict[str, str] = {}
    for canonical in (
        "gsis_player_id", "player_name", "team", "position", "passing_attempts",
        "rushing_attempts", "targets", *TARGET_STATS,
    ):
        resolved[canonical] = _resolved_column(raw.columns, canonical)  # type: ignore[assignment]
    for optional in ("age", "experience"):
        column = _resolved_column(raw.columns, optional, required=False)
        if column:
            resolved[optional] = column

    weekly = pd.DataFrame({
        "_source_index": raw.index,
        "season": pd.to_numeric(raw["season"], errors="coerce"),
        "season_type": raw["season_type"].astype("string"),
        "week": pd.to_numeric(raw["week"], errors="coerce"),
    })
    for canonical, source in resolved.items():
        weekly[canonical] = raw[source]

    if weekly[["season", "week"]].isna().any(axis=None):
        raise HistoricalDataError("historical.season_week_valid", "Season and week must be parseable integers")
    weekly["season"] = weekly["season"].astype(int)
    weekly["week"] = weekly["week"].astype(int)

    configured_type = historical_config["season_type"]
    weekly = weekly.loc[weekly["season_type"] == configured_type].copy()
    if weekly.empty:
        raise HistoricalDataError("historical.regular_season_rows", f"No {configured_type} historical rows were found")

    start = int(historical_config["ingest_start_season"])
    latest = int(historical_config["latest_completed_season"])
    weekly = weekly.loc[(weekly["season"] >= start) & (weekly["season"] <= latest)].copy()
    present = set(int(value) for value in weekly["season"].unique())
    expected = set(range(start, latest + 1))
    if present != expected:
        missing = sorted(expected - present)
        raise HistoricalDataError(
            "historical.seasons_complete",
            f"Historical input is missing configured season(s): {missing}",
            {"expected": sorted(expected), "observed": sorted(present), "missing": missing},
        )

    ids = weekly["gsis_player_id"].map(_nonempty_text)
    if ids.isna().any():
        raise HistoricalDataError("historical.player_ids_present", "Historical input contains missing GSIS player IDs", {"missing_rows": int(ids.isna().sum())})
    weekly["gsis_player_id"] = ids

    numeric = ["passing_attempts", "rushing_attempts", "targets", *TARGET_STATS]
    for column in numeric:
        weekly[column] = pd.to_numeric(weekly[column], errors="coerce")
    for stat in TARGET_STATS:
        values = weekly[stat].dropna().to_numpy(dtype=float)
        if weekly[stat].isna().any() or not np.isfinite(values).all():
            raise HistoricalDataError(
                "historical.target_values_valid",
                f"{stat} contains missing or non-finite values",
                {"stat": stat, "missing_rows": int(weekly[stat].isna().sum())},
            )
    for opportunity in ("passing_attempts", "rushing_attempts", "targets"):
        if not _finite_nonnegative(weekly[opportunity]):
            raise HistoricalDataError("historical.opportunity_values_valid", f"{opportunity} contains negative or non-finite values", {"field": opportunity})
    for optional in ("age", "experience"):
        if optional in weekly:
            weekly[optional] = pd.to_numeric(weekly[optional], errors="coerce")

    duplicate_key = ["season", "week", "gsis_player_id", "team"]
    duplicates = weekly.duplicated(duplicate_key, keep=False)
    duplicate_groups = 0
    if duplicates.any():
        duplicate_groups = int(weekly.loc[duplicates].groupby(duplicate_key, dropna=False).ngroups)
        identity_columns = ["player_name", "position"]
        conflicting = []
        exact_duplicate_groups = []
        for key, group in weekly.loc[duplicates].groupby(duplicate_key, dropna=False, sort=False):
            if group.drop(columns=["_source_index", *numeric, "age", "experience"], errors="ignore").drop_duplicates().shape[0] > 1:
                if any(group[column].map(_nonempty_text).nunique(dropna=True) > 1 for column in identity_columns):
                    conflicting.append(key)
            if group.drop(columns=["_source_index"], errors="ignore").drop_duplicates().shape[0] == 1:
                exact_duplicate_groups.append(key)
        if conflicting:
            raise HistoricalDataError(
                "historical.weekly_keys_unique",
                "Historical input contains duplicate player-week-team rows with conflicting identity fields",
                {"duplicate_groups": len(conflicting), "key": duplicate_key, "sample": conflicting[:5]},
            )
        if exact_duplicate_groups:
            raise HistoricalDataError(
                "historical.weekly_keys_unique",
                "Historical input contains exact duplicate player-week-team rows",
                {"duplicate_groups": len(exact_duplicate_groups), "key": duplicate_key, "sample": exact_duplicate_groups[:5]},
            )
        # nflverse can split one weekly player record into complementary rows
        # (for example, a passing row and an all-zero row). Sum numeric fields
        # and retain the latest nonempty descriptive values for that source week.
        grouped_rows: list[dict[str, Any]] = []
        for key, group in weekly.groupby(duplicate_key, dropna=False, sort=False):
            group = group.sort_values("_source_index")
            row = dict(zip(duplicate_key, key if isinstance(key, tuple) else (key,)))
            row["_source_index"] = int(group["_source_index"].iloc[0])
            row["season_type"] = str(group["season_type"].iloc[-1])
            for column in numeric:
                row[column] = float(group[column].sum(min_count=1)) if group[column].notna().any() else np.nan
            for column in ("player_name", "position", "age", "experience"):
                if column in group:
                    values = group[column].dropna()
                    row[column] = values.iloc[-1] if not values.empty else np.nan
            grouped_rows.append(row)
        nonduplicates = weekly.loc[~duplicates].to_dict(orient="records")
        weekly = pd.DataFrame.from_records(nonduplicates + grouped_rows)
        weekly.attrs["duplicate_week_groups_aggregated"] = duplicate_groups

    return weekly.sort_values(["season", "gsis_player_id", "week", "team"], na_position="last").reset_index(drop=True), resolved


def _aggregate_player_seasons(weekly: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (season, player_id), group in weekly.groupby(["season", "gsis_player_id"], sort=True, dropna=False):
        group = group.sort_values(["week", "team"], na_position="last")
        record: dict[str, Any] = {
            "season": int(season),
            "gsis_player_id": str(player_id),
            "player_name": _latest_text(group, "player_name"),
            "team": _ordered_join(group, "team"),
            "position": _latest_text(group, "position"),
            "games": int(group["week"].nunique()),
            "schedule_games": scheduled_games(int(season)),
            "source_week_rows": int(len(group)),
        }
        for opportunity in ("passing_attempts", "rushing_attempts", "targets"):
            record[opportunity] = float(group[opportunity].sum(min_count=1)) if group[opportunity].notna().any() else np.nan
        for stat in TARGET_STATS:
            record[stat] = float(group[stat].sum())
        for optional in ("age", "experience"):
            record[optional] = float(group[optional].dropna().iloc[-1]) if optional in group and group[optional].notna().any() else np.nan
        records.append(record)
    return pd.DataFrame.from_records(records).sort_values(["season", "gsis_player_id"]).reset_index(drop=True)


def _reconcile_weekly(weekly: pd.DataFrame, seasons: pd.DataFrame) -> None:
    # This loop intentionally does not share the groupby aggregation used above.
    manual: dict[tuple[int, str, str], list[float]] = {}
    for row in weekly.itertuples(index=False):
        for stat in TARGET_STATS:
            manual.setdefault((int(row.season), str(row.gsis_player_id), stat), []).append(float(getattr(row, stat)))
    actual = {
        (int(row.season), str(row.gsis_player_id), stat): float(getattr(row, stat))
        for row in seasons.itertuples(index=False)
        for stat in TARGET_STATS
    }
    mismatches = [key for key, parts in manual.items() if not math.isclose(math.fsum(parts), actual[key], rel_tol=0.0, abs_tol=1e-9)]
    if mismatches:
        raise HistoricalDataError("historical.weekly_to_season_reconciled", "Weekly rows do not reconcile to player-season totals", {"mismatch_count": len(mismatches), "sample": mismatches[:5]})


def _reconcile_optional_summaries(raw: pd.DataFrame, weekly: pd.DataFrame, resolved: dict[str, str], seasons: pd.DataFrame) -> None:
    # Small fixtures may carry an independent season total to exercise this gate.
    for stat in TARGET_STATS:
        raw_name = resolved[stat]
        candidates = (f"season_{raw_name}", f"{raw_name}_season", f"{stat}_season_total")
        summary_column = next((name for name in candidates if name in raw.columns), None)
        if summary_column is None:
            continue
        aligned = raw.loc[weekly["_source_index"], ["season", resolved["gsis_player_id"], summary_column]].copy()
        aligned["season"] = pd.to_numeric(aligned["season"], errors="coerce")
        aligned[summary_column] = pd.to_numeric(aligned[summary_column], errors="coerce")
        expected: dict[tuple[int, str], float] = {}
        for (season, player_id), group in aligned.groupby(["season", resolved["gsis_player_id"]], dropna=False):
            values = group[summary_column].dropna().unique()
            if len(values) > 1:
                raise HistoricalDataError("historical.summary_reconciled", f"Conflicting independent season summaries for {stat}", {"stat": stat})
            if len(values) == 1:
                expected[(int(season), str(player_id))] = float(values[0])
        observed = {(int(row.season), str(row.gsis_player_id)): float(getattr(row, stat)) for row in seasons.itertuples(index=False)}
        mismatches = [key for key, value in expected.items() if key in observed and not math.isclose(value, observed[key], rel_tol=0.0, abs_tol=1e-9)]
        if mismatches:
            raise HistoricalDataError("historical.summary_reconciled", f"Independent season summaries do not reconcile for {stat}", {"stat": stat, "mismatch_count": len(mismatches), "sample": mismatches[:5]})


def _long_player_seasons(seasons: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    shared = [
        "season", "gsis_player_id", "player_name", "team", "position", "games",
        "schedule_games", "source_week_rows", "passing_attempts", "rushing_attempts",
        "targets", "age", "experience",
    ]
    for record in seasons.to_dict(orient="records"):
        for stat in TARGET_STATS:
            total = float(record[stat])
            opportunity_name = _OPPORTUNITY_BY_STAT[stat]
            row = {key: record[key] for key in shared}
            for opportunity in ("passing_attempts", "rushing_attempts", "targets"):
                value = record[opportunity]
                row[f"{opportunity}_per_17"] = value * 17.0 / float(record["schedule_games"]) if not pd.isna(value) else np.nan
            row.update({
                "stat": stat,
                "stat_total": total,
                "stat_total_per_17": total * 17.0 / float(record["schedule_games"]),
                "stat_per_game": total / float(record["games"]) if record["games"] else np.nan,
                "opportunity_name": opportunity_name,
                "opportunity": record[opportunity_name],
                "opportunity_per_17": record[opportunity_name] * 17.0 / float(record["schedule_games"]) if not pd.isna(record[opportunity_name]) else np.nan,
            })
            rows.append(row)
    return pd.DataFrame.from_records(rows).sort_values(["season", "gsis_player_id", "team", "stat"], na_position="last").reset_index(drop=True)


def _weighted_mean(values: list[float], weights: list[float]) -> float | None:
    if not values:
        return None
    return float(np.average(np.asarray(values, dtype=float), weights=np.asarray(weights, dtype=float)))


def _backtest_predictions(seasons: pd.DataFrame, historical_config: dict[str, Any]) -> pd.DataFrame:
    start = int(historical_config["calibration_start_season"])
    latest = int(historical_config["latest_completed_season"])
    prior_seasons = int(historical_config["prior_seasons"])
    minimum_history = int(historical_config["minimum_training_seasons"])
    half_life = float(historical_config["recency_half_life_seasons"])
    lookup = {(int(row.season), str(row.gsis_player_id)): row for row in seasons.itertuples(index=False)}
    rows: list[dict[str, Any]] = []

    for target in seasons.loc[seasons["season"].between(start, latest)].itertuples(index=False):
        target_season = int(target.season)
        target_schedule = int(target.schedule_games)
        player_id = str(target.gsis_player_id)
        prior_rows = [lookup.get((target_season - lag, player_id)) for lag in range(1, prior_seasons + 1)]
        for stat in TARGET_STATS:
            opportunity_name = _OPPORTUNITY_BY_STAT[stat]
            minimum_opportunity = _minimum_opportunity(stat, historical_config)
            player_values: list[float] = []
            player_weights: list[float] = []
            history_seasons: list[int] = []
            output: dict[str, Any] = {
                "target_season": target_season,
                "gsis_player_id": player_id,
                "player_name": target.player_name,
                "team": target.team,
                "position": target.position,
                "stat": stat,
                "realized_total": float(getattr(target, stat)),
                "target_games": int(target.games),
                "target_schedule_games": target_schedule,
                "realized_target_opportunity": getattr(target, opportunity_name),
                "prior_opportunity_name": opportunity_name,
                "prior_opportunity_minimum": minimum_opportunity,
                "baseline_formula": "recency_weighted_player_plus_position_cohort_v1",
            }
            for lag in range(1, 4):
                prior = prior_rows[lag - 1] if lag <= prior_seasons else None
                output[f"lag_{lag}_season"] = int(prior.season) if prior is not None else np.nan
                output[f"lag_{lag}_total"] = float(getattr(prior, stat)) if prior is not None else np.nan
                output[f"lag_{lag}_total_per_17"] = float(getattr(prior, stat)) * 17.0 / float(prior.schedule_games) if prior is not None else np.nan
                output[f"lag_{lag}_games"] = int(prior.games) if prior is not None else np.nan
                output[f"lag_{lag}_schedule_games"] = int(prior.schedule_games) if prior is not None else np.nan
                output[f"lag_{lag}_per_game"] = float(getattr(prior, stat)) / float(prior.games) if prior is not None and prior.games else np.nan
                output[f"lag_{lag}_opportunity"] = getattr(prior, opportunity_name) if prior is not None else np.nan
                prior_opportunity_value = getattr(prior, opportunity_name) if prior is not None else np.nan
                output[f"lag_{lag}_opportunity_per_17"] = prior_opportunity_value * 17.0 / float(prior.schedule_games) if prior is not None and not pd.isna(prior_opportunity_value) else np.nan
                output[f"lag_{lag}_position"] = prior.position if prior is not None else None
                output[f"lag_{lag}_age"] = prior.age if prior is not None else np.nan
                output[f"lag_{lag}_experience"] = prior.experience if prior is not None else np.nan
                if prior is not None:
                    recency_weight = 0.5 ** ((lag - 1) / half_life)
                    player_values.append(float(getattr(prior, stat)) * 17.0 / float(prior.schedule_games))
                    player_weights.append(recency_weight)
                    history_seasons.append(int(prior.season))

            position = _nonempty_text(target.position)
            cohort_values: list[float] = []
            cohort_weights: list[float] = []
            cohort_seasons: list[int] = []
            if position is not None:
                candidates = seasons.loc[
                    (seasons["position"].map(_nonempty_text) == position)
                    & (seasons["season"] < target_season)
                    & (seasons["season"] >= target_season - prior_seasons)
                ]
                for candidate in candidates.itertuples(index=False):
                    opportunity = getattr(candidate, opportunity_name)
                    if pd.isna(opportunity) or float(opportunity) < minimum_opportunity:
                        continue
                    gap = target_season - int(candidate.season)
                    cohort_values.append(float(getattr(candidate, stat)) * 17.0 / float(candidate.schedule_games))
                    cohort_weights.append(0.5 ** ((gap - 1) / half_life))
                    cohort_seasons.append(int(candidate.season))

            player_component_17 = _weighted_mean(player_values, player_weights)
            cohort_component_17 = _weighted_mean(cohort_values, cohort_weights)
            history_count = len(player_values)
            player_weight = history_count / (history_count + minimum_history) if history_count and cohort_component_17 is not None else (1.0 if history_count else 0.0)
            if player_component_17 is not None and cohort_component_17 is not None:
                baseline_17 = player_weight * player_component_17 + (1.0 - player_weight) * cohort_component_17
            elif player_component_17 is not None:
                baseline_17 = player_component_17
            else:
                baseline_17 = cohort_component_17
            baseline = baseline_17 * target_schedule / 17.0 if baseline_17 is not None else None

            prior_opportunity = getattr(prior_rows[0], opportunity_name) if prior_rows and prior_rows[0] is not None else np.nan
            opportunity_known = not pd.isna(prior_opportunity)
            opportunity_eligible = bool(opportunity_known and float(prior_opportunity) >= minimum_opportunity)
            enough_history = history_count >= minimum_history
            training_eligible = baseline is not None and enough_history and opportunity_eligible
            if not history_count:
                cohort_status = "rookie_cohort_baseline" if baseline is not None else "rookie_no_baseline"
            elif not enough_history:
                cohort_status = "insufficient_history"
            elif not opportunity_known:
                cohort_status = "missing_prior_opportunity"
            elif not opportunity_eligible:
                cohort_status = "below_prior_opportunity"
            elif cohort_component_17 is None:
                cohort_status = "missing_cohort"
            else:
                cohort_status = "training_eligible"

            feature_seasons = sorted(set(history_seasons + cohort_seasons))
            output.update({
                "player_history_seasons": history_count,
                "player_component_per_17": player_component_17,
                "cohort_component_per_17": cohort_component_17,
                "cohort_player_seasons": len(cohort_values),
                "player_shrinkage_weight": player_weight,
                "baseline_mean": baseline,
                "baseline_available": baseline is not None,
                "prior_opportunity": prior_opportunity,
                "opportunity_eligible": opportunity_eligible,
                "training_eligible": training_eligible,
                "cohort_status": cohort_status,
                "feature_seasons": "|".join(str(value) for value in feature_seasons),
                "max_feature_season": max(feature_seasons) if feature_seasons else np.nan,
            })
            rows.append(output)
    return pd.DataFrame.from_records(rows).sort_values(["target_season", "gsis_player_id", "stat"]).reset_index(drop=True)


def prepare_historical_data(path: str | Path, historical_config: dict[str, Any]) -> HistoricalPreparation:
    """Prepare long-form player seasons and deterministic lag-only baselines."""

    raw = _read_input(path)
    weekly, resolved = _canonical_weekly(raw, historical_config)
    seasons = _aggregate_player_seasons(weekly)
    _reconcile_weekly(weekly, seasons)
    _reconcile_optional_summaries(raw, weekly, resolved, seasons)

    for stat, maximum in _PLAUSIBLE_SEASON_MAX.items():
        negative = seasons[stat] < 0
        if negative.any():
            raise HistoricalDataError(
                "historical.target_totals_nonnegative",
                f"Historical player-season {stat} contains negative totals",
                {"stat": stat, "rows": int(negative.sum()), "minimum": float(seasons.loc[negative, stat].min())},
            )
        bad = seasons[stat] > maximum
        if bad.any():
            raise HistoricalDataError(
                "historical.implausible_totals",
                f"Historical player-season {stat} exceeds the plausibility limit {maximum}",
                {"stat": stat, "limit": maximum, "rows": int(bad.sum())},
            )
    bad_games = seasons["games"] > seasons["schedule_games"]
    if bad_games.any():
        raise HistoricalDataError("historical.schedule_lengths", "Player games exceed the known regular-season schedule length", {"rows": int(bad_games.sum())})

    player_seasons = _long_player_seasons(seasons)
    duplicate_output = player_seasons.duplicated(["season", "gsis_player_id", "team", "stat"], keep=False)
    if duplicate_output.any():
        raise HistoricalDataError("historical.output_keys_unique", "Historical player-season-stat output keys are not unique", {"rows": int(duplicate_output.sum())})

    predictions = _backtest_predictions(seasons, historical_config)
    leaking = predictions["max_feature_season"].notna() & (predictions["max_feature_season"] >= predictions["target_season"])
    if leaking.any():
        raise HistoricalDataError("historical.no_lookahead", "A baseline feature uses its target or a future season", {"rows": int(leaking.sum())})

    checks = [
        CheckResult("historical.seasons_complete", True, details={"first": int(seasons["season"].min()), "last": int(seasons["season"].max()), "count": int(seasons["season"].nunique())}),
        CheckResult("historical.regular_season_only", True, details={"season_type": historical_config["season_type"]}),
        CheckResult("historical.weekly_keys_unique", True, details={"rows": int(len(weekly)), "duplicate_groups_aggregated": int(weekly.attrs.get("duplicate_week_groups_aggregated", 0))}),
        CheckResult("historical.weekly_to_season_reconciled", True, details={"player_seasons": int(len(seasons)), "stats": len(TARGET_STATS)}),
        CheckResult("historical.output_keys_unique", True, details={"rows": int(len(player_seasons)), "key": ["season", "gsis_player_id", "team", "stat"]}),
        CheckResult("historical.schedule_lengths", True, details={"sixteen_game_seasons": int((seasons["schedule_games"] == 16).sum()), "seventeen_game_seasons": int((seasons["schedule_games"] == 17).sum())}),
        CheckResult("historical.no_lookahead", True, details={"baselines": int(len(predictions))}),
    ]
    missingness_rates = {
        column: float(seasons[column].isna().mean())
        for column in ("player_name", "team", "position", "passing_attempts", "rushing_attempts", "targets", "age", "experience")
    }
    for column in ("player_name", "team", "position"):
        rate = missingness_rates[column]
        checks.append(CheckResult(
            f"historical.missingness.{column}",
            rate == 0.0,
            "warning",
            f"{column} missingness rate is {rate:.2%}",
            {"field": column, "missingness_rate": rate},
        ))
    minimum = int(historical_config["minimum_player_seasons_per_stat"])
    cohort_counts: dict[str, int] = {}
    for stat in TARGET_STATS:
        count = int(predictions.loc[(predictions["stat"] == stat) & predictions["training_eligible"], :].shape[0])
        cohort_counts[stat] = count
        checks.append(CheckResult(
            f"historical.cohort_size.{stat}",
            count >= minimum,
            "error",
            f"Eligible player-season count is {count}; required minimum is {minimum}",
            {"stat": stat, "eligible_player_seasons": count, "minimum": minimum},
        ))
    failed = [check for check in checks if not check.passed and check.severity == "error"]
    validation = {
        "status": "failed" if failed else "passed",
        "source_rows": int(len(raw)),
        "regular_season_rows": int(len(weekly)),
        "player_season_rows": int(len(seasons)),
        "historical_player_season_stat_rows": int(len(player_seasons)),
        "backtest_prediction_rows": int(len(predictions)),
        "covered_seasons": sorted(int(value) for value in seasons["season"].unique()),
        "calibration_window": [int(historical_config["calibration_start_season"]), int(historical_config["latest_completed_season"])],
        "prior_seasons": int(historical_config["prior_seasons"]),
        "recency_half_life_seasons": float(historical_config["recency_half_life_seasons"]),
        "minimum_training_seasons": int(historical_config["minimum_training_seasons"]),
        "prior_opportunity_filters": dict(historical_config["prior_opportunity_filters"]),
        "opportunity_by_stat": dict(_OPPORTUNITY_BY_STAT),
        "cohort_counts": cohort_counts,
        "missingness_rates": missingness_rates,
        "team_representation": "pipe-delimited ordered teams observed across the player season",
        "checks": [check.to_dict() for check in checks],
    }
    if failed:
        first = failed[0]
        raise HistoricalDataError(first.name, first.message, {**first.details, "validation": validation})
    return HistoricalPreparation(player_seasons, predictions, validation)
