"""Strict loading for the checked-in pipeline TOML contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


class ConfigError(ValueError):
    """A configuration violates the explicit pipeline contract."""


_SCHEMA: dict[str, Any] = {
    "run": {"season", "timezone", "fail_on_warning", "max_snapshot_age_hours", "max_cross_source_skew_minutes"},
    "sources": {
        "draftkings": {"enabled", "weight"},
        "fanduel": {"enabled", "weight", "state"},
        "kalshi": {"enabled", "weight", "require_two_sided_quote", "max_spread_probability_points", "min_open_interest_contracts"},
    },
    "historical": {
        "enabled", "source", "url", "ingest_start_season", "calibration_start_season", "latest_completed_season", "season_type", "prior_seasons", "recency_half_life_seasons", "minimum_training_seasons", "holdout_seasons", "minimum_player_seasons_per_stat", "prior_opportunity_filters", "baseline_bias_correction",
    },
    "names": {"automatic_fuzzy_match", "minimum_score", "minimum_runner_up_gap"},
    "pricing": {"sportsbook_devig_method", "probability_tolerance", "reject_ambiguous_integer_lines"},
    "model": {"family", "dispersion_mode", "minimum_calibration_groups", "minimum_thresholds_per_group", "probability_floor", "probability_ceiling", "robust_loss", "on_calibration_failure", "bootstrap_samples", "random_seed", "historical_calibration", "current_market"},
    "aggregation": {
        "minimum_sources", "renormalize_available_source_weights",
        "max_absolute_disagreement", "max_relative_disagreement",
    },
    "scoring": {"missing_stat_policy", "passing_yards", "passing_touchdowns", "passing_interceptions", "rushing_yards", "rushing_touchdowns", "receiving_yards", "receiving_touchdowns", "fumbles_lost", "two_point_conversions", "reception_bonus", "required_profiles"},
    "workbook": {"filename", "freeze_header", "autofilter"},
}
_REQUIRED_TOP_LEVEL = frozenset(_SCHEMA)
_REQUIRED_SOURCES = frozenset(_SCHEMA["sources"])
_TARGET_STATS = frozenset({
    "passing_yards", "passing_touchdowns", "rushing_yards", "rushing_touchdowns",
    "receiving_yards", "receiving_touchdowns", "receptions",
})
_HISTORICAL_CALIBRATION_KEYS = {
    "sensitivity_start_seasons", "bootstrap_confidence_level",
    "minimum_holdout_player_seasons_per_stat", "minimum_sensitivity_player_seasons_per_stat",
    "brier_event_cdf_levels", "interval_levels", "mean_calibration_bins",
    "max_holdout_nll_regret_per_player_season", "max_brier_calibration_gap",
    "max_interval_coverage_error", "max_abs_relative_bias",
    "max_sensitivity_log_dispersion_delta", "minimum_bootstrap_success_rate",
    "dispersion_bounds",
}
_BASELINE_BIAS_CORRECTION_KEYS = {
    "method", "minimum_seasons", "exponent_bounds",
    "recency_half_life_candidates", "minimum_validation_seasons",
}
_CURRENT_MARKET_KEYS = {
    "optimizer_tolerance", "optimizer_max_evaluations",
    "max_current_market_logit_rmse", "max_current_market_holdout_logit_mae",
    "max_current_market_log_dispersion_delta", "current_market_conflict_policy",
    "max_sportsbook_probability_residual", "max_kalshi_logit_rmse",
    "max_kalshi_holdout_logit_mae", "near_even_probability_band",
    "max_near_even_mean_shift_relative", "max_near_even_mean_shift_absolute", "mean_bounds",
}


@dataclass(frozen=True)
class PipelineConfig:
    """Validated configuration plus the original, copyable TOML bytes."""

    path: Path
    values: dict[str, Any]
    raw_toml: bytes

    @property
    def season(self) -> str:
        return self.values["run"]["season"]


def _assert_keys(mapping: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(f"Unknown setting(s) at {location}: {', '.join(unknown)}")


def _expect(mapping: dict[str, Any], key: str, location: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing required setting {location}.{key}")
    return mapping[key]


def _positive(value: Any, location: str, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 or (value == 0 and not allow_zero):
        comparator = "non-negative" if allow_zero else "positive"
        raise ConfigError(f"{location} must be a {comparator} number")


def _validate(values: dict[str, Any]) -> None:
    _assert_keys(values, set(_SCHEMA), "root")
    missing = sorted(_REQUIRED_TOP_LEVEL - set(values))
    if missing:
        raise ConfigError(f"Missing required section(s): {', '.join(missing)}")
    for section, schema in _SCHEMA.items():
        value = values[section]
        if not isinstance(value, dict):
            raise ConfigError(f"{section} must be a TOML table")
        _assert_keys(value, set(schema), section)

    run = values["run"]
    for key in ("season", "timezone", "fail_on_warning", "max_snapshot_age_hours", "max_cross_source_skew_minutes"):
        _expect(run, key, "run")
    if not isinstance(run["season"], str) or len(run["season"]) != 7 or run["season"][4] != "-":
        raise ConfigError("run.season must use YYYY-YY format")
    _positive(run["max_snapshot_age_hours"], "run.max_snapshot_age_hours")
    _positive(run["max_cross_source_skew_minutes"], "run.max_cross_source_skew_minutes", allow_zero=True)

    sources = values["sources"]
    _assert_keys(sources, set(_REQUIRED_SOURCES), "sources")
    missing_sources = sorted(_REQUIRED_SOURCES - set(sources))
    if missing_sources:
        raise ConfigError(f"Missing required source(s): {', '.join(missing_sources)}")
    for name, source in sources.items():
        if not isinstance(source, dict):
            raise ConfigError(f"sources.{name} must be a TOML table")
        _assert_keys(source, set(_SCHEMA["sources"][name]), f"sources.{name}")
        _expect(source, "enabled", f"sources.{name}")
        _positive(
            _expect(source, "weight", f"sources.{name}"),
            f"sources.{name}.weight",
            allow_zero=True,
        )

    historical = values["historical"]
    for key in _SCHEMA["historical"]:
        _expect(historical, key, "historical")
    if historical["season_type"] != "REG":
        raise ConfigError("historical.season_type must be REG")
    for key in ("ingest_start_season", "calibration_start_season", "latest_completed_season", "prior_seasons", "recency_half_life_seasons", "minimum_training_seasons", "holdout_seasons", "minimum_player_seasons_per_stat"):
        _positive(historical[key], f"historical.{key}")
    if not historical["ingest_start_season"] <= historical["calibration_start_season"] <= historical["latest_completed_season"]:
        raise ConfigError("historical season window must satisfy ingest_start <= calibration_start <= latest_completed")
    if historical["ingest_start_season"] < 1999:
        raise ConfigError("historical.ingest_start_season must be 1999 or later")
    if not 1 <= historical["prior_seasons"] <= 3:
        raise ConfigError("historical.prior_seasons must be between 1 and 3")
    if historical["minimum_training_seasons"] > historical["prior_seasons"]:
        raise ConfigError("historical.minimum_training_seasons cannot exceed historical.prior_seasons")
    calibration_seasons = historical["latest_completed_season"] - historical["calibration_start_season"] + 1
    if historical["holdout_seasons"] >= calibration_seasons:
        raise ConfigError("historical.holdout_seasons must leave at least one pre-holdout calibration season")
    run_start_season = int(run["season"][:4])
    if historical["latest_completed_season"] >= run_start_season:
        raise ConfigError("historical.latest_completed_season must precede the configured run season")
    filters = historical["prior_opportunity_filters"]
    if not isinstance(filters, dict):
        raise ConfigError("historical.prior_opportunity_filters must be a TOML table")
    _assert_keys(filters, {"passing_attempts", "rushing_attempts", "targets"}, "historical.prior_opportunity_filters")
    for key in ("passing_attempts", "rushing_attempts", "targets"):
        _positive(_expect(filters, key, "historical.prior_opportunity_filters"), f"historical.prior_opportunity_filters.{key}", allow_zero=True)
    bias_correction = historical["baseline_bias_correction"]
    if not isinstance(bias_correction, dict):
        raise ConfigError("historical.baseline_bias_correction must be a TOML table")
    _assert_keys(bias_correction, _BASELINE_BIAS_CORRECTION_KEYS, "historical.baseline_bias_correction")
    for key in _BASELINE_BIAS_CORRECTION_KEYS:
        _expect(bias_correction, key, "historical.baseline_bias_correction")
    if bias_correction["method"] != "rolling_power_poisson":
        raise ConfigError("historical.baseline_bias_correction.method must be rolling_power_poisson")
    minimum_bias_seasons = bias_correction["minimum_seasons"]
    if isinstance(minimum_bias_seasons, bool) or not isinstance(minimum_bias_seasons, int) or minimum_bias_seasons < 2:
        raise ConfigError("historical.baseline_bias_correction.minimum_seasons must be an integer of at least two")
    minimum_validation_seasons = bias_correction["minimum_validation_seasons"]
    if (
        isinstance(minimum_validation_seasons, bool)
        or not isinstance(minimum_validation_seasons, int)
        or minimum_validation_seasons < 1
    ):
        raise ConfigError("historical.baseline_bias_correction.minimum_validation_seasons must be a positive integer")
    exponent_bounds = bias_correction["exponent_bounds"]
    if (
        not isinstance(exponent_bounds, list) or len(exponent_bounds) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in exponent_bounds)
        or not 0 < exponent_bounds[0] < exponent_bounds[1]
    ):
        raise ConfigError("historical.baseline_bias_correction.exponent_bounds must be [positive_min, larger_max]")
    half_life_candidates = bias_correction["recency_half_life_candidates"]
    if (
        not isinstance(half_life_candidates, list) or not half_life_candidates
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0 for value in half_life_candidates)
        or half_life_candidates != sorted(set(half_life_candidates))
    ):
        raise ConfigError(
            "historical.baseline_bias_correction.recency_half_life_candidates must be unique increasing positive numbers"
        )

    pricing = values["pricing"]
    if pricing.get("sportsbook_devig_method") != "proportional":
        raise ConfigError("pricing.sportsbook_devig_method must be proportional")
    _positive(_expect(pricing, "probability_tolerance", "pricing"), "pricing.probability_tolerance")
    if not isinstance(_expect(pricing, "reject_ambiguous_integer_lines", "pricing"), bool):
        raise ConfigError("pricing.reject_ambiguous_integer_lines must be boolean")
    kalshi = sources["kalshi"]
    if not isinstance(kalshi["require_two_sided_quote"], bool):
        raise ConfigError("sources.kalshi.require_two_sided_quote must be boolean")
    _positive(kalshi["max_spread_probability_points"], "sources.kalshi.max_spread_probability_points", allow_zero=True)
    _positive(kalshi["min_open_interest_contracts"], "sources.kalshi.min_open_interest_contracts", allow_zero=True)

    model = values["model"]
    if model.get("family") != "negative_binomial":
        raise ConfigError("model.family must be negative_binomial")
    if model.get("dispersion_mode") not in {"historical_only", "historical_with_current_market_update"}:
        raise ConfigError("model.dispersion_mode is unsupported")
    if model.get("on_calibration_failure") != "fail":
        raise ConfigError("model.on_calibration_failure must be fail")
    _positive(_expect(model, "bootstrap_samples", "model"), "model.bootstrap_samples")
    if isinstance(model["bootstrap_samples"], bool) or not isinstance(model["bootstrap_samples"], int):
        raise ConfigError("model.bootstrap_samples must be an integer")
    if isinstance(_expect(model, "random_seed", "model"), bool) or not isinstance(model["random_seed"], int):
        raise ConfigError("model.random_seed must be an integer")
    floor, ceiling = _expect(model, "probability_floor", "model"), _expect(model, "probability_ceiling", "model")
    if not (isinstance(floor, (int, float)) and isinstance(ceiling, (int, float)) and 0 < floor < ceiling < 1):
        raise ConfigError("model probability_floor and probability_ceiling must satisfy 0 < floor < ceiling < 1")
    for key in ("minimum_calibration_groups", "minimum_thresholds_per_group"):
        value = _expect(model, key, "model")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError(f"model.{key} must be a positive integer")
    if model["minimum_thresholds_per_group"] < 2:
        raise ConfigError("model.minimum_thresholds_per_group must be at least two")
    if _expect(model, "robust_loss", "model") != "soft_l1":
        raise ConfigError("model.robust_loss must be soft_l1")

    calibration = _expect(model, "historical_calibration", "model")
    if not isinstance(calibration, dict):
        raise ConfigError("model.historical_calibration must be a TOML table")
    _assert_keys(calibration, _HISTORICAL_CALIBRATION_KEYS, "model.historical_calibration")
    for key in _HISTORICAL_CALIBRATION_KEYS:
        _expect(calibration, key, "model.historical_calibration")
    starts = calibration["sensitivity_start_seasons"]
    if (
        not isinstance(starts, list) or not starts
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1999 for value in starts)
        or starts != sorted(set(starts))
    ):
        raise ConfigError("model.historical_calibration.sensitivity_start_seasons must be unique increasing seasons from 1999 onward")
    for key in ("brier_event_cdf_levels", "interval_levels"):
        levels = calibration[key]
        if not isinstance(levels, list) or not levels or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1 for value in levels):
            raise ConfigError(f"model.historical_calibration.{key} must contain probabilities strictly between zero and one")
        if levels != sorted(set(levels)):
            raise ConfigError(f"model.historical_calibration.{key} must be unique and increasing")
    for key in ("bootstrap_confidence_level", "minimum_bootstrap_success_rate"):
        value = calibration[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
            raise ConfigError(f"model.historical_calibration.{key} must be strictly between zero and one")
    for key in ("minimum_holdout_player_seasons_per_stat", "minimum_sensitivity_player_seasons_per_stat", "mean_calibration_bins"):
        value = calibration[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError(f"model.historical_calibration.{key} must be a positive integer")
    for key in (
        "max_holdout_nll_regret_per_player_season", "max_brier_calibration_gap",
        "max_interval_coverage_error", "max_abs_relative_bias",
        "max_sensitivity_log_dispersion_delta",
    ):
        _positive(calibration[key], f"model.historical_calibration.{key}", allow_zero=True)
    for key in ("max_brier_calibration_gap", "max_interval_coverage_error", "max_abs_relative_bias"):
        if calibration[key] > 1:
            raise ConfigError(f"model.historical_calibration.{key} cannot exceed one")
    bounds = calibration["dispersion_bounds"]
    if not isinstance(bounds, dict):
        raise ConfigError("model.historical_calibration.dispersion_bounds must be a TOML table")
    _assert_keys(bounds, set(_TARGET_STATS), "model.historical_calibration.dispersion_bounds")
    missing_bounds = sorted(_TARGET_STATS - set(bounds))
    if missing_bounds:
        raise ConfigError(f"Missing dispersion bound(s) for: {', '.join(missing_bounds)}")
    for stat, stat_bounds in bounds.items():
        if (
            not isinstance(stat_bounds, list) or len(stat_bounds) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in stat_bounds)
            or not 0 < stat_bounds[0] < stat_bounds[1]
        ):
            raise ConfigError(f"model.historical_calibration.dispersion_bounds.{stat} must be [positive_min, larger_max]")

    current = _expect(model, "current_market", "model")
    if not isinstance(current, dict):
        raise ConfigError("model.current_market must be a TOML table")
    _assert_keys(current, _CURRENT_MARKET_KEYS, "model.current_market")
    for key in _CURRENT_MARKET_KEYS:
        _expect(current, key, "model.current_market")
    _positive(current["optimizer_tolerance"], "model.current_market.optimizer_tolerance")
    evaluations = current["optimizer_max_evaluations"]
    if isinstance(evaluations, bool) or not isinstance(evaluations, int) or evaluations <= 0:
        raise ConfigError("model.current_market.optimizer_max_evaluations must be a positive integer")
    for key in (
        "max_current_market_logit_rmse", "max_current_market_holdout_logit_mae",
        "max_current_market_log_dispersion_delta",
        "max_sportsbook_probability_residual", "max_kalshi_logit_rmse",
        "max_kalshi_holdout_logit_mae", "max_near_even_mean_shift_relative",
        "max_near_even_mean_shift_absolute",
    ):
        _positive(current[key], f"model.current_market.{key}", allow_zero=True)
    if current["max_sportsbook_probability_residual"] >= 1:
        raise ConfigError("model.current_market.max_sportsbook_probability_residual must be less than one")
    if current["current_market_conflict_policy"] not in {"warning", "fail"}:
        raise ConfigError("model.current_market.current_market_conflict_policy must be warning or fail")
    near_even_band = current["near_even_probability_band"]
    if (
        not isinstance(near_even_band, list) or len(near_even_band) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in near_even_band)
        or not 0 < near_even_band[0] < 0.5 < near_even_band[1] < 1
    ):
        raise ConfigError("model.current_market.near_even_probability_band must straddle 0.5 inside (0, 1)")
    if current["max_near_even_mean_shift_relative"] > 1:
        raise ConfigError("model.current_market.max_near_even_mean_shift_relative cannot exceed one")
    mean_bounds = current["mean_bounds"]
    if not isinstance(mean_bounds, dict):
        raise ConfigError("model.current_market.mean_bounds must be a TOML table")
    _assert_keys(mean_bounds, set(_TARGET_STATS), "model.current_market.mean_bounds")
    missing_mean_bounds = sorted(_TARGET_STATS - set(mean_bounds))
    if missing_mean_bounds:
        raise ConfigError(f"Missing mean bound(s) for: {', '.join(missing_mean_bounds)}")
    for stat, stat_bounds in mean_bounds.items():
        if (
            not isinstance(stat_bounds, list) or len(stat_bounds) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in stat_bounds)
            or not 0 < stat_bounds[0] < stat_bounds[1]
        ):
            raise ConfigError(f"model.current_market.mean_bounds.{stat} must be [positive_min, larger_max]")

    aggregation = values["aggregation"]
    for key in _SCHEMA["aggregation"]:
        _expect(aggregation, key, "aggregation")
    minimum_sources = aggregation["minimum_sources"]
    if isinstance(minimum_sources, bool) or not isinstance(minimum_sources, int) or minimum_sources <= 0:
        raise ConfigError("aggregation.minimum_sources must be a positive integer")
    if not isinstance(aggregation["renormalize_available_source_weights"], bool):
        raise ConfigError("aggregation.renormalize_available_source_weights must be boolean")
    for key in ("max_absolute_disagreement", "max_relative_disagreement"):
        _positive(aggregation[key], f"aggregation.{key}", allow_zero=True)

    scoring = values["scoring"]
    if scoring.get("missing_stat_policy") not in {"blank_total", "partial_total"}:
        raise ConfigError("scoring.missing_stat_policy must be blank_total or partial_total")
    scoring_stats = {
        "passing_yards", "passing_touchdowns", "passing_interceptions",
        "rushing_yards", "rushing_touchdowns", "receiving_yards",
        "receiving_touchdowns", "fumbles_lost", "two_point_conversions",
    }
    for stat in scoring_stats:
        value = _expect(scoring, stat, "scoring")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"scoring.{stat} must be numeric")
    reception_bonus = _expect(scoring, "reception_bonus", "scoring")
    if not isinstance(reception_bonus, dict):
        raise ConfigError("scoring.reception_bonus must be a TOML table")
    _assert_keys(reception_bonus, {"standard", "half_ppr", "three_quarter_ppr", "full_ppr"}, "scoring.reception_bonus")
    for mode in ("standard", "half_ppr", "three_quarter_ppr", "full_ppr"):
        value = _expect(reception_bonus, mode, "scoring.reception_bonus")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ConfigError(f"scoring.reception_bonus.{mode} must be non-negative numeric")
    if reception_bonus != {"standard": 0.0, "half_ppr": 0.5, "three_quarter_ppr": 0.75, "full_ppr": 1.0}:
        raise ConfigError("scoring.reception_bonus must define the standard, half_ppr, three_quarter_ppr, and full_ppr modes")
    profiles = _expect(scoring, "required_profiles", "scoring")
    if not isinstance(profiles, dict) or not profiles:
        raise ConfigError("scoring.required_profiles must be a non-empty TOML table")
    allowed_profile_stats = scoring_stats | {"receptions"}
    for profile, components in profiles.items():
        if not isinstance(profile, str) or not profile or not isinstance(components, list) or not components:
            raise ConfigError("scoring.required_profiles entries must be non-empty lists")
        if len(components) != len(set(components)) or any(component not in allowed_profile_stats for component in components):
            raise ConfigError(f"scoring.required_profiles.{profile} contains duplicate or unsupported components")


def load_config(path: str | Path) -> PipelineConfig:
    """Load TOML, rejecting malformed, duplicate, unknown, and invalid settings."""

    config_path = Path(path)
    try:
        raw = config_path.read_bytes()
        values = tomllib.loads(raw.decode("utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file does not exist: {config_path}") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(f"Configuration must be UTF-8: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {config_path}: {exc}") from exc
    if not isinstance(values, dict):
        raise ConfigError("Configuration root must be a TOML table")
    _validate(values)
    return PipelineConfig(path=config_path, values=values, raw_toml=raw)
