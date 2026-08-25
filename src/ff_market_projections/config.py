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
        "enabled", "source", "url", "ingest_start_season", "calibration_start_season", "latest_completed_season", "season_type", "prior_seasons", "recency_half_life_seasons", "minimum_training_seasons", "holdout_seasons", "minimum_player_seasons_per_stat", "prior_opportunity_filters",
    },
    "names": {"automatic_fuzzy_match", "minimum_score", "minimum_runner_up_gap"},
    "pricing": {"sportsbook_devig_method", "probability_tolerance", "reject_ambiguous_integer_lines"},
    "model": {"family", "dispersion_mode", "minimum_calibration_groups", "minimum_thresholds_per_group", "probability_floor", "probability_ceiling", "robust_loss", "on_calibration_failure", "bootstrap_samples", "random_seed"},
    "aggregation": {"minimum_sources", "renormalize_available_source_weights"},
    "scoring": {"missing_stat_policy", "passing_yards", "passing_touchdowns", "passing_interceptions", "rushing_yards", "rushing_touchdowns", "receiving_yards", "receiving_touchdowns", "fumbles_lost", "two_point_conversions", "reception_bonus", "required_profiles"},
    "workbook": {"filename", "freeze_header", "autofilter"},
}
_REQUIRED_TOP_LEVEL = frozenset(_SCHEMA)
_REQUIRED_SOURCES = frozenset(_SCHEMA["sources"])


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
        _positive(_expect(source, "weight", f"sources.{name}"), f"sources.{name}.weight")

    historical = values["historical"]
    for key in _SCHEMA["historical"]:
        _expect(historical, key, "historical")
    if historical["season_type"] != "REG":
        raise ConfigError("historical.season_type must be REG")
    for key in ("ingest_start_season", "calibration_start_season", "latest_completed_season", "prior_seasons", "minimum_training_seasons", "holdout_seasons", "minimum_player_seasons_per_stat"):
        _positive(historical[key], f"historical.{key}")
    if not historical["ingest_start_season"] <= historical["calibration_start_season"] <= historical["latest_completed_season"]:
        raise ConfigError("historical season window must satisfy ingest_start <= calibration_start <= latest_completed")
    filters = historical["prior_opportunity_filters"]
    if not isinstance(filters, dict):
        raise ConfigError("historical.prior_opportunity_filters must be a TOML table")
    _assert_keys(filters, {"passing_attempts", "rushing_attempts", "targets"}, "historical.prior_opportunity_filters")
    for key in ("passing_attempts", "rushing_attempts", "targets"):
        _positive(_expect(filters, key, "historical.prior_opportunity_filters"), f"historical.prior_opportunity_filters.{key}", allow_zero=True)

    pricing = values["pricing"]
    if pricing.get("sportsbook_devig_method") not in {"proportional", "power"}:
        raise ConfigError("pricing.sportsbook_devig_method must be proportional or power")
    _positive(_expect(pricing, "probability_tolerance", "pricing"), "pricing.probability_tolerance")

    model = values["model"]
    if model.get("family") != "negative_binomial":
        raise ConfigError("model.family must be negative_binomial")
    if model.get("dispersion_mode") not in {"historical_only", "historical_with_kalshi_update"}:
        raise ConfigError("model.dispersion_mode is unsupported")
    floor, ceiling = _expect(model, "probability_floor", "model"), _expect(model, "probability_ceiling", "model")
    if not (isinstance(floor, (int, float)) and isinstance(ceiling, (int, float)) and 0 < floor < ceiling < 1):
        raise ConfigError("model probability_floor and probability_ceiling must satisfy 0 < floor < ceiling < 1")


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
