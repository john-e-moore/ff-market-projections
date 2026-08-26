"""Leakage-free historical predictive-dispersion calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom, poisson

from .contracts import CheckResult
from .distributions import negative_binomial_interval, negative_binomial_logpmf, negative_binomial_survival
from .historical import TARGET_STATS


CALIBRATION_VERSION = "negative_binomial_historical_v2"
_REQUIRED_COLUMNS = {
    "target_season", "gsis_player_id", "position", "stat", "realized_total",
    "target_games", "baseline_mean", "training_eligible", "feature_seasons",
    "max_feature_season", "baseline_mean_raw", "baseline_bias_method",
    "baseline_bias_intercept", "baseline_bias_exponent",
    "baseline_bias_recency_half_life_seasons",
    "baseline_bias_max_observation_season", "baseline_bias_optimizer_converged",
    "baseline_bias_bound_hit",
}


class CalibrationError(ValueError):
    """Historical calibration input violates a hard contract."""

    def __init__(self, check_name: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.check = CheckResult(check_name, False, "error", message, details or {})


@dataclass(frozen=True)
class DispersionFit:
    dispersion: float
    log_dispersion: float
    objective_nll: float
    nll_per_weight: float
    effective_weight: float
    converged: bool
    bound_hit: bool
    optimizer_iterations: int
    optimizer_evaluations: int


@dataclass(frozen=True)
class HistoricalCalibration:
    dispersions: pd.DataFrame
    report: dict[str, Any]


def _as_float_array(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise CalibrationError("historical_calibration.fit_inputs", f"{name} must be a nonempty finite one-dimensional array")
    return result


def fit_historical_dispersion(
    outcomes: Any,
    means: Any,
    seasons: Any,
    bounds: tuple[float, float],
    recency_half_life_seasons: float,
    *,
    reference_season: int | None = None,
) -> DispersionFit:
    """Fit one dispersion with fixed means and recency-weighted NB likelihood."""

    realized = _as_float_array(outcomes, "outcomes")
    baseline = _as_float_array(means, "means")
    target_seasons = _as_float_array(seasons, "seasons")
    if not (len(realized) == len(baseline) == len(target_seasons)):
        raise CalibrationError("historical_calibration.fit_inputs", "outcomes, means, and seasons must have equal lengths")
    if (baseline <= 0).any():
        raise CalibrationError("historical_calibration.fit_inputs", "calibration means must be positive")
    if (realized < 0).any() or not np.allclose(realized, np.rint(realized), rtol=0.0, atol=1e-9):
        raise CalibrationError("historical_calibration.fit_inputs", "calibration outcomes must be nonnegative integers")
    lower, upper = (float(bounds[0]), float(bounds[1]))
    if not 0 < lower < upper:
        raise CalibrationError("historical_calibration.dispersion_bounds", "dispersion bounds must satisfy 0 < lower < upper")
    if not np.isfinite(recency_half_life_seasons) or recency_half_life_seasons <= 0:
        raise CalibrationError("historical_calibration.fit_inputs", "recency half-life must be finite and positive")

    reference = int(reference_season if reference_season is not None else np.max(target_seasons))
    weights = 0.5 ** ((reference - target_seasons) / float(recency_half_life_seasons))
    weights = weights / float(np.mean(weights))
    log_bounds = (math.log(lower), math.log(upper))

    def objective(log_dispersion: float) -> float:
        values = negative_binomial_logpmf(realized, baseline, math.exp(log_dispersion))
        if not np.isfinite(values).all():
            return math.inf
        return float(-np.dot(weights, values))

    optimized = minimize_scalar(
        objective,
        bounds=log_bounds,
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 500},
    )
    log_dispersion = float(optimized.x)
    tolerance = max(1e-5, (log_bounds[1] - log_bounds[0]) * 1e-5)
    bound_hit = log_dispersion - log_bounds[0] <= tolerance or log_bounds[1] - log_dispersion <= tolerance
    objective_value = float(optimized.fun)
    return DispersionFit(
        dispersion=float(math.exp(log_dispersion)),
        log_dispersion=log_dispersion,
        objective_nll=objective_value,
        nll_per_weight=objective_value / float(np.sum(weights)),
        effective_weight=float(np.sum(weights)),
        converged=bool(optimized.success and np.isfinite(objective_value)),
        bound_hit=bool(bound_hit),
        optimizer_iterations=int(getattr(optimized, "nit", 0)),
        optimizer_evaluations=int(getattr(optimized, "nfev", 0)),
    )


def _bootstrap_groups(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, list[np.ndarray]]]:
    base = frame.reset_index(drop=True)
    groups: dict[int, list[np.ndarray]] = {}
    for season, season_rows in base.groupby("target_season", sort=True):
        groups[int(season)] = [
            group.index.to_numpy(dtype=int)
            for _, group in season_rows.groupby("gsis_player_id", sort=True)
        ]
    return base, groups


def _bootstrap_sample(
    frame: pd.DataFrame,
    groups: dict[int, list[np.ndarray]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Hierarchically resample target seasons, then player groups within season."""

    season_values = np.asarray(sorted(groups), dtype=int)
    sampled_seasons = rng.choice(season_values, size=len(season_values), replace=True)
    sampled_indices: list[np.ndarray] = []
    for season in sampled_seasons:
        player_groups = groups[int(season)]
        sampled_players = rng.integers(0, len(player_groups), size=len(player_groups))
        sampled_indices.extend(player_groups[int(index)] for index in sampled_players)
    return frame.iloc[np.concatenate(sampled_indices)].reset_index(drop=True)


def grouped_bootstrap_log_dispersion(
    frame: pd.DataFrame,
    bounds: tuple[float, float],
    recency_half_life_seasons: float,
    *,
    samples: int,
    seed: int,
    confidence_level: float,
    minimum_success_rate: float,
    reference_season: int,
) -> dict[str, Any]:
    """Bootstrap season/player groups and return uncertainty in log dispersion."""

    rng = np.random.default_rng(seed)
    base, groups = _bootstrap_groups(frame)
    estimates: list[float] = []
    bound_hits = 0
    failures = 0
    for _ in range(samples):
        sampled = _bootstrap_sample(base, groups, rng)
        fit = fit_historical_dispersion(
            sampled["realized_total"], sampled["baseline_mean"], sampled["target_season"],
            bounds, recency_half_life_seasons, reference_season=reference_season,
        )
        if not fit.converged:
            failures += 1
        elif fit.bound_hit:
            bound_hits += 1
        else:
            estimates.append(fit.log_dispersion)
    successes = len(estimates)
    success_rate = successes / samples
    tail = (1.0 - confidence_level) / 2.0
    lower = float(np.quantile(estimates, tail)) if estimates else math.nan
    upper = float(np.quantile(estimates, 1.0 - tail)) if estimates else math.nan
    return {
        "samples_requested": int(samples),
        "successful_samples": successes,
        "failed_samples": failures,
        "bound_hit_samples": bound_hits,
        "success_rate": float(success_rate),
        "minimum_success_rate": float(minimum_success_rate),
        "confidence_level": float(confidence_level),
        "log_dispersion_lower": lower,
        "log_dispersion_upper": upper,
        "dispersion_lower": float(math.exp(lower)) if np.isfinite(lower) else math.nan,
        "dispersion_upper": float(math.exp(upper)) if np.isfinite(upper) else math.nan,
        "passed": bool(success_rate >= minimum_success_rate and np.isfinite([lower, upper]).all()),
        "grouping": "resample_target_seasons_then_players_within_season",
        "seed": int(seed),
    }


def _boolean_series(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = series.astype("string").str.strip().str.lower().map({"true": True, "false": False})
    if mapped.isna().any():
        raise CalibrationError("historical_calibration.input_types", f"{name} must contain only true/false values")
    return mapped.astype(bool)


def _validate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise CalibrationError("historical_calibration.required_columns", f"Historical backtest predictions are missing columns: {', '.join(missing)}")
    result = frame.copy()
    for column in (
        "target_season", "realized_total", "target_games", "baseline_mean", "baseline_mean_raw",
        "baseline_bias_intercept", "baseline_bias_exponent",
        "baseline_bias_recency_half_life_seasons",
        "baseline_bias_max_observation_season", "max_feature_season",
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[["target_season", "realized_total", "target_games"]].isna().any(axis=None):
        raise CalibrationError("historical_calibration.input_types", "Target seasons, outcomes, and games must be numeric and present")
    result["training_eligible"] = _boolean_series(result["training_eligible"], "training_eligible")
    result["baseline_bias_optimizer_converged"] = _boolean_series(
        result["baseline_bias_optimizer_converged"], "baseline_bias_optimizer_converged"
    )
    result["baseline_bias_bound_hit"] = _boolean_series(
        result["baseline_bias_bound_hit"], "baseline_bias_bound_hit"
    )
    unknown_stats = sorted(set(result["stat"].dropna().astype(str)) - set(TARGET_STATS))
    if unknown_stats:
        raise CalibrationError("historical_calibration.supported_stats", f"Unsupported historical stat(s): {', '.join(unknown_stats)}")
    duplicates = result.duplicated(["target_season", "gsis_player_id", "stat"], keep=False)
    if duplicates.any():
        raise CalibrationError("historical_calibration.input_keys", "Historical backtest prediction keys are not unique", {"rows": int(duplicates.sum())})

    leakage_rows: list[int] = []
    lineage_mismatches: list[int] = []
    for index, row in result.iterrows():
        target = int(row["target_season"])
        feature_values: list[int] = []
        raw_features = row.get("feature_seasons")
        if not pd.isna(raw_features) and str(raw_features).strip():
            try:
                feature_values.extend(int(value) for value in str(raw_features).split("|") if value)
            except ValueError as exc:
                raise CalibrationError("historical_calibration.feature_lineage", "feature_seasons must be pipe-delimited integer seasons") from exc
        for column in ("lag_1_season", "lag_2_season", "lag_3_season"):
            if column in result.columns and not pd.isna(row[column]):
                feature_values.append(int(float(row[column])))
        if not pd.isna(row["baseline_bias_max_observation_season"]):
            feature_values.append(int(float(row["baseline_bias_max_observation_season"])))
        if any(value >= target for value in feature_values):
            leakage_rows.append(int(index))
        declared = row["max_feature_season"]
        if feature_values and (pd.isna(declared) or int(float(declared)) != max(feature_values)):
            lineage_mismatches.append(int(index))
        if not feature_values and not pd.isna(declared):
            lineage_mismatches.append(int(index))
    if leakage_rows:
        raise CalibrationError("historical_calibration.no_lookahead", "A baseline uses its target or a future season", {"rows": len(leakage_rows), "sample_indices": leakage_rows[:10]})
    if lineage_mismatches:
        raise CalibrationError("historical_calibration.feature_lineage", "max_feature_season does not reconcile to recorded feature seasons", {"rows": len(lineage_mismatches), "sample_indices": lineage_mismatches[:10]})
    fitted_bias = result["baseline_bias_method"] == "rolling_power_poisson"
    invalid_bias = fitted_bias & (
        result["baseline_mean_raw"].isna()
        | ~np.isfinite(result["baseline_mean_raw"])
        | (result["baseline_mean_raw"] < 0)
        | result["baseline_bias_intercept"].isna()
        | ~np.isfinite(result["baseline_bias_intercept"])
        | result["baseline_bias_exponent"].isna()
        | ~np.isfinite(result["baseline_bias_exponent"])
        | (result["baseline_bias_exponent"] <= 0)
        | result["baseline_bias_recency_half_life_seasons"].isna()
        | ~np.isfinite(result["baseline_bias_recency_half_life_seasons"])
        | (result["baseline_bias_recency_half_life_seasons"] <= 0)
        | ~result["baseline_bias_optimizer_converged"]
        | result["baseline_bias_bound_hit"]
        | result["baseline_bias_max_observation_season"].isna()
    )
    if invalid_bias.any():
        raise CalibrationError(
            "historical_calibration.baseline_bias_correction",
            "Fitted baseline bias corrections must have finite positive inputs and converged interior parameters",
            {"rows": int(invalid_bias.sum())},
        )
    return result


def _simple_group_metrics(frame: pd.DataFrame, dispersion: float) -> dict[str, Any]:
    realized = frame["realized_total"].to_numpy(dtype=float)
    means = frame["baseline_mean"].to_numpy(dtype=float)
    errors = realized - means
    logpmf = negative_binomial_logpmf(realized, means, dispersion)
    return {
        "count": int(len(frame)),
        "mean_baseline": float(np.mean(means)),
        "mean_realized": float(np.mean(realized)),
        "bias": float(np.mean(errors)),
        "mean_absolute_error": float(np.mean(np.abs(errors))),
        "negative_log_likelihood_per_player_season": float(-np.mean(logpmf)),
    }


def _holdout_metrics(frame: pd.DataFrame, dispersion: float, calibration_config: dict[str, Any]) -> dict[str, Any]:
    realized = frame["realized_total"].to_numpy(dtype=float)
    means = frame["baseline_mean"].to_numpy(dtype=float)
    raw_means = frame["baseline_mean_raw"].to_numpy(dtype=float)
    errors = realized - means
    raw_errors = realized - raw_means
    model_logpmf = negative_binomial_logpmf(realized, means, dispersion)
    poisson_logpmf = poisson.logpmf(np.rint(realized), means)

    probability = dispersion / (dispersion + means)
    brier_events: list[dict[str, Any]] = []
    for cdf_level in calibration_config["brier_event_cdf_levels"]:
        thresholds = np.asarray(nbinom.ppf(float(cdf_level), dispersion, probability), dtype=int) + 1
        predicted = negative_binomial_survival(thresholds, means, dispersion)
        observed = (realized >= thresholds).astype(float)
        brier_events.append({
            "cdf_level": float(cdf_level),
            "rows": int(len(frame)),
            "brier_score": float(np.mean((predicted - observed) ** 2)),
            "mean_predicted_probability": float(np.mean(predicted)),
            "empirical_event_rate": float(np.mean(observed)),
            "calibration_gap": float(abs(np.mean(predicted) - np.mean(observed))),
        })

    coverage: list[dict[str, Any]] = []
    for level in calibration_config["interval_levels"]:
        lower, upper = negative_binomial_interval(means, dispersion, float(level))
        observed_coverage = float(np.mean((realized >= lower) & (realized <= upper)))
        model_coverage = float(np.mean(
            nbinom.cdf(upper, dispersion, probability)
            - nbinom.cdf(lower - 1, dispersion, probability)
        ))
        coverage.append({
            "nominal_level": float(level),
            "mean_model_implied_coverage": model_coverage,
            "empirical_coverage": observed_coverage,
            "absolute_error": float(abs(observed_coverage - model_coverage)),
        })

    ordered = frame.sort_values(["baseline_mean", "target_season", "gsis_player_id"]).copy()
    bin_count = min(int(calibration_config["mean_calibration_bins"]), len(ordered))
    ordered["mean_bin"] = pd.qcut(np.arange(len(ordered)), q=bin_count, labels=False) + 1
    mean_bins = [
        {"bin": int(bin_value), **_simple_group_metrics(group, dispersion)}
        for bin_value, group in ordered.groupby("mean_bin", sort=True)
    ]
    positions = [
        {"position": str(position) if not pd.isna(position) else "missing", **_simple_group_metrics(group, dispersion)}
        for position, group in frame.assign(position=frame["position"].fillna("missing")).groupby("position", sort=True, dropna=False)
    ]
    games = frame["target_games"].to_numpy(dtype=float)
    bands = np.where(games <= 8, "0-8", np.where(games <= 13, "9-13", "14+"))
    band_frame = frame.assign(availability_band=bands)
    availability = [
        {"availability_band": str(band), **_simple_group_metrics(group, dispersion)}
        for band, group in band_frame.groupby("availability_band", sort=True)
    ]

    return {
        "count": int(len(frame)),
        "seasons": sorted(int(value) for value in frame["target_season"].unique()),
        "negative_log_likelihood": float(-np.sum(model_logpmf)),
        "negative_log_likelihood_per_player_season": float(-np.mean(model_logpmf)),
        "poisson_negative_log_likelihood_per_player_season": float(-np.mean(poisson_logpmf)),
        "nll_regret_vs_poisson_per_player_season": float(np.mean(poisson_logpmf - model_logpmf)),
        "bias": float(np.mean(errors)),
        "mean_absolute_error": float(np.mean(np.abs(errors))),
        "relative_bias": float(np.sum(errors) / np.sum(means)),
        "raw_baseline_bias": float(np.mean(raw_errors)),
        "raw_baseline_mean_absolute_error": float(np.mean(np.abs(raw_errors))),
        "raw_baseline_relative_bias": float(np.sum(raw_errors) / np.sum(raw_means)),
        "brier_events": brier_events,
        "predictive_interval_coverage": coverage,
        "mean_calibration_bins": mean_bins,
        "position_cohorts": positions,
        "availability_cohorts": availability,
    }


def _check(checks: list[CheckResult], name: str, passed: bool, message: str, details: dict[str, Any]) -> None:
    checks.append(CheckResult(name, bool(passed), "error", message, details))


def calibrate_historical_distributions(
    predictions: pd.DataFrame,
    historical_config: dict[str, Any],
    model_config: dict[str, Any],
) -> HistoricalCalibration:
    """Calibrate one historical NB dispersion per supported stat and validate holdout performance."""

    frame = _validate_predictions(predictions)
    calibration = model_config["historical_calibration"]
    start = int(historical_config["calibration_start_season"])
    latest = int(historical_config["latest_completed_season"])
    holdout_count = int(historical_config["holdout_seasons"])
    holdout_start = latest - holdout_count + 1
    holdout_seasons = list(range(holdout_start, latest + 1))
    training_seasons = list(range(start, holdout_start))
    window = frame.loc[frame["target_season"].between(start, latest) & frame["training_eligible"]].copy()
    checks: list[CheckResult] = [
        CheckResult("historical_calibration.no_lookahead", True, details={"rows_checked": int(len(frame))}),
        CheckResult(
            "historical_calibration.holdout_excluded_from_fit", True,
            details={"training_seasons": training_seasons, "holdout_seasons": holdout_seasons},
        ),
    ]
    holdout_bias_rows = frame.loc[
        (frame["target_season"] >= holdout_start)
        & (frame["baseline_bias_method"] == "rolling_power_poisson")
    ]
    holdout_bias_frozen = bool(
        holdout_bias_rows.empty
        or (
            (holdout_bias_rows["baseline_bias_max_observation_season"] <= holdout_start - 1).all()
            and holdout_bias_rows.groupby("stat")[[
                "baseline_bias_intercept", "baseline_bias_exponent",
                "baseline_bias_recency_half_life_seasons",
            ]]
            .nunique(dropna=False)
            .le(1)
            .to_numpy()
            .all()
        )
    )
    _check(
        checks,
        "historical_calibration.holdout_mean_correction_frozen",
        holdout_bias_frozen,
        "Baseline mean-correction parameters are fitted before and frozen across the holdout",
        {"holdout_start": holdout_start, "maximum_observation_season": holdout_start - 1},
    )
    stat_reports: dict[str, Any] = {}
    dispersion_rows: list[dict[str, Any]] = []
    minimum_training = int(historical_config["minimum_player_seasons_per_stat"])
    minimum_holdout = int(calibration["minimum_holdout_player_seasons_per_stat"])
    half_life = float(historical_config["recency_half_life_seasons"])
    base_seed = int(model_config["random_seed"])

    for stat_index, stat in enumerate(TARGET_STATS):
        stat_check_start = len(checks)
        stat_frame = window.loc[window["stat"] == stat].copy()
        negatives = stat_frame["realized_total"] < 0
        negative_count = int(negatives.sum())
        if negative_count:
            checks.append(CheckResult(
                f"historical_calibration.{stat}.signed_negative_observations",
                False, "warning", "Signed negative outcomes were excluded from the negative-binomial likelihood",
                {"excluded_rows": negative_count},
            ))
        stat_frame = stat_frame.loc[~negatives].copy()
        noninteger = ~np.isclose(stat_frame["realized_total"], np.rint(stat_frame["realized_total"]), rtol=0.0, atol=1e-9)
        _check(checks, f"historical_calibration.{stat}.integer_outcomes", not noninteger.any(), "Negative-binomial outcomes are integer valued", {"invalid_rows": int(noninteger.sum())})
        invalid_baseline = stat_frame["baseline_mean"].isna() | ~np.isfinite(stat_frame["baseline_mean"]) | (stat_frame["baseline_mean"] < 0)
        _check(checks, f"historical_calibration.{stat}.baseline_domain", not invalid_baseline.any(), "Eligible baselines are finite and nonnegative", {"invalid_rows": int(invalid_baseline.sum())})
        zero_baseline = stat_frame["baseline_mean"] == 0
        impossible_zero = zero_baseline & (stat_frame["realized_total"] > 0)
        _check(checks, f"historical_calibration.{stat}.zero_baseline_consistency", not impossible_zero.any(), "Zero baselines cannot accompany positive realized totals", {"invalid_rows": int(impossible_zero.sum())})
        zero_count = int((zero_baseline & ~impossible_zero).sum())
        stat_frame = stat_frame.loc[~noninteger & ~invalid_baseline & ~zero_baseline].copy()

        training = stat_frame.loc[stat_frame["target_season"] < holdout_start].copy()
        holdout = stat_frame.loc[stat_frame["target_season"] >= holdout_start].copy()
        _check(checks, f"historical_calibration.{stat}.training_cohort", len(training) >= minimum_training, "Pre-holdout training cohort meets the configured minimum", {"observed": int(len(training)), "minimum": minimum_training})
        _check(checks, f"historical_calibration.{stat}.holdout_cohort", len(holdout) >= minimum_holdout, "Out-of-sample holdout cohort meets the configured minimum", {"observed": int(len(holdout)), "minimum": minimum_holdout})
        bounds = tuple(float(value) for value in calibration["dispersion_bounds"][stat])
        fit: DispersionFit | None = None
        bootstrap: dict[str, Any] = {"passed": False}
        holdout_metrics: dict[str, Any] = {}
        sensitivities: list[dict[str, Any]] = []

        if len(training) >= minimum_training and len(holdout) >= minimum_holdout and not any(
            not check.passed and check.severity == "error" for check in checks[stat_check_start:]
        ):
            fit = fit_historical_dispersion(
                training["realized_total"], training["baseline_mean"], training["target_season"],
                bounds, half_life, reference_season=holdout_start - 1,
            )
            _check(checks, f"historical_calibration.{stat}.optimizer_converged", fit.converged, "Historical dispersion optimizer converged", {"iterations": fit.optimizer_iterations, "evaluations": fit.optimizer_evaluations})
            _check(checks, f"historical_calibration.{stat}.dispersion_not_on_bound", not fit.bound_hit, "Historical dispersion is strictly inside configured bounds", {"dispersion": fit.dispersion, "bounds": list(bounds)})
            if fit.converged and not fit.bound_hit:
                bootstrap = grouped_bootstrap_log_dispersion(
                    training, bounds, half_life,
                    samples=int(model_config["bootstrap_samples"]),
                    seed=base_seed + stat_index * 1_000_003,
                    confidence_level=float(calibration["bootstrap_confidence_level"]),
                    minimum_success_rate=float(calibration["minimum_bootstrap_success_rate"]),
                    reference_season=holdout_start - 1,
                )
                _check(checks, f"historical_calibration.{stat}.bootstrap", bool(bootstrap["passed"]), "Grouped bootstrap produced sufficient finite interior fits", bootstrap)
                holdout_metrics = _holdout_metrics(holdout, fit.dispersion, calibration)
                nll_limit = holdout_metrics["poisson_negative_log_likelihood_per_player_season"] + float(calibration["max_holdout_nll_regret_per_player_season"])
                _check(checks, f"historical_calibration.{stat}.holdout_likelihood", holdout_metrics["negative_log_likelihood_per_player_season"] <= nll_limit, "Holdout likelihood is within the configured Poisson-benchmark tolerance", {"observed": holdout_metrics["negative_log_likelihood_per_player_season"], "maximum": nll_limit})
                max_brier_gap = max(value["calibration_gap"] for value in holdout_metrics["brier_events"])
                _check(checks, f"historical_calibration.{stat}.holdout_brier_calibration", max_brier_gap <= float(calibration["max_brier_calibration_gap"]), "Holdout threshold-event calibration gaps are within tolerance", {"observed": max_brier_gap, "maximum": float(calibration["max_brier_calibration_gap"])})
                max_coverage_error = max(value["absolute_error"] for value in holdout_metrics["predictive_interval_coverage"])
                _check(checks, f"historical_calibration.{stat}.holdout_interval_coverage", max_coverage_error <= float(calibration["max_interval_coverage_error"]), "Holdout coverage matches the model-implied mass of discrete predictive intervals within tolerance", {"observed": max_coverage_error, "maximum": float(calibration["max_interval_coverage_error"])})
                relative_bias = abs(float(holdout_metrics["relative_bias"]))
                _check(checks, f"historical_calibration.{stat}.holdout_bias", relative_bias <= float(calibration["max_abs_relative_bias"]), "Holdout aggregate relative bias is within tolerance", {"observed": relative_bias, "maximum": float(calibration["max_abs_relative_bias"])})

                sensitivity_failed = False
                for sensitivity_start in calibration["sensitivity_start_seasons"]:
                    effective_start = max(start, int(sensitivity_start))
                    sensitivity_rows = training.loc[training["target_season"] >= effective_start]
                    sensitivity: dict[str, Any] = {
                        "configured_start_season": int(sensitivity_start),
                        "effective_start_season": effective_start,
                        "player_seasons": int(len(sensitivity_rows)),
                    }
                    if len(sensitivity_rows) < int(calibration["minimum_sensitivity_player_seasons_per_stat"]):
                        sensitivity.update({"status": "insufficient_data", "dispersion": None, "log_dispersion_delta": None})
                        sensitivity_failed = True
                    else:
                        sensitivity_fit = fit_historical_dispersion(
                            sensitivity_rows["realized_total"], sensitivity_rows["baseline_mean"], sensitivity_rows["target_season"],
                            bounds, half_life, reference_season=holdout_start - 1,
                        )
                        delta = abs(sensitivity_fit.log_dispersion - fit.log_dispersion)
                        sensitivity.update({
                            "status": "passed" if sensitivity_fit.converged and not sensitivity_fit.bound_hit else "failed",
                            "dispersion": sensitivity_fit.dispersion,
                            "log_dispersion": sensitivity_fit.log_dispersion,
                            "log_dispersion_delta": delta,
                            "converged": sensitivity_fit.converged,
                            "bound_hit": sensitivity_fit.bound_hit,
                        })
                        sensitivity_failed = sensitivity_failed or sensitivity["status"] != "passed"
                    sensitivities.append(sensitivity)
                finite_deltas = [float(value["log_dispersion_delta"]) for value in sensitivities if value.get("log_dispersion_delta") is not None]
                maximum_delta = max(finite_deltas) if finite_deltas else math.inf
                stable = not sensitivity_failed and maximum_delta <= float(calibration["max_sensitivity_log_dispersion_delta"])
                _check(checks, f"historical_calibration.{stat}.sensitivity_stability", stable, "Configured historical windows produce stable interior dispersion fits", {"maximum_log_dispersion_delta": maximum_delta, "maximum": float(calibration["max_sensitivity_log_dispersion_delta"]), "windows": sensitivities})

        stat_failed = any(not check.passed and check.severity == "error" for check in checks[stat_check_start:])
        stat_report = {
            "status": "failed" if stat_failed else "passed",
            "training_player_seasons": int(len(training)),
            "training_seasons": sorted(int(value) for value in training["target_season"].unique()),
            "holdout_player_seasons": int(len(holdout)),
            "holdout_seasons": sorted(int(value) for value in holdout["target_season"].unique()),
            "negative_outcomes_excluded": negative_count,
            "zero_baselines_excluded": zero_count,
            "fit": asdict(fit) if fit is not None else None,
            "bootstrap": bootstrap,
            "sensitivity_windows": sensitivities,
            "holdout_metrics": holdout_metrics,
        }
        stat_reports[stat] = stat_report
        if fit is not None:
            dispersion_rows.append({
                "stat": stat,
                "distribution_family": "negative_binomial",
                "calibration_version": CALIBRATION_VERSION,
                "method": "historical_only",
                "historical_dispersion": fit.dispersion,
                "historical_log_dispersion": fit.log_dispersion,
                "historical_dispersion_lower": bootstrap.get("dispersion_lower"),
                "historical_dispersion_upper": bootstrap.get("dispersion_upper"),
                "historical_log_dispersion_lower": bootstrap.get("log_dispersion_lower"),
                "historical_log_dispersion_upper": bootstrap.get("log_dispersion_upper"),
                "final_dispersion": fit.dispersion,
                "dispersion_source": "historical_only",
                "training_player_seasons": len(training),
                "holdout_player_seasons": len(holdout),
                "training_seasons": "|".join(str(value) for value in sorted(training["target_season"].unique())),
                "holdout_seasons": "|".join(str(value) for value in sorted(holdout["target_season"].unique())),
                "negative_outcomes_excluded": negative_count,
                "zero_baselines_excluded": zero_count,
                "objective_nll": fit.objective_nll,
                "optimizer_converged": fit.converged,
                "dispersion_bound_hit": fit.bound_hit,
                "bootstrap_success_rate": bootstrap.get("success_rate"),
                "holdout_nll_per_player_season": holdout_metrics.get("negative_log_likelihood_per_player_season"),
                "holdout_poisson_nll_per_player_season": holdout_metrics.get("poisson_negative_log_likelihood_per_player_season"),
                "holdout_relative_bias": holdout_metrics.get("relative_bias"),
                "holdout_raw_baseline_relative_bias": holdout_metrics.get("raw_baseline_relative_bias"),
                "status": stat_report["status"],
            })

    failed = [check for check in checks if not check.passed and check.severity == "error"]
    report = {
        "status": "failed" if failed else "passed",
        "calibration_version": CALIBRATION_VERSION,
        "distribution_family": "negative_binomial",
        "method": "historical_only",
        "input_rows": int(len(frame)),
        "eligible_rows": int(len(window)),
        "calibration_window": [start, latest],
        "training_seasons": training_seasons,
        "holdout_seasons": holdout_seasons,
        "recency_half_life_seasons": half_life,
        "opportunity_filters": dict(historical_config["prior_opportunity_filters"]),
        "baseline_bias_correction": {
            **dict(historical_config["baseline_bias_correction"]),
            "holdout_parameters": {
                stat: {
                    "intercept": float(group["baseline_bias_intercept"].iloc[0]),
                    "exponent": float(group["baseline_bias_exponent"].iloc[0]),
                    "recency_half_life_seasons": float(group["baseline_bias_recency_half_life_seasons"].iloc[0]),
                    "max_observation_season": int(group["baseline_bias_max_observation_season"].iloc[0]),
                }
                for stat, group in holdout_bias_rows.groupby("stat", sort=True)
            },
        },
        "bootstrap": {
            "samples": int(model_config["bootstrap_samples"]),
            "seed": base_seed,
            "confidence_level": float(calibration["bootstrap_confidence_level"]),
            "uncertainty_scale": "log_dispersion",
            "interpretation": "calibration uncertainty, not player forecast confidence",
        },
        "stats": stat_reports,
        "checks": [check.to_dict() for check in checks],
    }
    dispersions = pd.DataFrame.from_records(dispersion_rows)
    if not dispersions.empty:
        dispersions = dispersions.sort_values("stat").reset_index(drop=True)
    return HistoricalCalibration(dispersions=dispersions, report=report)
