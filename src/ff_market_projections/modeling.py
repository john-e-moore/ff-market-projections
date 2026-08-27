"""Current-market dispersion updates and auditable source-level mean estimation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .contracts import CheckResult
from .distributions import (
    DistributionError,
    MeanCurveFit,
    fit_negative_binomial_mean_curve,
    invert_negative_binomial_mean,
    negative_binomial_survival,
)
from .historical import TARGET_STATS


MARKET_UPDATE_VERSION = "negative_binomial_current_market_v3"
SOURCE_ESTIMATION_VERSION = "negative_binomial_source_means_v3"
_SOURCES = frozenset({"draftkings", "fanduel", "kalshi"})
_MARKET_COLUMNS = {
    "run_id", "season", "source", "source_market_id", "snapshot_utc", "raw_player_name",
    "stat", "canonical_threshold", "modeling_probability", "kalshi_bid_probability",
    "kalshi_ask_probability", "inclusion_status", "exclusion_reason", "canonical_player_id",
    "canonical_player_name", "canonical_position",
}
_DISPERSION_COLUMNS = {
    "stat", "distribution_family", "calibration_version", "historical_dispersion",
    "historical_log_dispersion", "historical_dispersion_lower", "historical_dispersion_upper",
    "historical_log_dispersion_lower", "historical_log_dispersion_upper", "status",
}


class ModelingError(ValueError):
    """A current-market model input violates a hard contract."""

    def __init__(self, message: str, validation: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.validation = validation or {"status": "failed", "checks": [], "error": message}


@dataclass(frozen=True)
class SharedDispersionFit:
    dispersion: float
    log_dispersion: float
    group_means: dict[str, float]
    objective_total: float
    data_objective: float
    prior_penalty: float
    logit_rmse: float
    logit_mae: float
    max_abs_logit_residual: float
    converged: bool
    dispersion_bound_hit: bool
    nuisance_mean_bound_hit: bool
    optimizer_status: int
    optimizer_message: str
    optimizer_iterations: int
    optimizer_evaluations: int


@dataclass(frozen=True)
class MeanEstimationResult:
    dispersions: pd.DataFrame
    source_projections: pd.DataFrame
    priced_markets: pd.DataFrame
    historical_report: dict[str, Any]
    validation: dict[str, Any]


def _check(
    checks: list[CheckResult],
    name: str,
    passed: bool,
    message: str,
    *,
    severity: str = "error",
    **details: Any,
) -> None:
    checks.append(CheckResult(name, bool(passed), severity, message, details))


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value))


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), np.finfo(float).tiny, 1.0 - np.finfo(float).eps)
    return np.log(clipped) - np.log1p(-clipped)


def _soft_l1_objective(residuals: np.ndarray) -> float:
    return float(np.sum(np.sqrt(1.0 + np.asarray(residuals, dtype=float) ** 2) - 1.0))


def _bound_hit(value: float, lower: float, upper: float) -> bool:
    span = upper - lower
    tolerance = max(1e-8, span * 1e-7)
    return bool(value - lower <= tolerance or upper - value <= tolerance)


def _coerce_inputs(
    priced_markets: pd.DataFrame,
    dispersions: pd.DataFrame,
    historical_report: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_markets = sorted(_MARKET_COLUMNS - set(priced_markets.columns))
    if missing_markets:
        raise ModelingError(f"Priced markets are missing columns: {', '.join(missing_markets)}")
    missing_dispersions = sorted(_DISPERSION_COLUMNS - set(dispersions.columns))
    if missing_dispersions:
        raise ModelingError(f"Dispersion calibration is missing columns: {', '.join(missing_dispersions)}")
    if historical_report.get("status") != "passed":
        raise ModelingError("Historical calibration must pass before current-market estimation")

    markets = priced_markets.copy()
    numeric_market_columns = (
        "canonical_threshold", "modeling_probability", "kalshi_bid_probability", "kalshi_ask_probability",
    )
    for column in numeric_market_columns:
        markets[column] = pd.to_numeric(markets[column], errors="coerce")
    if markets.empty:
        raise ModelingError("Priced markets must not be empty")
    required_values = ["run_id", "season", "source", "source_market_id", "stat"]
    missing_values = markets[required_values].isna().any(axis=1)
    for column in required_values:
        missing_values |= markets[column].astype(str).str.strip().eq("")
    if missing_values.any():
        raise ModelingError(f"Priced markets contain {int(missing_values.sum())} rows with missing key values")
    duplicate_market_ids = markets.duplicated(["source", "source_market_id"], keep=False)
    if duplicate_market_ids.any():
        raise ModelingError(f"Priced market source IDs are not unique on {int(duplicate_market_ids.sum())} rows")
    unknown_sources = sorted(set(markets["source"].dropna().astype(str)) - _SOURCES)
    unknown_stats = sorted(set(markets["stat"].dropna().astype(str)) - set(TARGET_STATS))
    if unknown_sources or unknown_stats:
        raise ModelingError(
            "Priced markets contain unsupported source or stat values",
            {"status": "failed", "checks": [], "unknown_sources": unknown_sources, "unknown_stats": unknown_stats},
        )
    identity_missing = markets[["canonical_player_id", "canonical_player_name"]].isna().any(axis=1)
    identity_missing |= markets["canonical_player_id"].astype(str).str.strip().eq("")
    identity_missing |= markets["canonical_player_name"].astype(str).str.strip().eq("")
    if identity_missing.any():
        raise ModelingError(f"Priced markets contain {int(identity_missing.sum())} rows without canonical identity")

    dispersion_frame = dispersions.copy()
    numeric_dispersion_columns = (
        "historical_dispersion", "historical_log_dispersion", "historical_dispersion_lower",
        "historical_dispersion_upper", "historical_log_dispersion_lower", "historical_log_dispersion_upper",
    )
    for column in numeric_dispersion_columns:
        dispersion_frame[column] = pd.to_numeric(dispersion_frame[column], errors="coerce")
    duplicate_stats = dispersion_frame.duplicated("stat", keep=False)
    observed_stats = set(dispersion_frame["stat"].dropna().astype(str))
    if duplicate_stats.any() or observed_stats != set(TARGET_STATS):
        raise ModelingError(
            "Dispersion calibration must contain exactly one row for every supported stat",
            {
                "status": "failed", "checks": [], "duplicate_rows": int(duplicate_stats.sum()),
                "missing_stats": sorted(set(TARGET_STATS) - observed_stats),
                "unknown_stats": sorted(observed_stats - set(TARGET_STATS)),
            },
        )
    invalid_dispersion = (
        dispersion_frame[list(numeric_dispersion_columns)].isna().any(axis=1)
        | (dispersion_frame["historical_dispersion"] <= 0)
        | (dispersion_frame["historical_dispersion_lower"] <= 0)
        | (dispersion_frame["historical_dispersion_upper"] <= dispersion_frame["historical_dispersion_lower"])
        | dispersion_frame["status"].astype(str).ne("passed")
        | dispersion_frame["distribution_family"].astype(str).ne("negative_binomial")
    )
    if invalid_dispersion.any():
        raise ModelingError(f"Dispersion calibration contains {int(invalid_dispersion.sum())} invalid or failed rows")
    versions = set(dispersion_frame["calibration_version"].astype(str))
    if versions != {_text(historical_report.get("calibration_version"))}:
        raise ModelingError("Dispersion CSV and historical calibration version do not match")
    return markets, dispersion_frame


def _annotate_model_eligibility(markets: pd.DataFrame, model_config: dict[str, Any]) -> pd.DataFrame:
    frame = markets.copy()
    statuses: list[str] = []
    reasons: list[str] = []
    floor, ceiling = float(model_config["probability_floor"]), float(model_config["probability_ceiling"])
    for _, row in frame.iterrows():
        pricing_status = _text(row.get("inclusion_status"))
        if pricing_status != "included":
            statuses.append("excluded")
            reasons.append(_text(row.get("exclusion_reason")) or "excluded_by_pricing")
            continue
        threshold, probability = row.get("canonical_threshold"), row.get("modeling_probability")
        if not _finite(threshold) or float(threshold) < 1 or not float(threshold).is_integer():
            statuses.append("excluded")
            reasons.append("invalid_model_threshold")
        elif not _finite(probability) or not 0 < float(probability) < 1:
            statuses.append("excluded")
            reasons.append("invalid_model_probability")
        elif not floor <= float(probability) <= ceiling:
            statuses.append("excluded")
            reasons.append("probability_outside_model_range")
        else:
            statuses.append("included")
            reasons.append("")
    frame["model_inclusion_status"] = statuses
    frame["model_exclusion_reason"] = reasons
    frame["model_calibration_group_eligible"] = False
    for column in (
        "modeled_probability", "probability_residual", "logit_probability_residual",
        "source_projection_mean", "source_projection_method", "final_dispersion",
        "dispersion_source", "calibration_version", "market_update_version",
        "source_projection_quality_status", "source_projection_exclusion_reason",
    ):
        frame[column] = None
    return frame


def fit_shared_kalshi_dispersion(
    groups: dict[str, tuple[np.ndarray, np.ndarray]],
    historical_dispersion: float,
    dispersion_bounds: tuple[float, float],
    mean_bounds: tuple[float, float],
    *,
    robust_loss: str,
    optimizer_tolerance: float,
    max_evaluations: int,
    prior_log_sd: float | None,
) -> SharedDispersionFit:
    """Jointly fit nuisance player means and a shared log dispersion.

    The public name is retained for compatibility, but groups may contain any
    eligible current-market quotes. ``prior_log_sd=None`` yields the market-only
    fit. A positive value adds the Gaussian historical prior penalty used for the
    empirical-Bayes/MAP update.
    """

    if not groups:
        raise DistributionError("at least one Kalshi calibration group is required")
    if robust_loss != "soft_l1":
        raise DistributionError("robust_loss must be soft_l1")
    if prior_log_sd is not None and (not _finite(prior_log_sd) or float(prior_log_sd) <= 0):
        raise DistributionError("prior_log_sd must be finite and positive when supplied")
    dispersion_lower, dispersion_upper = (float(value) for value in dispersion_bounds)
    mean_lower, mean_upper = (float(value) for value in mean_bounds)
    if not 0 < dispersion_lower < dispersion_upper or not 0 < mean_lower < mean_upper:
        raise DistributionError("dispersion and mean bounds must be positive and increasing")
    if not _finite(historical_dispersion) or historical_dispersion <= 0:
        raise DistributionError("historical_dispersion must be finite and positive")

    keys = sorted(groups)
    thresholds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    initial_means: list[float] = []
    for key in keys:
        group_thresholds = np.asarray(groups[key][0], dtype=float)
        group_probabilities = np.asarray(groups[key][1], dtype=float)
        fit = fit_negative_binomial_mean_curve(
            group_thresholds, group_probabilities, historical_dispersion, (mean_lower, mean_upper),
            robust_loss=robust_loss, optimizer_tolerance=optimizer_tolerance,
            max_evaluations=max_evaluations,
        )
        thresholds.append(np.rint(group_thresholds).astype(int))
        targets.append(group_probabilities)
        initial_means.append(fit.mean)

    historical_log_dispersion = math.log(float(historical_dispersion))
    lower = np.asarray([math.log(dispersion_lower), *([math.log(mean_lower)] * len(keys))])
    upper = np.asarray([math.log(dispersion_upper), *([math.log(mean_upper)] * len(keys))])
    initial = np.asarray([np.clip(historical_log_dispersion, lower[0], upper[0]), *(math.log(value) for value in initial_means)])

    def data_residuals(parameters: np.ndarray) -> np.ndarray:
        dispersion = math.exp(float(parameters[0]))
        values: list[np.ndarray] = []
        for index, (group_thresholds, group_targets) in enumerate(zip(thresholds, targets, strict=True), start=1):
            modeled = negative_binomial_survival(group_thresholds, math.exp(float(parameters[index])), dispersion)
            values.append(_logit(modeled) - _logit(group_targets))
        return np.concatenate(values)

    def objective(parameters: np.ndarray) -> float:
        residuals = data_residuals(parameters)
        data_objective = _soft_l1_objective(residuals)
        if prior_log_sd is None:
            return data_objective
        prior_residual = (float(parameters[0]) - historical_log_dispersion) / float(prior_log_sd)
        return data_objective + 0.5 * prior_residual**2

    optimized = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=list(zip(lower, upper, strict=True)),
        options={
            "ftol": float(optimizer_tolerance), "gtol": float(optimizer_tolerance),
            "maxiter": int(max_evaluations), "maxfun": int(max_evaluations), "maxls": 50,
        },
    )
    parameters = np.asarray(optimized.x, dtype=float)
    residuals = data_residuals(parameters)
    data_objective = _soft_l1_objective(residuals)
    prior_penalty = 0.0
    if prior_log_sd is not None:
        prior_penalty = 0.5 * ((float(parameters[0]) - historical_log_dispersion) / float(prior_log_sd)) ** 2
    return SharedDispersionFit(
        dispersion=float(math.exp(float(parameters[0]))),
        log_dispersion=float(parameters[0]),
        group_means={key: float(math.exp(float(parameters[index]))) for index, key in enumerate(keys, start=1)},
        objective_total=float(data_objective + prior_penalty),
        data_objective=float(data_objective),
        prior_penalty=float(prior_penalty),
        logit_rmse=float(np.sqrt(np.mean(residuals**2))),
        logit_mae=float(np.mean(np.abs(residuals))),
        max_abs_logit_residual=float(np.max(np.abs(residuals))),
        converged=bool(optimized.success and np.isfinite(optimized.fun)),
        dispersion_bound_hit=_bound_hit(float(parameters[0]), float(lower[0]), float(upper[0])),
        nuisance_mean_bound_hit=any(
            _bound_hit(float(parameters[index]), float(lower[index]), float(upper[index]))
            for index in range(1, len(parameters))
        ),
        optimizer_status=int(optimized.status),
        optimizer_message=str(optimized.message),
        optimizer_iterations=int(getattr(optimized, "nit", 0)),
        optimizer_evaluations=int(getattr(optimized, "nfev", 0)),
    )


def _prior_log_sd(row: pd.Series, confidence_level: float) -> float:
    lower = float(row["historical_log_dispersion_lower"])
    upper = float(row["historical_log_dispersion_upper"])
    quantile = NormalDist().inv_cdf((1.0 + float(confidence_level)) / 2.0)
    standard_deviation = (upper - lower) / (2.0 * quantile)
    if not math.isfinite(standard_deviation) or standard_deviation <= 0:
        raise DistributionError("historical bootstrap bounds do not imply a positive finite log-dispersion variance")
    return standard_deviation


def _leave_one_threshold_out(
    thresholds: np.ndarray,
    probabilities: np.ndarray,
    dispersion: float,
    mean_bounds: tuple[float, float],
    current_config: dict[str, Any],
    robust_loss: str,
) -> dict[str, Any]:
    if len(thresholds) < 2:
        return {"count": 0, "failed_fits": 0, "logit_mae": None, "logit_rmse": None}
    residuals: list[float] = []
    failures = 0
    for held_out in range(len(thresholds)):
        keep = np.arange(len(thresholds)) != held_out
        try:
            fit = fit_negative_binomial_mean_curve(
                thresholds[keep], probabilities[keep], dispersion, mean_bounds,
                robust_loss=robust_loss,
                optimizer_tolerance=float(current_config["optimizer_tolerance"]),
                max_evaluations=int(current_config["optimizer_max_evaluations"]),
            )
        except DistributionError:
            failures += 1
            continue
        if not fit.converged or fit.bound_hit:
            failures += 1
            continue
        predicted = float(negative_binomial_survival(int(thresholds[held_out]), fit.mean, dispersion))
        residuals.append(float(_logit(np.asarray([predicted]))[0] - _logit(np.asarray([probabilities[held_out]]))[0]))
    values = np.asarray(residuals, dtype=float)
    return {
        "count": int(len(values)),
        "failed_fits": int(failures),
        "logit_mae": float(np.mean(np.abs(values))) if len(values) else None,
        "logit_rmse": float(np.sqrt(np.mean(values**2))) if len(values) else None,
    }


def _shared_curve_holdout(
    groups: dict[str, tuple[np.ndarray, np.ndarray]],
    dispersion: float,
    mean_bounds: tuple[float, float],
    current_config: dict[str, Any],
    robust_loss: str,
) -> dict[str, Any]:
    values: list[float] = []
    failures = 0
    count = 0
    for thresholds, probabilities in groups.values():
        group = _leave_one_threshold_out(
            thresholds, probabilities, dispersion, mean_bounds, current_config, robust_loss,
        )
        failures += int(group["failed_fits"])
        if group["logit_mae"] is not None:
            values.extend([float(group["logit_mae"])] * int(group["count"]))
            count += int(group["count"])
    array = np.asarray(values, dtype=float)
    return {
        "count": count,
        "failed_fits": failures,
        "logit_mae": float(np.mean(array)) if len(array) else None,
        "logit_rmse": float(np.sqrt(np.mean(array**2))) if len(array) else None,
    }


def _update_dispersions(
    markets: pd.DataFrame,
    dispersions: pd.DataFrame,
    historical_report: dict[str, Any],
    model_config: dict[str, Any],
    checks: list[CheckResult],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = dispersions.copy().set_index("stat", drop=False)
    current = model_config["current_market"]
    calibration = model_config["historical_calibration"]
    update_reports: dict[str, Any] = {}
    eligible_market = markets.loc[markets["model_inclusion_status"] == "included"].copy()

    duplicate_thresholds = eligible_market.duplicated(
        ["run_id", "season", "source", "canonical_player_id", "stat", "canonical_threshold"],
        keep=False,
    )
    _check(
        checks, "model.current_market_thresholds_unique_within_source", not duplicate_thresholds.any(),
        "Eligible current-market curves contain at most one quote per source/player/stat threshold",
        duplicate_rows=int(duplicate_thresholds.sum()),
    )

    for stat in TARGET_STATS:
        row = output.loc[stat]
        historical_dispersion = float(row["historical_dispersion"])
        stat_quotes = eligible_market.loc[eligible_market["stat"] == stat]
        qualified: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        qualified_indices: dict[str, list[int]] = {}
        for player_id, group in stat_quotes.groupby("canonical_player_id", sort=True):
            ordered = group.sort_values(["canonical_threshold", "source", "source_market_id"])
            if ordered["canonical_threshold"].nunique() >= int(model_config["minimum_thresholds_per_group"]):
                key = str(player_id)
                qualified[key] = (
                    ordered["canonical_threshold"].to_numpy(dtype=int),
                    ordered["modeling_probability"].to_numpy(dtype=float),
                )
                qualified_indices[key] = [int(value) for value in ordered.index]

        group_count = len(qualified)
        initial_group_count = group_count
        quote_count = int(sum(len(value[0]) for value in qualified.values()))
        enough_groups = group_count >= int(model_config["minimum_calibration_groups"])
        update_requested = model_config["dispersion_mode"] == "historical_with_current_market_update"
        method = "historical_only"
        fallback_reason = "dispersion_mode_historical_only" if not update_requested else "insufficient_eligible_current_market_groups"
        market_only: SharedDispersionFit | None = None
        map_fit: SharedDispersionFit | None = None
        prior_sd: float | None = None
        market_holdout = {"count": 0, "failed_fits": 0, "logit_mae": None, "logit_rmse": None}
        map_holdout = {"count": 0, "failed_fits": 0, "logit_mae": None, "logit_rmse": None}
        market_validation_passed = False
        map_validation_passed = False
        quarantined_groups: dict[str, list[str]] = {}

        if update_requested and enough_groups and not duplicate_thresholds.any():
            try:
                prior_sd = _prior_log_sd(row, float(calibration["bootstrap_confidence_level"]))
                dispersion_bounds = tuple(float(value) for value in calibration["dispersion_bounds"][stat])
                mean_bounds = tuple(float(value) for value in current["mean_bounds"][stat])
                shared_kwargs = {
                    "historical_dispersion": historical_dispersion,
                    "dispersion_bounds": dispersion_bounds,
                    "mean_bounds": mean_bounds,
                    "robust_loss": str(model_config["robust_loss"]),
                    "optimizer_tolerance": float(current["optimizer_tolerance"]),
                    "max_evaluations": int(current["optimizer_max_evaluations"]),
                }
                market_only = fit_shared_kalshi_dispersion(qualified, prior_log_sd=None, **shared_kwargs)
                for group_id, (thresholds, probabilities) in qualified.items():
                    reasons: list[str] = []
                    curve = _fit_curve(thresholds, probabilities, market_only.dispersion, mean_bounds, model_config)
                    group_holdout = _leave_one_threshold_out(
                        thresholds, probabilities, market_only.dispersion, mean_bounds, current,
                        str(model_config["robust_loss"]),
                    )
                    if not curve.converged or curve.bound_hit or curve.logit_rmse > float(current["max_current_market_logit_rmse"]):
                        reasons.append("curve_residual_exceeds_limit")
                    if (
                        group_holdout["failed_fits"] > 0 or group_holdout["logit_mae"] is None
                        or float(group_holdout["logit_mae"]) > float(current["max_current_market_holdout_logit_mae"])
                    ):
                        reasons.append("threshold_holdout_exceeds_limit")
                    if reasons:
                        quarantined_groups[group_id] = reasons
                retained = {
                    group_id: values for group_id, values in qualified.items()
                    if group_id not in quarantined_groups
                }
                if quarantined_groups and len(retained) >= int(model_config["minimum_calibration_groups"]):
                    qualified = retained
                    group_count = len(qualified)
                    quote_count = int(sum(len(value[0]) for value in qualified.values()))
                    market_only = fit_shared_kalshi_dispersion(qualified, prior_log_sd=None, **shared_kwargs)
                map_fit = fit_shared_kalshi_dispersion(qualified, prior_log_sd=prior_sd, **shared_kwargs)
            except DistributionError as exc:
                _check(checks, f"model.{stat}.current_market_update_inputs", False, str(exc))
                fallback_reason = "current_market_update_failed"
            if market_only is not None and map_fit is not None:
                _check(checks, f"model.{stat}.current_market_only_optimizer", market_only.converged, "Current-market-only shared-dispersion optimizer converged", **asdict(market_only))
                _check(checks, f"model.{stat}.current_market_map_optimizer", map_fit.converged, "Current-market MAP shared-dispersion optimizer converged", **asdict(map_fit))
                _check(
                    checks, f"model.{stat}.current_market_dispersion_bounds",
                    not market_only.dispersion_bound_hit and not map_fit.dispersion_bound_hit,
                    "Current-market-only and MAP dispersions are strictly inside configured bounds",
                    market_only_bound_hit=market_only.dispersion_bound_hit,
                    map_bound_hit=map_fit.dispersion_bound_hit,
                )
                _check(
                    checks, f"model.{stat}.current_market_nuisance_mean_bounds",
                    not market_only.nuisance_mean_bound_hit and not map_fit.nuisance_mean_bound_hit,
                    "Current-market nuisance player means are strictly inside configured bounds",
                    market_only_bound_hit=market_only.nuisance_mean_bound_hit,
                    map_bound_hit=map_fit.nuisance_mean_bound_hit,
                )
                market_holdout = _shared_curve_holdout(
                    qualified, market_only.dispersion, mean_bounds, current, str(model_config["robust_loss"]),
                )
                map_holdout = _shared_curve_holdout(
                    qualified, map_fit.dispersion, mean_bounds, current, str(model_config["robust_loss"]),
                )
                market_residual_passed = market_only.logit_rmse <= float(current["max_current_market_logit_rmse"])
                market_holdout_passed = (
                    market_holdout["failed_fits"] == 0 and market_holdout["logit_mae"] is not None
                    and float(market_holdout["logit_mae"]) <= float(current["max_current_market_holdout_logit_mae"])
                )
                market_validation_passed = market_residual_passed and market_holdout_passed
                map_residual_passed = map_fit.logit_rmse <= float(current["max_current_market_logit_rmse"])
                map_holdout_passed = (
                    map_holdout["failed_fits"] == 0 and map_holdout["logit_mae"] is not None
                    and float(map_holdout["logit_mae"]) <= float(current["max_current_market_holdout_logit_mae"])
                )
                map_validation_passed = map_residual_passed and map_holdout_passed
                update_validation_severity = (
                    "error" if current["current_market_conflict_policy"] == "fail" else "warning"
                )
                _check(
                    checks, f"model.{stat}.current_market_only_curve_residual",
                    market_residual_passed,
                    "Current-market-only curve logit RMSE is within tolerance",
                    observed=market_only.logit_rmse, maximum=float(current["max_current_market_logit_rmse"]),
                )
                _check(
                    checks, f"model.{stat}.current_market_only_threshold_holdout", market_holdout_passed,
                    "Leave-one-threshold-out current-market-only logit MAE is within tolerance",
                    observed=market_holdout["logit_mae"], maximum=float(current["max_current_market_holdout_logit_mae"]),
                    failed_fits=market_holdout["failed_fits"], predictions=market_holdout["count"],
                )
                _check(
                    checks, f"model.{stat}.current_market_map_curve_residual",
                    map_residual_passed,
                    "Current-market MAP curve logit RMSE is within tolerance",
                    severity=update_validation_severity,
                    observed=map_fit.logit_rmse, maximum=float(current["max_current_market_logit_rmse"]),
                )
                _check(
                    checks, f"model.{stat}.current_market_map_threshold_holdout", map_holdout_passed,
                    "Leave-one-threshold-out current-market MAP logit MAE is within tolerance",
                    severity=update_validation_severity,
                    observed=map_holdout["logit_mae"], maximum=float(current["max_current_market_holdout_logit_mae"]),
                    failed_fits=map_holdout["failed_fits"], predictions=map_holdout["count"],
                )
                delta = abs(market_only.log_dispersion - math.log(historical_dispersion))
                conflict_passed = delta <= float(current["max_current_market_log_dispersion_delta"])
                _check(
                    checks, f"model.{stat}.current_market_historical_conflict", conflict_passed,
                    "Current-market-only and historical log dispersions are within the configured conflict threshold",
                    severity="error" if current["current_market_conflict_policy"] == "fail" else "warning",
                    observed=delta, maximum=float(current["max_current_market_log_dispersion_delta"]),
                    historical_dispersion=historical_dispersion, current_market_only_dispersion=market_only.dispersion,
                )
                optimizer_valid = (
                    market_only.converged and map_fit.converged
                    and not market_only.dispersion_bound_hit and not map_fit.dispersion_bound_hit
                    and not market_only.nuisance_mean_bound_hit and not map_fit.nuisance_mean_bound_hit
                )
                if optimizer_valid and map_validation_passed and conflict_passed:
                    method = "historical_plus_current_market_map"
                    fallback_reason = ""
                elif (
                    optimizer_valid and market_validation_passed
                    and current["current_market_conflict_policy"] == "warning"
                ):
                    method = "current_market_conflict_override"
                    fallback_reason = "historical_map_misspecification_current_market_override"
                elif optimizer_valid and not market_validation_passed:
                    fallback_reason = "current_market_validation_failed"
                elif optimizer_valid and current["current_market_conflict_policy"] == "fail":
                    fallback_reason = "current_market_historical_conflict"
                else:
                    fallback_reason = "current_market_update_optimizer_failed"
        elif update_requested:
            _check(
                checks, f"model.{stat}.current_market_calibration_groups", enough_groups,
                "Eligible multi-threshold current-market groups meet the configured update minimum",
                severity="warning", observed=group_count,
                minimum=int(model_config["minimum_calibration_groups"]),
                minimum_thresholds_per_group=int(model_config["minimum_thresholds_per_group"]),
            )

        retained_indices = [
            index for group_id in qualified for index in qualified_indices[group_id]
        ]
        if retained_indices:
            markets.loc[retained_indices, "model_calibration_group_eligible"] = True

        if method == "historical_plus_current_market_map" and map_fit is not None:
            final_dispersion = map_fit.dispersion
        elif method == "current_market_conflict_override" and market_only is not None:
            final_dispersion = market_only.dispersion
        else:
            final_dispersion = historical_dispersion
        output.loc[stat, "method"] = method
        output.loc[stat, "dispersion_source"] = method
        output.loc[stat, "final_dispersion"] = final_dispersion
        output.loc[stat, "market_update_version"] = MARKET_UPDATE_VERSION
        output.loc[stat, "current_market_only_dispersion"] = market_only.dispersion if market_only else None
        output.loc[stat, "current_market_only_log_dispersion"] = market_only.log_dispersion if market_only else None
        output.loc[stat, "current_market_map_dispersion"] = map_fit.dispersion if map_fit else None
        output.loc[stat, "current_market_map_log_dispersion"] = map_fit.log_dispersion if map_fit else None
        output.loc[stat, "current_market_group_count"] = group_count
        output.loc[stat, "current_market_quote_count"] = quote_count
        output.loc[stat, "current_market_prior_log_sd"] = prior_sd
        output.loc[stat, "current_market_prior_log_variance"] = prior_sd**2 if prior_sd is not None else None
        output.loc[stat, "current_market_only_objective"] = market_only.objective_total if market_only else None
        output.loc[stat, "current_market_map_objective"] = map_fit.objective_total if map_fit else None
        output.loc[stat, "current_market_only_logit_rmse"] = market_only.logit_rmse if market_only else None
        output.loc[stat, "current_market_map_logit_rmse"] = map_fit.logit_rmse if map_fit else None
        output.loc[stat, "current_market_only_holdout_logit_mae"] = market_holdout.get("logit_mae")
        output.loc[stat, "current_market_map_holdout_logit_mae"] = map_holdout.get("logit_mae")
        output.loc[stat, "current_market_optimizer_converged"] = map_fit.converged if map_fit else None
        output.loc[stat, "current_market_dispersion_bound_hit"] = map_fit.dispersion_bound_hit if map_fit else None
        output.loc[stat, "current_market_nuisance_mean_bound_hit"] = map_fit.nuisance_mean_bound_hit if map_fit else None
        output.loc[stat, "current_market_only_validation_passed"] = market_validation_passed if market_only else None
        output.loc[stat, "current_market_map_validation_passed"] = map_validation_passed if map_fit else None
        output.loc[stat, "fallback_reason"] = fallback_reason

        update_reports[stat] = {
            "method": method,
            "fallback_reason": fallback_reason,
            "historical_dispersion": historical_dispersion,
            "current_market_only_fit": asdict(market_only) if market_only else None,
            "current_market_map_fit": asdict(map_fit) if map_fit else None,
            "final_dispersion": final_dispersion,
            "eligible_groups": group_count,
            "initial_eligible_groups": initial_group_count,
            "eligible_quotes": quote_count,
            "quarantined_groups": quarantined_groups,
            "minimum_groups": int(model_config["minimum_calibration_groups"]),
            "minimum_thresholds_per_group": int(model_config["minimum_thresholds_per_group"]),
            "prior_log_sd": prior_sd,
            "current_market_only_validation_passed": market_validation_passed if market_only else None,
            "current_market_map_validation_passed": map_validation_passed if map_fit else None,
            "current_market_only_threshold_holdout": market_holdout,
            "current_market_map_threshold_holdout": map_holdout,
        }
    return output.reset_index(drop=True), update_reports


def _fit_curve(
    thresholds: np.ndarray,
    probabilities: np.ndarray,
    dispersion: float,
    mean_bounds: tuple[float, float],
    model_config: dict[str, Any],
) -> MeanCurveFit:
    current = model_config["current_market"]
    return fit_negative_binomial_mean_curve(
        thresholds, probabilities, dispersion, mean_bounds,
        robust_loss=str(model_config["robust_loss"]),
        optimizer_tolerance=float(current["optimizer_tolerance"]),
        max_evaluations=int(current["optimizer_max_evaluations"]),
    )


def _sensitivity_bounds(
    source: str,
    thresholds: np.ndarray,
    probabilities: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    dispersion_row: pd.Series,
    mean_bounds: tuple[float, float],
    model_config: dict[str, Any],
    point_mean: float,
) -> tuple[float | None, float | None, int]:
    current = model_config["current_market"]
    estimates = [float(point_mean)]
    failures = 0
    dispersions = [
        float(dispersion_row["historical_dispersion_lower"]),
        float(dispersion_row["historical_dispersion_upper"]),
    ]
    probability_sets = [("midpoint", probabilities)]
    if source == "kalshi" and np.isfinite(bids).all() and np.isfinite(asks).all():
        probability_sets.extend((("bid", bids), ("ask", asks)))
    for dispersion in dispersions:
        for _, values in probability_sets:
            try:
                if source in {"draftkings", "fanduel"}:
                    fit = invert_negative_binomial_mean(
                        int(thresholds[0]), float(values[0]), dispersion, mean_bounds,
                        probability_tolerance=float(current["optimizer_tolerance"]),
                    )
                    if fit.bound_hit:
                        failures += 1
                    else:
                        estimates.append(fit.mean)
                else:
                    fit = _fit_curve(thresholds, values, dispersion, mean_bounds, model_config)
                    if not fit.converged or fit.bound_hit:
                        failures += 1
                    else:
                        estimates.append(fit.mean)
            except DistributionError:
                failures += 1
    if failures or len(estimates) < (3 if source in {"draftkings", "fanduel"} else 7):
        return None, None, failures
    return float(min(estimates)), float(max(estimates)), failures


def _projection_base(group: pd.DataFrame) -> dict[str, Any]:
    first = group.iloc[0]
    return {
        "run_id": _text(first["run_id"]),
        "season": _text(first["season"]),
        "canonical_player_id": _text(first["canonical_player_id"]),
        "canonical_player_name": _text(first["canonical_player_name"]),
        "display_name": _text(first["canonical_player_name"]),
        "canonical_position": _text(first.get("canonical_position")),
        "source": _text(first["source"]),
        "stat": _text(first["stat"]),
        "source_snapshot_time": "|".join(sorted({_text(value) for value in group["snapshot_utc"] if _text(value)})),
        "raw_player_names": "|".join(sorted({_text(value) for value in group["raw_player_name"] if _text(value)})),
        "quote_count": int(len(group)),
        "source_market_ids": "|".join(sorted({_text(value) for value in group["source_market_id"] if _text(value)})),
    }


def _estimate_source_projections(
    markets: pd.DataFrame,
    dispersions: pd.DataFrame,
    model_config: dict[str, Any],
    checks: list[CheckResult],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = markets.copy()
    dispersion_lookup = dispersions.set_index("stat")
    current = model_config["current_market"]
    projection_rows: list[dict[str, Any]] = []
    group_columns = ["run_id", "season", "canonical_player_id", "stat", "source"]
    sportsbook_residuals: list[float] = []
    kalshi_residuals: list[float] = []
    kalshi_holdouts: list[float] = []
    quarantined_kalshi_curves: list[dict[str, Any]] = []
    solvers_passed = True
    sensitivities_passed = True

    for _, group in frame.groupby(group_columns, sort=True, dropna=False):
        base = _projection_base(group)
        source, stat = base["source"], base["stat"]
        dispersion_row = dispersion_lookup.loc[stat]
        final_dispersion = float(dispersion_row["final_dispersion"])
        mean_bounds = tuple(float(value) for value in current["mean_bounds"][stat])
        eligible = group.loc[group["model_inclusion_status"] == "included"].sort_values(
            ["canonical_threshold", "source_market_id"]
        )
        excluded_reasons = sorted({_text(value) for value in group["model_exclusion_reason"] if _text(value)})
        common = {
            **base,
            "distribution_family": "negative_binomial",
            "fitted_dispersion": final_dispersion,
            "dispersion_source": _text(dispersion_row["dispersion_source"]),
            "calibration_version": _text(dispersion_row["calibration_version"]),
            "market_update_version": MARKET_UPDATE_VERSION,
            "source_estimation_version": SOURCE_ESTIMATION_VERSION,
            "sensitivity_label": "model_sensitivity_not_confidence_interval",
            "eligible_quote_count": int(len(eligible)),
        }
        if eligible.empty:
            projection_rows.append({
                **common, "mean": None, "sensitivity_low": None, "sensitivity_high": None,
                "projection_method": "unavailable", "thresholds_used": "", "fit_objective": None,
                "fit_error": None, "logit_rmse": None, "max_abs_logit_residual": None,
                "holdout_logit_mae": None, "holdout_failed_fits": None,
                "optimizer_converged": None, "mean_bound_hit": None,
                "quality_status": "excluded", "exclusion_reason": "|".join(excluded_reasons) or "no_eligible_quotes",
            })
            continue

        thresholds = eligible["canonical_threshold"].to_numpy(dtype=int)
        probabilities = eligible["modeling_probability"].to_numpy(dtype=float)
        bids = eligible["kalshi_bid_probability"].to_numpy(dtype=float)
        asks = eligible["kalshi_ask_probability"].to_numpy(dtype=float)
        fit_error: float | None = None
        objective: float | None = None
        logit_rmse: float | None = None
        max_logit_residual: float | None = None
        holdout_mae: float | None = None
        holdout_failed_fits = 0
        modeled: tuple[float, ...] = ()
        residuals: tuple[float, ...] = ()
        logit_residuals: tuple[float, ...] = ()
        mean: float | None = None
        converged = False
        bound_hit = False
        projection_method = ""
        exclusion_reason = ""

        try:
            if source in {"draftkings", "fanduel"}:
                if len(eligible) != 1:
                    raise DistributionError("sportsbook source/player/stat groups must contain exactly one eligible Over quote")
                inversion = invert_negative_binomial_mean(
                    int(thresholds[0]), float(probabilities[0]), final_dispersion, mean_bounds,
                    probability_tolerance=float(current["optimizer_tolerance"]),
                )
                mean = inversion.mean
                modeled = (inversion.modeled_probability,)
                residuals = (inversion.probability_residual,)
                logit_residuals = tuple(_logit(np.asarray(modeled)) - _logit(probabilities))
                fit_error = abs(inversion.probability_residual)
                objective = 0.5 * float(logit_residuals[0] ** 2)
                logit_rmse = abs(float(logit_residuals[0]))
                max_logit_residual = logit_rmse
                converged, bound_hit = inversion.converged, inversion.bound_hit
                projection_method = "single_threshold_bounded_inversion"
                sportsbook_residuals.append(fit_error)
            else:
                if len(np.unique(thresholds)) != len(thresholds):
                    raise DistributionError("Kalshi source/player/stat curves must contain distinct eligible thresholds")
                curve = _fit_curve(thresholds, probabilities, final_dispersion, mean_bounds, model_config)
                mean = curve.mean
                modeled = curve.modeled_probabilities
                residuals = curve.probability_residuals
                logit_residuals = curve.logit_residuals
                fit_error = curve.logit_rmse
                objective = curve.objective_cost
                logit_rmse = curve.logit_rmse
                max_logit_residual = curve.max_abs_logit_residual
                converged, bound_hit = curve.converged, curve.bound_hit
                projection_method = "robust_multi_threshold_logit_fit"
                holdout = _leave_one_threshold_out(
                    thresholds, probabilities, final_dispersion, mean_bounds, current,
                    str(model_config["robust_loss"]),
                )
                holdout_mae = holdout["logit_mae"]
                holdout_failed_fits = int(holdout["failed_fits"])
        except DistributionError as exc:
            exclusion_reason = f"mean_fit_failed:{exc}"

        sensitivity_low: float | None = None
        sensitivity_high: float | None = None
        sensitivity_failures = 0
        if mean is not None and converged and not bound_hit:
            sensitivity_low, sensitivity_high, sensitivity_failures = _sensitivity_bounds(
                source, thresholds, probabilities, bids, asks, dispersion_row,
                mean_bounds, model_config, mean,
            )
        if sensitivity_low is None or sensitivity_high is None:
            sensitivities_passed = False
        if not converged or bound_hit or mean is None:
            solvers_passed = False

        quality_status = "passed" if converged and not bound_hit and sensitivity_low is not None else "failed"
        if quality_status == "failed" and not exclusion_reason:
            exclusion_reason = (
                "mean_bound_hit" if bound_hit else
                "sensitivity_not_identifiable" if sensitivity_low is None else
                "optimizer_nonconvergence"
            )
        if quality_status == "passed" and source == "kalshi":
            curve_exclusion_reasons: list[str] = []
            if logit_rmse is None or logit_rmse > float(current["max_kalshi_logit_rmse"]):
                curve_exclusion_reasons.append("kalshi_curve_residual_exceeds_limit")
            if len(eligible) > 1 and (
                holdout_failed_fits > 0 or holdout_mae is None
                or holdout_mae > float(current["max_kalshi_holdout_logit_mae"])
            ):
                curve_exclusion_reasons.append("kalshi_curve_holdout_exceeds_limit")
            if curve_exclusion_reasons:
                quality_status = "excluded"
                exclusion_reason = "|".join(curve_exclusion_reasons)
                quarantined_kalshi_curves.append({
                    "stat": stat,
                    "canonical_player_id": base["canonical_player_id"],
                    "reasons": tuple(curve_exclusion_reasons),
                })
            else:
                kalshi_residuals.append(float(logit_rmse))
                if holdout_mae is not None:
                    kalshi_holdouts.append(float(holdout_mae))
        projection_rows.append({
            **common,
            "mean": mean,
            "sensitivity_low": sensitivity_low,
            "sensitivity_high": sensitivity_high,
            "projection_method": projection_method or "failed",
            "thresholds_used": "|".join(str(int(value)) for value in thresholds),
            "fit_objective": objective,
            "fit_error": fit_error,
            "logit_rmse": logit_rmse,
            "max_abs_logit_residual": max_logit_residual,
            "holdout_logit_mae": holdout_mae,
            "holdout_failed_fits": holdout_failed_fits,
            "optimizer_converged": converged,
            "mean_bound_hit": bound_hit,
            "quality_status": quality_status,
            "exclusion_reason": exclusion_reason,
            "sensitivity_failed_fits": sensitivity_failures,
        })

        if mean is not None and modeled:
            for index, modeled_probability, probability_residual, logit_residual in zip(
                eligible.index, modeled, residuals, logit_residuals, strict=True,
            ):
                frame.at[index, "modeled_probability"] = float(modeled_probability)
                frame.at[index, "probability_residual"] = float(probability_residual)
                frame.at[index, "logit_probability_residual"] = float(logit_residual)
                frame.at[index, "source_projection_mean"] = float(mean)
                frame.at[index, "source_projection_method"] = projection_method
                frame.at[index, "final_dispersion"] = final_dispersion
                frame.at[index, "dispersion_source"] = _text(dispersion_row["dispersion_source"])
                frame.at[index, "calibration_version"] = _text(dispersion_row["calibration_version"])
                frame.at[index, "market_update_version"] = MARKET_UPDATE_VERSION
                frame.at[index, "source_projection_quality_status"] = quality_status
                frame.at[index, "source_projection_exclusion_reason"] = exclusion_reason

    projections = pd.DataFrame.from_records(projection_rows)
    if not projections.empty:
        projections = projections.sort_values(group_columns).reset_index(drop=True)

    passed_rows = projections.loc[projections["quality_status"] == "passed"] if not projections.empty else projections
    keys_unique = not projections.duplicated(group_columns).any() if not projections.empty else True
    _check(checks, "model.source_projection_keys", keys_unique, "Source projection grain is unique", rows=int(len(projections)))
    _check(checks, "model.at_least_one_projection", not passed_rows.empty, "At least one source projection is modeling-eligible", eligible_rows=int(len(passed_rows)))
    _check(checks, "model.mean_optimizers", solvers_passed, "Every attempted mean optimizer converged away from bounds")
    _check(checks, "model.sensitivity_bounds", sensitivities_passed, "Every fitted mean has identifiable model-sensitivity bounds")
    max_sportsbook = max(sportsbook_residuals, default=0.0)
    _check(
        checks, "model.sportsbook_back_substitution",
        max_sportsbook <= float(current["max_sportsbook_probability_residual"]),
        "Sportsbook back-substitution residuals are within tolerance",
        observed=max_sportsbook, maximum=float(current["max_sportsbook_probability_residual"]),
    )
    near_even_lower, near_even_upper = (
        float(value) for value in current["near_even_probability_band"]
    )
    near_even = frame.loc[
        frame["source"].isin({"draftkings", "fanduel"})
        & frame["model_inclusion_status"].eq("included")
        & frame["source_projection_quality_status"].eq("passed")
        & frame["modeling_probability"].between(near_even_lower, near_even_upper)
    ].copy()
    implausible_near_even: list[dict[str, Any]] = []
    for _, row in near_even.iterrows():
        median_proxy = float(row["canonical_threshold"]) - 0.5
        shift = abs(float(row["source_projection_mean"]) - median_proxy)
        maximum = max(
            float(current["max_near_even_mean_shift_absolute"]),
            float(current["max_near_even_mean_shift_relative"]) * median_proxy,
        )
        if shift > maximum:
            implausible_near_even.append({
                "source": _text(row["source"]),
                "canonical_player_name": _text(row["canonical_player_name"]),
                "stat": _text(row["stat"]),
                "modeling_probability": float(row["modeling_probability"]),
                "median_proxy": median_proxy,
                "inferred_mean": float(row["source_projection_mean"]),
                "absolute_shift": shift,
                "maximum_shift": maximum,
                "dispersion_source": _text(row["dispersion_source"]),
            })
    _check(
        checks, "model.near_even_market_mean_plausibility", not implausible_near_even,
        "Near-even sportsbook lines do not imply means materially separated from their median proxies",
        evaluated_quotes=int(len(near_even)), violations=len(implausible_near_even),
        probability_band=[near_even_lower, near_even_upper],
        maximum_relative_shift=float(current["max_near_even_mean_shift_relative"]),
        maximum_absolute_shift=float(current["max_near_even_mean_shift_absolute"]),
        samples=implausible_near_even[:20],
    )
    max_kalshi = max(kalshi_residuals, default=0.0)
    _check(
        checks, "model.kalshi_source_residuals", max_kalshi <= float(current["max_kalshi_logit_rmse"]),
        "Every eligible Kalshi source curve has logit RMSE within tolerance",
        observed=max_kalshi, maximum=float(current["max_kalshi_logit_rmse"]),
        evaluated_curves=len(kalshi_residuals), quarantined_curves=len(quarantined_kalshi_curves),
    )
    max_holdout = max(kalshi_holdouts, default=0.0)
    _check(
        checks, "model.kalshi_source_holdout", max_holdout <= float(current["max_kalshi_holdout_logit_mae"]),
        "Every eligible multi-threshold Kalshi curve has leave-one-threshold-out error within tolerance",
        observed=max_holdout, maximum=float(current["max_kalshi_holdout_logit_mae"]),
        evaluated_curves=len(kalshi_holdouts), quarantined_curves=len(quarantined_kalshi_curves),
    )
    quarantined_by_stat: dict[str, int] = {}
    quarantined_by_reason: dict[str, int] = {}
    for value in quarantined_kalshi_curves:
        quarantined_by_stat[value["stat"]] = quarantined_by_stat.get(value["stat"], 0) + 1
        for reason in value["reasons"]:
            quarantined_by_reason[reason] = quarantined_by_reason.get(reason, 0) + 1
    _check(
        checks, "model.kalshi_curves_quarantined", not quarantined_kalshi_curves,
        "Kalshi curves that fail residual or holdout limits are excluded from downstream consensus",
        severity="warning", curves=len(quarantined_kalshi_curves),
        by_stat=quarantined_by_stat, by_reason=quarantined_by_reason,
    )
    excluded_contributed = (
        (frame["model_inclusion_status"] != "included") & frame["modeled_probability"].notna()
    )
    included_missing = (
        (frame["model_inclusion_status"] == "included") & frame["modeled_probability"].isna()
    )
    _check(
        checks, "model.quote_inclusion_lineage",
        not excluded_contributed.any() and not included_missing.any(),
        "Only included quotes contribute and every included quote is back-substituted",
        excluded_contributed=int(excluded_contributed.sum()), included_missing=int(included_missing.sum()),
    )

    monotone = True
    for _, group in frame.loc[frame["modeled_probability"].notna()].groupby(group_columns, sort=True):
        ordered = group.sort_values("canonical_threshold")
        if np.any(np.diff(ordered["modeled_probability"].to_numpy(dtype=float)) > 1e-12):
            monotone = False
            break
    _check(checks, "model.fitted_survival_monotonicity", monotone, "Fitted survival probabilities are non-increasing across thresholds")

    finite_means = True
    for _, row in passed_rows.iterrows():
        bounds = current["mean_bounds"][str(row["stat"])]
        finite_means = finite_means and _finite(row["mean"]) and float(bounds[0]) < float(row["mean"]) < float(bounds[1])
        finite_means = finite_means and _finite(row["sensitivity_low"]) and _finite(row["sensitivity_high"])
        finite_means = finite_means and float(row["sensitivity_low"]) <= float(row["mean"]) <= float(row["sensitivity_high"])
    _check(checks, "model.mean_domains", finite_means, "Fitted means and sensitivity bounds are finite, positive, ordered, and within configured stat bounds")
    return projections, frame


def estimate_market_means(
    priced_markets: pd.DataFrame,
    dispersions: pd.DataFrame,
    historical_report: dict[str, Any],
    model_config: dict[str, Any],
) -> MeanEstimationResult:
    """Update stat dispersion when eligible and fit every source/player/stat mean."""

    markets, historical_dispersions = _coerce_inputs(priced_markets, dispersions, historical_report)
    markets = _annotate_model_eligibility(markets, model_config)
    checks: list[CheckResult] = []
    invalid_priced_inclusions = (
        (markets["inclusion_status"] == "included")
        & markets["model_exclusion_reason"].isin({"invalid_model_threshold", "invalid_model_probability"})
    )
    _check(
        checks, "model.pricing_included_quote_contract",
        not invalid_priced_inclusions.any(),
        "Quotes accepted by pricing retain valid canonical model thresholds and probabilities",
        invalid_rows=int(invalid_priced_inclusions.sum()),
    )
    updated_dispersions, update_reports = _update_dispersions(
        markets, historical_dispersions, historical_report, model_config, checks,
    )
    projections, enriched_markets = _estimate_source_projections(
        markets, updated_dispersions, model_config, checks,
    )

    final_dispersion_valid = all(
        _finite(row["final_dispersion"]) and float(row["final_dispersion"]) > 0
        and _text(row["dispersion_source"]) in {
            "historical_only", "historical_plus_current_market_map", "current_market_conflict_override",
        }
        for _, row in updated_dispersions.iterrows()
    )
    _check(
        checks, "model.final_dispersions", final_dispersion_valid,
        "Every stat has a finite positive final dispersion and explicit source method",
    )
    errors = [check for check in checks if check.severity == "error" and not check.passed]
    warnings = [check for check in checks if check.severity == "warning" and not check.passed]
    validation = {
        "status": "failed" if errors else "passed",
        "market_update_version": MARKET_UPDATE_VERSION,
        "source_estimation_version": SOURCE_ESTIMATION_VERSION,
        "distribution_family": "negative_binomial",
        "sensitivity_interpretation": "model sensitivity, not a confidence interval",
        "quote_filters": {
            "probability_floor": float(model_config["probability_floor"]),
            "probability_ceiling": float(model_config["probability_ceiling"]),
            "pricing_inclusion_required": True,
        },
        "dispersion_updates": update_reports,
        "checks": [check.to_dict() for check in checks],
        "summary": {
            "priced_rows": int(len(enriched_markets)),
            "model_eligible_quotes": int((enriched_markets["model_inclusion_status"] == "included").sum()),
            "source_projection_rows": int(len(projections)),
            "eligible_source_projections": int((projections["quality_status"] == "passed").sum()) if not projections.empty else 0,
            "quarantined_kalshi_curves": int((
                projections["source"].eq("kalshi")
                & projections["quality_status"].eq("excluded")
                & projections["exclusion_reason"].astype(str).str.startswith("kalshi_curve_")
            ).sum()) if not projections.empty else 0,
            "historical_only_stats": sum(value["method"] == "historical_only" for value in update_reports.values()),
            "historical_plus_current_market_stats": sum(value["method"] == "historical_plus_current_market_map" for value in update_reports.values()),
            "current_market_conflict_override_stats": sum(value["method"] == "current_market_conflict_override" for value in update_reports.values()),
            "errors": len(errors),
            "warnings": len(warnings),
        },
    }
    report = deepcopy(historical_report)
    report["method"] = (
        "historical_with_current_market_update" if any(value["method"] != "historical_only" for value in update_reports.values())
        else "historical_only"
    )
    report["market_update_version"] = MARKET_UPDATE_VERSION
    report["current_market_update"] = {
        "status": validation["status"],
        "stats": update_reports,
        "model_validation_summary": validation["summary"],
    }
    for stat, update in update_reports.items():
        if stat in report.get("stats", {}):
            report["stats"][stat]["current_market_update"] = update
    return MeanEstimationResult(
        dispersions=updated_dispersions.sort_values("stat").reset_index(drop=True),
        source_projections=projections,
        priced_markets=enriched_markets,
        historical_report=report,
        validation=validation,
    )
