"""Probability primitives for explicit mean/dispersion model assumptions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import nbinom


class DistributionError(ValueError):
    """A distribution argument is outside the supported model contract."""


@dataclass(frozen=True)
class MeanInversion:
    """Auditable result of one bounded survival-probability inversion."""

    mean: float
    modeled_probability: float
    probability_residual: float
    converged: bool
    bound_hit: bool
    iterations: int
    function_evaluations: int


@dataclass(frozen=True)
class MeanCurveFit:
    """One mean fitted to a fixed-dispersion threshold curve."""

    mean: float
    objective_cost: float
    modeled_probabilities: tuple[float, ...]
    probability_residuals: tuple[float, ...]
    logit_residuals: tuple[float, ...]
    logit_rmse: float
    logit_mae: float
    max_abs_logit_residual: float
    converged: bool
    bound_hit: bool
    optimizer_status: int
    optimizer_message: str
    optimizer_evaluations: int


def _finite_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise DistributionError(f"{name} must be finite")
    return array


def _scalar(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise DistributionError(f"{name} must be a finite scalar")
    array = _finite_array(value, name)
    if array.ndim != 0:
        raise DistributionError(f"{name} must be a finite scalar")
    return float(array)


def _positive_bounds(bounds: tuple[float, float], name: str) -> tuple[float, float]:
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        raise DistributionError(f"{name} must contain exactly two values")
    lower, upper = _scalar(bounds[0], f"{name} lower"), _scalar(bounds[1], f"{name} upper")
    if not 0 < lower < upper:
        raise DistributionError(f"{name} must satisfy 0 < lower < upper")
    return lower, upper


def _logit(probability: np.ndarray) -> np.ndarray:
    return np.log(probability) - np.log1p(-probability)


def negative_binomial_parameters(mean: Any, dispersion: float) -> tuple[np.ndarray, float]:
    """Return scipy's success probability and shape for ``E[X] = mean``.

    A zero mean is supported as the degenerate distribution at zero. Historical
    calibration excludes zero-mean rows because they contain no dispersion
    information.
    """

    means = _finite_array(mean, "mean")
    if (means < 0).any():
        raise DistributionError("mean must be nonnegative")
    if isinstance(dispersion, bool) or not np.isfinite(dispersion) or dispersion <= 0:
        raise DistributionError("dispersion must be finite and positive")
    probability = dispersion / (dispersion + means)
    return probability, float(dispersion)


def negative_binomial_variance(mean: Any, dispersion: float) -> np.ndarray:
    """Return ``mu + mu**2 / r`` under the project parameterization."""

    means = _finite_array(mean, "mean")
    negative_binomial_parameters(means, dispersion)
    return means + means**2 / float(dispersion)


def negative_binomial_logpmf(outcome: Any, mean: Any, dispersion: float) -> np.ndarray:
    """Evaluate integer nonnegative outcomes under the project parameterization."""

    outcomes = _finite_array(outcome, "outcome")
    means = _finite_array(mean, "mean")
    if (outcomes < 0).any() or not np.allclose(outcomes, np.rint(outcomes), rtol=0.0, atol=1e-9):
        raise DistributionError("outcome must contain nonnegative integers")
    probability, shape = negative_binomial_parameters(means, dispersion)
    return np.asarray(nbinom.logpmf(np.rint(outcomes), shape, probability), dtype=float)


def negative_binomial_survival(threshold: Any, mean: Any, dispersion: float) -> np.ndarray:
    """Return ``P(X >= threshold)`` for nonnegative integer thresholds."""

    thresholds = _finite_array(threshold, "threshold")
    if (thresholds < 0).any() or not np.allclose(thresholds, np.rint(thresholds), rtol=0.0, atol=1e-9):
        raise DistributionError("threshold must contain nonnegative integers")
    probability, shape = negative_binomial_parameters(mean, dispersion)
    return np.asarray(nbinom.sf(np.rint(thresholds) - 1, shape, probability), dtype=float)


def negative_binomial_interval(mean: Any, dispersion: float, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Return equal-tailed inclusive predictive interval bounds."""

    if isinstance(level, bool) or not np.isfinite(level) or not 0 < level < 1:
        raise DistributionError("level must be strictly between zero and one")
    probability, shape = negative_binomial_parameters(mean, dispersion)
    tail = (1.0 - float(level)) / 2.0
    lower = np.asarray(nbinom.ppf(tail, shape, probability), dtype=float)
    upper = np.asarray(nbinom.ppf(1.0 - tail, shape, probability), dtype=float)
    return lower, upper


def invert_negative_binomial_mean(
    threshold: int,
    probability: float,
    dispersion: float,
    mean_bounds: tuple[float, float],
    *,
    probability_tolerance: float = 1e-10,
) -> MeanInversion:
    """Solve ``P(X >= threshold) = probability`` with dispersion held fixed.

    The root is found in log-mean space so the configured positive bounds remain
    explicit even when supported stat scales differ by orders of magnitude.
    """

    threshold_value = _scalar(threshold, "threshold")
    if threshold_value < 1 or not math.isclose(threshold_value, round(threshold_value), rel_tol=0.0, abs_tol=1e-9):
        raise DistributionError("threshold must be a positive integer for mean inversion")
    target = _scalar(probability, "probability")
    if not 0 < target < 1:
        raise DistributionError("probability must be strictly between zero and one")
    if isinstance(probability_tolerance, bool) or not np.isfinite(probability_tolerance) or probability_tolerance <= 0:
        raise DistributionError("probability_tolerance must be finite and positive")
    lower, upper = _positive_bounds(mean_bounds, "mean_bounds")
    # Validate dispersion through the shared parameterization before optimization.
    negative_binomial_parameters(lower, dispersion)
    integer_threshold = int(round(threshold_value))

    def residual(log_mean: float) -> float:
        modeled = float(negative_binomial_survival(integer_threshold, math.exp(log_mean), dispersion))
        return modeled - target

    log_lower, log_upper = math.log(lower), math.log(upper)
    lower_residual, upper_residual = residual(log_lower), residual(log_upper)
    tolerance = float(probability_tolerance)
    if lower_residual > tolerance or upper_residual < -tolerance:
        raise DistributionError(
            "target probability is not attainable within the configured mean bounds"
        )
    if abs(lower_residual) <= tolerance:
        log_mean, iterations, evaluations = log_lower, 0, 1
    elif abs(upper_residual) <= tolerance:
        log_mean, iterations, evaluations = log_upper, 0, 1
    else:
        root, details = brentq(
            residual,
            log_lower,
            log_upper,
            xtol=tolerance,
            rtol=max(tolerance, 4 * np.finfo(float).eps),
            maxiter=500,
            full_output=True,
            disp=False,
        )
        if not details.converged:
            raise DistributionError("bounded mean inversion did not converge")
        log_mean = float(root)
        iterations = int(details.iterations)
        evaluations = int(details.function_calls)
    mean = float(math.exp(log_mean))
    modeled = float(negative_binomial_survival(integer_threshold, mean, dispersion))
    log_span = log_upper - log_lower
    bound_tolerance = max(1e-8, log_span * 1e-7)
    return MeanInversion(
        mean=mean,
        modeled_probability=modeled,
        probability_residual=modeled - target,
        converged=True,
        bound_hit=bool(log_mean - log_lower <= bound_tolerance or log_upper - log_mean <= bound_tolerance),
        iterations=iterations,
        function_evaluations=evaluations,
    )


def fit_negative_binomial_mean_curve(
    thresholds: Any,
    probabilities: Any,
    dispersion: float,
    mean_bounds: tuple[float, float],
    *,
    robust_loss: str = "soft_l1",
    optimizer_tolerance: float = 1e-10,
    max_evaluations: int = 2000,
) -> MeanCurveFit:
    """Fit one mean to all threshold probabilities using robust logit residuals."""

    threshold_values = _finite_array(thresholds, "thresholds")
    probability_values = _finite_array(probabilities, "probabilities")
    if threshold_values.ndim != 1 or not len(threshold_values):
        raise DistributionError("thresholds must be a nonempty one-dimensional array")
    if probability_values.ndim != 1 or len(probability_values) != len(threshold_values):
        raise DistributionError("probabilities must be one-dimensional and match thresholds")
    if (threshold_values < 1).any() or not np.allclose(threshold_values, np.rint(threshold_values), rtol=0.0, atol=1e-9):
        raise DistributionError("thresholds must contain positive integers")
    if ((probability_values <= 0) | (probability_values >= 1)).any():
        raise DistributionError("probabilities must be strictly between zero and one")
    if robust_loss not in {"linear", "soft_l1", "huber", "cauchy", "arctan"}:
        raise DistributionError("robust_loss is unsupported")
    if isinstance(optimizer_tolerance, bool) or not np.isfinite(optimizer_tolerance) or optimizer_tolerance <= 0:
        raise DistributionError("optimizer_tolerance must be finite and positive")
    if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, int) or max_evaluations <= 0:
        raise DistributionError("max_evaluations must be a positive integer")
    lower, upper = _positive_bounds(mean_bounds, "mean_bounds")
    negative_binomial_parameters(lower, dispersion)
    integer_thresholds = np.rint(threshold_values).astype(int)
    target_logits = _logit(probability_values)
    log_bounds = (math.log(lower), math.log(upper))

    initial_roots: list[float] = []
    for threshold_value, probability_value in zip(integer_thresholds, probability_values, strict=True):
        try:
            inversion = invert_negative_binomial_mean(
                int(threshold_value), float(probability_value), dispersion, (lower, upper),
                probability_tolerance=max(float(optimizer_tolerance), 1e-12),
            )
        except DistributionError:
            continue
        if not inversion.bound_hit:
            initial_roots.append(math.log(inversion.mean))
    initial = float(np.median(initial_roots)) if initial_roots else (log_bounds[0] + log_bounds[1]) / 2.0

    def residuals(log_mean: np.ndarray) -> np.ndarray:
        modeled = negative_binomial_survival(integer_thresholds, math.exp(float(log_mean[0])), dispersion)
        # Finite positive means and thresholds yield probabilities in (0, 1),
        # but clipping guards floating-point saturation at extreme valid inputs.
        clipped = np.clip(modeled, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)
        return _logit(clipped) - target_logits

    optimized = least_squares(
        residuals,
        np.asarray([initial]),
        bounds=(np.asarray([log_bounds[0]]), np.asarray([log_bounds[1]])),
        loss=robust_loss,
        ftol=float(optimizer_tolerance),
        xtol=float(optimizer_tolerance),
        gtol=float(optimizer_tolerance),
        max_nfev=max_evaluations,
    )
    log_mean = float(optimized.x[0])
    mean = float(math.exp(log_mean))
    modeled = negative_binomial_survival(integer_thresholds, mean, dispersion)
    probability_residuals = modeled - probability_values
    logit_residuals = residuals(np.asarray([log_mean]))
    log_span = log_bounds[1] - log_bounds[0]
    bound_tolerance = max(1e-8, log_span * 1e-7)
    return MeanCurveFit(
        mean=mean,
        objective_cost=float(optimized.cost),
        modeled_probabilities=tuple(float(value) for value in modeled),
        probability_residuals=tuple(float(value) for value in probability_residuals),
        logit_residuals=tuple(float(value) for value in logit_residuals),
        logit_rmse=float(np.sqrt(np.mean(logit_residuals**2))),
        logit_mae=float(np.mean(np.abs(logit_residuals))),
        max_abs_logit_residual=float(np.max(np.abs(logit_residuals))),
        converged=bool(optimized.success and np.isfinite(optimized.cost)),
        bound_hit=bool(log_mean - log_bounds[0] <= bound_tolerance or log_bounds[1] - log_mean <= bound_tolerance),
        optimizer_status=int(optimized.status),
        optimizer_message=str(optimized.message),
        optimizer_evaluations=int(optimized.nfev),
    )
