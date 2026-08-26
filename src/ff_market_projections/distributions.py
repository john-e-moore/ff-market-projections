"""Probability primitives for explicit mean/dispersion model assumptions."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import nbinom


class DistributionError(ValueError):
    """A distribution argument is outside the supported model contract."""


def _finite_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise DistributionError(f"{name} must be finite")
    return array


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
