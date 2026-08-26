from __future__ import annotations

import numpy as np
import pytest

from ff_market_projections.distributions import (
    DistributionError,
    fit_negative_binomial_mean_curve,
    invert_negative_binomial_mean,
    negative_binomial_survival,
)


@pytest.mark.parametrize("probability", [0.0201, 0.20, 0.60, 0.9799])
def test_bounded_single_quote_inversion_recovers_extreme_valid_probabilities(probability: float) -> None:
    result = invert_negative_binomial_mean(801, probability, 4.0, (0.01, 100_000.0))
    assert result.converged
    assert not result.bound_hit
    assert result.modeled_probability == pytest.approx(probability, abs=1e-9)
    assert result.probability_residual == pytest.approx(0.0, abs=1e-9)


def test_probability_and_dispersion_assumptions_change_inferred_mean() -> None:
    low_probability = invert_negative_binomial_mean(801, 0.40, 4.0, (0.01, 100_000.0))
    high_probability = invert_negative_binomial_mean(801, 0.60, 4.0, (0.01, 100_000.0))
    different_dispersion = invert_negative_binomial_mean(801, 0.60, 40.0, (0.01, 100_000.0))
    assert high_probability.mean > low_probability.mean
    assert high_probability.mean != pytest.approx(different_dispersion.mean, rel=1e-3)


def test_robust_multi_threshold_fit_recovers_one_shared_mean() -> None:
    thresholds = np.asarray([500, 700, 900, 1100, 1300])
    probabilities = negative_binomial_survival(thresholds, 900.0, 5.0)
    fit = fit_negative_binomial_mean_curve(thresholds, probabilities, 5.0, (1.0, 5000.0))
    assert fit.converged
    assert not fit.bound_hit
    assert fit.mean == pytest.approx(900.0, rel=1e-8)
    assert fit.logit_rmse < 1e-8
    assert fit.modeled_probabilities == pytest.approx(probabilities)


def test_inversion_and_curve_fit_surface_impossible_inputs_and_bounds() -> None:
    with pytest.raises(DistributionError, match="positive integer"):
        invert_negative_binomial_mean(0, 0.5, 4.0, (0.1, 100.0))
    with pytest.raises(DistributionError, match="not attainable"):
        invert_negative_binomial_mean(1000, 0.9, 4.0, (0.1, 10.0))
    fit = fit_negative_binomial_mean_curve([1, 2, 3], [0.999, 0.999, 0.999], 4.0, (0.1, 1.0))
    assert fit.converged
    assert fit.bound_hit
