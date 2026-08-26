from __future__ import annotations

import numpy as np
import pytest

from ff_market_projections.distributions import (
    DistributionError,
    negative_binomial_interval,
    negative_binomial_logpmf,
    negative_binomial_parameters,
    negative_binomial_survival,
    negative_binomial_variance,
)


def test_negative_binomial_parameterization_and_exact_probabilities() -> None:
    probability, shape = negative_binomial_parameters(3.0, 2.0)
    assert float(probability) == pytest.approx(0.4)
    assert shape == 2.0
    assert float(negative_binomial_variance(3.0, 2.0)) == pytest.approx(7.5)
    assert float(negative_binomial_survival(3, 3.0, 2.0)) == pytest.approx(0.4752)
    assert float(np.exp(negative_binomial_logpmf(2, 3.0, 2.0))) == pytest.approx(0.1728)


def test_survival_is_monotone_and_intervals_contain_the_model_median() -> None:
    thresholds = np.arange(0, 25)
    survival = negative_binomial_survival(thresholds, 8.0, 4.0)
    assert survival[0] == 1.0
    assert np.all(np.diff(survival) <= 0)
    lower, upper = negative_binomial_interval(8.0, 4.0, 0.8)
    assert float(lower) <= 8 <= float(upper)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: negative_binomial_parameters(-1.0, 2.0), "mean must be nonnegative"),
        (lambda: negative_binomial_parameters(1.0, 0.0), "dispersion must be"),
        (lambda: negative_binomial_logpmf(1.5, 2.0, 3.0), "nonnegative integers"),
        (lambda: negative_binomial_survival(-1, 2.0, 3.0), "nonnegative integers"),
        (lambda: negative_binomial_interval(2.0, 3.0, 1.0), "strictly between"),
    ],
)
def test_invalid_distribution_inputs_fail_explicitly(call, message: str) -> None:
    with pytest.raises(DistributionError, match=message):
        call()
