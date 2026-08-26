"""Pure odds and sportsbook no-vig conversion functions."""

from __future__ import annotations

import math
from typing import Any


class OddsError(ValueError):
    """An odds value cannot be converted safely."""


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise OddsError(f"{name} must be a finite number")
    return float(value)


def american_to_decimal(american_odds: int | float | str) -> float:
    """Convert American odds (including ``EVEN``) to decimal odds."""

    if isinstance(american_odds, str):
        if american_odds.strip().upper() == "EVEN":
            return 2.0
        raise OddsError("American odds must be a nonzero number or EVEN")
    odds = _number(american_odds, "American odds")
    if odds == 0:
        raise OddsError("American odds cannot be zero")
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / abs(odds))


def decimal_to_american(decimal_odds: int | float) -> float:
    """Convert decimal odds back to American odds."""

    decimal = _number(decimal_odds, "Decimal odds")
    if decimal <= 1.0:
        raise OddsError("Decimal odds must exceed one")
    return 100.0 * (decimal - 1.0) if decimal >= 2.0 else -100.0 / (decimal - 1.0)


def implied_probability(decimal_odds: int | float) -> float:
    """Return the raw implied probability from decimal odds."""

    decimal = _number(decimal_odds, "Decimal odds")
    if decimal <= 1.0:
        raise OddsError("Decimal odds must exceed one")
    return 1.0 / decimal


def american_implied_probability(american_odds: int | float | str) -> float:
    """Return raw implied probability directly from American odds."""

    return implied_probability(american_to_decimal(american_odds))


def proportional_devig(over_probability: int | float, under_probability: int | float) -> tuple[float, float]:
    """Normalize a two-way sportsbook book to probabilities summing to one."""

    over = _number(over_probability, "Over implied probability")
    under = _number(under_probability, "Under implied probability")
    if not 0 < over < 1 or not 0 < under < 1:
        raise OddsError("Two-way implied probabilities must be strictly between zero and one")
    total = over + under
    return over / total, under / total


def overround(*probabilities: int | float) -> float:
    """Return the total implied probability for a market."""

    if not probabilities:
        raise OddsError("At least one probability is required")
    return sum(_number(value, "Implied probability") for value in probabilities)
