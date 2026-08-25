"""Price normalized market observations into canonical survival probabilities."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .contracts import CheckResult
from .odds import OddsError, american_to_decimal, implied_probability, overround, proportional_devig


class PricingError(ValueError):
    def __init__(self, message: str, validation: dict[str, Any]) -> None:
        super().__init__(message)
        self.validation = validation


@dataclass(frozen=True)
class PricingResult:
    rows: list[dict[str, Any]]
    validation: dict[str, Any]


def _check(name: str, passed: bool, *, severity: str = "error", message: str = "", **details: Any) -> CheckResult:
    return CheckResult(name, passed, severity, message, details)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def canonical_sportsbook_threshold(line: Any, *, reject_integer_lines: bool) -> int:
    """Turn an Over ``L`` market into its discrete ``P(X >= k)`` threshold."""

    if not _finite(line) or float(line) < 0:
        raise OddsError("Sportsbook market line must be a finite nonnegative number")
    numeric_line = float(line)
    if reject_integer_lines and numeric_line.is_integer():
        raise OddsError("Integer sportsbook lines have ambiguous push/settlement semantics")
    return math.floor(numeric_line) + 1


def _copy(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in row}


def _sportsbook_rows(rows: list[dict[str, Any]], pricing: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("source") in {"draftkings", "fanduel"}:
            grouped.setdefault((row.get("source"), row.get("source_market_id")), []).append(row)
    for (source, market_id), pair in sorted(grouped.items()):
        over = next((row for row in pair if row.get("outcome_side") == "over"), None)
        under = next((row for row in pair if row.get("outcome_side") == "under"), None)
        if len(pair) != 2 or over is None or under is None:
            raise PricingError(f"Sportsbook market {source}/{market_id} must have exactly one Over and one Under", {"status": "failed", "checks": []})
        if over.get("threshold") != under.get("threshold"):
            raise PricingError(f"Sportsbook market {source}/{market_id} has mismatched lines", {"status": "failed", "checks": []})
        try:
            over_decimal, under_decimal = american_to_decimal(over.get("american_odds")), american_to_decimal(under.get("american_odds"))
            raw_over, raw_under = implied_probability(over_decimal), implied_probability(under_decimal)
            no_vig_over, no_vig_under = proportional_devig(raw_over, raw_under)
            threshold = canonical_sportsbook_threshold(over.get("threshold"), reject_integer_lines=pricing["reject_ambiguous_integer_lines"])
        except OddsError as exc:
            raise PricingError(f"Cannot price sportsbook market {source}/{market_id}: {exc}", {"status": "failed", "checks": []}) from exc
        priced = _copy(over)
        priced.update({
            "canonical_event": "P(X >= k)", "canonical_operator": "ge", "canonical_threshold": threshold,
            "decimal_odds": over_decimal, "raw_implied_probability": raw_over,
            "market_overround": overround(raw_over, raw_under), "devig_method": "proportional",
            "no_vig_probability": no_vig_over, "modeling_probability": no_vig_over,
            "paired_under_selection_id": under.get("source_selection_id"), "paired_under_american_odds": under.get("american_odds"),
            "paired_under_decimal_odds": under_decimal, "paired_under_raw_implied_probability": raw_under,
            "paired_under_no_vig_probability": no_vig_under, "kalshi_bid_probability": None,
            "kalshi_ask_probability": None, "kalshi_midpoint_probability": None,
            "inclusion_status": "included", "exclusion_reason": "",
        })
        output.append(priced)
    return output


def _kalshi_row(row: dict[str, Any], pricing: dict[str, Any]) -> dict[str, Any]:
    priced = _copy(row)
    bid, ask = row.get("yes_bid_probability"), row.get("yes_ask_probability")
    reason = ""
    if not _finite(row.get("threshold")) or float(row["threshold"]) < 0 or not float(row["threshold"]).is_integer():
        reason = "invalid_kalshi_threshold"
    elif not (_finite(bid) and _finite(ask) and 0 < float(bid) < 1 and 0 < float(ask) < 1):
        reason = "missing_two_sided_yes_quote"
    elif float(bid) > float(ask):
        reason = "invalid_yes_quote_order"
    else:
        spread = float(ask) - float(bid)
        if spread > float(pricing["max_spread_probability_points"]) / 100.0:
            reason = "yes_spread_exceeds_configured_limit"
        elif row.get("open_interest") is not None and (not _finite(row["open_interest"]) or float(row["open_interest"]) < float(pricing.get("min_open_interest_contracts", 0))):
            reason = "open_interest_below_configured_minimum"
    midpoint = (float(bid) + float(ask)) / 2.0 if not reason and bid is not None and ask is not None else None
    priced.update({
        "canonical_event": "P(X >= k)", "canonical_operator": "ge", "canonical_threshold": int(float(row["threshold"])) if not reason else None,
        "decimal_odds": 1.0 / midpoint if midpoint else None,
        "raw_implied_probability": midpoint, "market_overround": None, "devig_method": "not_applicable_kalshi",
        "no_vig_probability": midpoint, "modeling_probability": midpoint,
        "paired_under_selection_id": None, "paired_under_american_odds": None, "paired_under_decimal_odds": None,
        "paired_under_raw_implied_probability": None, "paired_under_no_vig_probability": None,
        "kalshi_bid_probability": bid, "kalshi_ask_probability": ask, "kalshi_midpoint_probability": midpoint,
        "inclusion_status": "excluded" if reason else "included", "exclusion_reason": reason,
    })
    return priced


def price_markets(rows: list[dict[str, Any]], config: dict[str, Any]) -> PricingResult:
    """Create one usable sportsbook Over or Kalshi YES observation per market."""

    pricing = {**config["pricing"], **config["sources"]["kalshi"]}
    priced = _sportsbook_rows(rows, pricing)
    priced.extend(_kalshi_row(row, pricing) for row in rows if row.get("source") == "kalshi" and row.get("outcome_side") == "yes")
    priced.sort(key=lambda row: (str(row.get("source")), str(row.get("source_market_id"))))
    validation = validate_priced_markets(priced, pricing)
    if validation["status"] != "passed":
        raise PricingError("Pricing validation failed", validation)
    return PricingResult(priced, validation)


def validate_priced_markets(rows: list[dict[str, Any]], pricing: dict[str, Any]) -> dict[str, Any]:
    """Validate arithmetic, threshold semantics, and auditable exclusions."""

    required = {
        "source", "source_market_id", "stat", "canonical_event", "canonical_operator", "canonical_threshold",
        "decimal_odds", "raw_implied_probability", "market_overround", "devig_method", "no_vig_probability",
        "modeling_probability", "inclusion_status", "exclusion_reason", "kalshi_bid_probability",
        "kalshi_ask_probability", "kalshi_midpoint_probability", "paired_under_no_vig_probability",
    }
    checks = [
        _check("pricing.nonempty", bool(rows), message="priced markets must not be empty"),
        _check("pricing.required_columns", all(required <= set(row) for row in rows), message="priced rows must expose the pricing contract"),
    ]
    included = [row for row in rows if row.get("inclusion_status") == "included"]
    excluded = [row for row in rows if row.get("inclusion_status") == "excluded"]
    tolerance = float(pricing["probability_tolerance"])
    checks.extend([
        _check("pricing.included_canonical_probability", all(row.get("canonical_event") == "P(X >= k)" and row.get("canonical_operator") == "ge" and isinstance(row.get("canonical_threshold"), int) and row["canonical_threshold"] >= 0 and _finite(row.get("modeling_probability")) and 0 < float(row["modeling_probability"]) < 1 for row in included), message="included rows must have a canonical threshold and probability"),
        _check("pricing.exclusions_are_auditable", all(bool(row.get("exclusion_reason")) and row.get("modeling_probability") is None for row in excluded), message="excluded rows must state a reason and cannot supply a model probability"),
        _check("pricing.sportsbook_pairs_reconcile", all(abs(float(row["no_vig_probability"]) + float(row["paired_under_no_vig_probability"]) - 1.0) <= tolerance and abs(float(row["modeling_probability"]) - float(row["no_vig_probability"])) <= tolerance and float(row["market_overround"]) > 1.0 for row in included if row.get("source") in {"draftkings", "fanduel"}), message="sportsbook no-vig pairs must sum to one within tolerance"),
        _check("pricing.kalshi_not_devigged", all(row.get("devig_method") == "not_applicable_kalshi" and row.get("market_overround") is None and row.get("modeling_probability") == row.get("kalshi_midpoint_probability") and row.get("no_vig_probability") == row.get("kalshi_midpoint_probability") for row in rows if row.get("source") == "kalshi"), message="Kalshi point probabilities must remain midpoint probabilities"),
        _check("pricing.kalshi_quote_fields", all(_finite(row.get("kalshi_bid_probability")) and _finite(row.get("kalshi_ask_probability")) and float(row["kalshi_bid_probability"]) <= float(row["kalshi_ask_probability"]) for row in included if row.get("source") == "kalshi"), message="included Kalshi rows must preserve ordered two-sided quotes"),
    ])
    source_decimal_checks = [row for row in rows if row.get("source") in {"draftkings", "fanduel"} and row.get("source_decimal_odds") is not None]
    checks.append(_check("pricing.source_decimal_crosscheck", all(abs(float(row["source_decimal_odds"]) - float(row["decimal_odds"])) <= tolerance for row in source_decimal_checks), severity="warning", message="collector-derived decimal odds are checked but not authoritative", checked_rows=len(source_decimal_checks)))
    errors = [check for check in checks if check.severity == "error" and not check.passed]
    return {"status": "failed" if errors else "passed", "checks": [check.to_dict() for check in checks], "summary": {"rows": len(rows), "included_rows": len(included), "excluded_rows": len(excluded), "errors": len(errors), "warnings": sum(check.severity == "warning" and not check.passed for check in checks)}}
