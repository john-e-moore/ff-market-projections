"""Raw market gates and source-preserving canonical market rows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any

from .contracts import CheckResult


SUPPORTED_STATS = frozenset({
    "passing_yards", "passing_touchdowns", "rushing_yards", "rushing_touchdowns",
    "receiving_yards", "receiving_touchdowns", "receptions",
})
SUPPORTED_OPERATORS = frozenset({"gt", "ge", "lt", "le"})
_SOURCE_NAMES = ("draftkings", "fanduel", "kalshi")
_SPORTSBOOK_LABEL_LINE = re.compile(r"(?:Over|Under)\s+(-?[0-9]+(?:\.[0-9]+)?)$")


class MarketValidationError(ValueError):
    def __init__(self, message: str, validation: dict[str, Any]) -> None:
        super().__init__(message)
        self.validation = validation


@dataclass(frozen=True)
class CollectionValidation:
    source_data: dict[str, dict[str, Any]]
    report: dict[str, Any]


def _check(name: str, passed: bool, *, severity: str = "error", message: str = "", **details: Any) -> CheckResult:
    return CheckResult(name, passed, severity, message, details)


def _finite_number(value: Any, *, minimum: float | None = None, maximum: float | None = None) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and (minimum is None or value >= minimum) and (maximum is None or value <= maximum)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MarketValidationError(f"Unreadable JSON input {path}: {exc}", {"status": "failed", "checks": []}) from exc
    if not isinstance(value, dict):
        raise MarketValidationError(f"Market input must be a JSON object: {path}", {"status": "failed", "checks": []})
    return value


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else None


def _source_checks(source: str, payload: dict[str, Any], season: str, now: datetime, max_age_hours: float) -> list[CheckResult]:
    checks: list[CheckResult] = []
    markets = payload.get("markets")
    metadata = payload.get("metadata")
    quality = payload.get("data_quality")
    checks.append(_check(f"collections.{source}.schema", isinstance(metadata, dict) and isinstance(quality, dict) and isinstance(markets, list), message="metadata, data_quality, and markets are required"))
    if not isinstance(metadata, dict) or not isinstance(quality, dict) or not isinstance(markets, list):
        return checks
    snapshot = _parse_utc(metadata.get("snapshot_utc"))
    age_hours = (now - snapshot).total_seconds() / 3600 if snapshot else None
    checks.append(_check(f"collections.{source}.season", metadata.get("season") == season, message="source season must equal configured season", observed=metadata.get("season"), expected=season))
    checks.append(_check(f"collections.{source}.freshness", snapshot is not None and age_hours is not None and 0 <= age_hours <= max_age_hours, message="snapshot must be parseable and within configured age", age_hours=age_hours, max_age_hours=max_age_hours))
    checks.append(_check(f"collections.{source}.collector_quality", quality.get("checks_passed") is True, message="collector data_quality.checks_passed must be true"))
    checks.append(_check(f"collections.{source}.nonempty", bool(markets), message="markets must be nonempty", rows=len(markets)))
    id_key = "ticker" if source == "kalshi" else "market_id"
    ids = [row.get(id_key) for row in markets if isinstance(row, dict)]
    checks.append(_check(f"collections.{source}.unique_market_ids", len(ids) == len(markets) and all(isinstance(value, str) and value for value in ids) and len(ids) == len(set(ids)), message="source market IDs must be present and unique", id_key=id_key, rows=len(markets), unique_ids=len(set(ids))))
    for index, row in enumerate(markets):
        if not isinstance(row, dict):
            checks.append(_check(f"collections.{source}.market.{index}.object", False, message="market must be an object"))
            continue
        checks.extend(_market_checks(source, row, index, season))
    if source == "fanduel":
        placeholders = payload.get("unavailable_player_prop_references", [])
        checks.append(_check("collections.fanduel.expected_placeholders", not placeholders, severity="warning", message="collector-classified unavailable FanDuel references are retained as warnings", count=len(placeholders)))
        unrelated = payload.get("other_player_markets", [])
        checks.append(_check("collections.fanduel.unrelated_markets_excluded", not unrelated, severity="warning", message="non-O/U FanDuel player markets are intentionally excluded from the season-stat canonical contract", count=len(unrelated)))
    return checks


def _market_checks(source: str, row: dict[str, Any], index: int, season: str) -> list[CheckResult]:
    prefix = f"collections.{source}.market.{index}"
    checks = [_check(f"{prefix}.season", row.get("season") == season, message="market season must equal configured season"), _check(f"{prefix}.stat", row.get("stat") in SUPPORTED_STATS, message="stat must be supported", stat=row.get("stat"))]
    if source in {"draftkings", "fanduel"}:
        line = row.get("market_line")
        probabilities = row.get("probability")
        checks.append(_check(f"{prefix}.line", _finite_number(line), message="market line must be finite"))
        valid_sides = isinstance(probabilities, dict) and set(probabilities) >= {"over", "under"}
        sides = [probabilities.get(side) for side in ("over", "under")] if isinstance(probabilities, dict) else []
        valid_sides = valid_sides and all(isinstance(side, dict) for side in sides)
        checks.append(_check(f"{prefix}.two_sided", valid_sides, message="sportsbook market must contain Over and Under sides"))
        if valid_sides:
            operators = row.get("threshold_operators") or {}
            checks.append(_check(f"{prefix}.operators", operators.get("over") == ">" and operators.get("under") == "<", message="sportsbook operators must be > and <"))
            for side_name, side in zip(("over", "under"), sides):
                odds = side.get("american_odds")
                checks.append(_check(f"{prefix}.{side_name}.odds", isinstance(odds, int) and not isinstance(odds, bool) and odds != 0, message="American odds must be a nonzero integer"))
                checks.append(_check(f"{prefix}.{side_name}.selection_id", bool(side.get("selection_id")), message="selection ID is required"))
                label_match = _SPORTSBOOK_LABEL_LINE.search(str(side.get("label") or ""))
                parsed_line = float(label_match.group(1)) if label_match else None
                checks.append(_check(f"{prefix}.{side_name}.line", parsed_line == float(line) if _finite_number(line) else False, message="selection label line must match market line", label=side.get("label"), market_line=line))
    else:
        threshold = row.get("threshold")
        checks.append(_check(f"{prefix}.threshold", _finite_number(threshold, minimum=0) and float(threshold).is_integer(), message="Kalshi threshold must be a nonnegative integer"))
        checks.append(_check(f"{prefix}.operator", row.get("threshold_operator") == ">=", message="Kalshi threshold operator must be >="))
        probability = row.get("probability") or {}
        book = row.get("orderbook") or {}
        for name, value in (("yes_bid_percent", probability.get("yes_bid_percent")), ("yes_ask_percent", probability.get("yes_ask_percent")), ("yes_midpoint_percent", probability.get("yes_midpoint_percent"))):
            checks.append(_check(f"{prefix}.{name}", value is None or _finite_number(value, minimum=0, maximum=100), message="Kalshi probability must be in [0, 100]"))
        bid, ask = book.get("best_yes_bid_dollars"), book.get("best_yes_ask_dollars")
        checks.append(_check(f"{prefix}.book", (bid is None or _finite_number(bid, minimum=0, maximum=1)) and (ask is None or _finite_number(ask, minimum=0, maximum=1)) and (bid is None or ask is None or bid <= ask), message="Kalshi book must have probabilities in [0, 1] and bid <= ask"))
    return checks


def validate_collections(raw_dir: str | Path, config: dict[str, Any], *, now: datetime | None = None) -> CollectionValidation:
    raw = Path(raw_dir)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    season, run = config["run"]["season"], config["run"]
    payloads = {source: _load_json(raw / f"{source}.json") for source in _SOURCE_NAMES}
    checks = [check for source, payload in payloads.items() for check in _source_checks(source, payload, season, now, run["max_snapshot_age_hours"])]
    times = [_parse_utc(payload["metadata"].get("snapshot_utc")) for payload in payloads.values() if isinstance(payload.get("metadata"), dict)]
    valid_times = [value for value in times if value is not None]
    skew_minutes = (max(valid_times) - min(valid_times)).total_seconds() / 60 if len(valid_times) == len(_SOURCE_NAMES) else None
    checks.append(_check("collections.cross_source_snapshot_skew", skew_minutes is not None and skew_minutes <= run["max_cross_source_skew_minutes"], message="market snapshots must be within configured skew", skew_minutes=skew_minutes, max_skew_minutes=run["max_cross_source_skew_minutes"]))
    errors = [check for check in checks if check.severity == "error" and not check.passed]
    report = {"status": "failed" if errors else "passed", "season": season, "checks": [check.to_dict() for check in checks], "summary": {"errors": len(errors), "warnings": sum(check.severity == "warning" and not check.passed for check in checks), "source_market_rows": {source: len(payload["markets"]) for source, payload in payloads.items()}, "cross_source_snapshot_skew_minutes": skew_minutes}}
    if errors:
        raise MarketValidationError("Raw market collection validation failed", report)
    return CollectionValidation(payloads, report)


def _base_row(run_id: str, source: str, market: dict[str, Any], source_market_id: str, snapshot: str) -> dict[str, Any]:
    return {"run_id": run_id, "source": source, "source_market_id": source_market_id, "snapshot_utc": snapshot, "season": market["season"], "raw_player_name": market.get("player"), "team": market.get("team"), "team_abbreviation": market.get("team_abbreviation"), "stat": market["stat"], "unit": market.get("threshold_unit"), "normalization_status": "included", "exclusion_reason": ""}


def normalize_markets(collections: CollectionValidation, run_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, payload in collections.source_data.items():
        snapshot = payload["metadata"]["snapshot_utc"]
        for index, market in enumerate(payload["markets"]):
            if source in {"draftkings", "fanduel"}:
                for side_name, operator in (("over", "gt"), ("under", "lt")):
                    side = market["probability"][side_name]
                    row = _base_row(run_id, source, market, str(market["market_id"]), snapshot)
                    row.update({"source_selection_id": str(side["selection_id"]), "operator": operator, "threshold": market["market_line"], "outcome_side": side_name, "american_odds": side["american_odds"], "source_decimal_odds": side.get("decimal_odds"), "yes_bid_probability": None, "yes_ask_probability": None, "last_trade_probability": None, "volume": None, "open_interest": None, "spread": None, "source_url": (market.get("source_urls") or {}).get("sportsbook_page"), "raw_record_locator": f"$.markets[{index}].probability.{side_name}"})
                    rows.append(row)
            else:
                probability, book = market.get("probability") or {}, market.get("orderbook") or {}
                row = _base_row(run_id, source, market, str(market["ticker"]), snapshot)
                activity = market.get("trading_activity") or {}
                row.update({"source_selection_id": "", "operator": "ge", "threshold": market["threshold"], "outcome_side": "yes", "american_odds": None, "source_decimal_odds": None, "yes_bid_probability": _percent_to_probability(probability.get("yes_bid_percent")), "yes_ask_probability": _percent_to_probability(probability.get("yes_ask_percent")), "last_trade_probability": _percent_to_probability(probability.get("last_trade_percent")), "volume": activity.get("volume_contracts"), "open_interest": activity.get("open_interest_contracts"), "spread": book.get("yes_spread_dollars"), "source_url": (market.get("source_urls") or {}).get("market_api"), "raw_record_locator": f"$.markets[{index}]"})
                rows.append(row)
    return sorted(rows, key=lambda row: (row["source"], row["source_market_id"], row["outcome_side"]))


def _percent_to_probability(value: Any) -> float | None:
    return None if value is None else float(value) / 100


def validate_normalized(rows: list[dict[str, Any]], collections: CollectionValidation) -> dict[str, Any]:
    required = {"run_id", "source", "source_market_id", "source_selection_id", "snapshot_utc", "season", "raw_player_name", "stat", "unit", "operator", "threshold", "outcome_side", "american_odds", "source_decimal_odds", "yes_bid_probability", "yes_ask_probability", "last_trade_probability", "volume", "open_interest", "spread", "source_url", "raw_record_locator", "normalization_status", "exclusion_reason"}
    checks = [_check("normalized.nonempty", bool(rows), message="normalized rows must be nonempty"), _check("normalized.required_columns", all(required <= set(row) for row in rows), message="every canonical row must expose all required fields")]
    keys = [(row.get("source"), row.get("source_market_id"), row.get("outcome_side")) for row in rows]
    checks.append(_check("normalized.keys_unique", len(keys) == len(set(keys)), message="source/market/outcome keys must be unique"))
    checks.append(_check("normalized.types_and_domains", all(row.get("source") in _SOURCE_NAMES and row.get("stat") in SUPPORTED_STATS and row.get("operator") in SUPPORTED_OPERATORS and _finite_number(row.get("threshold")) for row in rows), message="source, stat, operator, and threshold values must be canonical"))
    sportsbook = [row for row in rows if row["source"] in {"draftkings", "fanduel"}]
    checks.append(_check("normalized.sportsbook_odds", all(isinstance(row["american_odds"], int) and row["american_odds"] != 0 for row in sportsbook), message="sportsbook rows must preserve valid American odds"))
    kalshi = [row for row in rows if row["source"] == "kalshi"]
    checks.append(_check("normalized.kalshi_probabilities", all(all(value is None or _finite_number(value, minimum=0, maximum=1) for value in (row["yes_bid_probability"], row["yes_ask_probability"], row["last_trade_probability"])) for row in kalshi), message="Kalshi probability fields must be 0-1 values"))
    expected = sum(len(payload["markets"]) * (1 if source == "kalshi" else 2) for source, payload in collections.source_data.items())
    checks.append(_check("normalized.row_count", len(rows) == expected, message="all validated raw market outcomes must be represented", rows=len(rows), expected=expected))
    errors = [check for check in checks if not check.passed]
    return {"status": "failed" if errors else "passed", "checks": [check.to_dict() for check in checks], "summary": {"rows": len(rows), "expected_rows": expected, "errors": len(errors)}}
