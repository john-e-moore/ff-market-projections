#!/usr/bin/env python3
"""Fetch and normalize Kalshi's open NFL Season Stats markets.

Uses only public, unauthenticated Trade API endpoints. The output preserves the
raw Kalshi payloads and adds point-in-time probability and order-book metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
SERIES_TICKERS = [
    "KXNFLSEASONPASSYDS",
    "KXNFLSEASONPASSTDS",
    "KXNFLSEASONRSHYDS",
    "KXNFLSEASONRSHTD",
    "KXNFLSEASONREC",
    "KXNFLSEASONRECYDS",
    "KXNFLSEASONRECTD",
]
DEPTH_BANDS = (Decimal("0.01"), Decimal("0.05"), Decimal("0.10"), Decimal("0.25"))
EXECUTION_SIZES = (1, 10, 100, 500, 1000)


def fetch_json(path: str, params: Any = None, retries: int = 5) -> dict[str, Any]:
    url = f"{BASE_URL}{path}"
    if params:
        url = f"{url}?{urlencode(params, doseq=True)}"
    request = Request(url, headers={"User-Agent": "fantasy-market-projections/1.0"})
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=45) as response:
                return json.load(response)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (2**attempt))
    raise RuntimeError("unreachable")


def decimal_or_none(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        result = Decimal(str(value))
    except Exception:
        return None
    return result if result.is_finite() else None


def number(value: Decimal | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def percent(value: Decimal | None) -> float | None:
    return number(value * 100 if value is not None else None, 2)


def chunked(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def summarize_levels(levels: list[list[str]]) -> dict[str, Any]:
    parsed = [(Decimal(price), Decimal(contracts)) for price, contracts in levels]
    return {
        "level_count": len(parsed),
        "contracts": number(sum((qty for _, qty in parsed), Decimal(0)), 2),
        "bid_notional_dollars": number(sum((price * qty for price, qty in parsed), Decimal(0)), 2),
    }


def band_depth(levels: list[tuple[Decimal, Decimal]], best: Decimal | None, side: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if best is None:
        for band in DEPTH_BANDS:
            output[f"within_{int(band * 100)}c"] = {"contracts": 0.0, "notional_dollars": 0.0}
        return output
    for band in DEPTH_BANDS:
        if side == "bid":
            selected = [(price, qty) for price, qty in levels if price >= best - band]
        else:
            selected = [(price, qty) for price, qty in levels if price <= best + band]
        output[f"within_{int(band * 100)}c"] = {
            "contracts": number(sum((qty for _, qty in selected), Decimal(0)), 2),
            "notional_dollars": number(sum((price * qty for price, qty in selected), Decimal(0)), 2),
        }
    return output


def execution_quote(levels: list[tuple[Decimal, Decimal]], requested: int) -> dict[str, Any]:
    remaining = Decimal(requested)
    filled = Decimal(0)
    consideration = Decimal(0)
    worst: Decimal | None = None
    for price, available in levels:
        take = min(available, remaining)
        if take <= 0:
            continue
        filled += take
        remaining -= take
        consideration += take * price
        worst = price
        if remaining <= 0:
            break
    return {
        "requested_contracts": requested,
        "fillable_contracts": number(filled, 2),
        "fully_fillable": remaining <= 0,
        "vwap_dollars": number(consideration / filled if filled else None, 4),
        "worst_price_dollars": number(worst, 4),
        "consideration_dollars": number(consideration, 2),
    }


def probability_metrics(market: dict[str, Any], best_bid: Decimal | None, best_ask: Decimal | None) -> dict[str, Any]:
    last = decimal_or_none(market.get("last_price_dollars"))
    midpoint = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else None
    if midpoint is not None:
        estimate, method = midpoint, "midpoint_of_best_yes_bid_and_ask"
    elif last is not None and last > 0:
        estimate, method = last, "last_trade_price_no_two_sided_quote"
    elif best_bid is not None:
        estimate, method = best_bid, "one_sided_yes_bid_reference"
    elif best_ask is not None:
        estimate, method = best_ask, "one_sided_yes_ask_reference"
    else:
        estimate, method = None, "unavailable"
    return {
        "estimated_chance_percent": percent(estimate),
        "estimate_method": method,
        "yes_bid_percent": percent(best_bid),
        "yes_ask_percent": percent(best_ask),
        "yes_midpoint_percent": percent(midpoint),
        "last_trade_percent": percent(last if last is not None and last > 0 else None),
        "bid_ask_probability_interval_percent": [percent(best_bid), percent(best_ask)],
        "note": "Contract prices are market-implied probabilities, not objective forecasts; wide spreads indicate greater uncertainty/illiquidity.",
    }


def orderbook_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    yes_raw = raw.get("yes_dollars") or []
    no_raw = raw.get("no_dollars") or []
    yes_bids = sorted(
        [(Decimal(price), Decimal(qty)) for price, qty in yes_raw],
        key=lambda level: level[0],
        reverse=True,
    )
    no_bids = sorted(
        [(Decimal(price), Decimal(qty)) for price, qty in no_raw],
        key=lambda level: level[0],
        reverse=True,
    )
    yes_asks = sorted([(Decimal(1) - price, qty) for price, qty in no_bids], key=lambda level: level[0])
    best_bid = yes_bids[0][0] if yes_bids else None
    best_ask = yes_asks[0][0] if yes_asks else None
    spread = best_ask - best_bid if best_bid is not None and best_ask is not None else None
    midpoint = (best_bid + best_ask) / 2 if spread is not None else None
    yes_summary = summarize_levels(yes_raw)
    no_summary = summarize_levels(no_raw)
    return {
        "best_yes_bid_dollars": number(best_bid, 4),
        "best_yes_ask_dollars": number(best_ask, 4),
        "yes_midpoint_dollars": number(midpoint, 4),
        "yes_spread_dollars": number(spread, 4),
        "yes_spread_cents": number(spread * 100 if spread is not None else None, 2),
        "best_yes_bid_size_contracts": number(yes_bids[0][1], 2) if yes_bids else None,
        "best_yes_ask_size_contracts": number(yes_asks[0][1], 2) if yes_asks else None,
        "yes_bid_book": yes_summary,
        "no_bid_book": no_summary,
        "total_resting_contracts": number(
            Decimal(str(yes_summary["contracts"])) + Decimal(str(no_summary["contracts"])), 2
        ),
        "total_resting_bid_notional_dollars": number(
            Decimal(str(yes_summary["bid_notional_dollars"]))
            + Decimal(str(no_summary["bid_notional_dollars"])),
            2,
        ),
        "yes_bid_depth": band_depth(yes_bids, best_bid, "bid"),
        "yes_ask_depth": band_depth(yes_asks, best_ask, "ask"),
        "execution_quotes": {
            "buy_yes": [execution_quote(yes_asks, size) for size in EXECUTION_SIZES],
            "sell_yes": [execution_quote(yes_bids, size) for size in EXECUTION_SIZES],
        },
        "raw_orderbook": {"yes_dollars": yes_raw, "no_dollars": no_raw},
    }


def infer_stat(series_ticker: str) -> tuple[str, str]:
    mapping = {
        "KXNFLSEASONPASSYDS": ("passing_yards", "yards"),
        "KXNFLSEASONPASSTDS": ("passing_touchdowns", "touchdowns"),
        "KXNFLSEASONRSHYDS": ("rushing_yards", "yards"),
        "KXNFLSEASONRSHTD": ("rushing_touchdowns", "touchdowns"),
        "KXNFLSEASONREC": ("receptions", "receptions"),
        "KXNFLSEASONRECYDS": ("receiving_yards", "yards"),
        "KXNFLSEASONRECTD": ("receiving_touchdowns", "touchdowns"),
    }
    return mapping[series_ticker]


def threshold_from_market(market: dict[str, Any]) -> int | float | None:
    floor = decimal_or_none(market.get("floor_strike"))
    if floor is None:
        return None
    threshold = floor + Decimal("0.5")
    return int(threshold) if threshold == threshold.to_integral() else float(threshold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/kalshi_nfl_season_stats.json"))
    args = parser.parse_args()

    series_records: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    markets_with_context: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for ticker in SERIES_TICKERS:
        series = fetch_json(f"/series/{ticker}", {"include_volume": "true"})["series"]
        series_records.append(series)
        cursor = ""
        while True:
            params = {
                "series_ticker": ticker,
                "status": "open",
                "with_nested_markets": "true",
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor
            page = fetch_json("/events", params)
            for event in page.get("events", []):
                event_without_markets = {key: value for key, value in event.items() if key != "markets"}
                event_records.append(event_without_markets)
                for market in event.get("markets", []):
                    markets_with_context.append((event_without_markets, market))
            cursor = page.get("cursor") or ""
            if not cursor:
                break

    tickers = [market["ticker"] for _, market in markets_with_context]
    orderbooks: dict[str, dict[str, Any]] = {}
    for batch in chunked(tickers, 100):
        page = fetch_json("/markets/orderbooks", [("tickers", ticker) for ticker in batch] + [("depth", 0)])
        for entry in page.get("orderbooks", []):
            orderbooks[entry["ticker"]] = entry.get("orderbook_fp") or {}

    normalized_markets: list[dict[str, Any]] = []
    for event, market in markets_with_context:
        stat, unit = infer_stat(event["series_ticker"])
        book = orderbook_metrics(orderbooks.get(market["ticker"], {}))
        best_bid = decimal_or_none(book["best_yes_bid_dollars"])
        best_ask = decimal_or_none(book["best_yes_ask_dollars"])
        normalized_markets.append(
            {
                "ticker": market["ticker"],
                "event_ticker": event["event_ticker"],
                "series_ticker": event["series_ticker"],
                "market_name": event.get("title"),
                "player": market.get("yes_sub_title") or market.get("subtitle"),
                "outcome": market.get("rules_primary"),
                "season": "2026-27",
                "stat": stat,
                "threshold_operator": ">=",
                "threshold": threshold_from_market(market),
                "threshold_unit": unit,
                "status": market.get("status"),
                "probability": probability_metrics(market, best_bid, best_ask),
                "trading_activity": {
                    "volume_contracts": number(decimal_or_none(market.get("volume_fp")), 2),
                    "volume_24h_contracts": number(decimal_or_none(market.get("volume_24h_fp")), 2),
                    "open_interest_contracts": number(decimal_or_none(market.get("open_interest_fp")), 2),
                    "legacy_liquidity_dollars": number(decimal_or_none(market.get("liquidity_dollars")), 4),
                    "legacy_liquidity_note": "Kalshi deprecated liquidity_dollars; use orderbook metrics instead.",
                },
                "orderbook": book,
                "timing": {
                    "created_time": market.get("created_time"),
                    "updated_time": market.get("updated_time"),
                    "open_time": market.get("open_time"),
                    "close_time": market.get("close_time"),
                    "expected_expiration_time": market.get("expected_expiration_time"),
                    "expiration_time": market.get("expiration_time"),
                },
                "participant": {
                    "primary_participant_key": market.get("primary_participant_key"),
                    "custom_strike": market.get("custom_strike"),
                },
                "source_urls": {
                    "market_api": f"{BASE_URL}/markets/{market['ticker']}",
                    "orderbook_api": f"{BASE_URL}/markets/{market['ticker']}/orderbook",
                    "event_api": f"{BASE_URL}/events/{event['event_ticker']}",
                },
                "raw_market": market,
            }
        )

    normalized_markets.sort(key=lambda row: (row["stat"], row["threshold"] or 0, row["player"] or ""))
    two_sided = sum(
        1
        for row in normalized_markets
        if row["orderbook"]["best_yes_bid_dollars"] is not None
        and row["orderbook"]["best_yes_ask_dollars"] is not None
    )
    one_sided = sum(
        1
        for row in normalized_markets
        if (row["orderbook"]["best_yes_bid_dollars"] is None)
        != (row["orderbook"]["best_yes_ask_dollars"] is None)
    )
    no_quote = len(normalized_markets) - two_sided - one_sided
    duplicate_tickers = len(tickers) - len(set(tickers))
    invalid_probabilities = sum(
        1
        for row in normalized_markets
        if (p := row["probability"]["estimated_chance_percent"]) is not None and not (0 <= p <= 100)
    )
    missing_books = len(set(tickers) - set(orderbooks))
    generated = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    totals = {
        "series": len(series_records),
        "events": len(event_records),
        "markets": len(normalized_markets),
        "unique_players": len({row["player"] for row in normalized_markets if row["player"]}),
        "volume_contracts": round(sum(row["trading_activity"]["volume_contracts"] or 0 for row in normalized_markets), 2),
        "volume_24h_contracts": round(
            sum(row["trading_activity"]["volume_24h_contracts"] or 0 for row in normalized_markets), 2
        ),
        "open_interest_contracts": round(
            sum(row["trading_activity"]["open_interest_contracts"] or 0 for row in normalized_markets), 2
        ),
        "resting_order_contracts": round(
            sum(row["orderbook"]["total_resting_contracts"] or 0 for row in normalized_markets), 2
        ),
        "resting_bid_notional_dollars": round(
            sum(row["orderbook"]["total_resting_bid_notional_dollars"] or 0 for row in normalized_markets), 2
        ),
    }

    output = {
        "metadata": {
            "dataset": "Kalshi NFL Season Stats markets",
            "snapshot_utc": generated,
            "scope": "All open events and nested player markets in the seven active FOOTBALLSEASONSTAT NFL series.",
            "season": "2026-27",
            "source": "Kalshi public Trade API v2",
            "source_base_url": BASE_URL,
            "source_category_url": "https://kalshi.com/category/sports/football/nfl",
            "api_documentation": {
                "markets": "https://docs.kalshi.com/api-reference/market/get-markets",
                "events": "https://docs.kalshi.com/api-reference/events/get-events",
                "orderbooks": "https://docs.kalshi.com/getting_started/orderbook_responses",
            },
            "series_tickers": SERIES_TICKERS,
            "probability_methodology": "Best YES bid/ask midpoint when both exist; otherwise last trade, then a clearly labeled one-sided quote reference.",
            "liquidity_methodology": "Full public YES/NO bid books, with YES asks derived as 1 - NO bid; includes top-of-book, spread, depth bands, and executable VWAP estimates.",
            "point_in_time_warning": "Prices, probabilities, volume, open interest, and resting orders change continuously.",
        },
        "summary": totals,
        "data_quality": {
            "intended_grain": "One row per Kalshi player contract per event threshold.",
            "market_ticker_duplicates": duplicate_tickers,
            "missing_orderbooks": missing_books,
            "markets_with_two_sided_quotes": two_sided,
            "markets_with_one_sided_quotes": one_sided,
            "markets_with_no_quotes": no_quote,
            "invalid_probability_estimates": invalid_probabilities,
            "checks_passed": duplicate_tickers == 0 and missing_books == 0 and invalid_probabilities == 0,
            "known_limitations": [
                "The snapshot covers open markets only; settled and closed historical seasons are excluded.",
                "Market prices are tradable quotes and may differ from objective probabilities, especially where spreads are wide.",
                "The deprecated liquidity_dollars field is preserved but should not be used; order-book metrics are the replacement.",
                "Aggregate volume sums contracts across threshold markets and should not be interpreted as unique bettors or unique player exposures.",
            ],
        },
        "series": series_records,
        "events": event_records,
        "markets": normalized_markets,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": totals, "data_quality": output["data_quality"]}, indent=2))


if __name__ == "__main__":
    main()
