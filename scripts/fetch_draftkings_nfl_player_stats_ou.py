#!/usr/bin/env python3
"""Fetch and normalize DraftKings NFL Player Stats O/U futures.

The DraftKings web page advertises a sports-content service in its product
configuration. This script discovers that service, fetches the seven NFL
Player Stats O/U subcategories, converts American odds to raw implied
probabilities, and removes the two-way overround with proportional
normalization.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import HTTPCookieProcessor, Request, build_opener


SPORTBOOK_ORIGIN = "https://sportsbook.draftkings.com"
TARGET_PAGE_URL = (
    f"{SPORTBOOK_ORIGIN}/leagues/football/nfl"
    "?category=futures&subcategory=player-stats-o-u"
)
LEAGUE_ID = 88808
CATEGORY_ID = 1759
TARGET_SUBCATEGORIES = {
    17147: {"stat": "passing_yards", "unit": "yards", "title": "Pass Yards"},
    17148: {"stat": "passing_touchdowns", "unit": "touchdowns", "title": "Pass TDs"},
    17314: {"stat": "receiving_yards", "unit": "yards", "title": "Rec Yards"},
    17315: {"stat": "receiving_touchdowns", "unit": "touchdowns", "title": "Rec TDs"},
    20168: {"stat": "receptions", "unit": "receptions", "title": "Receptions"},
    17223: {"stat": "rushing_yards", "unit": "yards", "title": "Rush Yards"},
    17224: {"stat": "rushing_touchdowns", "unit": "touchdowns", "title": "Rush TDs"},
}
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
LINE_PATTERN = re.compile(r"^(Over|Under)\s+([0-9]+(?:\.[0-9]+)?)$")
SEASON_PATTERN = re.compile(r"NFL\s+(\d{4})/(\d{2})")


def round_number(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


class DraftKingsClient:
    def __init__(self, retries: int = 5) -> None:
        self.retries = retries
        self.opener = build_opener(HTTPCookieProcessor(CookieJar()))
        self.referer = TARGET_PAGE_URL

    def _request(self, url: str, accept: str) -> bytes:
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": accept,
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": SPORTBOOK_ORIGIN,
            "Referer": self.referer,
            "Cache-Control": "no-cache",
            "Sec-Fetch-Dest": "empty" if accept == "application/json" else "document",
            "Sec-Fetch-Mode": "cors" if accept == "application/json" else "navigate",
            "Sec-Fetch-Site": "same-site" if accept == "application/json" else "same-origin",
        }
        request = Request(url, headers=headers)
        for attempt in range(self.retries):
            try:
                with self.opener.open(request, timeout=60) as response:
                    return response.read()
            except (HTTPError, URLError, TimeoutError):
                if attempt == self.retries - 1:
                    raise
                time.sleep(1.5 * (2**attempt))
        raise RuntimeError("unreachable")

    def fetch_text(self, url: str) -> str:
        return self._request(url, "text/html").decode("utf-8")

    def fetch_json(self, url: str) -> dict[str, Any]:
        return json.loads(self._request(url, "application/json"))


def extract_json_assignment(page: str, variable: str) -> dict[str, Any]:
    marker = f"window.{variable} = "
    start = page.find(marker)
    if start < 0:
        raise ValueError(f"Could not find {marker!r} in DraftKings page")
    value, _ = json.JSONDecoder().raw_decode(page[start + len(marker) :])
    if not isinstance(value, dict):
        raise TypeError(f"Expected {variable} to contain a JSON object")
    return value


def parse_american_odds(display_value: Any) -> int | None:
    if display_value is None:
        return None
    cleaned = str(display_value).strip().replace("−", "-").replace("–", "-")
    if cleaned.upper() in {"EVEN", "EVENS"}:
        return 100
    try:
        result = int(cleaned)
    except ValueError:
        return None
    return result if result != 0 else None


def implied_probability(american_odds: int | None) -> float | None:
    if american_odds is None:
        return None
    if american_odds > 0:
        return 100 / (american_odds + 100)
    return -american_odds / (-american_odds + 100)


def displayed_probability(selection: dict[str, Any]) -> float | None:
    value = selection.get("displayOdds", {}).get("percentage")
    if value is None:
        return None
    try:
        return float(str(value).strip().rstrip("%"))
    except ValueError:
        return None


def parse_line(selection: dict[str, Any]) -> float | int | None:
    match = LINE_PATTERN.fullmatch(str(selection.get("label") or ""))
    if not match:
        return None
    value = float(match.group(2))
    return int(value) if value.is_integer() else value


def player_and_team(event: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    participants = event.get("participants") or []
    teams = [
        participant
        for participant in participants
        if (participant.get("metadata") or {}).get("rosettaTeamId")
    ]
    people = [
        participant
        for participant in participants
        if not (participant.get("metadata") or {}).get("rosettaTeamId")
    ]
    player = people[0].get("name") if len(people) == 1 else None
    if not player:
        name = str(event.get("name") or "")
        player = name.rsplit(" - ", 1)[-1] if " - " in name else None
    team = teams[0].get("name") if teams else None
    team_abbreviation = (teams[0].get("metadata") or {}).get("shortName") if teams else None
    return player, team, team_abbreviation


def season_from_event(event: dict[str, Any]) -> str | None:
    match = SEASON_PATTERN.search(str(event.get("name") or ""))
    return f"{match.group(1)}-{match.group(2)}" if match else None


def normalize_selection(
    selection: dict[str, Any], raw_probability: float | None, no_vig_probability: float | None
) -> dict[str, Any]:
    display = selection.get("displayOdds") or {}
    return {
        "selection_id": selection.get("id"),
        "label": selection.get("label"),
        "outcome_type": selection.get("outcomeType"),
        "american_odds": parse_american_odds(display.get("american")),
        "american_odds_display": display.get("american"),
        "decimal_odds_display": display.get("decimal"),
        "fractional_odds_display": display.get("fractional"),
        "draftkings_display_implied_probability_percent": displayed_probability(selection),
        "computed_raw_implied_probability_percent": round_number(
            raw_probability * 100 if raw_probability is not None else None
        ),
        "no_vig_implied_probability_percent": round_number(
            no_vig_probability * 100 if no_vig_probability is not None else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/draftkings_nfl_player_stats_ou.json"),
    )
    args = parser.parse_args()

    client = DraftKingsClient()
    target_page = client.fetch_text(TARGET_PAGE_URL)
    product_config = extract_json_assignment(target_page, "__productConfig")
    service_base_url = str(product_config["sportsContentBff"]).rstrip("/") + "/"

    payloads: dict[int, dict[str, Any]] = {}
    source_urls: dict[int, str] = {}
    for subcategory_id in TARGET_SUBCATEGORIES:
        path = (
            f"v1/leagues/{LEAGUE_ID}/categories/{CATEGORY_ID}"
            f"/subcategories/{subcategory_id}"
        )
        source_url = urljoin(service_base_url, path)
        source_urls[subcategory_id] = source_url
        payloads[subcategory_id] = client.fetch_json(source_url)

    event_by_id: dict[str, dict[str, Any]] = {}
    event_conflicts = 0
    market_entries: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    selection_ids: list[str] = []
    market_ids: list[str] = []
    category_record: dict[str, Any] | None = None
    subcategory_records: list[dict[str, Any]] = []

    for subcategory_id, payload in payloads.items():
        if category_record is None:
            category_record = next(
                (row for row in payload.get("categories", []) if int(row.get("id")) == CATEGORY_ID),
                None,
            )
        subcategory = next(
            (
                row
                for row in payload.get("subcategories", [])
                if int(row.get("id")) == subcategory_id
            ),
            None,
        )
        if subcategory is None:
            raise ValueError(f"DraftKings omitted subcategory {subcategory_id}")
        subcategory_records.append(subcategory)

        for event in payload.get("events", []):
            event_id = str(event.get("id"))
            if event_id in event_by_id and event_by_id[event_id] != event:
                event_conflicts += 1
            else:
                event_by_id[event_id] = event

        selections_by_market: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for selection in payload.get("selections", []):
            selections_by_market[str(selection.get("marketId"))].append(selection)
            selection_ids.append(str(selection.get("id")))

        for market in payload.get("markets", []):
            market_id = str(market.get("id"))
            market_ids.append(market_id)
            market_entries.append((subcategory_id, market, selections_by_market.get(market_id, [])))

    normalized_markets: list[dict[str, Any]] = []
    missing_events = 0
    missing_over = 0
    missing_under = 0
    non_two_selection = 0
    mismatched_lines = 0
    invalid_odds = 0
    invalid_raw_probabilities = 0
    invalid_no_vig_probabilities = 0
    no_vig_pairs_not_100 = 0

    for subcategory_id, market, selections in market_entries:
        event = event_by_id.get(str(market.get("eventId")))
        if event is None:
            missing_events += 1
            event = {}
        by_outcome = {str(selection.get("outcomeType")): selection for selection in selections}
        over = by_outcome.get("Over")
        under = by_outcome.get("Under")
        missing_over += over is None
        missing_under += under is None
        non_two_selection += len(selections) != 2

        over_line = parse_line(over or {})
        under_line = parse_line(under or {})
        if over_line is None or under_line is None or over_line != under_line:
            mismatched_lines += 1
        line = over_line if over_line == under_line else None

        over_odds = parse_american_odds((over or {}).get("displayOdds", {}).get("american"))
        under_odds = parse_american_odds((under or {}).get("displayOdds", {}).get("american"))
        invalid_odds += over_odds is None
        invalid_odds += under_odds is None
        raw_over = implied_probability(over_odds)
        raw_under = implied_probability(under_odds)
        raw_total = (
            raw_over + raw_under if raw_over is not None and raw_under is not None else None
        )
        no_vig_over = raw_over / raw_total if raw_over is not None and raw_total else None
        no_vig_under = raw_under / raw_total if raw_under is not None and raw_total else None

        for probability in (raw_over, raw_under):
            invalid_raw_probabilities += probability is not None and not (0 < probability < 1)
        for probability in (no_vig_over, no_vig_under):
            invalid_no_vig_probabilities += probability is not None and not (0 < probability < 1)
        if no_vig_over is not None and no_vig_under is not None:
            no_vig_pairs_not_100 += abs(no_vig_over + no_vig_under - 1) > 1e-12

        player, team, team_abbreviation = player_and_team(event)
        stat_config = TARGET_SUBCATEGORIES[subcategory_id]
        normalized_markets.append(
            {
                "market_id": str(market.get("id")),
                "event_id": str(market.get("eventId")),
                "sport_id": str(market.get("sportId")),
                "league_id": str(market.get("leagueId")),
                "category_id": CATEGORY_ID,
                "subcategory_id": subcategory_id,
                "subcategory_name": next(
                    row.get("name")
                    for row in subcategory_records
                    if int(row.get("id")) == subcategory_id
                ),
                "market_type_id": str((market.get("marketType") or {}).get("id")),
                "market_type_name": (market.get("marketType") or {}).get("name"),
                "market_name": market.get("name"),
                "player": player,
                "team": team,
                "team_abbreviation": team_abbreviation,
                "season": season_from_event(event),
                "stat": stat_config["stat"],
                "market_line": line,
                "median_outcome_proxy": line,
                "threshold_unit": stat_config["unit"],
                "threshold_operators": {"over": ">", "under": "<"},
                "status": event.get("status"),
                "probability": {
                    "over": normalize_selection(over or {}, raw_over, no_vig_over),
                    "under": normalize_selection(under or {}, raw_under, no_vig_under),
                    "two_way_raw_implied_total_percent": round_number(
                        raw_total * 100 if raw_total is not None else None
                    ),
                    "two_way_overround_percentage_points": round_number(
                        (raw_total - 1) * 100 if raw_total is not None else None
                    ),
                    "no_vig_method": "Proportional normalization of the Over and Under raw implied probabilities.",
                    "note": "The no-vig values remove this market's two-way overround; they are not objective forecasts.",
                },
                "timing": {
                    "start_event_date": event.get("startEventDate"),
                    "settlement_date": (event.get("metadata") or {}).get("settlementDate"),
                },
                "source_urls": {
                    "sportsbook_page": TARGET_PAGE_URL,
                    "subcategory_service": source_urls[subcategory_id],
                },
                "raw_market": market,
                "raw_selections": selections,
            }
        )

    normalized_markets.sort(
        key=lambda row: (row["stat"], row["player"] or "", row["market_id"])
    )
    unique_players = {row["player"] for row in normalized_markets if row["player"]}
    markets_by_stat = Counter(row["stat"] for row in normalized_markets)
    players_by_stat: dict[str, set[str]] = defaultdict(set)
    for row in normalized_markets:
        if row["player"]:
            players_by_stat[row["stat"]].add(row["player"])

    duplicate_market_ids = len(market_ids) - len(set(market_ids))
    duplicate_selection_ids = len(selection_ids) - len(set(selection_ids))
    fatal_check_values = [
        duplicate_market_ids,
        duplicate_selection_ids,
        missing_events,
        missing_over,
        missing_under,
        non_two_selection,
        mismatched_lines,
        invalid_odds,
        invalid_raw_probabilities,
        invalid_no_vig_probabilities,
        no_vig_pairs_not_100,
        event_conflicts,
    ]
    generated = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    output = {
        "metadata": {
            "dataset": "DraftKings NFL Player Stats O/U futures",
            "snapshot_utc": generated,
            "scope": "All markets returned by the seven NFL Futures > Player Stats O/U subcategories.",
            "season": "2026-27",
            "source": "DraftKings Sportsbook sports-content service advertised by the web application",
            "source_page_url": TARGET_PAGE_URL,
            "source_service_base_url": service_base_url,
            "league_id": LEAGUE_ID,
            "category_id": CATEGORY_ID,
            "subcategory_ids": list(TARGET_SUBCATEGORIES),
            "probability_methodology": {
                "raw_implied_probability": "Positive American odds A: 100 / (A + 100); negative American odds A: -A / (-A + 100).",
                "no_vig_probability": "Each side's raw implied probability divided by the sum for Over and Under.",
            },
            "median_proxy_methodology": "market_line is copied to median_outcome_proxy. It is only an approximate median when the no-vig Over/Under probabilities are not exactly 50%/50%.",
            "point_in_time_warning": "Odds, lines, available players, and market availability can change continuously and can vary by jurisdiction.",
            "service_stability_warning": "The sports-content endpoint is used by DraftKings' web application but is not documented as a stable public API.",
        },
        "summary": {
            "subcategories": len(TARGET_SUBCATEGORIES),
            "events": len(event_by_id),
            "markets": len(normalized_markets),
            "selections": len(selection_ids),
            "unique_players": len(unique_players),
            "markets_by_stat": dict(sorted(markets_by_stat.items())),
            "unique_players_by_stat": {
                stat: len(players) for stat, players in sorted(players_by_stat.items())
            },
        },
        "data_quality": {
            "intended_grain": "One row per DraftKings player/stat O/U market, with Over and Under selections nested in the probability block.",
            "market_id_duplicates": duplicate_market_ids,
            "selection_id_duplicates": duplicate_selection_ids,
            "conflicting_duplicate_event_records": event_conflicts,
            "markets_missing_event": missing_events,
            "markets_missing_over": missing_over,
            "markets_missing_under": missing_under,
            "markets_without_exactly_two_selections": non_two_selection,
            "markets_with_mismatched_or_unparsed_lines": mismatched_lines,
            "selections_with_invalid_american_odds": invalid_odds,
            "invalid_raw_implied_probabilities": invalid_raw_probabilities,
            "invalid_no_vig_probabilities": invalid_no_vig_probabilities,
            "no_vig_pairs_not_summing_to_100_percent": no_vig_pairs_not_100,
            "checks_passed": not any(fatal_check_values),
            "known_limitations": [
                "The snapshot contains only markets currently returned by DraftKings for this site experience and jurisdiction.",
                "Sportsbook prices include margin; use no_vig_implied_probability_percent for a proportional two-way normalization.",
                "A posted O/U line is a median proxy, not a complete player-outcome distribution.",
                "The no-vig conversion removes the two-way overround mechanically and does not eliminate every form of bookmaker bias.",
                "Start and settlement dates are copied from DraftKings event metadata and do not indicate when a price first became available.",
            ],
        },
        "category": category_record,
        "subcategories": sorted(subcategory_records, key=lambda row: int(row["id"])),
        "events": sorted(event_by_id.values(), key=lambda row: str(row.get("id"))),
        "markets": normalized_markets,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **output["summary"], "checks_passed": output["data_quality"]["checks_passed"]}, indent=2))


if __name__ == "__main__":
    main()
