#!/usr/bin/env python3
"""Fetch and normalize FanDuel's season-long NFL player props.

FanDuel's NFL custom page has a Player Props tab backed by its public web
application's ``content-managed-page`` service.  This script discovers the
cards currently assigned to that tab, follows their coupon-to-market links,
and emits DraftKings-compatible O/U records with raw and no-vig implied
probabilities.  Non-O/U player markets on the tab are preserved separately.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SPORTSBOOK_ORIGIN = "https://sportsbook.fanduel.com"
TARGET_PAGE_URL = f"{SPORTSBOOK_ORIGIN}/navigation/nfl?tab=player-props"
CUSTOM_PAGE_ID = "nfl"
TARGET_TAB_TITLE = "Player Props"
EVENT_TYPE_ID = 6423

# This is the public application key sent by FanDuel's sportsbook web client.
# It is not an account credential.  --api-key can override it if FanDuel rotates
# the web-client key before this script is updated.
DEFAULT_PUBLIC_API_KEY = "FhMFpcPWXMeyZxOx"

STAT_CARDS = {
    "Regular Season Passing Yards": {"stat": "passing_yards", "unit": "yards"},
    "Regular Season Passing TDs": {
        "stat": "passing_touchdowns",
        "unit": "touchdowns",
    },
    "Regular Season Rushing Yards": {"stat": "rushing_yards", "unit": "yards"},
    "Regular Season Rushing TDs": {
        "stat": "rushing_touchdowns",
        "unit": "touchdowns",
    },
    "Regular Season Receiving Yards": {
        "stat": "receiving_yards",
        "unit": "yards",
    },
    "Regular Season Receiving TDs": {
        "stat": "receiving_touchdowns",
        "unit": "touchdowns",
    },
    "Regular Season Receiving Touchdowns": {
        "stat": "receiving_touchdowns",
        "unit": "touchdowns",
    },
    "Regular Season Receptions": {"stat": "receptions", "unit": "receptions"},
}
OTHER_CARD_CONFIG = {
    "Most Regular Season Rookie Receiving Yards": {
        "market_kind": "multiway_outright",
        "stat": "rookie_receiving_yards_leader",
        "unit": "player",
    }
}

RUNNER_PATTERN = re.compile(
    r"^(?P<player>.+)\s+(?P<outcome>Over|Under)\s+(?P<line>-?[0-9]+(?:\.[0-9]+)?)$"
)
SEASON_PATTERN = re.compile(r"\b(20\d{2}-\d{2})\b")
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def round_number(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


def number_from_text(value: str) -> float | int:
    number = float(value)
    return int(number) if number.is_integer() else number


def record(mapping: Any, key: Any) -> dict[str, Any] | None:
    if not isinstance(mapping, dict):
        return None
    value = mapping.get(str(key))
    if value is None:
        value = mapping.get(key)
    return value if isinstance(value, dict) else None


def mapping_values(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    return [row for row in value.values() if isinstance(row, dict)]


def american_odds(runner: dict[str, Any]) -> int | None:
    display = (runner.get("winRunnerOdds") or {}).get("americanDisplayOdds") or {}
    value = display.get("americanOddsInt")
    if value is None:
        value = display.get("americanOdds")
    if value is None:
        return None
    try:
        result = int(str(value).strip().replace("+", "").replace("−", "-"))
    except ValueError:
        return None
    return result if result != 0 else None


def decimal_odds(runner: dict[str, Any]) -> float | None:
    value = (
        (((runner.get("winRunnerOdds") or {}).get("trueOdds") or {}).get("decimalOdds") or {})
        .get("decimalOdds")
    )
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 1 else None


def fractional_odds_display(runner: dict[str, Any]) -> str | None:
    fractional = (
        ((runner.get("winRunnerOdds") or {}).get("trueOdds") or {}).get("fractionalOdds") or {}
    )
    numerator = fractional.get("numerator")
    denominator = fractional.get("denominator")
    if numerator is None or denominator in (None, 0):
        return None
    return f"{numerator}/{denominator}"


def implied_probability(odds: int | None) -> float | None:
    if odds is None:
        return None
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)


def parse_runner(runner: dict[str, Any]) -> tuple[str | None, str | None, float | int | None]:
    match = RUNNER_PATTERN.fullmatch(str(runner.get("runnerName") or "").strip())
    if not match:
        return None, None, None
    return match.group("player"), match.group("outcome"), number_from_text(match.group("line"))


def player_and_season(market_name: Any, card_title: str) -> tuple[str | None, str | None]:
    name = str(market_name or "").strip()
    marker = f" {card_title} "
    if marker in name:
        player, suffix = name.rsplit(marker, 1)
        season_match = SEASON_PATTERN.search(suffix)
        return player or None, season_match.group(1) if season_match else None
    season_match = SEASON_PATTERN.search(name)
    return None, season_match.group(1) if season_match else None


def format_american(odds: int | None) -> str | None:
    if odds is None:
        return None
    return f"+{odds}" if odds > 0 else str(odds)


def normalize_runner(
    runner: dict[str, Any], raw_probability: float | None, no_vig_probability: float | None
) -> dict[str, Any]:
    odds = american_odds(runner)
    decimal = decimal_odds(runner)
    _, outcome, _ = parse_runner(runner)
    return {
        "selection_id": str(runner.get("selectionId")),
        "label": runner.get("runnerName"),
        "outcome_type": outcome,
        "runner_status": runner.get("runnerStatus"),
        "american_odds": odds,
        "american_odds_display": format_american(odds),
        "decimal_odds": round_number(decimal, 12),
        "decimal_odds_display": f"{decimal:.2f}" if decimal is not None else None,
        "fractional_odds_display": fractional_odds_display(runner),
        "computed_raw_implied_probability_percent": round_number(
            raw_probability * 100 if raw_probability is not None else None
        ),
        "no_vig_implied_probability_percent": round_number(
            no_vig_probability * 100 if no_vig_probability is not None else None
        ),
    }


def source_service_url(state: str, api_key: str, timezone_name: str) -> str:
    params = urlencode(
        {
            "page": "CUSTOM",
            "customPageId": CUSTOM_PAGE_ID,
            "tab": TARGET_TAB_TITLE,
            "_ak": api_key,
            "timezone": timezone_name,
        }
    )
    return (
        f"https://sbapi.{state.lower()}.sportsbook.fanduel.com"
        f"/api/content-managed-page?{params}"
    )


def fetch_payload(url: str, retries: int = 5) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": SPORTSBOOK_ORIGIN,
            "Referer": TARGET_PAGE_URL,
            "Cache-Control": "no-cache",
        },
    )
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=60) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise TypeError("FanDuel response was not a JSON object")
            return payload
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (2**attempt))
    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/fanduel_nfl_player_season_props.json"),
    )
    parser.add_argument(
        "--state",
        default="nj",
        help="FanDuel sportsbook state subdomain/jurisdiction (default: nj).",
    )
    parser.add_argument(
        "--timezone",
        default="America/New_York",
        help="Timezone sent to FanDuel's content service.",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_PUBLIC_API_KEY,
        help="FanDuel public web-client API key.",
    )
    args = parser.parse_args()

    service_url = source_service_url(args.state, args.api_key, args.timezone)
    payload = fetch_payload(service_url)
    layout = payload.get("layout") or {}
    attachments = payload.get("attachments") or {}
    tabs = layout.get("tabs") or {}
    cards_by_id = layout.get("cards") or {}
    coupons_by_id = layout.get("coupons") or {}
    markets_by_id = attachments.get("markets") or {}
    events_by_id = attachments.get("events") or {}
    competitions_by_id = attachments.get("competitions") or {}

    target_tab = next(
        (tab for tab in mapping_values(tabs) if tab.get("title") == TARGET_TAB_TITLE),
        None,
    )
    if target_tab is None:
        raise ValueError(f"FanDuel page omitted the {TARGET_TAB_TITLE!r} tab")

    target_cards: list[dict[str, Any]] = []
    missing_cards = 0
    for card_ref in target_tab.get("cards") or []:
        card = record(cards_by_id, card_ref.get("id"))
        if card is None:
            missing_cards += 1
        else:
            target_cards.append(card)

    coupon_contexts: list[tuple[dict[str, Any], dict[str, Any]]] = []
    missing_coupons = 0
    unavailable_coupon_placeholders = 0
    unexpected_markets_missing_from_attachments = 0
    target_market_contexts: list[
        tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
    ] = []
    unavailable_player_prop_references: list[dict[str, Any]] = []
    raw_coupons: list[dict[str, Any]] = []
    market_ids: list[str] = []
    unclassified_card_titles: list[str] = []

    for card in target_cards:
        card_title = str(card.get("title") or "")
        if card_title not in STAT_CARDS and card_title not in OTHER_CARD_CONFIG:
            unclassified_card_titles.append(card_title)
        for coupon_ref in card.get("coupons") or []:
            coupon = record(coupons_by_id, coupon_ref.get("id"))
            if coupon is None:
                missing_coupons += 1
                continue
            coupon_contexts.append((card, coupon))
            raw_coupons.append(
                {"card_id": card.get("id"), "card_title": card_title, **coupon}
            )
            market_id = str(coupon.get("marketId") or "")
            market = record(markets_by_id, market_id)
            if not market_id or market is None:
                expected_placeholder = not bool(coupon.get("hasAttachments")) and not market_id
                unavailable_coupon_placeholders += expected_placeholder
                unexpected_markets_missing_from_attachments += not expected_placeholder
                unavailable_player_prop_references.append(
                    {
                        "reason": (
                            "coupon_placeholder_has_no_attached_market"
                            if expected_placeholder
                            else "referenced_market_missing_from_attachments"
                        ),
                        "card_id": card.get("id"),
                        "card_title": card_title,
                        "coupon_id": coupon.get("id"),
                        "market_id": market_id or None,
                        "raw_coupon": coupon,
                    }
                )
                continue
            market_ids.append(market_id)
            target_market_contexts.append((card, coupon, market))

    normalized_markets: list[dict[str, Any]] = []
    other_player_markets: list[dict[str, Any]] = []
    selection_ids: list[str] = []
    referenced_event_ids: set[str] = set()
    referenced_competition_ids: set[str] = set()

    missing_events = 0
    missing_competitions = 0
    missing_over = 0
    missing_under = 0
    non_two_selection = 0
    mismatched_lines = 0
    player_name_mismatches = 0
    invalid_odds = 0
    invalid_raw_probabilities = 0
    invalid_no_vig_probabilities = 0
    no_vig_pairs_not_100 = 0
    decimal_american_probability_mismatches = 0
    empty_attached_markets = 0

    for card, coupon, market in target_market_contexts:
        card_title = str(card.get("title") or "")
        market_id = str(market.get("marketId"))
        event_id = str(market.get("eventId"))
        competition_id = str(market.get("competitionId"))
        referenced_event_ids.add(event_id)
        referenced_competition_ids.add(competition_id)
        event = record(events_by_id, event_id)
        competition = record(competitions_by_id, competition_id)
        missing_events += event is None
        missing_competitions += competition is None
        runners = market.get("runners") or []
        selection_ids.extend(str(runner.get("selectionId")) for runner in runners)

        if not runners:
            empty_attached_markets += 1
            unavailable_player_prop_references.append(
                {
                    "reason": "attached_market_has_no_offered_runners",
                    "card_id": card.get("id"),
                    "card_title": card_title,
                    "coupon_id": coupon.get("id"),
                    "market_id": market_id,
                    "raw_coupon": coupon,
                    "raw_market": market,
                }
            )
            continue

        if card_title not in STAT_CARDS:
            probabilities = [implied_probability(american_odds(runner)) for runner in runners]
            raw_total = sum(value for value in probabilities if value is not None)
            valid_total = raw_total if all(value is not None for value in probabilities) else None
            selections: list[dict[str, Any]] = []
            for runner, raw_probability in zip(runners, probabilities):
                odds = american_odds(runner)
                decimal = decimal_odds(runner)
                normalized_probability = (
                    raw_probability / valid_total
                    if raw_probability is not None and valid_total
                    else None
                )
                invalid_odds += odds is None
                selections.append(
                    {
                        "selection_id": str(runner.get("selectionId")),
                        "player": runner.get("runnerName"),
                        "label": runner.get("runnerName"),
                        "runner_status": runner.get("runnerStatus"),
                        "american_odds": odds,
                        "american_odds_display": format_american(odds),
                        "decimal_odds": round_number(decimal, 12),
                        "decimal_odds_display": (
                            f"{decimal:.2f}" if decimal is not None else None
                        ),
                        "fractional_odds_display": fractional_odds_display(runner),
                        "computed_raw_implied_probability_percent": round_number(
                            raw_probability * 100 if raw_probability is not None else None
                        ),
                        "proportional_no_vig_implied_probability_percent": round_number(
                            normalized_probability * 100
                            if normalized_probability is not None
                            else None
                        ),
                    }
                )
            season_match = SEASON_PATTERN.search(str(market.get("marketName") or ""))
            other_config = OTHER_CARD_CONFIG.get(card_title) or {
                "market_kind": "unclassified",
                "stat": None,
                "unit": None,
            }
            other_player_markets.append(
                {
                    "market_id": market_id,
                    "event_id": event_id,
                    "event_type_id": str(market.get("eventTypeId")),
                    "competition_id": competition_id,
                    "competition_name": (competition or {}).get("name"),
                    "card_id": card.get("id"),
                    "card_title": card_title,
                    "coupon_id": coupon.get("id"),
                    "external_market_id": coupon.get("externalMarketId"),
                    "market_kind": other_config["market_kind"],
                    "market_name": market.get("marketName"),
                    "season": season_match.group(1) if season_match else None,
                    "stat": other_config["stat"],
                    "threshold_unit": other_config["unit"],
                    "market_type": market.get("marketType"),
                    "status": market.get("marketStatus"),
                    "timing": {
                        "market_time": market.get("marketTime"),
                        "generic_event_open_date": (event or {}).get("openDate"),
                    },
                    "probability": {
                        "selections": selections,
                        "multiway_raw_implied_total_percent": round_number(
                            valid_total * 100 if valid_total is not None else None
                        ),
                        "multiway_overround_percentage_points": round_number(
                            (valid_total - 1) * 100 if valid_total is not None else None
                        ),
                        "no_vig_method": "Proportional normalization across every active runner.",
                        "note": "The normalized values remove the displayed multiway overround mechanically; they are not objective forecasts.",
                    },
                    "source_urls": {
                        "sportsbook_page": TARGET_PAGE_URL,
                        "content_service": service_url,
                    },
                    "raw_market": {key: value for key, value in market.items() if key != "runners"},
                    "raw_runners": runners,
                }
            )
            continue

        non_two_selection += len(runners) != 2
        parsed = [(*parse_runner(runner), runner) for runner in runners]
        by_outcome = {
            str(outcome): (runner_player, line, runner)
            for runner_player, outcome, line, runner in parsed
            if outcome
        }
        over_context = by_outcome.get("Over")
        under_context = by_outcome.get("Under")
        missing_over += over_context is None
        missing_under += under_context is None
        over_line = over_context[1] if over_context else None
        under_line = under_context[1] if under_context else None
        if over_line is None or under_line is None or over_line != under_line:
            mismatched_lines += 1
        line = over_line if over_line == under_line else None

        over = over_context[2] if over_context else {}
        under = under_context[2] if under_context else {}
        over_odds = american_odds(over)
        under_odds = american_odds(under)
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
        for runner, probability in ((over, raw_over), (under, raw_under)):
            decimal = decimal_odds(runner)
            if probability is not None and decimal is not None:
                decimal_american_probability_mismatches += abs(probability - 1 / decimal) > 1e-9

        player, season = player_and_season(market.get("marketName"), card_title)
        runner_players = {
            context[0] for context in (over_context, under_context) if context and context[0]
        }
        player_name_mismatches += bool(
            player is None or len(runner_players) != 1 or player not in runner_players
        )
        stat_config = STAT_CARDS[card_title]
        normalized_markets.append(
            {
                "market_id": market_id,
                "event_id": event_id,
                "event_type_id": str(market.get("eventTypeId")),
                "competition_id": competition_id,
                "competition_name": (competition or {}).get("name"),
                "card_id": card.get("id"),
                "card_title": card_title,
                "coupon_id": coupon.get("id"),
                "external_market_id": coupon.get("externalMarketId"),
                "market_type": market.get("marketType"),
                "market_name": market.get("marketName"),
                "player": player,
                "team": None,
                "team_abbreviation": None,
                "season": season,
                "stat": stat_config["stat"],
                "market_line": line,
                "median_outcome_proxy": line,
                "threshold_unit": stat_config["unit"],
                "threshold_operators": {"over": ">", "under": "<"},
                "status": market.get("marketStatus"),
                "probability": {
                    "over": normalize_runner(over, raw_over, no_vig_over),
                    "under": normalize_runner(under, raw_under, no_vig_under),
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
                    "market_time": market.get("marketTime"),
                    "generic_event_open_date": (event or {}).get("openDate"),
                },
                "source_urls": {
                    "sportsbook_page": TARGET_PAGE_URL,
                    "content_service": service_url,
                },
                "raw_market": {key: value for key, value in market.items() if key != "runners"},
                "raw_runners": runners,
            }
        )

    normalized_markets.sort(
        key=lambda row: (row["stat"], row["player"] or "", row["market_id"])
    )
    other_player_markets.sort(key=lambda row: (row["card_title"], row["market_id"]))
    markets_by_stat = Counter(row["stat"] for row in normalized_markets)
    players_by_stat: dict[str, set[str]] = defaultdict(set)
    for row in normalized_markets:
        if row["player"]:
            players_by_stat[row["stat"]].add(row["player"])
    unique_ou_players = {row["player"] for row in normalized_markets if row["player"]}
    other_players = {
        selection["player"]
        for row in other_player_markets
        for selection in row["probability"]["selections"]
        if selection["player"]
    }

    selected_events: list[dict[str, Any]] = []
    for event_id in referenced_event_ids:
        event = record(events_by_id, event_id)
        if event is not None:
            selected_events.append(event)
    selected_competitions: list[dict[str, Any]] = []
    for competition_id in referenced_competition_ids:
        competition = record(competitions_by_id, competition_id)
        if competition is not None:
            selected_competitions.append(competition)

    duplicate_market_ids = len(market_ids) - len(set(market_ids))
    duplicate_selection_ids = len(selection_ids) - len(set(selection_ids))
    cards_without_attachments = sum(not bool(card.get("hasAttachments")) for card in target_cards)
    seasons = sorted(
        {
            row["season"]
            for row in [*normalized_markets, *other_player_markets]
            if row.get("season")
        }
    )
    fatal_check_values = [
        missing_cards,
        missing_coupons,
        unexpected_markets_missing_from_attachments,
        duplicate_market_ids,
        duplicate_selection_ids,
        cards_without_attachments,
        len(unclassified_card_titles),
        missing_events,
        missing_competitions,
        missing_over,
        missing_under,
        non_two_selection,
        mismatched_lines,
        player_name_mismatches,
        invalid_odds,
        invalid_raw_probabilities,
        invalid_no_vig_probabilities,
        no_vig_pairs_not_100,
        decimal_american_probability_mismatches,
    ]
    generated = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    output = {
        "metadata": {
            "dataset": "FanDuel NFL season-long player props",
            "snapshot_utc": generated,
            "scope": "All attached markets on FanDuel's NFL > Player Props tab; two-way season-stat O/U markets are normalized in markets and other player markets are preserved separately.",
            "season": seasons[0] if len(seasons) == 1 else seasons,
            "source": "FanDuel Sportsbook content-managed-page service used by the web application",
            "source_page_url": TARGET_PAGE_URL,
            "source_service_url": service_url,
            "jurisdiction_state": args.state.upper(),
            "timezone": args.timezone,
            "event_type_id": EVENT_TYPE_ID,
            "custom_page_id": CUSTOM_PAGE_ID,
            "tab_id": target_tab.get("id"),
            "tab_title": target_tab.get("title"),
            "player_prop_card_ids": [card.get("id") for card in target_cards],
            "player_prop_card_titles": [card.get("title") for card in target_cards],
            "probability_methodology": {
                "raw_implied_probability": "Positive American odds A: 100 / (A + 100); negative American odds A: -A / (-A + 100).",
                "two_way_no_vig_probability": "Each side's raw implied probability divided by the sum for Over and Under.",
                "multiway_no_vig_probability": "Each runner's raw implied probability divided by the sum across all active runners.",
            },
            "median_proxy_methodology": "market_line is copied to median_outcome_proxy. It is only an approximate median when the no-vig Over/Under probabilities are not exactly 50%/50%.",
            "point_in_time_warning": "Odds, lines, available players, market availability, and even the available card set can change continuously and can vary by jurisdiction.",
            "service_stability_warning": "The content-managed-page endpoint is used by FanDuel's web application but is not documented as a stable public API.",
        },
        "summary": {
            "cards": len(target_cards),
            "coupons": len(coupon_contexts),
            "market_references_on_tab": len(coupon_contexts),
            "player_markets_total": len(normalized_markets) + len(other_player_markets),
            "markets": len(normalized_markets),
            "other_player_markets": len(other_player_markets),
            "unavailable_player_prop_references": len(unavailable_player_prop_references),
            "selections": sum(len(row.get("runners") or []) for _, _, row in target_market_contexts),
            "unique_players_in_ou_markets": len(unique_ou_players),
            "unique_players_in_other_markets": len(other_players),
            "markets_by_stat": dict(sorted(markets_by_stat.items())),
            "unique_players_by_stat": {
                stat: len(players) for stat, players in sorted(players_by_stat.items())
            },
        },
        "data_quality": {
            "intended_grain": "One row per FanDuel player/stat O/U market in markets, with Over and Under selections nested in the probability block; non-O/U tab markets are stored in other_player_markets.",
            "target_tab_cards_missing": missing_cards,
            "target_card_coupons_missing": missing_coupons,
            "unavailable_coupon_placeholders_without_market": unavailable_coupon_placeholders,
            "unexpected_markets_missing_from_attachments": unexpected_markets_missing_from_attachments,
            "attached_markets_without_offered_runners": empty_attached_markets,
            "target_cards_without_attachments": cards_without_attachments,
            "unclassified_target_card_titles": unclassified_card_titles,
            "market_id_duplicates": duplicate_market_ids,
            "selection_id_duplicates": duplicate_selection_ids,
            "markets_missing_event": missing_events,
            "markets_missing_competition": missing_competitions,
            "markets_missing_over": missing_over,
            "markets_missing_under": missing_under,
            "ou_markets_without_exactly_two_selections": non_two_selection,
            "markets_with_mismatched_or_unparsed_lines": mismatched_lines,
            "markets_with_player_name_mismatch": player_name_mismatches,
            "selections_with_invalid_american_odds": invalid_odds,
            "invalid_raw_implied_probabilities": invalid_raw_probabilities,
            "invalid_no_vig_probabilities": invalid_no_vig_probabilities,
            "no_vig_pairs_not_summing_to_100_percent": no_vig_pairs_not_100,
            "american_decimal_probability_mismatches": decimal_american_probability_mismatches,
            "checks_passed": not any(fatal_check_values),
            "known_limitations": [
                "The snapshot contains only markets currently returned for the selected FanDuel site experience and state jurisdiction.",
                "FanDuel can retain empty coupon slots or attached markets without offered runners; these are preserved in unavailable_player_prop_references and excluded from projection-ready markets.",
                "FanDuel's season-specials feed does not provide player team metadata, so team and team_abbreviation are null rather than inferred from another source.",
                "The generic NFL Specials event date is a container value and is not a reliable settlement date; it is preserved as generic_event_open_date only.",
                "Sportsbook prices include margin; use no_vig_implied_probability_percent for proportional two-way normalization.",
                "A posted O/U line is a median proxy, not a complete player-outcome distribution.",
                "The no-vig conversion removes displayed overround mechanically and does not eliminate every form of bookmaker bias.",
            ],
        },
        "event_type": record(attachments.get("eventTypes") or {}, EVENT_TYPE_ID),
        "competitions": sorted(
            selected_competitions, key=lambda row: str(row.get("competitionId"))
        ),
        "events": sorted(selected_events, key=lambda row: str(row.get("eventId"))),
        "cards": target_cards,
        "coupons": raw_coupons,
        "markets": normalized_markets,
        "other_player_markets": other_player_markets,
        "unavailable_player_prop_references": unavailable_player_prop_references,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                **output["summary"],
                "checks_passed": output["data_quality"]["checks_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
