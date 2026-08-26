from __future__ import annotations

from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ff_market_projections.config import load_config
from ff_market_projections.markets import normalize_markets, validate_collections
from ff_market_projections.odds import OddsError, american_implied_probability, american_to_decimal, decimal_to_american, implied_probability, proportional_devig
from ff_market_projections.pricing import PricingError, canonical_sportsbook_threshold, price_markets
from ff_market_projections.runs import initialize_run


ROOT = Path(__file__).parents[1]


def _config() -> dict:
    return deepcopy(load_config(ROOT / "config" / "pipeline.toml").values)


def _raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"; raw.mkdir()
    for source in ("draftkings", "fanduel", "kalshi"):
        (raw / f"{source}.json").write_bytes((ROOT / "tests" / "fixtures" / f"{source}.json").read_bytes())
    return raw


def _rows(tmp_path: Path) -> list[dict]:
    from datetime import datetime, timezone
    return normalize_markets(validate_collections(_raw(tmp_path), _config(), now=datetime(2026, 8, 25, 1, tzinfo=timezone.utc)), "fixture-run")


@pytest.mark.parametrize(("odds", "decimal"), [(150, 2.5), (-200, 1.5), ("EVEN", 2.0)])
def test_american_odds_conversions_are_pure_and_round_trip(odds: int | str, decimal: float) -> None:
    assert american_to_decimal(odds) == pytest.approx(decimal)
    assert implied_probability(decimal) == pytest.approx(1 / decimal)
    assert american_implied_probability(odds) == pytest.approx(1 / decimal)
    if isinstance(odds, int):
        assert decimal_to_american(decimal) == pytest.approx(odds)


def test_invalid_american_odds_fail_closed() -> None:
    with pytest.raises(OddsError, match="zero"):
        american_to_decimal(0)
    with pytest.raises(OddsError):
        american_to_decimal("+110")


def test_known_two_way_proportional_devig_sums_to_one() -> None:
    over, under = proportional_devig(0.60, 0.50)
    assert over == pytest.approx(6 / 11)
    assert under == pytest.approx(5 / 11)
    assert over + under == pytest.approx(1.0)


def test_unimplemented_devig_methods_are_rejected_by_configuration(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.toml"
    config_path.write_text((ROOT / "config" / "pipeline.toml").read_text().replace('sportsbook_devig_method = "proportional"', 'sportsbook_devig_method = "power"'))
    with pytest.raises(Exception, match="sportsbook_devig_method"):
        load_config(config_path)


@pytest.mark.parametrize(("line", "expected"), [(0.5, 1), (1.5, 2), (3499.5, 3500), (3500.5, 3501)])
def test_half_point_sportsbook_lines_convert_to_canonical_ge_threshold(line: float, expected: int) -> None:
    assert canonical_sportsbook_threshold(line, reject_integer_lines=True) == expected


def test_integer_sportsbook_lines_fail_when_settlement_is_ambiguous(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    for row in rows:
        if row["source"] == "draftkings":
            row["threshold"] = 3500.0
    with pytest.raises(PricingError, match="Integer sportsbook lines"):
        price_markets(rows, _config())


def test_pricing_recomputes_no_vig_probabilities_and_preserves_under_lineage(tmp_path: Path) -> None:
    priced = price_markets(_rows(tmp_path), _config())
    sportsbook = next(row for row in priced.rows if row["source"] == "draftkings")
    assert sportsbook["canonical_event"] == "P(X >= k)"
    assert sportsbook["canonical_threshold"] == 3501
    assert sportsbook["decimal_odds"] == pytest.approx(1 + 100 / 110)
    assert sportsbook["raw_implied_probability"] == pytest.approx(110 / 210)
    assert sportsbook["market_overround"] == pytest.approx(220 / 210)
    assert sportsbook["no_vig_probability"] == pytest.approx(0.5)
    assert sportsbook["paired_under_no_vig_probability"] == pytest.approx(0.5)
    assert sportsbook["no_vig_probability"] + sportsbook["paired_under_no_vig_probability"] == pytest.approx(1.0)
    assert sportsbook["paired_under_selection_id"] == "under-redacted"
    assert priced.validation["status"] == "passed"


def test_kalshi_uses_yes_midpoint_without_sportsbook_devig(tmp_path: Path) -> None:
    priced = price_markets(_rows(tmp_path), _config())
    kalshi = next(row for row in priced.rows if row["source"] == "kalshi")
    assert kalshi["canonical_threshold"] == 900
    assert kalshi["kalshi_bid_probability"] == pytest.approx(0.51)
    assert kalshi["kalshi_ask_probability"] == pytest.approx(0.53)
    assert kalshi["modeling_probability"] == pytest.approx(0.52)
    assert kalshi["no_vig_probability"] == pytest.approx(0.52)
    assert kalshi["market_overround"] is None
    assert kalshi["devig_method"] == "not_applicable_kalshi"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda row: row.update(yes_ask_probability=None), "missing_two_sided_yes_quote"),
        (lambda row: row.update(yes_ask_probability=0.80), "yes_spread_exceeds_configured_limit"),
    ],
)
def test_one_sided_or_wide_kalshi_quotes_are_excluded_auditably(tmp_path: Path, mutation, reason: str) -> None:
    rows = _rows(tmp_path)
    kalshi = next(row for row in rows if row["source"] == "kalshi")
    mutation(kalshi)
    priced = price_markets(rows, _config())
    excluded = next(row for row in priced.rows if row["source"] == "kalshi")
    assert excluded["inclusion_status"] == "excluded"
    assert excluded["exclusion_reason"] == reason
    assert excluded["modeling_probability"] is None
    assert priced.validation["status"] == "passed"


def test_pricing_cli_writes_run_scoped_artifacts(tmp_path: Path) -> None:
    config = load_config(ROOT / "config" / "pipeline.toml")
    run_dir = initialize_run(config, ROOT / "config" / "player_aliases.csv", tmp_path / "runs")
    source_rows = _rows(tmp_path)
    with (run_dir / "artifacts" / "normalized_markets.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0])); writer.writeheader(); writer.writerows(source_rows)
    completed = subprocess.run([sys.executable, str(ROOT / "scripts" / "price_markets.py"), "--run-dir", str(run_dir)], text=True, capture_output=True, check=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert json.loads(completed.stdout)["state"] == "succeeded"
    with (run_dir / "artifacts" / "priced_markets.csv").open(newline="") as handle:
        priced_rows = list(csv.DictReader(handle))
    assert len(priced_rows) == 3
    assert {row["source"] for row in priced_rows} == {"draftkings", "fanduel", "kalshi"}
    assert json.loads((run_dir / "artifacts" / "pricing_validation.json").read_text())["status"] == "passed"
