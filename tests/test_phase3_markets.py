from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ff_market_projections.config import load_config
from ff_market_projections.markets import MarketValidationError, normalize_markets, validate_collections, validate_normalized
from ff_market_projections.runs import initialize_run


ROOT = Path(__file__).parents[1]
NOW = datetime(2026, 8, 25, 1, tzinfo=timezone.utc)


def _config() -> dict:
    return deepcopy(load_config(ROOT / "config/pipeline.toml").values)


def _raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"; raw.mkdir()
    for source in ("draftkings", "fanduel", "kalshi"):
        (raw / f"{source}.json").write_bytes((ROOT / "tests" / "fixtures" / f"{source}.json").read_bytes())
    return raw


def _payload(raw: Path, source: str) -> dict:
    return json.loads((raw / f"{source}.json").read_text())


def _write(raw: Path, source: str, payload: dict) -> None:
    (raw / f"{source}.json").write_text(json.dumps(payload))


def test_golden_fixtures_validate_and_flatten_to_canonical_rows(tmp_path: Path) -> None:
    collections = validate_collections(_raw(tmp_path), _config(), now=NOW)
    rows = normalize_markets(collections, "fixture-run")
    report = validate_normalized(rows, collections)
    assert collections.report["status"] == report["status"] == "passed"
    assert len(rows) == 5
    assert {(row["source"], row["outcome_side"]) for row in rows} == {("draftkings", "over"), ("draftkings", "under"), ("fanduel", "over"), ("fanduel", "under"), ("kalshi", "yes")}
    assert next(row for row in rows if row["source"] == "kalshi")["yes_bid_probability"] == pytest.approx(.51)
    assert all(row["raw_record_locator"].startswith("$.markets[") for row in rows)
    assert all(row["normalization_status"] == "included" and row["exclusion_reason"] == "" for row in rows)


@pytest.mark.parametrize(
    ("source", "mutation", "match"),
    [
        ("draftkings", lambda value: value["metadata"].update(season="2025-26"), "season"),
        ("fanduel", lambda value: value["metadata"].update(snapshot_utc="2026-08-24T00:00:00Z"), "freshness"),
        ("draftkings", lambda value: value["markets"].append(deepcopy(value["markets"][0])), "unique_market_ids"),
        ("fanduel", lambda value: value["markets"][0]["probability"]["under"].update(american_odds=0), "odds"),
        ("draftkings", lambda value: value["markets"][0]["probability"]["under"].update(label="Under 3499.5"), "line"),
        ("kalshi", lambda value: value["markets"][0].update(threshold=900.5), "threshold"),
        ("kalshi", lambda value: value["markets"][0].update(stat="unknown_stat"), "stat"),
    ],
)
def test_bad_raw_inputs_fail_closed(tmp_path: Path, source: str, mutation, match: str) -> None:
    raw = _raw(tmp_path); payload = _payload(raw, source); mutation(payload); _write(raw, source, payload)
    with pytest.raises(MarketValidationError) as captured:
        validate_collections(raw, _config(), now=NOW)
    assert any(match in check["name"] for check in captured.value.validation["checks"] if not check["passed"])


def test_mismatched_sportsbook_lines_are_rejected_even_if_collector_fields_are_present(tmp_path: Path) -> None:
    raw = _raw(tmp_path); payload = _payload(raw, "draftkings")
    payload["markets"][0]["market_line"] = 3501.5
    _write(raw, "draftkings", payload)
    # The raw fixture representation stores the parsed shared line. A disagreeing side
    # must be caught from its displayed label as well.
    payload["markets"][0]["probability"]["under"]["label"] = "Under 3500.5"
    _write(raw, "draftkings", payload)
    with pytest.raises(MarketValidationError):
        validate_collections(raw, _config(), now=NOW)


def test_fanduel_collector_placeholders_are_warnings_not_failures(tmp_path: Path) -> None:
    raw = _raw(tmp_path); payload = _payload(raw, "fanduel")
    payload["unavailable_player_prop_references"] = [{"reason": "coupon_placeholder_has_no_attached_market"}]
    _write(raw, "fanduel", payload)
    result = validate_collections(raw, _config(), now=NOW)
    warning = next(check for check in result.report["checks"] if check["name"] == "collections.fanduel.expected_placeholders")
    assert result.report["status"] == "passed"
    assert warning["severity"] == "warning" and not warning["passed"]


def test_normalization_neither_prices_nor_matches_names_or_drops_rows(tmp_path: Path) -> None:
    collections = validate_collections(_raw(tmp_path), _config(), now=NOW)
    rows = normalize_markets(collections, "fixture-run")
    sportsbook = next(row for row in rows if row["source"] == "draftkings" and row["outcome_side"] == "over")
    assert sportsbook["raw_player_name"] == "Example Player"
    assert "no_vig_probability" not in sportsbook
    assert "canonical_player_id" not in sportsbook
    assert len(rows) == 2 * len(collections.source_data["draftkings"]["markets"]) + 2 * len(collections.source_data["fanduel"]["markets"]) + len(collections.source_data["kalshi"]["markets"])


def test_cli_writes_both_validation_reports_and_normalized_csv(tmp_path: Path) -> None:
    config = load_config(ROOT / "config" / "pipeline.toml")
    run_dir = initialize_run(config, ROOT / "config" / "player_aliases.csv", tmp_path / "runs")
    for source in ("draftkings", "fanduel", "kalshi"):
        (run_dir / "raw" / f"{source}.json").write_bytes((ROOT / "tests" / "fixtures" / f"{source}.json").read_bytes())
    # The deterministic fixture has a fixed snapshot; allow its age for this CLI wiring test.
    effective = run_dir / "config" / "effective.toml"
    effective.write_text(effective.read_text().replace("max_snapshot_age_hours = 6", "max_snapshot_age_hours = 100000"))
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    validate = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_collections.py"), "--run-dir", str(run_dir), "--config", str(effective)], text=True, capture_output=True, env=environment, check=True)
    normalized = subprocess.run([sys.executable, str(ROOT / "scripts" / "normalize_markets.py"), "--run-dir", str(run_dir), "--config", str(effective)], text=True, capture_output=True, env=environment, check=True)
    assert json.loads(validate.stdout)["state"] == json.loads(normalized.stdout)["state"] == "succeeded"
    assert json.loads((run_dir / "artifacts" / "collections_validation.json").read_text())["status"] == "passed"
    assert json.loads((run_dir / "artifacts" / "normalized_validation.json").read_text())["status"] == "passed"
    with (run_dir / "artifacts" / "normalized_markets.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 5
