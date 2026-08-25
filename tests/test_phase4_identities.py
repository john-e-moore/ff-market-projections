from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ff_market_projections.config import load_config
from ff_market_projections.identities import (
    IdentityError,
    csv_bytes,
    load_aliases,
    normalize_match_key,
    promote_reviewed_suggestions,
    reconcile_players,
)
from ff_market_projections.runs import initialize_run


ROOT = Path(__file__).parents[1]


def _names(*, minimum_score: float = 96.0, minimum_runner_up_gap: float = 4.0) -> dict:
    return {"automatic_fuzzy_match": True, "minimum_score": minimum_score, "minimum_runner_up_gap": minimum_runner_up_gap}


def _market(source: str, name: str, stat: str = "passing_yards", team: str = "EXM") -> dict:
    return {"source": source, "raw_player_name": name, "stat": stat, "team": team, "source_market_id": f"{source}-{name}-{stat}"}


def _history(gsis_id: str, name: str, position: str = "QB", team: str = "EXM", stat: str = "passing_yards") -> dict:
    return {"gsis_player_id": gsis_id, "player_name": name, "position": position, "team": team, "stat": stat, "season": "2024"}


def _aliases(tmp_path: Path, rows: list[dict[str, str]] | None = None) -> Path:
    path = tmp_path / "player_aliases.csv"
    fields = ["source", "raw_player_name", "canonical_player_id", "canonical_player_name", "notes"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows or [])
    return path


@pytest.mark.parametrize(
    ("display_name", "expected"),
    [
        ("J. O’Neil", "j oneil"),
        ("D'Andre-Swift Jr.", "dandre swift"),
        ("José Núñez II", "jose nunez"),
        ("Amon-Ra St. Brown Sr", "amon ra st brown"),
    ],
)
def test_match_key_normalization_handles_initials_punctuation_suffixes_and_unicode(display_name: str, expected: str) -> None:
    assert normalize_match_key(display_name) == expected


def test_golden_cross_source_identity_uses_gsis_and_preserves_unmatched_player(tmp_path: Path) -> None:
    result = reconcile_players(
        [_market("draftkings", "Example Player"), _market("fanduel", "Example Player"), _market("kalshi", "Example Player"), _market("fanduel", "Unmatched Player", "receiving_yards")],
        [_history("00-EXAMPLE", "Example Player"), _history("00-EXAMPLE", "Example Player", stat="rushing_yards")],
        _names(), load_aliases(_aliases(tmp_path)),
    )
    matched = [row for row in result.rows if row["raw_player_name"] == "Example Player"]
    unmatched = next(row for row in result.rows if row["raw_player_name"] == "Unmatched Player")
    assert result.validation["status"] == "passed"
    assert {row["canonical_player_id"] for row in matched} == {"gsis:00-EXAMPLE"}
    assert {row["canonical_position"] for row in matched} == {"QB"}
    assert unmatched["canonical_player_id"].startswith("player:unmatched-player-")
    assert unmatched["identity_review_status"] == "unmatched"
    assert result.suggestions and result.suggestions[0]["reason"] == "no_candidate"


def test_initials_can_match_unique_historical_player_but_ambiguous_high_scores_stay_unmerged(tmp_path: Path) -> None:
    unique = reconcile_players(
        [_market("draftkings", "J. O’Neil")], [_history("00-ONEIL", "James ONeil")],
        _names(), load_aliases(_aliases(tmp_path)),
    )
    assert unique.rows[0]["canonical_player_id"] == "gsis:00-ONEIL"
    ambiguous = reconcile_players(
        [_market("draftkings", "J. Smith")], [_history("00-JOHN", "John Smith"), _history("00-JAMES", "James Smith")],
        _names(), load_aliases(_aliases(tmp_path)),
    )
    assert ambiguous.rows[0]["identity_review_status"] == "unmatched"
    assert ambiguous.suggestions[0]["reason"] == "ambiguous_fuzzy_match"
    assert ambiguous.suggestions[0]["match_score"] == 100.0


def test_source_collision_guard_keeps_two_source_names_distinct(tmp_path: Path) -> None:
    result = reconcile_players(
        [_market("draftkings", "J Smith"), _market("draftkings", "John Smith")],
        [_history("00-JOHN", "John Smith")], _names(), load_aliases(_aliases(tmp_path)),
    )
    by_name = {row["raw_player_name"]: row for row in result.rows}
    assert by_name["J Smith"]["canonical_player_id"] == "gsis:00-JOHN"
    assert by_name["John Smith"]["identity_review_status"] == "unmatched"
    assert any(row["reason"] == "source_level_collision" for row in result.suggestions)
    assert result.validation["status"] == "passed"


def test_explicit_aliases_override_matching_and_authorize_a_source_collision(tmp_path: Path) -> None:
    alias_file = _aliases(tmp_path, [
        {"source": "draftkings", "raw_player_name": "J Smith", "canonical_player_id": "gsis:00-JOHN", "canonical_player_name": "John Smith", "notes": "reviewed"},
        {"source": "draftkings", "raw_player_name": "John Smith", "canonical_player_id": "gsis:00-JOHN", "canonical_player_name": "John Smith", "notes": "reviewed"},
    ])
    result = reconcile_players(
        [_market("draftkings", "J Smith"), _market("draftkings", "John Smith")],
        [_history("00-JAMES", "James Smith")], _names(), load_aliases(alias_file),
    )
    assert {row["canonical_player_id"] for row in result.rows} == {"gsis:00-JOHN"}
    assert {row["identity_match_method"] for row in result.rows} == {"explicit_alias"}
    assert result.validation["status"] == "passed"


def test_one_explicit_alias_does_not_authorize_an_unaliased_source_collision(tmp_path: Path) -> None:
    alias_file = _aliases(tmp_path, [
        {"source": "draftkings", "raw_player_name": "J Smith", "canonical_player_id": "gsis:00-JOHN", "canonical_player_name": "John Smith", "notes": "reviewed"},
    ])
    result = reconcile_players(
        [_market("draftkings", "J Smith"), _market("draftkings", "John Smith")],
        [_history("00-JOHN", "John Smith")], _names(), load_aliases(alias_file),
    )
    by_name = {row["raw_player_name"]: row for row in result.rows}
    assert by_name["J Smith"]["identity_review_status"] == "alias"
    assert by_name["John Smith"]["identity_review_status"] == "unmatched"


def test_alias_conflicts_fail_closed(tmp_path: Path) -> None:
    alias_file = _aliases(tmp_path, [
        {"source": "fanduel", "raw_player_name": "A Player", "canonical_player_id": "one", "canonical_player_name": "A Player", "notes": ""},
        {"source": "fanduel", "raw_player_name": "A Player", "canonical_player_id": "two", "canonical_player_name": "A Player", "notes": ""},
    ])
    with pytest.raises(IdentityError, match="Conflicting aliases"):
        load_aliases(alias_file)


def test_same_input_and_aliases_produce_deterministic_maps(tmp_path: Path) -> None:
    rows = [_market("kalshi", "Other Player", "receiving_yards"), _market("draftkings", "Example Player"), _market("fanduel", "Example Player")]
    history = [_history("00-EXAMPLE", "Example Player"), _history("00-OTHER", "Other Player", position="WR", stat="receiving_yards")]
    aliases = load_aliases(_aliases(tmp_path))
    first = reconcile_players(rows, history, _names(), aliases)
    second = reconcile_players(list(reversed(rows)), list(reversed(history)), _names(), aliases)
    assert first.player_map == second.player_map
    assert sorted(first.rows, key=lambda row: row["source_market_id"]) == sorted(second.rows, key=lambda row: row["source_market_id"])


def test_cli_writes_identity_artifacts_and_decorates_normalized_markets(tmp_path: Path) -> None:
    config = load_config(ROOT / "config" / "pipeline.toml")
    run_dir = initialize_run(config, ROOT / "config" / "player_aliases.csv", tmp_path / "runs")
    artifacts = run_dir / "artifacts"
    normalized = [_market("draftkings", "Example Player")]
    history = [_history("00-EXAMPLE", "Example Player")]
    normalized_fields = list(normalized[0])
    (artifacts / "normalized_markets.csv").write_bytes(csv_bytes(normalized, normalized_fields))
    (artifacts / "historical_player_seasons.csv").write_bytes(csv_bytes(history, list(history[0])))
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "reconcile_players.py"), "--run-dir", str(run_dir)],
        text=True, capture_output=True, check=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert json.loads(completed.stdout)["state"] == "succeeded"
    with (artifacts / "normalized_markets.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["canonical_player_id"] == "gsis:00-EXAMPLE"
    assert (artifacts / "player_map.csv").is_file()
    assert (artifacts / "name_match_suggestions.csv").is_file()
    assert json.loads((artifacts / "identity_validation.json").read_text())["status"] == "passed"


def test_reviewed_suggestion_promotion_only_appends_explicit_approvals(tmp_path: Path) -> None:
    aliases = _aliases(tmp_path)
    suggestions = tmp_path / "name_match_suggestions.csv"
    suggestions.write_text(
        "source,raw_player_name,candidate_canonical_player_id,candidate_canonical_player_name,review_status\n"
        "draftkings,J. ONeil,gsis:00-ONEIL,James ONeil,approved\n"
        "fanduel,Needs Review,gsis:00-NO,No Match,needs_review\n",
        encoding="utf-8",
    )
    assert promote_reviewed_suggestions(suggestions, aliases) == 1
    loaded = load_aliases(aliases)
    assert [(item.source, item.raw_player_name, item.canonical_player_id) for item in loaded] == [("draftkings", "J. ONeil", "gsis:00-ONEIL")]
