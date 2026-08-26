from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pytest

from ff_market_projections.config import ConfigError, load_config
from ff_market_projections.contracts import CheckResult, atomic_write_bytes, canonical_json, sha256_file
from ff_market_projections.runs import initialize_run


ROOT = Path(__file__).parents[1]


def test_config_loads_and_effective_copy_preserves_semantics(tmp_path: Path) -> None:
    config = load_config(ROOT / "config/pipeline.toml")
    run_dir = initialize_run(config, ROOT / "config/player_aliases.csv", tmp_path / "runs")
    copied = run_dir / "config/effective.toml"
    assert tomllib.loads(copied.read_text()) == config.values
    assert copied.read_bytes() == config.raw_toml
    assert json.loads((run_dir / "logs/pipeline.log").read_text())["message"].startswith("run_initialized")
    assert json.loads((run_dir / "metadata/run_status.json").read_text())["state"] == "running"


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("unexpected = true\n", "Unknown setting"),
        ("weight = -1.0", "must be a positive number"),
        ("probability_floor = 0.99\nprobability_ceiling = 0.98", "probability_floor"),
        ("prior_seasons = 4", "prior_seasons must be between 1 and 3"),
        ("passing_yards = [2.0, 1.0]", "dispersion_bounds.passing_yards"),
        ("max_brier_calibration_gap = 1.1", "max_brier_calibration_gap"),
    ],
)
def test_invalid_settings_fail_usefully(tmp_path: Path, replacement: str, message: str) -> None:
    source = (ROOT / "config/pipeline.toml").read_text()
    if replacement.startswith("unexpected"):
        source += replacement
    elif replacement.startswith("weight"):
        source = source.replace("weight = 1.0", replacement, 1)
    elif replacement.startswith("probability_floor"):
        source = source.replace("probability_floor = 0.02\nprobability_ceiling = 0.98", replacement)
    elif replacement.startswith("passing_yards"):
        source = source.replace("passing_yards = [0.05, 1000.0]", replacement)
    elif replacement.startswith("max_brier"):
        source = source.replace("max_brier_calibration_gap = 0.10", replacement)
    else:
        source = source.replace("prior_seasons = 3", replacement)
    path = tmp_path / "bad.toml"
    path.write_text(source)
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_duplicate_toml_settings_fail_usefully(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.toml"
    path.write_text((ROOT / "config/pipeline.toml").read_text() + "\n[run]\nseason = '2026-27'\n")
    with pytest.raises(ConfigError, match="Invalid TOML"):
        load_config(path)


def test_atomic_writer_has_stable_hash_and_no_temporary_file(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "value.json"
    payload = canonical_json({"b": 2, "a": 1})
    first = atomic_write_bytes(output, payload)
    second = atomic_write_bytes(output, payload)
    assert first == second == sha256_file(output)
    assert not list(output.parent.glob("*.tmp"))


def test_check_result_serializes_consistently() -> None:
    check = CheckResult("keys_unique", True, "warning", "all good", {"rows": 2})
    assert CheckResult.from_dict(check.to_dict()) == check
    assert canonical_json(check.to_dict()) == canonical_json(check.to_dict())


def test_fixtures_parse_without_network() -> None:
    fixture_dir = ROOT / "tests/fixtures"
    for name in ("draftkings.json", "fanduel.json", "kalshi.json"):
        value = json.loads((fixture_dir / name).read_text())
        assert value["metadata"]["season"] == "2026-27"
        assert value["markets"]
    rows = (fixture_dir / "nflverse_player_stats.csv").read_text().splitlines()
    assert len(rows) == 3 and "player_id" in rows[0]
