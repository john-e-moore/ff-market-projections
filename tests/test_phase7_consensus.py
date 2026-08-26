from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from ff_market_projections.config import ConfigError, load_config
from ff_market_projections.consensus import ConsensusError, aggregate_source_projections


ROOT = Path(__file__).parents[1]
SOURCES = {
    "draftkings": {"enabled": True, "weight": 1.0},
    "fanduel": {"enabled": True, "weight": 1.0},
    "kalshi": {"enabled": True, "weight": 1.0},
}
AGGREGATION = {"minimum_sources": 1, "renormalize_available_source_weights": True, "max_absolute_disagreement": 250.0, "max_relative_disagreement": 0.25}


def _rows(*values: tuple[str, float, str]) -> pd.DataFrame:
    return pd.DataFrame.from_records([
        {"run_id": "run", "season": "2026-27", "canonical_player_id": "p1", "canonical_player_name": "Player", "stat": "receiving_yards", "source": source, "mean": mean, "sensitivity_low": mean - 10, "sensitivity_high": mean + 10, "quality_status": status}
        for source, mean, status in values
    ])


def test_equal_three_source_mean_and_visible_recalculation_inputs() -> None:
    result = aggregate_source_projections(_rows(("draftkings", 800.0, "passed"), ("fanduel", 900.0, "passed"), ("kalshi", 1000.0, "passed")), SOURCES, AGGREGATION)
    row = result.consensus_stats.iloc[0]
    assert result.validation["status"] == "passed"
    assert row["consensus_mean"] == pytest.approx(900.0)
    assert row["source_count"] == 3
    assert row["draftkings_effective_weight"] == pytest.approx(1 / 3)
    assert row["source_range"] == pytest.approx(200.0)
    assert row["disagreement_flag"] == False


def test_two_source_renormalization_one_source_and_ineligible_exclusion() -> None:
    result = aggregate_source_projections(_rows(("draftkings", 800.0, "passed"), ("fanduel", 1000.0, "excluded")), SOURCES, AGGREGATION)
    row = result.consensus_stats.iloc[0]
    assert row["consensus_mean"] == pytest.approx(800.0)
    assert row["source_list"] == "draftkings"
    assert row["kalshi_effective_weight"] == 0.0

    config = {**AGGREGATION, "minimum_sources": 2}
    result = aggregate_source_projections(_rows(("draftkings", 800.0, "passed")), SOURCES, config)
    assert pd.isna(result.consensus_stats.iloc[0]["consensus_mean"])
    assert result.consensus_stats.iloc[0]["exclusion_reason"] == "minimum_sources_not_met"


def test_custom_weights_disabled_source_and_disagreement_flag() -> None:
    sources = {**SOURCES, "kalshi": {"enabled": False, "weight": 5.0}, "draftkings": {"enabled": True, "weight": 3.0}, "fanduel": {"enabled": True, "weight": 1.0}}
    aggregation = {**AGGREGATION, "max_absolute_disagreement": 50.0}
    result = aggregate_source_projections(_rows(("draftkings", 700.0, "passed"), ("fanduel", 900.0, "passed"), ("kalshi", 2000.0, "passed")), sources, aggregation)
    row = result.consensus_stats.iloc[0]
    assert row["consensus_mean"] == pytest.approx(750.0)
    assert "kalshi_mean" not in row
    assert row["disagreement_flag"] == True


def test_zero_weight_keeps_collection_enabled_but_excludes_source_from_consensus() -> None:
    sources = {**SOURCES, "kalshi": {"enabled": True, "weight": 0.0}}
    result = aggregate_source_projections(
        _rows(
            ("draftkings", 800.0, "passed"),
            ("fanduel", 900.0, "passed"),
            ("kalshi", 2000.0, "passed"),
        ),
        sources,
        AGGREGATION,
    )
    row = result.consensus_stats.iloc[0]
    assert row["consensus_mean"] == pytest.approx(850.0)
    assert row["source_count"] == 2
    assert row["source_list"] == "draftkings|fanduel"
    assert "kalshi_mean" not in row


@pytest.mark.parametrize("weight", [-1])
def test_invalid_weights_rejected(weight: float) -> None:
    sources = {**SOURCES, "fanduel": {"enabled": True, "weight": weight}}
    with pytest.raises(ConsensusError, match="weight must be non-negative"):
        aggregate_source_projections(_rows(("draftkings", 800.0, "passed")), sources, AGGREGATION)


def test_config_rejects_invalid_aggregation_controls(tmp_path: Path) -> None:
    text = (ROOT / "config/pipeline.toml").read_text(encoding="utf-8").replace("max_absolute_disagreement = 100.0", "max_absolute_disagreement = -1.0")
    path = tmp_path / "bad.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="max_absolute_disagreement"):
        load_config(path)


def test_cli_writes_consensus_and_validation_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "config").mkdir(parents=True)
    (run_dir / "artifacts").mkdir()
    (run_dir / "config/effective.toml").write_bytes((ROOT / "config/pipeline.toml").read_bytes())
    _rows(("draftkings", 800.0, "passed")).to_csv(run_dir / "artifacts/source_projections.csv", index=False)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/aggregate_consensus.py"), "--run-dir", str(run_dir)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    output = pd.read_csv(run_dir / "artifacts/consensus_stats.csv")
    validation = (run_dir / "artifacts/aggregation_validation.json").read_text(encoding="utf-8")
    assert output.loc[0, "consensus_mean"] == pytest.approx(800.0)
    assert '"status":"passed"' in validation
