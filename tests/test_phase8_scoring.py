from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from ff_market_projections.config import ConfigError, load_config
from ff_market_projections.scoring import ScoringError, score_consensus


ROOT = Path(__file__).parents[1]
SCORING = {
    "missing_stat_policy": "blank_total",
    "passing_yards": 0.04, "passing_touchdowns": 4.0, "passing_interceptions": -2.0,
    "rushing_yards": 0.10, "rushing_touchdowns": 6.0, "receiving_yards": 0.10,
    "receiving_touchdowns": 6.0, "fumbles_lost": -2.0, "two_point_conversions": 2.0,
    "reception_bonus": {"standard": 0.0, "half_ppr": 0.5, "three_quarter_ppr": 0.75, "full_ppr": 1.0},
    "required_profiles": {"passing": ["passing_yards", "passing_touchdowns"], "rushing": ["rushing_yards", "rushing_touchdowns"], "receiving": ["receiving_yards", "receiving_touchdowns", "receptions"]},
}


def _consensus(rows: list[tuple[str, str, str, float | None, str]]):
    return pd.DataFrame.from_records([
        {"run_id": "run", "season": "2026-27", "canonical_player_id": player, "canonical_player_name": name, "stat": stat, "consensus_mean": mean, "quality_status": status}
        for player, name, stat, mean, status in rows
    ])


def test_all_four_modes_hand_calculate_and_reconcile() -> None:
    frame = _consensus([
        ("p1", "Quarterback", "passing_yards", 4000.0, "passed"),
        ("p1", "Quarterback", "passing_touchdowns", 30.0, "passed"),
        ("p1", "Quarterback", "receptions", 10.0, "passed"),
    ])
    result = score_consensus(frame, SCORING)
    row = result.fantasy_projections.iloc[0]
    assert result.validation["status"] == "passed"
    assert row["partial_fpts_standard"] == pytest.approx(280.0)
    assert row["partial_fpts_half_ppr"] == pytest.approx(285.0)
    assert row["partial_fpts_three_quarter_ppr"] == pytest.approx(287.5)
    assert row["partial_fpts_full_ppr"] == pytest.approx(290.0)
    assert row["fpts_full_ppr"] == pytest.approx(row["fpts_standard"] + 10.0)


def test_missing_projection_is_blank_and_unsupported_stats_are_disclosed() -> None:
    frame = _consensus([
        ("p1", "Receiver", "receiving_yards", 1000.0, "passed"),
        ("p1", "Receiver", "receiving_touchdowns", 8.0, "passed"),
    ])
    row = score_consensus(frame, SCORING).fantasy_projections.iloc[0]
    assert pd.isna(row["receptions"])
    assert pd.isna(row["fpts_standard"])
    assert row["partial_fpts_standard"] == pytest.approx(148.0)
    assert "passing_interceptions" in row["components_missing"]
    assert "fumbles_lost" in row["components_missing"]
    assert bool(row["projection_complete"]) is False


@pytest.mark.parametrize("receptions", [0.0, 1.0, 20.0])
def test_ppr_monotonicity(receptions: float) -> None:
    frame = _consensus([("p1", "Receiver", "receiving_yards", 100.0, "passed"), ("p1", "Receiver", "receiving_touchdowns", 1.0, "passed"), ("p1", "Receiver", "receptions", receptions, "passed")])
    row = score_consensus(frame, SCORING).fantasy_projections.iloc[0]
    assert row["partial_fpts_standard"] <= row["partial_fpts_half_ppr"] <= row["partial_fpts_three_quarter_ppr"] <= row["partial_fpts_full_ppr"]


def test_invalid_policy_rejected() -> None:
    with pytest.raises(ScoringError, match="missing_stat_policy"):
        score_consensus(_consensus([("p1", "Player", "receiving_yards", 1.0, "passed")]), {**SCORING, "missing_stat_policy": "zero_fill"})


def test_config_validates_scoring_controls(tmp_path: Path) -> None:
    text = (ROOT / "config/pipeline.toml").read_text(encoding="utf-8").replace('missing_stat_policy = "blank_total"', 'missing_stat_policy = "zero_fill"')
    path = tmp_path / "bad.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="missing_stat_policy"):
        load_config(path)


def test_cli_writes_fantasy_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "config").mkdir(parents=True)
    (run_dir / "artifacts").mkdir()
    (run_dir / "config/effective.toml").write_bytes((ROOT / "config/pipeline.toml").read_bytes())
    _consensus([("p1", "Receiver", "receiving_yards", 1000.0, "passed"), ("p1", "Receiver", "receiving_touchdowns", 8.0, "passed"), ("p1", "Receiver", "receptions", 100.0, "passed")]).to_csv(run_dir / "artifacts/consensus_stats.csv", index=False)
    completed = subprocess.run([sys.executable, str(ROOT / "scripts/score_fantasy.py"), "--run-dir", str(run_dir)], cwd=ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    output = pd.read_csv(run_dir / "artifacts/fantasy_projections.csv")
    assert output.loc[0, "partial_fpts_full_ppr"] == pytest.approx(248.0)
    assert '"status":"passed"' in (run_dir / "artifacts/scoring_validation.json").read_text(encoding="utf-8")
