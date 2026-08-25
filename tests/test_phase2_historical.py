from __future__ import annotations

from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from ff_market_projections.config import load_config
from ff_market_projections.historical import HistoricalDataError, TARGET_STATS, prepare_historical_data
from ff_market_projections.runs import initialize_run


ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests/fixtures/nflverse_history_phase2.csv"


def _historical_config() -> dict:
    values = deepcopy(load_config(ROOT / "config/pipeline.toml").values["historical"])
    values.update({
        "ingest_start_season": 2019,
        "calibration_start_season": 2021,
        "latest_completed_season": 2022,
        "minimum_player_seasons_per_stat": 1,
    })
    values["prior_opportunity_filters"] = {"passing_attempts": 10, "rushing_attempts": 5, "targets": 5}
    return values


def _write_frame(tmp_path: Path, frame: pd.DataFrame, name: str = "history.csv") -> Path:
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def _configured_toml(*, minimum_cohort: int = 1) -> str:
    return (
        (ROOT / "config/pipeline.toml").read_text()
        .replace('season = "2026-27"', 'season = "2023-24"')
        .replace("ingest_start_season = 1999", "ingest_start_season = 2019")
        .replace("calibration_start_season = 2011", "calibration_start_season = 2021")
        .replace("latest_completed_season = 2025", "latest_completed_season = 2022")
        .replace("minimum_player_seasons_per_stat = 200", f"minimum_player_seasons_per_stat = {minimum_cohort}")
        .replace("passing_attempts = 100", "passing_attempts = 10")
        .replace("rushing_attempts = 25", "rushing_attempts = 5")
        .replace("targets = 25", "targets = 5")
    )


def test_weekly_rows_reconcile_to_unique_long_player_seasons_and_schedule_normalization() -> None:
    result = prepare_historical_data(FIXTURE, _historical_config())
    seasons = result.player_seasons
    assert result.validation["status"] == "passed"
    assert not seasons.duplicated(["season", "gsis_player_id", "team", "stat"]).any()
    assert set(seasons["stat"]) == set(TARGET_STATS)
    assert len(seasons) == (4 * 3 + 1) * len(TARGET_STATS)

    qb_2019 = seasons.query("season == 2019 and gsis_player_id == '00-QB' and stat == 'passing_yards'").iloc[0]
    qb_2021 = seasons.query("season == 2021 and gsis_player_id == '00-QB' and stat == 'passing_yards'").iloc[0]
    assert qb_2019["stat_total"] == 500
    assert qb_2019["schedule_games"] == 16
    assert qb_2019["stat_total_per_17"] == pytest.approx(531.25)
    assert qb_2019["passing_attempts_per_17"] == pytest.approx(47.8125)
    assert qb_2019["opportunity_per_17"] == pytest.approx(47.8125)
    assert qb_2021["stat_total"] == 560
    assert qb_2021["schedule_games"] == 17
    assert qb_2021["stat_total_per_17"] == 560


def test_baselines_use_only_exact_prior_seasons_and_future_mutation_is_inert(tmp_path: Path) -> None:
    config = _historical_config()
    original = prepare_historical_data(FIXTURE, config).backtest_predictions
    frame = pd.read_csv(FIXTURE)
    frame.loc[(frame["season"] == 2022) & (frame["player_id"] == "00-QB"), "passing_yards"] = [3000, 3000]
    modified = prepare_historical_data(_write_frame(tmp_path, frame), config).backtest_predictions

    keys = ["target_season", "gsis_player_id", "stat"]
    stable_columns = [column for column in original.columns if column not in {*keys, "realized_total", "realized_target_opportunity", "target_games"}]
    left = original.loc[original["target_season"] <= 2022, keys + stable_columns].sort_values(keys).reset_index(drop=True)
    right = modified.loc[modified["target_season"] <= 2022, keys + stable_columns].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)

    qb = original.query("target_season == 2022 and gsis_player_id == '00-QB' and stat == 'passing_yards'").iloc[0]
    assert qb["lag_1_season"] == 2021
    assert qb["lag_2_season"] == 2020
    assert qb["lag_3_season"] == 2019
    assert qb["lag_1_schedule_games"] == 17
    assert qb["lag_2_schedule_games"] == 16
    assert qb["max_feature_season"] == 2021
    assert qb["max_feature_season"] < qb["target_season"]


def test_rookies_insufficient_history_and_missing_opportunity_are_explicit() -> None:
    predictions = prepare_historical_data(FIXTURE, _historical_config()).backtest_predictions
    rookie = predictions.query("target_season == 2022 and gsis_player_id == '00-RK' and stat == 'receiving_yards'").iloc[0]
    assert rookie["player_history_seasons"] == 0
    assert rookie["baseline_available"]
    assert not rookie["training_eligible"]
    assert rookie["cohort_status"] == "rookie_cohort_baseline"

    missing = predictions.query("target_season == 2022 and gsis_player_id == '00-WR' and stat == 'receiving_yards'").iloc[0]
    assert pd.isna(missing["prior_opportunity"])
    assert pd.isna(missing["lag_1_opportunity"])
    assert not missing["opportunity_eligible"]
    assert not missing["training_eligible"]
    assert missing["cohort_status"] == "missing_prior_opportunity"

    insufficient = predictions.query("target_season == 2021 and gsis_player_id == '00-QB' and stat == 'passing_yards'").iloc[0]
    assert insufficient["player_history_seasons"] == 2
    assert insufficient["training_eligible"]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda frame: frame.loc[frame["season"] != 2020], "missing configured season"),
        (lambda frame: frame.drop(columns=["passing_yards"]), "missing required field passing_yards"),
        (lambda frame: frame.assign(player_id=lambda value: value["player_id"].mask(value.index == 0, "")), "missing GSIS player IDs"),
        (lambda frame: frame.assign(passing_yards=lambda value: value["passing_yards"].mask(value.index == 0, 9000)), "plausibility limit"),
    ],
)
def test_missing_seasons_fields_ids_and_implausible_totals_fail(tmp_path: Path, mutation, match: str) -> None:
    path = _write_frame(tmp_path, mutation(pd.read_csv(FIXTURE)))
    with pytest.raises(HistoricalDataError, match=match):
        prepare_historical_data(path, _historical_config())


def test_independent_season_summary_mismatch_fails_reconciliation(tmp_path: Path) -> None:
    frame = pd.read_csv(FIXTURE)
    frame["season_passing_yards"] = frame.groupby(["season", "player_id"])["passing_yards"].transform("sum")
    frame.loc[(frame["season"] == 2021) & (frame["player_id"] == "00-QB"), "season_passing_yards"] = 999
    with pytest.raises(HistoricalDataError, match="do not reconcile"):
        prepare_historical_data(_write_frame(tmp_path, frame), _historical_config())


def test_inadequate_calibration_cohort_fails_closed() -> None:
    config = _historical_config()
    config["minimum_player_seasons_per_stat"] = 999
    with pytest.raises(HistoricalDataError, match="required minimum is 999") as captured:
        prepare_historical_data(FIXTURE, config)
    validation = captured.value.check.details["validation"]
    assert validation["status"] == "failed"
    assert all(stat in validation["cohort_counts"] for stat in TARGET_STATS)


def test_cli_writes_all_three_immutable_run_scoped_artifacts(tmp_path: Path) -> None:
    source_config = tmp_path / "source.toml"
    source_config.write_text(_configured_toml())
    config = load_config(source_config)
    run_dir = initialize_run(config, ROOT / "config/player_aliases.csv", tmp_path / "runs")
    raw = run_dir / "raw/nflverse_player_stats.csv.gz"
    with gzip.open(raw, "wb") as handle:
        handle.write(FIXTURE.read_bytes())
    command = [
        sys.executable,
        str(ROOT / "scripts/prepare_historical_stats.py"),
        "--run-dir", str(run_dir),
        "--config", str(run_dir / "config/effective.toml"),
        "--input", str(raw),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, check=True)
    summary = json.loads(completed.stdout)
    assert summary["state"] == "succeeded"
    artifacts = run_dir / "artifacts"
    assert (artifacts / "historical_player_seasons.csv").is_file()
    assert (artifacts / "historical_backtest_predictions.csv").is_file()
    validation = json.loads((artifacts / "historical_validation.json").read_text())
    assert validation["status"] == "passed"
    assert validation["artifacts"]["historical_player_seasons.csv"]["rows"] == (4 * 3 + 1) * len(TARGET_STATS)
    manifest = json.loads((run_dir / "metadata/manifest.json").read_text())
    status = json.loads((run_dir / "metadata/run_status.json").read_text())
    assert manifest["historical_state"] == "succeeded"
    assert manifest["tasks"]["prepare_historical_stats"]["state"] == "succeeded"
    assert manifest["task_dag"]["prepare_historical_stats"] == ["collect_nflverse_history"]
    assert status["state"] == "running"
    assert status["historical_state"] == "succeeded"

    repeated = subprocess.run(command, text=True, capture_output=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert repeated.returncode != 0
    assert "artifacts are immutable" in repeated.stderr


def test_cli_validation_failure_marks_run_failed_and_preserves_evidence(tmp_path: Path) -> None:
    source_config = tmp_path / "source.toml"
    source_config.write_text(_configured_toml(minimum_cohort=999))
    config = load_config(source_config)
    run_dir = initialize_run(config, ROOT / "config/player_aliases.csv", tmp_path / "runs")
    raw = run_dir / "raw/nflverse_player_stats.csv.gz"
    with gzip.open(raw, "wb") as handle:
        handle.write(FIXTURE.read_bytes())
    command = [
        sys.executable,
        str(ROOT / "scripts/prepare_historical_stats.py"),
        "--run-dir", str(run_dir),
        "--config", str(run_dir / "config/effective.toml"),
        "--input", str(raw),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert completed.returncode == 1
    summary = json.loads(completed.stdout)
    assert summary["state"] == "failed"
    validation = json.loads((run_dir / "artifacts/historical_validation.json").read_text())
    manifest = json.loads((run_dir / "metadata/manifest.json").read_text())
    status = json.loads((run_dir / "metadata/run_status.json").read_text())
    assert validation["status"] == "failed"
    assert manifest["tasks"]["prepare_historical_stats"]["state"] == "failed"
    assert status["state"] == "failed"
    assert status["failed_task"] == "prepare_historical_stats"
    assert not (run_dir / "artifacts/historical_player_seasons.csv").exists()
