from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ff_market_projections.calibration import (
    CalibrationError,
    calibrate_historical_distributions,
    fit_historical_dispersion,
    grouped_bootstrap_log_dispersion,
)
from ff_market_projections.config import load_config
from ff_market_projections.historical import TARGET_STATS
from ff_market_projections.runs import initialize_run


ROOT = Path(__file__).parents[1]
TRUE_DISPERSION = {
    "passing_yards": 7.0,
    "passing_touchdowns": 5.0,
    "rushing_yards": 4.0,
    "rushing_touchdowns": 3.0,
    "receiving_yards": 4.5,
    "receiving_touchdowns": 3.5,
    "receptions": 6.0,
}
BASE_MEAN = {
    "passing_yards": 3600.0,
    "passing_touchdowns": 25.0,
    "rushing_yards": 700.0,
    "rushing_touchdowns": 6.0,
    "receiving_yards": 750.0,
    "receiving_touchdowns": 6.0,
    "receptions": 65.0,
}


def _synthetic_predictions(*, seed: int = 219, players: int = 45) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for season in range(2011, 2026):
        for player_number in range(players):
            player_factor = float(rng.lognormal(mean=0.0, sigma=0.25))
            for stat in TARGET_STATS:
                mean = BASE_MEAN[stat] * player_factor
                dispersion = TRUE_DISPERSION[stat]
                probability = dispersion / (dispersion + mean)
                realized = int(rng.negative_binomial(dispersion, probability))
                rows.append({
                    "target_season": season,
                    "gsis_player_id": f"00-{player_number:04d}",
                    "position": "QB" if stat.startswith("passing") else "RB" if stat.startswith("rushing") else "WR",
                    "stat": stat,
                    "realized_total": realized,
                    "target_games": int(rng.integers(8, 18)),
                    "target_schedule_games": 17,
                    "baseline_mean": mean,
                    "training_eligible": True,
                    "feature_seasons": str(season - 1),
                    "max_feature_season": season - 1,
                    "lag_1_season": season - 1,
                    "lag_2_season": season - 2,
                    "lag_3_season": season - 3,
                })
    return pd.DataFrame.from_records(rows)


def _configs() -> tuple[dict, dict]:
    values = load_config(ROOT / "config/pipeline.toml").values
    historical = deepcopy(values["historical"])
    historical.update({
        "calibration_start_season": 2011,
        "latest_completed_season": 2024,
        "holdout_seasons": 3,
        "minimum_player_seasons_per_stat": 100,
    })
    model = deepcopy(values["model"])
    model["bootstrap_samples"] = 16
    calibration = model["historical_calibration"]
    calibration.update({
        "minimum_holdout_player_seasons_per_stat": 30,
        "minimum_sensitivity_player_seasons_per_stat": 30,
        "max_holdout_nll_regret_per_player_season": 10.0,
        "max_brier_calibration_gap": 1.0,
        "max_interval_coverage_error": 1.0,
        "max_abs_relative_bias": 1.0,
        "max_sensitivity_log_dispersion_delta": 10.0,
        "minimum_bootstrap_success_rate": 0.5,
    })
    return historical, model


def test_known_predictive_dispersion_is_recovered_with_fixed_means() -> None:
    rng = np.random.default_rng(440)
    true_dispersion = 4.0
    means = rng.lognormal(mean=5.7, sigma=0.5, size=12_000)
    probabilities = true_dispersion / (true_dispersion + means)
    outcomes = rng.negative_binomial(true_dispersion, probabilities)
    fit = fit_historical_dispersion(outcomes, means, np.full(len(means), 2020), (0.05, 1000.0), 5.0)
    assert fit.converged
    assert not fit.bound_hit
    assert fit.dispersion == pytest.approx(true_dispersion, rel=0.08)


def test_grouped_bootstrap_is_reproducible_with_fixed_seed() -> None:
    frame = _synthetic_predictions(players=20).query("stat == 'rushing_yards' and target_season < 2023")
    kwargs = {
        "samples": 12,
        "seed": 1234,
        "confidence_level": 0.9,
        "minimum_success_rate": 0.5,
        "reference_season": 2022,
    }
    first = grouped_bootstrap_log_dispersion(frame, (0.05, 1000.0), 5.0, **kwargs)
    second = grouped_bootstrap_log_dispersion(frame, (0.05, 1000.0), 5.0, **kwargs)
    assert first == second
    assert first["passed"]
    assert first["dispersion_lower"] < first["dispersion_upper"]


def test_holdout_is_excluded_from_fit_and_all_diagnostics_are_reported() -> None:
    predictions = _synthetic_predictions()
    historical, model = _configs()
    original = calibrate_historical_distributions(predictions, historical, model)
    changed = predictions.copy()
    changed.loc[changed["target_season"] >= 2022, "realized_total"] *= 5
    mutated = calibrate_historical_distributions(changed, historical, model)

    assert original.report["status"] == "passed"
    assert mutated.report["status"] == "failed"
    assert len(original.dispersions) == len(TARGET_STATS)
    assert set(original.dispersions["method"]) == {"historical_only"}
    assert original.dispersions["training_seasons"].str.endswith("2021").all()
    for stat in TARGET_STATS:
        assert original.report["stats"][stat]["fit"] == mutated.report["stats"][stat]["fit"]
        assert original.report["stats"][stat]["bootstrap"] == mutated.report["stats"][stat]["bootstrap"]
        metrics = original.report["stats"][stat]["holdout_metrics"]
        assert metrics["brier_events"]
        assert len(metrics["predictive_interval_coverage"]) == 3
        assert all("mean_model_implied_coverage" in value for value in metrics["predictive_interval_coverage"])
        assert metrics["mean_calibration_bins"]
        assert metrics["position_cohorts"]
        assert metrics["availability_cohorts"]


def test_negative_yardage_is_explicitly_excluded_without_clipping() -> None:
    predictions = _synthetic_predictions()
    historical, model = _configs()
    index = predictions.query("stat == 'rushing_yards' and target_season == 2015").index[0]
    predictions.loc[index, "realized_total"] = -7
    result = calibrate_historical_distributions(predictions, historical, model)
    assert result.report["stats"]["rushing_yards"]["negative_outcomes_excluded"] == 1
    warning = next(check for check in result.report["checks"] if check["name"].endswith("signed_negative_observations"))
    assert warning["severity"] == "warning"
    assert not warning["passed"]


def test_future_feature_lineage_fails_before_fitting() -> None:
    predictions = _synthetic_predictions(players=10)
    predictions.loc[0, "feature_seasons"] = str(predictions.loc[0, "target_season"])
    predictions.loc[0, "max_feature_season"] = predictions.loc[0, "target_season"]
    historical, model = _configs()
    with pytest.raises(CalibrationError, match="target or a future season"):
        calibrate_historical_distributions(predictions, historical, model)


def test_inadequate_cohorts_and_impossible_zero_baselines_fail_closed() -> None:
    historical, model = _configs()
    small = calibrate_historical_distributions(_synthetic_predictions(players=5), historical, model)
    assert small.report["status"] == "failed"
    assert any(
        check["name"].endswith("training_cohort") and not check["passed"]
        for check in small.report["checks"]
    )

    impossible = _synthetic_predictions(players=45)
    row = impossible.query("stat == 'receiving_touchdowns' and target_season == 2015").index[0]
    impossible.loc[row, ["baseline_mean", "realized_total"]] = [0.0, 1]
    result = calibrate_historical_distributions(impossible, historical, model)
    assert result.report["status"] == "failed"
    check = next(
        value for value in result.report["checks"]
        if value["name"] == "historical_calibration.receiving_touchdowns.zero_baseline_consistency"
    )
    assert not check["passed"]


def test_nonconvergence_and_bound_hits_are_exposed(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailedResult:
        x = np.log(0.05)
        fun = 10.0
        success = False
        nit = 1
        nfev = 2

    monkeypatch.setattr("ff_market_projections.calibration.minimize_scalar", lambda *args, **kwargs: FailedResult())
    fit = fit_historical_dispersion([1, 2], [2.0, 2.0], [2020, 2021], (0.05, 100.0), 5.0)
    assert not fit.converged
    assert fit.bound_hit


def _cli_toml() -> str:
    return (
        (ROOT / "config/pipeline.toml").read_text()
        .replace('season = "2026-27"', 'season = "2026-27"')
        .replace("minimum_player_seasons_per_stat = 200", "minimum_player_seasons_per_stat = 100")
        .replace("bootstrap_samples = 500", "bootstrap_samples = 8")
        .replace("sensitivity_start_seasons = [2011, 2016, 2021]", "sensitivity_start_seasons = [2011]")
        .replace("minimum_bootstrap_success_rate = 0.90", "minimum_bootstrap_success_rate = 0.50")
        .replace("max_holdout_nll_regret_per_player_season = 0.05", "max_holdout_nll_regret_per_player_season = 10.0")
        .replace("max_brier_calibration_gap = 0.10", "max_brier_calibration_gap = 1.0")
        .replace("max_interval_coverage_error = 0.10", "max_interval_coverage_error = 1.0")
        .replace("max_abs_relative_bias = 0.10", "max_abs_relative_bias = 1.0")
        .replace("max_sensitivity_log_dispersion_delta = 1.0", "max_sensitivity_log_dispersion_delta = 10.0")
    )


def test_cli_writes_run_scoped_calibration_artifacts_and_manifest(tmp_path: Path) -> None:
    source_config = tmp_path / "source.toml"
    source_config.write_text(_cli_toml())
    config = load_config(source_config)
    run_dir = initialize_run(config, ROOT / "config/player_aliases.csv", tmp_path / "runs")
    predictions = _synthetic_predictions()
    input_path = run_dir / "artifacts/historical_backtest_predictions.csv"
    predictions.to_csv(input_path, index=False)
    command = [sys.executable, str(ROOT / "scripts/calibrate_distributions.py"), "--run-dir", str(run_dir)]
    completed = subprocess.run(command, text=True, capture_output=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, check=True)
    summary = json.loads(completed.stdout)
    assert summary["state"] == "succeeded"
    dispersions = pd.read_csv(run_dir / "artifacts/dispersion_calibration.csv")
    report = json.loads((run_dir / "artifacts/historical_calibration.json").read_text())
    manifest = json.loads((run_dir / "metadata/manifest.json").read_text())
    assert len(dispersions) == len(TARGET_STATS)
    assert report["status"] == "passed"
    assert manifest["task_dag"]["calibrate_distributions"] == ["prepare_historical_stats"]
    assert manifest["historical_calibration"]["method"] == "historical_only"
    assert not (run_dir / "artifacts/source_projections.csv").exists()

    repeated = subprocess.run(command, text=True, capture_output=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert repeated.returncode != 0
    assert "artifacts are immutable" in repeated.stderr
