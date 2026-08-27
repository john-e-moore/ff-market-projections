from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ff_market_projections.config import ConfigError, load_config
from ff_market_projections.contracts import sha256_file
from ff_market_projections.distributions import negative_binomial_survival
from ff_market_projections.historical import TARGET_STATS
from ff_market_projections.modeling import estimate_market_means, fit_shared_kalshi_dispersion
from ff_market_projections.runs import initialize_run


ROOT = Path(__file__).parents[1]
CALIBRATION_VERSION = "negative_binomial_historical_v1"


def _config() -> dict:
    model = deepcopy(load_config(ROOT / "config/pipeline.toml").values["model"])
    model["minimum_calibration_groups"] = 4
    return model


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('robust_loss = "soft_l1"', 'robust_loss = "linear"', "robust_loss"),
        ("minimum_thresholds_per_group = 2", "minimum_thresholds_per_group = 1", "at least two"),
        ("passing_yards = [0.01, 10000.0]", "passing_yards = [100.0, 10.0]", "mean_bounds.passing_yards"),
        ('current_market_conflict_policy = "warning"', 'current_market_conflict_policy = "ignore"', "current_market_conflict_policy"),
    ],
)
def test_current_market_configuration_rejects_unsupported_or_invalid_controls(
    tmp_path: Path, old: str, new: str, message: str,
) -> None:
    path = tmp_path / "bad.toml"
    path.write_text((ROOT / "config/pipeline.toml").read_text().replace(old, new), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def _dispersions(*, receiving_yards: float = 9.0, overrides: dict[str, float] | None = None) -> pd.DataFrame:
    overrides = overrides or {}
    rows = []
    for stat in TARGET_STATS:
        historical = overrides.get(stat, receiving_yards if stat == "receiving_yards" else 6.0)
        lower, upper = historical / 3.0, historical * 3.0
        rows.append({
            "stat": stat,
            "distribution_family": "negative_binomial",
            "calibration_version": CALIBRATION_VERSION,
            "method": "historical_only",
            "historical_dispersion": historical,
            "historical_log_dispersion": math.log(historical),
            "historical_dispersion_lower": lower,
            "historical_dispersion_upper": upper,
            "historical_log_dispersion_lower": math.log(lower),
            "historical_log_dispersion_upper": math.log(upper),
            "final_dispersion": historical,
            "dispersion_source": "historical_only",
            "status": "passed",
        })
    return pd.DataFrame.from_records(rows)


def _historical_report() -> dict:
    return {
        "status": "passed",
        "calibration_version": CALIBRATION_VERSION,
        "method": "historical_only",
        "stats": {stat: {"status": "passed"} for stat in TARGET_STATS},
        "checks": [],
    }


def _market(
    source: str,
    player: str,
    stat: str,
    threshold: int,
    probability: float,
    *,
    market_id: str,
    inclusion_status: str = "included",
    exclusion_reason: str = "",
) -> dict:
    is_kalshi = source == "kalshi"
    return {
        "run_id": "fixture-run",
        "season": "2026-27",
        "source": source,
        "source_market_id": market_id,
        "snapshot_utc": "2026-08-25T00:00:00Z",
        "raw_player_name": player,
        "stat": stat,
        "canonical_threshold": threshold if inclusion_status == "included" else None,
        "modeling_probability": probability if inclusion_status == "included" else None,
        "kalshi_bid_probability": probability - 0.005 if is_kalshi and inclusion_status == "included" else None,
        "kalshi_ask_probability": probability + 0.005 if is_kalshi and inclusion_status == "included" else None,
        "inclusion_status": inclusion_status,
        "exclusion_reason": exclusion_reason,
        "canonical_player_id": f"player:{player.lower().replace(' ', '-')}",
        "canonical_player_name": player,
        "canonical_position": "WR" if stat.startswith("rece") else "QB",
    }


def _kalshi_curves(*, dispersion: float = 4.0) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict] = []
    means: dict[str, float] = {}
    thresholds = np.asarray([400, 700, 1000, 1300])
    for player_number, mean in enumerate([650.0, 800.0, 950.0, 1100.0, 1250.0]):
        player = f"Example Player {player_number}"
        means[f"player:{player.lower().replace(' ', '-')}"] = mean
        probabilities = negative_binomial_survival(thresholds, mean, dispersion)
        for threshold, probability in zip(thresholds, probabilities, strict=True):
            rows.append(_market(
                "kalshi", player, "receiving_yards", int(threshold), float(probability),
                market_id=f"kx-{player_number}-{threshold}",
            ))
    return pd.DataFrame.from_records(rows), means


def _steep_kalshi_curves() -> pd.DataFrame:
    rows: list[dict] = []
    thresholds = np.asarray([750, 1000, 1250])
    for player_number, mean in enumerate([850.0, 925.0, 1000.0, 1075.0, 1150.0]):
        player = f"Steep Curve {player_number}"
        probabilities = negative_binomial_survival(thresholds, mean, 20.0)
        for threshold, probability in zip(thresholds, probabilities, strict=True):
            rows.append(_market(
                "kalshi", player, "receiving_yards", int(threshold), float(probability),
                market_id=f"steep-{player_number}-{threshold}",
            ))
    return pd.DataFrame.from_records(rows)


def _narrow_historical_prior(*, receiving_yards: float = 1.5) -> pd.DataFrame:
    dispersions = _dispersions(receiving_yards=receiving_yards)
    target = dispersions["stat"].eq("receiving_yards")
    log_dispersion = math.log(receiving_yards)
    dispersions.loc[target, "historical_dispersion_lower"] = math.exp(log_dispersion - 0.02)
    dispersions.loc[target, "historical_dispersion_upper"] = math.exp(log_dispersion + 0.02)
    dispersions.loc[target, "historical_log_dispersion_lower"] = log_dispersion - 0.02
    dispersions.loc[target, "historical_log_dispersion_upper"] = log_dispersion + 0.02
    return dispersions


def _narrow_prior_for(dispersions: pd.DataFrame, stats: set[str]) -> pd.DataFrame:
    result = dispersions.copy()
    for stat in stats:
        target = result["stat"].eq(stat)
        log_dispersion = math.log(float(result.loc[target, "historical_dispersion"].iloc[0]))
        result.loc[target, "historical_dispersion_lower"] = math.exp(log_dispersion - 0.02)
        result.loc[target, "historical_dispersion_upper"] = math.exp(log_dispersion + 0.02)
        result.loc[target, "historical_log_dispersion_lower"] = log_dispersion - 0.02
        result.loc[target, "historical_log_dispersion_upper"] = log_dispersion + 0.02
    return result


def _cross_source_curves(
    stat: str,
    dispersion: float,
    thresholds: tuple[int, int],
    means: list[float],
) -> list[dict]:
    rows: list[dict] = []
    for number, mean in enumerate(means):
        player = f"{stat} Curve {number}"
        probabilities = negative_binomial_survival(thresholds, mean, dispersion)
        for source, threshold, probability in zip(
            ("draftkings", "kalshi"), thresholds, probabilities, strict=True,
        ):
            rows.append(_market(
                source, player, stat, threshold, float(probability),
                market_id=f"{source}-{stat}-{number}-{threshold}",
            ))
    return rows


def test_recovers_shared_kalshi_dispersion_and_player_means_with_map_update() -> None:
    markets, true_means = _kalshi_curves()
    result = estimate_market_means(
        markets, _dispersions(receiving_yards=6.0), _historical_report(), _config(),
    )

    assert result.validation["status"] == "passed"
    receiving = result.dispersions.set_index("stat").loc["receiving_yards"]
    assert receiving["method"] == "historical_plus_current_market_map"
    assert receiving["current_market_only_dispersion"] == pytest.approx(4.0, rel=0.03)
    assert 4.0 < receiving["final_dispersion"] < receiving["historical_dispersion"]
    assert receiving["current_market_group_count"] == len(true_means)
    assert set(result.dispersions.query("stat != 'receiving_yards'")["method"]) == {"historical_only"}

    projections = result.source_projections.set_index("canonical_player_id")
    assert len(projections) == len(true_means)
    assert set(projections["projection_method"]) == {"robust_multi_threshold_logit_fit"}
    assert set(projections["quality_status"]) == {"passed"}
    for player_id, true_mean in true_means.items():
        assert projections.loc[player_id, "mean"] == pytest.approx(true_mean, rel=0.08)
        assert projections.loc[player_id, "sensitivity_low"] <= projections.loc[player_id, "mean"]
        assert projections.loc[player_id, "sensitivity_high"] >= projections.loc[player_id, "mean"]
    included = result.priced_markets.query("model_inclusion_status == 'included'")
    assert included["modeled_probability"].notna().all()
    assert included["model_calibration_group_eligible"].all()
    assert included.sort_values("canonical_threshold").groupby("canonical_player_id")["modeled_probability"].apply(
        lambda values: np.all(np.diff(values.to_numpy(dtype=float)) <= 0)
    ).all()


def test_prior_conflict_warning_uses_validated_current_market_override() -> None:
    historical = 1.5
    sportsbook_probability = float(negative_binomial_survival(1001, 900.0, historical))
    markets = pd.DataFrame.from_records([
        *_steep_kalshi_curves().to_dict(orient="records"),
        _market(
            "draftkings", "Sportsbook Receiver", "receiving_yards", 1001,
            sportsbook_probability, market_id="dk-prior-conflict",
        ),
    ])

    result = estimate_market_means(
        markets, _narrow_historical_prior(receiving_yards=historical),
        _historical_report(), _config(),
    )

    receiving = result.dispersions.set_index("stat").loc["receiving_yards"]
    assert result.validation["status"] == "passed"
    assert receiving["method"] == "current_market_conflict_override"
    assert receiving["final_dispersion"] == pytest.approx(20.0, rel=0.05)
    assert receiving["fallback_reason"] == "historical_map_misspecification_current_market_override"
    assert receiving["current_market_only_dispersion"] == pytest.approx(20.0, rel=0.05)
    assert receiving["current_market_map_logit_rmse"] > _config()["current_market"]["max_current_market_logit_rmse"]

    projections = result.source_projections.set_index(["source", "canonical_player_name"])
    assert projections.loc[("draftkings", "Sportsbook Receiver"), "quality_status"] == "passed"
    kalshi = result.source_projections.query("source == 'kalshi'")
    assert set(kalshi["quality_status"]) == {"passed"}

    checks = {check["name"]: check for check in result.validation["checks"]}
    assert checks["model.receiving_yards.current_market_map_curve_residual"]["severity"] == "warning"
    assert checks["model.receiving_yards.current_market_map_threshold_holdout"]["severity"] == "warning"
    assert checks["model.kalshi_curves_quarantined"]["details"]["curves"] == 0


def test_prior_conflict_fail_policy_remains_a_hard_failure() -> None:
    config = _config()
    config["current_market"]["current_market_conflict_policy"] = "fail"
    result = estimate_market_means(
        _steep_kalshi_curves(), _narrow_historical_prior(), _historical_report(), config,
    )

    receiving = result.dispersions.set_index("stat").loc["receiving_yards"]
    checks = {check["name"]: check for check in result.validation["checks"]}
    assert result.validation["status"] == "failed"
    assert receiving["method"] == "historical_only"
    assert receiving["fallback_reason"] == "current_market_historical_conflict"
    assert checks["model.receiving_yards.current_market_map_curve_residual"]["severity"] == "error"
    assert checks["model.receiving_yards.current_market_map_threshold_holdout"]["severity"] == "error"


def test_current_market_shape_keeps_near_even_means_close_to_lines() -> None:
    markets = pd.DataFrame.from_records([
        *_cross_source_curves("passing_yards", 18.0, (3500, 4000), [3650.0, 3750.0, 3850.0, 3950.0]),
        *_cross_source_curves("receptions", 6.0, (80, 100), [86.0, 90.0, 94.0, 98.0]),
        _market("fanduel", "Drake Maye", "passing_yards", 3751, 0.5, market_id="fd-maye-3750"),
        _market("draftkings", "Trey McBride", "receptions", 98, 0.505, market_id="dk-mcbride-97.5"),
    ])
    dispersions = _narrow_prior_for(
        _dispersions(overrides={"passing_yards": 1.321611, "receptions": 2.070899}),
        {"passing_yards", "receptions"},
    )

    result = estimate_market_means(markets, dispersions, _historical_report(), _config())

    assert result.validation["status"] == "passed"
    projections = result.source_projections.set_index(["source", "canonical_player_name", "stat"])
    maye = projections.loc[("fanduel", "Drake Maye", "passing_yards"), "mean"]
    mcbride = projections.loc[("draftkings", "Trey McBride", "receptions"), "mean"]
    assert 3750.5 < maye < 3900.0
    assert 97.5 < mcbride < 105.0
    checks = {check["name"]: check for check in result.validation["checks"]}
    assert checks["model.near_even_market_mean_plausibility"]["passed"]
    fitted = result.dispersions.set_index("stat")
    assert fitted.loc["passing_yards", "method"] == "current_market_conflict_override"
    assert fitted.loc["receptions", "method"] == "current_market_conflict_override"


def test_near_even_plausibility_gate_rejects_historical_shape_failure() -> None:
    config = _config()
    config["dispersion_mode"] = "historical_only"
    markets = pd.DataFrame.from_records([
        _market("draftkings", "Trey McBride", "receptions", 98, 0.5, market_id="dk-bad-shape"),
    ])

    result = estimate_market_means(
        markets, _dispersions(overrides={"receptions": 2.070899}), _historical_report(), config,
    )

    assert result.validation["status"] == "failed"
    check = next(
        value for value in result.validation["checks"]
        if value["name"] == "model.near_even_market_mean_plausibility"
    )
    assert not check["passed"]
    assert check["details"]["violations"] == 1
    assert check["details"]["samples"][0]["canonical_player_name"] == "Trey McBride"


def test_bad_source_curve_is_quarantined_without_invalidating_good_curves() -> None:
    markets, _ = _kalshi_curves()
    outlier = pd.DataFrame.from_records([
        _market("kalshi", "Outlier Curve", "receiving_yards", 700, 0.95, market_id="outlier-700"),
        _market("kalshi", "Outlier Curve", "receiving_yards", 1000, 0.05, market_id="outlier-1000"),
    ])
    result = estimate_market_means(
        pd.concat([markets, outlier], ignore_index=True),
        _dispersions(), _historical_report(), _config(),
    )

    assert result.validation["status"] == "passed"
    projections = result.source_projections.set_index("canonical_player_name")
    assert set(projections.drop(index="Outlier Curve")["quality_status"]) == {"passed"}
    assert projections.loc["Outlier Curve", "quality_status"] == "excluded"
    assert "kalshi_curve_residual_exceeds_limit" in projections.loc["Outlier Curve", "exclusion_reason"]
    assert "kalshi_curve_holdout_exceeds_limit" in projections.loc["Outlier Curve", "exclusion_reason"]
    outlier_quotes = result.priced_markets.query("canonical_player_name == 'Outlier Curve'")
    assert set(outlier_quotes["source_projection_quality_status"]) == {"excluded"}
    assert outlier_quotes["source_projection_exclusion_reason"].str.contains(
        "kalshi_curve_residual_exceeds_limit"
    ).all()
    checks = {check["name"]: check for check in result.validation["checks"]}
    assert checks["model.kalshi_source_residuals"]["passed"]
    assert checks["model.kalshi_source_holdout"]["passed"]
    assert checks["model.kalshi_curves_quarantined"]["details"]["curves"] == 1


def test_historical_only_fallback_is_exact_and_single_quote_mean_is_reproducible() -> None:
    historical = 9.0
    true_mean = 875.0
    probability = float(negative_binomial_survival(801, true_mean, historical))
    markets = pd.DataFrame.from_records([
        _market("draftkings", "Example Receiver", "receiving_yards", 801, probability, market_id="dk-one"),
    ])
    result = estimate_market_means(markets, _dispersions(receiving_yards=historical), _historical_report(), _config())

    assert result.validation["status"] == "passed"
    assert set(result.dispersions["method"]) == {"historical_only"}
    assert np.array_equal(
        result.dispersions["final_dispersion"].to_numpy(dtype=float),
        result.dispersions["historical_dispersion"].to_numpy(dtype=float),
    )
    projection = result.source_projections.iloc[0]
    assert projection["mean"] == pytest.approx(true_mean, rel=1e-9)
    assert projection["fit_error"] <= 1e-9
    assert projection["sensitivity_label"] == "model_sensitivity_not_confidence_interval"
    quote = result.priced_markets.iloc[0]
    assert quote["modeled_probability"] == pytest.approx(probability, abs=1e-9)
    assert quote["probability_residual"] == pytest.approx(0.0, abs=1e-9)


def test_groups_below_minimum_threshold_count_cannot_update_dispersion() -> None:
    markets, _ = _kalshi_curves()
    markets = markets.groupby("canonical_player_id", sort=False).head(2).reset_index(drop=True)
    config = _config()
    config["minimum_thresholds_per_group"] = 3
    # This test isolates update eligibility from the deliberately misspecified
    # historical-dispersion source residuals produced by the two-point curves.
    config["current_market"]["max_kalshi_logit_rmse"] = 2.0
    config["current_market"]["max_kalshi_holdout_logit_mae"] = 2.0
    result = estimate_market_means(markets, _dispersions(), _historical_report(), config)
    receiving = result.dispersions.set_index("stat").loc["receiving_yards"]
    assert result.validation["status"] == "passed"
    assert receiving["method"] == "historical_only"
    assert receiving["final_dispersion"] == receiving["historical_dispersion"]
    assert receiving["current_market_group_count"] == 0
    assert receiving["fallback_reason"] == "insufficient_eligible_current_market_groups"


def test_excluded_quotes_never_contribute_and_remain_auditable() -> None:
    historical = 9.0
    probability = float(negative_binomial_survival(801, 875.0, historical))
    markets = pd.DataFrame.from_records([
        _market("draftkings", "Included Player", "receiving_yards", 801, probability, market_id="dk-included"),
        _market(
            "kalshi", "Excluded Player", "receiving_yards", 900, 0.5,
            market_id="kx-excluded", inclusion_status="excluded",
            exclusion_reason="missing_two_sided_yes_quote",
        ),
    ])
    result = estimate_market_means(markets, _dispersions(receiving_yards=historical), _historical_report(), _config())
    excluded_quote = result.priced_markets.query("source_market_id == 'kx-excluded'").iloc[0]
    excluded_projection = result.source_projections.query("canonical_player_name == 'Excluded Player'").iloc[0]
    assert result.validation["status"] == "passed"
    assert excluded_quote["model_inclusion_status"] == "excluded"
    assert pd.isna(excluded_quote["modeled_probability"])
    assert excluded_projection["quality_status"] == "excluded"
    assert excluded_projection["exclusion_reason"] == "missing_two_sided_yes_quote"


def test_duplicate_kalshi_thresholds_and_impossible_mean_bounds_fail_validation() -> None:
    markets, _ = _kalshi_curves()
    duplicate = markets.iloc[[0]].copy()
    duplicate.loc[:, "source_market_id"] = "different-contract-same-threshold"
    duplicated = estimate_market_means(
        pd.concat([markets, duplicate], ignore_index=True),
        _dispersions(), _historical_report(), _config(),
    )
    assert duplicated.validation["status"] == "failed"
    check = next(value for value in duplicated.validation["checks"] if value["name"] == "model.current_market_thresholds_unique_within_source")
    assert not check["passed"]

    config = _config()
    config["current_market"]["mean_bounds"]["receiving_yards"] = [0.01, 10.0]
    probability = float(negative_binomial_survival(801, 875.0, 9.0))
    impossible = estimate_market_means(
        pd.DataFrame.from_records([_market("fanduel", "Bound Player", "receiving_yards", 801, probability, market_id="fd-bound")]),
        _dispersions(), _historical_report(), config,
    )
    assert impossible.validation["status"] == "failed"
    assert impossible.source_projections.iloc[0]["quality_status"] == "failed"
    assert "not attainable" in impossible.source_projections.iloc[0]["exclusion_reason"]


def test_shared_dispersion_optimizer_nonconvergence_is_exposed(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailedResult:
        x = np.log(np.asarray([8.0, 700.0]))
        fun = 10.0
        success = False
        status = 0
        message = "synthetic failure"
        nit = 1
        nfev = 2

    monkeypatch.setattr("ff_market_projections.modeling.minimize", lambda *args, **kwargs: FailedResult())
    probabilities = negative_binomial_survival([400, 700, 1000], 700.0, 4.0)
    fit = fit_shared_kalshi_dispersion(
        {"player": (np.asarray([400, 700, 1000]), probabilities)},
        8.0, (0.05, 1000.0), (0.01, 5000.0), robust_loss="soft_l1",
        optimizer_tolerance=1e-10, max_evaluations=100, prior_log_sd=0.5,
    )
    assert not fit.converged
    assert fit.optimizer_message == "synthetic failure"


def _write_cli_inputs(run_dir: Path, *, mean: float = 875.0) -> None:
    probability = float(negative_binomial_survival(801, mean, 9.0))
    pd.DataFrame.from_records([
        _market("draftkings", "CLI Receiver", "receiving_yards", 801, probability, market_id="dk-cli"),
    ]).to_csv(run_dir / "artifacts/priced_markets.csv", index=False)
    _dispersions(receiving_yards=9.0).to_csv(run_dir / "artifacts/dispersion_calibration.csv", index=False)
    (run_dir / "artifacts/historical_calibration.json").write_text(json.dumps(_historical_report()), encoding="utf-8")


def test_cli_updates_calibration_and_writes_source_projection_validation_and_manifest(tmp_path: Path) -> None:
    config = load_config(ROOT / "config/pipeline.toml")
    run_dir = initialize_run(config, ROOT / "config/player_aliases.csv", tmp_path / "runs")
    _write_cli_inputs(run_dir)
    command = [sys.executable, str(ROOT / "scripts/estimate_means.py"), "--run-dir", str(run_dir)]
    completed = subprocess.run(
        command, text=True, capture_output=True, check=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    summary = json.loads(completed.stdout)
    projections = pd.read_csv(run_dir / "artifacts/source_projections.csv")
    dispersions = pd.read_csv(run_dir / "artifacts/dispersion_calibration.csv")
    historical = json.loads((run_dir / "artifacts/historical_calibration.json").read_text())
    validation = json.loads((run_dir / "artifacts/model_validation.json").read_text())
    manifest = json.loads((run_dir / "metadata/manifest.json").read_text())
    status = json.loads((run_dir / "metadata/run_status.json").read_text())

    assert summary["state"] == "succeeded"
    assert len(projections) == 1
    assert projections.iloc[0]["mean"] == pytest.approx(875.0, rel=1e-8)
    assert set(dispersions["method"]) == {"historical_only"}
    assert historical["current_market_update"]["status"] == "passed"
    assert validation["status"] == "passed"
    assert validation["artifacts"]["source_projections.csv"]["rows"] == 1
    assert manifest["task_dag"]["estimate_means"] == ["price_markets", "calibrate_distributions"]
    assert manifest["tasks"]["estimate_means"]["state"] == "succeeded"
    assert status["state"] == "running" and status["model_state"] == "succeeded"

    repeated = subprocess.run(command, text=True, capture_output=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert repeated.returncode != 0
    assert "artifacts are immutable" in repeated.stderr


def test_cli_failure_preserves_model_inputs_and_writes_failure_evidence(tmp_path: Path) -> None:
    config_path = tmp_path / "bounded.toml"
    config_path.write_text(
        (ROOT / "config/pipeline.toml").read_text().replace(
            "receiving_yards = [0.01, 5000.0]", "receiving_yards = [0.01, 10.0]"
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    run_dir = initialize_run(config, ROOT / "config/player_aliases.csv", tmp_path / "runs")
    _write_cli_inputs(run_dir)
    priced_path = run_dir / "artifacts/priced_markets.csv"
    dispersion_path = run_dir / "artifacts/dispersion_calibration.csv"
    input_hashes = (sha256_file(priced_path), sha256_file(dispersion_path))
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/estimate_means.py"), "--run-dir", str(run_dir)],
        text=True, capture_output=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    validation = json.loads((run_dir / "artifacts/model_validation.json").read_text())
    status = json.loads((run_dir / "metadata/run_status.json").read_text())
    assert completed.returncode == 1
    assert validation["status"] == "failed"
    assert status["state"] == "failed" and status["failed_task"] == "estimate_means"
    assert not (run_dir / "artifacts/source_projections.csv").exists()
    assert (sha256_file(priced_path), sha256_file(dispersion_path)) == input_hashes
