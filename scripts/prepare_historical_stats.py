#!/usr/bin/env python3
"""Prepare and validate leakage-free historical player-season baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from ff_market_projections.config import ConfigError, load_config
from ff_market_projections.contracts import atomic_write_bytes, atomic_write_json, sha256_file, utc_now
from ff_market_projections.historical import HistoricalDataError, prepare_historical_data


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _csv_bytes(frame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.10g").encode("utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mark_failed(run_dir: Path, started_utc: str, elapsed: float, error: HistoricalDataError, validation_path: Path) -> None:
    ended_utc = utc_now()
    manifest_path = run_dir / "metadata/manifest.json"
    manifest = _read_json(manifest_path)
    manifest["state"] = "failed"
    manifest["ended_utc"] = ended_utc
    manifest.setdefault("task_dag", {})["prepare_historical_stats"] = ["collect_nflverse_history"]
    manifest.setdefault("tasks", {})["prepare_historical_stats"] = {
        "state": "failed", "started_utc": started_utc, "ended_utc": ended_utc,
        "duration_seconds": elapsed, "failed_check": error.check.name,
        "reason": str(error), "validation": str(validation_path),
    }
    atomic_write_json(manifest_path, manifest)
    prior_status = _read_json(run_dir / "metadata/run_status.json")
    atomic_write_json(run_dir / "metadata/run_status.json", {
        "run_id": run_dir.name, "state": "failed",
        "started_utc": prior_status.get("started_utc", started_utc), "ended_utc": ended_utc,
        "failed_task": "prepare_historical_stats", "reason": str(error),
    })


def _mark_succeeded(run_dir: Path, started_utc: str, elapsed: float, validation: dict, validation_path: Path) -> None:
    ended_utc = utc_now()
    manifest_path = run_dir / "metadata/manifest.json"
    manifest = _read_json(manifest_path)
    manifest.setdefault("task_dag", {})["prepare_historical_stats"] = ["collect_nflverse_history"]
    manifest.setdefault("tasks", {})["prepare_historical_stats"] = {
        "state": "succeeded", "started_utc": started_utc, "ended_utc": ended_utc,
        "duration_seconds": elapsed, "command": [sys.executable, *sys.argv],
        "input": validation["input"], "outputs": validation["artifacts"],
        "validation": str(validation_path), "validation_sha256": sha256_file(validation_path),
    }
    manifest["historical"] = {
        "source": "nflverse_player_stats_release",
        "raw": validation["input"],
        "covered_seasons": validation["covered_seasons"],
        "calibration_window": validation["calibration_window"],
        "cohort_counts": validation["cohort_counts"],
        "prior_opportunity_filters": validation["prior_opportunity_filters"],
    }
    manifest["historical_state"] = "succeeded"
    atomic_write_json(manifest_path, manifest)
    status_path = run_dir / "metadata/run_status.json"
    status = _read_json(status_path)
    status.update({"state": "running", "historical_state": "succeeded"})
    atomic_write_json(status_path, status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backtest-output", type=Path)
    parser.add_argument("--validation-output", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output = (args.output or run_dir / "artifacts" / "historical_player_seasons.csv").resolve()
    backtest_output = (args.backtest_output or run_dir / "artifacts" / "historical_backtest_predictions.csv").resolve()
    validation_output = (args.validation_output or run_dir / "artifacts" / "historical_validation.json").resolve()
    paths = (args.config.resolve(), args.input.resolve(), output, backtest_output, validation_output)
    if not all(_inside(path, run_dir) for path in paths):
        parser.error("config, input, and outputs must all be inside the active run directory")
    if args.config.resolve() != run_dir / "config" / "effective.toml":
        parser.error("downstream historical preparation must use run-scoped config/effective.toml")
    if not args.input.is_file() or args.input.stat().st_size == 0:
        parser.error(f"historical input is missing or empty: {args.input}")
    existing = [str(path) for path in (output, backtest_output, validation_output) if path.exists()]
    if existing:
        parser.error(f"historical artifacts are immutable and already exist: {', '.join(existing)}")

    started_utc = utc_now()
    started = time.perf_counter()
    try:
        config = load_config(args.config)
        prepared = prepare_historical_data(args.input, config.values["historical"])
        failed_warnings = [
            check for check in prepared.validation["checks"]
            if check["severity"] == "warning" and not check["passed"]
        ]
        if config.values["run"]["fail_on_warning"] and failed_warnings:
            warning = failed_warnings[0]
            validation = {**prepared.validation, "status": "failed", "fail_on_warning": True}
            raise HistoricalDataError(warning["name"], f"Warning promoted to error: {warning['message']}", {**warning["details"], "validation": validation})
    except ConfigError as exc:
        parser.error(str(exc))
    except HistoricalDataError as exc:
        validation = exc.check.details.get("validation") if isinstance(exc.check.details, dict) else None
        if not isinstance(validation, dict):
            validation = {"status": "failed", "checks": [exc.check.to_dict()], "error": str(exc)}
        atomic_write_json(validation_output, validation)
        _mark_failed(run_dir, started_utc, time.perf_counter() - started, exc, validation_output)
        print(json.dumps({"state": "failed", "validation": str(validation_output), "failed_check": exc.check.name, "reason": str(exc)}, sort_keys=True))
        raise SystemExit(1) from exc

    player_hash = atomic_write_bytes(output, _csv_bytes(prepared.player_seasons))
    backtest_hash = atomic_write_bytes(backtest_output, _csv_bytes(prepared.backtest_predictions))
    validation = {
        **prepared.validation,
        "artifacts": {
            "historical_player_seasons.csv": {"path": str(output), "sha256": player_hash, "rows": int(len(prepared.player_seasons))},
            "historical_backtest_predictions.csv": {"path": str(backtest_output), "sha256": backtest_hash, "rows": int(len(prepared.backtest_predictions))},
        },
        "input": {"path": str(args.input.resolve()), "sha256": sha256_file(args.input), "bytes": args.input.stat().st_size},
    }
    validation_hash = atomic_write_json(validation_output, validation)
    _mark_succeeded(run_dir, started_utc, time.perf_counter() - started, validation, validation_output)
    print(json.dumps({
        "state": "succeeded",
        "historical_player_season_rows": len(prepared.player_seasons),
        "backtest_prediction_rows": len(prepared.backtest_predictions),
        "validation": str(validation_output),
        "validation_sha256": validation_hash,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
