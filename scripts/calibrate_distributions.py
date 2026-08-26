#!/usr/bin/env python3
"""Calibrate and validate historical predictive dispersion for every target stat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd

from ff_market_projections.calibration import CalibrationError, calibrate_historical_distributions
from ff_market_projections.config import ConfigError, load_config
from ff_market_projections.contracts import atomic_write_bytes, atomic_write_json, sha256_file, utc_now


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode("utf-8")


def _mark_run_failed(run_dir: Path, started_utc: str, elapsed: float, reason: str, report_path: Path) -> None:
    ended_utc = utc_now()
    manifest_path = run_dir / "metadata" / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["state"] = "failed"
    manifest["ended_utc"] = ended_utc
    manifest.setdefault("task_dag", {})["calibrate_distributions"] = ["prepare_historical_stats"]
    manifest.setdefault("tasks", {})["calibrate_distributions"] = {
        "state": "failed",
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "duration_seconds": elapsed,
        "reason": reason,
        "historical_calibration": str(report_path),
    }
    atomic_write_json(manifest_path, manifest)
    status_path = run_dir / "metadata" / "run_status.json"
    prior = _read_json(status_path)
    atomic_write_json(status_path, {
        "run_id": run_dir.name,
        "state": "failed",
        "started_utc": prior.get("started_utc", started_utc),
        "ended_utc": ended_utc,
        "failed_task": "calibrate_distributions",
        "reason": reason,
    })


def _mark_run_succeeded(
    run_dir: Path,
    started_utc: str,
    elapsed: float,
    input_path: Path,
    output_path: Path,
    report_path: Path,
    rows: int,
    calibration_version: str,
) -> None:
    ended_utc = utc_now()
    manifest_path = run_dir / "metadata" / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest.setdefault("task_dag", {})["calibrate_distributions"] = ["prepare_historical_stats"]
    manifest.setdefault("tasks", {})["calibrate_distributions"] = {
        "state": "succeeded",
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "duration_seconds": elapsed,
        "command": [sys.executable, *sys.argv],
        "input": {"path": str(input_path), "sha256": sha256_file(input_path)},
        "outputs": {
            output_path.name: {"path": str(output_path), "sha256": sha256_file(output_path), "rows": rows},
            report_path.name: {"path": str(report_path), "sha256": sha256_file(report_path)},
        },
        "calibration_version": calibration_version,
    }
    manifest["historical_calibration"] = {
        "state": "succeeded",
        "calibration_version": calibration_version,
        "method": "historical_only",
        "stats": rows,
    }
    atomic_write_json(manifest_path, manifest)
    status_path = run_dir / "metadata" / "run_status.json"
    status = _read_json(status_path)
    status.update({"state": "running", "historical_calibration_state": "succeeded"})
    atomic_write_json(status_path, status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config_path = (args.config or run_dir / "config" / "effective.toml").resolve()
    input_path = (args.input or run_dir / "artifacts" / "historical_backtest_predictions.csv").resolve()
    output_path = (args.output or run_dir / "artifacts" / "dispersion_calibration.csv").resolve()
    report_path = (args.report_output or run_dir / "artifacts" / "historical_calibration.json").resolve()
    if not all(_inside(path, run_dir) for path in (config_path, input_path, output_path, report_path)):
        parser.error("calibration inputs and artifacts must be inside the active run directory")
    if config_path != run_dir / "config" / "effective.toml":
        parser.error("calibration must use the run-scoped effective config")
    if not input_path.is_file() or input_path.stat().st_size == 0:
        parser.error(f"historical backtest input is missing or empty: {input_path}")
    existing = [str(path) for path in (output_path, report_path) if path.exists()]
    if existing:
        parser.error(f"calibration artifacts are immutable and already exist: {', '.join(existing)}")

    started_utc = utc_now()
    started = time.perf_counter()
    report: dict = {"status": "failed", "checks": []}
    try:
        config = load_config(config_path)
        predictions = pd.read_csv(input_path, low_memory=False)
        result = calibrate_historical_distributions(predictions, config.values["historical"], config.values["model"])
        report = {
            **result.report,
            "input": {"path": str(input_path), "sha256": sha256_file(input_path), "rows": int(len(predictions))},
        }
        failed_warnings = [check for check in report["checks"] if check["severity"] == "warning" and not check["passed"]]
        if config.values["run"]["fail_on_warning"] and failed_warnings:
            report["status"] = "failed"
            report["fail_on_warning"] = True
        if report["status"] != "passed":
            raise CalibrationError("historical_calibration.validation", "One or more historical calibrations failed validation", {"report": report})
    except (ConfigError, OSError, pd.errors.ParserError) as exc:
        report = {"status": "failed", "checks": [], "error": str(exc)}
        atomic_write_json(report_path, report)
        _mark_run_failed(run_dir, started_utc, time.perf_counter() - started, str(exc), report_path)
        print(json.dumps({"state": "failed", "reason": str(exc), "historical_calibration": str(report_path)}, sort_keys=True))
        raise SystemExit(1) from exc
    except CalibrationError as exc:
        embedded = exc.check.details.get("report") if isinstance(exc.check.details, dict) else None
        if isinstance(embedded, dict):
            report = embedded
        elif "error" not in report:
            report = {"status": "failed", "checks": [exc.check.to_dict()], "error": str(exc)}
        atomic_write_json(report_path, report)
        _mark_run_failed(run_dir, started_utc, time.perf_counter() - started, str(exc), report_path)
        print(json.dumps({"state": "failed", "reason": str(exc), "historical_calibration": str(report_path)}, sort_keys=True))
        raise SystemExit(1) from exc

    dispersion_hash = atomic_write_bytes(output_path, _csv_bytes(result.dispersions))
    report["artifacts"] = {
        output_path.name: {"path": str(output_path), "sha256": dispersion_hash, "rows": int(len(result.dispersions))},
    }
    report_hash = atomic_write_json(report_path, report)
    _mark_run_succeeded(
        run_dir, started_utc, time.perf_counter() - started, input_path, output_path,
        report_path, len(result.dispersions), str(report["calibration_version"]),
    )
    print(json.dumps({
        "state": "succeeded",
        "calibrated_stats": len(result.dispersions),
        "calibration_version": report["calibration_version"],
        "dispersion_calibration": str(output_path),
        "historical_calibration": str(report_path),
        "historical_calibration_sha256": report_hash,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
