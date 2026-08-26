#!/usr/bin/env python3
"""Update dispersion from eligible current-market curves and estimate source means."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd

from ff_market_projections.config import ConfigError, load_config
from ff_market_projections.contracts import atomic_write_bytes, atomic_write_json, sha256_file, utc_now
from ff_market_projections.modeling import ModelingError, estimate_market_means


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode("utf-8")


def _mark_failed(
    run_dir: Path,
    started_utc: str,
    elapsed: float,
    reason: str,
    validation_path: Path,
) -> None:
    ended_utc = utc_now()
    manifest_path = run_dir / "metadata" / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["state"] = "failed"
    manifest["ended_utc"] = ended_utc
    manifest.setdefault("task_dag", {})["estimate_means"] = ["price_markets", "calibrate_distributions"]
    manifest.setdefault("tasks", {})["estimate_means"] = {
        "state": "failed",
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "duration_seconds": elapsed,
        "reason": reason,
        "validation": str(validation_path),
    }
    atomic_write_json(manifest_path, manifest)
    status_path = run_dir / "metadata" / "run_status.json"
    prior = _read_json(status_path)
    atomic_write_json(status_path, {
        "run_id": run_dir.name,
        "state": "failed",
        "started_utc": prior.get("started_utc", started_utc),
        "ended_utc": ended_utc,
        "failed_task": "estimate_means",
        "reason": reason,
    })


def _mark_succeeded(
    run_dir: Path,
    started_utc: str,
    elapsed: float,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    validation_path: Path,
    summary: dict[str, Any],
) -> None:
    ended_utc = utc_now()
    manifest_path = run_dir / "metadata" / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest.setdefault("task_dag", {})["estimate_means"] = ["price_markets", "calibrate_distributions"]
    manifest.setdefault("tasks", {})["estimate_means"] = {
        "state": "succeeded",
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "duration_seconds": elapsed,
        "command": [sys.executable, *sys.argv],
        "inputs": inputs,
        "outputs": outputs,
        "validation": str(validation_path),
    }
    manifest["model_state"] = "succeeded"
    manifest["historical_calibration"] = {
        **manifest.get("historical_calibration", {}),
        "method": "historical_with_current_market_update",
        "historical_only_stats": summary["historical_only_stats"],
        "historical_plus_current_market_stats": summary["historical_plus_current_market_stats"],
        "current_market_conflict_override_stats": summary["current_market_conflict_override_stats"],
        "market_update_version": summary["market_update_version"],
    }
    atomic_write_json(manifest_path, manifest)
    status_path = run_dir / "metadata" / "run_status.json"
    status = _read_json(status_path)
    status.update({"state": "running", "model_state": "succeeded"})
    atomic_write_json(status_path, status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input", type=Path, help="priced_markets.csv input")
    parser.add_argument("--dispersion-input", type=Path)
    parser.add_argument("--historical-report-input", type=Path)
    parser.add_argument("--output", type=Path, help="source_projections.csv output")
    parser.add_argument("--validation-output", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config_path = (args.config or run_dir / "config" / "effective.toml").resolve()
    priced_path = (args.input or run_dir / "artifacts" / "priced_markets.csv").resolve()
    dispersion_path = (args.dispersion_input or run_dir / "artifacts" / "dispersion_calibration.csv").resolve()
    historical_path = (args.historical_report_input or run_dir / "artifacts" / "historical_calibration.json").resolve()
    output_path = (args.output or run_dir / "artifacts" / "source_projections.csv").resolve()
    validation_path = (args.validation_output or run_dir / "artifacts" / "model_validation.json").resolve()
    paths = (config_path, priced_path, dispersion_path, historical_path, output_path, validation_path)
    if not all(_inside(path, run_dir) for path in paths):
        parser.error("model inputs and artifacts must be inside the active run directory")
    if config_path != run_dir / "config" / "effective.toml":
        parser.error("mean estimation must use the run-scoped effective config")
    for path in (priced_path, dispersion_path, historical_path):
        if not path.is_file() or path.stat().st_size == 0:
            parser.error(f"model input is missing or empty: {path}")
    existing = [str(path) for path in (output_path, validation_path) if path.exists()]
    if existing:
        parser.error(f"model artifacts are immutable and already exist: {', '.join(existing)}")

    started_utc = utc_now()
    started = time.perf_counter()
    input_metadata = {
        path.name: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in (priced_path, dispersion_path, historical_path)
    }
    try:
        config = load_config(config_path)
        priced = pd.read_csv(priced_path, low_memory=False)
        dispersions = pd.read_csv(dispersion_path, low_memory=False)
        historical_report = _read_json(historical_path)
        result = estimate_market_means(priced, dispersions, historical_report, config.values["model"])
        validation = result.validation
        failed_warnings = [
            check for check in validation["checks"]
            if check["severity"] == "warning" and not check["passed"]
        ]
        if config.values["run"]["fail_on_warning"] and failed_warnings:
            validation = {**validation, "status": "failed", "fail_on_warning": True}
        if validation["status"] != "passed":
            raise ModelingError("One or more current-market model validations failed", validation)
    except (ConfigError, OSError, pd.errors.ParserError, json.JSONDecodeError) as exc:
        validation = {"status": "failed", "checks": [], "error": str(exc)}
        atomic_write_json(validation_path, validation)
        _mark_failed(run_dir, started_utc, time.perf_counter() - started, str(exc), validation_path)
        print(json.dumps({"state": "failed", "reason": str(exc), "model_validation": str(validation_path)}, sort_keys=True))
        raise SystemExit(1) from exc
    except ModelingError as exc:
        validation = exc.validation
        atomic_write_json(validation_path, validation)
        _mark_failed(run_dir, started_utc, time.perf_counter() - started, str(exc), validation_path)
        print(json.dumps({"state": "failed", "reason": str(exc), "model_validation": str(validation_path)}, sort_keys=True))
        raise SystemExit(1) from exc

    priced_hash = atomic_write_bytes(priced_path, _csv_bytes(result.priced_markets))
    dispersion_hash = atomic_write_bytes(dispersion_path, _csv_bytes(result.dispersions))
    historical_hash = atomic_write_json(historical_path, result.historical_report)
    projection_hash = atomic_write_bytes(output_path, _csv_bytes(result.source_projections))
    validation = {
        **result.validation,
        "artifacts": {
            priced_path.name: {"path": str(priced_path), "sha256": priced_hash, "rows": int(len(result.priced_markets))},
            dispersion_path.name: {"path": str(dispersion_path), "sha256": dispersion_hash, "rows": int(len(result.dispersions))},
            historical_path.name: {"path": str(historical_path), "sha256": historical_hash},
            output_path.name: {"path": str(output_path), "sha256": projection_hash, "rows": int(len(result.source_projections))},
        },
    }
    validation_hash = atomic_write_json(validation_path, validation)
    outputs = {
        **validation["artifacts"],
        validation_path.name: {"path": str(validation_path), "sha256": validation_hash},
    }
    summary = {
        **validation["summary"],
        "market_update_version": validation["market_update_version"],
    }
    _mark_succeeded(
        run_dir, started_utc, time.perf_counter() - started,
        input_metadata, outputs, validation_path, summary,
    )
    print(json.dumps({
        "state": "succeeded",
        "source_projections": str(output_path),
        "source_projection_rows": len(result.source_projections),
        "eligible_source_projections": validation["summary"]["eligible_source_projections"],
        "historical_plus_current_market_stats": validation["summary"]["historical_plus_current_market_stats"],
        "current_market_conflict_override_stats": validation["summary"]["current_market_conflict_override_stats"],
        "model_validation": str(validation_path),
        "model_validation_sha256": validation_hash,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
