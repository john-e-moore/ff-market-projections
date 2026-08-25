#!/usr/bin/env python3
"""Create an isolated run and collect its immutable Phase 1 inputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from ff_market_projections.config import ConfigError, load_config
from ff_market_projections.contracts import atomic_write_json, sha256_file, utc_now
from ff_market_projections.dag import Task, TaskState, run_dag
from ff_market_projections.runs import initialize_run


ROOT = Path(__file__).resolve().parents[1]


class CollectorError(RuntimeError):
    def __init__(self, message: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details


def _write_log(path: Path, stdout: str, stderr: str) -> None:
    path.write_text("[stdout]\n" + stdout + "\n[stderr]\n" + stderr, encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Expected JSON output is unreadable: {path}: {exc}") from exc


def _snapshot_utc(path: Path) -> str | None:
    return _read_json(path).get("metadata", {}).get("snapshot_utc")


def _copy_offline(source: Path, output: Path, *, historical: bool = False) -> dict[str, Any]:
    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError(f"Offline input is missing or empty: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    if historical:
        with gzip.open(output, "rb") as handle:
            while handle.read(1024 * 1024):
                pass
    return {"command": ["offline-copy", str(source), str(output)], "exit_code": 0, "stdout": "", "stderr": "", "offline_input": str(source)}


def _run_command(command: list[str], output: Path, log: Path, timeout_seconds: float) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\nTimed out after {timeout_seconds} seconds."
        _write_log(log, stdout, stderr)
        raise CollectorError(f"Collector timed out after {timeout_seconds} seconds", {"command": command, "exit_code": None, "log": str(log)}) from exc
    _write_log(log, completed.stdout, completed.stderr)
    if completed.returncode != 0:
        raise CollectorError(f"Collector exited with code {completed.returncode}", {"command": command, "exit_code": completed.returncode, "log": str(log)})
    if not output.is_file() or output.stat().st_size == 0:
        raise CollectorError("Collector exited successfully but did not produce a nonempty output", {"command": command, "exit_code": completed.returncode, "log": str(log)})
    return {"command": command, "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def _collector_task(name: str, script: str, output: Path, log: Path, arguments: list[str], timeout_seconds: float, offline_input: Path | None, *, historical: bool = False) -> dict[str, Any]:
    if offline_input is not None:
        command_result = _copy_offline(offline_input, output, historical=historical)
        _write_log(log, "offline input copied\n", "")
    else:
        command_result = _run_command([sys.executable, str(ROOT / "scripts" / script), *arguments, "--output", str(output)], output, log, timeout_seconds)
    details: dict[str, Any] = {
        "command": command_result["command"], "exit_code": command_result["exit_code"],
        "output": str(output), "output_sha256": sha256_file(output), "output_bytes": output.stat().st_size,
        "log": str(log),
    }
    if historical:
        with gzip.open(output, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            if "season" not in header:
                raise RuntimeError("nflverse history is missing its season column")
            seasons = sorted({int(row["season"]) for row in reader if (row.get("season") or "").isdigit()})
        if not seasons:
            raise RuntimeError("nflverse history has no parseable seasons")
        details["historical"] = {"gzip_valid": True, "covered_seasons": seasons, "columns": header}
        if offline_input is None:
            try:
                collector_summary = json.loads(command_result["stdout"])
                details["historical"]["response"] = collector_summary.get("response", {})
                if collector_summary.get("cache") is not None:
                    details["historical"]["cache"] = collector_summary["cache"]
            except json.JSONDecodeError:
                pass
    else:
        details["source_snapshot_utc"] = _snapshot_utc(output)
    return details


def _update_run(run_dir: Path, results: dict[str, Any], offline: bool) -> None:
    manifest_path = run_dir / "metadata" / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest.update({
        "mode": "offline_fixture" if offline else "live",
        "task_dag": {name: [] for name in sorted(results)},
        "tasks": {name: {"state": result.state.value, **result.details} for name, result in sorted(results.items())},
        "ended_utc": utc_now(),
    })
    snapshots = [result.details.get("source_snapshot_utc") for result in results.values() if result.state == TaskState.SUCCEEDED and result.details.get("source_snapshot_utc")]
    manifest["source_snapshot_times_utc"] = sorted(snapshots)
    failures = [result for result in results.values() if result.state != TaskState.SUCCEEDED]
    if failures:
        failed = sorted(failures, key=lambda item: item.name)[0]
        manifest["state"] = "failed"
        status = {"run_id": run_dir.name, "state": "failed", "started_utc": manifest["started_utc"], "ended_utc": manifest["ended_utc"], "failed_task": failed.name, "reason": failed.details.get("reason", "dependency_failed")}
    else:
        # Phase 1 stops after collection; later phases own completed-run finalization.
        manifest["collection_state"] = "succeeded"
        status = {"run_id": run_dir.name, "state": "running", "started_utc": manifest["started_utc"], "collection_state": "succeeded"}
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(run_dir / "metadata" / "run_status.json", status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/pipeline.toml"))
    parser.add_argument("--aliases", type=Path, default=Path("config/player_aliases.csv"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--offline-input-dir", type=Path)
    parser.add_argument("--collector-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--historical-cache-dir", type=Path, default=ROOT / "data" / "cache" / "nflverse_player_stats")
    parser.add_argument("--refresh-historical-cache", action="store_true")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        run_dir = initialize_run(config, args.aliases, args.runs_dir)
    except (ConfigError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
    offline = args.offline_input_dir is not None
    raw = run_dir / "raw"
    logs = run_dir / "logs"
    inputs = args.offline_input_dir
    source_config = config.values["sources"]
    tasks = [
        Task("collect_draftkings", lambda: _collector_task("draftkings", "fetch_draftkings_nfl_player_stats_ou.py", raw / "draftkings.json", logs / "collect_draftkings.log", [], args.collector_timeout_seconds, inputs / "draftkings.json" if inputs else None)),
        Task("collect_fanduel", lambda: _collector_task("fanduel", "fetch_fanduel_nfl_player_season_props.py", raw / "fanduel.json", logs / "collect_fanduel.log", ["--state", source_config["fanduel"]["state"], "--timezone", config.values["run"]["timezone"]], args.collector_timeout_seconds, inputs / "fanduel.json" if inputs else None)),
        Task("collect_kalshi", lambda: _collector_task("kalshi", "fetch_kalshi_nfl_season_stats.py", raw / "kalshi.json", logs / "collect_kalshi.log", [], args.collector_timeout_seconds, inputs / "kalshi.json" if inputs else None)),
        Task("collect_nflverse_history", lambda: _collector_task("nflverse", "fetch_nflverse_player_history.py", raw / "nflverse_player_stats.csv.gz", logs / "collect_nflverse_history.log", ["--url", config.values["historical"]["url"], "--cache-dir", str(args.historical_cache_dir), *( ["--refresh-cache"] if args.refresh_historical_cache else [])], args.collector_timeout_seconds, inputs / "nflverse_player_stats.csv.gz" if inputs else None, historical=True)),
    ]
    results = run_dag(tasks)
    _update_run(run_dir, results, offline)
    failed = [name for name, result in results.items() if result.state != TaskState.SUCCEEDED]
    print(json.dumps({"run_dir": str(run_dir.resolve()), "state": "failed" if failed else "running", "collection_state": "failed" if failed else "succeeded", "offline": offline, "failed_tasks": sorted(failed)}, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
