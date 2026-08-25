from __future__ import annotations

import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ff_market_projections.dag import Task, TaskState, run_dag, topological_order


ROOT = Path(__file__).parents[1]


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_pipeline", ROOT / "scripts/run_pipeline.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_topological_order_is_deterministic_and_failure_blocks_descendants() -> None:
    called: list[str] = []

    def fail() -> dict:
        called.append("a")
        raise RuntimeError("broken")

    tasks = [Task("c", lambda: called.append("c") or {}, ("b",)), Task("b", lambda: called.append("b") or {}, ("a",)), Task("a", fail)]
    assert topological_order(tasks) == ["a", "b", "c"]
    results = run_dag(tasks)
    assert called == ["a"]
    assert results["a"].state is TaskState.FAILED
    assert results["b"].state is TaskState.BLOCKED
    assert results["c"].state is TaskState.BLOCKED


@pytest.mark.parametrize("kind", ["nonzero", "missing_output", "timeout"])
def test_collector_subprocess_failures_are_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    runner = _load_runner()
    output, log = tmp_path / "raw.json", tmp_path / "collector.log"

    def fake_run(*_args, **_kwargs):
        if kind == "timeout":
            raise subprocess.TimeoutExpired(["collector"], 1, output="some output", stderr="some error")
        return subprocess.CompletedProcess(["collector"], 1 if kind == "nonzero" else 0, "out", "err")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError):
        runner._run_command(["collector"], output, log, 1)
    contents = log.read_text()
    assert "[stdout]" in contents and "[stderr]" in contents


def test_collector_subprocess_success_records_output_and_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _load_runner()
    output, log = tmp_path / "raw.json", tmp_path / "collector.log"

    def fake_run(*_args, **_kwargs):
        output.write_text('{"ok": true}')
        return subprocess.CompletedProcess(["collector"], 0, "done", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner._run_command(["collector"], output, log, 1)
    assert result["exit_code"] == 0
    assert "done" in log.read_text()


def test_offline_collection_creates_labeled_isolated_run(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "inputs"
    fixture_dir.mkdir()
    for name in ("draftkings.json", "fanduel.json", "kalshi.json"):
        (fixture_dir / name).write_bytes((ROOT / "tests/fixtures" / name).read_bytes())
    with gzip.open(fixture_dir / "nflverse_player_stats.csv.gz", "wb") as handle:
        handle.write((ROOT / "tests/fixtures/nflverse_player_stats.csv").read_bytes())
    command = [
        sys.executable, str(ROOT / "scripts/run_pipeline.py"), "--config", str(ROOT / "config/pipeline.toml"),
        "--aliases", str(ROOT / "config/player_aliases.csv"), "--runs-dir", str(tmp_path / "runs"),
        "--offline-input-dir", str(fixture_dir),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, check=True)
    summary = json.loads(completed.stdout)
    run_dir = Path(summary["run_dir"])
    manifest = json.loads((run_dir / "metadata/manifest.json").read_text())
    status = json.loads((run_dir / "metadata/run_status.json").read_text())
    assert summary["offline"] is True
    assert manifest["mode"] == "offline_fixture"
    assert status["state"] == "running"
    assert status["collection_state"] == "succeeded"
    assert set(manifest["tasks"]) == {"collect_draftkings", "collect_fanduel", "collect_kalshi", "collect_nflverse_history"}
    for task in manifest["tasks"].values():
        assert task["state"] == "succeeded"
        assert Path(task["output"]).is_file()
        assert Path(task["log"]).is_file()
