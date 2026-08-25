"""Run-directory bootstrap with no-overwrite guarantees."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import platform
import secrets
import sys
import time

from .config import PipelineConfig
from .contracts import atomic_write_bytes, atomic_write_json, sha256_file, utc_now


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(4)


def initialize_run(config: PipelineConfig, aliases_path: str | Path, runs_dir: str | Path = "runs") -> Path:
    """Create a new immutable run skeleton and return its path.

    Directory creation is exclusive, so a caller can never overwrite a prior run.
    """

    aliases = Path(aliases_path)
    if not aliases.is_file():
        raise FileNotFoundError(f"Alias file does not exist: {aliases}")
    root = Path(runs_dir)
    run_dir = root / new_run_id()
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"Run directory already exists and will not be overwritten: {run_dir}") from exc
    for name in ("config", "raw", "artifacts", "output", "logs", "metadata", ".tmp"):
        (run_dir / name).mkdir()
    effective_config = run_dir / "config" / "effective.toml"
    atomic_write_bytes(effective_config, config.raw_toml)
    atomic_write_bytes(run_dir / "config" / "player_aliases.csv", aliases.read_bytes())
    status = {"run_id": run_dir.name, "state": "running", "started_utc": utc_now()}
    atomic_write_json(run_dir / "metadata" / "run_status.json", status)
    environment = {
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "effective_config_sha256": sha256_file(effective_config),
    }
    atomic_write_json(run_dir / "metadata" / "environment.json", environment)
    manifest = {
        "run_id": run_dir.name,
        "season": config.season,
        "started_utc": status["started_utc"],
        "effective_config": str(effective_config),
        "effective_config_sha256": environment["effective_config_sha256"],
        "state": "running",
    }
    atomic_write_json(run_dir / "metadata" / "manifest.json", manifest)
    logger = logging.getLogger(f"ff_market_projections.run.{run_dir.name}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(run_dir / "logs" / "pipeline.log", encoding="utf-8")
    formatter = logging.Formatter('{"timestamp":"%(asctime)sZ","level":"%(levelname)s","message":"%(message)s"}')
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    logger.info("run_initialized run_id=%s pid=%s", run_dir.name, os.getpid())
    handler.close()
    logger.removeHandler(handler)
    return run_dir
