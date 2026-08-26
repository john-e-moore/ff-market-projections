#!/usr/bin/env python3
"""Write run-scoped fantasy projections from consensus statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ff_market_projections.config import ConfigError, load_config
from ff_market_projections.contracts import atomic_write_bytes, atomic_write_json, sha256_file
from ff_market_projections.scoring import ScoringError, score_consensus


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validation-output", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config_path = (args.config or run_dir / "config" / "effective.toml").resolve()
    input_path = (args.input or run_dir / "artifacts" / "consensus_stats.csv").resolve()
    output_path = (args.output or run_dir / "artifacts" / "fantasy_projections.csv").resolve()
    validation_path = (args.validation_output or run_dir / "artifacts" / "scoring_validation.json").resolve()
    paths = (config_path, input_path, output_path, validation_path)
    if not all(_inside(path, run_dir) for path in paths):
        parser.error("scoring inputs and artifacts must be inside the active run directory")
    if config_path != run_dir / "config" / "effective.toml":
        parser.error("fantasy scoring must use the run-scoped effective config")
    if not input_path.is_file() or input_path.stat().st_size == 0:
        parser.error(f"consensus input is missing or empty: {input_path}")
    existing = [str(path) for path in (output_path, validation_path) if path.exists()]
    if existing:
        parser.error(f"scoring artifacts are immutable and already exist: {', '.join(existing)}")

    try:
        config = load_config(config_path)
        result = score_consensus(pd.read_csv(input_path, low_memory=False), config.values["scoring"])
    except (ConfigError, ScoringError, OSError, pd.errors.ParserError) as exc:
        validation = {"status": "failed", "checks": [], "error": str(exc)}
        atomic_write_json(validation_path, validation)
        print(json.dumps({"state": "failed", "reason": str(exc)}, sort_keys=True))
        raise SystemExit(1) from exc

    if result.validation["status"] != "passed":
        atomic_write_json(validation_path, result.validation)
        print(json.dumps({"state": "failed", "reason": "scoring validation failed"}, sort_keys=True))
        raise SystemExit(1)
    csv_bytes = result.fantasy_projections.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode("utf-8")
    output_hash = atomic_write_bytes(output_path, csv_bytes)
    validation = {**result.validation, "artifacts": {output_path.name: {"path": str(output_path), "sha256": output_hash, "rows": len(result.fantasy_projections)}, "input": {"path": str(input_path), "sha256": sha256_file(input_path)}}}
    validation_hash = atomic_write_json(validation_path, validation)
    print(json.dumps({"state": "succeeded", "fantasy_projections": str(output_path), "rows": len(result.fantasy_projections), "validation": str(validation_path), "validation_sha256": validation_hash}, sort_keys=True))


if __name__ == "__main__":
    main()
