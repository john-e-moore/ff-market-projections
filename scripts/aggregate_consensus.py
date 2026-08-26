#!/usr/bin/env python3
"""Aggregate source projections into the run-scoped consensus artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from ff_market_projections.config import ConfigError, load_config
from ff_market_projections.consensus import ConsensusError, aggregate_source_projections
from ff_market_projections.contracts import atomic_write_bytes, atomic_write_json, sha256_file


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
    input_path = (args.input or run_dir / "artifacts" / "source_projections.csv").resolve()
    output_path = (args.output or run_dir / "artifacts" / "consensus_stats.csv").resolve()
    validation_path = (args.validation_output or run_dir / "artifacts" / "aggregation_validation.json").resolve()
    paths = (config_path, input_path, output_path, validation_path)
    if not all(_inside(path, run_dir) for path in paths):
        parser.error("consensus inputs and artifacts must be inside the active run directory")
    if config_path != run_dir / "config" / "effective.toml":
        parser.error("consensus aggregation must use the run-scoped effective config")
    if not input_path.is_file() or input_path.stat().st_size == 0:
        parser.error(f"source projections input is missing or empty: {input_path}")
    existing = [str(path) for path in (output_path, validation_path) if path.exists()]
    if existing:
        parser.error(f"consensus artifacts are immutable and already exist: {', '.join(existing)}")

    try:
        config = load_config(config_path)
        result = aggregate_source_projections(
            pd.read_csv(input_path, low_memory=False),
            config.values["sources"],
            config.values["aggregation"],
        )
    except (ConfigError, ConsensusError, OSError, pd.errors.ParserError) as exc:
        atomic_write_json(validation_path, {"status": "failed", "checks": [], "error": str(exc)})
        print(json.dumps({"state": "failed", "reason": str(exc)}, sort_keys=True))
        raise SystemExit(1) from exc

    if result.validation["status"] != "passed":
        atomic_write_json(validation_path, result.validation)
        print(json.dumps({"state": "failed", "reason": "aggregation validation failed"}, sort_keys=True))
        raise SystemExit(1)
    csv_bytes = result.consensus_stats.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode("utf-8")
    output_hash = atomic_write_bytes(output_path, csv_bytes)
    validation = {**result.validation, "artifacts": {output_path.name: {"path": str(output_path), "sha256": output_hash, "rows": len(result.consensus_stats)}}}
    validation_hash = atomic_write_json(validation_path, validation)
    print(json.dumps({"state": "succeeded", "consensus_stats": str(output_path), "rows": len(result.consensus_stats), "validation": str(validation_path), "validation_sha256": validation_hash, "input_sha256": sha256_file(input_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
