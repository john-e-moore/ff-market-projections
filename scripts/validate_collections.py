#!/usr/bin/env python3
"""Validate raw market snapshots before canonical normalization."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from ff_market_projections.config import load_config
from ff_market_projections.contracts import atomic_write_json
from ff_market_projections.markets import MarketValidationError, validate_collections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path)
    args = parser.parse_args()
    try:
        result = validate_collections(args.input_dir or args.run_dir / "raw", load_config(args.config).values)
        report = result.report
    except MarketValidationError as exc:
        report = exc.validation
    atomic_write_json(args.run_dir / "artifacts" / "collections_validation.json", report)
    print(json.dumps({"state": "succeeded" if report["status"] == "passed" else "failed", "artifact": str(args.run_dir / "artifacts" / "collections_validation.json")}, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
