#!/usr/bin/env python3
"""Flatten validated raw market snapshots into the canonical CSV contract."""
from __future__ import annotations
import argparse
import csv
import io
import json
from pathlib import Path

from ff_market_projections.config import load_config
from ff_market_projections.contracts import atomic_write_bytes, atomic_write_json
from ff_market_projections.markets import MarketValidationError, normalize_markets, validate_collections, validate_normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path)
    args = parser.parse_args()
    try:
        collections = validate_collections(args.input_dir or args.run_dir / "raw", load_config(args.config).values)
        rows = normalize_markets(collections, args.run_dir.name)
        validation = validate_normalized(rows, collections)
        if validation["status"] != "passed":
            raise MarketValidationError("Normalized market validation failed", validation)
    except MarketValidationError as exc:
        atomic_write_json(args.run_dir / "artifacts" / "normalized_validation.json", exc.validation)
        print(json.dumps({"state": "failed"}, sort_keys=True))
        raise SystemExit(1)
    output = args.run_dir / "artifacts" / "normalized_markets.csv"
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    atomic_write_bytes(output, buffer.getvalue().encode("utf-8"))
    atomic_write_json(args.run_dir / "artifacts" / "normalized_validation.json", validation)
    print(json.dumps({"state": "succeeded", "rows": len(rows), "artifact": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
