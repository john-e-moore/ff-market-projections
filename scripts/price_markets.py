#!/usr/bin/env python3
"""Price reconciled normalized markets into no-vig modeling observations."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

from ff_market_projections.config import ConfigError, load_config
from ff_market_projections.contracts import atomic_write_bytes, atomic_write_json
from ff_market_projections.pricing import PricingError, price_markets


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_bytes(rows: list[dict[str, object]], fields: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="raise")
    writer.writeheader(); writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _coerce(row: dict[str, object]) -> dict[str, object]:
    numeric = {"threshold", "american_odds", "source_decimal_odds", "yes_bid_probability", "yes_ask_probability", "last_trade_probability", "volume", "open_interest", "spread"}
    result = dict(row)
    for field in numeric:
        value = result.get(field)
        if value in (None, ""):
            result[field] = None
        elif field == "american_odds":
            result[field] = int(str(value))
        else:
            result[field] = float(str(value))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config_path = (args.config or run_dir / "config" / "effective.toml").resolve()
    input_path = (args.input or run_dir / "artifacts" / "normalized_markets.csv").resolve()
    output_path = (args.output or run_dir / "artifacts" / "priced_markets.csv").resolve()
    validation_path = run_dir / "artifacts" / "pricing_validation.json"
    if not all(_inside(path, run_dir) for path in (config_path, input_path, output_path, validation_path)):
        parser.error("pricing inputs and artifacts must be inside the active run directory")
    if config_path != run_dir / "config" / "effective.toml":
        parser.error("pricing must use the run-scoped effective config")
    try:
        config = load_config(config_path)
        result = price_markets([_coerce(row) for row in _read_csv(input_path)], config.values)
    except (ConfigError, PricingError, OSError, ValueError) as exc:
        validation = exc.validation if isinstance(exc, PricingError) else {"status": "failed", "checks": [], "error": str(exc)}
        atomic_write_json(validation_path, validation)
        print(json.dumps({"state": "failed", "reason": str(exc)}, sort_keys=True))
        raise SystemExit(1) from exc
    fields = list(result.rows[0]) if result.rows else []
    atomic_write_bytes(output_path, _csv_bytes(result.rows, fields))
    atomic_write_json(validation_path, result.validation)
    print(json.dumps({"state": "succeeded", "priced_rows": len(result.rows), **result.validation["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
