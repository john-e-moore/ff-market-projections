#!/usr/bin/env python3
"""Read back and validate a Phase 9 workbook."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from ff_market_projections.contracts import atomic_write_json
from ff_market_projections.validate_workbook import validate_workbook

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workbook", type=Path)
    args = parser.parse_args(); run_dir = args.run_dir.resolve()
    report = validate_workbook(run_dir, (args.workbook or run_dir / "output" / "fantasy_football_projections.xlsx").resolve())
    atomic_write_json(run_dir / "artifacts" / "workbook_validation.json", report)
    print(json.dumps({"state": "succeeded" if report["status"] == "passed" else "failed", "validation": str(run_dir / "artifacts" / "workbook_validation.json")}, sort_keys=True))
    if report["status"] != "passed": raise SystemExit(1)
if __name__ == "__main__": main()
