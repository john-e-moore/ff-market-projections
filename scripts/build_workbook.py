#!/usr/bin/env python3
"""Build the Phase 9 static Excel workbook from run-scoped artifacts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from ff_market_projections.config import load_config
from ff_market_projections.workbook import WorkbookError, build_workbook

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = (args.output or run_dir / "output" / "fantasy_football_projections.xlsx").resolve()
    if output.parent != run_dir / "output": parser.error("workbook output must be inside run output directory")
    if output.exists(): parser.error("workbook output is immutable and already exists")
    try:
        counts = build_workbook(run_dir, load_config(run_dir / "config" / "effective.toml").values, output)
    except (WorkbookError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"state": "succeeded", "workbook": str(output), "rows": counts}, sort_keys=True))
if __name__ == "__main__": main()
