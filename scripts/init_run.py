#!/usr/bin/env python3
"""Initialize a validated, isolated pipeline run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ff_market_projections.config import ConfigError, load_config
from ff_market_projections.runs import initialize_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/pipeline.toml"))
    parser.add_argument("--aliases", type=Path, default=Path("config/player_aliases.csv"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    try:
        run_dir = initialize_run(load_config(args.config), args.aliases, args.runs_dir)
    except (ConfigError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps({"run_dir": str(run_dir), "state": "running"}, sort_keys=True))


if __name__ == "__main__":
    main()
