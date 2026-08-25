#!/usr/bin/env python3
"""Reconcile normalized market names to conservative canonical player identities."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ff_market_projections.config import ConfigError, load_config
from ff_market_projections.contracts import atomic_write_bytes, atomic_write_json
from ff_market_projections.identities import (
    IDENTITY_COLUMNS,
    IdentityError,
    csv_bytes,
    load_aliases,
    promote_reviewed_suggestions,
    reconcile_players,
)


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or ())


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _promote(args: argparse.Namespace) -> None:
    if not args.aliases or not args.promote_suggestions:
        raise ValueError("--promote-suggestions and --aliases are required together")
    count = promote_reviewed_suggestions(args.promote_suggestions, args.aliases)
    print(json.dumps({"state": "succeeded", "promoted_aliases": count, "aliases": str(args.aliases)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--historical", type=Path)
    parser.add_argument("--aliases", type=Path)
    parser.add_argument("--promote-suggestions", type=Path)
    args = parser.parse_args()

    if args.promote_suggestions:
        try:
            _promote(args)
        except (IdentityError, OSError, ValueError) as exc:
            parser.error(str(exc))
        return
    if not args.run_dir:
        parser.error("--run-dir is required for reconciliation")
    run_dir = args.run_dir.resolve()
    config_path = (args.config or run_dir / "config" / "effective.toml").resolve()
    input_path = (args.input or run_dir / "artifacts" / "normalized_markets.csv").resolve()
    history_path = (args.historical or run_dir / "artifacts" / "historical_player_seasons.csv").resolve()
    aliases_path = (args.aliases or run_dir / "config" / "player_aliases.csv").resolve()
    output_paths = (input_path, run_dir / "artifacts" / "player_map.csv", run_dir / "artifacts" / "name_match_suggestions.csv", run_dir / "artifacts" / "identity_validation.json")
    if not all(_inside(path, run_dir) for path in (config_path, input_path, history_path, aliases_path, *output_paths)):
        parser.error("reconciliation inputs and artifacts must be inside the active run directory")
    if config_path != run_dir / "config" / "effective.toml" or aliases_path != run_dir / "config" / "player_aliases.csv":
        parser.error("downstream reconciliation must use run-scoped config and alias files")
    try:
        config = load_config(config_path)
        normalized_rows, normalized_fields = _read_csv(input_path)
        historical_rows, _ = _read_csv(history_path)
        result = reconcile_players(normalized_rows, historical_rows, config.values["names"], load_aliases(aliases_path))
    except (ConfigError, IdentityError, OSError, csv.Error) as exc:
        validation = exc.validation if isinstance(exc, IdentityError) else {"status": "failed", "checks": [], "error": str(exc)}
        atomic_write_json(run_dir / "artifacts" / "identity_validation.json", validation)
        print(json.dumps({"state": "failed", "reason": str(exc)}, sort_keys=True))
        raise SystemExit(1) from exc

    enriched_fields = [*normalized_fields, *(field for field in IDENTITY_COLUMNS if field not in normalized_fields)]
    atomic_write_bytes(input_path, csv_bytes(result.rows, enriched_fields))
    map_fields = list(result.player_map[0]) if result.player_map else ["source", "raw_player_name"]
    suggestion_fields = list(result.suggestions[0]) if result.suggestions else [
        "source", "raw_player_name", "normalized_match_key", "observed_stats",
        "candidate_canonical_player_id", "candidate_canonical_player_name", "candidate_gsis_player_id",
        "candidate_position", "candidate_source", "match_score", "runner_up_score", "reason", "review_status",
    ]
    atomic_write_bytes(run_dir / "artifacts" / "player_map.csv", csv_bytes(result.player_map, map_fields))
    atomic_write_bytes(run_dir / "artifacts" / "name_match_suggestions.csv", csv_bytes(result.suggestions, suggestion_fields))
    atomic_write_json(run_dir / "artifacts" / "identity_validation.json", result.validation)
    print(json.dumps({"state": "succeeded", "normalized_rows": len(result.rows), "player_map_rows": len(result.player_map), "suggestions": len(result.suggestions)}, sort_keys=True))


if __name__ == "__main__":
    main()
