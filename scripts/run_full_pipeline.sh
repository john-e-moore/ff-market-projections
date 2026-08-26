#!/usr/bin/env bash
# Run the implemented Phase 1–9 pipeline stages for one fresh, immutable run.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-$repo_root/.venv/bin/python}"

if [[ ! -x "$python_bin" ]]; then
  echo "Python executable is unavailable: $python_bin" >&2
  echo "Set PYTHON to a Python 3.11+ environment with the project installed." >&2
  exit 2
fi

summary_file="$(mktemp)"
trap 'rm -f "$summary_file"' EXIT

run_stage() {
  local script="$1"
  shift
  printf '\n==> %s\n' "$script"
  "$python_bin" "$repo_root/scripts/$script" "$@"
}

printf '==> run_pipeline.py\n'
"$python_bin" "$repo_root/scripts/run_pipeline.py" "$@" | tee "$summary_file"

run_dir="$($python_bin -c '
import json
import pathlib
import sys

lines = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
try:
    summary = json.loads(lines[-1])
    path = pathlib.Path(summary["run_dir"])
except (IndexError, KeyError, json.JSONDecodeError) as exc:
    raise SystemExit(f"Could not determine run directory from collection summary: {exc}")
if summary.get("state") != "running" or summary.get("collection_state") != "succeeded":
    raise SystemExit(f"Collection did not succeed: {summary}")
print(path)
' "$summary_file")"
config="$run_dir/config/effective.toml"

run_stage prepare_historical_stats.py --run-dir "$run_dir" --config "$config" --input "$run_dir/raw/nflverse_player_stats.csv.gz"
run_stage validate_collections.py --run-dir "$run_dir" --config "$config"
run_stage normalize_markets.py --run-dir "$run_dir" --config "$config"
run_stage reconcile_players.py --run-dir "$run_dir"
run_stage price_markets.py --run-dir "$run_dir"
run_stage calibrate_distributions.py --run-dir "$run_dir"
run_stage estimate_means.py --run-dir "$run_dir"
run_stage aggregate_consensus.py --run-dir "$run_dir"
run_stage score_fantasy.py --run-dir "$run_dir"
run_stage build_workbook.py --run-dir "$run_dir"
run_stage validate_workbook.py --run-dir "$run_dir"

printf '\nFull Phase 1–9 pipeline succeeded.\n'
printf 'Run directory: %s\n' "$run_dir"
printf 'Player projections: %s\n' "$run_dir/artifacts/fantasy_projections.csv"
printf 'Workbook: %s\n' "$run_dir/output/fantasy_football_projections.xlsx"
