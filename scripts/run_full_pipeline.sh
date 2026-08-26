#!/usr/bin/env bash
# Run every implemented pipeline stage for one new, immutable run directory.
# Extra arguments are forwarded to run_pipeline.py (for example,
# --offline-input-dir tests/fixtures/phase1).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "$(git branch --show-current)" != "main" ]]; then
  printf 'Refusing to run: expected the main branch, found %s.\n' "$(git branch --show-current)" >&2
  exit 2
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
elif [[ -x .venv/bin/python ]]; then
  PYTHON_BIN=.venv/bin/python
else
  PYTHON_BIN=python3
fi
"$PYTHON_BIN" -c 'import ff_market_projections' || {
  printf 'The project package is not installed for %s. Install the project dependencies first.\n' "$PYTHON_BIN" >&2
  exit 2
}

run_stage() {
  current_stage="$1"
  shift
  printf '\n==> %s\n' "$current_stage"
  "$@"
}

printf '==> collect_inputs\n'
set +e
collection_output="$("$PYTHON_BIN" scripts/run_pipeline.py "$@" 2>&1)"
collection_status=$?
set -e
printf '%s\n' "$collection_output"

run_dir="$("$PYTHON_BIN" -c '
import json, sys
for line in reversed(sys.stdin.read().splitlines()):
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        continue
    if "run_dir" in payload:
        print(payload["run_dir"])
        break
else:
    raise SystemExit("Could not locate run_dir in collection output")
' <<<"$collection_output")"

if [[ $collection_status -ne 0 ]]; then
  printf 'Pipeline stopped during collection. Run directory: %s\n' "$run_dir" >&2
  exit "$collection_status"
fi

config="$run_dir/config/effective.toml"
current_stage=""
trap 'status=$?; printf "Pipeline stopped during %s. Run directory: %s\\n" "$current_stage" "$run_dir" >&2; exit "$status"' ERR

run_stage validate_collections "$PYTHON_BIN" scripts/validate_collections.py --run-dir "$run_dir" --config "$config"
run_stage prepare_historical_stats "$PYTHON_BIN" scripts/prepare_historical_stats.py --run-dir "$run_dir" --config "$config" --input "$run_dir/raw/nflverse_player_stats.csv.gz"
run_stage normalize_markets "$PYTHON_BIN" scripts/normalize_markets.py --run-dir "$run_dir" --config "$config"
run_stage reconcile_players "$PYTHON_BIN" scripts/reconcile_players.py --run-dir "$run_dir"
run_stage price_markets "$PYTHON_BIN" scripts/price_markets.py --run-dir "$run_dir"
run_stage calibrate_distributions "$PYTHON_BIN" scripts/calibrate_distributions.py --run-dir "$run_dir"
run_stage estimate_means "$PYTHON_BIN" scripts/estimate_means.py --run-dir "$run_dir"
run_stage aggregate_consensus "$PYTHON_BIN" scripts/aggregate_consensus.py --run-dir "$run_dir"
run_stage score_fantasy "$PYTHON_BIN" scripts/score_fantasy.py --run-dir "$run_dir"
run_stage build_workbook "$PYTHON_BIN" scripts/build_workbook.py --run-dir "$run_dir"
run_stage validate_workbook "$PYTHON_BIN" scripts/validate_workbook.py --run-dir "$run_dir"

workbook="$run_dir/output/fantasy_football_projections.xlsx"
printf '\nPipeline stages passed. Workbook: %s\n' "$workbook"
printf 'Note: final run completion metadata is not updated because Phase 10 finalization is not yet implemented.\n'
