# Fantasy Football Market Projections

Local, reproducible pipeline foundations for auditable fantasy-football market projections.

## Local setup

Use Python 3.11 or newer. Create an environment and install the package plus test tools:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Run the deterministic offline tests:

```bash
pytest
```

Initialize a new run directory (this validates `config/pipeline.toml`, snapshots effective configuration and aliases, creates status/metadata, and starts a structured log):

```bash
python scripts/init_run.py
```

The command prints the immutable `runs/{run_id}` path. It never overwrites an existing run.

## Phase 1 collection

Collect fresh run-scoped inputs (the default is live network collection):

```bash
python scripts/run_pipeline.py
```

For deterministic development, use only an explicit offline input directory containing
`draftkings.json`, `fanduel.json`, `kalshi.json`, and `nflverse_player_stats.csv.gz`:

```bash
python scripts/run_pipeline.py --offline-input-dir path/to/fixtures
```

Offline runs are identified as `offline_fixture` in `metadata/manifest.json`. A
collection failure marks the run failed while retaining its collector logs and any
successfully collected raw inputs. A successful Phase 1 collection remains `running`:
later phases are responsible for validation, workbook production, and final completion.

The nflverse snapshot is persisted in a content-addressed cache under
`data/cache/nflverse_player_stats/`; each run receives its own immutable copy in
`runs/RUN_ID/raw/`. To force a new nflverse download, pass
`--refresh-historical-cache` to `run_pipeline.py`.

## Phase 2 historical preparation

Prepare the run-scoped nflverse snapshot after collection:

```bash
python scripts/prepare_historical_stats.py \
  --run-dir runs/RUN_ID \
  --config runs/RUN_ID/config/effective.toml \
  --input runs/RUN_ID/raw/nflverse_player_stats.csv.gz
```

The command atomically writes `historical_player_seasons.csv`,
`historical_backtest_predictions.csv`, and `historical_validation.json` under the
run's `artifacts/` directory. It fails closed for incomplete seasons, reconciliation
errors, implausible totals, insufficient calibration cohorts, or any look-ahead
feature. Baseline eligibility uses prior-season opportunity only; missing opportunity
is preserved as missing and never converted to zero.

## Phase 3 market validation and normalization

Validate the three collected market snapshots before any pricing or identity work:

```bash
python scripts/validate_collections.py \
  --run-dir runs/RUN_ID \
  --config runs/RUN_ID/config/effective.toml
```

Then create `artifacts/normalized_markets.csv` and its validation report:

```bash
python scripts/normalize_markets.py \
  --run-dir runs/RUN_ID \
  --config runs/RUN_ID/config/effective.toml
```

The collection gate checks freshness, season, source-reported quality, unique source IDs,
valid two-sided sportsbook markets, Kalshi threshold and order-book domains, and
cross-source timestamp skew. Normalization emits one row per sportsbook side and one YES
row per Kalshi threshold, preserving raw JSON locators. It intentionally does not de-vig
odds, infer means, reconcile player identities, or silently discard rows.
