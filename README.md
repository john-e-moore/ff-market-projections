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
