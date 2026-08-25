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

Phase 0 supplies contracts and test fixtures only. Collection and projection stages are deliberately deferred to later roadmap phases.
