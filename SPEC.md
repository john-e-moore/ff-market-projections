# Fantasy Football Market Projections: Technical Specification

## 1. Purpose

Build a lean, local, reproducible Python pipeline that combines historical NFL player outcomes with season-long player-stat markets from DraftKings, FanDuel, and Kalshi to estimate mean player-stat projections, combines the source projections, applies configurable fantasy scoring, and writes an auditable Excel workbook.

The primary deliverable of every successful run is:

```text
runs/{run_id}/output/fantasy_football_projections.xlsx
```

The workbook must be easy to sort and filter, expose source lineage, and contain enough validation and run metadata for a user to trust and reproduce the numbers.

This document is the implementation contract. `ROADMAP.md` defines the recommended build order.

## 2. Current Repository State

The repository already contains these collectors and example snapshots:

| Source | Collector | Current default output | Input market shape |
|---|---|---|---|
| DraftKings | `scripts/fetch_draftkings_nfl_player_stats_ou.py` | `data/draftkings_nfl_player_stats_ou.json` | One paired Over/Under market per player/stat |
| FanDuel | `scripts/fetch_fanduel_nfl_player_season_props.py` | `data/fanduel_nfl_player_season_props.json` | One paired Over/Under market per player/stat; unrelated player markets are stored separately |
| Kalshi | `scripts/fetch_kalshi_nfl_season_stats.py` | `data/kalshi_nfl_season_stats.json` | Multiple binary threshold contracts per player/stat, with order-book data |

All three collectors accept `--output`, so the orchestrator must direct their fresh output into the active run rather than relying on the checked-in `data/*.json` snapshots.

Historical calibration data will come from the official [nflverse-data player-stats release](https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.csv.gz). nflverse player stats aim to match official NFL box scores, expose stable GSIS player IDs, and are available from the play-by-play era beginning in 1999. The pipeline will ingest the full available history but use 2011 through the latest completed season as the default modeling window. Earlier seasons remain available for sensitivity analysis.

Use the release asset directly with an HTTP download and `pandas.read_csv`; record the resolved URL, response metadata, bytes, and SHA-256 hash. This is preferable for the lean core to adding a dataframe runtime solely for ingestion. The active [`nflreadpy`](https://github.com/nflverse/nflreadpy) package may be supported later as an adapter, but is not required. Do not use the archived `nfl_data_py` package as a new dependency.

The currently covered projection statistics are:

- `passing_yards`
- `passing_touchdowns`
- `rushing_yards`
- `rushing_touchdowns`
- `receiving_yards`
- `receiving_touchdowns`
- `receptions`

## 3. Goals and Non-goals

### 3.1 Goals

- Run one local command and receive one completed, timestamped run directory.
- Preserve every raw source response used in a run.
- Validate data between every material transformation.
- Reconcile source-specific player names conservatively and auditably.
- Independently recompute decimal odds, implied probabilities, and no-vig probabilities.
- Ingest and validate historical regular-season player statistics for distribution calibration, position enrichment, and out-of-sample model testing.
- Estimate a mean for each source/player/stat using an explicit, testable distribution model.
- Average source means using equal weights by default, with configurable weights.
- Calculate projections for standard, half-PPR, 0.75-PPR, and full-PPR scoring.
- Make scoring rules, source weights, validation limits, and model controls configurable.
- Produce a static `.xlsx` file with filters, frozen headers, readable formats, and detailed lineage sheets.
- Record configuration, logs, artifacts, checks, code/environment metadata, and checksums under the run directory.
- Fail closed: do not mark a run successful when a required validation gate fails.

### 3.2 Non-goals for the first version

- Weekly projections, lineup optimization, draft rankings, tiers, auction values, or simulations.
- A web application, database, cloud service, scheduler, or workflow framework.
- Scraping or adding current projection sources beyond the three existing collectors. Historical nflverse outcomes are calibration/enrichment data and never receive consensus source weight.
- Subjective news, injury, depth-chart, or schedule adjustments.
- Correlation modeling between player statistics.
- Ceiling, floor, percentile, or full-season simulation outputs beyond model-sensitivity bounds.
- Automatically publishing or emailing the workbook.
- Claiming that market-implied projections are objective forecasts.

## 4. Correctness Principles

1. **Raw inputs are immutable.** A later task never edits a collected market JSON or historical data file.
2. **Every transformation has a declared input and output grain.** Duplicate keys are errors unless the stage explicitly aggregates them.
3. **No silent coercion.** Missing odds, names, probabilities, scoring inputs, or calibration results remain missing or stop the run according to configuration.
4. **No silent zeroes.** An unavailable player-stat projection is not assumed to be zero.
5. **Source signals remain separate until aggregation.** Raw odds or market lines are never averaged across sources.
6. **Model assumptions are visible.** A mean inferred from one probability is labeled with the distribution family and dispersion calibration used.
7. **Uncertainty is not overstated.** Model-sensitivity bounds are not labeled confidence intervals unless a statistically valid interval is implemented.
8. **Point-in-time context is preserved.** Every record carries source snapshot time and the run captures cross-source time skew.
9. **Published workbooks contain values, not fragile external links or hidden calculation dependencies.**
10. **A run is successful only after the workbook is read back and reconciled to the final machine-readable artifacts.**

## 5. Technology and Project Shape

### 5.1 Runtime and libraries

Target Python 3.11 or newer. Keep the dependency set small:

- `numpy`: numeric operations
- `scipy`: probability distributions, root finding, and optimization
- `pandas`: tabular transformations and CSV handling
- `openpyxl`: Excel construction and read-back validation
- `rapidfuzz`: conservative name-match suggestions
- `pytest`: tests

Use the standard library for configuration (`tomllib`), CLI parsing (`argparse`), logging, subprocess execution, hashing, timestamps, and JSON.

Pin direct dependencies in `pyproject.toml` and commit a resolved lock file if the selected package manager supports one. Do not introduce Airflow, Prefect, Dagster, a database, or a service process.

### 5.2 Recommended repository layout

```text
config/
  pipeline.toml
  player_aliases.csv
scripts/
  fetch_nflverse_player_history.py
  fetch_draftkings_nfl_player_stats_ou.py
  fetch_fanduel_nfl_player_season_props.py
  fetch_kalshi_nfl_season_stats.py
  run_pipeline.py
  validate_collections.py
  prepare_historical_stats.py
  normalize_markets.py
  reconcile_players.py
  price_markets.py
  estimate_means.py
  aggregate_sources.py
  score_fantasy.py
  build_workbook.py
  validate_workbook.py
src/ff_market_projections/
  __init__.py
  config.py
  contracts.py
  dag.py
  distributions.py
  historical.py
  identities.py
  odds.py
  scoring.py
  validation.py
  workbook.py
tests/
  fixtures/
runs/
SPEC.md
ROADMAP.md
pyproject.toml
```

Scripts are thin CLI entry points. Reusable logic, especially statistical logic, belongs in `src/ff_market_projections/`. `distributions.py` must not contain source fetching, name matching, scoring, or Excel code.

## 6. Run Contract

### 6.1 Run identifier

Use an immutable UTC identifier such as:

```text
20260825T184512Z-a1b2c3d4
```

The random suffix prevents collision. Never reuse or overwrite an existing run directory. The orchestrator may update `runs/latest.json` only after a run is fully successful; that file contains the latest completed run ID and workbook path.

### 6.2 Run directory

```text
runs/{run_id}/
  config/
    effective.toml
    player_aliases.csv
  raw/
    nflverse_player_stats.csv.gz
    draftkings.json
    fanduel.json
    kalshi.json
  artifacts/
    collections_validation.json
    historical_player_seasons.csv
    historical_validation.json
    historical_backtest_predictions.csv
    historical_calibration.json
    normalized_markets.csv
    normalized_validation.json
    player_map.csv
    name_match_suggestions.csv
    identity_validation.json
    priced_markets.csv
    pricing_validation.json
    dispersion_calibration.csv
    source_projections.csv
    model_validation.json
    consensus_stats.csv
    aggregation_validation.json
    fantasy_projections.csv
    scoring_validation.json
    workbook_validation.json
  output/
    fantasy_football_projections.xlsx
  logs/
    pipeline.log
    collect_draftkings.log
    collect_fanduel.log
    collect_kalshi.log
    collect_nflverse_history.log
    {task_name}.log
  metadata/
    manifest.json
    run_status.json
    environment.json
    checksums.sha256
```

Temporary files may be created inside `runs/{run_id}/.tmp/`. A task writes to a temporary path, validates it, then atomically renames it to the declared artifact path.

### 6.3 Effective configuration

`config/pipeline.toml` is the checked-in editable default. At run start, validate it and copy the resolved configuration to `runs/{run_id}/config/effective.toml`. Copy the alias file as well. Downstream tasks read only the run-scoped copies.

The effective configuration must contain no credentials. Public web-client keys already embedded in a collector may remain collector defaults, but secrets must be passed through environment variables and their values must never enter configuration, logs, or metadata.

### 6.4 Manifest and status

`manifest.json` records:

- run ID and season
- invocation and task DAG
- UTC start/end times and duration for every task
- command and exit code for every collector
- input/output paths and SHA-256 hashes
- input/output row counts
- effective configuration hash
- source snapshot timestamps and maximum cross-source skew
- historical source URL/response metadata, covered seasons, raw hash, cohort counts, and calibration window
- Git commit and dirty-worktree flag, when Git is available
- Python and direct library versions
- validation results and warnings
- workbook path and hash

`run_status.json` transitions through `running`, `failed`, or `completed`. On failure it records the failed task and concise reason. It becomes `completed` only after all gates, including workbook read-back, pass.

## 7. Configuration Contract

Use TOML. The exact formatting may change, but the following concepts and defaults are required:

```toml
[run]
season = "2026-27"
timezone = "America/New_York"
fail_on_warning = false
max_snapshot_age_hours = 6
max_cross_source_skew_minutes = 30

[sources.draftkings]
enabled = true
weight = 1.0

[sources.fanduel]
enabled = true
weight = 1.0
state = "nj"

[sources.kalshi]
enabled = true
weight = 1.0
require_two_sided_quote = true
max_spread_probability_points = 20.0
min_open_interest_contracts = 0

[historical]
enabled = true
source = "nflverse_player_stats_release"
url = "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.csv.gz"
ingest_start_season = 1999
calibration_start_season = 2011
latest_completed_season = 2025
season_type = "REG"
prior_seasons = 3
recency_half_life_seasons = 5.0
minimum_training_seasons = 2
holdout_seasons = 3
minimum_player_seasons_per_stat = 200

[historical.prior_opportunity_filters]
passing_attempts = 100
rushing_attempts = 25
targets = 25

[historical.baseline_bias_correction]
method = "rolling_power_poisson"
minimum_seasons = 3
exponent_bounds = [0.25, 3.0]
recency_half_life_candidates = [1.0, 3.0, 5.0, 10.0]
minimum_validation_seasons = 3

[names]
automatic_fuzzy_match = true
minimum_score = 96.0
minimum_runner_up_gap = 4.0

[pricing]
sportsbook_devig_method = "proportional"
probability_tolerance = 1e-9
reject_ambiguous_integer_lines = true

[model]
family = "negative_binomial"
dispersion_mode = "historical_with_kalshi_update"
minimum_calibration_groups = 8
minimum_thresholds_per_group = 3
probability_floor = 0.02
probability_ceiling = 0.98
robust_loss = "soft_l1"
on_calibration_failure = "fail"
bootstrap_samples = 500
random_seed = 20260825

[aggregation]
minimum_sources = 1
renormalize_available_source_weights = true

[scoring]
missing_stat_policy = "blank_total"
passing_yards = 0.04
passing_touchdowns = 4.0
passing_interceptions = -2.0
rushing_yards = 0.10
rushing_touchdowns = 6.0
receiving_yards = 0.10
receiving_touchdowns = 6.0
fumbles_lost = -2.0
two_point_conversions = 2.0

[scoring.reception_bonus]
standard = 0.0
half_ppr = 0.5
three_quarter_ppr = 0.75
full_ppr = 1.0

[scoring.required_profiles]
passing = ["passing_yards", "passing_touchdowns"]
rushing = ["rushing_yards", "rushing_touchdowns"]
receiving = ["receiving_yards", "receiving_touchdowns", "receptions"]

[workbook]
filename = "fantasy_football_projections.xlsx"
freeze_header = true
autofilter = true
```

`latest_completed_season` must be explicit in the effective configuration; never infer that an in-progress season is complete. The implementation must also support configured per-stat opportunity filters and model bounds. Default behavior is to fail a stat's model stage when historical calibration is inadequate; do not ship unvalidated arbitrary dispersion constants merely to produce output.

Configuration validation must reject unknown keys, duplicate source names, negative weights, an incomplete historical target season, invalid historical/calibration windows, invalid probability ranges, invalid model bounds, unsupported de-vig/model methods, and output paths outside the active run.

## 8. DAG and Task Interfaces

Use a small in-process DAG runner with explicit task names, declared dependencies, deterministic topological ordering, and subprocess support for collectors. Collection tasks may run concurrently. Transformation tasks should remain separate runnable scripts for debugging.

Run bootstrap occurs before the domain DAG. The domain DAG is:

```text
collect_draftkings ─┐
collect_fanduel ────┼─> validate_collections -> normalize_markets -> validate_normalized ─┐
collect_kalshi ─────┘                                                                     ├─> reconcile_players
collect_nflverse_history -> validate_history_input -> prepare_historical_stats            │
                                                     -> validate_historical ───────────────┘
                         -> validate_identities
                         -> validate_identities
                         -> price_markets
                         -> validate_pricing
                         -> calibrate_distributions
                         -> estimate_means
                         -> validate_models
                         -> aggregate_sources
                         -> validate_aggregation
                         -> score_fantasy
                         -> validate_scoring
                         -> build_workbook
                         -> validate_workbook
                         -> finalize_run
```

Each script accepts, at minimum:

```text
--run-dir RUN_DIR
--config RUN_DIR/config/effective.toml
--input ...
--output ...
```

It exits nonzero on a failed gate and emits a compact JSON summary to stdout. It uses the shared logger and must never read an artifact from a different run.

## 9. Stage Specifications

### 9.1 Collection

Invoke the three existing collectors with explicit run-scoped outputs:

```text
python .../fetch_draftkings_nfl_player_stats_ou.py --output RUN/raw/draftkings.json
python .../fetch_fanduel_nfl_player_season_props.py --state CONFIG_STATE --output RUN/raw/fanduel.json
python .../fetch_kalshi_nfl_season_stats.py --output RUN/raw/kalshi.json
python .../fetch_nflverse_player_history.py --output RUN/raw/nflverse_player_stats.csv.gz
```

Requirements:

- Capture stdout/stderr separately in source logs.
- Record command, exit code, duration, and output hash.
- Treat network or schema failure as a failed run; do not silently fall back to checked-in `data/` snapshots.
- Download the nflverse gzip asset without transforming it, verify gzip integrity, and hash the exact bytes. Use a persistent local content-addressed cache by default so routine runs can reuse a verified snapshot without another download; the exact cached file used must be copied or hard-linked into the run and identified by hash. Provide an explicit refresh option for a new download.
- A future explicit `--offline-input-dir` mode may use supplied snapshots, but the manifest must label the run offline and identify every input hash.

### 9.2 Collection validation

Validate before normalization:

- every file exists and is nonempty; market files are UTF-8 JSON with expected top-level keys and the historical file is a valid gzip CSV
- each market `metadata.snapshot_utc` parses and satisfies configured freshness
- each market-source season equals configured season
- each market collector's `data_quality.checks_passed` is true
- normalized source market collection is nonempty
- source-specific IDs are unique at the documented grain
- expected stat labels are recognized
- DraftKings/FanDuel O/U markets contain exactly two parseable sides, matching lines, and valid odds
- Kalshi thresholds, operators, order-book prices, and probability fields are in valid ranges
- nflverse includes every configured historical season, regular-season records, stable player IDs, and all seven target stat columns
- market-source snapshot-time skew is within configuration; historical release time is recorded separately and is not part of live quote skew

Expected source placeholders already classified by a collector may be warnings. Unexpected missing records are failures.

### 9.3 Historical preparation and validation

Historical outcomes help estimate the spread around an unknown season mean, but raw historical totals are not themselves a conditional distribution for a current player. Players have different roles, talent, positions, schedules, and injury histories. Therefore, never estimate dispersion by pooling all raw totals around their cross-sectional average.

Build `historical_player_seasons.csv` at this grain:

```text
season + gsis_player_id + team + stat
```

Required preparation:

1. Keep regular-season games only and exclude the configured current/incomplete season.
2. Aggregate weekly records to player-season totals independently and reconcile them to any nflverse regular-season summary fields.
3. Retain `games`, position, team, passing attempts, rushing attempts, and targets where available so stat-specific opportunity rules can be applied.
4. Normalize comparison features for scheduled season length; preserve actual totals as the modeled outcome.
5. Create lagged features using only seasons before the target season: the prior one to three totals, games, per-game rates, opportunity, position, and age/experience when available.
6. Create a deterministic raw preseason baseline mean using a recency-weighted blend of prior player seasons shrunk toward a lagged-position/stat cohort. Do not use a position read from the target season to choose the cohort. Rookies or players without sufficient history do not train the primary residual model but remain a reported validation cohort when a baseline is available.
7. Correct systematic winner's-curse and availability/role attrition bias with a monotone positive power calibration of the raw baseline. Fit that correction by rolling origin on eligible prior target seasons with a recency-weighted Poisson mean score; select its recency half-life per stat by nested rolling-origin Poisson score using only pre-target seasons. The Poisson score calibrates the mean only and does not replace the configured predictive distribution. Freeze the correction and selected half-life before the final holdout, preserve the raw baseline, fitted parameters, training/validation seasons, convergence, and bound status, and fail closed on a nonconverged or bound-hitting fit.
8. Do not use any current-season outcome or future-season field in a target season's features.

Configure minimum **prior-season** opportunity thresholds by stat so the training population resembles players likely to have a posted season market. For example, passing distributions should not be learned from every player whose only prior usage was one trick-play attempt. Cohort eligibility for target season `t` may use only information through `t-1`; realized target-season opportunity may be used only for after-the-fact validation cuts. These thresholds are selection rules, not zero imputation, and must be shown in calibration metadata.

Historical validation must check:

- seasons are complete, ordered, and within the configured range
- regular-season totals independently reconcile to weekly sums
- player/season/stat keys are unique
- games and all seven target stats are finite; touchdowns and receptions are nonnegative, while passing, rushing, and receiving yardage may be signed because football totals can be negative
- known schedule-length changes are represented correctly; a player-season may exceed a single team schedule by one game when a trade crosses bye weeks, which is retained and reported as a warning, while larger excesses are errors
- lagged features contain no look-ahead leakage
- player IDs, display names, positions, and teams have documented missingness rates
- calibration cohort counts meet configured per-stat minimums
- changing the target season cannot change features belonging to an earlier target season

### 9.4 Market normalization

Flatten source records into `normalized_markets.csv`, one row per quoteable threshold outcome. Preserve raw JSON separately rather than embedding nested payloads in CSV.

Required columns:

| Column | Meaning |
|---|---|
| `run_id` | Active run ID |
| `source` | `draftkings`, `fanduel`, or `kalshi` |
| `source_market_id` | Source market/ticker ID |
| `source_selection_id` | Selection ID where applicable |
| `snapshot_utc` | Source snapshot time |
| `season` | Canonical season |
| `raw_player_name` | Source display name |
| `team` / `team_abbreviation` | Source value or blank; never guessed here |
| `stat` | Canonical stat name |
| `unit` | `yards`, `touchdowns`, or `receptions` |
| `operator` | Canonical `gt`, `ge`, `lt`, or `le` |
| `threshold` | Source threshold/line |
| `outcome_side` | `over`, `under`, or `yes` |
| `american_odds` | Raw American odds, if present |
| `source_decimal_odds` | Source decimal odds, if present |
| `yes_bid_probability` / `yes_ask_probability` | Kalshi quote bounds as 0-1 values |
| `last_trade_probability` | Kalshi last trade as a 0-1 value |
| `volume` / `open_interest` / `spread` | Available Kalshi quality fields |
| `source_url` | Most specific available source URL |
| `raw_record_locator` | JSON path or stable ID needed to trace back to raw input |

DraftKings/FanDuel emit two rows per paired market at this stage. Kalshi emits one YES row per threshold contract.

Normalization does not de-vig, fit a distribution, merge names, or discard an invalid record without recording an exclusion reason.

### 9.5 Player reconciliation

Produce `player_map.csv` with:

```text
source,raw_player_name,normalized_match_key,canonical_player_id,
canonical_player_name,match_method,match_score,alias_file_row,review_status
```

Algorithm:

1. Apply explicit aliases from the run-scoped `player_aliases.csv`.
2. Build a match key using Unicode normalization, case folding, punctuation removal, suffix normalization, and whitespace collapse. Preserve the original display name.
3. Merge exact match keys automatically.
4. Generate fuzzy candidates only within the same season and compatible observed stats. Use team as supporting evidence when present, never as a required field because FanDuel may omit it.
5. Auto-match only when the best score meets `minimum_score`, exceeds the runner-up by `minimum_runner_up_gap`, and introduces no source-level collision.
6. Otherwise keep the player as a distinct canonical identity and write the candidate to `name_match_suggestions.csv`.

False merges are worse than missed merges. A fuzzy suggestion must never silently change an existing explicit alias. Two different raw names from the same source may not map to one canonical player in the same stat unless an explicit alias authorizes it.

Canonical IDs must be deterministic across runs. Prefer an explicit stable ID in the alias file; for previously unseen exact keys, derive a documented slug-plus-hash ID and retain it in the generated map for promotion to the alias file.

Use nflverse GSIS IDs and full display names as strong identity evidence when available. Market-source names still require the same conservative alias and collision rules; a fuzzy name match must not acquire a GSIS ID without passing those rules.

### 9.6 Odds conversion and de-vigging

Recompute pricing centrally even though collectors currently include derived probabilities.

For American odds `A`:

```text
A > 0: decimal = 1 + A / 100
A < 0: decimal = 1 + 100 / abs(A)
raw implied probability = 1 / decimal
```

For a two-way sportsbook market with raw probabilities `q_over` and `q_under`, the default proportional no-vig probabilities are:

```text
p_over  = q_over  / (q_over + q_under)
p_under = q_under / (q_over + q_under)
```

Store raw decimal odds, raw implied probability, overround, de-vig method, and no-vig probability. Cross-check source-provided derived fields but do not use them as authoritative inputs.

An optional configured power-method de-vig may be added behind tests. Proportional normalization remains the MVP default.

Kalshi contract prices are probabilities, not sportsbook odds with a two-way overround. Do not apply sportsbook de-vigging. For auditability, derive `decimal_odds = 1 / p` for an eligible point probability. By default an eligible Kalshi point probability requires a two-sided YES market and uses the bid/ask midpoint. Preserve bid, ask, spread, and last trade separately. One-sided quotes and stale last trades are excluded from point estimation by default; any future inclusion must use a separately configured, clearly labeled rule.

`priced_markets.csv` has one modeling row per sportsbook Over threshold and one per Kalshi YES threshold. Under probabilities remain available for validation and lineage.

### 9.7 Threshold semantics

Market thresholds for the supported season totals are nonnegative integers. Historical passing, rushing, and receiving yardage totals may be signed and must remain auditable; do not silently clip or replace them. Convert each quote to the canonical event `P(X >= k)`:

- sportsbook Over `L`: `k = floor(L) + 1`
- Kalshi `N+`: `k = N`
- an Under side is used as the paired complement check, not as a second independent observation

Reject an integer sportsbook line unless its push and settlement semantics are explicitly known. Do not assume that `Over 800` and `800+` settle identically.

### 9.8 Mean-estimation model

#### Identifiability

A probability at one threshold does not determine a mean by itself. For example, if a normal approximation with standard deviation `sigma` were assumed,

```text
P(X > 800) = 0.60  =>  mean is approximately 800 + 0.253 * sigma
```

The mean changes with the unknown spread. Therefore, never treat an O/U line as a mean and never convert one probability to a mean without a recorded distribution/dispersion assumption.

#### MVP distribution

Use a negative binomial distribution for market-supported nonnegative season totals because the family has an explicit mean with flexible overdispersion. Historical signed-yardage observations remain in the Phase 2 artifact. Before fitting a negative-binomial historical likelihood in Phase 6, exclude those observations from that likelihood with an explicit count and warning, or use a separately validated signed-support model; never silently clip them. Parameterize the negative-binomial component by mean `mu` and dispersion `r`:

```text
E[X]   = mu
Var[X] = mu + mu^2 / r
p      = r / (r + mu)          # scipy parameterization
P(X >= k) = scipy.stats.nbinom.sf(k - 1, r, p)
```

This is a pragmatic first model, not a claim that every NFL stat is generated by a literal negative-binomial process. The model family is configurable so it can later be compared with discretized lognormal, gamma, zero-inflated, or other families without changing downstream contracts.

#### Historical predictive-dispersion calibration

Historical data supplies the otherwise unidentified distribution shape. It does not directly supply the current mean; the current market probability still determines that.

For every eligible player/stat target season `t`, construct a preseason baseline `mu_hat_t` using only information through `t-1`. First form a raw baseline from the configured recency-weighted blend of the player's prior one to three seasons, normalized for schedule length and shrunk toward the eligible lagged-position/stat cohort. Then apply the configured rolling-origin power calibration learned only from earlier eligible target seasons. The final holdout uses correction parameters frozen at the last pre-holdout season. Save the raw baseline, corrected baseline, correction parameters and lineage for every historical baseline in `historical_backtest_predictions.csv`.

Estimate one historical predictive dispersion `r_hist_stat` per stat by maximizing the recency-weighted negative-binomial likelihood of realized totals `Y_t` with each historical baseline `mu_hat_t` held fixed:

```text
Y_t ~ NegativeBinomial(mean = mu_hat_t, dispersion = r_hist_stat)
```

This estimates forecast uncertainty around a simple preseason expectation rather than variance across unrelated raw player totals. It intentionally includes role, availability, and performance uncertainty that is visible in out-of-sample player-season errors. Because the simple baseline may be less informed than betting markets, its dispersion can still be too wide; that limitation must remain visible.

Use rolling-origin evaluation. The final configured holdout seasons are never used to fit the baseline shrinkage or dispersion. Report, by stat and relevant position cohort:

- training and holdout player-season counts
- negative log likelihood and Brier scores for selected threshold events
- empirical coverage of 50%, 80%, and 90% predictive intervals
- calibration by predicted-mean decile
- bias and absolute error of the historical baseline
- results by games played/availability band
- dispersion estimates under the default window and configured sensitivity windows

Bootstrap target seasons and players in groups with a fixed seed to obtain uncertainty in `log(r_hist_stat)`. These bootstrap bounds describe calibration uncertainty, not player forecast confidence.

#### Current-market update from Kalshi

Eligible multi-threshold Kalshi curves may update the historical dispersion but are no longer required to identify it:

1. Select player/stat groups with the configured number of distinct, eligible, two-sided thresholds.
2. Exclude probabilities outside the configured floor/ceiling and quotes wider than the spread limit.
3. Jointly fit nuisance `mu_player` values and `log(r_stat)` to robust logit-probability residuals.
4. Penalize departure from `log(r_hist_stat)` using the historical bootstrap variance. This is an empirical-Bayes/MAP update: the current curve supplies likelihood and the historical calibration supplies the prior.
5. If current curves are insufficient, use `r_hist_stat` unchanged and label the method `historical_only`.
6. If curves are sufficient but conflict materially with historical calibration, retain both estimates, fail or warn according to configured thresholds, and publish sensitivity results. Do not silently average them.

Write historical, Kalshi-only, and final dispersion values, bootstrap bounds, sample sizes, objective values, convergence status, update method, and validation metrics to `dispersion_calibration.csv` and `historical_calibration.json`.

The checked-in Kalshi example contains many one-sided contracts, so the historical path is essential. One-sided quotes remain excluded from the point-estimate update by default. Historical calibration data and a Kalshi shape update do not receive consensus source weights; they define the shared probability-to-mean conversion applied to all three current sources.

If historical calibration lacks enough eligible out-of-sample player-seasons, performs poorly on holdout coverage, does not converge, or lands on a parameter bound, fail that stat. Do not weaken cohort filters or invent a fixed constant merely to produce output.

#### Source-specific mean fitting

With `r_stat` fixed:

- **DraftKings/FanDuel:** solve the negative-binomial survival equation for `mu` at the single no-vig Over probability using a bounded monotonic root solver.
- **Kalshi:** fit one `mu` to all eligible thresholds for the source/player/stat using robust logit-probability residuals. Do not average per-threshold means.

Back-substitute every fitted mean into the distribution and store modeled probabilities and residuals. Produce sensitivity bounds by repeating the inversion/fit at the dispersion bootstrap bounds and, for Kalshi, with bid/ask endpoints where identifiable. Label these as model-sensitivity bounds, not confidence intervals.

`source_projections.csv` grain is one row per:

```text
run_id + season + canonical_player_id + stat + source
```

Required fields include canonical/display names, source, stat, mean, sensitivity low/high, distribution family, fitted dispersion, dispersion source, quote count, thresholds used, fit error, source snapshot time, quality status, exclusion reason, and source market IDs.

#### Model validation

At minimum:

- all means and dispersions are finite, positive, and within configured per-stat bounds
- historical baselines for season `t` use no field from season `t` or later
- rolling holdout likelihood, calibration, and predictive-interval coverage meet configured tolerances
- the final dispersion records whether it is `historical_only` or `historical_plus_kalshi`
- the root solver/optimizer converged
- fitted survival probabilities are monotone non-increasing across thresholds
- back-substitution matches a single sportsbook target within tolerance
- Kalshi residual and holdout metrics are below configured limits
- no excluded quote contributed to a fit
- bootstrap output is reproducible with the configured seed
- synthetic known-parameter tests recover the expected mean and dispersion
- changing a probability in the expected direction changes the inferred mean in the expected direction

### 9.9 Source aggregation

Aggregate source means only after source-level fitting:

```text
consensus_mean = sum(weight_source * source_mean) / sum(weight_source)
```

Default every enabled source weight to `1.0`. If a source is unavailable for a player/stat, renormalize across available sources when configured. Do not impute a missing source. Record:

- every source mean and configured weight
- effective normalized weights
- source count and source list
- minimum, maximum, and range of source means
- weighted sensitivity low/high
- disagreement flag based on configured absolute and relative thresholds

If `source_count < minimum_sources`, leave the consensus blank or exclude it according to configuration. Source quality determines whether a source projection is eligible; quality must not silently alter the configured source weight.

`consensus_stats.csv` grain is one row per player/stat.

### 9.10 Fantasy scoring

For each player, pivot consensus stat means and calculate:

```text
base_points =
    passing_yards       * score.passing_yards
  + passing_touchdowns  * score.passing_touchdowns
  + passing_interceptions * score.passing_interceptions
  + rushing_yards       * score.rushing_yards
  + rushing_touchdowns  * score.rushing_touchdowns
  + receiving_yards     * score.receiving_yards
  + receiving_touchdowns * score.receiving_touchdowns
  + fumbles_lost        * score.fumbles_lost
  + two_point_conversions * score.two_point_conversions

fantasy_points(ppr) = base_points + receptions * ppr
```

Required outputs are standard (`ppr=0`), half-PPR (`0.5`), 0.75-PPR, and full-PPR (`1.0`).

The current market inputs do not project interceptions, fumbles lost, or two-point conversions. Those scoring values still belong in configuration, but their missing projections must not be silently treated as observed zeroes. The workbook and metadata must state `scoring_scope = market_supported_stats_only` until those inputs exist.

Implement both:

- `partial_fpts_*`: sum of all available projected scoring components
- `fpts_*`: publishable total governed by `missing_stat_policy`

The default `missing_stat_policy = "blank_total"` leaves `fpts_*` blank when a configured required component is missing. Configuration defines required component profiles for passing, rushing, and receiving. The row carries `components_used`, `components_missing`, `projection_complete`, and `scoring_scope` so a partial number cannot be mistaken for a complete total.

Scoring validation must independently recompute sampled and aggregate rows, confirm full-PPR is exactly standard plus receptions, verify the four PPR outputs are monotone when receptions are nonnegative, and confirm that no blank projection was converted to zero.

### 9.11 Workbook construction

Create one workbook with static values and these sheets:

1. **Projections** — one row per player; consensus stat means, source counts, completeness fields, and the four publishable and partial fantasy-point projections.
2. **Source Projections** — one row per source/player/stat; source mean, sensitivity bounds, method, calibration, quote count, fit metrics, and source market IDs.
3. **Market Inputs** — modeling-ready market rows; source, raw name, canonical name, stat, threshold, raw/decimal odds, raw/no-vig probability, Kalshi bid/ask/spread, inclusion status, and exclusion reason.
4. **Name Map** — source names, canonical identity, match method, score, and review status.
5. **Calibration** — historical window/cohorts, backtest metrics, historical/Kalshi/final dispersion values, uncertainty, and method status by stat.
6. **Scoring** — scoring settings and reception bonuses from the effective config, including currently unsupported scoring inputs.
7. **Validation** — every gate with task, check name, severity, observed value, expected rule, pass/fail, and message.
8. **Run Info** — run ID, season, timestamps, source snapshots, historical-data provenance, Git/environment metadata, model family/calibration summary, scoring scope, warnings, and hashes.

Workbook usability requirements:

- Excel tables with unique table names on all tabular sheets
- autofilters and frozen top row
- bold, readable headers; sensible column widths; no merged data cells
- numeric values stored as numbers, not formatted strings
- consistent decimals: probabilities as percentages, means/FPTS to two decimals, odds to appropriate precision
- top-level Projections columns ordered for practical sorting: identity, team/position if available, FPTS, completeness, stat means, source/quality fields
- conditional formatting for incomplete projections, low source count, model warnings, and source disagreement
- hyperlinks only to noncredentialed source URLs
- no macros, external workbook links, volatile formulas, hidden validation failures, or error cells
- explicit explanatory note at the top or in Run Info that projections are point-in-time market-derived estimates

The **Source Projections** and **Market Inputs** sheets provide the requested source transparency; `sources_used` and per-stat source counts also appear on **Projections**.

### 9.12 Workbook validation and finalization

Reopen the workbook with `openpyxl` and verify:

- all eight required sheets exist and names are correct
- expected row counts match the final CSV artifacts
- headers are unique and required columns exist
- filters, freeze panes, and tables are present
- representative numeric cells have numeric types
- no cell contains `#REF!`, `#VALUE!`, `#DIV/0!`, `NaN`, or infinity
- sampled projection and scoring values exactly match the machine-readable artifacts within tolerance
- source lineage rows can be joined back to every published consensus stat
- run ID and configuration hash match the manifest
- workbook can be saved and reopened without repair

Only after this validation passes should the orchestrator write final checksums, mark the run completed, and update `runs/latest.json`.

## 10. Data-Quality Gate Policy

Every check has a stable ID, severity (`error` or `warning`), observed value, expected rule, status, and message. Errors stop the DAG. Warnings flow to the workbook and manifest.

Minimum hard failures include:

- collector nonzero exit or missing raw file
- invalid JSON/schema, wrong season, stale data beyond configured limit, or excessive snapshot skew
- failed source-reported critical quality checks
- missing historical seasons/columns, weekly-to-season reconciliation failure, future-data leakage, or inadequate calibration cohorts
- duplicate keys at any declared grain
- unknown stats/operators or ambiguous threshold semantics
- invalid odds/probabilities or failed no-vig arithmetic
- identity collision or alias contradiction
- failed model calibration/convergence/back-substitution
- nonfinite or out-of-bound means
- aggregation arithmetic mismatch
- scoring arithmetic mismatch or silent missing-to-zero conversion
- workbook read-back mismatch

Examples of warnings include expected unavailable FanDuel coupon placeholders, single-source projections when allowed, wide but still eligible spreads, fuzzy match suggestions not used, source disagreement, and market-supported rather than full-stat scoring coverage.

## 11. Testing Strategy

### 11.1 Unit tests

- American-to-decimal and implied-probability conversions, including EVEN and invalid odds
- proportional de-vig and overround checks
- threshold/operator canonicalization and half-point semantics
- match-key normalization, explicit aliases, fuzzy thresholds, and collision guards
- negative-binomial parameterization, survival probability, inversion, monotonicity, and bounds
- lag-only historical baseline construction and look-ahead leakage guards
- synthetic historical dispersion recovery and grouped bootstrap reproducibility
- empirical-Bayes Kalshi update and historical-only fallback
- source-weight renormalization and missing-source behavior
- every scoring mode and missing-stat policy
- deterministic run IDs/hashes where applicable

### 11.2 Contract and fixture tests

Create small redacted fixtures from all three existing JSON schemas plus nflverse weekly player stats spanning several seasons. Test successful parsing plus deliberate failures for missing keys, wrong season, duplicate IDs, malformed probabilities, one-sided Kalshi quotes, missing historical seasons, weekly/season mismatches, future leakage, and unexpected stat labels.

Collector network behavior should not be exercised in the ordinary unit suite.

### 11.3 Integration tests

Run the entire post-collection DAG against fixed fixtures and compare canonical CSV outputs to golden files. Include at least:

- one player whose name differs across sources
- one unmatched player
- one player/stat available from all sources
- one single-source player/stat
- a Kalshi multi-threshold curve with a known fitted mean
- historical player-seasons with known lagged baselines and predictive dispersion
- a final dispersion that uses historical-only fallback because Kalshi curves are insufficient
- an excluded wide-spread/one-sided quote
- an incomplete fantasy scoring row

### 11.4 End-to-end smoke test

Run the pipeline in explicit offline-fixture mode, build the workbook, reopen it, and verify hashes/counts. A separate opt-in live smoke test executes all collectors and must never be required for routine deterministic tests.

## 12. Observability and Failure Behavior

- Use structured, timestamped logs with run ID and task name.
- Never log full raw responses, secrets, or enormous tracebacks to stdout; raw responses already exist as artifacts.
- On task failure, preserve completed artifacts and logs for diagnosis, mark the run failed, and stop dependent tasks.
- Re-running creates a new run. Do not mutate or resume a failed run in the MVP.
- The final console message prints the completed workbook absolute path, run ID, player count, projection count, warnings, and validation status.

## 13. Definition of Done

The MVP is complete when a clean environment can:

1. Install the pinned package.
2. Run one documented command.
3. Execute all three existing market collectors and snapshot nflverse history into a new timestamped run.
4. Build leakage-free historical baselines and pass rolling holdout calibration checks.
5. Pass each validation gate.
6. Reconcile names with no unresolved collision.
7. Recompute odds and de-vig sportsbook markets.
8. Calibrate historical predictive dispersion, optionally update it with eligible Kalshi curves, and produce validated source-level means.
9. Aggregate sources with equal default weights.
10. Apply all four reception-scoring settings without silently filling missing stats.
11. Build and read back the Excel workbook.
12. Mark the run completed and provide checksums and an auditable manifest.
13. Pass unit, integration, and offline end-to-end tests.

The user should be able to open the workbook, filter or sort the Projections sheet, trace any number through source projections and market inputs, see every material assumption or warning, and reproduce the result from the same raw inputs and effective configuration.
