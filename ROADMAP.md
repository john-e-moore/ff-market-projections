# Fantasy Football Market Projections: Implementation Roadmap

## 1. Delivery Strategy

Implement the pipeline as a sequence of small, testable vertical slices. Do not begin with workbook styling or a workflow framework. Establish data contracts and validation first, then make each DAG stage independently runnable, and only connect the full live pipeline after the deterministic fixture path passes.

The phases below are ordered. Each phase has an exit gate; an implementation agent should not treat a phase as complete merely because the happy path runs.

## 2. Phase 0 — Foundation and Contracts

### Build

- Add `pyproject.toml` with Python 3.11+ and the minimal dependencies named in `SPEC.md`.
- Create the `src/ff_market_projections/` package and thin script entry points.
- Add `config/pipeline.toml` and `config/player_aliases.csv` templates.
- Implement strict configuration loading and validation.
- Define shared column contracts, check-result structure, artifact naming, hashing, atomic writes, and structured logging.
- Create small redacted fixtures for all three current collector schemas and a multi-season nflverse player-stats fixture.
- Document the local install and test commands in `README.md`.

### Tests

- Valid config loads and is copied without semantic change.
- Unknown/duplicate/invalid settings fail with a useful error.
- Artifact writers are atomic and produce stable hashes.
- Check results serialize consistently.
- Fixtures parse without network access.

### Exit gate

`pytest` passes in a clean environment, and one command can initialize a new run directory containing effective config, status, environment metadata, and a log without overwriting an existing run.

## 3. Phase 1 — Lightweight DAG and Fresh Collection

### Build

- Implement `scripts/run_pipeline.py` and the small dependency-aware runner.
- Register the three existing collectors as the first domain tasks.
- Add `fetch_nflverse_player_history.py` to download the official gzip CSV release asset unchanged into the run.
- Run collection tasks concurrently when practical.
- Route each collector to `runs/{run_id}/raw/{source}.json` with separate logs.
- Record commands, timing, exit codes, hashes, and source snapshot times.
- Record nflverse URL/response metadata, gzip integrity, hash, and covered seasons; allow a content-addressed local cache while preserving the exact run input.
- Add explicit offline-fixture mode for deterministic development and tests; never make it an implicit live fallback.

### Tests

- DAG topological ordering and dependency failure behavior.
- Collector subprocess success, timeout, nonzero exit, and missing-output behavior with mocked commands.
- A failed collector blocks every downstream task while preserving logs/status.
- Offline mode is clearly labeled in the manifest.

### Exit gate

A live development run can collect all three market sources plus nflverse history into an isolated run, while a deterministic offline run exercises the identical downstream path.

## 4. Phase 2 — Historical Outcomes and Leakage-free Baselines

### Build

- Implement `prepare_historical_stats.py` using nflverse player stats from 1999 onward.
- Default calibration targets to 2011 through the latest explicitly completed season; retain earlier data for sensitivity checks.
- Aggregate regular-season weekly rows to player-season/stat totals and reconcile them independently.
- Preserve GSIS player ID, name, team, position, games, attempts/carries/targets, and all seven target stats.
- Add schedule-length normalization features while keeping realized totals unchanged.
- Build prior-one-to-three-season player baselines with recency weighting and cohort shrinkage using only information available before each target season.
- Apply configurable stat-specific prior-season opportunity rules matching the market-relevant population; never select a training row using realized target-season opportunity.
- Write `historical_player_seasons.csv`, `historical_backtest_predictions.csv`, and `historical_validation.json`.

### Tests

- Weekly-to-season reconciliation and unique keys.
- Sixteen- and seventeen-game schedule handling.
- Lag features never use target/future season values.
- Modifying a future season cannot change an earlier baseline.
- Rookies and insufficient-history players follow documented cohort behavior.
- Opportunity filters are deterministic and never convert missing values to zero.
- Missing seasons, fields, IDs, and implausible totals produce the intended errors/warnings.

### Exit gate

Every historical baseline can be recomputed from prior-season rows only, all target stats reconcile, calibration cohort sizes are adequate, and leakage tests pass.

## 5. Phase 3 — Raw Market Validation and Canonical Market Rows

### Build

- Implement `validate_collections.py` with source-specific schema and quality checks.
- Enforce season, freshness, uniqueness, nonempty markets, and cross-source snapshot-skew rules.
- Implement `normalize_markets.py` and the `normalized_markets.csv` contract.
- Preserve stable links from every normalized row to its raw source record.
- Add `validate_normalized` for keys, types, supported stats, operators, odds fields, and row counts.

### Tests

- Parse all three fixtures into expected canonical rows.
- Deliberately fail wrong season, stale input, duplicate IDs, mismatched O/U lines, malformed odds, invalid Kalshi thresholds, and unknown stats.
- Confirm expected FanDuel placeholders warn rather than fail.
- Confirm normalization does not de-vig, match names, or discard rows invisibly.

### Exit gate

Every raw fixture record is either represented in canonical rows or accounted for by a named exclusion/warning, and both validation reports pass on the golden fixtures.

## 6. Phase 4 — Conservative Player Identity Reconciliation

### Build

- Implement match-key normalization and deterministic canonical player IDs.
- Use nflverse GSIS IDs, positions, and full names as strong enrichment evidence without bypassing collision rules.
- Apply explicit aliases before all automated matching.
- Add exact matching, constrained fuzzy suggestions, auto-match thresholds, and collision guards.
- Write `player_map.csv` and `name_match_suggestions.csv`.
- Join canonical identity back to normalized markets.
- Implement identity validation and an easy workflow for promoting reviewed suggestions into `config/player_aliases.csv` for future runs.

### Tests

- Initials, punctuation, apostrophes, hyphens, suffixes, and Unicode variations.
- High-scoring but ambiguous candidates remain unmerged.
- One source cannot collapse two different players into one identity.
- Explicit aliases override fuzzy logic and remain stable across runs.
- Same raw inputs and alias file produce identical IDs and maps.

### Exit gate

Golden fixtures reconcile the intended cross-source player while preserving the intended unmatched player, with no collision or unaudited fuzzy merge.

## 7. Phase 5 — Central Pricing and No-vig Conversion

### Build

- Implement pure odds functions in `odds.py`.
- Recompute American-to-decimal odds, raw implied probabilities, overround, and no-vig probabilities.
- Cross-check but do not trust collector-derived probability fields.
- Canonicalize every modeling observation to `P(X >= k)`.
- Use eligible two-sided Kalshi midpoints by default and preserve bid/ask/spread fields.
- Write `priced_markets.csv` with inclusion status and exclusion reason.
- Implement pricing validation.

### Tests

- Positive/negative American odds, EVEN, invalid zero, and round-trip tolerances.
- Known two-way de-vig examples and sum-to-one checks.
- Half-point threshold semantics for every stat.
- Ambiguous integer sportsbook lines fail.
- Kalshi is not sportsbook-de-vigged.
- One-sided/wide-spread Kalshi quotes are excluded under default config and remain auditable.

### Exit gate

Every included modeling row has a valid canonical threshold and probability, every sportsbook pair reconciles exactly within tolerance, and excluded pricing rows have explicit reasons.

## 8. Phase 6 — Historical Distribution Calibration and Mean Estimation

This is the highest-risk phase and should receive the most review.

### Build

- Implement the mean/dispersion parameterization and survival functions in `distributions.py`.
- Implement bounded inversion for a single threshold probability.
- Implement robust multi-threshold fitting for a player/stat curve.
- Estimate historical predictive dispersion against rolling, preseason-like baselines with each baseline mean held fixed.
- Add recency weighting, grouped bootstrap uncertainty, and the final configured holdout seasons.
- Evaluate likelihood, threshold Brier scores, interval coverage, mean-decile calibration, bias, and availability cohorts.
- Implement the empirical-Bayes update from eligible Kalshi groups, with historical bootstrap variance supplying the prior strength.
- Use historical-only dispersion when Kalshi curves are insufficient and label the method explicitly.
- Back-substitute fitted parameters to quote-level predicted probabilities/residuals.
- Write `dispersion_calibration.csv` and `source_projections.csv`.
- Implement convergence, monotonicity, holdout, residual, bounds, and sensitivity validation.
- Surface calibration failure rather than substituting arbitrary constants.

### Tests

- Exact synthetic negative-binomial probabilities.
- Recover known means from single quotes when dispersion is fixed.
- Recover known predictive dispersion from synthetic rolling historical forecasts.
- Recover known shared dispersion and player means from synthetic multi-threshold Kalshi groups.
- Verify the Kalshi MAP update moves dispersion in the expected direction and historical-only fallback is exact.
- Probability/mean monotonicity and extreme-but-valid probabilities.
- Optimizer nonconvergence, bound hits, insufficient thresholds, and impossible inputs.
- Reproducible bootstrap with fixed seed.
- Strict rolling-origin leakage tests and holdout exclusion.
- A demonstration test proving that the same `60% over 800` input yields different means under different dispersion assumptions.

### Review checkpoint

Before continuing, inspect calibration and residual diagnostics using a fresh real snapshot. Answer:

- Does the historical model meet holdout likelihood, calibration, and interval-coverage tolerances for every stat?
- Are results stable across the 2011+, 2016+, and 2021+ sensitivity windows?
- Are enough multi-threshold Kalshi groups eligible to update any stat?
- Do fitted survival curves decrease monotonically and visually track bid/ask ranges?
- Do any stats consistently hit dispersion or mean bounds?
- How sensitive are source means to dispersion bootstrap bounds and quote filters?
- Are one-source means plausible compared with posted thresholds without treating thresholds as means?

The checked-in Kalshi snapshot contains many one-sided contracts, so multi-threshold availability must be measured rather than assumed. Historical calibration is the default identification path; lack of Kalshi curves should produce a labeled historical-only result, not a failure by itself.

If the negative-binomial family fails materially for a stat, stop and compare an alternative configured family for that stat. Do not conceal poor fit by loosening all validation limits.

### Exit gate

Synthetic recovery and leakage tests pass, every historical calibration passes holdout diagnostics, and any mean can be reproduced from its threshold probability, distribution, final dispersion, calibration version, and method fields.

## 9. Phase 7 — Configurable Source Consensus

### Build

- Implement aggregation at player/stat grain.
- Default all enabled source weights to `1.0` and renormalize over available eligible sources.
- Store configured and effective weights, source list/count, range, and disagreement flags.
- Aggregate sensitivity bounds without labeling them confidence intervals.
- Write `consensus_stats.csv` and aggregation validation.

### Tests

- Equal three-source mean.
- Two-source and one-source behavior.
- Custom weights and disabled source.
- Missing/zero/negative weight rejection.
- Ineligible projections do not enter the denominator.
- Recomputed consensus exactly matches saved source values and effective weights.

### Exit gate

Every consensus value joins to one or more eligible source projection rows and can be recalculated exactly from workbook-visible inputs.

## 10. Phase 8 — Fantasy Scoring and Coverage

### Build

- Implement scoring in `scoring.py` from config only.
- Pivot consensus stats to one row per player.
- Calculate standard, half-PPR, 0.75-PPR, and full-PPR outputs.
- Calculate partial points separately from publishable totals.
- Implement required-component profiles and `missing_stat_policy`.
- Carry `components_used`, `components_missing`, `projection_complete`, and `scoring_scope`.
- Write `fantasy_projections.csv` and scoring validation.

### Tests

- Hand-calculated player examples for all four PPR modes.
- `full_ppr - standard == receptions` and analogous 0.5/0.75 identities.
- PPR monotonicity for nonnegative receptions.
- Missing projection remains missing under the default policy.
- Unsupported market stats such as interceptions/fumbles are disclosed rather than zero-filled.
- Config changes alter only the expected scoring outputs.

### Exit gate

All scoring values independently reconcile, and no user can mistake a partial market-supported total for a complete fantasy total based on row fields or metadata.

## 11. Phase 9 — Excel Workbook

### Build

- Implement the eight sheets specified in `SPEC.md`, including Calibration.
- Use static numeric values, Excel tables, filters, frozen headers, formats, sensible widths, and targeted conditional formatting.
- Put source lineage and quality fields in easily filterable columns.
- Add Run Info explanations for point-in-time data, shared dispersion calibration, scoring scope, and warnings.
- Implement workbook read-back validation and artifact reconciliation.

### Tests

- Required sheets, headers, row counts, tables, filters, and freeze panes.
- Numeric types and absence of Excel error values.
- Sample source, consensus, and fantasy values match CSV artifacts.
- Long names, URLs, blank optional fields, and warning text do not corrupt output.
- Workbook saves, closes, reopens, and validates.

### Manual QA checkpoint

Open the workbook in desktop Excel and verify:

- Projections is useful immediately without rearranging columns.
- Sorting FPTS and filtering source count/completeness works.
- A projection can be traced to Source Projections and Market Inputs.
- Warning formatting is visible but not overwhelming.
- All columns are readable and no table is clipped or misleadingly formatted.

### Exit gate

Automated read-back passes and the manual Excel smoke test finds no usability or repair issue.

## 12. Phase 10 — End-to-end Hardening and Release

### Build

- Connect all task gates and final manifest/checksum generation.
- Mark completion only after workbook validation.
- Add `runs/latest.json` update after success.
- Add concise CLI success/failure summaries and documented exit codes.
- Add the deterministic offline end-to-end test to CI.
- Add an opt-in live collector smoke test for local use.
- Review logs and artifacts for accidental credentials or oversized duplication.

### Acceptance run

Execute one fresh live run from a clean environment. Preserve it as a release candidate and confirm:

- three fresh market inputs with acceptable timestamp skew
- a hashed nflverse history snapshot with the expected completed-season coverage
- leakage-free historical baselines and passing rolling holdout diagnostics
- no unexplained exclusions or identity collisions
- all seven historical stat calibrations pass and each optional Kalshi update/fallback is documented
- source and consensus means pass residual/arithmetic checks
- scoring coverage is clearly represented
- workbook read-back passes
- manifest hashes every final artifact
- run ends `completed` and the printed absolute workbook path exists

### Exit gate

All items in the `SPEC.md` Definition of Done are satisfied and the user can reproduce the candidate run from its raw inputs and effective configuration.

## 13. Deferred Enhancements

Consider only after the MVP acceptance run:

- Historical preseason market/projection archives to calibrate uncertainty around a stronger baseline than lagged player outcomes.
- Stat-specific model-family selection beyond the initial negative binomial.
- Empirically calibrated injury/zero-inflated mixtures.
- Additional sources for interceptions, fumbles, two-point conversions, positions, and teams.
- Full probabilistic projections and player-level simulation.
- Market liquidity/staleness-aware source weights, kept distinct from user-configured source weights.
- Weekly scheduling, incremental runs, or a database.
- Rankings, tiers, replacement value, or league-specific roster settings.

Each enhancement must preserve the existing run contract, source lineage, and validation behavior or introduce an explicit versioned schema migration.

## 14. Implementation-Agent Checklist

Before closing any phase, an LLM agent should:

- read `SPEC.md` and the current phase in this roadmap
- inspect the actual upstream artifact rather than assuming its schema
- preserve unrelated user changes in the repository
- add or update tests with the code change
- run the smallest relevant tests, then the cumulative offline pipeline suite
- inspect generated validation artifacts, not only process exit codes
- update docs/config examples when a contract changes
- state any assumption that changes numerical interpretation
- avoid proceeding past a failed gate solely to produce a workbook
