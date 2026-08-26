from __future__ import annotations

import contextlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from analysis import RUN_DIR, RUN_ID, compute_audit


HERE = Path(__file__).resolve().parent
GENERATED_AT = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def execute_cells(cells: list[dict]) -> list[dict]:
    namespace: dict = {"__name__": "__notebook__"}
    execution_count = 0
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        execution_count += 1
        stream = io.StringIO()
        try:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                exec(cell["source"], namespace)
        except Exception as exc:
            cell["execution_count"] = execution_count
            cell["outputs"] = [
                {
                    "output_type": "error",
                    "ename": type(exc).__name__,
                    "evalue": str(exc),
                    "traceback": [],
                }
            ]
            raise
        cell["execution_count"] = execution_count
        output = stream.getvalue()
        cell["outputs"] = (
            [{"output_type": "stream", "name": "stdout", "text": output}] if output else []
        )
    return cells


def build_notebook(results: dict) -> dict:
    s = results["summary"]
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": (
                "# Projection pipeline audit\n\n"
                "## tl;dr\n\n"
                f"- DraftKings and FanDuel overlap closely, while Kalshi-involved published rows trigger disagreement flags {pct(s['kalshi_disagreement_rate'])} of the time.\n"
                f"- Kalshi updated historical dispersion for {s['kalshi_dispersion_updates']} stats; {s['quarantined_kalshi_curves']} source curves were quarantined.\n"
                f"- {s['kalshi_curves_no_24h_volume']} of {s['kalshi_eligible_curves']} eligible Kalshi curves had no 24-hour trades.\n"
                f"- All {s['fantasy_rows']} fantasy rows have blank player names and none is marked complete."
            ),
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": (
                "## Context & Methods\n\n"
                f"This diagnostic notebook audits run `{RUN_ID}` at player/stat/source grain. "
                "Source means are compared only where both sources passed model validation. "
                "Kalshi liquidity is measured on the exact quote rows that contributed to passed source curves.\n\n"
                "### Key Assumptions\n\n"
                "- Pairwise percentage gaps use the symmetric absolute difference.\n"
                "- The two sportsbooks form the comparison baseline for three-source overlaps; this is a consistency check, not proof that the books are correct.\n"
                "- Cumulative volume, open interest, 24-hour volume, spread, and top-of-book size describe different aspects of liquidity and are not interchangeable."
            ),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": (
                "from pathlib import Path\n"
                "import pandas as pd\n"
                "from analysis import RUN_DIR, compute_audit\n"
                "results = compute_audit()\n"
                "print(f'Run directory: {RUN_DIR}')\n"
                "print(f\"Published consensus rows: {results['summary']['published_consensus_rows']}\")\n"
            ),
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Data\n\nThe analysis reads the run-scoped priced markets, source projections, consensus statistics, fantasy projections, model validation report, manifest, and raw Kalshi order-book snapshot.",
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": (
                "print(pd.Series(results['summary'], name='value').to_string())\n"
            ),
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Results\n\n### Pairwise source agreement",
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "print(pd.DataFrame(results['pair_summary']).to_string(index=False))\n",
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "### Three-source behavior by stat",
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "print(pd.DataFrame(results['triple_by_stat']).to_string(index=False))\n",
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "### Kalshi liquidity funnel and largest disagreements",
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": (
                "print(pd.DataFrame(results['kalshi_funnel']).to_string(index=False))\n"
                "print('\\nLiquidity correlations with absolute Kalshi/book disagreement:')\n"
                "print(pd.DataFrame(results['liquidity_correlations']).T.to_string())\n"
                "print('\\nLargest disagreements:')\n"
                "print(pd.DataFrame(results['triple_outliers']).to_string(index=False))\n"
            ),
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "### Fantasy output reasonableness",
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "print(pd.DataFrame(results['top_scores']).to_string(index=False))\n",
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": (
                "## Takeaways\n\n"
                "1. Kalshi is useful as a third signal, but the current equal weighting is not supported by the observed liquidity and cross-source disagreement.\n"
                "2. The dispersion-update fallback and curve quarantine are working as safety mechanisms.\n"
                "3. The final fantasy output needs a blocking validation for player names and clearer treatment of incomplete scores.\n"
                "4. Source-specific inversion should be validated against historical market snapshots and realized outcomes before the resulting means are treated as conventional fantasy projections."
            ),
        },
    ]
    executed = execute_cells(cells)
    return {
        "cells": executed,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def source_spec(source_id: str, label: str, path: str, description: str) -> dict:
    return {
        "id": source_id,
        "label": label,
        "path": path,
    }


def sql_literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def build_report_sql(datasets: dict[str, list[dict]]) -> str:
    statements = [
        "-- DuckDB-compatible rendering views for the validated report datasets.",
        "-- The joined calculations are executed and documented in projection_audit.ipynb.",
    ]
    for name, rows in datasets.items():
        if not rows:
            continue
        columns = list(rows[0])
        values = []
        for row in rows:
            values.append("(" + ", ".join(sql_literal(row.get(column)) for column in columns) + ")")
        quoted_columns = ", ".join(f'"{column}"' for column in columns)
        statements.append(
            f'CREATE OR REPLACE VIEW "{name}" AS\n'
            f"SELECT * FROM (VALUES\n  "
            + ",\n  ".join(values)
            + f"\n) AS dataset({quoted_columns});"
        )
    return "\n\n".join(statements) + "\n"


def build_report(results: dict) -> dict:
    s = results["summary"]
    pair_summary = {row["pair"]: row for row in results["pair_summary"]}
    dk_fd = pair_summary["DraftKings vs FanDuel"]
    dk_k = pair_summary["DraftKings vs Kalshi"]
    fd_k = pair_summary["FanDuel vs Kalshi"]

    report_sources = [
        source_spec(
            "report_sql",
            "Validated report dataset SQL",
            f"reports/{RUN_ID}-projection-audit/report_data.sql",
            "DuckDB-compatible rendering views for every chart, table, and metric card in the report.",
        ),
        source_spec(
            "audit_notebook",
            "Executed projection audit notebook",
            f"reports/{RUN_ID}-projection-audit/projection_audit.ipynb",
            "Reproducible joined analysis of source agreement, Kalshi liquidity, consensus impact, and final fantasy outputs.",
        ),
        source_spec(
            "source_projections",
            "Run source projections",
            f"runs/{RUN_ID}/artifacts/source_projections.csv",
            "Player/stat/source means and eligibility from the Phase 6B estimator.",
        ),
        source_spec(
            "priced_markets",
            "Run priced markets",
            f"runs/{RUN_ID}/artifacts/priced_markets.csv",
            "Canonical quote probabilities, exclusions, liquidity fields, fitted means, and curve status.",
        ),
        source_spec(
            "model_validation",
            "Run model validation",
            f"runs/{RUN_ID}/artifacts/model_validation.json",
            "Phase 6B global calibration checks, curve diagnostics, warnings, and quarantine counts.",
        ),
        source_spec(
            "fantasy_output",
            "Run fantasy projections",
            f"runs/{RUN_ID}/artifacts/fantasy_projections.csv",
            "Final scoring output used by the workbook Projections sheet.",
        ),
    ]
    manifest_sources = [
        {"id": item["id"], "label": item["label"], "path": item["path"]}
        for item in report_sources
    ]

    headline = [{
        "kalshi_updates": s["kalshi_dispersion_updates"],
        "kalshi_disagreement_rate": s["kalshi_disagreement_rate"],
        "inactive_curve_rate": s["kalshi_curves_no_24h_volume"] / s["kalshi_eligible_curves"],
        "blank_names": s["fantasy_blank_names"],
    }]
    output_quality = [
        {"issue": "Blank player names", "affected": s["fantasy_blank_names"], "total": s["fantasy_rows"], "severity": "High", "confidence": "High"},
        {"issue": "Rows marked projection_complete=false", "affected": s["fantasy_rows"] - s["fantasy_complete_rows"], "total": s["fantasy_rows"], "severity": "High", "confidence": "High"},
        {"issue": "Rows without a numeric standard score", "affected": s["fantasy_rows"] - s["fantasy_numeric_scores"], "total": s["fantasy_rows"], "severity": "Medium", "confidence": "High"},
        {"issue": "Unmatched source/player identities", "affected": s["identity_unmatched"], "total": 363, "severity": "Medium", "confidence": "High"},
    ]

    charts = [
        {
            "id": "source_agreement_chart",
            "title": "Median absolute source difference by stat",
            "subtitle": "Eligible overlapping player/stat projections; lower is closer agreement",
            "type": "bar",
            "dataset": "pair_by_stat",
            "sourceId": "report_sql",
            "valueFormat": "percent",
            "encodings": {
                "x": {"field": "comparison", "type": "nominal", "label": "Source pair and stat"},
                "y": {"field": "median_abs_difference", "type": "quantitative", "label": "Median absolute difference", "format": "percent"},
                "tooltip": [
                    {"field": "overlap", "type": "quantitative", "label": "Overlapping projections"},
                    {"field": "median_right_to_left_ratio", "type": "quantitative", "label": "Right/left median ratio"},
                ],
            },
            "layout": "full",
        },
        {
            "id": "kalshi_funnel_chart",
            "title": "Kalshi eligibility funnel",
            "subtitle": "Contracts and quote rows surviving each trust gate in the August 26 run",
            "type": "bar",
            "dataset": "kalshi_funnel",
            "sourceId": "report_sql",
            "valueFormat": "number",
            "encodings": {
                "x": {"field": "stage", "type": "ordinal", "label": "Gate"},
                "y": {"field": "count", "type": "quantitative", "label": "Contracts / curves"},
            },
            "layout": "full",
        },
    ]

    tables = [
        {
            "id": "triple_stat_table",
            "title": "Three-source behavior by stat",
            "subtitle": "Median Kalshi difference and resulting equal-weight consensus shift versus the two-book average",
            "dataset": "triple_by_stat",
            "sourceId": "report_sql",
            "defaultSort": {"field": "median_abs_kalshi_vs_books", "direction": "desc"},
            "density": "spacious",
            "columns": [
                {"field": "stat_label", "label": "Stat", "type": "text"},
                {"field": "overlaps", "label": "Three-source overlaps", "format": "number"},
                {"field": "median_kalshi_vs_books", "label": "Median Kalshi vs books", "format": "percent", "movement": True},
                {"field": "median_abs_kalshi_vs_books", "label": "Median absolute gap", "format": "percent"},
                {"field": "median_consensus_shift", "label": "Median consensus shift", "format": "percent", "movement": True},
                {"field": "disagreement_flags", "label": "Disagreement flags", "format": "number"},
            ],
        },
        {
            "id": "outlier_table",
            "title": "Largest Kalshi differences on three-source overlaps",
            "subtitle": "Top 15 absolute gaps, with liquidity for the contributing Kalshi thresholds",
            "dataset": "triple_outliers",
            "sourceId": "report_sql",
            "defaultSort": {"field": "kalshi_vs_books", "direction": "asc"},
            "density": "dense",
            "columns": [
                {"field": "canonical_player_name", "label": "Player", "type": "text"},
                {"field": "stat_label", "label": "Stat", "type": "text"},
                {"field": "draftkings_mean", "label": "DK mean", "format": "number"},
                {"field": "fanduel_mean", "label": "FD mean", "format": "number"},
                {"field": "kalshi_mean", "label": "Kalshi mean", "format": "number"},
                {"field": "kalshi_vs_books", "label": "Kalshi vs books", "format": "percent", "movement": True},
                {"field": "consensus_vs_books", "label": "Consensus shift", "format": "percent", "movement": True},
                {"field": "total_volume", "label": "Lifetime volume", "format": "number"},
                {"field": "minimum_open_interest", "label": "Min open interest", "format": "number"},
                {"field": "volume_24h", "label": "24h volume", "format": "number"},
                {"field": "median_spread", "label": "Median spread", "format": "percent"},
                {"field": "minimum_top_size", "label": "Min top size", "format": "number"},
            ],
        },
        {
            "id": "output_quality_table",
            "title": "Final-output quality defects",
            "subtitle": "Counts are taken from the run-scoped fantasy and identity artifacts",
            "dataset": "output_quality",
            "sourceId": "report_sql",
            "defaultSort": {"field": "affected", "direction": "desc"},
            "density": "spacious",
            "columns": [
                {"field": "issue", "label": "Issue", "type": "text"},
                {"field": "affected", "label": "Affected", "format": "number"},
                {"field": "total", "label": "Total", "format": "number"},
                {"field": "severity", "label": "Severity", "type": "text"},
                {"field": "confidence", "label": "Confidence", "type": "text"},
            ],
        },
        {
            "id": "top_scores_table",
            "title": "Highest partial standard-scoring outputs",
            "subtitle": "Names recovered from source projections because the final fantasy artifact leaves them blank",
            "dataset": "top_scores",
            "sourceId": "report_sql",
            "defaultSort": {"field": "fpts_standard", "direction": "desc"},
            "density": "dense",
            "columns": [
                {"field": "canonical_player_name", "label": "Player", "type": "text"},
                {"field": "canonical_position", "label": "Position", "type": "text"},
                {"field": "scoring_profile", "label": "Profile", "type": "text"},
                {"field": "fpts_standard", "label": "Partial standard points", "format": "number"},
                {"field": "passing_yards", "label": "Passing yards", "format": "number"},
                {"field": "passing_touchdowns", "label": "Passing TD", "format": "number"},
                {"field": "rushing_yards", "label": "Rushing yards", "format": "number"},
                {"field": "rushing_touchdowns", "label": "Rushing TD", "format": "number"},
            ],
        },
        {
            "id": "book_line_ratio_table",
            "title": "Sportsbook mean-to-threshold conversion",
            "subtitle": "Median fitted mean divided by the canonical market threshold",
            "dataset": "book_line_ratios",
            "sourceId": "report_sql",
            "defaultSort": {"field": "median_mean_to_threshold", "direction": "desc"},
            "density": "spacious",
            "columns": [
                {"field": "source", "label": "Source", "type": "text"},
                {"field": "stat", "label": "Stat", "type": "text"},
                {"field": "quotes", "label": "Quotes", "format": "number"},
                {"field": "median_mean_to_threshold", "label": "Mean / threshold", "format": "number"},
                {"field": "median_overround", "label": "Median raw probability total", "format": "number"},
            ],
        },
    ]

    cards = [
        {
            "id": "kalshi_updates_card",
            "description": "Number of the seven stat dispersions updated by Kalshi after all global checks.",
            "dataset": "headline",
            "sourceId": "report_sql",
            "metrics": [{"label": "Kalshi-updated stats", "field": "kalshi_updates", "format": "number"}],
        },
        {
            "id": "kalshi_disagreement_card",
            "description": "Share of published consensus rows involving Kalshi that breach a configured disagreement threshold.",
            "dataset": "headline",
            "sourceId": "report_sql",
            "metrics": [{"label": "Kalshi-involved disagreement", "field": "kalshi_disagreement_rate", "format": "percent"}],
        },
        {
            "id": "inactive_curves_card",
            "description": "Share of eligible Kalshi player/stat curves with no traded contracts in the prior 24 hours.",
            "dataset": "headline",
            "sourceId": "report_sql",
            "metrics": [{"label": "Eligible curves with zero 24h volume", "field": "inactive_curve_rate", "format": "percent"}],
        },
        {
            "id": "blank_names_card",
            "description": "Fantasy projection rows with no player name in the final CSV and workbook.",
            "dataset": "headline",
            "sourceId": "report_sql",
            "metrics": [{"label": "Blank player names", "field": "blank_names", "format": "number"}],
        },
    ]

    title = "Fantasy Projection Pipeline Audit"
    technical_summary = (
        "## Technical summary\n\n"
        f"**Kalshi is the outlying source in this run, and it is not yet trustworthy enough for an equal vote in consensus.** "
        f"DraftKings and FanDuel agree closely on {dk_fd['overlap']} overlapping player/stat projections (median absolute gap {pct(dk_fd['median_abs_difference'])}), "
        f"while DraftKings–Kalshi and FanDuel–Kalshi gaps are {pct(dk_k['median_abs_difference'])} and {pct(fd_k['median_abs_difference'])}. "
        f"Rows involving Kalshi triggered disagreement flags {pct(s['kalshi_disagreement_rate'])} of the time, versus {pct(s['non_kalshi_disagreement_rate'])} without Kalshi.\n\n"
        f"**The Phase 6B safety fallbacks worked, but the remaining Kalshi means still enter consensus too aggressively.** "
        f"All seven dispersions stayed historical, {s['quarantined_kalshi_curves']} bad Kalshi curves were excluded, and only {s['kalshi_contributing_quotes']} of {s['kalshi_raw_quotes']} Kalshi contracts contributed. "
        f"However, {s['kalshi_curves_no_24h_volume']} of {s['kalshi_eligible_curves']} eligible curves had no 24-hour trades, and Kalshi receives the same effective source weight as either sportsbook whenever it survives the hard gates.\n\n"
        f"**The final fantasy artifact has a separate blocking defect.** Every one of its {s['fantasy_rows']} player names is blank; none of the rows is marked complete, and only {s['fantasy_numeric_scores']} have numeric scores."
    )

    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {"id": "technical_summary", "type": "markdown", "body": technical_summary},
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": [card["id"] for card in cards]},
        {
            "id": "source_behavior_intro",
            "type": "markdown",
            "body": (
                "## Kalshi, not FanDuel, is the source behaving differently\n\n"
                "The two sportsbooks are internally consistent: the median gap is small across shared yards and touchdown markets, and FanDuel is only modestly lower on average. "
                "Kalshi is systematically lower for yardage and receptions, while rushing-touchdown estimates are often higher and more variable. Correlation remains high because all sources rank stars above fringe players; the level differences are what matter for consensus."
            ),
        },
        {"id": "source_agreement", "type": "chart", "chartId": "source_agreement_chart"},
        {
            "id": "source_stat_interpretation",
            "type": "markdown",
            "body": (
                "## Equal weighting turns those source gaps into material projection shifts\n\n"
                "On three-source overlaps, Kalshi is a median 31.7% below the sportsbook average for rushing yards, 28.9% below for passing yards, and 14.4% below for receiving yards. "
                "Because all three sources receive one-third weight, this moves the final consensus roughly 10.6%, 9.6%, and 4.8% lower, respectively. Rushing-touchdown effects run the other direction in several prominent cases."
            ),
        },
        {"id": "triple_stat", "type": "table", "tableId": "triple_stat_table"},
        {
            "id": "liquidity_intro",
            "type": "markdown",
            "body": (
                "## Kalshi filtering is conservative on quotes but incomplete on liquidity\n\n"
                f"The pipeline correctly rejects one-sided markets and spreads wider than 20 percentage points, then quarantines curves that fail fit or holdout diagnostics. That leaves {s['kalshi_contributing_quotes']} contributing quotes on {s['kalshi_eligible_curves']} curves. "
                f"The weak point is what happens next: the minimum open-interest setting is zero, no lifetime or 24-hour volume floor is enforced, and {s['kalshi_single_eligible_threshold_curves']} eligible curves have only one usable threshold. A single low-activity midpoint can therefore receive the same source weight as a complete sportsbook O/U pair."
            ),
        },
        {"id": "kalshi_funnel", "type": "chart", "chartId": "kalshi_funnel_chart"},
        {
            "id": "liquidity_evidence",
            "type": "markdown",
            "body": (
                f"## Weak liquidity is associated with larger Kalshi disagreements\n\n"
                f"Among the 42 three-source overlaps, absolute Kalshi/book disagreement falls as cumulative volume rises (Spearman ρ={results['liquidity_correlations']['total_volume']['rho']:.2f}, p={results['liquidity_correlations']['total_volume']['p_value']:.3f}) and as minimum open interest rises (ρ={results['liquidity_correlations']['minimum_open_interest']['rho']:.2f}, p={results['liquidity_correlations']['minimum_open_interest']['p_value']:.3f}). "
                "This is observational and the sample is small, but it supports using liquidity in the trust decision. The largest gaps include zero-volume or very low-open-interest curves and spreads near the current maximum."
            ),
        },
        {"id": "outliers", "type": "table", "tableId": "outlier_table"},
        {
            "id": "output_defects_intro",
            "type": "markdown",
            "body": (
                "## The fantasy output is not release-ready even though workbook validation passed\n\n"
                f"All {s['fantasy_rows']} player names are blank in the final CSV and workbook. The workbook validator compares a null sample to the null source artifact, so it verifies propagation of the defect rather than enforcing a required display field. "
                f"All rows also carry `projection_complete=false`; {s['fantasy_rows'] - s['fantasy_numeric_scores']} rows have no numeric score, while the other {s['fantasy_numeric_scores']} display partial totals that omit unsupported components such as interceptions, fumbles lost, and two-point conversions."
            ),
        },
        {"id": "output_quality", "type": "table", "tableId": "output_quality_table"},
        {
            "id": "score_reasonableness",
            "type": "markdown",
            "body": (
                "## Several headline fantasy totals need clearer interpretation\n\n"
                "The displayed values are partial market-supported scores, not conventional complete fantasy projections. For example, Josh Allen reaches about 450 standard points without an interception deduction, and multiple quarterbacks exceed 5,000 passing yards after threshold-to-mean inversion. "
                "Those values are mechanically consistent with the fitted negative-binomial distribution, but they should not be presented without an explicit partial-score label and an empirical check that the source-specific inversion maps preseason market prices to realized season means."
            ),
        },
        {"id": "top_scores", "type": "table", "tableId": "top_scores_table"},
        {
            "id": "inversion_risk",
            "type": "markdown",
            "body": (
                "## Sportsbook agreement does not prove the inferred means are calibrated\n\n"
                "Both books use the same historical dispersion to convert a near-median O/U threshold into a mean, so close agreement mostly shows that their posted lines are similar. The conversion lifts median means above thresholds by roughly 31% for passing yards, 26% for receiving yards, and 28% for rushing yards. "
                "That may be intentional under an injury-sensitive, right-skewed season distribution, but it is a model-risk concentration: both sportsbook projections can move together even if the shared distributional assumption is off."
            ),
        },
        {"id": "book_line_ratios", "type": "table", "tableId": "book_line_ratio_table"},
        {
            "id": "scope_definitions",
            "type": "markdown",
            "body": (
                "## Scope, data, and definitions\n\n"
                f"This audit covers the live run collected on August 26, 2026; the three market snapshots were only {s['source_skew_seconds']:.1f} seconds apart. The analytical grain is one canonical player/stat/source projection after pricing and Phase 6B validation. "
                "A published consensus row has at least one eligible source because `minimum_sources=1`. A disagreement flag means the source range breaches the configured absolute or relative threshold; it does not currently block publication. A contributing Kalshi quote is a two-sided midpoint within the 20-point spread cap and on a source curve that passed residual and holdout validation."
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "body": (
                "## Methodology\n\n"
                "Source means were joined on canonical player and stat. Pairwise agreement uses symmetric absolute percentage difference so neither source is privileged. For three-source overlaps, the simple DraftKings/FanDuel average is used only as an internal comparison baseline. "
                "Kalshi liquidity was aggregated over the exact thresholds contributing to each passed curve; cumulative volume, minimum open interest, 24-hour volume, median spread, and minimum top-of-book size were retained separately. Rank correlations test association between liquidity and absolute Kalshi/book disagreement. Final fantasy outputs were profiled for name completeness, score completeness, and visible outliers."
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## Limitations and robustness\n\n"
                "This is a single point-in-time run, not a forecast-accuracy backtest. The sportsbook average is a consistency baseline, not ground truth, and the three-source sample is only 42 player/stat overlaps. Kalshi order-book depth can change without recent trades, so zero 24-hour volume does not imply a quote is unusable; it does imply that stale resting orders deserve less confidence. "
                "The source-level direction is robust across multiple internal checks: source-pair gaps, disagreement flags, the liquidity relationship, the rejected global Kalshi calibration, and individual outliers all point the same way."
            ),
        },
        {
            "id": "recommendations",
            "type": "markdown",
            "body": (
                "## Recommended next steps\n\n"
                "1. **Block release when projection names are blank.** Carry canonical/display names into consensus and scoring, and add not-null/coverage assertions to CSV and workbook validation.\n"
                "2. **Stop presenting partial totals as ordinary fantasy projections.** Rename them to partial market-supported points or leave final point columns blank until required scoring components are available; add a completeness gate to workbook publication.\n"
                "3. **Make Kalshi weight depend on evidence quality.** Use a liquidity score based on spread, two-sided top-of-book depth, open interest, cumulative volume, and 24-hour activity; cap or exclude single-threshold, zero-volume, and very wide-spread curves.\n"
                "4. **Do not let flagged disagreements average through unchanged.** Quarantine, cap the outlying source's weight, or publish an uncertainty band when source disagreement is material.\n"
                "5. **Require two sources for headline consensus and fantasy scoring.** Keep single-source rows in an audit sheet, especially the 53 Kalshi-only rows, but do not treat them as equally decision-ready.\n"
                "6. **Backtest the full market-to-mean pipeline by source.** Preserve historical market snapshots and evaluate mean bias, calibration, and fantasy-score error against realized seasons; fit source-specific reliability and disagreement thresholds.\n"
                "7. **Use stat-specific disagreement limits.** A single 100-unit absolute cutoff has very different meaning for passing yards, receiving yards, and touchdowns."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## Further questions\n\n"
                "- Do Kalshi prices converge toward sportsbook-derived means as volume and open interest build closer to kickoff?\n"
                "- How stable are the single-threshold inversions across daily snapshots, especially when the midpoint is unchanged but top-of-book size changes?\n"
                "- Would a bounded or zero-inflated availability model better represent full-season outcomes than one negative-binomial family for every player within a stat?\n"
                "- Which single-source projections should remain visible for audit but be excluded from fantasy rankings?"
            ),
        },
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Technical audit of source agreement, Kalshi liquidity, mean inversion, consensus behavior, and final fantasy output quality.",
            "generatedAt": GENERATED_AT,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": manifest_sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": GENERATED_AT,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "pair_by_stat": results["pair_by_stat"],
                "kalshi_funnel": results["kalshi_funnel"],
                "triple_by_stat": results["triple_by_stat"],
                "triple_outliers": results["triple_outliers"],
                "output_quality": output_quality,
                "top_scores": results["top_scores"],
                "book_line_ratios": results["book_line_ratios"],
            },
            "accessIssues": [],
        },
        "sources": report_sources,
    }


def main() -> None:
    results = compute_audit()
    (HERE / "audit_results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    notebook = build_notebook(results)
    (HERE / "projection_audit.ipynb").write_text(json.dumps(notebook, indent=1, allow_nan=False) + "\n")
    artifact = build_report(results)
    (HERE / "artifact.json").write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    (HERE / "report_data.sql").write_text(build_report_sql(artifact["snapshot"]["datasets"]))
    print(f"Wrote analysis artifacts to {HERE}")


if __name__ == "__main__":
    main()
