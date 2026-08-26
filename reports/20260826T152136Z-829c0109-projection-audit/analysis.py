from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20260826T152136Z-829c0109"
RUN_DIR = REPO_ROOT / "runs" / RUN_ID
ARTIFACTS = RUN_DIR / "artifacts"


def _records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.replace({np.nan: None, np.inf: None, -np.inf: None})
    return clean.to_dict(orient="records")


def compute_audit() -> dict:
    priced = pd.read_csv(ARTIFACTS / "priced_markets.csv")
    source = pd.read_csv(ARTIFACTS / "source_projections.csv")
    consensus = pd.read_csv(ARTIFACTS / "consensus_stats.csv")
    fantasy = pd.read_csv(ARTIFACTS / "fantasy_projections.csv")
    player_map = pd.read_csv(ARTIFACTS / "player_map.csv")
    dispersion = pd.read_csv(ARTIFACTS / "dispersion_calibration.csv")
    model_validation = json.loads((ARTIFACTS / "model_validation.json").read_text())
    manifest = json.loads((RUN_DIR / "metadata" / "manifest.json").read_text())
    kalshi_raw = json.loads((RUN_DIR / "raw" / "kalshi.json").read_text())

    eligible = source[(source["quality_status"] == "passed") & source["mean"].notna()].copy()
    pivot = eligible.pivot_table(
        index=["canonical_player_id", "canonical_player_name", "stat"],
        columns="source",
        values="mean",
        aggfunc="first",
    )

    pair_rows = []
    pair_summary = []
    for left, right, pair_label in [
        ("draftkings", "fanduel", "DraftKings vs FanDuel"),
        ("draftkings", "kalshi", "DraftKings vs Kalshi"),
        ("fanduel", "kalshi", "FanDuel vs Kalshi"),
    ]:
        overlap = pivot[[left, right]].dropna().copy()
        overlap["symmetric_abs_difference"] = (
            2 * (overlap[right] - overlap[left]).abs() / (overlap[right] + overlap[left])
        )
        overlap["right_vs_left"] = (overlap[right] - overlap[left]) / overlap[left]
        pair_summary.append(
            {
                "pair": pair_label,
                "overlap": int(len(overlap)),
                "correlation": float(overlap[left].corr(overlap[right])),
                "median_abs_difference": float(overlap["symmetric_abs_difference"].median()),
                "mean_signed_difference": float(overlap["right_vs_left"].mean()),
            }
        )
        for stat, group in overlap.groupby(level="stat"):
            pair_rows.append(
                {
                    "pair": pair_label,
                    "stat": stat.replace("_", " ").title(),
                    "comparison": f"{pair_label.replace('DraftKings', 'DK').replace('FanDuel', 'FD')} · {stat.replace('_', ' ').title()}",
                    "overlap": int(len(group)),
                    "median_abs_difference": float(group["symmetric_abs_difference"].median()),
                    "median_right_to_left_ratio": float((group[right] / group[left]).median()),
                }
            )

    raw_liquidity_rows = []
    for market in kalshi_raw["markets"]:
        orderbook = market["orderbook"]
        activity = market["trading_activity"]
        bid_size = orderbook.get("best_yes_bid_size_contracts")
        ask_size = orderbook.get("best_yes_ask_size_contracts")
        raw_liquidity_rows.append(
            {
                "source_market_id": market["ticker"],
                "volume_24h": activity["volume_24h_contracts"],
                "min_top_size": min(bid_size, ask_size)
                if bid_size is not None and ask_size is not None
                else np.nan,
            }
        )
    raw_liquidity = pd.DataFrame(raw_liquidity_rows)
    kalshi_quotes = priced[priced["source"] == "kalshi"].merge(
        raw_liquidity, on="source_market_id", how="left"
    )
    contributing_quotes = kalshi_quotes[
        kalshi_quotes["source_projection_quality_status"] == "passed"
    ].copy()
    curve_liquidity = (
        contributing_quotes.groupby(["canonical_player_id", "stat"])
        .agg(
            contributing_quotes=("source_market_id", "size"),
            total_volume=("volume", "sum"),
            total_open_interest=("open_interest", "sum"),
            minimum_open_interest=("open_interest", "min"),
            volume_24h=("volume_24h", "sum"),
            median_spread=("spread", "median"),
            maximum_spread=("spread", "max"),
            minimum_top_size=("min_top_size", "min"),
        )
        .reset_index()
    )
    eligible_kalshi = eligible[eligible["source"] == "kalshi"].merge(
        curve_liquidity, on=["canonical_player_id", "stat"], how="left"
    )

    triples = consensus[consensus["source_list"] == "draftkings|fanduel|kalshi"].copy()
    triples["sportsbook_mean"] = (triples["draftkings_mean"] + triples["fanduel_mean"]) / 2
    triples["kalshi_vs_books"] = (
        triples["kalshi_mean"] - triples["sportsbook_mean"]
    ) / triples["sportsbook_mean"]
    triples["absolute_kalshi_vs_books"] = triples["kalshi_vs_books"].abs()
    triples["consensus_vs_books"] = (
        triples["consensus_mean"] - triples["sportsbook_mean"]
    ) / triples["sportsbook_mean"]
    names = source[["canonical_player_id", "canonical_player_name", "canonical_position"]].drop_duplicates(
        "canonical_player_id"
    )
    triples = triples.merge(
        names[["canonical_player_id", "canonical_player_name"]],
        on="canonical_player_id",
        how="left",
    ).merge(curve_liquidity, on=["canonical_player_id", "stat"], how="left")
    triples["stat_label"] = triples["stat"].str.replace("_", " ").str.title()

    correlations = {}
    for field in ["total_volume", "minimum_open_interest", "volume_24h", "median_spread"]:
        valid = triples[["absolute_kalshi_vs_books", field]].dropna()
        rho, p_value = spearmanr(valid["absolute_kalshi_vs_books"], valid[field])
        correlations[field] = {"rho": float(rho), "p_value": float(p_value), "n": int(len(valid))}

    published = consensus[consensus["quality_status"] == "passed"].copy()
    published["has_kalshi"] = published["source_list"].str.contains("kalshi", na=False)
    disagreement = (
        published.groupby("has_kalshi")
        .agg(rows=("stat", "size"), flags=("disagreement_flag", "sum"), rate=("disagreement_flag", "mean"))
        .reset_index()
    )

    kalshi_passed = source[(source["source"] == "kalshi") & (source["quality_status"] == "passed")]
    kalshi_single_threshold = int((kalshi_passed["eligible_quote_count"] == 1).sum())
    kalshi_only = consensus[consensus["source_list"] == "kalshi"].merge(
        curve_liquidity, on=["canonical_player_id", "stat"], how="left"
    )

    source_snapshots = pd.to_datetime(manifest["source_snapshot_times_utc"], utc=True)
    source_skew_seconds = float((source_snapshots.max() - source_snapshots.min()).total_seconds())

    sportsbook_quotes = priced[
        priced["source"].isin(["draftkings", "fanduel"])
        & (priced["inclusion_status"] == "included")
    ].copy()
    sportsbook_quotes["mean_to_threshold"] = (
        sportsbook_quotes["source_projection_mean"] / sportsbook_quotes["canonical_threshold"]
    )
    book_line_ratios = (
        sportsbook_quotes.groupby(["source", "stat"])
        .agg(
            quotes=("mean_to_threshold", "size"),
            median_mean_to_threshold=("mean_to_threshold", "median"),
            median_overround=("market_overround", "median"),
        )
        .reset_index()
    )
    book_line_ratios["source"] = book_line_ratios["source"].str.title()
    book_line_ratios["stat"] = book_line_ratios["stat"].str.replace("_", " ").str.title()

    fantasy_named = fantasy.drop(columns=["canonical_player_name"]).merge(
        names, on="canonical_player_id", how="left"
    )
    top_scores = fantasy_named[fantasy_named["fpts_standard"].notna()].nlargest(15, "fpts_standard")

    warning_checks = [
        check
        for check in model_validation["checks"]
        if check["severity"] == "warning" and not check["passed"]
    ]
    warning_rows = []
    for check in warning_checks:
        warning_rows.append(
            {
                "check": check["name"].replace("model.", ""),
                "message": check["message"],
                "details": json.dumps(check.get("details", {}), sort_keys=True),
            }
        )

    funnel = [
        {"stage": "Raw Kalshi contracts", "count": int(len(kalshi_quotes))},
        {
            "stage": "Two-sided and spread-eligible",
            "count": int((kalshi_quotes["inclusion_status"] == "included").sum()),
        },
        {"stage": "Quotes on curves used", "count": int(len(contributing_quotes))},
        {"stage": "Eligible player/stat curves", "count": int(len(eligible_kalshi))},
        {"stage": "Stats with Kalshi dispersion update", "count": int((dispersion["dispersion_source"] != "historical_only").sum())},
    ]

    summary = {
        "run_id": RUN_ID,
        "source_skew_seconds": source_skew_seconds,
        "eligible_source_curves": {
            key: int(value) for key, value in eligible.groupby("source").size().to_dict().items()
        },
        "published_consensus_rows": int(len(published)),
        "single_source_consensus_rows": int((published["source_count"] == 1).sum()),
        "single_source_consensus_rate": float((published["source_count"] == 1).mean()),
        "kalshi_disagreement_rows": int(disagreement.loc[disagreement["has_kalshi"], "flags"].iloc[0]),
        "kalshi_disagreement_rate": float(disagreement.loc[disagreement["has_kalshi"], "rate"].iloc[0]),
        "non_kalshi_disagreement_rows": int(disagreement.loc[~disagreement["has_kalshi"], "flags"].iloc[0]),
        "non_kalshi_disagreement_rate": float(disagreement.loc[~disagreement["has_kalshi"], "rate"].iloc[0]),
        "kalshi_raw_quotes": int(len(kalshi_quotes)),
        "kalshi_pricing_eligible_quotes": int((kalshi_quotes["inclusion_status"] == "included").sum()),
        "kalshi_contributing_quotes": int(len(contributing_quotes)),
        "kalshi_eligible_curves": int(len(eligible_kalshi)),
        "kalshi_curves_no_24h_volume": int((eligible_kalshi["volume_24h"] == 0).sum()),
        "kalshi_curves_zero_lifetime_volume": int((eligible_kalshi["total_volume"] == 0).sum()),
        "kalshi_curves_min_oi_le_10": int((eligible_kalshi["minimum_open_interest"] <= 10).sum()),
        "kalshi_curves_max_spread_gt_10c": int((eligible_kalshi["maximum_spread"] > 0.10).sum()),
        "kalshi_single_eligible_threshold_curves": kalshi_single_threshold,
        "kalshi_only_consensus_rows": int(len(kalshi_only)),
        "kalshi_only_zero_24h_volume": int((kalshi_only["volume_24h"] == 0).sum()),
        "kalshi_dispersion_updates": int((dispersion["dispersion_source"] != "historical_only").sum()),
        "quarantined_kalshi_curves": int(model_validation["summary"]["quarantined_kalshi_curves"]),
        "fantasy_rows": int(len(fantasy)),
        "fantasy_blank_names": int(fantasy["canonical_player_name"].isna().sum()),
        "fantasy_numeric_scores": int(fantasy["fpts_standard"].notna().sum()),
        "fantasy_complete_rows": int(fantasy["projection_complete"].sum()),
        "identity_unmatched": int((player_map["review_status"] == "unmatched").sum()),
        "model_warnings": int(model_validation["summary"]["warnings"]),
        "model_errors": int(model_validation["summary"]["errors"]),
    }

    return {
        "summary": summary,
        "pair_summary": pair_summary,
        "pair_by_stat": pair_rows,
        "kalshi_funnel": funnel,
        "triple_outliers": _records(
            triples.nlargest(15, "absolute_kalshi_vs_books")[[
                "canonical_player_name",
                "stat_label",
                "draftkings_mean",
                "fanduel_mean",
                "kalshi_mean",
                "kalshi_vs_books",
                "consensus_vs_books",
                "contributing_quotes",
                "total_volume",
                "minimum_open_interest",
                "volume_24h",
                "median_spread",
                "minimum_top_size",
            ]]
        ),
        "triple_by_stat": _records(
            triples.groupby("stat_label")
            .agg(
                overlaps=("stat_label", "size"),
                median_kalshi_vs_books=("kalshi_vs_books", "median"),
                median_abs_kalshi_vs_books=("absolute_kalshi_vs_books", "median"),
                median_consensus_shift=("consensus_vs_books", "median"),
                disagreement_flags=("disagreement_flag", "sum"),
            )
            .reset_index()
        ),
        "liquidity_correlations": correlations,
        "book_line_ratios": _records(book_line_ratios),
        "model_warning_rows": warning_rows,
        "top_scores": _records(
            top_scores[[
                "canonical_player_name",
                "canonical_position",
                "scoring_profile",
                "fpts_standard",
                "fpts_full_ppr",
                "passing_yards",
                "passing_touchdowns",
                "rushing_yards",
                "rushing_touchdowns",
                "receiving_yards",
                "receiving_touchdowns",
                "receptions",
            ]]
        ),
    }


def print_audit(results: dict) -> None:
    summary = results["summary"]
    print("Projection pipeline audit")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nPairwise source agreement")
    print(pd.DataFrame(results["pair_summary"]).to_string(index=False))
    print("\nThree-source behavior by stat")
    print(pd.DataFrame(results["triple_by_stat"]).to_string(index=False))
    print("\nKalshi liquidity funnel")
    print(pd.DataFrame(results["kalshi_funnel"]).to_string(index=False))
    print("\nLargest Kalshi disagreements")
    print(pd.DataFrame(results["triple_outliers"]).to_string(index=False))


if __name__ == "__main__":
    print_audit(compute_audit())
