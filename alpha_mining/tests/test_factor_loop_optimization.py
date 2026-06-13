from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd

from alpha_mining.mining.candidate_ranker import CandidateRanker, CandidateRankerConfig
from alpha_mining.mining.factor_family import (
    infer_factor_family,
    infer_factor_family_mix,
)
from alpha_mining.workflow.closed_loop import _scoreboard_score
from factor_research import (
    SampleSplitConfig,
    assign_sample_split,
    build_portfolio_pnl_table,
    calculate_long10_portfolio_returns,
    calculate_long_only_portfolio_turnover,
    double_sort_analysis,
    newey_west_stats,
    summarize_long10_portfolio_returns,
    summarize_split_metrics,
)
from factor_research.single_factor import _group_frames_by_trade_date


def test_factor_family_inference_uses_four_institutional_families() -> None:
    assert infer_factor_family("close", category="price") == "price_volume"
    assert infer_factor_family("moneyflow_net_amount", category="moneyflow") == "moneyflow"
    assert infer_factor_family("fin_roe", category="finance") == "fundamental"
    assert infer_factor_family("report_rc_eps_mean", category="analyst") == "analyst"
    family, mix_json = infer_factor_family_mix(["close", "fin_roe", "fin_roa"])
    mix = json.loads(mix_json)
    assert family == "fundamental,price_volume"
    assert mix["counts"] == {"fundamental": 2, "price_volume": 1}
    assert np.isclose(mix["ratios"]["fundamental"], 2.0 / 3.0)
    assert mix["primary_factor_family"] == "fundamental"


def test_candidate_ranker_uses_factor_family_feedback_weights() -> None:
    df = pd.DataFrame(
        {
            "candidate_id": ["c1", "c2"],
            "expression": ["rank(close)", "rank(report_rc_eps_mean)"],
            "prefilter_status": ["pass", "pass"],
            "sample_status": ["", ""],
            "fields": ["close", "report_rc_eps_mean"],
            "operators": ["rank", "rank"],
            "family": ["cross_sectional", "cross_sectional"],
            "factor_family": ["price_volume", "analyst"],
            "layer": ["L1", "L1"],
            "windows": ["", ""],
            "groups": ["", ""],
            "operator_count": [1, 1],
            "field_count": [1, 1],
            "depth": [1, 1],
        }
    )
    ranked = CandidateRanker(CandidateRankerConfig(min_explore_ratio=0.0)).rank(
        df,
        feedback_hints={"factor_family_weights": {"analyst": 5.0}},
        max_eval=2,
    )
    assert ranked.iloc[0]["candidate_id"] == "c2"
    assert ranked.iloc[0]["factor_family_feedback_score"] > 0


def test_candidate_ranker_enforces_factor_family_max_ratio_when_quota_enabled() -> None:
    rows = []
    for i in range(8):
        rows.append(
            {
                "candidate_id": f"pv{i}",
                "expression": f"rank(close_{i})",
                "prefilter_status": "pass",
                "sample_status": "",
                "fields": f"close_{i}",
                "operators": "rank",
                "family": "cross_sectional",
                "factor_family": "price_volume",
                "layer": "L1",
                "windows": "",
                "groups": "",
                "operator_count": 1,
                "field_count": 1,
                "depth": 1,
            }
        )
    for i, family in enumerate(["fundamental", "moneyflow", "analyst", "fundamental"]):
        rows.append(
            {
                "candidate_id": f"other{i}",
                "expression": f"rank(field_{i})",
                "prefilter_status": "pass",
                "sample_status": "",
                "fields": f"field_{i}",
                "operators": "rank",
                "family": "cross_sectional",
                "factor_family": family,
                "layer": "L1",
                "windows": "",
                "groups": "",
                "operator_count": 1,
                "field_count": 1,
                "depth": 1,
            }
        )
    ranked = CandidateRanker(CandidateRankerConfig(min_explore_ratio=0.0, family_max_selected_ratio=0.5)).rank(
        pd.DataFrame(rows),
        feedback_hints={"factor_family_weights": {"price_volume": 10.0}},
        max_eval=6,
    )
    counts = ranked["factor_family"].value_counts()
    assert int(counts.get("price_volume", 0)) <= 3

    unbounded = CandidateRanker(
        CandidateRankerConfig(
            min_explore_ratio=0.0,
            use_factor_family_quota=False,
            family_max_selected_ratio=0.5,
        )
    ).rank(
        pd.DataFrame(rows),
        feedback_hints={"factor_family_weights": {"price_volume": 10.0}},
        max_eval=6,
    )
    assert int(unbounded["factor_family"].value_counts().get("price_volume", 0)) > 3


def test_double_sort_analysis_returns_matrix_spread_and_summary() -> None:
    dates = pd.date_range("2025-01-01", periods=4)
    rows = []
    for date in dates:
        for i in range(25):
            rows.append(
                {
                    "trade_date": date,
                    "znz_code": f"S{i:03d}",
                    "alpha_a": float(i),
                    "total_mv": float(25 - i),
                    "pct_chg_1d": float(i) / 1000.0,
                }
            )
    out = double_sort_analysis(
        pd.DataFrame(rows),
        factor_cols=["alpha_a"],
        return_col="pct_chg_1d",
        control_col="total_mv",
        factor_bins=5,
        control_bins=5,
        method="conditional",
    )
    assert not out["matrix_returns_df"].empty
    assert not out["spread_returns_df"].empty
    assert not out["summary_df"].empty
    row = out["summary_df"].iloc[0]
    assert row["control_col_used"] == "total_mv"
    assert "double_sort_p_value" in out["summary_df"].columns
    assert "double_sort_positive_ratio" in out["summary_df"].columns
    assert "double_sort_annualized_return" in out["summary_df"].columns
    assert "double_sort_monotonicity_spearman" in out["summary_df"].columns
    assert "double_sort_group_min_count" in out["summary_df"].columns
    stats = newey_west_stats(out["spread_returns_df"]["double_sort_spread_return"])
    assert {"mean", "nw_t", "p_value", "obs"}.issubset(stats)


def test_sample_split_scores_validation_without_using_oos() -> None:
    df = pd.DataFrame(
        {
            "factor": ["alpha_a", "alpha_a", "alpha_a"],
            "trade_date": pd.to_datetime(["2024-12-31", "2025-06-01", "2026-02-01"]),
            "score_total": [10.0, 80.0, 5.0],
            "ir": [0.1, 0.8, 0.05],
        }
    )
    split_df = assign_sample_split(df, config=SampleSplitConfig())
    summary = summarize_split_metrics(split_df, factor_col="factor", metric_cols=["score_total", "ir"])
    row = summary.iloc[0]
    assert row["train_score_total_mean"] == 10.0
    assert row["validation_score_total_mean"] == 80.0
    assert row["oos_score_total_mean"] == 5.0
    assert row["train_score"] == 10.0
    assert row["validation_score"] == 80.0
    assert row["oos_score"] == 5.0
    assert row["validation_decay_ratio"] == 8.0
    assert row["oos_decay_ratio"] == 0.5
    assert bool(row["split_pass"])
    assert row["split_warning_reasons"] == ""

    scores = _scoreboard_score(pd.DataFrame({"score_total": [100.0, 1.0], "validation_score": [1.0, 90.0]}))
    assert list(scores) == [100.0, 1.0]


def test_long_only_turnover_entry_exit_respects_can_buy_and_can_sell() -> None:
    rows = [
        {
            "trade_date": "2025-01-01",
            "znz_code": "A",
            "layer": 1,
            "alpha_a": 0.1,
            "pct_chg_1d": 0.00,
            "can_buy": 1,
            "can_sell": 1,
        },
        {
            "trade_date": "2025-01-01",
            "znz_code": "B",
            "layer": 2,
            "alpha_a": 0.9,
            "pct_chg_1d": 0.01,
            "can_buy": 1,
            "can_sell": 1,
        },
        {
            "trade_date": "2025-01-02",
            "znz_code": "B",
            "layer": 1,
            "alpha_a": 0.2,
            "pct_chg_1d": 0.02,
            "can_buy": 1,
            "can_sell": 0,
        },
        {
            "trade_date": "2025-01-02",
            "znz_code": "C",
            "layer": 2,
            "alpha_a": 0.8,
            "pct_chg_1d": 0.04,
            "can_buy": 1,
            "can_sell": 1,
        },
    ]
    result = calculate_long_only_portfolio_turnover(
        {"alpha_a": pd.DataFrame(rows)},
        ic_summary_df=pd.DataFrame({"factor": ["alpha_a"], "ic_mean": [0.1]}),
        period=1,
        apply_tradability_constraints=True,
        tradability_mode="entry_exit",
    )["alpha_a"]
    second = result.sort_values("trade_date").iloc[1]
    assert int(second["blocked_sell_count"]) == 1
    assert np.isclose(float(second["blocked_sell_ratio"]), 1.0)
    assert np.isclose(float(second["blocked_buy_ratio"]), 0.0)
    assert int(second["holding_count"]) == 2
    assert np.isclose(float(second["portfolio_return_long_only"]), 0.03)
    assert np.isclose(float(second["tradability_return_drag"]), 0.01)


def test_trade_date_frame_cache_normalizes_dates_without_mutating_source() -> None:
    source = pd.DataFrame(
        {
            "trade_date": ["2025-01-02", "2025-01-01", "2025-01-01"],
            "znz_code": ["B", "A", "C"],
            "alpha_a": [0.2, 0.1, 0.3],
        }
    )

    work, grouped = _group_frames_by_trade_date(source)

    assert not pd.api.types.is_datetime64_any_dtype(source["trade_date"])
    assert pd.api.types.is_datetime64_any_dtype(work["trade_date"])
    assert list(grouped.keys()) == [
        pd.Timestamp("2025-01-02"),
        pd.Timestamp("2025-01-01"),
    ]
    assert grouped[pd.Timestamp("2025-01-01")]["znz_code"].tolist() == ["A", "C"]


def test_long10_portfolio_follows_ic_direction() -> None:
    dates = pd.date_range("2025-01-01", periods=2)
    rows = []
    for date in dates:
        for i in range(12):
            rows.append(
                {
                    "trade_date": date,
                    "znz_code": f"S{i:02d}",
                    "layer": 1 if i < 6 else 2,
                    "alpha_a": float(i),
                    "pct_chg_1d": float(i) / 100.0,
                    "can_buy": 1,
                    "can_sell": 1,
                }
            )
    layer_results = {"alpha_a": pd.DataFrame(rows)}

    positive = calculate_long10_portfolio_returns(
        layer_results,
        ic_summary_df=pd.DataFrame({"factor": ["alpha_a"], "ic_mean": [0.1]}),
        top_n=10,
    )["alpha_a"].sort_values("trade_date")
    negative = calculate_long10_portfolio_returns(
        layer_results,
        ic_summary_df=pd.DataFrame({"factor": ["alpha_a"], "ic_mean": [-0.1]}),
        top_n=10,
    )["alpha_a"].sort_values("trade_date")

    assert np.isclose(
        float(positive.iloc[0]["portfolio_return_long10"]),
        sum(range(2, 12)) / 10 / 100.0,
    )
    assert np.isclose(
        float(negative.iloc[0]["portfolio_return_long10"]),
        sum(range(0, 10)) / 10 / 100.0,
    )
    assert int(positive.iloc[0]["holding_count_long10"]) == 10
    summary = summarize_long10_portfolio_returns({"alpha_a": positive}, factors=["alpha_a"])
    assert summary.loc[0, "holding_count_long10_mean"] == 10.0


def test_portfolio_pnl_table_is_compact_and_includes_expected_portfolios() -> None:
    dates = pd.date_range("2025-01-01", periods=2)
    layer_visual = {
        "alpha_a": pd.DataFrame(
            {
                "trade_date": [
                    dates[0],
                    dates[0],
                    dates[0],
                    dates[1],
                    dates[1],
                    dates[1],
                ],
                "layer": [1, 2, "long_short", 1, 2, "long_short"],
                "pct_chg_1d": [0.01, 0.03, 0.01, -0.02, 0.04, 0.03],
            }
        )
    }
    long_only = {
        "alpha_a": pd.DataFrame(
            {
                "trade_date": dates,
                "factor": ["alpha_a", "alpha_a"],
                "portfolio_return_long_only": [0.03, 0.04],
                "turnover_long_only": [0.0, 0.5],
                "holding_count": [6, 6],
                "blocked_buy_ratio": [0.0, 0.0],
                "blocked_sell_ratio": [0.0, 0.0],
                "tradability_return_drag": [0.0, 0.0],
            }
        )
    }
    long10 = {
        "alpha_a": pd.DataFrame(
            {
                "trade_date": dates,
                "factor": ["alpha_a", "alpha_a"],
                "portfolio_return_long10": [0.025, 0.035],
                "turnover_long10": [0.0, 0.4],
                "holding_count_long10": [10, 10],
                "blocked_buy_ratio_long10": [0.0, 0.1],
                "blocked_sell_ratio_long10": [0.0, 0.0],
                "tradability_return_drag_long10": [0.0, 0.002],
            }
        )
    }
    turnover_results = {
        "alpha_a": pd.DataFrame(
            {
                "trade_date": [dates[1]],
                "min_layer_turnover": [0.2],
                "max_layer_turnover": [0.6],
            }
        )
    }

    pnl = build_portfolio_pnl_table(
        layer_results_for_visualization=layer_visual,
        long_only_turnover_results=long_only,
        long10_portfolio_returns=long10,
        turnover_results=turnover_results,
    )

    legacy_columns = [
        "factor",
        "trade_date",
        "portfolio",
        "return",
        "cum_return",
        "holding_count",
        "turnover",
        "blocked_buy_ratio",
        "blocked_sell_ratio",
        "tradability_return_drag",
    ]
    assert legacy_columns == [col for col in legacy_columns if col in pnl.columns]
    assert {"return_gross", "cum_return_gross", "has_net_pnl"}.issubset(set(pnl.columns))
    assert np.allclose(pnl["return"], pnl["return_gross"], equal_nan=True)
    assert np.allclose(pnl["cum_return"], pnl["cum_return_gross"], equal_nan=True)
    assert not bool(pnl["has_net_pnl"].fillna(False).any())
    assert set(pnl["portfolio"]) == {
        "layer_1",
        "layer_2",
        "long_short",
        "long_only",
        "long_10",
    }
    long10_rows = pnl[pnl["portfolio"] == "long_10"].sort_values("trade_date")
    assert np.isclose(float(long10_rows.iloc[1]["cum_return"]), (1.0 + 0.025) * (1.0 + 0.035) - 1.0)
    long_short_rows = pnl[pnl["portfolio"] == "long_short"].sort_values("trade_date")
    assert np.isnan(long_short_rows.iloc[0]["turnover"])
    assert np.isclose(float(long_short_rows.iloc[1]["turnover"]), 0.4)
    assert "raw_return" not in pnl.columns
    assert "source_return_col" not in pnl.columns


def test_reproduce_alpha_oos_help_exposes_oos_review_interface() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/reproduce_alpha_oos.py", "--help"],
        cwd=".",
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--oos-start" in result.stdout
    assert "--alpha-names" in result.stdout


def test_closed_loop_cli_help_exposes_family_ratio_and_split_date_options() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_closed_loop.py", "--help"],
        cwd=".",
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    for flag in [
        "--family-max-selected-ratio",
        "--family-min-explore-ratio",
        "--train-start",
        "--train-end",
        "--validation-start",
        "--validation-end",
        "--oos-start",
        "--neutralization",
        "--transaction-cost-enabled",
        "--no-transaction-cost",
    ]:
        assert flag in result.stdout
