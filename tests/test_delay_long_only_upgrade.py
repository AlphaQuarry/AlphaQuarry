from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_mining.simulation.delay import apply_delay
from alpha_mining.workflow.closed_loop import _scoreboard_score
from factor_research import (
    build_return_semantics_metadata,
    calculate_best_layer_metrics,
    calculate_long_only_portfolio_turnover,
    calculate_margin_metrics,
    process_future_return,
    summarize_long_only_turnover,
)
from factor_research.screening import compute_effectiveness_score_parts, evaluate_factor_effectiveness


def test_delay_return_semantics_metadata_matches_pct_chg_shift() -> None:
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=4),
            "znz_code": ["000001.SZ"] * 4,
            "close": [10.0, 11.0, 12.1, 13.31],
        }
    )
    df["pct_chg"] = df.groupby("znz_code")["close"].pct_change()
    out = process_future_return(df, return_col="pct_chg", period=1)
    assert np.isclose(out.loc[1, "pct_chg_1d"], 12.1 / 11.0 - 1.0)

    alpha = pd.DataFrame({"000001.SZ": [1.0, 2.0, 3.0]}, index=pd.date_range("2024-01-01", periods=3))
    delayed = apply_delay(alpha, 1)
    assert np.isnan(delayed.iloc[0, 0])
    assert delayed.iloc[1, 0] == 1.0

    meta = build_return_semantics_metadata("pct_chg", period=1, signal_delay=1)
    assert meta["future_return_col"] == "pct_chg_1d"
    assert meta["effective_raw_signal_return_window"] == "close[t+1] -> close[t+2]"
    assert meta["equivalent_exec_return_formula"] == "close[t+2] / close[t+1] - 1"
    assert meta["ret_exec_cc_main_col"] is False


def test_long_only_best_layer_turnover_and_margin() -> None:
    dates = pd.date_range("2024-01-01", periods=3)
    rows = []
    for date, top_ret, bottom_ret in zip(dates, [0.02, 0.01, 0.03], [-0.01, 0.00, -0.02]):
        rows.extend(
            [
                {"trade_date": date, "znz_code": "A", "layer": 1, "factor_a": -1.0, "pct_chg_1d": bottom_ret},
                {"trade_date": date, "znz_code": "B", "layer": 2, "factor_a": 1.0, "pct_chg_1d": top_ret},
            ]
        )
    layer_results = {"factor_a": pd.DataFrame(rows)}
    summary = pd.DataFrame({"factor": ["factor_a"], "ic_mean": [0.05]})

    best = calculate_best_layer_metrics(layer_results, ic_summary_df=summary, period=1)
    assert int(best.loc[0, "best_layer_label"]) == 2
    assert best.loc[0, "best_layer_annualized_return"] > 0

    turnover = calculate_long_only_portfolio_turnover(layer_results, ic_summary_df=summary, period=1)
    summary_to = summarize_long_only_turnover(turnover, factors=["factor_a"])
    assert summary_to.loc[0, "turnover_long_only_mean"] == 0.0

    margin = calculate_margin_metrics(turnover, factors=["factor_a"])
    assert bool(margin.loc[0, "margin_long_only_valid"]) is False


def test_long_only_scoring_and_scoreboard_prefer_score_total() -> None:
    summary = pd.DataFrame(
        {
            "factor": ["factor_a"],
            "ic_mean": [0.04],
            "ic_std": [0.10],
            "ir": [0.40],
            "positive_ic_ratio": [0.55],
            "t_stat": [2.0],
            "p_value": [0.05],
        }
    )
    best = pd.DataFrame(
        {
            "factor": ["factor_a"],
            "best_layer_annualized_return": [0.12],
            "best_minus_universe_annualized_return": [0.06],
            "best_layer_sharpe": [1.0],
            "best_layer_max_drawdown": [0.20],
            "best_layer_positive_month_ratio": [0.60],
        }
    )
    turnover = pd.DataFrame({"factor": ["factor_a"], "turnover_long_only_mean": [0.30]})
    margin = pd.DataFrame({"factor": ["factor_a"], "margin_long_only": [0.001], "best_layer_margin": [0.001]})
    coverage = pd.DataFrame({"factor": ["factor_a"], "coverage_rate": [0.95], "total_obs": [300], "non_missing_obs": [285]})
    stability = pd.DataFrame({"factor": ["factor_a"], "obs_count": [300]})
    mono = pd.DataFrame({"factor": ["factor_a"], "sign_adjusted_monotonicity": [0.20]})

    out = evaluate_factor_effectiveness(
        summary_df=summary,
        long_short_metrics={},
        best_layer_metrics_df=best,
        long_only_turnover_summary_df=turnover,
        margin_metrics_df=margin,
        coverage_overall_df=coverage,
        ic_stability_df=stability,
        monotonicity_summary_df=mono,
    )["factor_effectiveness_table"]
    assert bool(out.loc[0, "passed_hard_filter"]) is True
    assert out.loc[0, "effectiveness_score"] == out.loc[0, "score_total"]

    scores = _scoreboard_score(pd.DataFrame({"score_total": [10.0, 90.0], "long_short_sharpe_ratio": [99.0, -99.0]}))
    assert list(scores) == [10.0, 90.0]


def test_long_only_scoring_uses_requested_weights_and_abs_robust_ir() -> None:
    base = {
        "ic_mean": -0.08,
        "ir": -0.80,
        "sign_adjusted_monotonicity": 0.40,
        "best_layer_annualized_return": 0.50,
        "best_layer_sharpe": 2.50,
        "best_layer_max_drawdown": 0.05,
        "best_layer_margin": 0.030,
        "margin_long_only": 0.030,
        "best_minus_universe_annualized_return": 0.48,
        "best_layer_positive_month_ratio": 0.75,
        "yearly_sign_consistency": 0.50,
        "monthly_sign_consistency": 0.90,
        "turnover_long_only_mean": 0.15,
        "robust_ir_median": -0.31,
    }
    summary = pd.DataFrame(
        {
            "factor": ["factor_a"],
            "ic_mean": [base["ic_mean"]],
            "ic_std": [0.10],
            "ir": [base["ir"]],
            "positive_ic_ratio": [0.40],
            "t_stat": [-2.0],
            "p_value": [0.05],
        }
    )
    best = pd.DataFrame(
        {
            "factor": ["factor_a"],
            "best_layer_annualized_return": [base["best_layer_annualized_return"]],
            "best_minus_universe_annualized_return": [base["best_minus_universe_annualized_return"]],
            "best_layer_sharpe": [base["best_layer_sharpe"]],
            "best_layer_max_drawdown": [base["best_layer_max_drawdown"]],
            "best_layer_positive_month_ratio": [base["best_layer_positive_month_ratio"]],
        }
    )
    turnover = pd.DataFrame({"factor": ["factor_a"], "turnover_long_only_mean": [base["turnover_long_only_mean"]]})
    margin = pd.DataFrame({"factor": ["factor_a"], "margin_long_only": [base["margin_long_only"]], "best_layer_margin": [base["best_layer_margin"]]})
    coverage = pd.DataFrame({"factor": ["factor_a"], "coverage_rate": [0.70], "total_obs": [300], "non_missing_obs": [210]})
    stability = pd.DataFrame({"factor": ["factor_a"], "obs_count": [250]})
    mono = pd.DataFrame({"factor": ["factor_a"], "monotonicity_mean": [-0.40]})
    yearly = pd.DataFrame({"factor": ["factor_a"], "ic_mean": [-0.01]})
    monthly = pd.DataFrame({"factor": ["factor_a"], "ic_mean": [-0.01]})
    periods = pd.DataFrame(
        {
            "factor": ["factor_a", "factor_a"],
            "period": [1, 5],
            "ic_mean": [-0.08, -0.04],
            "ir": [-0.31, -0.50],
            "long_short_total_return": [0.01, 0.02],
        }
    )

    out = evaluate_factor_effectiveness(
        summary_df=summary,
        long_short_metrics={},
        best_layer_metrics_df=best,
        long_only_turnover_summary_df=turnover,
        margin_metrics_df=margin,
        coverage_overall_df=coverage,
        ic_stability_df=stability,
        monotonicity_summary_df=mono,
        ic_yearly_df=yearly,
        ic_monthly_df=monthly,
        period_comparison_df=periods,
    )["factor_effectiveness_table"]

    row = out.iloc[0]
    assert bool(row["passed_hard_filter"]) is True
    assert row["score_stability"] == 100.0
    assert row["score_tradeability"] == 100.0
    assert row["score_total"] == 100.0


def test_predictive_power_ic_decay_spearman_is_optional_and_equally_weighted() -> None:
    base = {
        "ic_mean": 0.08,
        "ir": 0.80,
        "sign_adjusted_monotonicity": 0.40,
        "best_layer_annualized_return": 0.05,
        "best_layer_sharpe": 0.50,
        "best_layer_max_drawdown": 0.50,
        "best_layer_margin": 0.001,
        "margin_long_only": 0.001,
        "best_minus_universe_annualized_return": 0.03,
        "best_layer_positive_month_ratio": 0.48,
        "yearly_sign_consistency": 0.50,
        "monthly_sign_consistency": 0.55,
        "turnover_long_only_mean": 0.75,
    }

    without_decay = compute_effectiveness_score_parts(base)
    with_full_decay = compute_effectiveness_score_parts({**base, "ic_decay_spearman": -0.40})
    with_half_decay = compute_effectiveness_score_parts({**base, "ic_decay_spearman": 0.20})

    assert without_decay["score_predictive_power"] == pytest.approx(100.0)
    assert with_full_decay["score_predictive_power"] == pytest.approx(100.0)
    assert with_half_decay["score_predictive_power"] == pytest.approx(87.5)


def test_evaluate_factor_effectiveness_uses_preferred_phase_ic_decay_spearman() -> None:
    summary = pd.DataFrame(
        {
            "factor": ["factor_a"],
            "ic_mean": [0.08],
            "ic_std": [0.10],
            "ir": [0.80],
            "positive_ic_ratio": [0.65],
            "t_stat": [3.0],
            "p_value": [0.01],
        }
    )
    best = pd.DataFrame(
        {
            "factor": ["factor_a"],
            "best_layer_annualized_return": [0.20],
            "best_minus_universe_annualized_return": [0.10],
            "best_layer_sharpe": [1.50],
            "best_layer_max_drawdown": [0.10],
            "best_layer_positive_month_ratio": [0.65],
        }
    )
    turnover = pd.DataFrame({"factor": ["factor_a"], "turnover_long_only_mean": [0.30]})
    margin = pd.DataFrame({"factor": ["factor_a"], "margin_long_only": [0.002], "best_layer_margin": [0.002]})
    coverage = pd.DataFrame({"factor": ["factor_a"], "coverage_rate": [0.95], "total_obs": [300], "non_missing_obs": [285]})
    stability = pd.DataFrame({"factor": ["factor_a"], "obs_count": [300]})
    mono = pd.DataFrame({"factor": ["factor_a"], "monotonicity_mean": [0.40]})
    decay = pd.DataFrame(
        {
            "factor": ["factor_a", "factor_a"],
            "phase": ["train", "val"],
            "ic_decay_rank_corr": [0.25, -0.40],
        }
    )

    out = evaluate_factor_effectiveness(
        summary_df=summary,
        long_short_metrics={},
        best_layer_metrics_df=best,
        long_only_turnover_summary_df=turnover,
        margin_metrics_df=margin,
        coverage_overall_df=coverage,
        ic_stability_df=stability,
        monotonicity_summary_df=mono,
        ic_decay_df=decay,
    )["factor_effectiveness_table"]

    assert out.loc[0, "ic_decay_spearman"] == pytest.approx(0.25)
