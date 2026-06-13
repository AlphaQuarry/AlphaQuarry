import unittest

import numpy as np
import pandas as pd

from factor_alalyze_lib import (
    analyze_holding_period_robustness,
    build_factor_summary_report,
    calculate_factor_weights,
    calculate_factor_correlation,
    calculate_factor_returns,
    calculate_factor_coverage,
    calculate_icir,
    calculate_ic_stability,
    calculate_ic_time_breakdown,
    calculate_layer_monotonicity,
    calculate_turnover_rate,
    factor_layer_analysis,
    filter_effective_factors,
    filter_factors_by_correlation_advanced,
    calculate_long_short_metrics,
    predict_stock_returns,
    process_factor_data,
    process_future_return,
)
from factor_research.utils import calculate_risk_metrics


def make_synthetic_panel(n_dates: int = 30, n_stocks: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    codes = [f"S{i:03d}" for i in range(n_stocks)]
    rows = []
    for d in dates:
        for c in codes:
            base = rng.normal()
            rows.append(
                {
                    "trade_date": d,
                    "znz_code": c,
                    "pct_chg": rng.normal(0.0005, 0.02),
                    "circ_mv": float(rng.uniform(1e8, 1e11)),
                    "factor_a": base + rng.normal(0, 0.1),
                    "factor_b": -base + rng.normal(0, 0.1),
                }
            )
    return pd.DataFrame(rows)


class TestMinimalPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.df = make_synthetic_panel()

    def test_process_future_return(self):
        out = process_future_return(self.df, return_col="pct_chg", period=3)
        self.assertIn("pct_chg_3d", out.columns)
        self.assertGreater(out["pct_chg_3d"].notna().sum(), 0)

    def test_process_future_return_unsorted_input_alignment(self):
        sorted_df = self.df.sort_values(["znz_code", "trade_date"]).reset_index(drop=True)
        unsorted_df = sorted_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        out_sorted = process_future_return(sorted_df, return_col="pct_chg", period=3)
        out_unsorted = process_future_return(unsorted_df, return_col="pct_chg", period=3)

        lhs = out_sorted[["znz_code", "trade_date", "pct_chg_3d"]].sort_values(["znz_code", "trade_date"]).reset_index(drop=True)
        rhs = out_unsorted[["znz_code", "trade_date", "pct_chg_3d"]].sort_values(["znz_code", "trade_date"]).reset_index(drop=True)
        pd.testing.assert_series_equal(lhs["pct_chg_3d"], rhs["pct_chg_3d"], check_names=False)

    def test_process_factor_data(self):
        out = process_factor_data(
            self.df[["trade_date", "znz_code", "pct_chg", "circ_mv", "factor_a"]].copy(),
            ["factor_a"],
            market_value_column="circ_mv",
            is_timeseries=True,
        )
        self.assertIn("factor_a", out.columns)
        per_date_mean = out.groupby("trade_date")["factor_a"].mean().abs().mean()
        self.assertLess(per_date_mean, 1e-6 + 0.05)

    def test_process_factor_data_multi_factor_dropna_independence(self):
        raw = self.df[["trade_date", "znz_code", "pct_chg", "circ_mv", "factor_a"]].copy()
        raw["factor_b"] = np.nan
        out = process_factor_data(
            raw,
            ["factor_a", "factor_b"],
            market_value_column="circ_mv",
            is_timeseries=True,
            do_clip=False,
            do_neutralize=True,
            do_standardize=False,
        )
        self.assertGreater(len(out), 0)
        self.assertGreater(out["factor_a"].notna().sum(), 0)
        self.assertEqual(int(out["factor_b"].notna().sum()), 0)

    def test_process_factor_data_bypass_without_market_value(self):
        raw = self.df[["trade_date", "znz_code", "factor_a"]].copy()
        out = process_factor_data(
            raw,
            ["factor_a"],
            market_value_column="circ_mv",
            is_timeseries=True,
            do_clip=False,
            do_neutralize=False,
            do_standardize=False,
        )
        self.assertIn("factor_a", out.columns)
        self.assertGreater(len(out), 0)

    def test_calculate_icir(self):
        df2 = process_future_return(self.df, return_col="pct_chg", period=1)
        ic_df, summary_df = calculate_icir(df2, ["factor_a"], return_col="pct_chg", period=1)
        self.assertIn("factor_a_ic", ic_df.columns)
        self.assertIn("factor", summary_df.columns)
        self.assertGreaterEqual(len(summary_df), 1)

    def test_calculate_icir_with_future_only_return_column(self):
        period = 3
        df2 = process_future_return(self.df, return_col="pct_chg", period=period)
        df2 = df2.drop(columns=["pct_chg"])
        ic_df, summary_df = calculate_icir(df2, ["factor_a"], return_col="pct_chg", period=period)
        self.assertIn("factor_a_ic", ic_df.columns)
        self.assertIn("factor", summary_df.columns)

    def test_factor_layer_analysis(self):
        df2 = process_future_return(self.df, return_col="pct_chg", period=1)
        layer_results = factor_layer_analysis(df2, ["factor_a"], return_col="pct_chg", period=1, layers=5)
        self.assertIn("factor_a", layer_results)
        layer_df = layer_results["factor_a"]
        self.assertIn("layer", layer_df.columns)
        self.assertTrue(layer_df["layer"].between(1, 5).all())

    def test_period_sampling_alignment_between_ic_and_layer(self):
        period = 3
        df2 = process_future_return(self.df, return_col="pct_chg", period=period)
        ic_df, _ = calculate_icir(df2, ["factor_a"], return_col="pct_chg", period=period)
        layer_results = factor_layer_analysis(df2, ["factor_a"], return_col="pct_chg", period=period, layers=5)
        self.assertIn("factor_a", layer_results)

        ic_dates = set(pd.to_datetime(ic_df["trade_date"]).dropna().unique())
        layer_dates = set(pd.to_datetime(layer_results["factor_a"]["trade_date"]).dropna().unique())
        self.assertSetEqual(ic_dates, layer_dates)

    def test_filter_effective_factors(self):
        summary_df = pd.DataFrame(
            [
                {
                    "factor": "factor_a",
                    "ic_mean": 0.03,
                    "ic_std": 0.1,
                    "ir": 0.4,
                    "positive_ic_ratio": 0.6,
                    "t_stat": 2.0,
                    "p_value": 0.03,
                }
            ]
        )
        long_short_metrics = {
            "factor_a": {
                "total_return": 0.1,
                "annualized_return": 0.12,
                "volatility": 0.2,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.1,
                "fitness_ratio": 1.1,
            }
        }
        layer_results = {"factor_a": pd.DataFrame({"layer": [1, 2], "r": [0.01, 0.02]})}
        out = filter_effective_factors(summary_df, long_short_metrics, layer_results, apply_filtering=False)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["factor"], "factor_a")

    def test_calculate_factor_weights(self):
        summary_df = pd.DataFrame(
            [
                {"factor": "factor_a", "ir": 0.3, "sharpe_ratio": 1.0},
                {"factor": "factor_b", "ir": -0.2, "sharpe_ratio": -0.5},
            ]
        )
        w = calculate_factor_weights(summary_df, weighting_method="icir")
        self.assertIn("factor_a", w)
        self.assertIn("factor_b", w)

    def test_diagnostics_functions(self):
        df2 = process_future_return(self.df, return_col="pct_chg", period=1)
        dfp = process_factor_data(
            df2[["trade_date", "znz_code", "pct_chg", "pct_chg_1d", "circ_mv", "factor_a", "factor_b"]].copy(),
            ["factor_a", "factor_b"],
            market_value_column="circ_mv",
            is_timeseries=True,
        )
        ic_df, summary_df = calculate_icir(dfp, ["factor_a", "factor_b"], return_col="pct_chg", period=1)
        layer_results = factor_layer_analysis(dfp, ["factor_a", "factor_b"], return_col="pct_chg", period=1, layers=5)

        cov = calculate_factor_coverage(dfp, ["factor_a", "factor_b"])
        self.assertIn("overall", cov)
        self.assertIn("by_date", cov)
        self.assertEqual(set(cov["overall"]["factor"]), {"factor_a", "factor_b"})

        st = calculate_ic_stability(ic_df, ["factor_a", "factor_b"])
        self.assertTrue(set(["factor", "ic_mean", "ir"]).issubset(st.columns))

        yearly = calculate_ic_time_breakdown(ic_df, ["factor_a", "factor_b"], freq="Y")
        monthly = calculate_ic_time_breakdown(ic_df, ["factor_a", "factor_b"], freq="M")
        self.assertIn("period_label", yearly.columns)
        self.assertIn("period_label", monthly.columns)

        mono = calculate_layer_monotonicity(layer_results)
        self.assertIn("summary", mono)
        self.assertIn("daily", mono)
        self.assertIn("monotonicity_mean", mono["summary"].columns)

        long_short_metrics = {
            f: {
                "total_return": 0.1,
                "annualized_return": 0.12,
                "volatility": 0.2,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.1,
                "fitness_ratio": 1.1,
            }
            for f in ["factor_a", "factor_b"]
        }
        turnover_results = {}
        summary_report = build_factor_summary_report(
            factor_cols=["factor_a", "factor_b"],
            coverage_overall_df=cov["overall"],
            ic_summary_df=summary_df,
            monotonicity_summary_df=mono["summary"],
            long_short_metrics=long_short_metrics,
            turnover_results=turnover_results,
            effective_factors_df=pd.DataFrame({"factor": ["factor_a"]}),
            apply_filtering=True,
        )
        self.assertIn("passed_filter", summary_report.columns)

    def test_holding_period_robustness(self):
        res = analyze_holding_period_robustness(
            self.df,
            factor_cols=["factor_a"],
            periods=[1, 5],
            return_col="pct_chg",
            layers=5,
            market_value_column="circ_mv",
            is_timeseries=True,
        )
        self.assertIn("comparison", res)
        self.assertIn("details", res)
        self.assertTrue(set(["period", "factor", "ir"]).issubset(res["comparison"].columns))

    def test_direction_modes_and_corr_abs_option(self):
        df2 = process_future_return(self.df, return_col="pct_chg", period=1)
        layer_results = factor_layer_analysis(df2, ["factor_a"], return_col="pct_chg", period=1, layers=5)
        ic_df, summary_df = calculate_icir(df2, ["factor_a"], return_col="pct_chg", period=1)
        _ = ic_df

        ls_auto, _ = calculate_long_short_metrics(layer_results, period=1, direction_mode="auto_by_final_return")
        ls_top, _ = calculate_long_short_metrics(layer_results, period=1, direction_mode="top_minus_bottom")
        ls_ic, _ = calculate_long_short_metrics(
            layer_results, period=1, direction_mode="by_ic_sign", ic_summary_df=summary_df
        )
        self.assertIn("factor_a", ls_auto)
        self.assertIn("factor_a", ls_top)
        self.assertIn("factor_a", ls_ic)

        corr = pd.DataFrame(
            [[1.0, -0.9, 0.1], [-0.9, 1.0, 0.2], [0.1, 0.2, 1.0]],
            index=["f1", "f2", "f3"],
            columns=["f1", "f2", "f3"],
        )
        ef = pd.DataFrame({"factor": ["f1", "f2", "f3"], "sharpe_ratio": [2.0, 1.9, 1.0]})
        out_legacy = filter_factors_by_correlation_advanced(corr, ef, threshold=0.8, use_absolute_corr=False)
        out_abs = filter_factors_by_correlation_advanced(corr, ef, threshold=0.8, use_absolute_corr=True)
        self.assertGreaterEqual(len(out_legacy), len(out_abs))

    def test_ewma_factor_returns_factorwise_independent(self):
        df2 = self.df.copy()
        df2["factor_b"] = np.nan
        out = calculate_factor_returns(
            df=df2,
            factor_cols=["factor_a", "factor_b"],
            return_col="pct_chg",
            period=1,
            independent_by_factor=True,
        )
        self.assertIn("factor", out.columns)
        self.assertGreater((out["factor"] == "factor_a").sum(), 0)

    def test_predict_stock_returns_with_partial_ewma_factor_set(self):
        dates = sorted(self.df["trade_date"].unique())[:6]
        df_small = self.df[self.df["trade_date"].isin(dates)][["znz_code", "trade_date", "factor_a", "factor_b"]].copy()
        ewma = pd.DataFrame(
            {
                "trade_date": list(dates),
                "factor": ["factor_a"] * len(dates),
                "ewma_return": [0.01] * len(dates),
            }
        )
        pred = predict_stock_returns(df_small, ["factor_a", "factor_b"], ewma, period=1)
        self.assertIn("predicted_return", pred.columns)
        self.assertGreater(len(pred), 0)

    def test_factor_correlation_long_short_pairwise_dropna(self):
        long_short = {
            "f1": pd.Series([0.01, np.nan, 0.02, 0.03]),
            "f2": pd.Series([0.02, 0.01, np.nan, 0.01]),
            "f3": pd.Series([0.03, 0.02, 0.01, 0.00]),
        }
        corr_pairwise = calculate_factor_correlation(
            self.df,
            factor_cols=["f1", "f2", "f3"],
            method="long_short_returns",
            long_short_returns_dict=long_short,
            pairwise_complete_obs=True,
        )
        corr_listwise = calculate_factor_correlation(
            self.df,
            factor_cols=["f1", "f2", "f3"],
            method="long_short_returns",
            long_short_returns_dict=long_short,
            pairwise_complete_obs=False,
        )
        self.assertTrue(set(["f1", "f2", "f3"]).issubset(corr_pairwise.columns))
        self.assertGreaterEqual(corr_pairwise.notna().sum().sum(), corr_listwise.notna().sum().sum())

    def test_turnover_period_step_on_sampled_dates(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"])
        layer_df = pd.DataFrame(
            {
                "trade_date": [dates[0], dates[0], dates[1], dates[1], dates[2], dates[2]],
                "znz_code": ["A", "B", "A", "C", "C", "D"],
                "layer": [1, 5, 1, 5, 1, 5],
                "factor_a": [0, 1, 0, 1, 0, 1],
                "pct_chg_5d": [0.01, -0.01, 0.02, -0.02, 0.03, -0.03],
            }
        )
        res = calculate_turnover_rate({"factor_a": layer_df}, period=5)
        self.assertIn("factor_a", res)
        self.assertEqual(len(res["factor_a"]), 2)

    def test_risk_metrics_dropna(self):
        s = pd.Series([0.01, np.nan, 0.02, -0.01])
        m = calculate_risk_metrics(s, period=1, sharpe_penalty_divisor=2.0)
        self.assertFalse(pd.isna(m["total_return"]))


if __name__ == "__main__":
    unittest.main()
