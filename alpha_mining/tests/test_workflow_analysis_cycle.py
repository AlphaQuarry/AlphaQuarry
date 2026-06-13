from __future__ import annotations

import unittest
import logging

import numpy as np
import pandas as pd

from alpha_mining.workflow import (
    BatchAnalysisConfig,
    run_factor_analysis_batch,
    run_factor_analysis_batch_light,
)


class TestWorkflowAnalysisCycle(unittest.TestCase):
    def test_run_factor_analysis_batch(self) -> None:
        rng = np.random.RandomState(42)
        dates = pd.date_range("2024-01-01", periods=40, freq="D")
        codes = ["A", "B", "C", "D"]
        rows = []
        for d in dates:
            for c in codes:
                rows.append(
                    {
                        "trade_date": d,
                        "znz_code": c,
                        "pct_chg": float(rng.normal(0.001, 0.02)),
                        "circ_mv": float(rng.uniform(5e8, 3e9)),
                    }
                )
        df = pd.DataFrame(rows).sort_values(["znz_code", "trade_date"]).reset_index(drop=True)
        df["alpha_0001"] = df.groupby("znz_code")["pct_chg"].transform(lambda s: s.rolling(5, min_periods=1).mean())
        df["alpha_0002"] = df.groupby("trade_date")["pct_chg"].rank(pct=True) - 0.5

        outputs = run_factor_analysis_batch(
            df_raw=df,
            factor_cols=["alpha_0001", "alpha_0002"],
            config=BatchAnalysisConfig(
                period=1,
                layers=5,
                include_robustness=False,
                is_timeseries=True,
                include_full_ic_lag_analysis=False,
            ),
        )

        self.assertIn("summary_df", outputs)
        self.assertIn("factor_metrics_df", outputs)
        self.assertIn("phase_metrics_df", outputs)
        self.assertIn("analysis_distribution_histogram_df", outputs)
        self.assertIn("analysis_ic_decay_df", outputs)
        self.assertIn("analysis_factor_coverage_by_date_df", outputs)
        summary_df = outputs["summary_df"]
        factor_metrics_df = outputs["factor_metrics_df"]
        phase_metrics_df = outputs["phase_metrics_df"]
        histogram_df = outputs["analysis_distribution_histogram_df"]
        decay_df = outputs["analysis_ic_decay_df"]
        yearly_df = outputs["ic_yearly_df"]
        monthly_df = outputs["ic_monthly_df"]
        coverage_by_date_df = outputs["analysis_factor_coverage_by_date_df"]
        lag_results = outputs["lag_analysis_results"]
        self.assertTrue(isinstance(summary_df, pd.DataFrame))
        self.assertTrue(isinstance(factor_metrics_df, pd.DataFrame))
        self.assertTrue(isinstance(phase_metrics_df, pd.DataFrame))
        self.assertTrue(isinstance(histogram_df, pd.DataFrame))
        self.assertTrue(isinstance(decay_df, pd.DataFrame))
        self.assertTrue(isinstance(yearly_df, pd.DataFrame))
        self.assertTrue(isinstance(monthly_df, pd.DataFrame))
        self.assertTrue(isinstance(coverage_by_date_df, pd.DataFrame))
        self.assertEqual(lag_results, [])
        self.assertGreaterEqual(len(summary_df), 1)
        self.assertIn("alpha_0001", factor_metrics_df["factor"].tolist())
        self.assertIn("train_score_total", factor_metrics_df.columns)
        self.assertIn("feedback_score", factor_metrics_df.columns)
        self.assertIn("ic_decay_spearman", factor_metrics_df.columns)
        self.assertFalse(yearly_df.empty)
        self.assertFalse(monthly_df.empty)
        self.assertFalse(coverage_by_date_df.empty)
        self.assertFalse(decay_df.empty)


if __name__ == "__main__":
    unittest.main()


def test_light_analysis_emits_stage_timing_logs(caplog) -> None:
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    codes = ["A", "B", "C"]
    rows = []
    for date_idx, d in enumerate(dates):
        for code_idx, c in enumerate(codes):
            rows.append(
                {
                    "trade_date": d,
                    "znz_code": c,
                    "pct_chg": float((date_idx + code_idx + 1) / 1000.0),
                    "circ_mv": float(1_000_000 + code_idx),
                    "alpha_a": float(code_idx - date_idx / 100.0),
                }
            )
    df = pd.DataFrame(rows)

    caplog.set_level(logging.INFO, logger="factor_research")
    outputs = run_factor_analysis_batch_light(
        df_raw=df,
        factor_cols=["alpha_a"],
        config=BatchAnalysisConfig(
            period=1,
            layers=3,
            include_robustness=False,
            include_phase_metrics=False,
            include_full_ic_lag_analysis=False,
        ),
    )

    assert "factor_metrics_df" in outputs
    messages = [record.getMessage() for record in caplog.records]
    assert any("[analysis_timing] stage=process_future_return" in message for message in messages)
    assert any("[analysis_timing] stage=factor_layer_analysis" in message for message in messages)
    assert any("[analysis_timing] stage=run_factor_analysis_batch_light_total" in message for message in messages)
