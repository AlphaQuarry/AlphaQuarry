from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_mining.workflow.analysis_cycle import (
    BatchAnalysisConfig,
    build_phase_metrics_table,
    phase_metrics_to_legacy_sample_split_metrics,
)


def _sample_inputs() -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    factors = ["alpha_a"]
    ic_df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                [
                    "2024-12-30",
                    "2024-12-31",
                    "2025-01-02",
                    "2025-01-03",
                    "2026-01-02",
                    "2026-01-05",
                ]
            ),
            "alpha_a_ic": [0.10, 0.20, 0.03, -0.01, 0.05, 0.07],
        }
    )
    factor_metrics = pd.DataFrame(
        {
            "factor": ["alpha_a"],
            "ic_mean": [0.07],
            "ir": [0.4],
            "score_total": [55.0],
            "turnover_long_only_mean": [0.3],
            "margin_long_only": [0.002],
        }
    )
    pnl = pd.DataFrame(
        {
            "factor": ["alpha_a"] * 12,
            "trade_date": pd.to_datetime(
                [
                    "2024-12-30",
                    "2024-12-31",
                    "2025-01-02",
                    "2025-01-03",
                    "2026-01-02",
                    "2026-01-05",
                ]
                * 2
            ),
            "portfolio": ["long_short"] * 6 + ["long_only"] * 6,
            "return": [
                0.01,
                0.02,
                0.005,
                -0.002,
                0.004,
                0.006,
                0.01,
                0.01,
                0.003,
                0.002,
                0.005,
                0.005,
            ],
            "cum_return": np.nan,
            "turnover": [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    return factors, ic_df, factor_metrics, pnl


def test_build_phase_metrics_table_defaults_to_train_feedback_score() -> None:
    factors, ic_df, factor_metrics, pnl = _sample_inputs()

    phase_metrics, phase_meta = build_phase_metrics_table(
        factors=factors,
        ic_df=ic_df,
        factor_metrics_df=factor_metrics,
        portfolio_pnl_df=pnl,
        cfg=BatchAnalysisConfig(),
    )

    row = phase_metrics.iloc[0]
    assert phase_meta["available_phases"] == ["train", "val", "test"]
    assert row["train_obs"] == 2
    assert row["val_obs"] == 2
    assert row["test_obs"] == 2
    assert np.isfinite(row["train_score_total"])
    assert row["feedback_phase"] == "train"
    assert row["feedback_score"] == row["train_score_total"]
    assert "train_long_short_sharpe_ratio" in phase_metrics.columns
    assert "train_turnover_long_short_mean" in phase_metrics.columns
    assert "train_margin_long_short" in phase_metrics.columns
    assert "train_turnover_long_only_mean" in phase_metrics.columns
    assert np.isclose(row["train_turnover_long_short_mean"], 0.1)
    assert np.isclose(row["train_margin_long_short"], 0.15)


def test_build_phase_metrics_uses_benchmark_relative_excess_for_feedback_score() -> None:
    factors, ic_df, factor_metrics, pnl = _sample_inputs()
    benchmark = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03"]),
            "portfolio": ["benchmark"] * 4,
            "return": [0.001, 0.001, 0.002, 0.002],
            "cum_return": [0.001, 0.002001, 0.004005, 0.006013],
        }
    )
    factor_metrics["best_minus_benchmark_annualized_return"] = 0.20

    phase_metrics, _ = build_phase_metrics_table(
        factors=factors,
        ic_df=ic_df,
        factor_metrics_df=factor_metrics,
        portfolio_pnl_df=pnl,
        benchmark_pnl_df=benchmark,
        cfg=BatchAnalysisConfig(),
    )

    row = phase_metrics.iloc[0]
    assert "train_long_short_excess_annualized_return_vs_benchmark" in phase_metrics.columns
    assert "train_best_minus_benchmark_annualized_return" in phase_metrics.columns
    assert pd.notna(row["train_benchmark_annualized_return"])
    assert pd.notna(row["train_long_short_excess_annualized_return_vs_benchmark"])
    assert row["feedback_score"] == row["train_score_total"]


def test_build_phase_metrics_respects_min_obs() -> None:
    factors, ic_df, factor_metrics, pnl = _sample_inputs()

    phase_metrics, phase_meta = build_phase_metrics_table(
        factors=factors,
        ic_df=ic_df,
        factor_metrics_df=factor_metrics,
        portfolio_pnl_df=pnl,
        cfg=BatchAnalysisConfig(phase_metric_min_obs=3),
    )

    row = phase_metrics.iloc[0]
    assert phase_meta["available_phases"] == ["train", "val", "test"]
    assert pd.isna(row["train_ic_mean"])
    assert pd.isna(row["val_ic_mean"])
    assert pd.isna(row["test_ic_mean"])
    assert row["feedback_score"] == 55.0


def test_phase_metrics_to_legacy_sample_split_metrics() -> None:
    frame = pd.DataFrame(
        {
            "factor": ["alpha_a"],
            "train_score_total": [70.0],
            "val_score_total": [35.0],
            "test_score_total": [14.0],
        }
    )

    legacy = phase_metrics_to_legacy_sample_split_metrics(frame)

    row = legacy.iloc[0]
    assert row["train_score"] == 70.0
    assert row["validation_score"] == 35.0
    assert row["oos_score"] == 14.0
    assert row["validation_decay_ratio"] == 0.5
    assert row["oos_decay_ratio"] == 0.2
