from __future__ import annotations

import numpy as np
import pandas as pd

from factor_research import FactorEffectivenessConfig
from factor_research.screening import compute_effectiveness_score_for_basis
from alpha_mining.workflow.analysis_cycle import (
    BatchAnalysisConfig,
    build_phase_metrics_table,
)


def _score_row() -> dict[str, float]:
    return {
        "ic_mean": 0.04,
        "ir": 0.55,
        "positive_ic_ratio": 0.58,
        "sign_adjusted_monotonicity": 0.20,
        "yearly_sign_consistency": 0.70,
        "monthly_sign_consistency": 0.70,
        "turnover_long_only_mean": 0.30,
        "best_layer_annualized_return_gross": 0.35,
        "best_layer_annualized_return_net": 0.02,
        "best_layer_sharpe_gross": 1.50,
        "best_layer_sharpe_net": 0.20,
        "best_layer_max_drawdown_gross": 0.12,
        "best_layer_max_drawdown_net": 0.45,
        "best_layer_margin_gross": 0.006,
        "best_layer_margin_net": -0.001,
        "margin_long_only_gross": 0.006,
        "margin_long_only_net": -0.001,
        "best_minus_universe_annualized_return_gross": 0.20,
        "best_minus_universe_annualized_return_net": -0.02,
        "best_layer_positive_month_ratio_gross": 0.65,
        "best_layer_positive_month_ratio_net": 0.45,
    }


def test_compute_effectiveness_score_for_basis_uses_net_inputs_when_requested() -> None:
    row = _score_row()

    gross = compute_effectiveness_score_for_basis(row, basis="gross", config=FactorEffectivenessConfig())
    net = compute_effectiveness_score_for_basis(row, basis="net", config=FactorEffectivenessConfig())

    assert gross["score_total"] > net["score_total"]
    assert gross["score_long_only_performance"] > net["score_long_only_performance"]


def test_phase_metrics_select_net_feedback_when_net_pnl_available() -> None:
    factors = ["alpha_a"]
    ic_df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-12-30", "2024-12-31"]),
            "alpha_a_ic": [0.08, 0.09],
        }
    )
    factor_metrics = pd.DataFrame(
        {
            "factor": ["alpha_a"],
            "ic_mean": [0.08],
            "ir": [0.5],
            "score_total": [70.0],
            "score_total_basis": ["net"],
            "turnover_long_only_mean": [0.2],
            "margin_long_only": [0.002],
        }
    )
    pnl = pd.DataFrame(
        {
            "factor": ["alpha_a"] * 4,
            "trade_date": pd.to_datetime(["2024-12-30", "2024-12-31"] * 2),
            "portfolio": ["long_short", "long_short", "long_only", "long_only"],
            "return": [0.02, 0.02, 0.04, 0.04],
            "return_gross": [0.02, 0.02, 0.04, 0.04],
            "return_net": [np.nan, np.nan, -0.01, -0.01],
            "has_net_pnl": [False, False, True, True],
            "turnover": [0.0, 0.2, 0.0, 0.2],
        }
    )

    phase_metrics, _ = build_phase_metrics_table(
        factors=factors,
        ic_df=ic_df,
        factor_metrics_df=factor_metrics,
        portfolio_pnl_df=pnl,
        cfg=BatchAnalysisConfig(),
    )

    row = phase_metrics.iloc[0]
    assert row["train_score_total_basis"] == "net"
    assert np.isfinite(row["train_score_total_gross"])
    assert np.isfinite(row["train_score_total_net"])
    assert row["train_score_total_gross"] > row["train_score_total_net"]
    assert row["feedback_score"] == row["feedback_score_net"]
    assert np.isfinite(row["train_long_short_total_return"])
    assert row["train_long_short_total_return"] == row["train_long_short_total_return_gross"]
    assert "train_long_short_total_return_net" not in row or not np.isfinite(row["train_long_short_total_return_net"])
    assert row["train_long_only_total_return"] == row["train_long_only_total_return_net"]
