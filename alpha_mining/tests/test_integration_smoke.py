"""Minimal integration smoke test: data → analysis → artifact schema → dashboard-readable.

Generates synthetic data in-memory (no DuckDB/external dependency).
Exercises the full analysis pipeline and verifies output schema compatibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_synthetic_panel(
    n_dates: int = 30,
    n_codes: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic factor_research-ready DataFrame."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    codes = [f"S{i:03d}" for i in range(n_codes)]

    rows = []
    for d in dates:
        for c in codes:
            rows.append(
                {
                    "trade_date": d,
                    "znz_code": c,
                    "pct_chg": rng.normal(0.001, 0.02),
                    "close": rng.uniform(10, 100),
                    "circ_mv": rng.uniform(1e8, 1e10),
                    "industry": rng.choice(["tech", "finance", "consumer"]),
                    "can_buy": 1,
                    "can_sell": 1,
                }
            )
    df = pd.DataFrame(rows)
    # Add a synthetic alpha signal
    df["alpha_test"] = rng.randn(len(df))
    return df


def test_analysis_pipeline_smoke() -> None:
    """Full pipeline: synthetic data → analysis → verify output keys and schema."""
    from alpha_mining.workflow.analysis_cycle import (
        BatchAnalysisConfig,
        run_factor_analysis_batch_light,
    )

    df = _make_synthetic_panel()
    cfg = BatchAnalysisConfig(
        period=1,
        layers=5,
        is_timeseries=False,
        include_robustness=False,
        apply_filtering=False,
        include_phase_metrics=False,
        apply_tradability_constraints=False,
        include_full_ic_lag_analysis=False,
    )

    outputs = run_factor_analysis_batch_light(
        df_raw=df,
        factor_cols=["alpha_test"],
        config=cfg,
    )

    # Verify essential output keys exist
    assert "summary_df" in outputs
    assert "ic_df" in outputs
    assert "layer_results" in outputs
    assert "factor_metrics_df" in outputs
    assert "factor_effectiveness_table" in outputs

    # Verify summary_df schema
    summary = outputs["summary_df"]
    assert isinstance(summary, pd.DataFrame)
    assert "factor" in summary.columns
    assert "ic_mean" in summary.columns
    assert "ir" in summary.columns
    assert "ic_valid_count" in summary.columns
    assert len(summary) >= 1

    # Verify ic_valid_count is populated
    row = summary[summary["factor"] == "alpha_test"].iloc[0]
    assert int(row["ic_valid_count"]) > 0

    # Verify factor_metrics_df schema
    metrics = outputs["factor_metrics_df"]
    assert isinstance(metrics, pd.DataFrame)
    assert "factor" in metrics.columns
    assert "ic_mean" in metrics.columns

    # Verify factor_effectiveness_table schema
    eff = outputs["factor_effectiveness_table"]
    assert isinstance(eff, pd.DataFrame)
    assert "factor" in eff.columns

    # Verify no inf values in numeric outputs
    for key in ["summary_df", "factor_metrics_df"]:
        frame = outputs[key]
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            numeric_cols = frame.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                assert np.isinf(frame[col].dropna()).sum() == 0, f"inf found in {key}.{col}"


def test_analysis_layer_diagnostics_smoke() -> None:
    """Verify layer diagnostics (quantile fallback, insufficient stock dates) are captured."""
    from alpha_mining.workflow.analysis_cycle import (
        BatchAnalysisConfig,
        run_factor_analysis_batch_light,
    )

    df = _make_synthetic_panel()
    cfg = BatchAnalysisConfig(
        period=1,
        layers=5,
        is_timeseries=False,
        include_robustness=False,
        apply_filtering=False,
        include_phase_metrics=False,
        apply_tradability_constraints=False,
        include_full_ic_lag_analysis=False,
    )

    outputs = run_factor_analysis_batch_light(
        df_raw=df,
        factor_cols=["alpha_test"],
        config=cfg,
    )

    metrics = outputs["factor_metrics_df"]
    # Check layer diagnostic columns exist
    diag_cols = [
        "layer_qcut_fallback_count",
        "layer_fewer_groups_count",
        "layer_total_dates",
        "layer_insufficient_stock_dates",
    ]
    for col in diag_cols:
        assert col in metrics.columns, f"missing diagnostic column: {col}"

    row = metrics[metrics["factor"] == "alpha_test"].iloc[0]
    assert int(row["layer_total_dates"]) > 0
    # With 20 codes and 5 layers, no insufficient stock dates expected
    assert int(row["layer_insufficient_stock_dates"]) == 0


def test_layer_results_attrs_diagnostics() -> None:
    """Verify layer_results DataFrame attrs contain diagnostic info."""
    from alpha_mining.workflow.analysis_cycle import (
        BatchAnalysisConfig,
        run_factor_analysis_batch_light,
    )

    df = _make_synthetic_panel()
    cfg = BatchAnalysisConfig(
        period=1,
        layers=5,
        is_timeseries=False,
        include_robustness=False,
        apply_filtering=False,
        include_phase_metrics=False,
        apply_tradability_constraints=False,
        include_full_ic_lag_analysis=False,
    )

    outputs = run_factor_analysis_batch_light(
        df_raw=df,
        factor_cols=["alpha_test"],
        config=cfg,
    )

    lr = outputs["layer_results"]
    assert "alpha_test" in lr
    attrs = lr["alpha_test"].attrs
    assert "qcut_fallback_count" in attrs
    assert "fewer_groups_count" in attrs
    assert "total_dates" in attrs
    assert "insufficient_stock_dates" in attrs
    assert attrs["total_dates"] > 0


def test_binary_op_clean_nonfinite_integration() -> None:
    """Verify _clean_nonfinite works end-to-end through expression engine."""
    from alpha_mining.engine import ExpressionEngine
    from alpha_mining.panel_store import PanelStore

    dates = pd.date_range("2024-01-01", periods=3)
    rows = []
    for d in dates:
        rows.extend(
            [
                {"date": d, "code": "A", "x": 0.0, "y": 1.0, "industry": "g1"},
                {"date": d, "code": "B", "x": 2.0, "y": 0.0, "industry": "g1"},
                {"date": d, "code": "C", "x": -3.0, "y": 1.0, "industry": "g1"},
            ]
        )
    store = PanelStore.from_long_frame(pd.DataFrame(rows), group_fields=["industry"])
    engine = ExpressionEngine(store)

    # Division by zero should produce nan, not inf
    result = engine.eval("x / y")
    assert np.isinf(result.to_numpy(dtype=float, na_value=np.nan)).sum() == 0

    # Power with negative base should produce nan, not inf
    result_pow = engine.eval("x ** (-1)")
    assert np.isinf(result_pow.to_numpy(dtype=float, na_value=np.nan)).sum() == 0
