from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_mining.simulation import AlphaSimulationConfig, apply_simulation_settings
from alpha_mining.simulation.scaling import apply_portfolio_scale
from alpha_mining.simulation.truncation import apply_truncation_long_short


def test_wq_portfolio_construction_masks_before_neutralize_scale_and_truncate() -> None:
    dt = pd.Timestamp("2024-03-01")
    alpha = pd.DataFrame([[10.0, 2.0, -3.0, -9.0, 1000.0]], index=[dt], columns=list("ABCDE"))
    universe = pd.DataFrame([[1, 1, 1, 1, 0]], index=[dt], columns=list("ABCDE"))
    cfg = AlphaSimulationConfig(
        delay=0,
        decay=0,
        neutralization="MARKET",
        universe="in_universe",
        portfolio_construction="wq_weight",
        scale_value=1.0,
        truncation=0.30,
        truncation_mode="long_short_capped_rescale",
    )

    out = apply_simulation_settings(alpha, cfg, universe_panel=universe)
    row = out.loc[dt, list("ABCD")]

    assert np.isnan(out.loc[dt, "E"])
    assert abs(float(row.sum())) <= 1.0e-12
    assert abs(float(row.abs().sum()) - 1.0) <= 1.0e-12
    assert float(row.abs().max()) <= 0.30 + 1.0e-12


def test_signal_portfolio_construction_keeps_legacy_clip_behavior() -> None:
    dt = pd.Timestamp("2024-03-02")
    alpha = pd.DataFrame([[2.0, -3.0]], index=[dt], columns=["A", "B"])
    cfg = AlphaSimulationConfig(delay=0, decay=0, neutralization="NONE", truncation=0.5)

    out = apply_simulation_settings(alpha, cfg)

    assert out.loc[dt, "A"] == 0.5
    assert out.loc[dt, "B"] == -0.5


def test_apply_portfolio_scale_and_truncation_helpers_preserve_long_short_gross() -> None:
    dt = pd.Timestamp("2024-03-03")
    weights = pd.DataFrame([[4.0, 2.0, -3.0, -1.0]], index=[dt], columns=list("ABCD"))

    scaled = apply_portfolio_scale(weights, scale_value=1.0)
    truncated = apply_truncation_long_short(scaled, cap=0.30)
    row = truncated.loc[dt]

    assert abs(float(scaled.loc[dt].abs().sum()) - 1.0) <= 1.0e-12
    assert float(row.abs().max()) <= 0.30 + 1.0e-12
    assert abs(float(row[row > 0].sum()) - float(scaled.loc[dt][scaled.loc[dt] > 0].sum())) <= 1.0e-12
    assert abs(float(row[row < 0].abs().sum()) - float(scaled.loc[dt][scaled.loc[dt] < 0].abs().sum())) <= 1.0e-12
