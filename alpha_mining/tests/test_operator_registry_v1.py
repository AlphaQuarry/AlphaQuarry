from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_mining.engine import ExpressionEngine
from alpha_mining.mining.operator_signatures import (
    SCALAR,
    WINDOW,
    build_default_operator_signature_registry,
)
from alpha_mining.panel_store import PanelStore
from alpha_mining.registry import build_default_registry


def _store() -> PanelStore:
    rows = []
    for date in pd.date_range("2024-01-01", periods=4):
        for code, close, volume in [("A", 1.0, 0.0), ("B", -4.0, 2.0)]:
            rows.append({"date": date, "code": code, "close": close, "volume": volume})
    return PanelStore.from_long_frame(pd.DataFrame(rows))


def test_registry_signatures_cover_searchable_builtin_operators() -> None:
    impl = set(build_default_registry().list_names())
    sigs = build_default_operator_signature_registry()
    allowed_runtime_only = {"bucket", "densify", "group_cartesian_product"}
    missing = sorted(name for name in impl if not sigs.has(name) and name not in allowed_runtime_only)

    assert missing == []
    required_phase2 = {
        "ts_min",
        "ts_max",
        "ts_median",
        "ts_av_diff",
        "ts_covariance",
        "ts_count_nans",
        "group_median",
        "group_scale",
        "quantile",
        "cs_quantile",
        "truncate",
        "left_tail",
        "right_tail",
        "hump",
        "trade_when_hold",
    }
    assert required_phase2 <= impl
    assert all(sigs.has(name) for name in required_phase2)
    assert sigs.match("ts_regression", (SCALAR, SCALAR, WINDOW)) is not None
    assert sigs.match("ts_regression", (SCALAR, SCALAR, WINDOW, "literal")) is not None


def test_aliases_and_numeric_guards_are_available_in_engine() -> None:
    engine = ExpressionEngine(_store())

    div_out = engine.eval("divide(close, volume)")
    assert np.isfinite(div_out.to_numpy(dtype=float)).sum() > 0
    assert np.isinf(div_out.to_numpy(dtype=float)).sum() == 0

    log_out = engine.eval("log(close)")
    assert log_out.loc[pd.Timestamp("2024-01-01"), "B"] != np.inf
    assert np.isnan(log_out.loc[pd.Timestamp("2024-01-01"), "B"])

    sqrt_out = engine.eval("sqrt(close)")
    assert np.isnan(sqrt_out.loc[pd.Timestamp("2024-01-01"), "B"])

    inverse_out = engine.eval("inverse(volume)")
    assert np.isnan(inverse_out.loc[pd.Timestamp("2024-01-01"), "A"])

    signed = engine.eval("signed_power(close, 2)")
    assert signed.loc[pd.Timestamp("2024-01-01"), "A"] == 1.0
    assert signed.loc[pd.Timestamp("2024-01-01"), "B"] == -16.0

    maximum = engine.eval("max(close, volume)")
    minimum = engine.eval("min(close, volume)")
    assert maximum.loc[pd.Timestamp("2024-01-01"), "B"] == 2.0
    assert minimum.loc[pd.Timestamp("2024-01-01"), "B"] == -4.0
