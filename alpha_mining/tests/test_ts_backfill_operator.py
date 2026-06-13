from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_mining.engine import ExpressionEngine
from alpha_mining.panel_store import PanelStore


def test_ts_backfill_uses_past_values_only_with_limit() -> None:
    dates = pd.date_range("2024-01-01", periods=5)
    rows = []
    values_a = [1.0, np.nan, np.nan, 4.0, np.nan]
    values_b = [np.nan, 2.0, np.nan, np.nan, np.nan]
    for dt, a, b in zip(dates, values_a, values_b):
        rows.append({"date": dt, "code": "A", "x": a})
        rows.append({"date": dt, "code": "B", "x": b})

    engine = ExpressionEngine(PanelStore.from_long_frame(pd.DataFrame(rows)))
    out = engine.eval("ts_backfill(x, 1)")

    assert out.loc[pd.Timestamp("2024-01-01"), "A"] == 1.0
    assert out.loc[pd.Timestamp("2024-01-02"), "A"] == 1.0
    assert np.isnan(out.loc[pd.Timestamp("2024-01-03"), "A"])
    assert out.loc[pd.Timestamp("2024-01-04"), "A"] == 4.0
    assert out.loc[pd.Timestamp("2024-01-05"), "A"] == 4.0

    assert np.isnan(out.loc[pd.Timestamp("2024-01-01"), "B"])
    assert out.loc[pd.Timestamp("2024-01-02"), "B"] == 2.0
    assert out.loc[pd.Timestamp("2024-01-03"), "B"] == 2.0
    assert np.isnan(out.loc[pd.Timestamp("2024-01-04"), "B"])
