from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_mining.engine import ExpressionEngine
from alpha_mining.panel_store import PanelStore


def test_scale_operator_accepts_total_scale_argument() -> None:
    dt = pd.Timestamp("2024-02-01")
    df = pd.DataFrame(
        [
            {"date": dt, "code": "A", "x": 2.0},
            {"date": dt, "code": "B", "x": -1.0},
            {"date": dt, "code": "C", "x": 1.0},
        ]
    )
    out = ExpressionEngine(PanelStore.from_long_frame(df)).eval("scale(x, 2.0)")

    assert abs(float(out.loc[dt].abs().sum()) - 2.0) <= 1.0e-12
    assert out.loc[dt, "A"] == 1.0
    assert out.loc[dt, "B"] == -0.5
    assert out.loc[dt, "C"] == 0.5


def test_scale_operator_accepts_long_short_scale_arguments() -> None:
    dt = pd.Timestamp("2024-02-02")
    df = pd.DataFrame(
        [
            {"date": dt, "code": "A", "x": 2.0},
            {"date": dt, "code": "B", "x": -3.0},
            {"date": dt, "code": "C", "x": 1.0},
            {"date": dt, "code": "D", "x": -1.0},
        ]
    )
    out = ExpressionEngine(PanelStore.from_long_frame(df)).eval("scale(x, 1.0, 0.6, 0.4)")
    row = out.loc[dt]

    assert abs(float(row[row > 0].sum()) - 0.6) <= 1.0e-12
    assert abs(float(row[row < 0].abs().sum()) - 0.4) <= 1.0e-12
    assert np.isfinite(row.to_numpy(dtype=float)).all()
