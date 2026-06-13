from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from alpha_mining.engine import ExpressionEngine
from alpha_mining.panel_store import PanelStore


class TestTimeSeriesOps(unittest.TestCase):
    def test_ts_mean(self) -> None:
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"] * 2,
                "code": ["A"] * 3 + ["B"] * 3,
                "close": [1, 2, 3, 4, 5, 6],
            }
        )
        store = PanelStore.from_long_frame(df)
        engine = ExpressionEngine(store)
        out = engine.eval("ts_mean(close, 2)")
        self.assertEqual(out.shape[0], 3)

    def test_ts_rank_matches_reference(self) -> None:
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"] * 2,
                "code": ["A"] * 4 + ["B"] * 4,
                "close": [1.0, 2.0, 2.0, 1.0, 3.0, 1.0, 2.0, 2.0],
            }
        )
        store = PanelStore.from_long_frame(df)
        engine = ExpressionEngine(store)
        out = engine.eval("ts_rank(close, 3)")

        ref = store.get_scalar("close").rolling(3, min_periods=1).apply(lambda s: s.rank(pct=True).iloc[-1], raw=False)
        pd.testing.assert_frame_equal(out, ref)

    def test_ts_rank_with_nan_matches_reference(self) -> None:
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"] * 2,
                "code": ["A"] * 4 + ["B"] * 4,
                "close": [1.0, float("nan"), 2.0, 2.0, 3.0, 1.0, float("nan"), 2.0],
            }
        )
        store = PanelStore.from_long_frame(df)
        engine = ExpressionEngine(store)
        out = engine.eval("ts_rank(close, 3)")

        ref = store.get_scalar("close").rolling(3, min_periods=1).apply(lambda s: s.rank(pct=True).iloc[-1], raw=False)
        pd.testing.assert_frame_equal(out, ref)

    def test_ts_product_matches_reference_when_finite(self) -> None:
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"] * 2,
                "code": ["A"] * 4 + ["B"] * 4,
                "ret": [0.1, -0.2, 0.3, 0.0, -0.1, 0.2, -0.5, 0.4],
            }
        )
        store = PanelStore.from_long_frame(df)
        engine = ExpressionEngine(store)
        out = engine.eval("ts_product(ret, 3)")
        ref = (1 + store.get_scalar("ret")).rolling(3, min_periods=1).apply(np.prod, raw=True) - 1
        pd.testing.assert_frame_equal(out, ref)

    def test_ts_decay_linear_matches_reference_when_finite(self) -> None:
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"] * 2,
                "code": ["A"] * 4 + ["B"] * 4,
                "close": [1.0, 2.0, 4.0, 8.0, 10.0, 9.0, 8.0, 7.0],
            }
        )
        store = PanelStore.from_long_frame(df)
        engine = ExpressionEngine(store)
        out = engine.eval("ts_decay_linear(close, 3)")

        weights = np.arange(1, 4, dtype=float)
        weights = weights / weights.sum()

        def _weighted(v):
            w = weights[-len(v) :]
            return float(np.dot(v, w / w.sum()))

        ref = store.get_scalar("close").rolling(3, min_periods=1).apply(_weighted, raw=True)
        pd.testing.assert_frame_equal(out, ref)


if __name__ == "__main__":
    unittest.main()
