from __future__ import annotations

import unittest

import pandas as pd

from alpha_mining.engine import ExpressionEngine
from alpha_mining.panel_store import PanelStore


class TestVectorExploded(unittest.TestCase):
    def test_vec_avg_with_exploded_columns(self) -> None:
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
                "code": ["A", "B", "A", "B"],
                "analyst_eps__0": [1.0, 2.0, 3.0, 4.0],
                "analyst_eps__1": [3.0, 4.0, 5.0, 6.0],
            }
        )
        store = PanelStore.from_long_frame(df)
        engine = ExpressionEngine(panel_store=store)
        out = engine.eval("vec_avg(analyst_eps)")

        self.assertAlmostEqual(float(out.loc[pd.Timestamp("2024-01-01"), "A"]), 2.0)
        self.assertAlmostEqual(float(out.loc[pd.Timestamp("2024-01-02"), "B"]), 5.0)


if __name__ == "__main__":
    unittest.main()
