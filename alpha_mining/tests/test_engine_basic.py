from __future__ import annotations

import unittest

import pandas as pd

from alpha_mining.engine import ExpressionEngine
from alpha_mining.panel_store import PanelStore


class TestEngineBasic(unittest.TestCase):
    def test_eval_rank(self) -> None:
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
                "code": ["A", "B", "A", "B"],
                "close": [1.0, 2.0, 1.5, 1.0],
            }
        )
        store = PanelStore.from_long_frame(df)
        engine = ExpressionEngine(store)
        out = engine.eval("rank(close)")
        self.assertEqual(out.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
