from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from alpha_mining.config import AlphaMiningConfig, AlphaSimulationConfig
from alpha_mining.mining.pipeline import AlphaMiningPipeline
from alpha_mining.panel_store import PanelStore


class TestPipelineSimulation(unittest.TestCase):
    def test_external_universe_mask(self) -> None:
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
                "code": ["A", "B", "A", "B"],
                "close": [1.0, 2.0, 1.5, 2.5],
                "in_universe": [1, 0, 1, 0],
            }
        )
        store = PanelStore.from_long_frame(df)
        cfg = AlphaMiningConfig(
            simulation=AlphaSimulationConfig(delay=0, decay=0, neutralization="NONE", universe="in_universe")
        )
        pipeline = AlphaMiningPipeline.from_panel_store(store, config=cfg)
        out, failed = pipeline.run_expressions(["rank(close)"])

        self.assertEqual(failed, {})
        b_rows = out[out["code"] == "B"]["alpha_0001"]
        self.assertTrue(b_rows.isna().all())

    def test_run_prepared_expressions_with_profile(self) -> None:
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
                "code": ["A", "B", "A", "B"],
                "close": [1.0, 2.0, 1.5, 2.5],
            }
        )
        store = PanelStore.from_long_frame(df)
        cfg = AlphaMiningConfig(simulation=AlphaSimulationConfig(delay=0, decay=0, neutralization="NONE"))
        pipeline = AlphaMiningPipeline.from_panel_store(store, config=cfg)

        alpha_df, expr_timing_df, op_timing_df = pipeline.run_prepared_expressions_with_profile(["rank(close)"])

        self.assertIn("alpha_0001", alpha_df.columns)
        self.assertEqual(len(expr_timing_df), 1)
        self.assertIn("operator", op_timing_df.columns)
        self.assertTrue((op_timing_df["operator"] == "rank").any())

    def test_output_dtype_and_drop_all_nan_rows(self) -> None:
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
                "code": ["A", "B", "A", "B"],
                "close": [1.0, 2.0, 1.5, 2.5],
                "in_universe": [0, 0, 1, 1],
            }
        )
        store = PanelStore.from_long_frame(df)
        cfg = AlphaMiningConfig(
            simulation=AlphaSimulationConfig(delay=0, decay=0, neutralization="NONE", universe="in_universe")
        )
        pipeline = AlphaMiningPipeline.from_panel_store(store, config=cfg)

        out = pipeline.run_prepared_expressions(
            ["rank(close)"],
            output_dtype="float32",
            drop_all_nan_rows=True,
        )

        self.assertEqual(len(out), 2)
        self.assertEqual(str(out["alpha_0001"].dtype), "float32")

    def test_sector_neutralization_uses_panel_store_group_field(self) -> None:
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"] * 4,
                "code": ["A", "B", "C", "D"],
                "close": [1.0, 3.0, 10.0, 14.0],
                "sector": ["S1", "S1", "S2", "S2"],
            }
        )
        store = PanelStore.from_long_frame(df, group_fields=["sector"])
        cfg = AlphaMiningConfig(simulation=AlphaSimulationConfig(delay=0, decay=0, neutralization="sector"))
        pipeline = AlphaMiningPipeline.from_panel_store(store, config=cfg)

        out = pipeline.run_prepared_expressions(["close"])

        values = out.set_index("code")["alpha_0001"]
        self.assertTrue(np.isclose(float(values.loc["A"] + values.loc["B"]), 0.0))
        self.assertTrue(np.isclose(float(values.loc["C"] + values.loc["D"]), 0.0))

    def test_missing_group_for_neutralization_fails_fast(self) -> None:
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-01"],
                "code": ["A", "B"],
                "close": [1.0, 2.0],
            }
        )
        store = PanelStore.from_long_frame(df)
        cfg = AlphaMiningConfig(simulation=AlphaSimulationConfig(delay=0, decay=0, neutralization="SECTOR"))
        pipeline = AlphaMiningPipeline.from_panel_store(store, config=cfg)

        with self.assertRaisesRegex(ValueError, "requires group field 'sector'"):
            pipeline.run_prepared_expressions(["close"])

    def test_universe_mask_is_applied_before_market_neutralization(self) -> None:
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-01", "2024-01-01"],
                "code": ["A", "B", "C"],
                "close": [1.0, 3.0, 100.0],
                "in_universe": [1, 1, 0],
            }
        )
        store = PanelStore.from_long_frame(df)
        cfg = AlphaMiningConfig(
            simulation=AlphaSimulationConfig(
                delay=0,
                decay=0,
                neutralization="MARKET",
                universe="in_universe",
            )
        )
        pipeline = AlphaMiningPipeline.from_panel_store(store, config=cfg)

        out = pipeline.run_prepared_expressions(["close"])
        values = out.set_index("code")["alpha_0001"]

        self.assertTrue(np.isnan(values.loc["C"]))
        self.assertTrue(np.isclose(float(values.loc["A"] + values.loc["B"]), 0.0))


if __name__ == "__main__":
    unittest.main()
