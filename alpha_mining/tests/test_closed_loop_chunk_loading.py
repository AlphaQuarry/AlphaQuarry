from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import duckdb  # type: ignore
import pandas as pd

from alpha_mining.config import AlphaMiningConfig, AlphaSimulationConfig
from alpha_mining.datasource.loader import load_panel_from_duckdb
from alpha_mining.workflow.closed_loop import (
    ClosedLoopConfig,
    _build_panel_store,
    _build_sample_prefilter_panel_store,
    _materialize_alpha_batch,
    _materialize_alpha_batch_from_source,
    _sample_prefilter_date_range,
)
from alpha_mining.workflow.universe_store import load_universe_alpha_values
from alpha_mining.mining import AlphaMiningPipeline


class TestClosedLoopChunkLoading(unittest.TestCase):
    def test_stratified_layered_preview_keeps_late_layers(self) -> None:
        from alpha_mining.datasource.loader import _stratified_limit_layered_candidates

        candidates = [{"expression": f"rank(f{i})", "layer": "L1"} for i in range(10)] + [
            {"expression": "if_else(g, alpha, zero_like(alpha))", "layer": "L3"},
            {
                "expression": "group_neutralize(alpha, bucket(rank(circ_mv), '0,1,0.2'))",
                "layer": "L4",
            },
        ]

        limited = _stratified_limit_layered_candidates(candidates, max_n=6, layer_min_counts={"L3": 1, "L4": 1})
        layers = [row["layer"] for row in limited]

        self.assertLessEqual(len(limited), 6)
        self.assertIn("L3", layers)
        self.assertIn("L4", layers)

    def test_chunk_source_materialize_matches_in_memory(self) -> None:
        base_dir = Path("data") / f"_closed_loop_chunk_{uuid.uuid4().hex}"
        duckdb_path = base_dir / "duckdb" / "market.duckdb"
        universe_dir = base_dir / "universe_store"
        try:
            self._prepare_duckdb(duckdb_path)
            cfg = ClosedLoopConfig(
                universe_name="chunk_test",
                universe_base_dir=str(universe_dir.as_posix()),
                date_col="date",
                code_col="code",
                group_fields=(),
                base_frame_cols=("date", "code", "pct_chg", "circ_mv"),
                source_backend="duckdb",
                duckdb_path=str(duckdb_path.as_posix()),
                source_view="v_project_panel_cn_a",
                source_date_range=("2026-04-01", "2026-04-03"),
                run_filters={"universe_only": True},
                enable_source_chunk_loading=True,
                mining_config=AlphaMiningConfig(
                    simulation=AlphaSimulationConfig(
                        delay=0,
                        decay=0,
                        neutralization="NONE",
                        truncation=None,
                        pasteurization=True,
                        universe="universe",
                    )
                ),
            )

            source_result = _materialize_alpha_batch_from_source(
                expressions=["close"],
                alpha_names=["alpha_chunk"],
                config=cfg,
            )
            self.assertIn("paths", source_result)
            self.assertIn("chunk_meta", source_result)
            self.assertGreaterEqual(float(source_result["chunk_meta"]["mem_mb"]), 0.0)
            chunk_df = load_universe_alpha_values(
                alpha_name="alpha_chunk",
                base_dir=cfg.universe_base_dir,
                universe_name=cfg.universe_name,
            )

            full_raw_df = load_panel_from_duckdb(
                duckdb_path=str(duckdb_path.as_posix()),
                source_view="v_project_panel_cn_a",
                required_fields=["close", "universe"],
                start_date="2026-04-01",
                end_date="2026-04-03",
                date_col="date",
                code_col="code",
                base_fields=cfg.base_frame_cols,
                group_fields=cfg.group_fields,
                run_filters=cfg.run_filters,
            )
            panel_store = _build_panel_store(full_raw_df, cfg)
            pipeline = AlphaMiningPipeline.from_panel_store(panel_store, config=cfg.mining_config)
            _materialize_alpha_batch(
                pipeline=pipeline,
                expressions=["close"],
                alpha_names=["alpha_full"],
                config=cfg,
            )
            full_df = load_universe_alpha_values(
                alpha_name="alpha_full",
                base_dir=cfg.universe_base_dir,
                universe_name=cfg.universe_name,
            )

            left = chunk_df[["date", "code", "alpha_chunk"]].copy()
            right = full_df[["date", "code", "alpha_full"]].copy()
            merged = pd.merge(left, right, on=["date", "code"], how="inner")
            self.assertFalse(merged.empty)
            diff = (
                pd.to_numeric(merged["alpha_chunk"], errors="coerce")
                - pd.to_numeric(merged["alpha_full"], errors="coerce")
            ).abs()
            self.assertLessEqual(float(diff.max()), 1e-12)
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_chunk_source_materialize_reports_soft_memory_warning(self) -> None:
        base_dir = Path("data") / f"_closed_loop_chunk_warn_{uuid.uuid4().hex}"
        duckdb_path = base_dir / "duckdb" / "market.duckdb"
        universe_dir = base_dir / "universe_store"
        try:
            self._prepare_duckdb(duckdb_path)
            cfg = ClosedLoopConfig(
                universe_name="chunk_warn_test",
                universe_base_dir=str(universe_dir.as_posix()),
                date_col="date",
                code_col="code",
                group_fields=(),
                base_frame_cols=("date", "code", "pct_chg", "circ_mv"),
                source_backend="duckdb",
                duckdb_path=str(duckdb_path.as_posix()),
                source_view="v_project_panel_cn_a",
                source_date_range=("2026-04-01", "2026-04-03"),
                run_filters={"universe_only": True},
                enable_source_chunk_loading=True,
                source_chunk_mem_warn_mb=0.000001,
                mining_config=AlphaMiningConfig(
                    simulation=AlphaSimulationConfig(
                        delay=0,
                        decay=0,
                        neutralization="NONE",
                        truncation=None,
                        pasteurization=True,
                        universe="universe",
                    )
                ),
            )

            source_result = _materialize_alpha_batch_from_source(
                expressions=["close"],
                alpha_names=["alpha_chunk_warn"],
                config=cfg,
            )

            meta = dict(source_result["chunk_meta"])
            self.assertTrue(bool(meta["mem_warning"]))
            self.assertEqual(float(meta["mem_warn_threshold_mb"]), 0.000001)
            self.assertEqual(meta["alpha_names"], ["alpha_chunk_warn"])
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_chunk_source_prefers_hot_base_view_when_fields_are_available(self) -> None:
        base_dir = Path("data") / f"_closed_loop_hot_{uuid.uuid4().hex}"
        duckdb_path = base_dir / "duckdb" / "market.duckdb"
        universe_dir = base_dir / "universe_store"
        try:
            self._prepare_duckdb(duckdb_path, include_hot_base=True)
            cfg = ClosedLoopConfig(
                universe_name="chunk_hot_test",
                universe_base_dir=str(universe_dir.as_posix()),
                date_col="date",
                code_col="code",
                group_fields=(),
                base_frame_cols=("date", "code", "pct_chg", "circ_mv"),
                source_backend="duckdb",
                duckdb_path=str(duckdb_path.as_posix()),
                source_view="v_project_panel_cn_a",
                source_date_range=("2026-04-01", "2026-04-03"),
                run_filters={"universe_only": True},
                enable_source_chunk_loading=True,
                mining_config=AlphaMiningConfig(
                    simulation=AlphaSimulationConfig(
                        delay=0,
                        decay=0,
                        neutralization="NONE",
                        truncation=None,
                        pasteurization=True,
                        universe="universe",
                    )
                ),
            )

            source_result = _materialize_alpha_batch_from_source(
                expressions=["close"],
                alpha_names=["alpha_hot"],
                config=cfg,
            )

            self.assertEqual(
                source_result["chunk_meta"]["effective_source_view"],
                "v_project_market_daily_base_hot",
            )
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_sample_prefilter_loader_fetches_candidate_fields_for_chunk_loading(
        self,
    ) -> None:
        base_dir = Path("data") / f"_closed_loop_sample_prefilter_{uuid.uuid4().hex}"
        duckdb_path = base_dir / "duckdb" / "market.duckdb"
        universe_dir = base_dir / "universe_store"
        try:
            self._prepare_duckdb(duckdb_path, include_hot_base=True)
            cfg = ClosedLoopConfig(
                universe_name="sample_prefilter_test",
                universe_base_dir=str(universe_dir.as_posix()),
                date_col="date",
                code_col="code",
                group_fields=(),
                base_frame_cols=("date", "code", "pct_chg", "circ_mv"),
                source_backend="duckdb",
                duckdb_path=str(duckdb_path.as_posix()),
                source_view="v_project_panel_cn_a",
                source_date_range=("2026-04-01", "2026-04-03"),
                run_filters={"universe_only": True},
                enable_source_chunk_loading=True,
                enable_sample_prefilter=True,
                mining_config=AlphaMiningConfig(
                    simulation=AlphaSimulationConfig(
                        delay=0,
                        decay=0,
                        neutralization="NONE",
                        truncation=None,
                        pasteurization=True,
                        universe="universe",
                    )
                ),
            )
            candidate_df = pd.DataFrame(
                [
                    {
                        "expression": "rank(close)",
                        "prefilter_status": "pass",
                    }
                ]
            )

            sample_store = _build_sample_prefilter_panel_store(candidate_df=candidate_df, config=cfg)

            self.assertIsNotNone(sample_store)
            assert sample_store is not None
            self.assertIn("close", sample_store.available_scalar_fields())
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_chunk_source_loads_neutralization_group_field_on_demand(self) -> None:
        base_dir = Path("data") / f"_closed_loop_chunk_neutral_{uuid.uuid4().hex}"
        duckdb_path = base_dir / "duckdb" / "market.duckdb"
        universe_dir = base_dir / "universe_store"
        try:
            self._prepare_duckdb(duckdb_path, include_hot_base=True)
            cfg = ClosedLoopConfig(
                universe_name="chunk_neutral_test",
                universe_base_dir=str(universe_dir.as_posix()),
                date_col="date",
                code_col="code",
                group_fields=("industry", "sector"),
                base_frame_cols=("date", "code", "pct_chg", "circ_mv"),
                source_backend="duckdb",
                duckdb_path=str(duckdb_path.as_posix()),
                source_view="v_project_panel_cn_a",
                source_date_range=("2026-04-01", "2026-04-03"),
                run_filters={"universe_only": True},
                enable_source_chunk_loading=True,
                mining_config=AlphaMiningConfig(
                    simulation=AlphaSimulationConfig(
                        delay=0,
                        decay=0,
                        neutralization="SUBINDUSTRY",
                        truncation=None,
                        pasteurization=True,
                        universe="universe",
                    )
                ),
            )

            source_result = _materialize_alpha_batch_from_source(
                expressions=["close"],
                alpha_names=["alpha_subindustry_neutral"],
                config=cfg,
            )

            self.assertIn("subindustry", source_result["chunk_meta"]["required_fields"])
            out = load_universe_alpha_values(
                alpha_name="alpha_subindustry_neutral",
                base_dir=cfg.universe_base_dir,
                universe_name=cfg.universe_name,
            )
            values = pd.to_numeric(out["alpha_subindustry_neutral"], errors="coerce").dropna()
            self.assertTrue((values.abs() <= 1.0e-12).all())
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_sample_prefilter_date_range_uses_recent_lookback_window(self) -> None:
        cfg = ClosedLoopConfig(
            source_date_range=("2021-01-01", "2026-01-01"),
            sample_prefilter_lookback_days=120,
        )

        start_date, end_date = _sample_prefilter_date_range(cfg)

        self.assertEqual(start_date, "2025-09-03")
        self.assertEqual(end_date, "2026-01-01")

    def _prepare_duckdb(self, duckdb_path: Path, include_hot_base: bool = False) -> None:
        duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(str(duckdb_path))
        try:
            df = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-03"]),
                    "code": ["600001.SH", "600001.SH", "600001.SH"],
                    "close": [10.0, 10.5, 11.0],
                    "pct_chg": [0.01, 0.02, 0.03],
                    "circ_mv": [1e9, 1.01e9, 1.02e9],
                    "universe": [1, 1, 1],
                    "industry": ["I1", "I1", "I1"],
                    "sector": ["S1", "S1", "S1"],
                    "subindustry": ["SI1", "SI1", "SI1"],
                }
            )
            conn.register("tmp_panel", df)
            conn.execute("CREATE TABLE t_panel AS SELECT * FROM tmp_panel")
            conn.execute("CREATE VIEW v_project_panel_cn_a AS SELECT * FROM t_panel")
            if include_hot_base:
                conn.execute("CREATE TABLE project_market_daily_base AS SELECT * FROM tmp_panel")
                conn.execute("CREATE VIEW v_project_market_daily_base_hot AS SELECT * FROM project_market_daily_base")
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
