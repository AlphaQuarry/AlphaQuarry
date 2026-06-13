from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import pandas as pd

from alpha_mining.workflow.closed_loop import (
    ClosedLoopConfig,
    _save_or_update_input_manifest,
)
from alpha_mining.workflow.universe_store import load_universe_input_manifest


class TestClosedLoopManifestVersion(unittest.TestCase):
    def test_manifest_contains_schema_and_source_meta(self) -> None:
        base_dir = Path("data") / f"_manifest_version_{uuid.uuid4().hex}"
        universe_name = "manifest_test"
        try:
            raw_df = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-04-01", "2026-04-02"]),
                    "code": ["600001.SH", "600001.SH"],
                    "pct_chg": [0.01, 0.02],
                    "circ_mv": [1e9, 1.01e9],
                }
            )
            cfg = ClosedLoopConfig(
                universe_name=universe_name,
                universe_base_dir=str(base_dir.as_posix()),
                source_backend="duckdb",
                duckdb_path="data/duckdb/market.duckdb",
                source_view="v_project_panel_cn_a",
                source_date_range=("2026-04-01", "2026-04-02"),
                field_catalog_version="v_test",
                manifest_schema_version="v2",
                run_filters={"universe_only": True},
            )
            saved = _save_or_update_input_manifest(raw_df=raw_df, config=cfg)
            loaded = load_universe_input_manifest(
                base_dir=str(base_dir.as_posix()),
                universe_name=universe_name,
                manifest_id=str(saved["manifest_id"]),
            )
            payload = loaded.get("payload", {}) if isinstance(loaded, dict) else {}
            self.assertEqual(str(payload.get("manifest_schema_version", "")), "v2")
            self.assertEqual(str(payload.get("source_backend", "")), "duckdb")
            self.assertEqual(str(payload.get("duckdb_path", "")), "data/duckdb/market.duckdb")
            self.assertEqual(str(payload.get("source_view", "")), "v_project_panel_cn_a")
            self.assertTrue(isinstance(payload.get("date_range", {}), dict))
            self.assertEqual(str(payload.get("field_catalog_version", "")), "v_test")
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
