from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import pandas as pd

from alpha_mining.workflow.reproduce import reproduce_alpha_by_expression
from alpha_mining.workflow.universe_store import save_universe_input_manifest


class TestWorkflowReproduceSourceMeta(unittest.TestCase):
    def test_reproduce_marks_duckdb_fallback_non_strict(self) -> None:
        try:
            import duckdb  # type: ignore
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"duckdb not available: {exc}")
            return

        base_dir = Path("data") / f"_reproduce_source_meta_{uuid.uuid4().hex}"
        universe = "reproduce_source_meta"
        duckdb_path = base_dir / "duckdb" / "market.duckdb"
        try:
            duckdb_path.parent.mkdir(parents=True, exist_ok=True)
            conn = duckdb.connect(str(duckdb_path))
            panel_df = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-04-01", "2026-04-02"]),
                    "code": ["600001.SH", "600001.SH"],
                    "close": [10.5, 10.7],
                    "pct_chg": [0.01, 0.02],
                    "circ_mv": [8e8, 8.1e8],
                    "universe": [1, 1],
                }
            )
            conn.register("tmp_panel", panel_df)
            conn.execute("CREATE TABLE t_panel AS SELECT * FROM tmp_panel")
            conn.execute("CREATE VIEW v_project_panel_cn_a AS SELECT * FROM t_panel")
            conn.close()

            manifest_payload = {
                "date_col": "date",
                "code_col": "code",
                "group_fields": [],
                "vector_fields": [],
                "snapshot_path": str((base_dir / "missing_snapshot.parquet").as_posix()),
                "source_backend": "duckdb",
                "duckdb_path": str(duckdb_path.as_posix()),
                "source_view": "v_project_panel_cn_a",
                "date_range": {"start": "2026-04-01", "end": "2026-04-02"},
                "base_frame_cols": ["date", "code", "pct_chg", "circ_mv"],
                "run_filters": {"universe_only": True},
            }
            manifest_id = "manifest_test_duckdb_fallback"
            save_universe_input_manifest(
                manifest=manifest_payload,
                base_dir=base_dir,
                universe_name=universe,
                manifest_id=manifest_id,
            )

            out = reproduce_alpha_by_expression(
                expression="close",
                base_dir=base_dir,
                universe_name=universe,
                manifest_id=manifest_id,
            )
            self.assertEqual(str(out.get("reproduce_source_mode")), "duckdb_fallback")
            self.assertFalse(bool(out.get("strict_reproducibility", True)))
            self.assertIn("snapshot unavailable", str(out.get("reproduce_warning", "")))
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
