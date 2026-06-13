from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import pandas as pd

from alpha_mining.datasource.config import LakePathSettings
from alpha_mining.datasource.duckdb_catalog import (
    _safe_view_row_count,
    build_duckdb_catalog,
)


class TestDuckdbCatalogEmptyParquetGuard(unittest.TestCase):
    def test_build_catalog_skips_zero_column_parquet(self) -> None:
        try:
            import duckdb  # type: ignore
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"duckdb not available: {exc}")
            return

        base_dir = Path("data") / f"_duckdb_empty_guard_{uuid.uuid4().hex}"
        settings = LakePathSettings(
            lake_root=str((base_dir / "lake").as_posix()),
            duckdb_path=str((base_dir / "duckdb" / "market.duckdb").as_posix()),
        )
        try:
            bad_file = settings.vendor_raw_path / "namechange" / "snapshot_date=2026-04-22" / "part-000.parquet"
            bad_file.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame().to_parquet(bad_file, index=False)

            out = build_duckdb_catalog(paths=settings)
            self.assertEqual(int(out.get("project_rows", -1)), 0)

            conn = duckdb.connect(str(settings.duckdb_path_obj), read_only=True)
            try:
                rows = int(conn.execute("SELECT COUNT(*) FROM raw_namechange").fetchone()[0])
            finally:
                conn.close()
            self.assertEqual(rows, 0)
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_safe_view_row_count_fallback(self) -> None:
        class _FakeConn:
            def execute(self, _sql: str):  # type: ignore[no-untyped-def]
                raise RuntimeError("mock oom")

        row_count, warning = _safe_view_row_count(conn=_FakeConn(), view_name="v_project_panel_cn_a")
        self.assertEqual(int(row_count), -1)
        self.assertIn("failed to compute row count", str(warning))


if __name__ == "__main__":
    unittest.main()
