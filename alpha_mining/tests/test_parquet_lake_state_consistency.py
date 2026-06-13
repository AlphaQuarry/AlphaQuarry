from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import pandas as pd

from alpha_mining.datasource.config import LakePathSettings
from alpha_mining.datasource.parquet_lake import ParquetLake


class TestParquetLakeStateConsistency(unittest.TestCase):
    def test_ingestion_state_keeps_max_last_trade_date(self) -> None:
        lake, base_dir = self._new_lake()
        try:
            lake.update_ingestion_state(table="daily", last_trade_date="2026-04-21", row_count=100)
            lake.update_ingestion_state(table="daily", last_trade_date="2021-04-20", row_count=1)

            payload = lake.load_ingestion_state()
            daily = payload.get("tables", {}).get("daily", {})
            self.assertEqual(str(daily.get("last_trade_date", "")), "2026-04-21")
            self.assertIn("incoming_last_trade_date_ignored", daily.get("extra", {}))
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_infer_vendor_table_max_trade_date(self) -> None:
        lake, base_dir = self._new_lake()
        try:
            df = pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000001.SZ"],
                    "trade_date": pd.to_datetime(["2026-04-20", "2026-04-21"]),
                    "close": [10.0, 10.2],
                }
            )
            target = lake.vendor_table_root("daily") / "year=2026" / "month=04" / "part-000.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(target, index=False)
            inferred = lake.infer_vendor_table_max_trade_date(table="daily", date_col="trade_date")
            self.assertEqual(inferred, "2026-04-21")
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_ingestion_state_allows_rewind_when_enabled(self) -> None:
        lake, base_dir = self._new_lake()
        try:
            lake.update_ingestion_state(
                table="fina_indicator_vip",
                last_trade_date="2026-04-23",
                row_count=12000,
            )
            lake.update_ingestion_state(
                table="fina_indicator_vip",
                last_trade_date="2026-03-17",
                row_count=85659,
                allow_rewind=True,
            )

            payload = lake.load_ingestion_state()
            state = payload.get("tables", {}).get("fina_indicator_vip", {})
            self.assertEqual(str(state.get("last_trade_date", "")), "2026-03-17")
            self.assertEqual(int(state.get("row_count", 0)), 85659)
            self.assertNotIn("incoming_last_trade_date_ignored", state.get("extra", {}))
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_snapshot_write_skips_empty_schema(self) -> None:
        lake, base_dir = self._new_lake()
        try:
            out = lake.write_vendor_snapshot(
                table="namechange",
                snapshot_date="2026-04-22",
                df=pd.DataFrame(),
            )
            self.assertEqual(str(out.get("status", "")), "skipped_empty_schema")
            target = lake.vendor_table_root("namechange") / "snapshot_date=2026-04-22" / "part-000.parquet"
            self.assertFalse(target.exists())
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def _new_lake(self) -> tuple[ParquetLake, Path]:
        base_dir = Path("data") / f"_lake_state_test_{uuid.uuid4().hex}"
        settings = LakePathSettings(
            lake_root=str((base_dir / "lake").as_posix()),
            duckdb_path=str((base_dir / "duckdb" / "market.duckdb").as_posix()),
        )
        lake = ParquetLake(paths=settings)
        return lake, base_dir


if __name__ == "__main__":
    unittest.main()
