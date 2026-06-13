from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import pandas as pd

from alpha_mining.datasource.config import LakePathSettings
from alpha_mining.datasource.curation import curate_security_namechange
from alpha_mining.datasource.duckdb_catalog import build_duckdb_catalog
from alpha_mining.datasource.loader import load_panel_from_duckdb


class TestDatasourceUniverseRules(unittest.TestCase):
    def test_curate_security_namechange_st_prefix(self) -> None:
        raw = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "name": "ST AAA", "start_date": "20240101"},
                {"ts_code": "000002.SZ", "name": "*ST BBB", "start_date": "20240101"},
                {"ts_code": "000003.SZ", "name": "SST CCC", "start_date": "20240101"},
                {"ts_code": "000004.SZ", "name": "S*ST DDD", "start_date": "20240101"},
                {"ts_code": "000005.SZ", "name": "PT EEE", "start_date": "20240101"},
                {"ts_code": "000006.SZ", "name": "NORMAL F", "start_date": "20240101"},
            ]
        )
        out = curate_security_namechange(raw)
        flag_map = {str(r["code"]): int(r["is_st"]) for _, r in out.iterrows()}
        self.assertEqual(flag_map["000001.SZ"], 1)
        self.assertEqual(flag_map["000002.SZ"], 1)
        self.assertEqual(flag_map["000003.SZ"], 1)
        self.assertEqual(flag_map["000004.SZ"], 1)
        self.assertEqual(flag_map["000005.SZ"], 1)
        self.assertEqual(flag_map["000006.SZ"], 0)

    def test_duckdb_panel_universe_and_tradable(self) -> None:
        base_dir = Path("data") / f"_duckdb_universe_test_{uuid.uuid4().hex}"
        lake_root = base_dir / "lake"
        duckdb_path = base_dir / "duckdb" / "market.duckdb"
        settings = LakePathSettings(
            lake_root=str(lake_root.as_posix()),
            duckdb_path=str(duckdb_path.as_posix()),
        )
        try:
            self._prepare_minimal_lake(settings)
            build_duckdb_catalog(
                paths=settings,
                source_view="v_project_panel_cn_a",
                field_catalog_version="v_test",
                universe_min_days_since_listed=1,
                universe_exclude_st=True,
                include_bj=False,
                tradable_require_close=True,
                tradable_require_positive_volume=True,
                tradable_require_positive_amount=True,
            )

            all_df = load_panel_from_duckdb(
                duckdb_path=str(duckdb_path.as_posix()),
                source_view="v_project_panel_cn_a",
                required_fields=[
                    "close",
                    "up_limit",
                    "down_limit",
                    "is_suspended",
                    "is_st",
                    "days_since_listed",
                    "tradable",
                    "universe",
                ],
                start_date="2026-04-01",
                end_date="2026-04-02",
                run_filters={"universe_only": False, "include_bj": True},
            )
            self.assertFalse(all_df.empty)

            st_rows = all_df[all_df["code"] == "000001.SZ"]
            self.assertTrue((st_rows["is_st"] == 1).all())
            self.assertTrue((st_rows["universe"] == 0).all())

            bj_rows = all_df[all_df["code"] == "430001.BJ"]
            self.assertFalse(bj_rows.empty)
            self.assertTrue((bj_rows["universe"] == 0).all())

            sh_rows = all_df[all_df["code"] == "600001.SH"]
            self.assertFalse(sh_rows.empty)
            self.assertTrue((sh_rows["tradable"] == 1).all())
            self.assertTrue((sh_rows["universe"] == 1).all())
            self.assertTrue((pd.to_numeric(sh_rows["days_since_listed"], errors="coerce") >= 1).all())

            suspended = all_df[(all_df["code"] == "000001.SZ") & (all_df["date"] == pd.Timestamp("2026-04-02"))]
            self.assertEqual(int(suspended.iloc[0]["is_suspended"]), 1)
            self.assertEqual(int(suspended.iloc[0]["tradable"]), 0)

            filtered_df = load_panel_from_duckdb(
                duckdb_path=str(duckdb_path.as_posix()),
                source_view="v_project_panel_cn_a",
                required_fields=["close", "universe", "tradable"],
                start_date="2026-04-01",
                end_date="2026-04-02",
                run_filters={"universe_only": True, "include_bj": False},
            )
            self.assertEqual(set(filtered_df["code"].astype(str).tolist()), {"600001.SH"})
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def _prepare_minimal_lake(self, settings: LakePathSettings) -> None:
        vendor_root = settings.lake_root_path / settings.vendor_raw_subdir
        curated_root = settings.curated_path
        settings.duckdb_path_obj.parent.mkdir(parents=True, exist_ok=True)

        dim_trade_calendar = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-03-31", "2026-04-01", "2026-04-02"]),
                "exchange": ["SSE", "SSE", "SSE"],
                "is_open": [1, 1, 1],
                "pretrade_date": pd.to_datetime(["2026-03-30", "2026-03-31", "2026-04-01"]),
                "cal_date": ["20260331", "20260401", "20260402"],
            }
        )
        dim_security_master = pd.DataFrame(
            {
                "code": ["000001.SZ", "600001.SH", "430001.BJ"],
                "name": ["ST AAA", "NORMAL SH", "NORMAL BJ"],
                "industry": ["I1", "I1", "I2"],
                "market": ["SZ", "SH", "BJ"],
                "list_status": ["L", "L", "L"],
                "list_date": pd.to_datetime(["2026-03-31", "2026-03-31", "2026-03-31"]),
                "delist_date": [pd.NaT, pd.NaT, pd.NaT],
                "snapshot_date": pd.to_datetime(["2026-04-02", "2026-04-02", "2026-04-02"]),
            }
        )
        dim_namechange = pd.DataFrame(
            {
                "code": ["000001.SZ"],
                "name": ["ST AAA"],
                "normalized_name": ["ST AAA"],
                "start_date": pd.to_datetime(["2026-03-15"]),
                "end_date": [pd.NaT],
                "ann_date": pd.to_datetime(["2026-03-15"]),
                "change_reason": [""],
                "is_st": [1],
            }
        )

        rows = []
        for d in ["2026-04-01", "2026-04-02"]:
            rows.extend(
                [
                    {"date": d, "code": "000001.SZ"},
                    {"date": d, "code": "600001.SH"},
                    {"date": d, "code": "430001.BJ"},
                ]
            )
        fact_daily = pd.DataFrame(rows)
        fact_daily["date"] = pd.to_datetime(fact_daily["date"])
        fact_daily["open"] = 10.0
        fact_daily["high"] = 11.0
        fact_daily["low"] = 9.0
        fact_daily["close"] = 10.5
        fact_daily["ret_1d"] = 0.01
        fact_daily["pct_chg"] = 0.01
        fact_daily["volume"] = 100000.0
        fact_daily["amount"] = 1000000.0
        fact_daily["adj_factor"] = 1.0
        fact_daily["bfq_open"] = fact_daily["open"]
        fact_daily["bfq_high"] = fact_daily["high"]
        fact_daily["bfq_low"] = fact_daily["low"]
        fact_daily["bfq_close"] = fact_daily["close"]
        fact_daily["qfq_open"] = fact_daily["open"]
        fact_daily["qfq_high"] = fact_daily["high"]
        fact_daily["qfq_low"] = fact_daily["low"]
        fact_daily["qfq_close"] = fact_daily["close"]
        fact_daily["hfq_open"] = fact_daily["open"]
        fact_daily["hfq_high"] = fact_daily["high"]
        fact_daily["hfq_low"] = fact_daily["low"]
        fact_daily["hfq_close"] = fact_daily["close"]
        fact_daily["price_adjust_mode"] = "qfq"

        fact_daily_basic = fact_daily[["date", "code"]].copy()
        fact_daily_basic["turnover_rate"] = 1.0
        fact_daily_basic["turnover_rate_f"] = 1.0
        fact_daily_basic["volume_ratio"] = 1.0
        fact_daily_basic["pe"] = 10.0
        fact_daily_basic["pe_ttm"] = 10.0
        fact_daily_basic["pb"] = 1.5
        fact_daily_basic["ps"] = 1.2
        fact_daily_basic["ps_ttm"] = 1.2
        fact_daily_basic["dv_ratio"] = 0.0
        fact_daily_basic["dv_ttm"] = 0.0
        fact_daily_basic["total_mv"] = 1e9
        fact_daily_basic["circ_mv"] = 8e8
        fact_daily_basic["total_mv_raw_wan"] = 1e5
        fact_daily_basic["circ_mv_raw_wan"] = 8e4

        fact_stk_limit = fact_daily[["date", "code"]].copy()
        fact_stk_limit["pre_close"] = 10.0
        fact_stk_limit["up_limit"] = 11.0
        fact_stk_limit["down_limit"] = 9.0

        fact_suspend_d = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-04-02"]),
                "code": ["000001.SZ"],
                "suspend_timing": [""],
                "suspend_type": [""],
                "is_suspended": [1],
            }
        )

        self._write_parquet(
            dim_trade_calendar,
            curated_root / "dims/trade_calendar/snapshot_date=2026-04-02/part-000.parquet",
        )
        self._write_parquet(
            dim_security_master,
            curated_root / "dims/security_master/snapshot_date=2026-04-02/part-000.parquet",
        )
        self._write_parquet(
            dim_namechange,
            curated_root / "dims/security_namechange/snapshot_date=2026-04-02/part-000.parquet",
        )
        self._write_parquet(
            fact_daily,
            curated_root / "facts/market_daily/year=2026/month=04/part-000.parquet",
        )
        self._write_parquet(
            fact_daily_basic,
            curated_root / "facts/market_daily_basic/year=2026/month=04/part-000.parquet",
        )
        self._write_parquet(
            fact_stk_limit,
            curated_root / "facts/market_stk_limit/year=2026/month=04/part-000.parquet",
        )
        self._write_parquet(
            fact_suspend_d,
            curated_root / "facts/market_suspend_d/year=2026/month=04/part-000.parquet",
        )

        for table in [
            "trade_cal",
            "stock_basic",
            "index_classify",
            "index_member_all",
            "daily",
            "daily_basic",
            "adj_factor",
            "stk_limit",
            "suspend_d",
            "namechange",
        ]:
            (vendor_root / table).mkdir(parents=True, exist_ok=True)

    def _write_parquet(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)


if __name__ == "__main__":
    unittest.main()
