from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import duckdb  # type: ignore
import pandas as pd

from alpha_mining.datasource.config import LakePathSettings
from alpha_mining.datasource.duckdb_catalog import build_duckdb_catalog
from alpha_mining.datasource.loader import (
    get_searchable_fields_from_field_catalog,
    load_panel_from_duckdb,
    plan_required_fields_for_closed_loop,
)
from alpha_mining.workflow.closed_loop import ClosedLoopConfig


class TestDuckDBFinanceAsOf(unittest.TestCase):
    def test_finance_asof_fields_no_lookahead(self) -> None:
        base_dir = Path("data") / f"_duckdb_fin_asof_{uuid.uuid4().hex}"
        settings = LakePathSettings(
            lake_root=str((base_dir / "lake").as_posix()),
            duckdb_path=str((base_dir / "duckdb" / "market.duckdb").as_posix()),
        )
        try:
            self._prepare_minimal_lake(settings)
            build_duckdb_catalog(paths=settings)

            conn = duckdb.connect(str(settings.duckdb_path_obj), read_only=True)
            try:
                out = conn.execute(
                    """
                    SELECT
                        date,
                        code,
                        fin_total_revenue,
                        fin_total_assets,
                        fin_roe,
                        fin_current_ratio,
                        fin_q_roe
                    FROM v_project_panel_cn_a
                    WHERE code = '600001.SH'
                    ORDER BY date
                    """
                ).fetchdf()
                base_cols = (
                    conn.execute(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'v_project_market_daily_base' ORDER BY ordinal_position"
                    )
                    .fetchdf()["column_name"]
                    .astype(str)
                    .tolist()
                )
                fin_cols = (
                    conn.execute(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'v_project_financial_asof_daily' ORDER BY ordinal_position"
                    )
                    .fetchdf()["column_name"]
                    .astype(str)
                    .tolist()
                )
                panel_cols = (
                    conn.execute(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'v_project_panel_cn_a' ORDER BY ordinal_position"
                    )
                    .fetchdf()["column_name"]
                    .astype(str)
                    .tolist()
                )
                hot_count = int(conn.execute("SELECT COUNT(*) FROM project_market_daily_base").fetchone()[0])
                catalog = conn.execute(
                    """
                    SELECT field_name, source_table
                    FROM v_project_field_catalog
                    WHERE field_name IN ('close', 'industry', 'fin_total_revenue')
                    """
                ).fetchdf()
            finally:
                conn.close()

            self.assertEqual(len(out), 3)
            self.assertIn("close", base_cols)
            self.assertIn("industry", base_cols)
            self.assertNotIn("fin_total_revenue", base_cols)
            self.assertIn("fin_total_revenue", fin_cols)
            self.assertEqual(hot_count, 3)
            self.assertLess(panel_cols.index("fin_total_revenue"), panel_cols.index("tradable"))
            catalog_sources = dict(zip(catalog["field_name"], catalog["source_table"]))
            self.assertEqual(catalog_sources["close"], "v_project_market_daily_base")
            self.assertEqual(catalog_sources["industry"], "v_project_market_daily_base")
            self.assertEqual(catalog_sources["fin_total_revenue"], "v_project_financial_asof_daily")
            searchable = get_searchable_fields_from_field_catalog(
                duckdb_path=str(settings.duckdb_path_obj),
                catalog_view="v_project_field_catalog",
            )
            self.assertIn("close", searchable)
            self.assertNotIn("industry", searchable)
            self.assertNotIn("sector", searchable)
            self.assertNotIn("fin_total_revenue", searchable)
            self.assertNotIn("list_status", searchable)
            include_searchable = get_searchable_fields_from_field_catalog(
                duckdb_path=str(settings.duckdb_path_obj),
                catalog_view="v_project_field_catalog",
                include_fields=(
                    "fin_total_revenue",
                    "fin_roe",
                    "list_status",
                    "sector",
                ),
            )
            self.assertIn("fin_total_revenue", include_searchable)
            self.assertIn("fin_roe", include_searchable)
            self.assertNotIn("list_status", include_searchable)
            self.assertNotIn("sector", include_searchable)
            plan = plan_required_fields_for_closed_loop(
                duckdb_path=str(settings.duckdb_path_obj),
                source_view="v_project_panel_cn_a",
                closed_loop_config=ClosedLoopConfig(
                    universe_base_dir=str(base_dir / "alpha_store"),
                    universe_name="ut",
                    group_fields=(),
                    include_fields=("fin_total_revenue",),
                    request_new_alphas=1,
                    max_eval_expressions=8,
                ),
                universe_base_dir=str(base_dir / "alpha_store"),
                universe_name="ut",
            )
            self.assertTrue(plan["selected_expressions"])
            self.assertIn("fin_total_revenue", plan["required_fields"])
            # 2026-04-10 before first ann_date, should be null
            self.assertTrue(pd.isna(out.iloc[0]["fin_total_revenue"]))
            # 2026-04-16 uses 2026-04-15 announcement
            self.assertAlmostEqual(float(out.iloc[1]["fin_total_revenue"]), 100.0, places=8)
            self.assertAlmostEqual(float(out.iloc[1]["fin_total_assets"]), 1000.0, places=8)
            self.assertAlmostEqual(float(out.iloc[1]["fin_roe"]), 0.10, places=8)
            self.assertAlmostEqual(float(out.iloc[1]["fin_current_ratio"]), 1.5, places=8)
            self.assertAlmostEqual(float(out.iloc[1]["fin_q_roe"]), 0.03, places=8)
            # 2026-04-20 uses latest announcement on same day
            self.assertAlmostEqual(float(out.iloc[2]["fin_total_revenue"]), 200.0, places=8)
            self.assertAlmostEqual(float(out.iloc[2]["fin_total_assets"]), 1500.0, places=8)
            self.assertAlmostEqual(float(out.iloc[2]["fin_roe"]), 0.20, places=8)
            loaded = load_panel_from_duckdb(
                duckdb_path=str(settings.duckdb_path_obj),
                source_view="v_project_panel_cn_a",
                required_fields=("close", "fin_current_ratio"),
                start_date="2026-04-10",
                end_date="2026-04-20",
            )
            self.assertEqual(
                str(loaded.attrs.get("duckdb_effective_source_view")),
                "dynamic_project_panel_finance_asof",
            )
            self.assertEqual(
                ["date", "code", "pct_chg", "circ_mv", "close", "fin_current_ratio"],
                list(loaded.columns),
            )
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def _prepare_minimal_lake(self, settings: LakePathSettings) -> None:
        curated = settings.curated_path

        trade_cal = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-04-10", "2026-04-16", "2026-04-20"]),
                "exchange": ["SSE", "SSE", "SSE"],
                "is_open": [1, 1, 1],
                "pretrade_date": pd.to_datetime(["2026-04-09", "2026-04-15", "2026-04-16"]),
                "cal_date": ["20260410", "20260416", "20260420"],
            }
        )
        security_master = pd.DataFrame(
            {
                "code": ["600001.SH"],
                "name": ["TEST SH"],
                "industry": ["I1"],
                "market": ["SH"],
                "list_status": ["L"],
                "list_date": pd.to_datetime(["2026-01-01"]),
                "delist_date": [pd.NaT],
                "snapshot_date": pd.to_datetime(["2026-04-20"]),
            }
        )
        daily = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-04-10", "2026-04-16", "2026-04-20"]),
                "code": ["600001.SH", "600001.SH", "600001.SH"],
                "open": [10.0, 10.1, 10.2],
                "high": [10.5, 10.6, 10.7],
                "low": [9.8, 9.9, 10.0],
                "close": [10.2, 10.3, 10.4],
                "ret_1d": [0.01, 0.02, 0.03],
                "pct_chg": [0.01, 0.02, 0.03],
                "volume": [1000.0, 1200.0, 1300.0],
                "amount": [10000.0, 12000.0, 13000.0],
                "adj_factor": [1.0, 1.0, 1.0],
                "bfq_open": [10.0, 10.1, 10.2],
                "bfq_high": [10.5, 10.6, 10.7],
                "bfq_low": [9.8, 9.9, 10.0],
                "bfq_close": [10.2, 10.3, 10.4],
                "qfq_open": [10.0, 10.1, 10.2],
                "qfq_high": [10.5, 10.6, 10.7],
                "qfq_low": [9.8, 9.9, 10.0],
                "qfq_close": [10.2, 10.3, 10.4],
                "hfq_open": [10.0, 10.1, 10.2],
                "hfq_high": [10.5, 10.6, 10.7],
                "hfq_low": [9.8, 9.9, 10.0],
                "hfq_close": [10.2, 10.3, 10.4],
                "price_adjust_mode": ["qfq", "qfq", "qfq"],
            }
        )
        daily_basic = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-04-10", "2026-04-16", "2026-04-20"]),
                "code": ["600001.SH", "600001.SH", "600001.SH"],
                "turnover_rate": [1.0, 1.0, 1.0],
                "turnover_rate_f": [1.0, 1.0, 1.0],
                "volume_ratio": [1.0, 1.0, 1.0],
                "pe": [10.0, 10.0, 10.0],
                "pe_ttm": [10.0, 10.0, 10.0],
                "pb": [1.0, 1.0, 1.0],
                "ps": [1.0, 1.0, 1.0],
                "ps_ttm": [1.0, 1.0, 1.0],
                "dv_ratio": [0.0, 0.0, 0.0],
                "dv_ttm": [0.0, 0.0, 0.0],
                "total_mv": [1e9, 1e9, 1e9],
                "circ_mv": [8e8, 8e8, 8e8],
                "total_mv_raw_wan": [1e5, 1e5, 1e5],
                "circ_mv_raw_wan": [8e4, 8e4, 8e4],
            }
        )
        income = pd.DataFrame(
            {
                "code": ["600001.SH", "600001.SH"],
                "ann_date": pd.to_datetime(["2026-04-15", "2026-04-20"]),
                "end_date": pd.to_datetime(["2025-12-31", "2026-03-31"]),
                "total_revenue": [100.0, 200.0],
                "revenue": [90.0, 180.0],
                "operate_profit": [10.0, 20.0],
                "total_profit": [8.0, 16.0],
                "n_income_attr_p": [6.0, 12.0],
                "basic_eps": [0.10, 0.20],
                "diluted_eps": [0.10, 0.20],
            }
        )
        balance = pd.DataFrame(
            {
                "code": ["600001.SH", "600001.SH"],
                "ann_date": pd.to_datetime(["2026-04-15", "2026-04-20"]),
                "end_date": pd.to_datetime(["2025-12-31", "2026-03-31"]),
                "total_assets": [1000.0, 1500.0],
                "total_liab": [600.0, 900.0],
                "total_hldr_eqy_exc_min_int": [400.0, 600.0],
            }
        )
        indicator = pd.DataFrame(
            {
                "code": ["600001.SH", "600001.SH"],
                "ann_date": pd.to_datetime(["2026-04-15", "2026-04-20"]),
                "end_date": pd.to_datetime(["2025-12-31", "2026-03-31"]),
                "roe": [0.10, 0.20],
                "roa": [0.05, 0.10],
                "grossprofit_margin": [0.30, 0.35],
                "netprofit_margin": [0.10, 0.12],
                "assets_turn": [0.4, 0.5],
                "current_ratio": [1.5, 1.8],
                "q_roe": [0.03, 0.05],
            }
        )
        cashflow = pd.DataFrame(
            {
                "code": ["600001.SH", "600001.SH"],
                "ann_date": pd.to_datetime(["2026-04-15", "2026-04-20"]),
                "end_date": pd.to_datetime(["2025-12-31", "2026-03-31"]),
                "n_cashflow_act": [50.0, 70.0],
                "n_cashflow_inv_act": [-20.0, -30.0],
                "n_cash_flows_fnc_act": [10.0, 15.0],
            }
        )

        self._write_parquet(
            trade_cal,
            curated / "dims/trade_calendar/snapshot_date=2026-04-20/part-000.parquet",
        )
        self._write_parquet(
            security_master,
            curated / "dims/security_master/snapshot_date=2026-04-20/part-000.parquet",
        )
        self._write_parquet(daily, curated / "facts/market_daily/year=2026/month=04/part-000.parquet")
        self._write_parquet(
            daily_basic,
            curated / "facts/market_daily_basic/year=2026/month=04/part-000.parquet",
        )
        self._write_parquet(
            income,
            curated / "facts/finance_income_q/year=2026/month=04/part-000.parquet",
        )
        self._write_parquet(
            balance,
            curated / "facts/finance_balancesheet_q/year=2026/month=04/part-000.parquet",
        )
        self._write_parquet(
            cashflow,
            curated / "facts/finance_cashflow_q/year=2026/month=04/part-000.parquet",
        )
        self._write_parquet(
            indicator,
            curated / "facts/finance_indicator_q/year=2026/month=04/part-000.parquet",
        )

    def _write_parquet(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)


if __name__ == "__main__":
    unittest.main()
