from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from alpha_mining.datasource.field_catalog_builder import build_field_catalog_dataframe


class TestFieldCatalogBuilder(unittest.TestCase):
    def test_build_field_catalog_dataframe(self) -> None:
        try:
            import duckdb  # type: ignore
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"duckdb not available: {exc}")
            return

        base_dir = Path("data") / f"_field_catalog_builder_test_{uuid.uuid4().hex}"
        db_path = base_dir / "market.duckdb"
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            conn = duckdb.connect(str(db_path))
            conn.execute(
                """
                CREATE TABLE t_panel (
                    date DATE,
                    code VARCHAR,
                    close DOUBLE,
                    universe INTEGER,
                    days_since_listed INTEGER
                )
                """
            )
            conn.execute(
                """
                INSERT INTO t_panel VALUES
                ('2026-04-01', '600001.SH', 10.5, 1, 120),
                ('2026-04-02', '600001.SH', 10.7, 1, 121)
                """
            )
            conn.execute("CREATE VIEW v_project_panel_cn_a AS SELECT * FROM t_panel")

            df = build_field_catalog_dataframe(
                conn=conn,
                source_view="v_project_panel_cn_a",
                field_catalog_version="v_test",
            )
            self.assertFalse(df.empty)
            required_cols = {
                "field_name",
                "field_type",
                "category",
                "source_table",
                "dtype",
                "unit",
                "available_start",
                "available_end",
                "is_default_enabled",
                "is_searchable",
                "description",
                "field_catalog_version",
                "field_role",
                "available_at",
                "preprocessing_policy",
                "leakage_safe",
            }
            self.assertTrue(required_cols.issubset(set(df.columns)))

            row_close = df[df["field_name"] == "close"].iloc[0]
            self.assertEqual(str(row_close["category"]), "price")
            self.assertEqual(str(row_close["field_role"]), "signal_input")
            self.assertEqual(
                str(row_close["preprocessing_policy"]),
                "expression_wrapper:ts_backfill+winsorize",
            )
            self.assertTrue(bool(row_close["is_default_enabled"]))
            self.assertTrue(bool(row_close["is_searchable"]))

            row_universe = df[df["field_name"] == "universe"].iloc[0]
            self.assertFalse(bool(row_universe["is_searchable"]))

            row_days = df[df["field_name"] == "days_since_listed"].iloc[0]
            self.assertFalse(bool(row_days["is_searchable"]))
            conn.close()
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_panel_catalog_date_range_prefers_light_fact_table(self) -> None:
        try:
            import duckdb  # type: ignore
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"duckdb not available: {exc}")
            return

        base_dir = Path("data") / f"_field_catalog_builder_range_{uuid.uuid4().hex}"
        db_path = base_dir / "market.duckdb"
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            conn = duckdb.connect(str(db_path))
            conn.execute(
                """
                CREATE TABLE fact_market_daily (
                    date DATE,
                    code VARCHAR,
                    close DOUBLE
                )
                """
            )
            conn.execute(
                """
                INSERT INTO fact_market_daily VALUES
                ('2026-01-01', '600001.SH', 9.5),
                ('2026-01-03', '600001.SH', 9.7)
                """
            )
            conn.execute(
                """
                CREATE TABLE t_heavy_panel (
                    date DATE,
                    code VARCHAR,
                    close DOUBLE,
                    universe INTEGER
                )
                """
            )
            conn.execute(
                """
                INSERT INTO t_heavy_panel VALUES
                ('2026-04-01', '600001.SH', 10.5, 1),
                ('2026-04-02', '600001.SH', 10.7, 1)
                """
            )
            conn.execute("CREATE VIEW v_project_panel_cn_a AS SELECT * FROM t_heavy_panel")

            df = build_field_catalog_dataframe(
                conn=conn,
                source_view="v_project_panel_cn_a",
                field_catalog_version="v_test",
            )
            self.assertEqual(set(df["available_start"].unique()), {"2026-01-01"})
            self.assertEqual(set(df["available_end"].unique()), {"2026-01-03"})
            conn.close()
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_tech_prefixed_fields_are_classified_as_technical(self) -> None:
        try:
            import duckdb  # type: ignore
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"duckdb not available: {exc}")
            return

        base_dir = Path("data") / f"_field_catalog_builder_tech_{uuid.uuid4().hex}"
        db_path = base_dir / "market.duckdb"
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            conn = duckdb.connect(str(db_path))
            conn.execute(
                """
                CREATE TABLE t_panel (
                    date DATE,
                    code VARCHAR,
                    tech_asi_qfq DOUBLE
                )
                """
            )
            conn.execute("CREATE VIEW v_project_panel_cn_a AS SELECT * FROM t_panel")

            df = build_field_catalog_dataframe(
                conn=conn,
                source_view="v_project_panel_cn_a",
                field_catalog_version="v_test",
                default_enabled_categories=("technical",),
            )

            row = df[df["field_name"] == "tech_asi_qfq"].iloc[0]
            self.assertEqual(str(row["category"]), "technical")
            self.assertTrue(bool(row["is_default_enabled"]))
            self.assertTrue(bool(row["is_searchable"]))
            conn.close()
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_moneyflow_catalog_uses_official_tushare_amount_fields(self) -> None:
        try:
            import duckdb  # type: ignore
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"duckdb not available: {exc}")
            return

        base_dir = Path("data") / f"_field_catalog_builder_moneyflow_{uuid.uuid4().hex}"
        db_path = base_dir / "market.duckdb"
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            conn = duckdb.connect(str(db_path))
            conn.execute(
                """
                CREATE TABLE t_panel (
                    date DATE,
                    code VARCHAR,
                    moneyflow_buy_sm_amount DOUBLE,
                    moneyflow_sell_sm_amount DOUBLE,
                    moneyflow_buy_md_amount DOUBLE,
                    moneyflow_sell_md_amount DOUBLE,
                    moneyflow_buy_lg_amount DOUBLE,
                    moneyflow_sell_lg_amount DOUBLE,
                    moneyflow_buy_elg_amount DOUBLE,
                    moneyflow_sell_elg_amount DOUBLE,
                    moneyflow_net_mf_amount DOUBLE
                )
                """
            )
            conn.execute("CREATE VIEW v_project_panel_cn_a AS SELECT * FROM t_panel")

            df = build_field_catalog_dataframe(
                conn=conn,
                source_view="v_project_panel_cn_a",
                field_catalog_version="v_test",
                default_enabled_categories=("moneyflow",),
            )

            fields = set(df["field_name"].astype(str))
            for field in [
                "moneyflow_buy_sm_amount",
                "moneyflow_sell_sm_amount",
                "moneyflow_buy_md_amount",
                "moneyflow_sell_md_amount",
                "moneyflow_buy_lg_amount",
                "moneyflow_sell_lg_amount",
                "moneyflow_buy_elg_amount",
                "moneyflow_sell_elg_amount",
                "moneyflow_net_mf_amount",
            ]:
                row = df[df["field_name"] == field].iloc[0]
                self.assertEqual(str(row["category"]), "moneyflow")
                self.assertEqual(str(row["factor_family"]), "moneyflow")
                self.assertTrue(bool(row["is_searchable"]))
            self.assertNotIn("moneyflow_buy_lg_amount_rate", fields)
            self.assertNotIn("moneyflow_net_d5_amount", fields)
            conn.close()
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_cyq_catalog_uses_all_official_perf_fields(self) -> None:
        try:
            import duckdb  # type: ignore
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"duckdb not available: {exc}")
            return

        base_dir = Path("data") / f"_field_catalog_builder_cyq_{uuid.uuid4().hex}"
        db_path = base_dir / "market.duckdb"
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            conn = duckdb.connect(str(db_path))
            conn.execute(
                """
                CREATE TABLE t_panel (
                    date DATE,
                    code VARCHAR,
                    cyq_his_low DOUBLE,
                    cyq_his_high DOUBLE,
                    cyq_cost_5pct DOUBLE,
                    cyq_cost_15pct DOUBLE,
                    cyq_cost_50pct DOUBLE,
                    cyq_cost_85pct DOUBLE,
                    cyq_cost_95pct DOUBLE,
                    cyq_weight_avg DOUBLE,
                    cyq_winner_rate DOUBLE
                )
                """
            )
            conn.execute("CREATE VIEW v_project_panel_cn_a AS SELECT * FROM t_panel")

            df = build_field_catalog_dataframe(
                conn=conn,
                source_view="v_project_panel_cn_a",
                field_catalog_version="v_test",
                default_enabled_categories=("chip",),
            )

            for field in [
                "cyq_his_low",
                "cyq_his_high",
                "cyq_cost_5pct",
                "cyq_cost_15pct",
                "cyq_cost_50pct",
                "cyq_cost_85pct",
                "cyq_cost_95pct",
                "cyq_weight_avg",
                "cyq_winner_rate",
            ]:
                row = df[df["field_name"] == field].iloc[0]
                self.assertEqual(str(row["category"]), "chip")
                self.assertTrue(bool(row["is_searchable"]))
            conn.close()
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_cyq_chips_daily_catalog_uses_chip_category(self) -> None:
        try:
            import duckdb  # type: ignore
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"duckdb not available: {exc}")
            return

        base_dir = Path("data") / f"_field_catalog_builder_cyq_chips_{uuid.uuid4().hex}"
        db_path = base_dir / "market.duckdb"
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            conn = duckdb.connect(str(db_path))
            conn.execute(
                """
                CREATE TABLE t_panel (
                    date DATE,
                    code VARCHAR,
                    cyq_chip_price_count DOUBLE,
                    cyq_chip_percent_sum DOUBLE,
                    cyq_chip_weight_avg_price DOUBLE,
                    cyq_chip_cost_50pct DOUBLE
                )
                """
            )
            conn.execute("CREATE VIEW v_project_panel_cn_a AS SELECT * FROM t_panel")

            df = build_field_catalog_dataframe(
                conn=conn,
                source_view="v_project_panel_cn_a",
                field_catalog_version="v_test",
                default_enabled_categories=("chip",),
            )

            for field in [
                "cyq_chip_price_count",
                "cyq_chip_percent_sum",
                "cyq_chip_weight_avg_price",
                "cyq_chip_cost_50pct",
            ]:
                row = df[df["field_name"] == field].iloc[0]
                self.assertEqual(str(row["category"]), "chip")
                self.assertEqual(str(row["source_table"]), "v_project_market_daily_base")
                self.assertTrue(bool(row["is_searchable"]))
                self.assertTrue(bool(row["leakage_safe"]))
            conn.close()
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
