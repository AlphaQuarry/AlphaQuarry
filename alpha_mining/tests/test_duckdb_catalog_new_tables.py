from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pandas as pd

from alpha_mining.datasource.config import LakePathSettings
from alpha_mining.datasource.duckdb_catalog import build_duckdb_catalog


def test_duckdb_catalog_registers_new_tushare_fact_views_and_panel_columns() -> None:
    try:
        import duckdb  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise AssertionError(f"duckdb required for this test: {exc}")

    base_dir = Path("data") / f"_duckdb_new_tables_{uuid.uuid4().hex}"
    settings = LakePathSettings(
        lake_root=str((base_dir / "lake").as_posix()),
        duckdb_path=str((base_dir / "duckdb" / "market.duckdb").as_posix()),
    )
    try:
        _write_minimal_panel_lake(settings)
        build_duckdb_catalog(paths=settings)
        conn = duckdb.connect(str(settings.duckdb_path_obj), read_only=True)
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info('v_project_panel_cn_a')").fetchall()}
            assert "moneyflow_net_mf_amount" in cols
            assert "moneyflow_buy_sm_amount" in cols
            assert "moneyflow_sell_sm_amount" in cols
            assert "moneyflow_buy_md_amount" in cols
            assert "moneyflow_sell_md_amount" in cols
            assert "moneyflow_buy_lg_amount" in cols
            assert "moneyflow_sell_lg_amount" in cols
            assert "moneyflow_buy_elg_amount" in cols
            assert "moneyflow_sell_elg_amount" in cols
            assert "moneyflow_net_amount" not in cols
            assert "moneyflow_net_d5_amount" not in cols
            assert "moneyflow_buy_lg_amount_rate" not in cols
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
                assert field in cols
            for field in [
                "cyq_chip_price_count",
                "cyq_chip_percent_sum",
                "cyq_chip_price_min",
                "cyq_chip_price_max",
                "cyq_chip_mode_price",
                "cyq_chip_mode_percent",
                "cyq_chip_weight_avg_price",
                "cyq_chip_price_std",
                "cyq_chip_cost_10pct",
                "cyq_chip_cost_25pct",
                "cyq_chip_cost_50pct",
                "cyq_chip_cost_75pct",
                "cyq_chip_cost_90pct",
            ]:
                assert field in cols
            chip_long_cols = {row[1] for row in conn.execute("PRAGMA table_info('fact_cyq_chips')").fetchall()}
            assert {"date", "code", "chip_price", "chip_percent"} <= chip_long_cols
            assert "tech_rsi_qfq_6" in cols
            assert "tech_ma_qfq_20" in cols
            assert "tech_asi_qfq" in cols
            assert "tech_macd_qfq" in cols
            assert "tech_rsi_qfq_24" in cols
            assert "tech_xsii_td4_qfq" in cols
            assert "tech_rsi_hfq_6" not in cols
            assert "tech_ma_hfq_20" not in cols
            assert "tech_updays" in cols
            assert "tech_downdays" in cols
            assert "tech_topdays" in cols
            assert "tech_lowdays" in cols
            assert "auction_o_open" not in cols
            assert "auction_c_close" not in cols
            assert "report_rc_eps_mean" in cols
            hot_cols = {
                row[1] for row in conn.execute("PRAGMA table_info('v_project_market_daily_base_hot')").fetchall()
            }
            assert "close" in hot_cols
            assert "universe" in hot_cols
            assert "tech_asi_qfq" not in hot_cols
            assert "moneyflow_net_mf_amount" not in hot_cols
            index_cols = {row[1] for row in conn.execute("PRAGMA table_info('v_project_index_daily')").fetchall()}
            assert {"date", "code", "close", "pct_chg", "return"} <= index_cols
            index_row = conn.execute("SELECT code, close, pct_chg, return FROM v_project_index_daily").fetchone()
            assert index_row == ("000300.SH", 4010.0, 0.0025, 0.0025)
            membership_cols = {
                row[1] for row in conn.execute("PRAGMA table_info('v_project_index_membership_asof')").fetchall()
            }
            assert {
                "date",
                "universe_name",
                "index_code",
                "code",
                "weight",
                "membership_date",
            } <= membership_cols
            hs300_row = conn.execute("SELECT code, close FROM v_project_panel_cn_a_hs300").fetchone()
            assert hs300_row == ("000001.SZ", 1.0)
            csi500_count = conn.execute("SELECT COUNT(*) FROM v_project_panel_cn_a_csi500").fetchone()[0]
            assert csi500_count == 0
        finally:
            conn.close()
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_duckdb_catalog_uses_qfq_stk_factor_pro_even_when_adjust_mode_is_hfq() -> None:
    try:
        import duckdb  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise AssertionError(f"duckdb required for this test: {exc}")

    base_dir = Path("data") / f"_duckdb_hfq_tech_{uuid.uuid4().hex}"
    settings = LakePathSettings(
        lake_root=str((base_dir / "lake").as_posix()),
        duckdb_path=str((base_dir / "duckdb" / "market.duckdb").as_posix()),
    )
    try:
        _write_minimal_panel_lake(settings, adjust_mode="hfq")
        build_duckdb_catalog(paths=settings, adjust_mode="hfq")
        conn = duckdb.connect(str(settings.duckdb_path_obj), read_only=True)
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info('v_project_panel_cn_a')").fetchall()}
            assert "tech_rsi_qfq_6" in cols
            assert "tech_rsi_hfq_6" not in cols
        finally:
            conn.close()
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_duckdb_catalog_preserves_index_source_view_when_requested() -> None:
    try:
        import duckdb  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise AssertionError(f"duckdb required for this test: {exc}")

    base_dir = Path("data") / f"_duckdb_index_source_{uuid.uuid4().hex}"
    settings = LakePathSettings(
        lake_root=str((base_dir / "lake").as_posix()),
        duckdb_path=str((base_dir / "duckdb" / "market.duckdb").as_posix()),
    )
    try:
        _write_minimal_panel_lake(settings)
        build_duckdb_catalog(paths=settings, source_view="v_project_panel_cn_a_hs300")
        conn = duckdb.connect(str(settings.duckdb_path_obj), read_only=True)
        try:
            count = conn.execute("SELECT COUNT(*) FROM v_project_panel_cn_a_hs300").fetchone()[0]
            assert count == 1
        finally:
            conn.close()
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def _write_minimal_panel_lake(settings: LakePathSettings, adjust_mode: str = "qfq") -> None:
    def write(table: str, df: pd.DataFrame) -> None:
        path = settings.curated_path / table / "part-000.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    date = pd.Timestamp("2026-04-21")
    code = "000001.SZ"
    index_universe_df = pd.DataFrame(
        {
            "universe_name": ["hs300"],
            "display_name": ["沪深300"],
            "resolved_index_code": ["000300.SH"],
            "index_daily_code": ["000300.SH"],
            "index_weight_code": ["000300.SH"],
            "resolved_name": ["沪深300"],
            "market": ["CSI"],
            "publisher": ["中证公司"],
            "category": ["规模指数"],
            "required": [True],
            "enabled": [True],
            "status": ["active"],
            "candidate_codes_json": ['["000300.SH"]'],
            "candidate_symbols_json": ['["000300"]'],
            "candidate_names_json": ['["沪深300"]'],
            "resolved_at": ["2026-04-21T00:00:00+00:00"],
            "snapshot_date": [date],
        }
    )
    write(
        "dims/trade_calendar",
        pd.DataFrame(
            {
                "date": [date],
                "exchange": ["SSE"],
                "is_open": [1],
                "pretrade_date": [pd.NaT],
                "cal_date": ["20260421"],
            }
        ),
    )
    write("dims/index_universe", index_universe_df)
    write(
        "dims/security_master",
        pd.DataFrame(
            {
                "code": [code],
                "name": ["A"],
                "industry": ["Bank"],
                "market": ["主板"],
                "list_status": ["L"],
                "list_date": [pd.Timestamp("2020-01-01")],
                "delist_date": [pd.NaT],
                "snapshot_date": [date],
            }
        ),
    )
    write(
        "dims/sw_membership_history",
        pd.DataFrame(
            {
                "code": [code],
                "index_code": ["801000"],
                "in_date": [pd.Timestamp("2020-01-01")],
                "out_date": [pd.NaT],
                "sector": ["金融"],
                "industry": ["银行"],
                "subindustry": ["银行"],
            }
        ),
    )
    write(
        "dims/security_namechange",
        pd.DataFrame(
            {
                "code": [code],
                "name": ["A"],
                "normalized_name": ["A"],
                "start_date": [pd.Timestamp("2020-01-01")],
                "end_date": [pd.NaT],
                "ann_date": [pd.Timestamp("2020-01-01")],
                "change_reason": [""],
                "is_st": [0],
            }
        ),
    )
    write(
        "facts/market_daily",
        pd.DataFrame(
            {
                "date": [date],
                "code": [code],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "pct_chg": [0.0],
                "ret_1d": [0.0],
                "volume": [100.0],
                "amount": [1000.0],
                "adj_factor": [1.0],
                "bfq_open": [1.0],
                "bfq_high": [1.0],
                "bfq_low": [1.0],
                "bfq_close": [1.0],
                "qfq_open": [1.0],
                "qfq_high": [1.0],
                "qfq_low": [1.0],
                "qfq_close": [1.0],
                "hfq_open": [1.0],
                "hfq_high": [1.0],
                "hfq_low": [1.0],
                "hfq_close": [1.0],
                "price_adjust_mode": [adjust_mode],
            }
        ),
    )
    write(
        "facts/market_daily_basic",
        pd.DataFrame(
            {
                "date": [date],
                "code": [code],
                "turnover_rate": [1.0],
                "turnover_rate_f": [1.0],
                "volume_ratio": [1.0],
                "pe": [1.0],
                "pe_ttm": [1.0],
                "pb": [1.0],
                "ps": [1.0],
                "ps_ttm": [1.0],
                "dv_ratio": [1.0],
                "dv_ttm": [1.0],
                "total_mv": [1.0],
                "circ_mv": [1.0],
                "total_mv_raw_wan": [1.0],
                "circ_mv_raw_wan": [1.0],
            }
        ),
    )
    write(
        "facts/market_stk_limit",
        pd.DataFrame(
            {
                "date": [date],
                "code": [code],
                "pre_close": [1.0],
                "up_limit": [1.1],
                "down_limit": [0.9],
            }
        ),
    )
    write(
        "facts/market_suspend_d",
        pd.DataFrame(
            {
                "date": [date],
                "code": [code],
                "suspend_timing": [""],
                "suspend_type": [""],
                "is_suspended": [0],
            }
        ),
    )
    write(
        "facts/moneyflow",
        pd.DataFrame(
            {
                "date": [date],
                "code": [code],
                "moneyflow_buy_sm_amount": [1.0],
                "moneyflow_sell_sm_amount": [2.0],
                "moneyflow_buy_md_amount": [3.0],
                "moneyflow_sell_md_amount": [4.0],
                "moneyflow_buy_lg_amount": [5.0],
                "moneyflow_sell_lg_amount": [6.0],
                "moneyflow_buy_elg_amount": [7.0],
                "moneyflow_sell_elg_amount": [8.0],
                "moneyflow_net_mf_amount": [9.0],
            }
        ),
    )
    write(
        "facts/cyq_perf",
        pd.DataFrame(
            {
                "date": [date],
                "code": [code],
                "cyq_his_low": [0.5],
                "cyq_his_high": [2.0],
                "cyq_cost_5pct": [0.8],
                "cyq_cost_15pct": [0.9],
                "cyq_cost_50pct": [1.0],
                "cyq_cost_85pct": [1.1],
                "cyq_cost_95pct": [1.2],
                "cyq_weight_avg": [1.05],
                "cyq_winner_rate": [0.5],
            }
        ),
    )
    write(
        "facts/cyq_chips",
        pd.DataFrame(
            {
                "date": [date, date],
                "code": [code, code],
                "chip_price": [0.9, 1.1],
                "chip_percent_raw_pct": [40.0, 60.0],
                "chip_percent": [0.4, 0.6],
            }
        ),
    )
    write(
        "facts/cyq_chips_daily",
        pd.DataFrame(
            {
                "date": [date],
                "code": [code],
                "cyq_chip_price_count": [2],
                "cyq_chip_percent_sum": [1.0],
                "cyq_chip_price_min": [0.9],
                "cyq_chip_price_max": [1.1],
                "cyq_chip_mode_price": [1.1],
                "cyq_chip_mode_percent": [0.6],
                "cyq_chip_weight_avg_price": [1.02],
                "cyq_chip_price_std": [0.1414213562],
                "cyq_chip_cost_10pct": [0.9],
                "cyq_chip_cost_25pct": [0.9],
                "cyq_chip_cost_50pct": [1.1],
                "cyq_chip_cost_75pct": [1.1],
                "cyq_chip_cost_90pct": [1.1],
            }
        ),
    )
    write(
        "facts/stk_factor_pro",
        pd.DataFrame(
            {
                "date": [date],
                "code": [code],
                "tech_rsi_qfq_6": [50.0],
                "tech_ma_qfq_20": [1.0],
                "tech_asi_qfq": [2.0],
                "tech_macd_qfq": [3.0],
                "tech_rsi_qfq_24": [4.0],
                "tech_xsii_td4_qfq": [5.0],
                "tech_rsi_hfq_6": [60.0],
                "tech_ma_hfq_20": [6.0],
                "tech_updays": [2.0],
                "tech_downdays": [1.0],
                "tech_topdays": [3.0],
                "tech_lowdays": [4.0],
            }
        ),
    )
    write(
        "facts/stk_auction_o",
        pd.DataFrame({"date": [date], "code": [code], "auction_o_open": [1.0]}),
    )
    write(
        "facts/stk_auction_c",
        pd.DataFrame({"date": [date], "code": [code], "auction_c_close": [1.0]}),
    )
    write(
        "facts/report_rc_daily",
        pd.DataFrame({"date": [date], "code": [code], "report_rc_eps_mean": [1.0]}),
    )
    write(
        "dims/index_basic",
        pd.DataFrame(
            {
                "code": ["000300.SH"],
                "name": ["沪深300"],
                "market": ["CSI"],
                "publisher": ["CSI"],
                "category": ["broad"],
                "base_date": [pd.Timestamp("2004-12-31")],
                "base_point": [1000.0],
                "list_date": [pd.Timestamp("2005-04-08")],
                "weight_rule": [""],
                "description": [""],
                "exp_date": [pd.NaT],
                "snapshot_date": [date],
            }
        ),
    )
    write(
        "facts/index_daily",
        pd.DataFrame(
            {
                "date": [date],
                "code": ["000300.SH"],
                "open": [4000.0],
                "high": [4020.0],
                "low": [3990.0],
                "close": [4010.0],
                "pre_close": [4000.0],
                "change": [10.0],
                "vendor_pct_chg_raw_pct": [0.25],
                "pct_chg": [0.0025],
                "return": [0.0025],
                "vol": [100.0],
                "amount": [200.0],
            }
        ),
    )
    write(
        "facts/index_weight",
        pd.DataFrame(
            {
                "date": [date],
                "index_code": ["000300.SH"],
                "code": [code],
                "weight": [1.5],
                "weight_decimal": [0.015],
                "is_member": [1],
            }
        ),
    )
