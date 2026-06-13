from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from alpha_mining.datasource.config import LakePathSettings
from alpha_mining.datasource.ingestion_scope import (
    DIM_GROUP_TABLES,
    FACT_GROUP_TABLES,
    resolve_dim_table_selection,
    resolve_fact_table_selection,
)
from alpha_mining.datasource.parquet_lake import ParquetLake


def _load_update_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "update_tushare_lake.py"
    spec = importlib.util.spec_from_file_location("update_tushare_lake_new_tables", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Client:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def fetch_moneyflow_by_trade_date(self, trade_date: str) -> pd.DataFrame:
        self.calls.append("moneyflow")
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": [trade_date],
                "buy_sm_amount": [1.0],
                "sell_sm_amount": [2.0],
                "buy_md_amount": [3.0],
                "sell_md_amount": [4.0],
                "buy_lg_amount": [5.0],
                "sell_lg_amount": [6.0],
                "buy_elg_amount": [7.0],
                "sell_elg_amount": [8.0],
                "net_mf_amount": [9.0],
            }
        )

    def fetch_cyq_perf_by_trade_date(self, trade_date: str) -> pd.DataFrame:
        self.calls.append("cyq_perf")
        return pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": [trade_date], "winner_rate": [1.0]})

    def fetch_cyq_perf_by_ts_code(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        self.calls.append("cyq_perf_by_code")
        return pd.DataFrame({"ts_code": [ts_code], "trade_date": [end_date], "winner_rate": [1.0]})

    def fetch_cyq_chips_by_ts_code(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        self.calls.append("cyq_chips_by_code")
        return pd.DataFrame(
            {
                "ts_code": [ts_code, ts_code],
                "trade_date": [end_date, end_date],
                "price": [10.0, 10.5],
                "percent": [40.0, 60.0],
            }
        )

    def fetch_stk_factor_pro_by_trade_date(self, trade_date: str) -> pd.DataFrame:
        self.calls.append("stk_factor_pro")
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": [trade_date],
                "rsi_qfq_6": [1.0],
                "rsi_hfq_6": [2.0],
                "asi_qfq": [3.0],
            }
        )

    def fetch_stk_factor_pro_by_ts_code(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        self.calls.append("stk_factor_pro_by_code")
        return pd.DataFrame(
            {
                "ts_code": [ts_code],
                "trade_date": [end_date],
                "rsi_qfq_6": [1.0],
                "asi_qfq": [3.0],
            }
        )

    def fetch_index_weight_by_index_code(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        self.calls.append("index_weight_by_code")
        return pd.DataFrame(
            {
                "index_code": [index_code],
                "con_code": ["000001.SZ"],
                "trade_date": [end_date],
                "weight": [1.5],
            }
        )

    def fetch_stk_auction_o_by_trade_date(self, trade_date: str) -> pd.DataFrame:
        self.calls.append("stk_auction_o")
        return pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": [trade_date], "open": [1.0]})

    def fetch_stk_auction_c_by_trade_date(self, trade_date: str) -> pd.DataFrame:
        self.calls.append("stk_auction_c")
        return pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": [trade_date], "close": [1.0]})

    def fetch_report_rc(self, start_date: str, end_date: str) -> pd.DataFrame:
        self.calls.append("report_rc")
        return pd.DataFrame({"ts_code": ["000001.SZ"], "report_date": [end_date], "eps": [1.0]})


def test_fact_groups_default_moneyflow_and_legacy_group() -> None:
    assert FACT_GROUP_TABLES["p3"] == ("moneyflow",)
    assert FACT_GROUP_TABLES["p3_legacy"] == ("moneyflow_ths",)
    assert FACT_GROUP_TABLES["p4"] == ("cyq_perf", "cyq_chips", "stk_factor_pro")
    assert FACT_GROUP_TABLES["p4_auction"] == ("stk_auction_o", "stk_auction_c")
    assert FACT_GROUP_TABLES["p5"] == ("report_rc",)
    assert FACT_GROUP_TABLES["index"] == ("index_daily", "index_weight")
    assert DIM_GROUP_TABLES["index"] == ("index_basic",)

    assert resolve_fact_table_selection(groups_raw="p3") == ["moneyflow"]
    assert resolve_fact_table_selection(groups_raw="p3_legacy") == ["moneyflow_ths"]
    assert "report_rc" in resolve_fact_table_selection(groups_raw="p5")
    assert resolve_fact_table_selection(groups_raw="index") == [
        "index_daily",
        "index_weight",
    ]
    assert resolve_dim_table_selection(groups_raw="index") == ["index_basic"]


def test_update_fetch_and_curate_new_trade_date_tables() -> None:
    mod = _load_update_module()
    client = _Client()

    fetched = mod._fetch_selected_fact_frames(
        client=client,
        selected_fact_tables=[
            "moneyflow",
            "stk_factor_pro",
            "stk_auction_o",
            "stk_auction_c",
        ],
        trade_date="20260421",
    )
    curated = mod._build_curated_fact_frames(
        selected_fact_tables=list(fetched),
        raw_table_data=fetched,
        adjust_mode="qfq",
    )

    assert set(fetched) == {
        "moneyflow",
        "stk_factor_pro",
        "stk_auction_o",
        "stk_auction_c",
    }
    assert "moneyflow_net_mf_amount" in curated["moneyflow"].columns
    assert "moneyflow_sell_lg_amount" in curated["moneyflow"].columns
    assert "moneyflow_buy_lg_amount_rate" not in curated["moneyflow"].columns
    assert "moneyflow_net_d5_amount" not in curated["moneyflow"].columns
    assert "moneyflow_net_mf_vol" not in curated["moneyflow"].columns
    assert "tech_rsi_qfq_6" in curated["stk_factor_pro"].columns
    assert "tech_asi_qfq" in curated["stk_factor_pro"].columns
    assert "tech_rsi_hfq_6" not in curated["stk_factor_pro"].columns
    assert "auction_o_open" in curated["stk_auction_o"].columns
    assert "auction_c_close" in curated["stk_auction_c"].columns


def test_update_curates_stk_factor_pro_qfq_even_when_adjust_mode_is_hfq() -> None:
    mod = _load_update_module()
    client = _Client()
    fetched = mod._fetch_selected_fact_frames(
        client=client,
        selected_fact_tables=["stk_factor_pro"],
        trade_date="20260421",
    )

    curated = mod._build_curated_fact_frames(
        selected_fact_tables=["stk_factor_pro"],
        raw_table_data=fetched,
        adjust_mode="hfq",
    )

    assert "tech_rsi_qfq_6" in curated["stk_factor_pro"].columns
    assert "tech_asi_qfq" in curated["stk_factor_pro"].columns
    assert "tech_rsi_hfq_6" not in curated["stk_factor_pro"].columns


def test_cyq_perf_is_code_range_table_not_trade_date_table() -> None:
    mod = _load_update_module()
    client = _Client()

    trade_tables, code_range_tables, range_tables = mod._split_selected_fact_tables(
        ["moneyflow", "cyq_perf", "report_rc"]
    )
    raw = mod._fetch_code_range_fact_frame(
        client=client,
        table="cyq_perf",
        ts_code="000001.SZ",
        start_date="2016-01-01",
        end_date="2016-01-08",
    )
    curated = mod._curate_code_range_fact_table("cyq_perf", raw)

    assert trade_tables == ["moneyflow"]
    assert code_range_tables == ["cyq_perf"]
    assert range_tables == ["report_rc"]
    assert client.calls == ["cyq_perf_by_code"]
    assert "cyq_winner_rate" in curated.columns


def test_cyq_chips_is_code_range_table_and_derives_daily_features() -> None:
    mod = _load_update_module()
    client = _Client()

    trade_tables, code_range_tables, range_tables = mod._split_selected_fact_tables(
        ["moneyflow", "cyq_chips", "report_rc"]
    )
    raw = mod._fetch_code_range_fact_frame(
        client=client,
        table="cyq_chips",
        ts_code="000001.SZ",
        start_date="2016-01-01",
        end_date="2016-01-08",
    )
    curated = mod._curate_code_range_fact_table("cyq_chips", raw)

    assert trade_tables == ["moneyflow"]
    assert code_range_tables == ["cyq_chips"]
    assert range_tables == ["report_rc"]
    assert client.calls == ["cyq_chips_by_code"]
    assert "long" in curated
    assert "daily" in curated
    assert "chip_percent" in curated["long"].columns
    assert "cyq_chip_weight_avg_price" in curated["daily"].columns


def test_cyq_chips_dry_run_plan_uses_code_range_calls() -> None:
    mod = _load_update_module()

    plan = mod._build_execution_plan(
        selected_trade_fact_tables=[],
        selected_code_range_fact_tables=["cyq_chips"],
        selected_range_fact_tables=[],
        selected_dim_tables=[],
        refresh_dims=False,
        start_date="2018-01-01",
        end_date="2018-01-08",
        open_trade_dates=[],
        pending_trade_dates=[],
        flush_trade_days=20,
        range_window_days=180,
        prune_out_of_range=False,
        skip_duckdb=True,
        trade_calendar_source="not_required",
        need_trade_bundle=False,
        code_range_ts_code_count=2,
    )

    assert plan["selected_trade_fact_tables"] == []
    assert plan["selected_code_range_fact_tables"] == ["cyq_chips"]
    assert plan["selected_range_fact_tables"] == []
    assert plan["trade_fact_calls_by_table"] == {}
    assert plan["code_range_fact_calls_by_table"] == {"cyq_chips": 2}
    assert plan["range_fact_calls_by_table"] == {}


def test_stk_factor_pro_can_use_code_range_repair_mode() -> None:
    mod = _load_update_module()
    client = _Client()

    trade_tables, code_range_tables, range_tables = mod._split_selected_fact_tables(
        ["moneyflow", "stk_factor_pro"],
        stk_factor_pro_fetch_mode="ts_code_range",
    )
    raw = mod._fetch_code_range_fact_frame(
        client=client,
        table="stk_factor_pro",
        ts_code="000002.SZ",
        start_date="2025-08-01",
        end_date="2025-08-07",
    )
    curated = mod._curate_code_range_fact_table("stk_factor_pro", raw)

    assert trade_tables == ["moneyflow"]
    assert code_range_tables == ["stk_factor_pro"]
    assert range_tables == []
    assert client.calls == ["stk_factor_pro_by_code"]
    assert "tech_rsi_qfq_6" in curated.columns
    assert "tech_asi_qfq" in curated.columns


def test_stk_factor_pro_sparse_repair_plan_detects_all_null_dates(tmp_path) -> None:
    mod = _load_update_module()
    lake = ParquetLake(paths=LakePathSettings(lake_root=str(tmp_path / "lake")))
    lake.write_curated_trade_table(
        table="facts/stk_factor_pro",
        df=pd.DataFrame(
            {
                "code": ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"],
                "date": ["2025-08-07", "2025-08-07", "2025-08-08", "2025-08-08"],
                "tech_dpo_qfq": [float("nan"), float("nan"), 1.0, 2.0],
                "tech_macd_qfq": [float("nan"), float("nan"), 3.0, 4.0],
            }
        ),
        date_col="date",
        key_cols=("code", "date"),
        mode="overwrite",
    )

    plan = mod._build_stk_factor_pro_sparse_repair_plan(
        lake=lake,
        start_date="2025-08-01",
        end_date="2025-08-31",
        fields_raw="dpo_qfq,macd_qfq",
        min_rows=1,
        max_finite_rate=0.01,
        output_csv=str(tmp_path / "sparse.csv"),
    )

    assert plan["summary"]["sparse_dates"] == 1
    assert plan["summary"]["sparse_months"] == 1
    assert plan["rows"][0]["date"] == "2025-08-07"
    assert plan["windows"] == [{"month": "2025-08", "start_date": "2025-08-01", "end_date": "2025-08-31"}]
    assert (tmp_path / "sparse.csv").exists()


def test_update_range_report_rc_uses_daily_aggregate() -> None:
    mod = _load_update_module()
    client = _Client()

    raw = mod._fetch_range_fact_frame(client, "report_rc", "2026-04-01", "2026-04-21")
    curated = mod._curate_range_fact_table("report_rc", raw)

    assert client.calls == ["report_rc"]
    assert "report_rc_count" in curated.columns
    assert "report_rc_eps_mean" in curated.columns


def test_index_daily_uses_common_broad_index_default_codes() -> None:
    mod = _load_update_module()

    codes = mod._resolve_index_daily_ts_codes("")

    assert codes == [
        "000300.SH",
        "000905.SH",
        "000852.SH",
        "000016.SH",
        "000001.SH",
        "399001.SZ",
        "399006.SZ",
    ]


def test_index_weight_is_code_range_table_and_curates_membership() -> None:
    mod = _load_update_module()
    client = _Client()

    trade_tables, code_range_tables, range_tables = mod._split_selected_fact_tables(["index_weight"])
    raw = mod._fetch_code_range_fact_frame(
        client=client,
        table="index_weight",
        ts_code="000300.SH",
        start_date="2026-04-01",
        end_date="2026-04-30",
    )
    curated = mod._curate_code_range_fact_table("index_weight", raw)

    assert trade_tables == []
    assert code_range_tables == ["index_weight"]
    assert range_tables == []
    assert client.calls == ["index_weight_by_code"]
    assert list(curated.columns) == [
        "date",
        "index_code",
        "code",
        "weight",
        "weight_decimal",
        "is_member",
    ]
    assert curated.iloc[0]["index_code"] == "000300.SH"
    assert curated.iloc[0]["code"] == "000001.SZ"
    assert curated.iloc[0]["weight_decimal"] == 0.015
