from __future__ import annotations

import pandas as pd

from alpha_mining.datasource.config import TushareSettings
from alpha_mining.datasource.tushare_client import TushareClient


class _NoInitClient(TushareClient):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._settings = TushareSettings(token="dummy")

    def _call(self, api_name: str, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append((api_name, kwargs))
        if api_name == "index_weight":
            return pd.DataFrame(
                {
                    "index_code": [str(kwargs.get("index_code", "000300.SH"))],
                    "con_code": ["000001.SZ"],
                    "trade_date": [str(kwargs.get("end_date", "20260421"))],
                    "weight": [1.0],
                }
            )
        return pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260421"]})


def test_tushare_client_exposes_new_trade_date_interfaces() -> None:
    client = _NoInitClient()

    client.fetch_moneyflow_by_trade_date("2026-04-21")
    client.fetch_cyq_perf_by_trade_date("2026-04-21")
    client.fetch_stk_factor_pro_by_trade_date("2026-04-21", fields="ts_code,trade_date,rsi_qfq_6")
    client.fetch_stk_auction_o_by_trade_date("2026-04-21")
    client.fetch_stk_auction_c_by_trade_date("2026-04-21")

    assert [name for name, _kwargs in client.calls] == [
        "moneyflow",
        "cyq_perf",
        "stk_factor_pro",
        "stk_auction_o",
        "stk_auction_c",
    ]
    assert all(call[1]["trade_date"] == "20260421" for call in client.calls)


def test_tushare_client_exposes_cyq_chips_code_range_interface() -> None:
    client = _NoInitClient()

    client.fetch_cyq_chips_by_ts_code("000001.SZ", start_date="2026-04-01", end_date="2026-04-21")

    api_name, kwargs = client.calls[-1]
    assert api_name == "cyq_chips"
    assert kwargs["ts_code"] == "000001.SZ"
    assert kwargs["start_date"] == "20260401"
    assert kwargs["end_date"] == "20260421"
    assert kwargs["fields"] == "ts_code,trade_date,price,percent"


def test_tushare_moneyflow_uses_official_amount_fields_only() -> None:
    client = _NoInitClient()

    client.fetch_moneyflow_by_trade_date("2026-04-21")

    _api_name, kwargs = client.calls[-1]
    fields = str(kwargs["fields"]).split(",")
    assert fields == [
        "ts_code",
        "trade_date",
        "buy_sm_amount",
        "sell_sm_amount",
        "buy_md_amount",
        "sell_md_amount",
        "buy_lg_amount",
        "sell_lg_amount",
        "buy_elg_amount",
        "sell_elg_amount",
        "net_mf_amount",
    ]
    assert not any(field.endswith("_vol") for field in fields)
    assert "net_d5_amount" not in fields


def test_tushare_client_exposes_report_rc_range_interface() -> None:
    client = _NoInitClient()

    client.fetch_report_rc(start_date="2026-04-01", end_date="2026-04-21")

    api_name, kwargs = client.calls[-1]
    assert api_name == "report_rc"
    assert kwargs["start_date"] == "20260401"
    assert kwargs["end_date"] == "20260421"


def test_tushare_client_exposes_index_basic_and_index_daily_interfaces() -> None:
    client = _NoInitClient()

    client.fetch_index_basic(market="CSI")
    client.fetch_index_daily_by_ts_code("000300.SH", start_date="2026-01-01", end_date="2026-04-30")
    client.fetch_index_weight_by_index_code("000300.SH", start_date="2026-04-01", end_date="2026-04-30")

    basic_name, basic_kwargs = client.calls[-3]
    daily_name, daily_kwargs = client.calls[-2]
    weight_name, weight_kwargs = client.calls[-1]
    assert basic_name == "index_basic"
    assert basic_kwargs["market"] == "CSI"
    assert {"ts_code", "name", "market", "publisher", "category"} <= set(str(basic_kwargs["fields"]).split(","))
    assert daily_name == "index_daily"
    assert daily_kwargs["ts_code"] == "000300.SH"
    assert daily_kwargs["start_date"] == "20260101"
    assert daily_kwargs["end_date"] == "20260430"
    assert {"ts_code", "trade_date", "close", "pct_chg"} <= set(str(daily_kwargs["fields"]).split(","))
    assert weight_name == "index_weight"
    assert weight_kwargs["index_code"] == "000300.SH"
    assert weight_kwargs["start_date"] == "20260401"
    assert weight_kwargs["end_date"] == "20260430"
    assert {"index_code", "con_code", "trade_date", "weight"} <= set(str(weight_kwargs["fields"]).split(","))


def test_stk_factor_pro_default_fields_include_matching_adjust_modes() -> None:
    client = _NoInitClient()

    client.fetch_stk_factor_pro_by_trade_date("2026-04-21")

    _api_name, kwargs = client.calls[-1]
    fields = set(str(kwargs["fields"]).split(","))
    assert {"rsi_qfq_6", "ma_qfq_20"} <= fields
    assert not any("_hfq" in field for field in fields)
    assert {"updays", "downdays", "topdays", "lowdays"} <= fields
    assert {"asi_qfq", "macd_qfq", "rsi_qfq_24", "xsii_td4_qfq"} <= fields


def test_stk_factor_pro_ts_code_range_interface_uses_stock_date_window() -> None:
    client = _NoInitClient()

    client.fetch_stk_factor_pro_by_ts_code(
        ts_code="000002.SZ",
        start_date="2025-08-01",
        end_date="2025-08-07",
        fields="ts_code,trade_date,dpo_qfq,macd_qfq",
    )

    api_name, kwargs = client.calls[-1]
    assert api_name == "stk_factor_pro"
    assert kwargs["ts_code"] == "000002.SZ"
    assert kwargs["start_date"] == "20250801"
    assert kwargs["end_date"] == "20250807"
    assert kwargs["fields"] == "ts_code,trade_date,dpo_qfq,macd_qfq"


def test_stk_factor_pro_ts_code_range_default_omits_fields_for_vendor_defaults() -> None:
    client = _NoInitClient()

    client.fetch_stk_factor_pro_by_ts_code(
        ts_code="000002.SZ",
        start_date="2025-08-01",
        end_date="2025-08-07",
    )

    api_name, kwargs = client.calls[-1]
    assert api_name == "stk_factor_pro"
    assert kwargs["ts_code"] == "000002.SZ"
    assert kwargs["start_date"] == "20250801"
    assert kwargs["end_date"] == "20250807"
    assert "fields" not in kwargs


def test_stk_factor_pro_falls_back_to_field_chunks_after_wide_request_failure() -> None:
    class _ChunkFallbackClient(_NoInitClient):
        def _call(self, api_name: str, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append((api_name, kwargs))
            fields = str(kwargs.get("fields", ""))
            if "asi_qfq" in fields and "xsii_td4_qfq" in fields:
                raise RuntimeError("wide request failed")
            payload = {"ts_code": ["000001.SZ"], "trade_date": ["20260421"]}
            for field in fields.split(","):
                field = field.strip()
                if field not in payload:
                    payload[field] = [1.0]
            return pd.DataFrame(payload)

    client = _ChunkFallbackClient()
    out = client.fetch_stk_factor_pro_by_trade_date("2026-04-21")

    assert len(client.calls) > 1
    assert all(call[0] == "stk_factor_pro" for call in client.calls)
    assert "asi_qfq" in out.columns
    assert "xsii_td4_qfq" in out.columns
    assert out.shape[0] == 1


def test_stk_factor_pro_recursively_splits_failed_field_chunks_and_skips_bad_single_field() -> None:
    class _RecursiveFallbackClient(_NoInitClient):
        def _call(self, api_name: str, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append((api_name, kwargs))
            fields = [x.strip() for x in str(kwargs.get("fields", "")).split(",") if x.strip()]
            payload_fields = [x for x in fields if x not in {"ts_code", "trade_date"}]
            if len(payload_fields) > 2:
                raise RuntimeError("chunk too wide")
            if "bad_qfq" in payload_fields:
                raise RuntimeError("single field unavailable")
            payload = {"ts_code": ["000001.SZ"], "trade_date": ["20260421"]}
            for field in payload_fields:
                payload[field] = [1.0]
            return pd.DataFrame(payload)

    client = _RecursiveFallbackClient()
    out = client.fetch_stk_factor_pro_by_trade_date(
        "2026-04-21",
        fields="ts_code,trade_date,asi_qfq,macd_qfq,bad_qfq,rsi_qfq_6,xsii_td4_qfq",
    )

    assert "asi_qfq" in out.columns
    assert "macd_qfq" in out.columns
    assert "rsi_qfq_6" in out.columns
    assert "xsii_td4_qfq" in out.columns
    assert "bad_qfq" not in out.columns
    assert out.attrs["skipped_stk_factor_pro_fields"] == ["bad_qfq"]
    assert len(client.calls) > 3


def test_finance_vip_interfaces_request_project_fields() -> None:
    client = _NoInitClient()

    client.fetch_income_vip(start_date="2026-01-01", end_date="2026-04-30")
    client.fetch_balancesheet_vip(start_date="2026-01-01", end_date="2026-04-30")
    client.fetch_cashflow_vip(start_date="2026-01-01", end_date="2026-04-30")
    client.fetch_fina_indicator_vip(start_date="2026-01-01", end_date="2026-04-30")

    calls = {name: kwargs for name, kwargs in client.calls[-4:]}
    assert {
        "income_vip",
        "balancesheet_vip",
        "cashflow_vip",
        "fina_indicator_vip",
    } == set(calls)
    assert {
        "ts_code",
        "ann_date",
        "end_date",
        "total_revenue",
        "n_income_attr_p",
    } <= set(str(calls["income_vip"]["fields"]).split(","))
    assert {"ts_code", "ann_date", "end_date", "total_assets", "total_liab"} <= set(
        str(calls["balancesheet_vip"]["fields"]).split(",")
    )
    assert {"ts_code", "ann_date", "end_date", "n_cashflow_act"} <= set(str(calls["cashflow_vip"]["fields"]).split(","))
    indicator_fields = set(str(calls["fina_indicator_vip"]["fields"]).split(","))
    assert {
        "ts_code",
        "ann_date",
        "end_date",
        "current_ratio",
        "quick_ratio",
        "assets_turn",
        "roe",
        "q_roe",
        "cfps_yoy",
        "ocf_yoy",
    } <= indicator_fields
