from __future__ import annotations

import pandas as pd

from alpha_mining.datasource.curation import (
    aggregate_cyq_chips_daily,
    curate_cyq_chips,
    curate_cyq_perf,
    curate_finance_fina_indicator_vip,
    curate_index_basic,
    curate_index_daily,
    curate_moneyflow,
    curate_report_rc_daily,
    curate_report_rc_detail,
    curate_stk_auction_c,
    curate_stk_auction_o,
    curate_stk_factor_pro,
)


def test_curate_index_basic_and_daily_normalize_index_lake_fields() -> None:
    basic = curate_index_basic(
        pd.DataFrame(
            {
                "ts_code": ["000300.SH"],
                "name": ["沪深300"],
                "market": ["CSI"],
                "publisher": ["中证公司"],
                "category": ["规模指数"],
                "base_date": ["20041231"],
                "base_point": [1000.0],
                "list_date": ["20050408"],
                "weight_rule": ["市值加权"],
                "desc": ["sample"],
                "exp_date": [""],
            }
        ),
        snapshot_date="2026-04-30",
    )
    daily = curate_index_daily(
        pd.DataFrame(
            {
                "ts_code": ["000300.SH"],
                "trade_date": ["20260430"],
                "open": [4000.0],
                "high": [4020.0],
                "low": [3990.0],
                "close": [4010.0],
                "pre_close": [4000.0],
                "change": [10.0],
                "pct_chg": [0.25],
                "vol": [100.0],
                "amount": [200.0],
            }
        )
    )

    assert basic.loc[0, "code"] == "000300.SH"
    assert basic.loc[0, "name"] == "沪深300"
    assert basic.loc[0, "market"] == "CSI"
    assert str(pd.Timestamp(basic.loc[0, "snapshot_date"]).date()) == "2026-04-30"
    assert daily.loc[0, "code"] == "000300.SH"
    assert str(pd.Timestamp(daily.loc[0, "date"]).date()) == "2026-04-30"
    assert daily.loc[0, "vendor_pct_chg_raw_pct"] == 0.25
    assert daily.loc[0, "pct_chg"] == 0.0025
    assert daily.loc[0, "return"] == 0.0025


def test_curate_moneyflow_units_prefixes_official_amount_fields() -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20260421"],
            "buy_sm_amount": [1.5],
            "sell_sm_amount": [0.5],
            "buy_md_amount": [2.5],
            "sell_md_amount": [1.5],
            "buy_lg_amount": [3.5],
            "sell_lg_amount": [2.5],
            "buy_elg_amount": [4.5],
            "sell_elg_amount": [3.5],
            "net_mf_amount": [2.0],
        }
    )

    out = curate_moneyflow(raw)

    assert out.loc[0, "moneyflow_buy_sm_amount"] == 15000.0
    assert out.loc[0, "moneyflow_sell_sm_amount"] == 5000.0
    assert out.loc[0, "moneyflow_buy_elg_amount"] == 45000.0
    assert out.loc[0, "moneyflow_sell_elg_amount"] == 35000.0
    assert out.loc[0, "moneyflow_net_mf_amount"] == 20000.0
    assert "moneyflow_net_mf_amount_rate" not in out.columns
    assert "moneyflow_net_mf_vol" not in out.columns


def test_curate_cyq_perf_and_auction_tables() -> None:
    cyq = curate_cyq_perf(
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20260421"],
                "winner_rate": [65.0],
                "cost_50pct": [10.0],
            }
        )
    )
    auction_o = curate_stk_auction_o(
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20260421"],
                "open": [10.0],
                "amount": [100.0],
            }
        )
    )
    auction_c = curate_stk_auction_c(
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20260421"],
                "close": [10.5],
                "vol": [20.0],
            }
        )
    )

    assert cyq.loc[0, "cyq_winner_rate"] == 0.65
    assert "cyq_his_low" in cyq.columns
    assert "cyq_his_high" in cyq.columns
    assert "cyq_cost_5pct" in cyq.columns
    assert "cyq_cost_15pct" in cyq.columns
    assert cyq.loc[0, "cyq_cost_50pct"] == 10.0
    assert "cyq_cost_85pct" in cyq.columns
    assert "cyq_cost_95pct" in cyq.columns
    assert "cyq_weight_avg" in cyq.columns
    assert auction_o.loc[0, "auction_o_open"] == 10.0
    assert auction_c.loc[0, "auction_c_close"] == 10.5


def test_curate_cyq_chips_preserves_long_distribution_and_ratio_units() -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "trade_date": ["20260421", "20260421", "20260421"],
            "price": [10.00, 10.00, 10.50],
            "percent": [0.50, 0.56, 1.00],
        }
    )

    out = curate_cyq_chips(raw)

    assert list(out.columns) == [
        "date",
        "code",
        "chip_price",
        "chip_percent_raw_pct",
        "chip_percent",
    ]
    assert len(out) == 2
    row = out[out["chip_price"] == 10.00].iloc[0]
    assert row["code"] == "000001.SZ"
    assert str(pd.Timestamp(row["date"]).date()) == "2026-04-21"
    assert row["chip_percent_raw_pct"] == 0.56
    assert abs(row["chip_percent"] - 0.0056) < 1e-12


def test_aggregate_cyq_chips_daily_derives_weighted_distribution_features() -> None:
    chips = curate_cyq_chips(
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ", "000001.SZ"],
                "trade_date": ["20260421", "20260421", "20260421", "20260421"],
                "price": [9.0, 10.0, 11.0, 12.0],
                "percent": [10.0, 20.0, 30.0, 40.0],
            }
        )
    )

    out = aggregate_cyq_chips_daily(chips)

    assert list(out.columns) == [
        "date",
        "code",
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
    ]
    row = out.iloc[0]
    assert row["cyq_chip_price_count"] == 4
    assert row["cyq_chip_percent_sum"] == 1.0
    assert row["cyq_chip_mode_price"] == 12.0
    assert row["cyq_chip_mode_percent"] == 0.4
    assert row["cyq_chip_weight_avg_price"] == 11.0
    assert row["cyq_chip_cost_10pct"] == 9.0
    assert row["cyq_chip_cost_25pct"] == 10.0
    assert row["cyq_chip_cost_50pct"] == 11.0
    assert row["cyq_chip_cost_75pct"] == 12.0
    assert row["cyq_chip_cost_90pct"] == 12.0


def test_aggregate_cyq_chips_daily_keeps_count_when_weights_are_empty() -> None:
    chips = curate_cyq_chips(
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000001.SZ"],
                "trade_date": ["20260421", "20260421"],
                "price": [9.0, 10.0],
                "percent": [0.0, 0.0],
            }
        )
    )

    out = aggregate_cyq_chips_daily(chips)

    assert out.loc[0, "cyq_chip_price_count"] == 2
    assert out.loc[0, "cyq_chip_percent_sum"] == 0.0
    assert pd.isna(out.loc[0, "cyq_chip_weight_avg_price"])
    assert pd.isna(out.loc[0, "cyq_chip_cost_50pct"])


def test_curate_stk_factor_pro_uses_allowlist_and_excludes_leakage_fields() -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20260421"],
            "pct_chg": [1.2],
            "pct_chg_qfq": [1.2],
            "change": [0.1],
            "pre_close": [9.9],
            "close_qfq": [10.1],
            "open_qfq": [10.0],
            "rsi_bfq_6": [45.0],
            "rsi_qfq_6": [55.0],
            "rsi_hfq_6": [65.0],
            "ma_qfq_20": [10.2],
            "asi_qfq": [1.1],
            "macd_qfq": [0.2],
            "rsi_qfq_24": [61.0],
            "xsii_td4_qfq": [4.0],
            "updays": [3],
        }
    )

    out = curate_stk_factor_pro(raw)

    assert "tech_rsi_qfq_6" in out.columns
    assert "tech_ma_qfq_20" in out.columns
    assert "tech_asi_qfq" in out.columns
    assert "tech_macd_qfq" in out.columns
    assert "tech_rsi_qfq_24" in out.columns
    assert "tech_xsii_td4_qfq" in out.columns
    assert "tech_updays" in out.columns
    assert "tech_rsi_bfq_6" not in out.columns
    assert "tech_rsi_hfq_6" not in out.columns
    assert "pct_chg" not in out.columns
    assert "tech_pct_chg" not in out.columns
    assert "tech_pct_chg_qfq" not in out.columns
    assert "tech_pre_close" not in out.columns
    assert "tech_close_qfq" not in out.columns
    assert "tech_open_qfq" not in out.columns


def test_curate_stk_factor_pro_keeps_qfq_when_adjust_mode_is_hfq() -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20260421"],
            "rsi_bfq_6": [45.0],
            "rsi_qfq_6": [55.0],
            "rsi_hfq_6": [65.0],
            "ma_qfq_20": [10.2],
            "ma_hfq_20": [10.8],
            "asi_qfq": [1.1],
            "updays": [3],
        }
    )

    out = curate_stk_factor_pro(raw, adjust_mode="hfq")

    assert "tech_rsi_qfq_6" in out.columns
    assert "tech_ma_qfq_20" in out.columns
    assert "tech_asi_qfq" in out.columns
    assert "tech_updays" in out.columns
    assert "tech_rsi_bfq_6" not in out.columns
    assert "tech_rsi_hfq_6" not in out.columns
    assert "tech_ma_hfq_20" not in out.columns


def test_curate_stk_factor_pro_rejects_unsupported_adjust_mode() -> None:
    raw = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260421"], "rsi_qfq_6": [55.0]})

    try:
        curate_stk_factor_pro(raw, adjust_mode="bfq")
    except ValueError as exc:
        assert "Unsupported adjust_mode" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("curate_stk_factor_pro should reject unsupported adjust_mode")


def test_curate_finance_indicator_vip_keeps_requested_fields() -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "ann_date": ["20260430"],
            "end_date": ["20260331"],
            "current_ratio": [1.5],
            "quick_ratio": [1.2],
            "assets_turn": [0.4],
            "roe": [0.1],
            "q_roe": [0.03],
            "cfps_yoy": [12.5],
            "ocf_yoy": [8.5],
        }
    )

    out = curate_finance_fina_indicator_vip(raw)

    assert out.loc[0, "code"] == "000001.SZ"
    assert out.loc[0, "current_ratio"] == 1.5
    assert out.loc[0, "quick_ratio"] == 1.2
    assert out.loc[0, "q_roe"] == 0.03
    assert out.loc[0, "cfps_yoy"] == 12.5
    assert out.loc[0, "ocf_yoy"] == 8.5


def test_curate_report_rc_detail_and_daily_aggregate() -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "report_date": ["20260421", "20260421"],
            "org_name": ["A", "B"],
            "author_name": ["aa", "bb"],
            "eps": [1.0, 1.4],
            "pe": [10.0, 12.0],
            "roe": [0.10, 0.12],
            "max_price": [20.0, 22.0],
            "min_price": [18.0, 19.0],
            "rating": ["买入", "增持"],
            "imp_dg": [80.0, 90.0],
        }
    )

    detail = curate_report_rc_detail(raw)
    daily = curate_report_rc_daily(raw)

    assert len(detail) == 2
    assert daily.loc[0, "report_rc_count"] == 2
    assert daily.loc[0, "report_rc_org_count"] == 2
    assert daily.loc[0, "report_rc_eps_mean"] == 1.2
    assert daily.loc[0, "report_rc_rating_score_mean"] == 4.5
