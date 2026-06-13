from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .finance_fields import (
    FINANCE_BALANCESHEET_VIP_FIELDS,
    FINANCE_CASHFLOW_VIP_FIELDS,
    FINANCE_INCOME_VIP_FIELDS,
    FINA_INDICATOR_VIP_FIELDS,
)


_ST_PREFIXES = ("ST", "*ST", "SST", "S*ST", "PT")


def curate_trade_cal(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=["date", "exchange", "is_open", "pretrade_date", "cal_date"])
    work = raw_df.copy()
    if "cal_date" in work.columns:
        work["date"] = pd.to_datetime(work["cal_date"], errors="coerce")
    elif "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
    else:
        raise ValueError("trade_cal requires 'cal_date' or 'date'")

    if "pretrade_date" in work.columns:
        work["pretrade_date"] = pd.to_datetime(work["pretrade_date"], errors="coerce")
    if "is_open" in work.columns:
        work["is_open"] = pd.to_numeric(work["is_open"], errors="coerce").fillna(0).astype(int)

    for col in ["exchange", "cal_date"]:
        if col not in work.columns:
            work[col] = ""
    out = work[["date", "exchange", "is_open", "pretrade_date", "cal_date"]].copy()
    return out.sort_values(["exchange", "date"], kind="mergesort").reset_index(drop=True)


def curate_stock_basic(raw_df: pd.DataFrame, snapshot_date: str | None = None) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(
            columns=[
                "code",
                "symbol",
                "name",
                "area",
                "industry",
                "market",
                "exchange",
                "curr_type",
                "list_status",
                "list_date",
                "delist_date",
                "is_hs",
                "snapshot_date",
            ]
        )
    work = raw_df.copy()
    if "ts_code" in work.columns and "code" not in work.columns:
        work = work.rename(columns={"ts_code": "code"})
    if "code" not in work.columns:
        raise ValueError("stock_basic requires 'ts_code' or 'code'")

    for col in ["list_date", "delist_date"]:
        if col in work.columns:
            work[col] = pd.to_datetime(work[col], errors="coerce")
        else:
            work[col] = pd.NaT

    out_cols = [
        "code",
        "symbol",
        "name",
        "area",
        "industry",
        "market",
        "exchange",
        "curr_type",
        "list_status",
        "list_date",
        "delist_date",
        "is_hs",
    ]
    for col in out_cols:
        if col not in work.columns:
            work[col] = np.nan if col not in {"code", "list_status"} else ""

    if snapshot_date is None:
        snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    work["snapshot_date"] = pd.to_datetime(snapshot_date, errors="coerce")

    out = work[out_cols + ["snapshot_date"]].copy()
    out = out.drop_duplicates(subset=["code", "list_status"], keep="last")
    return out.sort_values(["code", "list_status"], kind="mergesort").reset_index(drop=True)


def curate_index_classify(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(
            columns=[
                "index_code",
                "industry_name",
                "level",
                "parent_code",
                "src",
                "sw_level",
            ]
        )

    work = raw_df.copy()
    for col in ["index_code", "industry_name", "parent_code", "src"]:
        if col not in work.columns:
            work[col] = ""
    if "level" not in work.columns:
        work["level"] = np.nan

    level_num = pd.to_numeric(work["level"], errors="coerce")
    work["level"] = level_num.astype("Int64")
    level_map = {1: "L1", 2: "L2", 3: "L3"}
    work["sw_level"] = work["level"].map(level_map).fillna("UNKNOWN")

    out = work[["index_code", "industry_name", "level", "parent_code", "src", "sw_level"]].copy()
    out = out.drop_duplicates(subset=["index_code"], keep="last")
    return out.sort_values(["index_code"], kind="mergesort").reset_index(drop=True)


def curate_index_member_all(raw_df: pd.DataFrame, index_classify_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(
            columns=[
                "code",
                "index_code",
                "in_date",
                "out_date",
                "is_new",
                "sw_l1_code",
                "sw_l1_name",
                "sw_l2_code",
                "sw_l2_name",
                "sw_l3_code",
                "sw_l3_name",
                "sector",
                "industry",
                "subindustry",
            ]
        )

    work = raw_df.copy()
    if "con_code" in work.columns and "code" not in work.columns:
        work = work.rename(columns={"con_code": "code"})
    for col in ["code", "index_code", "in_date", "out_date", "is_new"]:
        if col not in work.columns:
            work[col] = np.nan

    work["in_date"] = pd.to_datetime(work["in_date"], errors="coerce")
    work["out_date"] = pd.to_datetime(work["out_date"], errors="coerce")

    hierarchy = _build_sw_hierarchy(index_classify_df)

    def _resolve(index_code: Any) -> dict[str, Any]:
        code = str(index_code or "").strip()
        if not code:
            return {
                "sw_l1_code": "",
                "sw_l1_name": "",
                "sw_l2_code": "",
                "sw_l2_name": "",
                "sw_l3_code": "",
                "sw_l3_name": "",
            }
        level = hierarchy.get(code, {}).get("level")
        parent = hierarchy.get(code, {}).get("parent_code", "")

        l1 = ""
        l2 = ""
        l3 = ""
        if level == 1:
            l1 = code
        elif level == 2:
            l2 = code
            l1 = parent
        elif level == 3:
            l3 = code
            l2 = parent
            l1 = hierarchy.get(l2, {}).get("parent_code", "")

        return {
            "sw_l1_code": l1,
            "sw_l1_name": hierarchy.get(l1, {}).get("name", ""),
            "sw_l2_code": l2,
            "sw_l2_name": hierarchy.get(l2, {}).get("name", ""),
            "sw_l3_code": l3,
            "sw_l3_name": hierarchy.get(l3, {}).get("name", ""),
        }

    resolved = work["index_code"].map(_resolve).apply(pd.Series)
    out = pd.concat([work, resolved], axis=1)
    out["sector"] = out["sw_l1_name"].fillna("")
    out["industry"] = out["sw_l2_name"].fillna("")
    out["subindustry"] = out["sw_l3_name"].fillna("")

    cols = [
        "code",
        "index_code",
        "in_date",
        "out_date",
        "is_new",
        "sw_l1_code",
        "sw_l1_name",
        "sw_l2_code",
        "sw_l2_name",
        "sw_l3_code",
        "sw_l3_name",
        "sector",
        "industry",
        "subindustry",
    ]
    out = out[cols].copy()
    return out.sort_values(["code", "in_date", "index_code"], kind="mergesort").reset_index(drop=True)


def curate_index_basic(raw_df: pd.DataFrame, snapshot_date: str | None = None) -> pd.DataFrame:
    cols = [
        "code",
        "name",
        "market",
        "publisher",
        "category",
        "base_date",
        "base_point",
        "list_date",
        "weight_rule",
        "description",
        "exp_date",
        "snapshot_date",
    ]
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=cols)

    work = raw_df.copy()
    if "ts_code" in work.columns and "code" not in work.columns:
        work = work.rename(columns={"ts_code": "code"})
    if "desc" in work.columns and "description" not in work.columns:
        work = work.rename(columns={"desc": "description"})
    if "code" not in work.columns:
        raise ValueError("index_basic requires 'ts_code' or 'code'")

    for col in [
        "code",
        "name",
        "market",
        "publisher",
        "category",
        "weight_rule",
        "description",
    ]:
        if col not in work.columns:
            work[col] = ""
        work[col] = work[col].fillna("").astype(str)
    for col in ["base_date", "list_date", "exp_date"]:
        work[col] = pd.to_datetime(work[col], errors="coerce") if col in work.columns else pd.NaT
    if "base_point" not in work.columns:
        work["base_point"] = np.nan
    work["base_point"] = pd.to_numeric(work["base_point"], errors="coerce")
    if snapshot_date is None:
        snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    work["snapshot_date"] = pd.to_datetime(snapshot_date, errors="coerce")

    out = work[cols].copy()
    out = out.drop_duplicates(subset=["code"], keep="last")
    return out.sort_values(["code"], kind="mergesort").reset_index(drop=True)


def curate_index_daily(raw_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "date",
        "code",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "vendor_pct_chg_raw_pct",
        "pct_chg",
        "return",
        "vol",
        "amount",
    ]
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=cols)

    work = _prepare_trade_date_common(raw_df, table_name="index_daily")
    for col in [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
    ]:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["vendor_pct_chg_raw_pct"] = work["pct_chg"]
    work["return"] = work["vendor_pct_chg_raw_pct"] / 100.0
    work["pct_chg"] = work["return"]

    out = work[cols].copy()
    out = out.drop_duplicates(subset=["date", "code"], keep="last")
    return out.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)


def curate_index_weight(raw_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["date", "index_code", "code", "weight", "weight_decimal", "is_member"]
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=cols)

    work = raw_df.copy()
    if "trade_date" in work.columns:
        work["date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    elif "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
    else:
        raise ValueError("index_weight requires trade_date/date")
    if "con_code" in work.columns and "code" not in work.columns:
        work = work.rename(columns={"con_code": "code"})
    if "index_code" not in work.columns:
        raise ValueError("index_weight requires index_code")
    if "code" not in work.columns:
        raise ValueError("index_weight requires con_code/code")
    if "weight" not in work.columns:
        work["weight"] = np.nan

    work["index_code"] = work["index_code"].fillna("").astype(str).str.strip().str.upper()
    work["code"] = work["code"].fillna("").astype(str).str.strip().str.upper()
    work["weight"] = pd.to_numeric(work["weight"], errors="coerce")
    work["weight_decimal"] = work["weight"] / 100.0
    work["is_member"] = 1
    work = work[(work["date"].notna()) & (work["index_code"] != "") & (work["code"] != "")].copy()

    out = work[cols].copy()
    out = out.drop_duplicates(subset=["date", "index_code", "code"], keep="last")
    return out.sort_values(["index_code", "date", "code"], kind="mergesort").reset_index(drop=True)


def curate_market_adj_factor(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=["date", "code", "adj_factor"])
    work = raw_df.copy()
    if "ts_code" in work.columns and "code" not in work.columns:
        work = work.rename(columns={"ts_code": "code"})
    if "trade_date" in work.columns and "date" not in work.columns:
        work["date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    elif "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
    else:
        raise ValueError("adj_factor requires trade_date/date column")

    if "adj_factor" not in work.columns:
        work["adj_factor"] = 1.0
    work["adj_factor"] = pd.to_numeric(work["adj_factor"], errors="coerce")
    out = work[["date", "code", "adj_factor"]].copy()
    out = out.drop_duplicates(subset=["date", "code"], keep="last")
    return out.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)


def curate_market_daily(
    daily_raw_df: pd.DataFrame,
    adj_factor_raw_df: pd.DataFrame,
    adjust_mode: str = "qfq",
) -> pd.DataFrame:
    adjust_mode_norm = str(adjust_mode or "qfq").strip().lower()
    if adjust_mode_norm not in {"qfq", "hfq"}:
        raise ValueError(f"Unsupported adjust_mode: {adjust_mode}")

    if daily_raw_df is None or daily_raw_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "code",
                "bfq_open",
                "bfq_high",
                "bfq_low",
                "bfq_close",
                "bfq_pre_close",
                "bfq_change",
                "vendor_pct_chg_raw_pct",
                "adj_factor",
                "qfq_open",
                "qfq_high",
                "qfq_low",
                "qfq_close",
                "hfq_open",
                "hfq_high",
                "hfq_low",
                "hfq_close",
                "open",
                "high",
                "low",
                "close",
                "ret_1d",
                "pct_chg",
                "vol_hand",
                "amount_k",
                "volume",
                "amount",
                "price_adjust_mode",
            ]
        )

    daily = daily_raw_df.copy()
    if "ts_code" in daily.columns and "code" not in daily.columns:
        daily = daily.rename(columns={"ts_code": "code"})
    if "trade_date" in daily.columns:
        daily["date"] = pd.to_datetime(daily["trade_date"], errors="coerce")
    elif "date" in daily.columns:
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    else:
        raise ValueError("daily data requires trade_date/date")

    price_cols = [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
    ]
    for col in price_cols:
        if col not in daily.columns:
            daily[col] = np.nan
        daily[col] = pd.to_numeric(daily[col], errors="coerce")

    adj = curate_market_adj_factor(adj_factor_raw_df)
    out = pd.merge(daily, adj, on=["date", "code"], how="left")
    out["adj_factor"] = pd.to_numeric(out["adj_factor"], errors="coerce").fillna(1.0)

    out = out.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)
    out["bfq_open"] = out["open"]
    out["bfq_high"] = out["high"]
    out["bfq_low"] = out["low"]
    out["bfq_close"] = out["close"]
    out["bfq_pre_close"] = out["pre_close"]
    out["bfq_change"] = out["change"]
    out["vendor_pct_chg_raw_pct"] = out["pct_chg"]

    max_adj = out.groupby("code", sort=False)["adj_factor"].transform("max")
    min_adj = out.groupby("code", sort=False)["adj_factor"].transform("min")
    qfq_ratio = out["adj_factor"] / max_adj.replace(0, np.nan)
    hfq_ratio = out["adj_factor"] / min_adj.replace(0, np.nan)

    for src, dst in [
        ("bfq_open", "qfq_open"),
        ("bfq_high", "qfq_high"),
        ("bfq_low", "qfq_low"),
        ("bfq_close", "qfq_close"),
        ("bfq_open", "hfq_open"),
        ("bfq_high", "hfq_high"),
        ("bfq_low", "hfq_low"),
        ("bfq_close", "hfq_close"),
    ]:
        ratio = qfq_ratio if dst.startswith("qfq_") else hfq_ratio
        out[dst] = pd.to_numeric(out[src], errors="coerce") * ratio

    if adjust_mode_norm == "qfq":
        out["open"] = out["qfq_open"]
        out["high"] = out["qfq_high"]
        out["low"] = out["qfq_low"]
        out["close"] = out["qfq_close"]
    else:
        out["open"] = out["hfq_open"]
        out["high"] = out["hfq_high"]
        out["low"] = out["hfq_low"]
        out["close"] = out["hfq_close"]

    out["ret_1d"] = out.groupby("code", sort=False)["close"].pct_change()
    out["pct_chg"] = out["ret_1d"]

    out["vol_hand"] = pd.to_numeric(out["vol"], errors="coerce")
    out["amount_k"] = pd.to_numeric(out["amount"], errors="coerce")
    out["volume"] = out["vol_hand"] * 100.0
    out["amount"] = out["amount_k"] * 1000.0
    out["price_adjust_mode"] = adjust_mode_norm

    cols = [
        "date",
        "code",
        "bfq_open",
        "bfq_high",
        "bfq_low",
        "bfq_close",
        "bfq_pre_close",
        "bfq_change",
        "vendor_pct_chg_raw_pct",
        "adj_factor",
        "qfq_open",
        "qfq_high",
        "qfq_low",
        "qfq_close",
        "hfq_open",
        "hfq_high",
        "hfq_low",
        "hfq_close",
        "open",
        "high",
        "low",
        "close",
        "ret_1d",
        "pct_chg",
        "vol_hand",
        "amount_k",
        "volume",
        "amount",
        "price_adjust_mode",
    ]
    return out[cols].sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)


def curate_market_daily_basic(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "code",
                "turnover_rate",
                "turnover_rate_f",
                "volume_ratio",
                "pe",
                "pe_ttm",
                "pb",
                "ps",
                "ps_ttm",
                "dv_ratio",
                "dv_ttm",
                "total_share",
                "float_share",
                "free_share",
                "total_mv_raw_wan",
                "circ_mv_raw_wan",
                "total_mv",
                "circ_mv",
            ]
        )

    work = raw_df.copy()
    if "ts_code" in work.columns and "code" not in work.columns:
        work = work.rename(columns={"ts_code": "code"})
    if "trade_date" in work.columns:
        work["date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    elif "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
    else:
        raise ValueError("daily_basic requires trade_date/date")

    numeric_cols = [
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe",
        "pe_ttm",
        "pb",
        "ps",
        "ps_ttm",
        "dv_ratio",
        "dv_ttm",
        "total_share",
        "float_share",
        "free_share",
        "total_mv",
        "circ_mv",
    ]
    for col in numeric_cols:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work["total_mv_raw_wan"] = work["total_mv"]
    work["circ_mv_raw_wan"] = work["circ_mv"]
    work["total_mv"] = work["total_mv_raw_wan"] * 10000.0
    work["circ_mv"] = work["circ_mv_raw_wan"] * 10000.0

    cols = [
        "date",
        "code",
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe",
        "pe_ttm",
        "pb",
        "ps",
        "ps_ttm",
        "dv_ratio",
        "dv_ttm",
        "total_share",
        "float_share",
        "free_share",
        "total_mv_raw_wan",
        "circ_mv_raw_wan",
        "total_mv",
        "circ_mv",
    ]
    out = work[cols].copy()
    out = out.drop_duplicates(subset=["date", "code"], keep="last")
    return out.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)


def curate_market_stk_limit(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=["date", "code", "pre_close", "up_limit", "down_limit"])

    work = raw_df.copy()
    if "ts_code" in work.columns and "code" not in work.columns:
        work = work.rename(columns={"ts_code": "code"})
    if "trade_date" in work.columns:
        work["date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    elif "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
    else:
        raise ValueError("stk_limit requires trade_date/date")

    for col in ["pre_close", "up_limit", "down_limit"]:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")

    out = work[["date", "code", "pre_close", "up_limit", "down_limit"]].copy()
    out = out.drop_duplicates(subset=["date", "code"], keep="last")
    return out.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)


def curate_market_suspend_d(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=["date", "code", "suspend_timing", "suspend_type", "is_suspended"])

    work = raw_df.copy()
    if "ts_code" in work.columns and "code" not in work.columns:
        work = work.rename(columns={"ts_code": "code"})
    if "trade_date" in work.columns:
        work["date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    elif "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
    else:
        raise ValueError("suspend_d requires trade_date/date")

    for col in ["suspend_timing", "suspend_type"]:
        if col not in work.columns:
            work[col] = ""
        work[col] = work[col].fillna("").astype(str)
    work["is_suspended"] = 1

    out = work[["date", "code", "suspend_timing", "suspend_type", "is_suspended"]].copy()
    out = out.drop_duplicates(subset=["date", "code"], keep="last")
    return out.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)


def curate_security_namechange(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(
            columns=[
                "code",
                "name",
                "normalized_name",
                "start_date",
                "end_date",
                "ann_date",
                "change_reason",
                "is_st",
            ]
        )

    work = raw_df.copy()
    if "ts_code" in work.columns and "code" not in work.columns:
        work = work.rename(columns={"ts_code": "code"})
    if "code" not in work.columns:
        raise ValueError("namechange requires ts_code/code")

    if "name" not in work.columns:
        work["name"] = ""
    work["name"] = work["name"].fillna("").astype(str)

    if "start_date" in work.columns:
        work["start_date"] = pd.to_datetime(work["start_date"], errors="coerce")
    elif "ann_date" in work.columns:
        work["start_date"] = pd.to_datetime(work["ann_date"], errors="coerce")
    else:
        work["start_date"] = pd.NaT

    if "end_date" in work.columns:
        work["end_date"] = pd.to_datetime(work["end_date"], errors="coerce")
    else:
        work["end_date"] = pd.NaT

    if "ann_date" in work.columns:
        work["ann_date"] = pd.to_datetime(work["ann_date"], errors="coerce")
    else:
        work["ann_date"] = pd.NaT

    if "change_reason" not in work.columns:
        work["change_reason"] = ""
    work["change_reason"] = work["change_reason"].fillna("").astype(str)

    work["normalized_name"] = work["name"].map(_normalize_security_name)
    work["is_st"] = work["normalized_name"].map(_is_st_name_prefix).astype(int)

    out = work[
        [
            "code",
            "name",
            "normalized_name",
            "start_date",
            "end_date",
            "ann_date",
            "change_reason",
            "is_st",
        ]
    ].copy()
    out = out.drop_duplicates(subset=["code", "name", "start_date", "end_date"], keep="last")
    return out.sort_values(["code", "start_date", "ann_date"], kind="mergesort").reset_index(drop=True)


def curate_moneyflow_ths(raw_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "date",
        "code",
        "moneyflow_name",
        "moneyflow_pct_change",
        "moneyflow_latest",
        "moneyflow_net_amount",
        "moneyflow_net_d5_amount",
        "moneyflow_buy_lg_amount",
        "moneyflow_buy_lg_amount_rate",
        "moneyflow_buy_md_amount",
        "moneyflow_buy_md_amount_rate",
        "moneyflow_buy_sm_amount",
        "moneyflow_buy_sm_amount_rate",
    ]
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=cols)

    work = raw_df.copy()
    if "ts_code" in work.columns and "code" not in work.columns:
        work = work.rename(columns={"ts_code": "code"})
    if "trade_date" in work.columns:
        work["date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    elif "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
    else:
        raise ValueError("moneyflow_ths requires trade_date/date")
    if "code" not in work.columns:
        raise ValueError("moneyflow_ths requires ts_code/code")

    work["moneyflow_name"] = work["name"].fillna("").astype(str) if "name" in work.columns else ""
    numeric_map = {
        "moneyflow_pct_change": ("pct_change", "pct_chg"),
        "moneyflow_latest": ("latest", "close"),
        "moneyflow_net_amount": ("net_amount",),
        "moneyflow_net_d5_amount": ("net_d5_amount",),
        "moneyflow_buy_lg_amount": ("buy_lg_amount",),
        "moneyflow_buy_lg_amount_rate": ("buy_lg_amount_rate",),
        "moneyflow_buy_md_amount": ("buy_md_amount",),
        "moneyflow_buy_md_amount_rate": ("buy_md_amount_rate",),
        "moneyflow_buy_sm_amount": ("buy_sm_amount",),
        "moneyflow_buy_sm_amount_rate": ("buy_sm_amount_rate",),
    }
    for target, candidates in numeric_map.items():
        _coalesce_numeric_column(work, target, candidates)
    for col in [
        "moneyflow_net_amount",
        "moneyflow_net_d5_amount",
        "moneyflow_buy_lg_amount",
        "moneyflow_buy_md_amount",
        "moneyflow_buy_sm_amount",
    ]:
        work[col] = pd.to_numeric(work[col], errors="coerce") * 10000.0
    for col in [
        "moneyflow_pct_change",
        "moneyflow_buy_lg_amount_rate",
        "moneyflow_buy_md_amount_rate",
        "moneyflow_buy_sm_amount_rate",
    ]:
        work[col] = pd.to_numeric(work[col], errors="coerce") / 100.0

    out = work[cols].copy()
    out = out.drop_duplicates(subset=["date", "code"], keep="last")
    return out.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)


def curate_moneyflow(raw_df: pd.DataFrame) -> pd.DataFrame:
    amount_cols = [
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
    out_cols = ["date", "code"] + [f"moneyflow_{c}" for c in amount_cols]
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=out_cols)

    work = _prepare_trade_date_common(raw_df, table_name="moneyflow")
    for col in amount_cols:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")
    for col in amount_cols:
        work[f"moneyflow_{col}"] = work[col] * 10000.0
    out = work[out_cols].copy()
    out = out.drop_duplicates(subset=["date", "code"], keep="last")
    return out.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)


def curate_cyq_perf(raw_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "his_low",
        "his_high",
        "cost_5pct",
        "cost_15pct",
        "cost_50pct",
        "cost_85pct",
        "cost_95pct",
        "weight_avg",
        "winner_rate",
    ]
    out_cols = ["date", "code"] + [f"cyq_{c}" for c in numeric_cols]
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=out_cols)
    work = _prepare_trade_date_common(raw_df, table_name="cyq_perf")
    for col in numeric_cols:
        if col not in work.columns:
            work[col] = np.nan
        values = pd.to_numeric(work[col], errors="coerce")
        if col == "winner_rate":
            values = values.where(values.abs() <= 1.0, values / 100.0)
        work[f"cyq_{col}"] = values
    out = work[out_cols].copy()
    out = out.drop_duplicates(subset=["date", "code"], keep="last")
    return out.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)


CYQ_CHIPS_LONG_COLUMNS: list[str] = [
    "date",
    "code",
    "chip_price",
    "chip_percent_raw_pct",
    "chip_percent",
]
CYQ_CHIPS_DAILY_COLUMNS: list[str] = [
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


def curate_cyq_chips(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=CYQ_CHIPS_LONG_COLUMNS)
    work = _prepare_trade_date_common(raw_df, table_name="cyq_chips")
    if "price" not in work.columns:
        work["price"] = np.nan
    if "percent" not in work.columns:
        work["percent"] = np.nan
    work["chip_price"] = pd.to_numeric(work["price"], errors="coerce")
    work["chip_percent_raw_pct"] = pd.to_numeric(work["percent"], errors="coerce")
    work["chip_percent"] = work["chip_percent_raw_pct"] / 100.0
    out = work[CYQ_CHIPS_LONG_COLUMNS].copy()
    out = out.dropna(subset=["date", "code", "chip_price"])
    out = out.drop_duplicates(subset=["date", "code", "chip_price"], keep="last")
    return out.sort_values(["code", "date", "chip_price"], kind="mergesort").reset_index(drop=True)


def aggregate_cyq_chips_daily(chips_df: pd.DataFrame) -> pd.DataFrame:
    if chips_df is None or chips_df.empty:
        return pd.DataFrame(columns=CYQ_CHIPS_DAILY_COLUMNS)
    work = chips_df.copy()
    if "date" not in work.columns or "code" not in work.columns:
        work = curate_cyq_chips(work)
    for col in ["chip_price", "chip_percent"]:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["code"] = work["code"].fillna("").astype(str).str.strip().str.upper()
    work = work[(work["date"].notna()) & (work["code"] != "") & (work["chip_price"].notna())].copy()
    if work.empty:
        return pd.DataFrame(columns=CYQ_CHIPS_DAILY_COLUMNS)

    rows: list[dict[str, Any]] = []
    for (dt, code), group in work.groupby(["date", "code"], dropna=False, sort=True):
        g = group.sort_values("chip_price", kind="mergesort").copy()
        prices = pd.to_numeric(g["chip_price"], errors="coerce")
        weights = pd.to_numeric(g["chip_percent"], errors="coerce")
        valid_price = prices.notna()
        valid_weights = weights.where(weights > 0)
        weight_sum = float(valid_weights.sum(skipna=True)) if valid_weights.notna().any() else 0.0
        count = int(valid_price.sum())
        row: dict[str, Any] = {
            "date": dt,
            "code": code,
            "cyq_chip_price_count": count,
            "cyq_chip_percent_sum": float(weights.sum(skipna=True)) if weights.notna().any() else 0.0,
            "cyq_chip_price_min": float(prices.min(skipna=True)) if valid_price.any() else np.nan,
            "cyq_chip_price_max": float(prices.max(skipna=True)) if valid_price.any() else np.nan,
            "cyq_chip_mode_price": np.nan,
            "cyq_chip_mode_percent": np.nan,
            "cyq_chip_weight_avg_price": np.nan,
            "cyq_chip_price_std": float(prices.std(skipna=True)) if count > 1 else np.nan,
            "cyq_chip_cost_10pct": np.nan,
            "cyq_chip_cost_25pct": np.nan,
            "cyq_chip_cost_50pct": np.nan,
            "cyq_chip_cost_75pct": np.nan,
            "cyq_chip_cost_90pct": np.nan,
        }
        if weight_sum > 0:
            mode_idx = valid_weights.idxmax()
            row["cyq_chip_mode_price"] = float(g.loc[mode_idx, "chip_price"])
            row["cyq_chip_mode_percent"] = float(g.loc[mode_idx, "chip_percent"])
            row["cyq_chip_weight_avg_price"] = float((prices * valid_weights.fillna(0.0)).sum() / weight_sum)
            normalized = valid_weights.fillna(0.0) / weight_sum
            cumulative = normalized.cumsum()
            for pct in (10, 25, 50, 75, 90):
                row[f"cyq_chip_cost_{pct}pct"] = _weighted_quantile_from_cdf(
                    prices=prices,
                    cumulative=cumulative,
                    threshold=float(pct) / 100.0,
                )
        rows.append(row)
    out = pd.DataFrame(rows)
    return out[CYQ_CHIPS_DAILY_COLUMNS].sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)


def _weighted_quantile_from_cdf(prices: pd.Series, cumulative: pd.Series, threshold: float) -> float:
    valid = prices.notna() & cumulative.notna()
    if not valid.any():
        return float("nan")
    idx = cumulative[valid].ge(float(threshold)).idxmax()
    return float(prices.loc[idx])


_STK_FACTOR_PRO_EXCLUDE_FIELDS = {
    "pct_chg",
    "change",
    "pre_close",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
    "adj_factor",
    "total_mv",
    "circ_mv",
}
_STK_FACTOR_PRO_EXACT_ALLOWLIST = {"updays", "downdays", "topdays", "lowdays"}


def _stk_factor_pro_base_name(name: str) -> str:
    out = str(name or "")
    for marker in ("_bfq", "_qfq", "_hfq"):
        out = out.replace(marker, "")
    return out


def curate_stk_factor_pro(raw_df: pd.DataFrame, adjust_mode: str = "qfq") -> pd.DataFrame:
    adjust_mode_norm = str(adjust_mode or "qfq").strip().lower()
    if adjust_mode_norm not in {"qfq", "hfq"}:
        raise ValueError(f"Unsupported adjust_mode: {adjust_mode}")
    # stk_factor_pro exposes many duplicated price-adjusted variants. The local
    # project keeps qfq technical indicators only, independent of the global
    # price adjust mode used for OHLC prices.
    adjust_marker = "_qfq"
    base_cols = ["date", "code"]
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=base_cols)
    work = _prepare_trade_date_common(raw_df, table_name="stk_factor_pro")
    feature_cols: list[str] = []
    for col in work.columns:
        name = str(col)
        if name in {"date", "code", "ts_code", "trade_date"}:
            continue
        if name in _STK_FACTOR_PRO_EXCLUDE_FIELDS or _stk_factor_pro_base_name(name) in _STK_FACTOR_PRO_EXCLUDE_FIELDS:
            continue
        if adjust_marker in name or name in _STK_FACTOR_PRO_EXACT_ALLOWLIST:
            feature_cols.append(name)

    for col in feature_cols:
        work[f"tech_{col}"] = pd.to_numeric(work[col], errors="coerce")
    out_cols = base_cols + [f"tech_{c}" for c in feature_cols]
    out = work[out_cols].copy()
    out = out.drop_duplicates(subset=["date", "code"], keep="last")
    return out.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)


def curate_stk_auction_o(raw_df: pd.DataFrame) -> pd.DataFrame:
    return _curate_stk_auction(raw_df=raw_df, prefix="auction_o", table_name="stk_auction_o")


def curate_stk_auction_c(raw_df: pd.DataFrame) -> pd.DataFrame:
    return _curate_stk_auction(raw_df=raw_df, prefix="auction_c", table_name="stk_auction_c")


def curate_report_rc_detail(raw_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "date",
        "code",
        "report_rc_org_name",
        "report_rc_author_name",
        "report_rc_eps",
        "report_rc_pe",
        "report_rc_roe",
        "report_rc_max_price",
        "report_rc_min_price",
        "report_rc_rating",
        "report_rc_rating_score",
        "report_rc_imp_dg",
    ]
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=cols)
    work = _prepare_report_rc_common(raw_df)
    work["report_rc_org_name"] = work.get("org_name", "").fillna("").astype(str) if "org_name" in work.columns else ""
    work["report_rc_author_name"] = (
        work.get("author_name", "").fillna("").astype(str) if "author_name" in work.columns else ""
    )
    work["report_rc_rating"] = work.get("rating", "").fillna("").astype(str) if "rating" in work.columns else ""
    for col in ["eps", "pe", "roe", "max_price", "min_price", "imp_dg"]:
        if col not in work.columns:
            work[col] = np.nan
        work[f"report_rc_{col}"] = pd.to_numeric(work[col], errors="coerce")
    work["report_rc_rating_score"] = work["report_rc_rating"].map(_rating_score)
    return work[cols].copy().sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)


def curate_report_rc_daily(raw_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "date",
        "code",
        "report_rc_count",
        "report_rc_org_count",
        "report_rc_author_count",
        "report_rc_eps_mean",
        "report_rc_eps_median",
        "report_rc_eps_max",
        "report_rc_eps_min",
        "report_rc_pe_mean",
        "report_rc_roe_mean",
        "report_rc_target_price_mean",
        "report_rc_target_price_max",
        "report_rc_target_price_min",
        "report_rc_rating_score_mean",
        "report_rc_rating_score_max",
        "report_rc_imp_dg_mean",
    ]
    detail = curate_report_rc_detail(raw_df)
    if detail.empty:
        return pd.DataFrame(columns=cols)
    work = detail.copy()
    work["target_price"] = work[["report_rc_max_price", "report_rc_min_price"]].mean(axis=1, skipna=True)
    grouped = work.groupby(["date", "code"], dropna=False, sort=True)
    out = grouped.agg(
        report_rc_count=("code", "size"),
        report_rc_org_count=(
            "report_rc_org_name",
            lambda s: int(s.replace("", np.nan).dropna().nunique()),
        ),
        report_rc_author_count=(
            "report_rc_author_name",
            lambda s: int(s.replace("", np.nan).dropna().nunique()),
        ),
        report_rc_eps_mean=("report_rc_eps", "mean"),
        report_rc_eps_median=("report_rc_eps", "median"),
        report_rc_eps_max=("report_rc_eps", "max"),
        report_rc_eps_min=("report_rc_eps", "min"),
        report_rc_pe_mean=("report_rc_pe", "mean"),
        report_rc_roe_mean=("report_rc_roe", "mean"),
        report_rc_target_price_mean=("target_price", "mean"),
        report_rc_target_price_max=("report_rc_max_price", "max"),
        report_rc_target_price_min=("report_rc_min_price", "min"),
        report_rc_rating_score_mean=("report_rc_rating_score", "mean"),
        report_rc_rating_score_max=("report_rc_rating_score", "max"),
        report_rc_imp_dg_mean=("report_rc_imp_dg", "mean"),
    ).reset_index()
    return out[cols].sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)


def curate_moneyflow_dc(raw_df: pd.DataFrame) -> pd.DataFrame:
    _ = raw_df
    raise NotImplementedError("moneyflow_dc is separate from moneyflow_ths and is not wired into P3")


def curate_ths_index(raw_df: pd.DataFrame, snapshot_date: str | None = None) -> pd.DataFrame:
    cols = [
        "ths_code",
        "ths_name",
        "member_count",
        "exchange",
        "list_date",
        "ths_type",
        "snapshot_date",
    ]
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=cols)

    work = raw_df.copy()
    if "ts_code" in work.columns and "ths_code" not in work.columns:
        work = work.rename(columns={"ts_code": "ths_code"})
    if "name" in work.columns and "ths_name" not in work.columns:
        work = work.rename(columns={"name": "ths_name"})
    if "count" in work.columns and "member_count" not in work.columns:
        work = work.rename(columns={"count": "member_count"})
    for col in ["ths_code", "ths_name", "exchange", "ths_type"]:
        if col not in work.columns:
            work[col] = ""
        work[col] = work[col].fillna("").astype(str)
    if "type" in raw_df.columns and "ths_type" in work.columns:
        work["ths_type"] = raw_df["type"].fillna("").astype(str)
    if "member_count" not in work.columns:
        work["member_count"] = np.nan
    work["member_count"] = pd.to_numeric(work["member_count"], errors="coerce")
    work["list_date"] = pd.to_datetime(work["list_date"], errors="coerce") if "list_date" in work.columns else pd.NaT
    if snapshot_date is None:
        snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    work["snapshot_date"] = pd.to_datetime(snapshot_date, errors="coerce")

    out = work[cols].copy()
    out = out.drop_duplicates(subset=["ths_code"], keep="last")
    return out.sort_values(["ths_code"], kind="mergesort").reset_index(drop=True)


def curate_ths_member(raw_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["ths_code", "code", "name", "weight", "in_date", "out_date", "is_new"]
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=cols)

    work = raw_df.copy()
    if "ts_code" in work.columns and "ths_code" not in work.columns:
        work = work.rename(columns={"ts_code": "ths_code"})
    if "con_code" in work.columns and "code" not in work.columns:
        work = work.rename(columns={"con_code": "code"})
    if "con_name" in work.columns and "name" not in work.columns:
        work = work.rename(columns={"con_name": "name"})
    for col in ["ths_code", "code", "name", "is_new"]:
        if col not in work.columns:
            work[col] = ""
        work[col] = work[col].fillna("").astype(str)
    if "weight" not in work.columns:
        work["weight"] = np.nan
    work["weight"] = pd.to_numeric(work["weight"], errors="coerce")
    work["in_date"] = pd.to_datetime(work["in_date"], errors="coerce") if "in_date" in work.columns else pd.NaT
    work["out_date"] = pd.to_datetime(work["out_date"], errors="coerce") if "out_date" in work.columns else pd.NaT

    out = work[cols].copy()
    out = out.drop_duplicates(subset=["ths_code", "code", "in_date"], keep="last")
    return out.sort_values(["ths_code", "code", "in_date"], kind="mergesort").reset_index(drop=True)


def curate_finance_income_vip(raw_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["code", "ann_date", "end_date", *FINANCE_INCOME_VIP_FIELDS]
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=cols)
    work = _prepare_finance_common(raw_df)
    _coalesce_numeric_column(work, "total_revenue", ("total_revenue", "revenue"))
    _coalesce_numeric_column(work, "revenue", ("revenue", "total_revenue"))
    _coalesce_numeric_column(work, "operate_profit", ("operate_profit",))
    _coalesce_numeric_column(work, "total_profit", ("total_profit",))
    _coalesce_numeric_column(work, "n_income_attr_p", ("n_income_attr_p", "n_income", "netprofit"))
    _coalesce_numeric_column(work, "basic_eps", ("basic_eps",))
    _coalesce_numeric_column(work, "diluted_eps", ("diluted_eps",))
    out = work[cols].copy()
    out = out.drop_duplicates(subset=["code", "ann_date", "end_date"], keep="last")
    return out.sort_values(["code", "ann_date", "end_date"], kind="mergesort").reset_index(drop=True)


def curate_finance_balancesheet_vip(raw_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["code", "ann_date", "end_date", *FINANCE_BALANCESHEET_VIP_FIELDS]
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=cols)
    work = _prepare_finance_common(raw_df)
    _coalesce_numeric_column(work, "total_assets", ("total_assets",))
    _coalesce_numeric_column(work, "total_liab", ("total_liab",))
    _coalesce_numeric_column(
        work,
        "total_hldr_eqy_exc_min_int",
        ("total_hldr_eqy_exc_min_int", "total_hldr_eqy_inc_min_int"),
    )
    out = work[cols].copy()
    out = out.drop_duplicates(subset=["code", "ann_date", "end_date"], keep="last")
    return out.sort_values(["code", "ann_date", "end_date"], kind="mergesort").reset_index(drop=True)


def curate_finance_cashflow_vip(raw_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["code", "ann_date", "end_date", *FINANCE_CASHFLOW_VIP_FIELDS]
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=cols)
    work = _prepare_finance_common(raw_df)
    _coalesce_numeric_column(work, "n_cashflow_act", ("n_cashflow_act", "n_cashflow_act"))
    _coalesce_numeric_column(work, "n_cashflow_inv_act", ("n_cashflow_inv_act",))
    _coalesce_numeric_column(work, "n_cash_flows_fnc_act", ("n_cash_flows_fnc_act",))
    out = work[cols].copy()
    out = out.drop_duplicates(subset=["code", "ann_date", "end_date"], keep="last")
    return out.sort_values(["code", "ann_date", "end_date"], kind="mergesort").reset_index(drop=True)


def curate_finance_fina_indicator_vip(raw_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["code", "ann_date", "end_date", *FINA_INDICATOR_VIP_FIELDS]
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=cols)
    work = _prepare_finance_common(raw_df)
    for col in FINA_INDICATOR_VIP_FIELDS:
        _coalesce_numeric_column(work, col, (col,))
    out = work[cols].copy()
    out = out.drop_duplicates(subset=["code", "ann_date", "end_date"], keep="last")
    return out.sort_values(["code", "ann_date", "end_date"], kind="mergesort").reset_index(drop=True)


def _prepare_finance_common(raw_df: pd.DataFrame) -> pd.DataFrame:
    work = raw_df.copy()
    if "ts_code" in work.columns and "code" not in work.columns:
        work = work.rename(columns={"ts_code": "code"})
    if "code" not in work.columns:
        raise ValueError("finance table requires ts_code/code")

    if "ann_date" not in work.columns:
        if "f_ann_date" in work.columns:
            work["ann_date"] = work["f_ann_date"]
        elif "end_date" in work.columns:
            work["ann_date"] = work["end_date"]
        else:
            work["ann_date"] = pd.NaT
    if "end_date" not in work.columns:
        if "period" in work.columns:
            work["end_date"] = work["period"]
        else:
            work["end_date"] = pd.NaT

    work["ann_date"] = pd.to_datetime(work["ann_date"], errors="coerce")
    work["end_date"] = pd.to_datetime(work["end_date"], errors="coerce")
    return work


def _prepare_trade_date_common(raw_df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    work = raw_df.copy()
    if "ts_code" in work.columns and "code" not in work.columns:
        work = work.rename(columns={"ts_code": "code"})
    if "trade_date" in work.columns:
        work["date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    elif "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
    else:
        raise ValueError(f"{table_name} requires trade_date/date")
    if "code" not in work.columns:
        raise ValueError(f"{table_name} requires ts_code/code")
    return work


def _curate_stk_auction(raw_df: pd.DataFrame, prefix: str, table_name: str) -> pd.DataFrame:
    numeric_cols = ["open", "high", "low", "close", "vol", "amount", "vwap"]
    out_cols = ["date", "code"] + [f"{prefix}_{c}" for c in numeric_cols]
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=out_cols)
    work = _prepare_trade_date_common(raw_df, table_name=table_name)
    for col in numeric_cols:
        if col not in work.columns:
            work[col] = np.nan
        work[f"{prefix}_{col}"] = pd.to_numeric(work[col], errors="coerce")
    out = work[out_cols].copy()
    out = out.drop_duplicates(subset=["date", "code"], keep="last")
    return out.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)


def _prepare_report_rc_common(raw_df: pd.DataFrame) -> pd.DataFrame:
    work = raw_df.copy()
    if "ts_code" in work.columns and "code" not in work.columns:
        work = work.rename(columns={"ts_code": "code"})
    if "report_date" in work.columns:
        work["date"] = pd.to_datetime(work["report_date"], errors="coerce")
    elif "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
    else:
        raise ValueError("report_rc requires report_date/date")
    if "code" not in work.columns:
        raise ValueError("report_rc requires ts_code/code")
    return work


def _rating_score(value: Any) -> float:
    text = str(value or "").strip().lower()
    if not text:
        return float("nan")
    if any(token in text for token in ["买", "buy", "涔"]):
        return 5.0
    if any(token in text for token in ["增", "outperform", "澧"]):
        return 4.0
    if any(token in text for token in ["持有", "hold", "中性"]):
        return 3.0
    if any(token in text for token in ["减", "underperform"]):
        return 2.0
    if any(token in text for token in ["卖", "sell"]):
        return 1.0
    return float("nan")


def _coalesce_numeric_column(work: pd.DataFrame, target: str, candidates: tuple[str, ...]) -> None:
    if target in work.columns:
        work[target] = pd.to_numeric(work[target], errors="coerce")
        return

    values = pd.Series(np.nan, index=work.index, dtype="float64")
    for col in candidates:
        if col not in work.columns:
            continue
        col_values = pd.to_numeric(work[col], errors="coerce")
        values = values.where(values.notna(), col_values)
    work[target] = values


def _build_sw_hierarchy(
    index_classify_df: pd.DataFrame | None,
) -> dict[str, dict[str, Any]]:
    if index_classify_df is None or index_classify_df.empty:
        return {}
    work = index_classify_df.copy()
    if "index_code" not in work.columns:
        return {}

    out: dict[str, dict[str, Any]] = {}
    for _, row in work.iterrows():
        code = str(row.get("index_code", "") or "").strip()
        if not code:
            continue
        level = pd.to_numeric(row.get("level", np.nan), errors="coerce")
        out[code] = {
            "name": str(row.get("industry_name", "") or "").strip(),
            "parent_code": str(row.get("parent_code", "") or "").strip(),
            "level": int(level) if pd.notna(level) else 0,
        }
    return out


def _normalize_security_name(name: Any) -> str:
    text = str(name or "").upper()
    return re.sub(r"[\s\u3000]+", "", text)


def _is_st_name_prefix(normalized_name: str) -> bool:
    text = str(normalized_name or "")
    for prefix in _ST_PREFIXES:
        if text.startswith(prefix):
            return True
    return False
