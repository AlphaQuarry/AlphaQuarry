from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from ..mining.factor_family import infer_factor_family


def build_field_catalog_dataframe(
    conn: Any,
    source_view: str,
    field_catalog_version: str = "v1",
    default_enabled_categories: Sequence[str] = (
        "price",
        "return",
        "liquidity",
        "valuation",
        "industry",
        "event",
    ),
    non_searchable_fields: Sequence[str] = (
        "date",
        "code",
        "universe",
        "tradable",
        "can_trade",
        "can_buy",
        "can_sell",
        "is_one_price_up_limit",
        "is_one_price_down_limit",
        "is_limit_up_close",
        "is_limit_down_close",
        "is_st",
        "is_suspended",
        "days_since_listed",
    ),
    date_range_source_views: Sequence[str] | None = None,
) -> pd.DataFrame:
    columns_df = _get_view_columns(conn=conn, source_view=source_view)
    if columns_df.empty:
        return pd.DataFrame(
            columns=[
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
                "factor_family",
                "field_role",
                "available_at",
                "preprocessing_policy",
                "leakage_safe",
                "field_catalog_version",
            ]
        )

    metadata = _field_metadata_map()
    enabled_categories = {str(x).strip().lower() for x in default_enabled_categories if str(x).strip()}
    blocked_fields = {str(x).strip() for x in non_searchable_fields if str(x).strip()}
    available_start, available_end = _infer_catalog_date_range(
        conn=conn,
        source_view=source_view,
        date_range_source_views=date_range_source_views,
    )

    rows: list[dict[str, Any]] = []
    for _, row in columns_df.iterrows():
        field_name = str(row.get("field_name", "") or "")
        dtype = str(row.get("dtype", "") or "")
        info = metadata.get(field_name) or _pattern_field_metadata(field_name)
        field_type = str(info.get("field_type", "SCALAR"))
        category = str(info.get("category", "other")).strip().lower() or "other"
        source_table = str(info.get("source_table", _default_source_table_for_field(field_name, source_view)))
        unit = str(info.get("unit", ""))
        description = str(info.get("description", ""))
        factor_family = str(
            info.get(
                "factor_family",
                infer_factor_family(field_name, category=category, source_table=source_table),
            )
        )
        field_role = _infer_field_role(field_name=field_name, field_type=field_type, category=category)
        available_at = _infer_available_at(field_name=field_name, category=category, source_table=source_table)
        preprocessing_policy = _infer_preprocessing_policy(
            field_type=field_type, category=category, field_name=field_name
        )

        is_default_enabled = bool(category in enabled_categories)
        is_searchable = bool(
            (field_name not in blocked_fields)
            and field_type in {"SCALAR", "GROUP"}
            and category not in {"identity", "metadata"}
        )
        leakage_safe = bool(is_searchable and field_role == "signal_input")

        rows.append(
            {
                "field_name": field_name,
                "field_type": field_type,
                "category": category,
                "source_table": source_table,
                "dtype": dtype,
                "unit": unit,
                "available_start": available_start,
                "available_end": available_end,
                "is_default_enabled": is_default_enabled,
                "is_searchable": is_searchable,
                "description": description,
                "factor_family": factor_family,
                "field_role": field_role,
                "available_at": available_at,
                "preprocessing_policy": preprocessing_policy,
                "leakage_safe": leakage_safe,
                "field_catalog_version": str(field_catalog_version),
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values("field_name", kind="mergesort").reset_index(drop=True)


def _get_view_columns(conn: Any, source_view: str) -> pd.DataFrame:
    schema_name, table_name = _split_view_name(source_view)
    if schema_name:
        query = (
            "SELECT column_name AS field_name, data_type AS dtype, ordinal_position "
            "FROM information_schema.columns WHERE table_schema = ? AND table_name = ? "
            "ORDER BY ordinal_position"
        )
        df = conn.execute(query, [schema_name, table_name]).fetchdf()
    else:
        query = (
            "SELECT column_name AS field_name, data_type AS dtype, ordinal_position "
            "FROM information_schema.columns WHERE table_name = ? "
            "ORDER BY ordinal_position"
        )
        df = conn.execute(query, [table_name]).fetchdf()
    if not df.empty:
        return df

    pragma_df = conn.execute(f"PRAGMA table_info('{table_name}')").fetchdf()
    if pragma_df.empty:
        return pd.DataFrame(columns=["field_name", "dtype", "ordinal_position"])
    return pd.DataFrame(
        {
            "field_name": pragma_df["name"].astype(str),
            "dtype": pragma_df["type"].astype(str),
            "ordinal_position": pragma_df["cid"],
        }
    ).sort_values("ordinal_position", kind="mergesort")


def _infer_view_date_range(conn: Any, source_view: str) -> tuple[str, str]:
    try:
        out = conn.execute(
            f"SELECT CAST(MIN(date) AS VARCHAR) AS min_date, CAST(MAX(date) AS VARCHAR) AS max_date FROM {_qident(source_view)}"
        ).fetchdf()
    except Exception:
        return "", ""
    if out.empty:
        return "", ""
    min_date = str(out.iloc[0].get("min_date", "") or "")
    max_date = str(out.iloc[0].get("max_date", "") or "")
    return min_date, max_date


def _infer_catalog_date_range(
    conn: Any,
    source_view: str,
    date_range_source_views: Sequence[str] | None = None,
) -> tuple[str, str]:
    candidates = _date_range_source_candidates(
        source_view=source_view,
        date_range_source_views=date_range_source_views,
    )
    for candidate in candidates:
        min_date, max_date = _infer_view_date_range(conn=conn, source_view=candidate)
        if min_date or max_date:
            return min_date, max_date
    return "", ""


def _date_range_source_candidates(
    source_view: str,
    date_range_source_views: Sequence[str] | None = None,
) -> list[str]:
    source = str(source_view or "").strip()
    out: list[str] = []
    if date_range_source_views is not None:
        out.extend(str(x).strip() for x in date_range_source_views if str(x).strip())
    elif source == "v_project_panel_cn_a":
        out.extend(["fact_market_daily", "v_project_market_daily_base"])
    out.append(source)

    deduped: list[str] = []
    for item in out:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def _split_view_name(source_view: str) -> tuple[str | None, str]:
    parts = [p for p in str(source_view or "").split(".") if p]
    if not parts:
        raise ValueError("source_view is empty")
    if len(parts) == 1:
        return None, parts[0]
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError(f"Unsupported source_view format: {source_view}")


def _qident(name: str) -> str:
    parts = [p for p in str(name).split(".") if p]
    quoted: list[str] = []
    for part in parts:
        quoted.append('"' + str(part).replace('"', '""') + '"')
    return ".".join(quoted)


def _default_source_table_for_field(field_name: str, source_view: str) -> str:
    field = str(field_name or "").strip()
    view = str(source_view or "").strip()
    if view == "v_project_panel_cn_a":
        if field.startswith("fin_"):
            return "v_project_financial_asof_daily"
        return "v_project_market_daily_base"
    return view


def _infer_field_role(*, field_name: str, field_type: str, category: str) -> str:
    field = str(field_name or "").strip().lower()
    kind = str(field_type or "").strip().upper()
    cat = str(category or "").strip().lower()
    if field in {"date", "trade_date", "code", "znz_code"}:
        return "identity"
    if field in {"pct_chg", "ret_1d"} or "return" in field or field in {"target", "label", "y"}:
        return "target_or_return"
    if field in {
        "universe",
        "tradable",
        "can_trade",
        "can_buy",
        "can_sell",
        "is_st",
        "is_suspended",
    } or cat in {"event", "metadata"}:
        return "filter_or_diagnostic"
    if kind == "GROUP" or cat in {"industry", "group"}:
        return "group"
    return "signal_input"


def _infer_available_at(*, field_name: str, category: str, source_table: str) -> str:
    field = str(field_name or "").strip().lower()
    cat = str(category or "").strip().lower()
    source = str(source_table or "").strip().lower()
    if field in {"date", "trade_date", "code", "znz_code"}:
        return "identity"
    if cat == "finance" or "financial" in source or field.startswith("fin_"):
        return "after_announcement_asof"
    if field in {"pct_chg", "ret_1d"} or "return" in field:
        return "close_to_close_return"
    if field.startswith("tech_"):
        return "post_close_derived"
    return "same_day_close_available"


def _infer_preprocessing_policy(*, field_type: str, category: str, field_name: str) -> str:
    kind = str(field_type or "").strip().upper()
    cat = str(category or "").strip().lower()
    field = str(field_name or "").strip().lower()
    if kind in {"GROUP", "EVENT"} or cat in {
        "identity",
        "metadata",
        "industry",
        "event",
    }:
        return "none"
    if field in {"date", "trade_date", "code", "znz_code"}:
        return "none"
    if field.startswith("fin_"):
        return "asof_then_expression_wrapper"
    if kind == "SCALAR":
        return "expression_wrapper:ts_backfill+winsorize"
    return "none"


def _pattern_field_metadata(field_name: str) -> dict[str, str]:
    field = str(field_name or "").strip()
    if field.startswith("fin_"):
        if field.endswith("_ann_date") or field.endswith("_end_date"):
            return {
                "field_type": "EVENT",
                "category": "finance",
                "source_table": "v_project_financial_asof_daily",
                "unit": "date",
                "description": "latest finance announcement/report date as-of trade date",
            }
        return {
            "field_type": "SCALAR",
            "category": "finance",
            "source_table": "v_project_financial_asof_daily",
            "unit": "",
            "description": "finance metric as-of trade date",
        }
    if field.startswith("cyq_chip_"):
        return {
            "field_type": "SCALAR",
            "category": "chip",
            "source_table": "v_project_market_daily_base",
            "unit": "ratio" if field.endswith("_percent") or field.endswith("_sum") else "",
            "description": "Tushare cyq_chips derived daily chip distribution feature",
        }
    if field.startswith("tech_"):
        return {
            "field_type": "SCALAR",
            "category": "technical",
            "source_table": "v_project_market_daily_base",
            "unit": "",
            "description": "Tushare stk_factor_pro technical indicator",
        }
    return {}


def _field_metadata_map() -> dict[str, dict[str, str]]:
    return {
        "date": {
            "field_type": "ID",
            "category": "identity",
            "source_table": "v_project_market_daily_base",
            "unit": "date",
            "description": "trade date",
        },
        "code": {
            "field_type": "ID",
            "category": "identity",
            "source_table": "v_project_market_daily_base",
            "unit": "",
            "description": "security code",
        },
        "open": {
            "field_type": "SCALAR",
            "category": "price",
            "source_table": "v_project_market_daily_base",
            "unit": "CNY",
            "description": "canonical adjusted open",
        },
        "high": {
            "field_type": "SCALAR",
            "category": "price",
            "source_table": "v_project_market_daily_base",
            "unit": "CNY",
            "description": "canonical adjusted high",
        },
        "low": {
            "field_type": "SCALAR",
            "category": "price",
            "source_table": "v_project_market_daily_base",
            "unit": "CNY",
            "description": "canonical adjusted low",
        },
        "close": {
            "field_type": "SCALAR",
            "category": "price",
            "source_table": "v_project_market_daily_base",
            "unit": "CNY",
            "description": "canonical adjusted close",
        },
        "ret_1d": {
            "field_type": "SCALAR",
            "category": "return",
            "source_table": "v_project_market_daily_base",
            "unit": "ratio",
            "description": "1-day return recomputed from adjusted close",
        },
        "pct_chg": {
            "field_type": "SCALAR",
            "category": "return",
            "source_table": "v_project_market_daily_base",
            "unit": "ratio",
            "description": "same as ret_1d in curated layer",
        },
        "volume": {
            "field_type": "SCALAR",
            "category": "liquidity",
            "source_table": "v_project_market_daily_base",
            "unit": "shares",
            "description": "volume standardized from vendor vol(hand)",
        },
        "amount": {
            "field_type": "SCALAR",
            "category": "liquidity",
            "source_table": "v_project_market_daily_base",
            "unit": "CNY",
            "description": "amount standardized from vendor amount(k CNY)",
        },
        "circ_mv": {
            "field_type": "SCALAR",
            "category": "valuation",
            "source_table": "v_project_market_daily_base",
            "unit": "CNY",
            "description": "circulation market cap standardized to CNY",
        },
        "total_mv": {
            "field_type": "SCALAR",
            "category": "valuation",
            "source_table": "v_project_market_daily_base",
            "unit": "CNY",
            "description": "total market cap standardized to CNY",
        },
        "sector": {
            "field_type": "GROUP",
            "category": "industry",
            "source_table": "v_project_market_daily_base",
            "unit": "",
            "description": "SW level-1 sector",
        },
        "industry": {
            "field_type": "GROUP",
            "category": "industry",
            "source_table": "v_project_market_daily_base",
            "unit": "",
            "description": "SW level-2 industry",
        },
        "subindustry": {
            "field_type": "GROUP",
            "category": "industry",
            "source_table": "v_project_market_daily_base",
            "unit": "",
            "description": "SW level-3 subindustry",
        },
        "up_limit": {
            "field_type": "SCALAR",
            "category": "event",
            "source_table": "v_project_market_daily_base",
            "unit": "CNY",
            "description": "daily upper price limit",
        },
        "down_limit": {
            "field_type": "SCALAR",
            "category": "event",
            "source_table": "v_project_market_daily_base",
            "unit": "CNY",
            "description": "daily lower price limit",
        },
        "is_suspended": {
            "field_type": "EVENT",
            "category": "event",
            "source_table": "v_project_market_daily_base",
            "unit": "0/1",
            "description": "suspend_d has record on trade date",
        },
        "moneyflow_buy_sm_amount": {
            "field_type": "SCALAR",
            "category": "moneyflow",
            "source_table": "v_project_market_daily_base",
            "unit": "CNY",
            "description": "Tushare moneyflow small-order buy amount",
        },
        "moneyflow_sell_sm_amount": {
            "field_type": "SCALAR",
            "category": "moneyflow",
            "source_table": "v_project_market_daily_base",
            "unit": "CNY",
            "description": "Tushare moneyflow small-order sell amount",
        },
        "moneyflow_buy_md_amount": {
            "field_type": "SCALAR",
            "category": "moneyflow",
            "source_table": "v_project_market_daily_base",
            "unit": "CNY",
            "description": "Tushare moneyflow medium-order buy amount",
        },
        "moneyflow_sell_md_amount": {
            "field_type": "SCALAR",
            "category": "moneyflow",
            "source_table": "v_project_market_daily_base",
            "unit": "CNY",
            "description": "Tushare moneyflow medium-order sell amount",
        },
        "moneyflow_buy_lg_amount": {
            "field_type": "SCALAR",
            "category": "moneyflow",
            "source_table": "v_project_market_daily_base",
            "unit": "CNY",
            "description": "Tushare moneyflow large-order buy amount",
        },
        "moneyflow_sell_lg_amount": {
            "field_type": "SCALAR",
            "category": "moneyflow",
            "source_table": "v_project_market_daily_base",
            "unit": "CNY",
            "description": "Tushare moneyflow large-order sell amount",
        },
        "moneyflow_buy_elg_amount": {
            "field_type": "SCALAR",
            "category": "moneyflow",
            "source_table": "v_project_market_daily_base",
            "unit": "CNY",
            "description": "Tushare moneyflow extra-large-order buy amount",
        },
        "moneyflow_sell_elg_amount": {
            "field_type": "SCALAR",
            "category": "moneyflow",
            "source_table": "v_project_market_daily_base",
            "unit": "CNY",
            "description": "Tushare moneyflow extra-large-order sell amount",
        },
        "moneyflow_net_mf_amount": {
            "field_type": "SCALAR",
            "category": "moneyflow",
            "source_table": "v_project_market_daily_base",
            "unit": "CNY",
            "description": "Tushare moneyflow net inflow amount",
        },
        "cyq_his_low": {
            "field_type": "SCALAR",
            "category": "chip",
            "source_table": "v_project_market_daily_base",
            "unit": "CNY",
            "description": "Tushare cyq_perf historical low price",
        },
        "cyq_his_high": {
            "field_type": "SCALAR",
            "category": "chip",
            "source_table": "v_project_market_daily_base",
            "unit": "CNY",
            "description": "Tushare cyq_perf historical high price",
        },
        "cyq_cost_5pct": {
            "field_type": "SCALAR",
            "category": "chip",
            "source_table": "v_project_market_daily_base",
            "unit": "CNY",
            "description": "Tushare cyq_perf 5 percentile cost",
        },
        "cyq_cost_15pct": {
            "field_type": "SCALAR",
            "category": "chip",
            "source_table": "v_project_market_daily_base",
            "unit": "CNY",
            "description": "Tushare cyq_perf 15 percentile cost",
        },
        "cyq_cost_50pct": {
            "field_type": "SCALAR",
            "category": "chip",
            "source_table": "v_project_market_daily_base",
            "unit": "CNY",
            "description": "Tushare cyq_perf 50 percentile cost",
        },
        "cyq_cost_85pct": {
            "field_type": "SCALAR",
            "category": "chip",
            "source_table": "v_project_market_daily_base",
            "unit": "CNY",
            "description": "Tushare cyq_perf 85 percentile cost",
        },
        "cyq_cost_95pct": {
            "field_type": "SCALAR",
            "category": "chip",
            "source_table": "v_project_market_daily_base",
            "unit": "CNY",
            "description": "Tushare cyq_perf 95 percentile cost",
        },
        "cyq_weight_avg": {
            "field_type": "SCALAR",
            "category": "chip",
            "source_table": "v_project_market_daily_base",
            "unit": "CNY",
            "description": "Tushare cyq_perf weighted average cost",
        },
        "cyq_winner_rate": {
            "field_type": "SCALAR",
            "category": "chip",
            "source_table": "v_project_market_daily_base",
            "unit": "ratio",
            "description": "Tushare cyq_perf winner rate",
        },
        "tech_rsi_qfq_6": {
            "field_type": "SCALAR",
            "category": "technical",
            "source_table": "v_project_market_daily_base",
            "unit": "",
            "description": "Tushare stk_factor_pro RSI indicator",
        },
        "report_rc_eps_mean": {
            "field_type": "SCALAR",
            "category": "analyst",
            "source_table": "v_project_market_daily_base",
            "unit": "",
            "description": "daily mean analyst forecast EPS",
        },
        "is_st": {
            "field_type": "EVENT",
            "category": "event",
            "source_table": "v_project_market_daily_base",
            "unit": "0/1",
            "description": "namechange prefix indicates ST status",
        },
        "days_since_listed": {
            "field_type": "EVENT",
            "category": "event",
            "source_table": "v_project_market_daily_base",
            "unit": "trade_days",
            "description": "trading-day distance from list_date",
        },
        "tradable": {
            "field_type": "EVENT",
            "category": "filter",
            "source_table": "v_project_market_daily_base",
            "unit": "0/1",
            "description": "daily tradability mask",
        },
        "can_trade": {
            "field_type": "EVENT",
            "category": "filter",
            "source_table": "v_project_market_daily_base",
            "unit": "0/1",
            "description": "same-day tradability mask",
        },
        "can_buy": {
            "field_type": "EVENT",
            "category": "filter",
            "source_table": "v_project_market_daily_base",
            "unit": "0/1",
            "description": "directional buy eligibility",
        },
        "can_sell": {
            "field_type": "EVENT",
            "category": "filter",
            "source_table": "v_project_market_daily_base",
            "unit": "0/1",
            "description": "directional sell eligibility",
        },
        "is_one_price_up_limit": {
            "field_type": "EVENT",
            "category": "filter",
            "source_table": "v_project_market_daily_base",
            "unit": "0/1",
            "description": "open/high/low/close locked at upper limit",
        },
        "is_one_price_down_limit": {
            "field_type": "EVENT",
            "category": "filter",
            "source_table": "v_project_market_daily_base",
            "unit": "0/1",
            "description": "open/high/low/close locked at lower limit",
        },
        "is_limit_up_close": {
            "field_type": "EVENT",
            "category": "filter",
            "source_table": "v_project_market_daily_base",
            "unit": "0/1",
            "description": "close at upper limit",
        },
        "is_limit_down_close": {
            "field_type": "EVENT",
            "category": "filter",
            "source_table": "v_project_market_daily_base",
            "unit": "0/1",
            "description": "close at lower limit",
        },
        "universe": {
            "field_type": "EVENT",
            "category": "filter",
            "source_table": "v_project_market_daily_base",
            "unit": "0/1",
            "description": "research universe mask",
        },
        "price_adjust_mode": {
            "field_type": "EVENT",
            "category": "metadata",
            "source_table": "v_project_market_daily_base",
            "unit": "",
            "description": "canonical adjust mode",
        },
        "fin_income_ann_date": {
            "field_type": "EVENT",
            "category": "finance",
            "source_table": "v_project_financial_asof_daily",
            "unit": "date",
            "description": "latest income ann_date as-of trade date",
        },
        "fin_income_end_date": {
            "field_type": "EVENT",
            "category": "finance",
            "source_table": "v_project_financial_asof_daily",
            "unit": "date",
            "description": "latest income period end_date as-of trade date",
        },
        "fin_total_revenue": {
            "field_type": "SCALAR",
            "category": "finance",
            "source_table": "v_project_financial_asof_daily",
            "unit": "CNY",
            "description": "income total_revenue as-of trade date",
        },
        "fin_revenue": {
            "field_type": "SCALAR",
            "category": "finance",
            "source_table": "v_project_financial_asof_daily",
            "unit": "CNY",
            "description": "income revenue as-of trade date",
        },
        "fin_n_income_attr_p": {
            "field_type": "SCALAR",
            "category": "finance",
            "source_table": "v_project_financial_asof_daily",
            "unit": "CNY",
            "description": "income n_income_attr_p as-of trade date",
        },
        "fin_balance_ann_date": {
            "field_type": "EVENT",
            "category": "finance",
            "source_table": "v_project_financial_asof_daily",
            "unit": "date",
            "description": "latest balancesheet ann_date as-of trade date",
        },
        "fin_balance_end_date": {
            "field_type": "EVENT",
            "category": "finance",
            "source_table": "v_project_financial_asof_daily",
            "unit": "date",
            "description": "latest balancesheet end_date as-of trade date",
        },
        "fin_total_assets": {
            "field_type": "SCALAR",
            "category": "finance",
            "source_table": "v_project_financial_asof_daily",
            "unit": "CNY",
            "description": "balancesheet total_assets as-of trade date",
        },
        "fin_total_liab": {
            "field_type": "SCALAR",
            "category": "finance",
            "source_table": "v_project_financial_asof_daily",
            "unit": "CNY",
            "description": "balancesheet total_liab as-of trade date",
        },
        "fin_total_hldr_eqy_exc_min_int": {
            "field_type": "SCALAR",
            "category": "finance",
            "source_table": "v_project_financial_asof_daily",
            "unit": "CNY",
            "description": "balancesheet equity as-of trade date",
        },
        "fin_cashflow_ann_date": {
            "field_type": "EVENT",
            "category": "finance",
            "source_table": "v_project_financial_asof_daily",
            "unit": "date",
            "description": "latest cashflow ann_date as-of trade date",
        },
        "fin_cashflow_end_date": {
            "field_type": "EVENT",
            "category": "finance",
            "source_table": "v_project_financial_asof_daily",
            "unit": "date",
            "description": "latest cashflow end_date as-of trade date",
        },
        "fin_n_cashflow_act": {
            "field_type": "SCALAR",
            "category": "finance",
            "source_table": "v_project_financial_asof_daily",
            "unit": "CNY",
            "description": "cashflow n_cashflow_act as-of trade date",
        },
        "fin_indicator_ann_date": {
            "field_type": "EVENT",
            "category": "finance",
            "source_table": "v_project_financial_asof_daily",
            "unit": "date",
            "description": "latest indicator ann_date as-of trade date",
        },
        "fin_indicator_end_date": {
            "field_type": "EVENT",
            "category": "finance",
            "source_table": "v_project_financial_asof_daily",
            "unit": "date",
            "description": "latest indicator end_date as-of trade date",
        },
        "fin_roe": {
            "field_type": "SCALAR",
            "category": "finance",
            "source_table": "v_project_financial_asof_daily",
            "unit": "ratio",
            "description": "indicator ROE as-of trade date",
        },
        "fin_roa": {
            "field_type": "SCALAR",
            "category": "finance",
            "source_table": "v_project_financial_asof_daily",
            "unit": "ratio",
            "description": "indicator ROA as-of trade date",
        },
        "fin_grossprofit_margin": {
            "field_type": "SCALAR",
            "category": "finance",
            "source_table": "v_project_financial_asof_daily",
            "unit": "ratio",
            "description": "indicator grossprofit_margin as-of trade date",
        },
    }
