from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .config import LakePathSettings
from .duckdb_runtime import normalize_duckdb_connection_config
from .field_catalog_builder import build_field_catalog_dataframe
from .finance_fields import (
    FINANCE_ASOF_FIELD_MAP,
    FINANCE_ASOF_PANEL_FIELDS,
    FINANCE_BALANCESHEET_VIP_FIELDS,
    FINANCE_CASHFLOW_VIP_FIELDS,
    FINANCE_INCOME_VIP_FIELDS,
    FINA_INDICATOR_VIP_FIELDS,
)

STK_FACTOR_PRO_PANEL_FIELDS: tuple[str, ...] = (
    "tech_asi_qfq",
    "tech_asit_qfq",
    "tech_atr_qfq",
    "tech_bbi_qfq",
    "tech_bias1_qfq",
    "tech_bias2_qfq",
    "tech_bias3_qfq",
    "tech_boll_lower_qfq",
    "tech_boll_mid_qfq",
    "tech_boll_upper_qfq",
    "tech_brar_ar_qfq",
    "tech_brar_br_qfq",
    "tech_cci_qfq",
    "tech_cr_qfq",
    "tech_dfma_dif_qfq",
    "tech_dfma_difma_qfq",
    "tech_dmi_adx_qfq",
    "tech_dmi_adxr_qfq",
    "tech_dmi_mdi_qfq",
    "tech_dmi_pdi_qfq",
    "tech_dpo_qfq",
    "tech_madpo_qfq",
    "tech_ema_qfq_5",
    "tech_ema_qfq_10",
    "tech_ema_qfq_20",
    "tech_ema_qfq_30",
    "tech_ema_qfq_60",
    "tech_ema_qfq_90",
    "tech_emv_qfq",
    "tech_maemv_qfq",
    "tech_expma_12_qfq",
    "tech_expma_50_qfq",
    "tech_kdj_qfq",
    "tech_kdj_d_qfq",
    "tech_kdj_k_qfq",
    "tech_ktn_down_qfq",
    "tech_ktn_mid_qfq",
    "tech_ktn_upper_qfq",
    "tech_ma_qfq_20",
    "tech_macd_qfq",
    "tech_macd_dea_qfq",
    "tech_macd_dif_qfq",
    "tech_mass_qfq",
    "tech_ma_mass_qfq",
    "tech_mfi_qfq",
    "tech_mtm_qfq",
    "tech_mtmma_qfq",
    "tech_obv_qfq",
    "tech_psy_qfq",
    "tech_psyma_qfq",
    "tech_roc_qfq",
    "tech_maroc_qfq",
    "tech_rsi_qfq_6",
    "tech_rsi_qfq_24",
    "tech_taq_down_qfq",
    "tech_taq_mid_qfq",
    "tech_taq_up_qfq",
    "tech_trix_qfq",
    "tech_trma_qfq",
    "tech_vr_qfq",
    "tech_wr_qfq",
    "tech_wr1_qfq",
    "tech_xsii_td1_qfq",
    "tech_xsii_td2_qfq",
    "tech_xsii_td3_qfq",
    "tech_xsii_td4_qfq",
    "tech_updays",
    "tech_downdays",
    "tech_topdays",
    "tech_lowdays",
)

CYQ_CHIPS_DAILY_PANEL_FIELDS: tuple[str, ...] = (
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
)

PROJECT_BASE_HOT_COLUMNS: tuple[str, ...] = (
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "ret_1d",
    "pct_chg",
    "volume",
    "amount",
    "circ_mv",
    "total_mv",
    "sector",
    "industry",
    "subindustry",
    "up_limit",
    "down_limit",
    "is_suspended",
    "is_st",
    "days_since_listed",
    "tradable",
    "can_trade",
    "can_buy",
    "can_sell",
    "is_one_price_up_limit",
    "is_one_price_down_limit",
    "is_limit_up_close",
    "is_limit_down_close",
    "universe",
)


def _normalize_adjust_mode(adjust_mode: str) -> str:
    mode = str(adjust_mode or "qfq").strip().lower()
    if mode not in {"qfq", "hfq"}:
        raise ValueError(f"Unsupported adjust_mode: {adjust_mode}")
    return mode


def build_duckdb_catalog(
    paths: LakePathSettings,
    source_view: str = "v_project_panel_cn_a",
    field_catalog_version: str = "v1",
    adjust_mode: str = "qfq",
    universe_min_days_since_listed: int = 60,
    universe_exclude_st: bool = True,
    include_bj: bool = True,
    tradable_require_close: bool = True,
    tradable_require_positive_volume: bool = True,
    tradable_require_positive_amount: bool = True,
    materialize_project_base: bool = True,
    duckdb_settings: dict[str, Any] | None = None,
    field_catalog_enabled_categories: Sequence[str] = (
        "price",
        "return",
        "liquidity",
        "valuation",
        "industry",
        "event",
    ),
    field_catalog_non_searchable_fields: Sequence[str] = (
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
) -> dict[str, Any]:
    try:
        import duckdb  # type: ignore
    except Exception as exc:
        raise RuntimeError("duckdb is required but not installed") from exc

    db_path = paths.duckdb_path_obj
    db_path.parent.mkdir(parents=True, exist_ok=True)
    adjust_mode_norm = _normalize_adjust_mode(adjust_mode)

    effective_duckdb_settings = normalize_duckdb_connection_config(db_path, duckdb_settings or {})
    conn = duckdb.connect(str(db_path), config=effective_duckdb_settings)
    try:
        _create_base_views(conn=conn, paths=paths)
        conn.execute(
            _project_panel_sql(
                universe_min_days_since_listed=max(0, int(universe_min_days_since_listed)),
                universe_exclude_st=bool(universe_exclude_st),
                include_bj=bool(include_bj),
                tradable_require_close=bool(tradable_require_close),
                tradable_require_positive_volume=bool(tradable_require_positive_volume),
                tradable_require_positive_amount=bool(tradable_require_positive_amount),
                adjust_mode=adjust_mode_norm,
            )
        )
        _create_index_universe_project_views(conn=conn)
        materialized_base_out = _materialize_project_base(conn=conn, enabled=bool(materialize_project_base))
        if (
            str(source_view).strip()
            and str(source_view).strip() != "v_project_panel_cn_a"
            and not _relation_exists(conn=conn, relation_name=str(source_view).strip())
        ):
            conn.execute(f"CREATE OR REPLACE VIEW {_qident(source_view)} AS SELECT * FROM v_project_panel_cn_a")

        catalog_out = _refresh_field_catalog(
            conn=conn,
            paths=paths,
            source_view=source_view,
            field_catalog_version=field_catalog_version,
            field_catalog_enabled_categories=field_catalog_enabled_categories,
            field_catalog_non_searchable_fields=field_catalog_non_searchable_fields,
            date_range_source_views=_field_catalog_date_range_sources(source_view),
        )

        row_count, row_count_warning = _safe_view_row_count(
            conn=conn,
            view_name="v_project_panel_cn_a",
            fallback_table="fact_market_daily",
        )
        out = {
            "duckdb_path": str(db_path.as_posix()),
            "source_view": str(source_view),
            "field_catalog_path": str(catalog_out["field_catalog_path"]),
            "field_catalog_rows": int(catalog_out["field_catalog_rows"]),
            "project_rows": int(row_count),
            "materialized_project_base": materialized_base_out,
            "duckdb_settings": effective_duckdb_settings,
        }
        if row_count_warning:
            out["project_rows_warning"] = str(row_count_warning)
        return out
    finally:
        conn.close()


def refresh_duckdb_field_catalog(
    paths: LakePathSettings,
    source_view: str = "v_project_panel_cn_a",
    field_catalog_version: str = "v1",
    duckdb_settings: dict[str, Any] | None = None,
    field_catalog_enabled_categories: Sequence[str] = (
        "price",
        "return",
        "liquidity",
        "valuation",
        "industry",
        "event",
    ),
    field_catalog_non_searchable_fields: Sequence[str] = (
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
) -> dict[str, Any]:
    try:
        import duckdb  # type: ignore
    except Exception as exc:
        raise RuntimeError("duckdb is required but not installed") from exc

    db_path = paths.duckdb_path_obj
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")

    effective_duckdb_settings = normalize_duckdb_connection_config(db_path, duckdb_settings or {})
    conn = duckdb.connect(str(db_path), config=effective_duckdb_settings)
    try:
        out = _refresh_field_catalog(
            conn=conn,
            paths=paths,
            source_view=source_view,
            field_catalog_version=field_catalog_version,
            field_catalog_enabled_categories=field_catalog_enabled_categories,
            field_catalog_non_searchable_fields=field_catalog_non_searchable_fields,
            date_range_source_views=_field_catalog_date_range_sources(source_view),
        )
        out["duckdb_path"] = str(db_path.as_posix())
        out["source_view"] = str(source_view)
        out["duckdb_settings"] = effective_duckdb_settings
        return out
    finally:
        conn.close()


def _refresh_field_catalog(
    conn: Any,
    paths: LakePathSettings,
    source_view: str,
    field_catalog_version: str,
    field_catalog_enabled_categories: Sequence[str],
    field_catalog_non_searchable_fields: Sequence[str],
    date_range_source_views: Sequence[str] | None = None,
) -> dict[str, Any]:
    catalog_df = build_field_catalog_dataframe(
        conn=conn,
        source_view=str(source_view),
        field_catalog_version=str(field_catalog_version),
        default_enabled_categories=tuple(str(x) for x in field_catalog_enabled_categories),
        non_searchable_fields=tuple(str(x) for x in field_catalog_non_searchable_fields),
        date_range_source_views=date_range_source_views,
    )
    field_catalog_path = paths.meta_path / "field_catalog.parquet"
    field_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_df.to_parquet(field_catalog_path, index=False)
    conn.execute(
        "CREATE OR REPLACE VIEW v_project_field_catalog AS "
        f"SELECT * FROM read_parquet('{_escape_sql_path(field_catalog_path)}')"
    )
    return {
        "field_catalog_path": str(field_catalog_path.as_posix()),
        "field_catalog_rows": int(len(catalog_df)),
    }


def _field_catalog_date_range_sources(source_view: str) -> tuple[str, ...]:
    source = str(source_view or "").strip()
    candidates = ["fact_market_daily", "v_project_market_daily_base"]
    if source:
        candidates.append(source)
    out: list[str] = []
    for item in candidates:
        if item and item not in out:
            out.append(item)
    return tuple(out)


def _materialize_project_base(conn: Any, enabled: bool = True) -> dict[str, Any]:
    if not bool(enabled):
        try:
            conn.execute(
                "CREATE OR REPLACE VIEW v_project_market_daily_base_hot AS SELECT * FROM v_project_market_daily_base"
            )
        except Exception:
            pass
        return {"enabled": False, "table": "", "view": "v_project_market_daily_base"}

    try:
        hot_columns = _available_columns(
            conn=conn,
            view_name="v_project_market_daily_base",
            wanted=PROJECT_BASE_HOT_COLUMNS,
        )
        if not hot_columns:
            hot_columns = ["date", "code"]
        select_sql = ", ".join(_qident(col) for col in hot_columns)
        conn.execute(
            f"CREATE OR REPLACE TABLE project_market_daily_base AS SELECT {select_sql} FROM v_project_market_daily_base"
        )
        conn.execute(
            "CREATE OR REPLACE VIEW v_project_market_daily_base_hot AS SELECT * FROM project_market_daily_base"
        )
        row_count = int(conn.execute("SELECT COUNT(*) FROM project_market_daily_base").fetchone()[0])
        return {
            "enabled": True,
            "table": "project_market_daily_base",
            "view": "v_project_market_daily_base_hot",
            "rows": int(row_count),
            "columns": list(hot_columns),
        }
    except Exception as exc:
        warning = f"failed to materialize project_market_daily_base: {type(exc).__name__}: {exc}"
        print(f"[build_duckdb_catalog][warn] {warning}")
        try:
            conn.execute(
                "CREATE OR REPLACE VIEW v_project_market_daily_base_hot AS SELECT * FROM v_project_market_daily_base"
            )
        except Exception:
            pass
        return {
            "enabled": False,
            "table": "",
            "view": "v_project_market_daily_base",
            "warning": warning,
        }


def _safe_view_row_count(
    conn: Any,
    view_name: str,
    fallback_table: str = "fact_market_daily",
) -> tuple[int, str]:
    fallback_warning = ""
    fallback_name = str(fallback_table or "").strip()
    if fallback_name:
        try:
            row_count = int(conn.execute(f"SELECT COUNT(*) FROM {_qident(fallback_name)}").fetchone()[0])
            warning = f"project_rows approximated from '{fallback_name}' (faster than full panel count)"
            print(f"[build_duckdb_catalog][warn] {warning}")
            return row_count, warning
        except Exception as exc:
            fallback_warning = f"fallback table count failed for '{fallback_name}': {type(exc).__name__}: {exc}"

    try:
        row_count = int(conn.execute(f"SELECT COUNT(*) FROM {_qident(view_name)}").fetchone()[0])
        return row_count, fallback_warning
    except Exception as exc:
        warning = f"failed to compute row count for view '{view_name}': {type(exc).__name__}: {exc}"
        if fallback_warning:
            warning = f"{fallback_warning}; {warning}"
        print(f"[build_duckdb_catalog][warn] {warning}")
        return -1, warning


def _available_columns(conn: Any, view_name: str, wanted: Sequence[str]) -> list[str]:
    try:
        df = conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position",
            [str(view_name)],
        ).fetchdf()
    except Exception:
        return []
    available = set(df["column_name"].astype(str).tolist()) if not df.empty else set()
    return [str(col) for col in wanted if str(col) in available]


def _relation_exists(conn: Any, relation_name: str) -> bool:
    name = str(relation_name or "").strip()
    if not name:
        return False
    try:
        count = conn.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = ?
            """,
            [name],
        ).fetchone()[0]
        return int(count or 0) > 0
    except Exception:
        return False


def _create_base_views(conn: Any, paths: LakePathSettings) -> None:
    vendor_root = paths.vendor_raw_path
    curated_root = paths.curated_path

    view_specs = [
        (
            "raw_trade_cal",
            vendor_root / "trade_cal",
            {
                "exchange": "VARCHAR",
                "cal_date": "VARCHAR",
                "is_open": "INTEGER",
                "pretrade_date": "VARCHAR",
            },
        ),
        (
            "raw_stock_basic",
            vendor_root / "stock_basic",
            {
                "ts_code": "VARCHAR",
                "symbol": "VARCHAR",
                "name": "VARCHAR",
                "industry": "VARCHAR",
                "market": "VARCHAR",
                "list_status": "VARCHAR",
                "list_date": "VARCHAR",
                "delist_date": "VARCHAR",
            },
        ),
        (
            "raw_index_classify",
            vendor_root / "index_classify",
            {
                "index_code": "VARCHAR",
                "industry_name": "VARCHAR",
                "level": "INTEGER",
                "parent_code": "VARCHAR",
                "src": "VARCHAR",
            },
        ),
        (
            "raw_index_member_all",
            vendor_root / "index_member_all",
            {
                "index_code": "VARCHAR",
                "con_code": "VARCHAR",
                "in_date": "VARCHAR",
                "out_date": "VARCHAR",
                "is_new": "VARCHAR",
            },
        ),
        (
            "raw_daily",
            vendor_root / "daily",
            {
                "ts_code": "VARCHAR",
                "trade_date": "VARCHAR",
                "open": "DOUBLE",
                "high": "DOUBLE",
                "low": "DOUBLE",
                "close": "DOUBLE",
                "pre_close": "DOUBLE",
                "change": "DOUBLE",
                "pct_chg": "DOUBLE",
                "vol": "DOUBLE",
                "amount": "DOUBLE",
            },
        ),
        (
            "raw_daily_basic",
            vendor_root / "daily_basic",
            {
                "ts_code": "VARCHAR",
                "trade_date": "VARCHAR",
                "turnover_rate": "DOUBLE",
                "turnover_rate_f": "DOUBLE",
                "volume_ratio": "DOUBLE",
                "pe": "DOUBLE",
                "pe_ttm": "DOUBLE",
                "pb": "DOUBLE",
                "ps_ttm": "DOUBLE",
                "dv_ttm": "DOUBLE",
                "total_mv": "DOUBLE",
                "circ_mv": "DOUBLE",
            },
        ),
        (
            "raw_adj_factor",
            vendor_root / "adj_factor",
            {
                "ts_code": "VARCHAR",
                "trade_date": "VARCHAR",
                "adj_factor": "DOUBLE",
            },
        ),
        (
            "raw_stk_limit",
            vendor_root / "stk_limit",
            {
                "ts_code": "VARCHAR",
                "trade_date": "VARCHAR",
                "pre_close": "DOUBLE",
                "up_limit": "DOUBLE",
                "down_limit": "DOUBLE",
            },
        ),
        (
            "raw_suspend_d",
            vendor_root / "suspend_d",
            {
                "ts_code": "VARCHAR",
                "trade_date": "VARCHAR",
                "suspend_timing": "VARCHAR",
                "suspend_type": "VARCHAR",
            },
        ),
        (
            "raw_namechange",
            vendor_root / "namechange",
            {
                "ts_code": "VARCHAR",
                "name": "VARCHAR",
                "start_date": "VARCHAR",
                "end_date": "VARCHAR",
                "ann_date": "VARCHAR",
                "change_reason": "VARCHAR",
            },
        ),
        (
            "raw_moneyflow_ths",
            vendor_root / "moneyflow_ths",
            {
                "trade_date": "VARCHAR",
                "ts_code": "VARCHAR",
                "name": "VARCHAR",
                "pct_change": "DOUBLE",
                "latest": "DOUBLE",
                "net_amount": "DOUBLE",
                "net_d5_amount": "DOUBLE",
                "buy_lg_amount": "DOUBLE",
                "buy_lg_amount_rate": "DOUBLE",
                "buy_md_amount": "DOUBLE",
                "buy_md_amount_rate": "DOUBLE",
                "buy_sm_amount": "DOUBLE",
                "buy_sm_amount_rate": "DOUBLE",
            },
        ),
        (
            "raw_moneyflow",
            vendor_root / "moneyflow",
            {
                "ts_code": "VARCHAR",
                "trade_date": "VARCHAR",
                "buy_sm_amount": "DOUBLE",
                "sell_sm_amount": "DOUBLE",
                "buy_md_amount": "DOUBLE",
                "sell_md_amount": "DOUBLE",
                "buy_lg_amount": "DOUBLE",
                "sell_lg_amount": "DOUBLE",
                "buy_elg_amount": "DOUBLE",
                "sell_elg_amount": "DOUBLE",
                "net_mf_amount": "DOUBLE",
            },
        ),
        (
            "raw_cyq_perf",
            vendor_root / "cyq_perf",
            {
                "ts_code": "VARCHAR",
                "trade_date": "VARCHAR",
                "his_low": "DOUBLE",
                "his_high": "DOUBLE",
                "cost_5pct": "DOUBLE",
                "cost_15pct": "DOUBLE",
                "cost_50pct": "DOUBLE",
                "cost_85pct": "DOUBLE",
                "cost_95pct": "DOUBLE",
                "weight_avg": "DOUBLE",
                "winner_rate": "DOUBLE",
            },
        ),
        (
            "raw_cyq_chips",
            vendor_root / "cyq_chips",
            {
                "ts_code": "VARCHAR",
                "trade_date": "VARCHAR",
                "price": "DOUBLE",
                "percent": "DOUBLE",
            },
        ),
        (
            "raw_stk_factor_pro",
            vendor_root / "stk_factor_pro",
            {
                "ts_code": "VARCHAR",
                "trade_date": "VARCHAR",
                **{field.removeprefix("tech_"): "DOUBLE" for field in STK_FACTOR_PRO_PANEL_FIELDS},
            },
        ),
        (
            "raw_stk_auction_o",
            vendor_root / "stk_auction_o",
            {
                "ts_code": "VARCHAR",
                "trade_date": "VARCHAR",
                "open": "DOUBLE",
                "high": "DOUBLE",
                "low": "DOUBLE",
                "close": "DOUBLE",
                "vol": "DOUBLE",
                "amount": "DOUBLE",
                "vwap": "DOUBLE",
            },
        ),
        (
            "raw_stk_auction_c",
            vendor_root / "stk_auction_c",
            {
                "ts_code": "VARCHAR",
                "trade_date": "VARCHAR",
                "open": "DOUBLE",
                "high": "DOUBLE",
                "low": "DOUBLE",
                "close": "DOUBLE",
                "vol": "DOUBLE",
                "amount": "DOUBLE",
                "vwap": "DOUBLE",
            },
        ),
        (
            "raw_report_rc",
            vendor_root / "report_rc",
            {
                "ts_code": "VARCHAR",
                "report_date": "VARCHAR",
                "org_name": "VARCHAR",
                "author_name": "VARCHAR",
                "eps": "DOUBLE",
                "pe": "DOUBLE",
                "roe": "DOUBLE",
                "max_price": "DOUBLE",
                "min_price": "DOUBLE",
                "rating": "VARCHAR",
                "imp_dg": "DOUBLE",
            },
        ),
        (
            "raw_index_basic",
            vendor_root / "index_basic",
            {
                "ts_code": "VARCHAR",
                "name": "VARCHAR",
                "market": "VARCHAR",
                "publisher": "VARCHAR",
                "category": "VARCHAR",
                "base_date": "VARCHAR",
                "base_point": "DOUBLE",
                "list_date": "VARCHAR",
                "weight_rule": "VARCHAR",
                "desc": "VARCHAR",
                "exp_date": "VARCHAR",
            },
        ),
        (
            "raw_index_daily",
            vendor_root / "index_daily",
            {
                "ts_code": "VARCHAR",
                "trade_date": "VARCHAR",
                "open": "DOUBLE",
                "high": "DOUBLE",
                "low": "DOUBLE",
                "close": "DOUBLE",
                "pre_close": "DOUBLE",
                "change": "DOUBLE",
                "pct_chg": "DOUBLE",
                "vol": "DOUBLE",
                "amount": "DOUBLE",
            },
        ),
        (
            "raw_index_weight",
            vendor_root / "index_weight",
            {
                "index_code": "VARCHAR",
                "con_code": "VARCHAR",
                "trade_date": "VARCHAR",
                "weight": "DOUBLE",
            },
        ),
        (
            "raw_ths_index",
            vendor_root / "ths_index",
            {
                "ts_code": "VARCHAR",
                "name": "VARCHAR",
                "count": "DOUBLE",
                "exchange": "VARCHAR",
                "list_date": "VARCHAR",
                "type": "VARCHAR",
            },
        ),
        (
            "raw_ths_member",
            vendor_root / "ths_member",
            {
                "ts_code": "VARCHAR",
                "con_code": "VARCHAR",
                "con_name": "VARCHAR",
                "weight": "DOUBLE",
                "in_date": "VARCHAR",
                "out_date": "VARCHAR",
                "is_new": "VARCHAR",
            },
        ),
        (
            "raw_income_vip",
            vendor_root / "income_vip",
            {
                "ts_code": "VARCHAR",
                "ann_date": "VARCHAR",
                "end_date": "VARCHAR",
                **{field: "DOUBLE" for field in FINANCE_INCOME_VIP_FIELDS},
            },
        ),
        (
            "raw_balancesheet_vip",
            vendor_root / "balancesheet_vip",
            {
                "ts_code": "VARCHAR",
                "ann_date": "VARCHAR",
                "end_date": "VARCHAR",
                **{field: "DOUBLE" for field in FINANCE_BALANCESHEET_VIP_FIELDS},
            },
        ),
        (
            "raw_cashflow_vip",
            vendor_root / "cashflow_vip",
            {
                "ts_code": "VARCHAR",
                "ann_date": "VARCHAR",
                "end_date": "VARCHAR",
                **{field: "DOUBLE" for field in FINANCE_CASHFLOW_VIP_FIELDS},
            },
        ),
        (
            "raw_fina_indicator_vip",
            vendor_root / "fina_indicator_vip",
            {
                "ts_code": "VARCHAR",
                "ann_date": "VARCHAR",
                "end_date": "VARCHAR",
                **{field: "DOUBLE" for field in FINA_INDICATOR_VIP_FIELDS},
            },
        ),
        (
            "dim_trade_calendar",
            curated_root / "dims/trade_calendar",
            {
                "date": "DATE",
                "exchange": "VARCHAR",
                "is_open": "INTEGER",
                "pretrade_date": "DATE",
                "cal_date": "VARCHAR",
            },
        ),
        (
            "dim_security_master",
            curated_root / "dims/security_master",
            {
                "code": "VARCHAR",
                "name": "VARCHAR",
                "industry": "VARCHAR",
                "market": "VARCHAR",
                "list_status": "VARCHAR",
                "list_date": "DATE",
                "delist_date": "DATE",
                "snapshot_date": "DATE",
            },
        ),
        (
            "dim_sw_classify",
            curated_root / "dims/sw_classify",
            {
                "index_code": "VARCHAR",
                "industry_name": "VARCHAR",
                "level": "INTEGER",
                "parent_code": "VARCHAR",
                "src": "VARCHAR",
                "sw_level": "VARCHAR",
            },
        ),
        (
            "dim_sw_membership_history",
            curated_root / "dims/sw_membership_history",
            {
                "code": "VARCHAR",
                "index_code": "VARCHAR",
                "in_date": "DATE",
                "out_date": "DATE",
                "sector": "VARCHAR",
                "industry": "VARCHAR",
                "subindustry": "VARCHAR",
            },
        ),
        (
            "dim_security_namechange",
            curated_root / "dims/security_namechange",
            {
                "code": "VARCHAR",
                "name": "VARCHAR",
                "normalized_name": "VARCHAR",
                "start_date": "DATE",
                "end_date": "DATE",
                "ann_date": "DATE",
                "change_reason": "VARCHAR",
                "is_st": "INTEGER",
            },
        ),
        (
            "dim_ths_index",
            curated_root / "dims/ths_index",
            {
                "ths_code": "VARCHAR",
                "ths_name": "VARCHAR",
                "member_count": "DOUBLE",
                "exchange": "VARCHAR",
                "list_date": "DATE",
                "ths_type": "VARCHAR",
                "snapshot_date": "DATE",
            },
        ),
        (
            "dim_index_basic",
            curated_root / "dims/index_basic",
            {
                "code": "VARCHAR",
                "name": "VARCHAR",
                "market": "VARCHAR",
                "publisher": "VARCHAR",
                "category": "VARCHAR",
                "base_date": "DATE",
                "base_point": "DOUBLE",
                "list_date": "DATE",
                "weight_rule": "VARCHAR",
                "description": "VARCHAR",
                "exp_date": "DATE",
                "snapshot_date": "DATE",
            },
        ),
        (
            "dim_index_universe",
            curated_root / "dims/index_universe",
            {
                "universe_name": "VARCHAR",
                "display_name": "VARCHAR",
                "resolved_index_code": "VARCHAR",
                "index_daily_code": "VARCHAR",
                "index_weight_code": "VARCHAR",
                "resolved_name": "VARCHAR",
                "market": "VARCHAR",
                "publisher": "VARCHAR",
                "category": "VARCHAR",
                "required": "BOOLEAN",
                "enabled": "BOOLEAN",
                "status": "VARCHAR",
                "candidate_codes_json": "VARCHAR",
                "candidate_symbols_json": "VARCHAR",
                "candidate_names_json": "VARCHAR",
                "resolved_at": "VARCHAR",
                "snapshot_date": "DATE",
            },
        ),
        (
            "dim_ths_member",
            curated_root / "dims/ths_member",
            {
                "ths_code": "VARCHAR",
                "code": "VARCHAR",
                "name": "VARCHAR",
                "weight": "DOUBLE",
                "in_date": "DATE",
                "out_date": "DATE",
                "is_new": "VARCHAR",
            },
        ),
        (
            "fact_market_daily",
            curated_root / "facts/market_daily",
            {
                "date": "DATE",
                "code": "VARCHAR",
                "open": "DOUBLE",
                "high": "DOUBLE",
                "low": "DOUBLE",
                "close": "DOUBLE",
                "pct_chg": "DOUBLE",
                "ret_1d": "DOUBLE",
                "volume": "DOUBLE",
                "amount": "DOUBLE",
                "adj_factor": "DOUBLE",
                "bfq_open": "DOUBLE",
                "bfq_high": "DOUBLE",
                "bfq_low": "DOUBLE",
                "bfq_close": "DOUBLE",
                "qfq_open": "DOUBLE",
                "qfq_high": "DOUBLE",
                "qfq_low": "DOUBLE",
                "qfq_close": "DOUBLE",
                "hfq_open": "DOUBLE",
                "hfq_high": "DOUBLE",
                "hfq_low": "DOUBLE",
                "hfq_close": "DOUBLE",
                "price_adjust_mode": "VARCHAR",
            },
        ),
        (
            "fact_market_daily_basic",
            curated_root / "facts/market_daily_basic",
            {
                "date": "DATE",
                "code": "VARCHAR",
                "turnover_rate": "DOUBLE",
                "turnover_rate_f": "DOUBLE",
                "volume_ratio": "DOUBLE",
                "pe": "DOUBLE",
                "pe_ttm": "DOUBLE",
                "pb": "DOUBLE",
                "ps": "DOUBLE",
                "ps_ttm": "DOUBLE",
                "dv_ratio": "DOUBLE",
                "dv_ttm": "DOUBLE",
                "total_mv": "DOUBLE",
                "circ_mv": "DOUBLE",
                "total_mv_raw_wan": "DOUBLE",
                "circ_mv_raw_wan": "DOUBLE",
            },
        ),
        (
            "fact_market_adj_factor",
            curated_root / "facts/market_adj_factor",
            {
                "date": "DATE",
                "code": "VARCHAR",
                "adj_factor": "DOUBLE",
            },
        ),
        (
            "fact_market_stk_limit",
            curated_root / "facts/market_stk_limit",
            {
                "date": "DATE",
                "code": "VARCHAR",
                "pre_close": "DOUBLE",
                "up_limit": "DOUBLE",
                "down_limit": "DOUBLE",
            },
        ),
        (
            "fact_market_suspend_d",
            curated_root / "facts/market_suspend_d",
            {
                "date": "DATE",
                "code": "VARCHAR",
                "suspend_timing": "VARCHAR",
                "suspend_type": "VARCHAR",
                "is_suspended": "INTEGER",
            },
        ),
        (
            "fact_moneyflow_ths",
            curated_root / "facts/moneyflow_ths",
            {
                "date": "DATE",
                "code": "VARCHAR",
                "moneyflow_name": "VARCHAR",
                "moneyflow_pct_change": "DOUBLE",
                "moneyflow_latest": "DOUBLE",
                "moneyflow_net_amount": "DOUBLE",
                "moneyflow_net_d5_amount": "DOUBLE",
                "moneyflow_buy_lg_amount": "DOUBLE",
                "moneyflow_buy_lg_amount_rate": "DOUBLE",
                "moneyflow_buy_md_amount": "DOUBLE",
                "moneyflow_buy_md_amount_rate": "DOUBLE",
                "moneyflow_buy_sm_amount": "DOUBLE",
                "moneyflow_buy_sm_amount_rate": "DOUBLE",
            },
        ),
        (
            "fact_moneyflow",
            curated_root / "facts/moneyflow",
            {
                "date": "DATE",
                "code": "VARCHAR",
                "moneyflow_buy_sm_amount": "DOUBLE",
                "moneyflow_sell_sm_amount": "DOUBLE",
                "moneyflow_buy_md_amount": "DOUBLE",
                "moneyflow_sell_md_amount": "DOUBLE",
                "moneyflow_buy_lg_amount": "DOUBLE",
                "moneyflow_sell_lg_amount": "DOUBLE",
                "moneyflow_buy_elg_amount": "DOUBLE",
                "moneyflow_sell_elg_amount": "DOUBLE",
                "moneyflow_net_mf_amount": "DOUBLE",
            },
        ),
        (
            "fact_cyq_perf",
            curated_root / "facts/cyq_perf",
            {
                "date": "DATE",
                "code": "VARCHAR",
                "cyq_his_low": "DOUBLE",
                "cyq_his_high": "DOUBLE",
                "cyq_cost_5pct": "DOUBLE",
                "cyq_cost_15pct": "DOUBLE",
                "cyq_winner_rate": "DOUBLE",
                "cyq_cost_50pct": "DOUBLE",
                "cyq_cost_85pct": "DOUBLE",
                "cyq_cost_95pct": "DOUBLE",
                "cyq_weight_avg": "DOUBLE",
            },
        ),
        (
            "fact_cyq_chips",
            curated_root / "facts/cyq_chips",
            {
                "date": "DATE",
                "code": "VARCHAR",
                "chip_price": "DOUBLE",
                "chip_percent_raw_pct": "DOUBLE",
                "chip_percent": "DOUBLE",
            },
        ),
        (
            "fact_cyq_chips_daily",
            curated_root / "facts/cyq_chips_daily",
            {
                "date": "DATE",
                "code": "VARCHAR",
                **{field: "DOUBLE" for field in CYQ_CHIPS_DAILY_PANEL_FIELDS},
            },
        ),
        (
            "fact_stk_factor_pro",
            curated_root / "facts/stk_factor_pro",
            {
                "date": "DATE",
                "code": "VARCHAR",
                **{field: "DOUBLE" for field in STK_FACTOR_PRO_PANEL_FIELDS},
            },
        ),
        (
            "fact_stk_auction_o",
            curated_root / "facts/stk_auction_o",
            {
                "date": "DATE",
                "code": "VARCHAR",
                "auction_o_open": "DOUBLE",
                "auction_o_high": "DOUBLE",
                "auction_o_low": "DOUBLE",
                "auction_o_close": "DOUBLE",
                "auction_o_vol": "DOUBLE",
                "auction_o_amount": "DOUBLE",
                "auction_o_vwap": "DOUBLE",
            },
        ),
        (
            "fact_stk_auction_c",
            curated_root / "facts/stk_auction_c",
            {
                "date": "DATE",
                "code": "VARCHAR",
                "auction_c_open": "DOUBLE",
                "auction_c_high": "DOUBLE",
                "auction_c_low": "DOUBLE",
                "auction_c_close": "DOUBLE",
                "auction_c_vol": "DOUBLE",
                "auction_c_amount": "DOUBLE",
                "auction_c_vwap": "DOUBLE",
            },
        ),
        (
            "fact_report_rc_daily",
            curated_root / "facts/report_rc_daily",
            {
                "date": "DATE",
                "code": "VARCHAR",
                "report_rc_count": "BIGINT",
                "report_rc_org_count": "BIGINT",
                "report_rc_eps_mean": "DOUBLE",
                "report_rc_rating_score_mean": "DOUBLE",
            },
        ),
        (
            "fact_index_daily",
            curated_root / "facts/index_daily",
            {
                "date": "DATE",
                "code": "VARCHAR",
                "open": "DOUBLE",
                "high": "DOUBLE",
                "low": "DOUBLE",
                "close": "DOUBLE",
                "pre_close": "DOUBLE",
                "change": "DOUBLE",
                "vendor_pct_chg_raw_pct": "DOUBLE",
                "pct_chg": "DOUBLE",
                "return": "DOUBLE",
                "vol": "DOUBLE",
                "amount": "DOUBLE",
            },
        ),
        (
            "fact_index_weight",
            curated_root / "facts/index_weight",
            {
                "date": "DATE",
                "index_code": "VARCHAR",
                "code": "VARCHAR",
                "weight": "DOUBLE",
                "weight_decimal": "DOUBLE",
                "is_member": "INTEGER",
            },
        ),
        (
            "fact_finance_income_q",
            curated_root / "facts/finance_income_q",
            {
                "code": "VARCHAR",
                "ann_date": "DATE",
                "end_date": "DATE",
                **{field: "DOUBLE" for field in FINANCE_INCOME_VIP_FIELDS},
            },
        ),
        (
            "fact_finance_balancesheet_q",
            curated_root / "facts/finance_balancesheet_q",
            {
                "code": "VARCHAR",
                "ann_date": "DATE",
                "end_date": "DATE",
                **{field: "DOUBLE" for field in FINANCE_BALANCESHEET_VIP_FIELDS},
            },
        ),
        (
            "fact_finance_cashflow_q",
            curated_root / "facts/finance_cashflow_q",
            {
                "code": "VARCHAR",
                "ann_date": "DATE",
                "end_date": "DATE",
                **{field: "DOUBLE" for field in FINANCE_CASHFLOW_VIP_FIELDS},
            },
        ),
        (
            "fact_finance_indicator_q",
            curated_root / "facts/finance_indicator_q",
            {
                "code": "VARCHAR",
                "ann_date": "DATE",
                "end_date": "DATE",
                **{field: "DOUBLE" for field in FINA_INDICATOR_VIP_FIELDS},
            },
        ),
    ]

    for view_name, table_root, schema in view_specs:
        sql = _create_read_parquet_view_sql(view_name=view_name, table_root=table_root, empty_schema=schema)
        conn.execute(sql)
    conn.execute(
        """
CREATE OR REPLACE VIEW v_project_index_daily AS
SELECT
    d.date,
    d.code,
    b.name,
    b.market,
    b.publisher,
    b.category,
    d.open,
    d.high,
    d.low,
    d.close,
    d.pre_close,
    d.change,
    d.vendor_pct_chg_raw_pct,
    d.pct_chg,
    d."return",
    d.vol,
    d.amount
FROM fact_index_daily d
LEFT JOIN (
    SELECT *
    FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (PARTITION BY code ORDER BY snapshot_date DESC NULLS LAST) AS rn
        FROM dim_index_basic
    ) t
    WHERE rn = 1
) b
  ON d.code = b.code
"""
    )


def _create_index_universe_project_views(conn: Any) -> None:
    conn.execute(
        """
CREATE OR REPLACE VIEW v_project_index_membership_asof AS
WITH panel_dates AS (
    SELECT DISTINCT date
    FROM v_project_panel_cn_a
    WHERE date IS NOT NULL
),
active_universes AS (
    SELECT *
    FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (PARTITION BY universe_name ORDER BY snapshot_date DESC NULLS LAST) AS rn
        FROM dim_index_universe
        WHERE COALESCE(enabled, FALSE)
          AND LOWER(COALESCE(status, '')) = 'active'
    ) t
    WHERE rn = 1
),
asof_dates AS (
    SELECT
        d.date,
        u.universe_name,
        u.display_name,
        u.index_weight_code AS index_code,
        (
            SELECT MAX(w.date)
            FROM fact_index_weight w
            WHERE w.index_code = u.index_weight_code
              AND w.date <= d.date
        ) AS membership_date
    FROM panel_dates d
    CROSS JOIN active_universes u
)
SELECT
    a.date,
    a.universe_name,
    a.display_name,
    a.index_code,
    w.code,
    w.weight,
    w.weight_decimal,
    w.is_member,
    a.membership_date
FROM asof_dates a
JOIN fact_index_weight w
  ON w.index_code = a.index_code
 AND w.date = a.membership_date
WHERE a.membership_date IS NOT NULL
"""
    )
    conn.execute(
        """
CREATE OR REPLACE VIEW v_project_index_membership_monthly AS
SELECT
    w.date,
    u.universe_name,
    u.display_name,
    w.index_code,
    w.code,
    w.weight,
    w.weight_decimal,
    w.is_member
FROM fact_index_weight w
JOIN (
    SELECT *
    FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (PARTITION BY universe_name ORDER BY snapshot_date DESC NULLS LAST) AS rn
        FROM dim_index_universe
        WHERE COALESCE(enabled, FALSE)
          AND LOWER(COALESCE(status, '')) = 'active'
    ) t
    WHERE rn = 1
) u
  ON w.index_code = u.index_weight_code
"""
    )
    for universe_name in [
        "hs300",
        "csi500",
        "csi1000",
        "csi2000",
        "csi_all_share",
        "cnindex2000",
        "sme_composite",
    ]:
        conn.execute(
            f"""
CREATE OR REPLACE VIEW {_qident(f"v_project_panel_cn_a_{universe_name}")} AS
SELECT p.*
FROM v_project_panel_cn_a p
JOIN v_project_index_membership_asof m
  ON p.date = m.date
 AND p.code = m.code
WHERE m.universe_name = '{universe_name}'
"""
        )


def _create_read_parquet_view_sql(view_name: str, table_root: Path, empty_schema: dict[str, str]) -> str:
    valid_files = _collect_readable_parquet_files(table_root=table_root)
    if valid_files:
        files_sql = _sql_parquet_file_list(valid_files)
        available = _parquet_column_names(valid_files)
        expected = list(empty_schema)
        ordered = [col for col in expected if col in available]
        ordered.extend([col for col in available if col not in set(expected)])
        select_parts = [_qident(col) for col in ordered]
        for col, dtype in empty_schema.items():
            if col not in available:
                select_parts.append(f"CAST(NULL AS {dtype}) AS {_qident(col)}")
        select_sql = ", ".join(select_parts) if select_parts else "*"
        return (
            f"CREATE OR REPLACE VIEW {_qident(view_name)} AS "
            f"SELECT {select_sql} FROM read_parquet([{files_sql}], union_by_name=True)"
        )

    cols = ", ".join([f"CAST(NULL AS {dtype}) AS {_qident(col)}" for col, dtype in empty_schema.items()])
    return f"CREATE OR REPLACE VIEW {_qident(view_name)} AS SELECT {cols} WHERE 1=0"


def _project_panel_sql(
    universe_min_days_since_listed: int,
    universe_exclude_st: bool,
    include_bj: bool,
    tradable_require_close: bool,
    tradable_require_positive_volume: bool,
    tradable_require_positive_amount: bool,
    adjust_mode: str,
) -> str:
    _normalize_adjust_mode(adjust_mode)
    tech_cols = list(STK_FACTOR_PRO_PANEL_FIELDS)
    tech_expr = _select_exprs("tech", tech_cols, indent=8)
    tech_base_expr = _select_exprs("b", tech_cols, indent=8)
    tech_enriched_expr = _select_exprs("e", tech_cols, indent=4)
    cyq_cols = [
        "cyq_his_low",
        "cyq_his_high",
        "cyq_cost_5pct",
        "cyq_cost_15pct",
        "cyq_cost_50pct",
        "cyq_cost_85pct",
        "cyq_cost_95pct",
        "cyq_weight_avg",
        "cyq_winner_rate",
    ]
    cyq_expr = _select_exprs("cyq", cyq_cols, indent=8)
    cyq_base_expr = _select_exprs("b", cyq_cols, indent=8)
    cyq_enriched_expr = _select_exprs("e", cyq_cols, indent=4)
    cyq_chip_cols = list(CYQ_CHIPS_DAILY_PANEL_FIELDS)
    cyq_chip_expr = _select_exprs("chips", cyq_chip_cols, indent=8)
    cyq_chip_base_expr = _select_exprs("b", cyq_chip_cols, indent=8)
    cyq_chip_enriched_expr = _select_exprs("e", cyq_chip_cols, indent=4)
    tradable_conditions = ["COALESCE(e.is_suspended, 0) = 0"]
    if tradable_require_close:
        tradable_conditions.append("e.close IS NOT NULL")
    if tradable_require_positive_volume:
        tradable_conditions.append("COALESCE(e.volume, 0) > 0")
    if tradable_require_positive_amount:
        tradable_conditions.append("COALESCE(e.amount, 0) > 0")
    tradable_expr = " AND ".join(tradable_conditions) if tradable_conditions else "TRUE"
    limit_up_close_expr = (
        "e.up_limit IS NOT NULL AND e.close IS NOT NULL "
        "AND ABS(e.close - e.up_limit) <= GREATEST(ABS(e.up_limit) * 0.0001, 0.0001)"
    )
    limit_down_close_expr = (
        "e.down_limit IS NOT NULL AND e.close IS NOT NULL "
        "AND ABS(e.close - e.down_limit) <= GREATEST(ABS(e.down_limit) * 0.0001, 0.0001)"
    )
    one_price_up_expr = (
        f"({limit_up_close_expr}) AND e.open IS NOT NULL AND e.high IS NOT NULL AND e.low IS NOT NULL "
        "AND ABS(e.open - e.up_limit) <= GREATEST(ABS(e.up_limit) * 0.0001, 0.0001) "
        "AND ABS(e.high - e.up_limit) <= GREATEST(ABS(e.up_limit) * 0.0001, 0.0001) "
        "AND ABS(e.low - e.up_limit) <= GREATEST(ABS(e.up_limit) * 0.0001, 0.0001)"
    )
    one_price_down_expr = (
        f"({limit_down_close_expr}) AND e.open IS NOT NULL AND e.high IS NOT NULL AND e.low IS NOT NULL "
        "AND ABS(e.open - e.down_limit) <= GREATEST(ABS(e.down_limit) * 0.0001, 0.0001) "
        "AND ABS(e.high - e.down_limit) <= GREATEST(ABS(e.down_limit) * 0.0001, 0.0001) "
        "AND ABS(e.low - e.down_limit) <= GREATEST(ABS(e.down_limit) * 0.0001, 0.0001)"
    )

    universe_conditions = [
        "COALESCE(e.list_status, '') = 'L'",
        f"COALESCE(e.days_since_listed, 0) >= {int(max(0, universe_min_days_since_listed))}",
        "COALESCE(e.is_suspended, 0) = 0",
        "e.close IS NOT NULL",
    ]
    if universe_exclude_st:
        universe_conditions.append("COALESCE(e.is_st, 0) = 0")
    if not include_bj:
        universe_conditions.append("e.code NOT LIKE '%.BJ'")
    universe_expr = " AND ".join(universe_conditions) if universe_conditions else "TRUE"
    finance_asof_select = _finance_asof_select_exprs()
    finance_panel_select = _select_exprs("f", FINANCE_ASOF_PANEL_FIELDS, indent=4)
    income_lateral_select = _finance_lateral_select_exprs("income")
    balance_lateral_select = _finance_lateral_select_exprs("balance")
    cashflow_lateral_select = _finance_lateral_select_exprs("cashflow")
    indicator_lateral_select = _finance_lateral_select_exprs("indicator")

    return f"""
CREATE OR REPLACE VIEW v_project_market_daily_base AS
WITH security_master AS (
    SELECT *
    FROM (
        SELECT
            code,
            name,
            industry AS fallback_industry,
            market,
            list_status,
            list_date,
            delist_date,
            snapshot_date,
            ROW_NUMBER() OVER (PARTITION BY code ORDER BY snapshot_date DESC NULLS LAST) AS rn
        FROM dim_security_master
    ) t
    WHERE rn = 1
),
trade_days AS (
    SELECT
        date,
        ROW_NUMBER() OVER (ORDER BY date) AS trade_day_no
    FROM (
        SELECT DISTINCT date
        FROM dim_trade_calendar
        WHERE is_open = 1
          AND date IS NOT NULL
    ) td
),
membership AS (
    SELECT
        code,
        sector,
        industry,
        subindustry,
        in_date,
        COALESCE(out_date, DATE '2099-12-31') AS out_date
    FROM dim_sw_membership_history
),
namechange AS (
    SELECT
        code,
        normalized_name,
        is_st,
        start_date,
        COALESCE(end_date, DATE '2099-12-31') AS end_date,
        ann_date
    FROM dim_security_namechange
),
base AS (
    SELECT
        d.date,
        d.code,
        d.open,
        d.high,
        d.low,
        d.close,
        d.ret_1d,
        d.pct_chg,
        d.volume,
        d.amount,
        d.adj_factor,
        d.bfq_open,
        d.bfq_high,
        d.bfq_low,
        d.bfq_close,
        d.qfq_open,
        d.qfq_high,
        d.qfq_low,
        d.qfq_close,
        d.hfq_open,
        d.hfq_high,
        d.hfq_low,
        d.hfq_close,
        d.price_adjust_mode,
        b.turnover_rate,
        b.turnover_rate_f,
        b.volume_ratio,
        b.pe,
        b.pe_ttm,
        b.pb,
        b.ps,
        b.ps_ttm,
        b.dv_ratio,
        b.dv_ttm,
        b.total_mv,
        b.circ_mv,
        b.total_mv_raw_wan,
        b.circ_mv_raw_wan,
        s.name AS security_name,
        s.market,
        s.list_status,
        s.list_date,
        s.delist_date,
        s.fallback_industry,
        l.pre_close AS limit_pre_close,
        l.up_limit,
        l.down_limit,
        COALESCE(sp.is_suspended, 0) AS is_suspended,
        mf.moneyflow_buy_sm_amount,
        mf.moneyflow_sell_sm_amount,
        mf.moneyflow_buy_md_amount,
        mf.moneyflow_sell_md_amount,
        mf.moneyflow_buy_lg_amount,
        mf.moneyflow_sell_lg_amount,
        mf.moneyflow_buy_elg_amount,
        mf.moneyflow_sell_elg_amount,
        mf.moneyflow_net_mf_amount,
{cyq_expr}
{cyq_chip_expr}
{tech_expr}
        rr.report_rc_eps_mean,
        td.trade_day_no AS trade_day_no,
        ld.trade_day_no AS list_trade_day_no
    FROM fact_market_daily d
    LEFT JOIN fact_market_daily_basic b
      ON d.code = b.code
     AND d.date = b.date
    LEFT JOIN security_master s
      ON d.code = s.code
    LEFT JOIN fact_market_stk_limit l
      ON d.code = l.code
     AND d.date = l.date
    LEFT JOIN fact_market_suspend_d sp
      ON d.code = sp.code
     AND d.date = sp.date
    LEFT JOIN fact_moneyflow mf
      ON d.code = mf.code
     AND d.date = mf.date
    LEFT JOIN fact_cyq_perf cyq
      ON d.code = cyq.code
     AND d.date = cyq.date
    LEFT JOIN fact_cyq_chips_daily chips
      ON d.code = chips.code
     AND d.date = chips.date
    LEFT JOIN fact_stk_factor_pro tech
      ON d.code = tech.code
     AND d.date = tech.date
    LEFT JOIN fact_report_rc_daily rr
      ON d.code = rr.code
     AND d.date = rr.date
    LEFT JOIN trade_days td
      ON d.date = td.date
    LEFT JOIN trade_days ld
      ON s.list_date = ld.date
),
enriched AS (
    SELECT
        b.date,
        b.code,
        b.open,
        b.high,
        b.low,
        b.close,
        b.ret_1d,
        b.pct_chg,
        b.volume,
        b.amount,
        b.adj_factor,
        b.bfq_open,
        b.bfq_high,
        b.bfq_low,
        b.bfq_close,
        b.qfq_open,
        b.qfq_high,
        b.qfq_low,
        b.qfq_close,
        b.hfq_open,
        b.hfq_high,
        b.hfq_low,
        b.hfq_close,
        b.price_adjust_mode,
        b.turnover_rate,
        b.turnover_rate_f,
        b.volume_ratio,
        b.pe,
        b.pe_ttm,
        b.pb,
        b.ps,
        b.ps_ttm,
        b.dv_ratio,
        b.dv_ttm,
        b.total_mv,
        b.circ_mv,
        b.total_mv_raw_wan,
        b.circ_mv_raw_wan,
        b.limit_pre_close,
        b.up_limit,
        b.down_limit,
        b.is_suspended,
        b.moneyflow_buy_sm_amount,
        b.moneyflow_sell_sm_amount,
        b.moneyflow_buy_md_amount,
        b.moneyflow_sell_md_amount,
        b.moneyflow_buy_lg_amount,
        b.moneyflow_sell_lg_amount,
        b.moneyflow_buy_elg_amount,
        b.moneyflow_sell_elg_amount,
        b.moneyflow_net_mf_amount,
{cyq_base_expr}
{cyq_chip_base_expr}
{tech_base_expr}
        b.report_rc_eps_mean,
        COALESCE(m.sector, b.market, 'UNKNOWN') AS sector,
        COALESCE(m.industry, b.fallback_industry, 'UNKNOWN') AS industry,
        COALESCE(m.subindustry, '') AS subindustry,
        b.security_name,
        b.market,
        b.list_status,
        b.list_date,
        b.delist_date,
        COALESCE(st.is_st, 0) AS is_st,
        CASE
            WHEN b.trade_day_no IS NULL OR b.list_trade_day_no IS NULL THEN NULL
            ELSE b.trade_day_no - b.list_trade_day_no + 1
        END AS days_since_listed
    FROM base b
    LEFT JOIN LATERAL (
        SELECT
            sector,
            industry,
            subindustry
        FROM membership m
        WHERE m.code = b.code
          AND b.date >= m.in_date
          AND b.date <= m.out_date
        ORDER BY m.in_date DESC
        LIMIT 1
    ) m ON TRUE
    LEFT JOIN LATERAL (
        SELECT is_st
        FROM namechange n
        WHERE n.code = b.code
          AND b.date >= n.start_date
          AND b.date <= n.end_date
        ORDER BY n.start_date DESC NULLS LAST, n.ann_date DESC NULLS LAST
        LIMIT 1
    ) st ON TRUE
)
SELECT
    e.date,
    e.code,
    e.open,
    e.high,
    e.low,
    e.close,
    e.ret_1d,
    e.pct_chg,
    e.volume,
    e.amount,
    e.adj_factor,
    e.bfq_open,
    e.bfq_high,
    e.bfq_low,
    e.bfq_close,
    e.qfq_open,
    e.qfq_high,
    e.qfq_low,
    e.qfq_close,
    e.hfq_open,
    e.hfq_high,
    e.hfq_low,
    e.hfq_close,
    e.price_adjust_mode,
    e.turnover_rate,
    e.turnover_rate_f,
    e.volume_ratio,
    e.pe,
    e.pe_ttm,
    e.pb,
    e.ps,
    e.ps_ttm,
    e.dv_ratio,
    e.dv_ttm,
    e.total_mv,
    e.circ_mv,
    e.total_mv_raw_wan,
    e.circ_mv_raw_wan,
    e.limit_pre_close,
    e.up_limit,
    e.down_limit,
    e.is_suspended,
    e.moneyflow_buy_sm_amount,
    e.moneyflow_sell_sm_amount,
    e.moneyflow_buy_md_amount,
    e.moneyflow_sell_md_amount,
    e.moneyflow_buy_lg_amount,
    e.moneyflow_sell_lg_amount,
    e.moneyflow_buy_elg_amount,
    e.moneyflow_sell_elg_amount,
    e.moneyflow_net_mf_amount,
{cyq_enriched_expr}
{cyq_chip_enriched_expr}
{tech_enriched_expr}
    e.report_rc_eps_mean,
    e.is_st,
    e.days_since_listed,
    e.sector,
    e.industry,
    e.subindustry,
    e.security_name,
    e.market,
    e.list_status,
    e.list_date,
    e.delist_date,
    CAST(CASE WHEN {tradable_expr} THEN 1 ELSE 0 END AS INTEGER) AS tradable,
    CAST(CASE WHEN {tradable_expr} THEN 1 ELSE 0 END AS INTEGER) AS can_trade,
    CAST(CASE WHEN ({tradable_expr}) AND NOT ({one_price_up_expr}) THEN 1 ELSE 0 END AS INTEGER) AS can_buy,
    CAST(CASE WHEN ({tradable_expr}) AND NOT ({one_price_down_expr}) THEN 1 ELSE 0 END AS INTEGER) AS can_sell,
    CAST(CASE WHEN {one_price_up_expr} THEN 1 ELSE 0 END AS INTEGER) AS is_one_price_up_limit,
    CAST(CASE WHEN {one_price_down_expr} THEN 1 ELSE 0 END AS INTEGER) AS is_one_price_down_limit,
    CAST(CASE WHEN {limit_up_close_expr} THEN 1 ELSE 0 END AS INTEGER) AS is_limit_up_close,
    CAST(CASE WHEN {limit_down_close_expr} THEN 1 ELSE 0 END AS INTEGER) AS is_limit_down_close,
    CAST(CASE WHEN {universe_expr} THEN 1 ELSE 0 END AS INTEGER) AS universe
FROM enriched e
WHERE regexp_matches(e.code, '^[0-9]{{6}}\\.(SZ|SH|BJ)$');

CREATE OR REPLACE VIEW v_project_financial_asof_daily AS
SELECT
{finance_asof_select}
FROM v_project_market_daily_base b
    LEFT JOIN LATERAL (
        SELECT
{income_lateral_select}
        FROM fact_finance_income_q fi
        WHERE fi.code = b.code
          AND fi.ann_date <= b.date
        ORDER BY fi.ann_date DESC NULLS LAST, fi.end_date DESC NULLS LAST
        LIMIT 1
    ) income_asof ON TRUE
    LEFT JOIN LATERAL (
        SELECT
{balance_lateral_select}
        FROM fact_finance_balancesheet_q fb
        WHERE fb.code = b.code
          AND fb.ann_date <= b.date
        ORDER BY fb.ann_date DESC NULLS LAST, fb.end_date DESC NULLS LAST
        LIMIT 1
    ) balance_asof ON TRUE
    LEFT JOIN LATERAL (
        SELECT
{cashflow_lateral_select}
        FROM fact_finance_cashflow_q fc
        WHERE fc.code = b.code
          AND fc.ann_date <= b.date
        ORDER BY fc.ann_date DESC NULLS LAST, fc.end_date DESC NULLS LAST
        LIMIT 1
    ) cashflow_asof ON TRUE
    LEFT JOIN LATERAL (
        SELECT
{indicator_lateral_select}
        FROM fact_finance_indicator_q ff
        WHERE ff.code = b.code
          AND ff.ann_date <= b.date
        ORDER BY ff.ann_date DESC NULLS LAST, ff.end_date DESC NULLS LAST
        LIMIT 1
    ) indicator_asof ON TRUE;

CREATE OR REPLACE VIEW v_project_panel_cn_a AS
SELECT
    b.date,
    b.code,
    b.open,
    b.high,
    b.low,
    b.close,
    b.ret_1d,
    b.pct_chg,
    b.volume,
    b.amount,
    b.adj_factor,
    b.bfq_open,
    b.bfq_high,
    b.bfq_low,
    b.bfq_close,
    b.qfq_open,
    b.qfq_high,
    b.qfq_low,
    b.qfq_close,
    b.hfq_open,
    b.hfq_high,
    b.hfq_low,
    b.hfq_close,
    b.price_adjust_mode,
    b.turnover_rate,
    b.turnover_rate_f,
    b.volume_ratio,
    b.pe,
    b.pe_ttm,
    b.pb,
    b.ps,
    b.ps_ttm,
    b.dv_ratio,
    b.dv_ttm,
    b.total_mv,
    b.circ_mv,
    b.total_mv_raw_wan,
    b.circ_mv_raw_wan,
    b.limit_pre_close,
    b.up_limit,
    b.down_limit,
    b.is_suspended,
    b.moneyflow_buy_sm_amount,
    b.moneyflow_sell_sm_amount,
    b.moneyflow_buy_md_amount,
    b.moneyflow_sell_md_amount,
    b.moneyflow_buy_lg_amount,
    b.moneyflow_sell_lg_amount,
    b.moneyflow_buy_elg_amount,
    b.moneyflow_sell_elg_amount,
    b.moneyflow_net_mf_amount,
{cyq_base_expr}
{cyq_chip_base_expr}
{tech_base_expr}
    b.report_rc_eps_mean,
    b.is_st,
    b.days_since_listed,
    b.sector,
    b.industry,
    b.subindustry,
    b.security_name,
    b.market,
    b.list_status,
    b.list_date,
    b.delist_date,
{finance_panel_select}
    b.tradable,
    b.can_trade,
    b.can_buy,
    b.can_sell,
    b.is_one_price_up_limit,
    b.is_one_price_down_limit,
    b.is_limit_up_close,
    b.is_limit_down_close,
    b.universe
FROM v_project_market_daily_base b
LEFT JOIN v_project_financial_asof_daily f
  ON b.date = f.date
 AND b.code = f.code
"""


def _select_exprs(alias: str, columns: Sequence[str], indent: int) -> str:
    pad = " " * int(indent)
    return "\n".join(f"{pad}{_qident(alias)}.{_qident(col)}," for col in columns)


def _finance_lateral_select_exprs(source: str, fields: Sequence[str] | None = None) -> str:
    source_key = str(source or "").strip()
    if fields is None:
        if source_key == "income":
            fields = ("ann_date", "end_date", *FINANCE_INCOME_VIP_FIELDS)
        elif source_key == "balance":
            fields = ("ann_date", "end_date", *FINANCE_BALANCESHEET_VIP_FIELDS)
        elif source_key == "cashflow":
            fields = ("ann_date", "end_date", *FINANCE_CASHFLOW_VIP_FIELDS)
        elif source_key == "indicator":
            fields = ("ann_date", "end_date", *FINA_INDICATOR_VIP_FIELDS)
        else:
            fields = ("ann_date", "end_date")
    pad = " " * 12
    return ",\n".join(f"{pad}{_qident(field)}" for field in fields)


def _finance_asof_select_exprs(fields: Sequence[str] | None = None, indent: int = 4) -> str:
    selected = list(fields or FINANCE_ASOF_PANEL_FIELDS)
    pad = " " * int(indent)
    lines = [f"{pad}b.date", f"{pad}b.code"]
    for field in selected:
        source, raw_col = FINANCE_ASOF_FIELD_MAP[str(field)]
        lines.append(f"{pad}{_qident(source + '_asof')}.{_qident(raw_col)} AS {_qident(field)}")
    return ",\n".join(lines)


def _qident(name: str) -> str:
    parts = [p for p in str(name).split(".") if p]
    quoted: list[str] = []
    for part in parts:
        quoted.append('"' + str(part).replace('"', '""') + '"')
    return ".".join(quoted)


def _escape_sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def _sql_parquet_file_list(paths: list[Path]) -> str:
    quoted = [f"'{_escape_sql_path(path)}'" for path in paths]
    return ", ".join(quoted)


def _collect_readable_parquet_files(table_root: Path) -> list[Path]:
    if not table_root.exists():
        return []
    files = sorted(table_root.rglob("*.parquet"))
    out: list[Path] = []
    for path in files:
        if _parquet_has_non_root_columns(path):
            out.append(path)
    return out


def _parquet_has_non_root_columns(path: Path) -> bool:
    try:
        import pyarrow.parquet as pq  # type: ignore

        schema = pq.read_schema(path)
        names = list(schema.names or []) if schema is not None else []
        return len(names) > 0
    except Exception:
        pass

    try:
        sample = pd.read_parquet(path)
    except Exception:
        return False
    return len(sample.columns) > 0


def _parquet_column_names(paths: Sequence[Path]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for path in paths:
        names: list[str] = []
        try:
            import pyarrow.parquet as pq  # type: ignore

            schema = pq.read_schema(path)
            names = [str(name) for name in (schema.names or [])]
        except Exception:
            try:
                sample = pd.read_parquet(path)
                names = [str(col) for col in sample.columns]
            except Exception:
                names = []
        for name in names:
            if name not in seen:
                out.append(name)
                seen.add(name)
    return out
