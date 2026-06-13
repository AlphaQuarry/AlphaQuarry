from __future__ import annotations

import argparse
import gc
import json
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_mining.datasource import (
    DEFAULT_DIM_GROUPS,
    DEFAULT_FACT_GROUPS,
    DIM_GROUP_TABLES,
    FACT_GROUP_TABLES,
    ParquetLake,
    TushareApiError,
    TushareErrorCategory,
    aggregate_cyq_chips_daily,
    build_checkpoint_signature,
    build_duckdb_catalog,
    build_tushare_client_from_settings,
    curate_cyq_chips,
    curate_cyq_perf,
    curate_finance_balancesheet_vip,
    curate_finance_cashflow_vip,
    curate_finance_fina_indicator_vip,
    curate_finance_income_vip,
    curate_index_basic,
    curate_index_classify,
    curate_index_daily,
    curate_index_weight,
    curate_index_member_all,
    curate_market_adj_factor,
    curate_market_daily,
    curate_market_daily_basic,
    curate_market_stk_limit,
    curate_market_suspend_d,
    curate_moneyflow,
    curate_moneyflow_ths,
    curate_report_rc_daily,
    curate_security_namechange,
    curate_stock_basic,
    curate_stk_auction_c,
    curate_stk_auction_o,
    curate_stk_factor_pro,
    curate_ths_index,
    curate_ths_member,
    curate_trade_cal,
    get_checkpoint_path,
    load_checkpoint,
    load_datasource_settings,
    load_index_universe_config,
    known_dim_tables,
    known_fact_tables,
    normalize_index_code,
    reset_checkpoint,
    resolve_index_universes,
    resolve_dim_table_selection,
    resolve_fact_table_selection,
    save_checkpoint,
)
from alpha_mining.datasource.duckdb_runtime import build_duckdb_runtime_settings


_FACT_VENDOR_SPECS: dict[str, dict[str, object]] = {
    "daily": {
        "table": "daily",
        "date_col": "trade_date",
        "key_cols": ("ts_code", "trade_date"),
    },
    "daily_basic": {
        "table": "daily_basic",
        "date_col": "trade_date",
        "key_cols": ("ts_code", "trade_date"),
    },
    "adj_factor": {
        "table": "adj_factor",
        "date_col": "trade_date",
        "key_cols": ("ts_code", "trade_date"),
    },
    "stk_limit": {
        "table": "stk_limit",
        "date_col": "trade_date",
        "key_cols": ("ts_code", "trade_date"),
    },
    "suspend_d": {
        "table": "suspend_d",
        "date_col": "trade_date",
        "key_cols": ("ts_code", "trade_date"),
    },
    "income_vip": {
        "table": "income_vip",
        "date_col": "ann_date",
        "key_cols": ("ts_code", "ann_date", "end_date"),
    },
    "balancesheet_vip": {
        "table": "balancesheet_vip",
        "date_col": "ann_date",
        "key_cols": ("ts_code", "ann_date", "end_date"),
    },
    "cashflow_vip": {
        "table": "cashflow_vip",
        "date_col": "ann_date",
        "key_cols": ("ts_code", "ann_date", "end_date"),
    },
    "fina_indicator_vip": {
        "table": "fina_indicator_vip",
        "date_col": "ann_date",
        "key_cols": ("ts_code", "ann_date", "end_date"),
    },
    "moneyflow": {
        "table": "moneyflow",
        "date_col": "trade_date",
        "key_cols": ("ts_code", "trade_date"),
    },
    "moneyflow_ths": {
        "table": "moneyflow_ths",
        "date_col": "trade_date",
        "key_cols": ("ts_code", "trade_date"),
    },
    "cyq_perf": {
        "table": "cyq_perf",
        "date_col": "trade_date",
        "key_cols": ("ts_code", "trade_date"),
    },
    "cyq_chips": {
        "table": "cyq_chips",
        "date_col": "trade_date",
        "key_cols": ("ts_code", "trade_date", "price"),
    },
    "index_daily": {
        "table": "index_daily",
        "date_col": "trade_date",
        "key_cols": ("ts_code", "trade_date"),
    },
    "index_weight": {
        "table": "index_weight",
        "date_col": "trade_date",
        "key_cols": ("index_code", "con_code", "trade_date"),
    },
    "stk_factor_pro": {
        "table": "stk_factor_pro",
        "date_col": "trade_date",
        "key_cols": ("ts_code", "trade_date"),
    },
    "stk_auction_o": {
        "table": "stk_auction_o",
        "date_col": "trade_date",
        "key_cols": ("ts_code", "trade_date"),
    },
    "stk_auction_c": {
        "table": "stk_auction_c",
        "date_col": "trade_date",
        "key_cols": ("ts_code", "trade_date"),
    },
    "report_rc": {
        "table": "report_rc",
        "date_col": "report_date",
        "key_cols": ("ts_code", "report_date", "org_name", "author_name"),
    },
}

_FACT_CURATED_SPECS: dict[str, dict[str, object]] = {
    "daily": {
        "table": "facts/market_daily",
        "date_col": "date",
        "key_cols": ("code", "date"),
    },
    "daily_basic": {
        "table": "facts/market_daily_basic",
        "date_col": "date",
        "key_cols": ("code", "date"),
    },
    "adj_factor": {
        "table": "facts/market_adj_factor",
        "date_col": "date",
        "key_cols": ("code", "date"),
    },
    "stk_limit": {
        "table": "facts/market_stk_limit",
        "date_col": "date",
        "key_cols": ("code", "date"),
    },
    "suspend_d": {
        "table": "facts/market_suspend_d",
        "date_col": "date",
        "key_cols": ("code", "date"),
    },
    "income_vip": {
        "table": "facts/finance_income_q",
        "date_col": "ann_date",
        "key_cols": ("code", "ann_date", "end_date"),
    },
    "balancesheet_vip": {
        "table": "facts/finance_balancesheet_q",
        "date_col": "ann_date",
        "key_cols": ("code", "ann_date", "end_date"),
    },
    "cashflow_vip": {
        "table": "facts/finance_cashflow_q",
        "date_col": "ann_date",
        "key_cols": ("code", "ann_date", "end_date"),
    },
    "fina_indicator_vip": {
        "table": "facts/finance_indicator_q",
        "date_col": "ann_date",
        "key_cols": ("code", "ann_date", "end_date"),
    },
    "moneyflow_ths": {
        "table": "facts/moneyflow_ths",
        "date_col": "date",
        "key_cols": ("code", "date"),
    },
    "moneyflow": {
        "table": "facts/moneyflow",
        "date_col": "date",
        "key_cols": ("code", "date"),
    },
    "cyq_perf": {
        "table": "facts/cyq_perf",
        "date_col": "date",
        "key_cols": ("code", "date"),
    },
    "cyq_chips": {
        "table": "facts/cyq_chips",
        "date_col": "date",
        "key_cols": ("code", "date", "chip_price"),
    },
    "index_daily": {
        "table": "facts/index_daily",
        "date_col": "date",
        "key_cols": ("code", "date"),
    },
    "index_weight": {
        "table": "facts/index_weight",
        "date_col": "date",
        "key_cols": ("index_code", "code", "date"),
    },
    "stk_factor_pro": {
        "table": "facts/stk_factor_pro",
        "date_col": "date",
        "key_cols": ("code", "date"),
    },
    "stk_auction_o": {
        "table": "facts/stk_auction_o",
        "date_col": "date",
        "key_cols": ("code", "date"),
    },
    "stk_auction_c": {
        "table": "facts/stk_auction_c",
        "date_col": "date",
        "key_cols": ("code", "date"),
    },
    "report_rc": {
        "table": "facts/report_rc_daily",
        "date_col": "date",
        "key_cols": ("code", "date"),
    },
}

_RANGE_FACT_TABLES: tuple[str, ...] = (
    "income_vip",
    "balancesheet_vip",
    "cashflow_vip",
    "fina_indicator_vip",
    "report_rc",
)
_WIDE_TRADE_FACT_TABLES: frozenset[str] = frozenset({"stk_factor_pro"})
_WIDE_TRADE_FACT_FLUSH_DAYS = 5
_CODE_RANGE_FACT_TABLES: tuple[str, ...] = (
    "cyq_perf",
    "cyq_chips",
    "index_daily",
    "index_weight",
)
_CODE_RANGE_FLUSH_CODES = 50
_CYQ_CHIPS_START_DATE = "2018-01-01"
_DEFAULT_INDEX_DAILY_TS_CODES: tuple[str, ...] = (
    "000300.SH",
    "000905.SH",
    "000852.SH",
    "000016.SH",
    "000001.SH",
    "399001.SZ",
    "399006.SZ",
)


def main() -> None:
    _configure_stdio_for_live_logs()
    parser = argparse.ArgumentParser(description="Incremental update for Tushare lake")
    parser.add_argument("--config", default="", help="Datasource config yaml path")
    parser.add_argument("--start-date", default="", help="YYYY-MM-DD, default from ingestion state")
    parser.add_argument("--end-date", default="", help="YYYY-MM-DD, default=today")
    parser.add_argument("--exchange", default="SSE")
    parser.add_argument("--adjust-mode", default="", choices=["", "qfq", "hfq"])
    parser.add_argument("--source-view", default="")
    parser.add_argument("--field-catalog-version", default="")
    parser.add_argument("--refresh-dims", action="store_true", help="Refresh stock/index dimensions")
    parser.add_argument(
        "--fact-groups",
        default=",".join(DEFAULT_FACT_GROUPS),
        help=f"Comma-separated fact groups (default: {','.join(DEFAULT_FACT_GROUPS)}; use 'none' to disable).",
    )
    parser.add_argument(
        "--fact-tables",
        default="",
        help=f"Additional fact tables to include. Allowed: {','.join(known_fact_tables())}",
    )
    parser.add_argument(
        "--stk-factor-pro-fetch-mode",
        default="trade_date",
        choices=["trade_date", "ts_code_range"],
        help=(
            "Fetch mode for stk_factor_pro. trade_date is faster but may return sparse "
            "advanced indicators on some dates; ts_code_range matches ts_code/date queries "
            "and is recommended for repairing low-coverage stk_factor_pro fields."
        ),
    )
    parser.add_argument(
        "--code-range-ts-codes",
        default="",
        help=(
            "Optional comma-separated ts_codes for code-range fact pulls. "
            "Useful for smoke tests such as 000002.SZ before running all stocks."
        ),
    )
    parser.add_argument(
        "--index-daily-ts-codes",
        default=",".join(_DEFAULT_INDEX_DAILY_TS_CODES),
        help=(
            "Comma-separated index ts_codes for index_daily pulls. "
            "Default is a compact broad-index set for benchmark research."
        ),
    )
    parser.add_argument(
        "--index-universe-config",
        default="configs/index_universes.yaml",
        help="YAML config for supported index universes.",
    )
    parser.add_argument(
        "--index-universe-names",
        default="hs300,csi500,csi1000,csi2000,csi_all_share,cnindex2000,sme_composite",
        help="Comma-separated index universe names used to resolve index_weight codes.",
    )
    parser.add_argument(
        "--index-universe-missing-policy",
        default="warn",
        choices=["warn", "fail"],
        help="How to handle configured index universes that are missing from index_basic.",
    )
    parser.add_argument(
        "--index-weight-index-codes",
        default="",
        help="Optional comma-separated index codes for index_weight pulls; overrides index universe resolution.",
    )
    parser.add_argument(
        "--index-basic-markets",
        default="SSE,SZSE,CSI,CNI",
        help="Comma-separated markets for index_basic dimension refresh.",
    )
    parser.add_argument(
        "--code-range-workers",
        type=int,
        default=1,
        help=(
            "Parallel workers for code-range fact pulls. Default 1 is conservative; "
            "use 4-8 for stk_factor_pro repair if your Tushare endpoint tolerates it."
        ),
    )
    parser.add_argument(
        "--stk-factor-pro-repair-sparse-dates",
        action="store_true",
        help=(
            "When stk_factor_pro runs in ts_code_range mode, scan local curated data "
            "for dates where selected tech fields are entirely NULL and repair only "
            "the affected months."
        ),
    )
    parser.add_argument(
        "--stk-factor-pro-sparse-fields",
        default="",
        help=(
            "Comma-separated fields used to detect sparse stk_factor_pro dates. "
            "Accepts tech_dpo_qfq or dpo_qfq names. Default: all local tech_* fields."
        ),
    )
    parser.add_argument(
        "--stk-factor-pro-sparse-min-rows",
        type=int,
        default=100,
        help="Minimum rows per date before a zero-non-null field is treated as a sparse-date defect.",
    )
    parser.add_argument(
        "--stk-factor-pro-sparse-max-finite-rate",
        type=float,
        default=0.05,
        help=(
            "A field/date is sparse when finite_count / row_count is at or below this ratio. "
            "Default 0.05 still catches all-null dates and dates only touched by smoke/partial repairs."
        ),
    )
    parser.add_argument(
        "--stk-factor-pro-sparse-output-csv",
        default="artifacts/data_quality/stk_factor_pro_sparse_dates.csv",
        help="CSV report path for detected sparse stk_factor_pro dates.",
    )
    parser.add_argument(
        "--exclude-fact-tables",
        default="",
        help="Fact tables to exclude after group expansion.",
    )
    parser.add_argument(
        "--dim-groups",
        default=",".join(DEFAULT_DIM_GROUPS),
        help=f"Comma-separated dim groups used with --refresh-dims (default: {','.join(DEFAULT_DIM_GROUPS)}; use 'none' to disable).",
    )
    parser.add_argument(
        "--dim-tables",
        default="",
        help=f"Additional dim tables to include when --refresh-dims. Allowed: {','.join(known_dim_tables())}",
    )
    parser.add_argument(
        "--exclude-dim-tables",
        default="",
        help="Dim tables to exclude after group expansion.",
    )
    parser.add_argument("--skip-duckdb", action="store_true")
    parser.add_argument("--no-materialize-project-base", action="store_true")
    parser.add_argument(
        "--duckdb-memory-limit",
        default="4GB",
        help="DuckDB memory_limit used when rebuilding catalog",
    )
    parser.add_argument(
        "--duckdb-threads",
        type=int,
        default=2,
        help="DuckDB threads used when rebuilding catalog",
    )
    parser.add_argument(
        "--duckdb-temp-directory",
        default="",
        help="DuckDB temp directory for catalog rebuild; default is <duckdb-path>.tmp.",
    )
    parser.add_argument("--duckdb-temp-run-id", default="")
    parser.add_argument(
        "--duckdb-temp-isolate-run",
        dest="duckdb_temp_isolate_run",
        action="store_true",
        default=True,
        help="Use a per-run DuckDB temp subdirectory for catalog rebuild.",
    )
    parser.add_argument(
        "--no-duckdb-temp-isolate-run",
        dest="duckdb_temp_isolate_run",
        action="store_false",
    )
    parser.add_argument("--duckdb-max-temp-directory-size", default="12GB")
    parser.add_argument(
        "--lookback-trade-days",
        type=int,
        default=-1,
        help="Rolling trade-day backfill window when --start-date is omitted",
    )
    parser.add_argument(
        "--flush-trade-days",
        type=int,
        default=20,
        help="Flush batch size by trade days",
    )
    parser.add_argument(
        "--range-window-days",
        type=int,
        default=180,
        help="Window size (days) for P2 range facts fetch",
    )
    parser.add_argument(
        "--range-empty-policy",
        default="warn",
        choices=["warn", "fail"],
        help="How to handle empty range-fact pulls",
    )
    parser.add_argument(
        "--keep-out-of-range-ann-date",
        action="store_true",
        help="Disable local ann_date boundary filter for range facts",
    )
    parser.add_argument(
        "--prune-out-of-range",
        action="store_true",
        help="Physically prune existing range-fact rows with ann_date > end_date before refill",
    )
    parser.add_argument("--resume", dest="resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable checkpoint resume",
    )
    parser.set_defaults(resume=True)
    parser.add_argument("--reset-checkpoint", action="store_true", help="Reset checkpoint before run")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print execution plan and exit without pulling dims/facts",
    )
    args = parser.parse_args()

    settings = load_datasource_settings(str(args.config or "") or None)
    lake = ParquetLake(paths=settings.paths)
    client = build_tushare_client_from_settings(settings.tushare)

    selected_fact_tables = resolve_fact_table_selection(
        groups_raw=str(args.fact_groups or ""),
        include_raw=str(args.fact_tables or ""),
        exclude_raw=str(args.exclude_fact_tables or ""),
    )
    selected_dim_tables = resolve_dim_table_selection(
        groups_raw=str(args.dim_groups or ""),
        include_raw=str(args.dim_tables or ""),
        exclude_raw=str(args.exclude_dim_tables or ""),
    )
    if not bool(args.refresh_dims):
        selected_dim_tables = []

    if not selected_fact_tables and not selected_dim_tables:
        print("[update] no selected fact/dim tables, nothing to do")
        return

    stk_factor_pro_fetch_mode = str(args.stk_factor_pro_fetch_mode or "trade_date").strip().lower()
    (
        selected_trade_fact_tables,
        selected_code_range_fact_tables,
        selected_range_fact_tables,
    ) = _split_selected_fact_tables(
        selected_fact_tables,
        stk_factor_pro_fetch_mode=stk_factor_pro_fetch_mode,
    )

    end_date = str(args.end_date).strip() or datetime.now().strftime("%Y-%m-%d")
    start_date = str(args.start_date).strip()
    lookback_trade_days = (
        max(0, int(args.lookback_trade_days))
        if int(args.lookback_trade_days) >= 0
        else max(0, int(settings.update_lookback_trade_days))
    )
    if not start_date:
        start_date = _resolve_default_start_date(
            lake=lake,
            client=client,
            end_date=end_date,
            exchange=str(args.exchange),
            lookback_trade_days=lookback_trade_days,
            anchor_tables=[
                *selected_trade_fact_tables,
                *selected_code_range_fact_tables,
            ],
        )

    adjust_mode = str(args.adjust_mode).strip().lower() or settings.adjust_mode
    source_view = str(args.source_view).strip() or settings.source_view
    field_catalog_version = str(args.field_catalog_version).strip() or settings.field_catalog_version
    code_range_ts_codes = _valid_ts_codes(_parse_csv_tokens(str(args.code_range_ts_codes or "")))
    index_daily_ts_codes = _resolve_index_daily_ts_codes(str(args.index_daily_ts_codes or ""))
    index_universe_config = str(args.index_universe_config or "").strip()
    index_universe_names = [x.lower() for x in _parse_csv_tokens(str(args.index_universe_names or ""))]
    index_weight_index_codes = _resolve_index_weight_index_codes(str(args.index_weight_index_codes or ""))
    index_universe_missing_policy = str(args.index_universe_missing_policy or "warn").strip().lower()
    index_basic_markets = _parse_csv_tokens(str(args.index_basic_markets or ""))
    code_range_workers = max(1, int(args.code_range_workers))
    repair_stk_sparse = (
        bool(args.stk_factor_pro_repair_sparse_dates) and "stk_factor_pro" in selected_code_range_fact_tables
    )
    flush_trade_days = max(1, int(args.flush_trade_days))
    effective_flush_trade_days = _effective_flush_trade_days(
        requested_flush_trade_days=flush_trade_days,
        selected_trade_fact_tables=selected_trade_fact_tables,
    )
    range_window_days = max(1, int(args.range_window_days))
    range_empty_policy = str(args.range_empty_policy).strip().lower()
    enforce_range_ann_date_boundary = not bool(args.keep_out_of_range_ann_date)
    prune_out_of_range = bool(args.prune_out_of_range)

    checkpoint_task = "update_tushare_lake_daily"
    checkpoint_signature = build_checkpoint_signature(
        task_name=checkpoint_task,
        payload={
            "start_date": str(start_date),
            "end_date": str(end_date),
            "exchange": str(args.exchange),
            "adjust_mode": str(adjust_mode),
            "lake_root": str(settings.paths.lake_root_path.as_posix()),
            "http_url": str(settings.tushare.http_url or ""),
            "source_view": str(source_view),
            "include_bj": bool(settings.include_bj),
            "universe_min_days_since_listed": int(settings.universe_min_days_since_listed),
            "universe_exclude_st": bool(settings.universe_exclude_st),
            "selected_fact_tables": list(selected_fact_tables),
            "selected_dim_tables": list(selected_dim_tables),
            "refresh_dims": bool(args.refresh_dims),
            "stk_factor_pro_fetch_mode": str(stk_factor_pro_fetch_mode),
            "code_range_ts_codes": list(code_range_ts_codes),
            "index_daily_ts_codes": list(index_daily_ts_codes),
            "index_universe_config": str(index_universe_config),
            "index_universe_names": list(index_universe_names),
            "index_weight_index_codes": list(index_weight_index_codes),
            "index_universe_missing_policy": str(index_universe_missing_policy),
            "index_basic_markets": list(index_basic_markets),
            "code_range_workers": int(code_range_workers),
            "stk_factor_pro_repair_sparse_dates": bool(repair_stk_sparse),
            "range_window_days": int(range_window_days),
            "range_empty_policy": str(range_empty_policy),
            "enforce_range_ann_date_boundary": bool(enforce_range_ann_date_boundary),
            "prune_out_of_range": bool(prune_out_of_range),
        },
    )
    checkpoint_path = get_checkpoint_path(
        paths=settings.paths,
        task_name=checkpoint_task,
        signature=checkpoint_signature,
    )

    if bool(args.reset_checkpoint):
        removed = reset_checkpoint(
            paths=settings.paths,
            task_name=checkpoint_task,
            signature=checkpoint_signature,
        )
        print(f"[update] checkpoint reset: {removed} -> {checkpoint_path.as_posix()}")

    print(f"[update] date range: {start_date} -> {end_date}")
    print(f"[update] selected fact tables: {selected_fact_tables} (groups={sorted(FACT_GROUP_TABLES.keys())})")
    if bool(args.refresh_dims):
        print(f"[update] selected dim tables: {selected_dim_tables} (groups={sorted(DIM_GROUP_TABLES.keys())})")
    else:
        print("[update] refresh-dims disabled")
    if not str(args.start_date).strip():
        print(f"[update] auto start-date resolved with lookback_trade_days={lookback_trade_days}")
    if "stk_factor_pro" in selected_fact_tables:
        print(f"[update] stk_factor_pro_fetch_mode={stk_factor_pro_fetch_mode}")
    if code_range_ts_codes:
        print(f"[update] code_range_ts_codes={code_range_ts_codes}")
    if "index_daily" in selected_code_range_fact_tables:
        print(f"[update] index_daily_ts_codes={index_daily_ts_codes}")
    if "index_weight" in selected_code_range_fact_tables:
        if index_weight_index_codes:
            print(f"[update] index_weight_index_codes={index_weight_index_codes}")
        else:
            print(f"[update] index_universe_names={index_universe_names} config={index_universe_config}")
    if bool(args.refresh_dims) and "index_basic" in selected_dim_tables:
        print(f"[update] index_basic_markets={index_basic_markets}")
    if selected_code_range_fact_tables:
        print(f"[update] code_range_workers={code_range_workers}")
    if repair_stk_sparse:
        print("[update] stk_factor_pro_repair_sparse_dates=True")
    if effective_flush_trade_days != flush_trade_days:
        print(
            f"[update] effective flush_trade_days={effective_flush_trade_days} "
            f"(requested={flush_trade_days}, wide table memory guard)"
        )
    need_trade_bundle = bool(selected_trade_fact_tables) or (
        bool(args.refresh_dims) and ("trade_cal" in selected_dim_tables)
    )
    trade_cal = pd.DataFrame()
    open_trade_dates: list[str] = []
    trade_calendar_source = "not_required"
    if need_trade_bundle:
        trade_cal, open_trade_dates, trade_calendar_source = client.fetch_trade_calendar_bundle(
            start_date=start_date,
            end_date=end_date,
            exchange=str(args.exchange),
        )
    print(f"[update] trade day source: {trade_calendar_source}")
    if not open_trade_dates and selected_trade_fact_tables:
        print("[update] no open trade dates in range, nothing to update")
        if not selected_range_fact_tables and not selected_code_range_fact_tables:
            return

    total_target_days = int(len(open_trade_dates)) if selected_trade_fact_tables else 0
    completed_set: set[str] = set()
    checkpoint_payload: dict[str, object] = {
        "status": "running",
        "total_trade_dates": int(total_target_days),
        "completed_trade_dates": [],
        "start_date": str(start_date),
        "end_date": str(end_date),
        "exchange": str(args.exchange),
        "adjust_mode": str(adjust_mode),
        "selected_fact_tables": list(selected_fact_tables),
        "selected_dim_tables": list(selected_dim_tables),
        "stk_factor_pro_fetch_mode": str(stk_factor_pro_fetch_mode),
        "code_range_ts_codes": list(code_range_ts_codes),
        "index_daily_ts_codes": list(index_daily_ts_codes),
        "index_basic_markets": list(index_basic_markets),
        "code_range_workers": int(code_range_workers),
        "stk_factor_pro_repair_sparse_dates": bool(repair_stk_sparse),
        "range_window_days": int(range_window_days),
        "range_empty_policy": str(range_empty_policy),
        "enforce_range_ann_date_boundary": bool(enforce_range_ann_date_boundary),
        "prune_out_of_range": bool(prune_out_of_range),
        "effective_flush_trade_days": int(effective_flush_trade_days),
    }
    if bool(args.resume):
        loaded = load_checkpoint(
            paths=settings.paths,
            task_name=checkpoint_task,
            signature=checkpoint_signature,
        )
        loaded_dates = [str(x) for x in loaded.get("completed_trade_dates", []) if str(x)]
        completed_set = set(loaded_dates)
        checkpoint_payload.update({k: v for k, v in loaded.items() if k not in {"completed_trade_dates"}})
        checkpoint_payload["completed_trade_dates"] = loaded_dates
        print(
            f"[update] resume={args.resume} checkpoint={checkpoint_path.as_posix()} "
            f"completed={len(completed_set)}/{total_target_days}"
        )
    else:
        print("[update] resume disabled; running from scratch")

    pending_trade_dates = (
        [d for d in open_trade_dates if str(d) not in completed_set] if selected_trade_fact_tables else []
    )
    print(
        f"[update] open trade days: {len(open_trade_dates)}, "
        f"target_fact_days: {total_target_days}, pending: {len(pending_trade_dates)}"
    )
    sparse_repair_plan: dict[str, Any] = {}
    if repair_stk_sparse:
        sparse_repair_plan = _build_stk_factor_pro_sparse_repair_plan(
            lake=lake,
            start_date=str(start_date),
            end_date=str(end_date),
            fields_raw=str(args.stk_factor_pro_sparse_fields or ""),
            min_rows=max(0, int(args.stk_factor_pro_sparse_min_rows)),
            max_finite_rate=max(0.0, float(args.stk_factor_pro_sparse_max_finite_rate)),
            output_csv=str(args.stk_factor_pro_sparse_output_csv or ""),
        )
        print(
            f"[update][stk_factor_pro_sparse] {json.dumps(sparse_repair_plan.get('summary', {}), ensure_ascii=False)}"
        )
    plan = _build_execution_plan(
        selected_trade_fact_tables=selected_trade_fact_tables,
        selected_code_range_fact_tables=selected_code_range_fact_tables,
        selected_range_fact_tables=selected_range_fact_tables,
        selected_dim_tables=selected_dim_tables,
        refresh_dims=bool(args.refresh_dims),
        start_date=str(start_date),
        end_date=str(end_date),
        open_trade_dates=open_trade_dates,
        pending_trade_dates=pending_trade_dates,
        flush_trade_days=effective_flush_trade_days,
        range_window_days=range_window_days,
        prune_out_of_range=prune_out_of_range,
        skip_duckdb=bool(args.skip_duckdb),
        trade_calendar_source=trade_calendar_source,
        need_trade_bundle=need_trade_bundle,
        code_range_ts_code_count=_estimate_code_range_ts_code_count(
            lake=lake,
            selected_code_range_fact_tables=selected_code_range_fact_tables,
            code_range_ts_codes=code_range_ts_codes,
            index_daily_ts_codes=index_daily_ts_codes,
            index_weight_index_codes=index_weight_index_codes,
        ),
    )
    _print_execution_plan(plan)
    if bool(args.dry_run):
        print("[update] dry-run enabled, exit before dim/fact pulls")
        return

    snapshot_date = end_date
    if bool(args.refresh_dims):
        print("[update] refreshing dimensions...")
        _refresh_selected_dimensions(
            lake=lake,
            client=client,
            snapshot_date=snapshot_date,
            selected_dim_tables=selected_dim_tables,
            trade_cal=trade_cal,
            index_basic_markets=index_basic_markets,
            index_universe_config=index_universe_config,
            index_universe_names=index_universe_names,
            index_universe_missing_policy=index_universe_missing_policy,
        )

    if selected_trade_fact_tables:
        print(f"[update] loading facts by trade date: {selected_trade_fact_tables}")
    else:
        print("[update] no selected trade-date fact tables, skipping trade-date refresh")
    if selected_range_fact_tables:
        print(f"[update] loading range facts by ann_date: {selected_range_fact_tables}")
    else:
        print("[update] no selected range fact tables")
    if selected_code_range_fact_tables:
        print(f"[update] loading code-range facts: {selected_code_range_fact_tables}")
    else:
        print("[update] no selected code-range fact tables")

    batch_dates: list[str] = []
    fact_frames: dict[str, list[pd.DataFrame]] = {table: [] for table in selected_trade_fact_tables}
    _perm_denied: set[str] = set()

    for idx, trade_date in enumerate(pending_trade_dates, start=1):
        try:
            fetched = _fetch_selected_fact_frames(
                client=client,
                selected_fact_tables=[t for t in selected_trade_fact_tables if t not in _perm_denied],
                trade_date=trade_date,
            )
        except TushareApiError as exc:
            if exc.category == TushareErrorCategory.AUTH:
                _perm_denied.add(exc.api_name)
                print(f"[update] ❌ 跳过无权限接口: {exc.api_name}", file=sys.stderr)
                fetched = {}
            elif exc.category == TushareErrorCategory.RATE_LIMIT:
                print(f"[update] ⏳ 频率限制，跳过本批次: {exc.api_name}", file=sys.stderr)
                fetched = {}
            else:
                raise
        _print_fetch_warnings(fetched=fetched, trade_date=str(trade_date))
        for table, frame in fetched.items():
            fact_frames[table].append(frame)
        batch_dates.append(str(trade_date))

        should_flush = (len(batch_dates) >= effective_flush_trade_days) or (idx == len(pending_trade_dates))
        if not should_flush:
            continue

        flushed = _flush_trade_batch(
            lake=lake,
            adjust_mode=adjust_mode,
            batch_dates=batch_dates,
            selected_fact_tables=selected_trade_fact_tables,
            fact_frames=fact_frames,
        )

        completed_set.update(batch_dates)
        ordered_completed = [d for d in open_trade_dates if str(d) in completed_set]
        checkpoint_payload["status"] = "running"
        checkpoint_payload["completed_trade_dates"] = ordered_completed
        checkpoint_payload["last_completed_trade_date"] = str(ordered_completed[-1]) if ordered_completed else ""
        checkpoint_payload["last_flush_rows"] = dict(flushed)
        if bool(args.resume):
            save_checkpoint(
                paths=settings.paths,
                task_name=checkpoint_task,
                signature=checkpoint_signature,
                payload=checkpoint_payload,
            )

        print(
            f"  [update] flushed {len(batch_dates)} trade days, completed={len(completed_set)}/{len(open_trade_dates)}"
        )

        batch_dates = []
        fact_frames = {table: [] for table in selected_trade_fact_tables}
        _release_process_memory()

    if selected_code_range_fact_tables:
        if repair_stk_sparse:
            code_range_summary = _refresh_sparse_stk_factor_pro_months(
                lake=lake,
                client=client,
                sparse_repair_plan=sparse_repair_plan,
                ts_codes_filter=code_range_ts_codes,
                workers=code_range_workers,
            )
        else:
            code_range_summary = _refresh_code_range_fact_tables(
                lake=lake,
                client=client,
                selected_code_range_fact_tables=selected_code_range_fact_tables,
                start_date=start_date,
                end_date=end_date,
                ts_codes_filter=code_range_ts_codes,
                index_daily_ts_codes=index_daily_ts_codes,
                index_universe_config=index_universe_config,
                index_universe_names=index_universe_names,
                index_weight_index_codes=index_weight_index_codes,
                index_universe_missing_policy=index_universe_missing_policy,
                workers=code_range_workers,
            )
        print(f"[update] code-range fact refresh summary: {code_range_summary}")

    if selected_range_fact_tables:
        range_summary = _refresh_range_fact_tables(
            lake=lake,
            client=client,
            selected_range_fact_tables=selected_range_fact_tables,
            start_date=start_date,
            end_date=end_date,
            range_window_days=range_window_days,
            range_empty_policy=range_empty_policy,
            enforce_ann_date_boundary=enforce_range_ann_date_boundary,
            prune_out_of_range=prune_out_of_range,
        )
        print(f"[update] range fact refresh summary: {range_summary}")

    if bool(args.resume):
        ordered_completed = [d for d in open_trade_dates if str(d) in completed_set]
        checkpoint_payload["status"] = "completed"
        checkpoint_payload["completed_trade_dates"] = ordered_completed
        checkpoint_payload["last_completed_trade_date"] = str(ordered_completed[-1]) if ordered_completed else ""
        save_checkpoint(
            paths=settings.paths,
            task_name=checkpoint_task,
            signature=checkpoint_signature,
            payload=checkpoint_payload,
        )

    if not bool(args.skip_duckdb):
        print("[update] rebuilding duckdb catalog...")
        duckdb_runtime = build_duckdb_runtime_settings(
            duckdb_path=settings.paths.duckdb_path,
            temp_directory=str(args.duckdb_temp_directory or "").strip(),
            isolate_run=bool(args.duckdb_temp_isolate_run),
            run_id=str(args.duckdb_temp_run_id or "").strip(),
            run_prefix="run_update_duckdb_catalog",
            memory_limit=str(args.duckdb_memory_limit or "").strip(),
            threads=max(0, int(args.duckdb_threads)),
            max_temp_directory_size=str(args.duckdb_max_temp_directory_size or "").strip(),
        )
        catalog_out = build_duckdb_catalog(
            paths=settings.paths,
            source_view=source_view,
            field_catalog_version=field_catalog_version,
            adjust_mode=adjust_mode,
            universe_min_days_since_listed=int(settings.universe_min_days_since_listed),
            universe_exclude_st=bool(settings.universe_exclude_st),
            include_bj=bool(settings.include_bj),
            tradable_require_close=bool(settings.tradable_require_close),
            tradable_require_positive_volume=bool(settings.tradable_require_positive_volume),
            tradable_require_positive_amount=bool(settings.tradable_require_positive_amount),
            materialize_project_base=not bool(args.no_materialize_project_base),
            duckdb_settings=duckdb_runtime,
            field_catalog_enabled_categories=tuple(settings.field_catalog_enabled_categories),
            field_catalog_non_searchable_fields=tuple(settings.field_catalog_non_searchable_fields),
        )
        print("[update] duckdb catalog:", catalog_out)

    if _perm_denied:
        print(
            f"\n[update] ⚠️ 以下接口因权限不足被跳过: {sorted(_perm_denied)}\n"
            f"  如需这些数据，请升级 Tushare 积分后重新运行\n"
            f"  参考: docs/tushare_points_reference.md",
            file=sys.stderr,
        )

    print("[update] done")
    print(
        {
            "open_trade_days": int(len(open_trade_dates)),
            "completed_trade_days": int(len(completed_set)),
            "pending_trade_days": int(len(pending_trade_dates)),
            "flush_trade_days": int(flush_trade_days),
            "adjust_mode": adjust_mode,
            "checkpoint_path": str(checkpoint_path.as_posix()),
        }
    )


def _fetch_fact_frame(client: Any, table: str, trade_date: str) -> pd.DataFrame:
    table_key = str(table or "").strip().lower()
    if table_key == "daily":
        return client.fetch_daily_by_trade_date(trade_date)
    if table_key == "daily_basic":
        return client.fetch_daily_basic_by_trade_date(trade_date)
    if table_key == "adj_factor":
        return client.fetch_adj_factor_by_trade_date(trade_date)
    if table_key == "stk_limit":
        return client.fetch_stk_limit_by_trade_date(trade_date)
    if table_key == "suspend_d":
        return client.fetch_suspend_d_by_trade_date(trade_date)
    if table_key == "moneyflow":
        return client.fetch_moneyflow_by_trade_date(trade_date)
    if table_key == "moneyflow_ths":
        return client.fetch_moneyflow_ths_by_trade_date(trade_date)
    if table_key == "stk_factor_pro":
        return client.fetch_stk_factor_pro_by_trade_date(trade_date)
    if table_key == "stk_auction_o":
        return client.fetch_stk_auction_o_by_trade_date(trade_date)
    if table_key == "stk_auction_c":
        return client.fetch_stk_auction_c_by_trade_date(trade_date)
    if table_key in set(_RANGE_FACT_TABLES):
        raise ValueError(f"Range fact table '{table_key}' must be fetched with date range")
    raise ValueError(f"Unsupported fact table: {table}")


def _fetch_code_range_fact_frame(
    client: Any,
    table: str,
    ts_code: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    table_key = str(table or "").strip().lower()
    if table_key == "cyq_perf":
        return client.fetch_cyq_perf_by_ts_code(
            ts_code=str(ts_code),
            start_date=start_date,
            end_date=end_date,
        )
    if table_key == "cyq_chips":
        return client.fetch_cyq_chips_by_ts_code(
            ts_code=str(ts_code),
            start_date=start_date,
            end_date=end_date,
        )
    if table_key == "stk_factor_pro":
        return client.fetch_stk_factor_pro_by_ts_code(
            ts_code=str(ts_code),
            start_date=start_date,
            end_date=end_date,
        )
    if table_key == "index_daily":
        return client.fetch_index_daily_by_ts_code(
            ts_code=str(ts_code),
            start_date=start_date,
            end_date=end_date,
        )
    if table_key == "index_weight":
        return client.fetch_index_weight_by_index_code(
            index_code=str(ts_code),
            start_date=start_date,
            end_date=end_date,
        )
    raise ValueError(f"Unsupported code-range fact table: {table}")


def _fetch_selected_fact_frames(
    client: Any,
    selected_fact_tables: list[str],
    trade_date: str,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for table in selected_fact_tables:
        out[str(table)] = _fetch_fact_frame(client=client, table=table, trade_date=trade_date)
    return out


def _print_fetch_warnings(fetched: dict[str, pd.DataFrame], trade_date: str) -> None:
    for table, frame in (fetched or {}).items():
        if str(table) != "stk_factor_pro" or not isinstance(frame, pd.DataFrame):
            continue
        skipped = frame.attrs.get("skipped_stk_factor_pro_fields", [])
        if skipped:
            print(
                f"[update][warn] stk_factor_pro skipped fields trade_date={trade_date} "
                f"fields={','.join(str(x) for x in skipped)}"
            )


def _build_curated_fact_frames(
    selected_fact_tables: list[str],
    raw_table_data: dict[str, pd.DataFrame],
    adjust_mode: str,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    if "daily" in selected_fact_tables:
        out["daily"] = curate_market_daily(
            raw_table_data.get("daily", pd.DataFrame()),
            raw_table_data.get("adj_factor", pd.DataFrame()),
            adjust_mode=adjust_mode,
        )
    if "daily_basic" in selected_fact_tables:
        out["daily_basic"] = curate_market_daily_basic(raw_table_data.get("daily_basic", pd.DataFrame()))
    if "adj_factor" in selected_fact_tables:
        out["adj_factor"] = curate_market_adj_factor(raw_table_data.get("adj_factor", pd.DataFrame()))
    if "stk_limit" in selected_fact_tables:
        out["stk_limit"] = curate_market_stk_limit(raw_table_data.get("stk_limit", pd.DataFrame()))
    if "suspend_d" in selected_fact_tables:
        out["suspend_d"] = curate_market_suspend_d(raw_table_data.get("suspend_d", pd.DataFrame()))
    if "moneyflow" in selected_fact_tables:
        out["moneyflow"] = curate_moneyflow(raw_table_data.get("moneyflow", pd.DataFrame()))
    if "moneyflow_ths" in selected_fact_tables:
        out["moneyflow_ths"] = curate_moneyflow_ths(raw_table_data.get("moneyflow_ths", pd.DataFrame()))
    if "cyq_perf" in selected_fact_tables:
        out["cyq_perf"] = curate_cyq_perf(raw_table_data.get("cyq_perf", pd.DataFrame()))
    if "stk_factor_pro" in selected_fact_tables:
        out["stk_factor_pro"] = curate_stk_factor_pro(
            raw_table_data.get("stk_factor_pro", pd.DataFrame()),
            adjust_mode=adjust_mode,
        )
    if "stk_auction_o" in selected_fact_tables:
        out["stk_auction_o"] = curate_stk_auction_o(raw_table_data.get("stk_auction_o", pd.DataFrame()))
    if "stk_auction_c" in selected_fact_tables:
        out["stk_auction_c"] = curate_stk_auction_c(raw_table_data.get("stk_auction_c", pd.DataFrame()))
    return out


def _curate_code_range_fact_table(table: str, raw_df: pd.DataFrame) -> pd.DataFrame:
    table_key = str(table or "").strip().lower()
    if table_key == "cyq_perf":
        return curate_cyq_perf(raw_df)
    if table_key == "cyq_chips":
        long_df = curate_cyq_chips(raw_df)
        return {"long": long_df, "daily": aggregate_cyq_chips_daily(long_df)}
    if table_key == "stk_factor_pro":
        return curate_stk_factor_pro(raw_df)
    if table_key == "index_daily":
        return curate_index_daily(raw_df)
    if table_key == "index_weight":
        return curate_index_weight(raw_df)
    raise ValueError(f"Unsupported code-range fact table: {table}")


def _fetch_range_fact_frame(
    client: Any,
    table: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    table_key = str(table or "").strip().lower()
    if table_key == "income_vip":
        return client.fetch_income_vip(start_date=start_date, end_date=end_date)
    if table_key == "balancesheet_vip":
        return client.fetch_balancesheet_vip(start_date=start_date, end_date=end_date)
    if table_key == "cashflow_vip":
        return client.fetch_cashflow_vip(start_date=start_date, end_date=end_date)
    if table_key == "fina_indicator_vip":
        return client.fetch_fina_indicator_vip(start_date=start_date, end_date=end_date)
    if table_key == "report_rc":
        return client.fetch_report_rc(start_date=start_date, end_date=end_date)
    raise ValueError(f"Unsupported range fact table: {table}")


def _fetch_range_fact_frame_windowed(
    client: Any,
    table: str,
    start_date: str,
    end_date: str,
    range_window_days: int,
) -> dict[str, Any]:
    windows = _iter_date_windows(
        start_date=start_date,
        end_date=end_date,
        window_days=max(1, int(range_window_days)),
    )
    frames: list[pd.DataFrame] = []
    window_rows: list[int] = []
    api_calls = 0
    for idx, (win_start, win_end) in enumerate(windows, start=1):
        frame = _fetch_range_fact_frame(
            client=client,
            table=table,
            start_date=win_start,
            end_date=win_end,
        )
        api_calls += 1
        rows = int(len(frame)) if isinstance(frame, pd.DataFrame) else 0
        window_rows.append(rows)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frames.append(frame)
        if idx % 10 == 0 or idx == len(windows):
            print(
                f"  [update][range] table={table} window={idx}/{len(windows)} rows={rows} span={win_start}->{win_end}"
            )

    return {
        "raw_df": _concat_frames(frames),
        "api_calls": int(api_calls),
        "window_count": int(len(windows)),
        "window_rows": window_rows,
    }


def _curate_range_fact_table(table: str, raw_df: pd.DataFrame) -> pd.DataFrame:
    table_key = str(table or "").strip().lower()
    if table_key == "income_vip":
        return curate_finance_income_vip(raw_df)
    if table_key == "balancesheet_vip":
        return curate_finance_balancesheet_vip(raw_df)
    if table_key == "cashflow_vip":
        return curate_finance_cashflow_vip(raw_df)
    if table_key == "fina_indicator_vip":
        return curate_finance_fina_indicator_vip(raw_df)
    if table_key == "report_rc":
        return curate_report_rc_daily(raw_df)
    raise ValueError(f"Unsupported range fact table: {table}")


def _enforce_ann_date_range(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    enabled: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if df is None or df.empty:
        return pd.DataFrame(), {
            "enabled": bool(enabled),
            "has_ann_date": False,
            "dropped_rows": 0,
            "kept_rows": 0,
        }
    if not bool(enabled):
        return df.copy(), {
            "enabled": False,
            "has_ann_date": "ann_date" in df.columns,
            "dropped_rows": 0,
            "kept_rows": int(len(df)),
        }
    if "ann_date" not in df.columns:
        return df.copy(), {
            "enabled": True,
            "has_ann_date": False,
            "dropped_rows": 0,
            "kept_rows": int(len(df)),
        }

    work = df.copy()
    ann = pd.to_datetime(work["ann_date"], errors="coerce")
    start_dt = pd.to_datetime(str(start_date), errors="coerce")
    end_dt = pd.to_datetime(str(end_date), errors="coerce")
    valid = ann.notna()
    if pd.notna(start_dt):
        valid = valid & (ann >= start_dt)
    if pd.notna(end_dt):
        valid = valid & (ann <= end_dt)

    out = work.loc[valid].copy()
    dropped = int(len(work) - len(out))
    return out, {
        "enabled": True,
        "has_ann_date": True,
        "dropped_rows": dropped,
        "kept_rows": int(len(out)),
    }


def _dedupe_range_rows(df: pd.DataFrame, key_cols: tuple[str, ...]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    work = df.copy()
    dedupe_cols = [str(c) for c in key_cols if str(c) in work.columns]
    if dedupe_cols:
        work = work.drop_duplicates(subset=dedupe_cols, keep="last")
    else:
        work = work.drop_duplicates(keep="last")
    return work.reset_index(drop=True)


def _iter_date_windows(start_date: str, end_date: str, window_days: int) -> list[tuple[str, str]]:
    start_dt = _coerce_date(start_date)
    end_dt = _coerce_date(end_date)
    if start_dt > end_dt:
        raise ValueError(f"start_date > end_date for range windows: {start_date} > {end_date}")
    step = max(1, int(window_days))
    windows: list[tuple[str, str]] = []
    cur = start_dt
    while cur <= end_dt:
        win_end = min(cur + timedelta(days=step - 1), end_dt)
        windows.append((_fmt_ymd(cur), _fmt_ymd(win_end)))
        cur = win_end + timedelta(days=1)
    return windows


def _prune_range_table_out_of_bound_records(
    lake: ParquetLake,
    table: str,
    vendor_spec: dict[str, object],
    curated_spec: dict[str, object],
    end_date: str,
    enabled: bool,
) -> dict[str, Any]:
    if not bool(enabled):
        return {
            "enabled": False,
            "vendor": {},
            "curated": {},
            "skipped_reason": "disabled",
        }
    if not hasattr(lake, "vendor_table_root") or not hasattr(lake, "curated_table_root"):
        return {
            "enabled": False,
            "vendor": {},
            "curated": {},
            "skipped_reason": "lake_path_api_unavailable",
        }

    vendor_root = lake.vendor_table_root(str(vendor_spec["table"]))
    curated_root = lake.curated_table_root(str(curated_spec["table"]))
    vendor_stats = _prune_partitioned_table_after_end_date(
        root=vendor_root,
        date_col=str(vendor_spec["date_col"]),
        end_date=end_date,
    )
    curated_stats = _prune_partitioned_table_after_end_date(
        root=curated_root,
        date_col=str(curated_spec["date_col"]),
        end_date=end_date,
    )
    if int(vendor_stats.get("dropped_rows", 0)) > 0 or int(curated_stats.get("dropped_rows", 0)) > 0:
        print(
            f"[update][range] pruned out-of-range rows table={table} "
            f"vendor_dropped={vendor_stats.get('dropped_rows', 0)} "
            f"curated_dropped={curated_stats.get('dropped_rows', 0)} "
            f"cutoff={end_date}"
        )
    return {"enabled": True, "vendor": vendor_stats, "curated": curated_stats}


def _prune_partitioned_table_after_end_date(
    root: Path,
    date_col: str,
    end_date: str,
) -> dict[str, int | str]:
    out: dict[str, int | str] = {
        "files_scanned": 0,
        "files_rewritten": 0,
        "files_deleted": 0,
        "dropped_rows": 0,
        "cutoff_date": str(end_date),
    }
    if not root.exists():
        return out

    end_dt = pd.to_datetime(str(end_date), errors="coerce")
    if pd.isna(end_dt):
        return out

    parquet_files = sorted(root.rglob("*.parquet"))
    for parquet_path in parquet_files:
        out["files_scanned"] = int(out["files_scanned"]) + 1
        try:
            df = pd.read_parquet(parquet_path)
        except Exception:
            continue
        if df is None or df.empty or str(date_col) not in df.columns:
            continue

        series = pd.to_datetime(df[str(date_col)], errors="coerce")
        keep_mask = series.isna() | (series <= end_dt)
        dropped_rows = int((~keep_mask).sum())
        if dropped_rows <= 0:
            continue

        kept = df.loc[keep_mask].copy()
        out["dropped_rows"] = int(out["dropped_rows"]) + dropped_rows
        if kept.empty:
            parquet_path.unlink(missing_ok=True)
            out["files_deleted"] = int(out["files_deleted"]) + 1
            continue

        _atomic_replace_parquet(path=parquet_path, df=kept)
        out["files_rewritten"] = int(out["files_rewritten"]) + 1

    return out


def _atomic_replace_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    if tmp.exists():
        tmp.unlink()
    df.to_parquet(tmp, index=False)
    if path.exists():
        path.unlink()
    tmp.replace(path)


def _coerce_date(value: str | date | datetime) -> date:
    dt = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(dt):
        raise ValueError(f"Invalid date: {value}")
    return pd.Timestamp(dt).date()


def _fmt_ymd(value: date) -> str:
    return value.strftime("%Y-%m-%d")


def _refresh_range_fact_tables(
    lake: ParquetLake,
    client: Any,
    selected_range_fact_tables: list[str],
    start_date: str,
    end_date: str,
    range_window_days: int = 180,
    range_empty_policy: str = "warn",
    enforce_ann_date_boundary: bool = True,
    prune_out_of_range: bool = False,
) -> dict[str, Any]:
    policy = str(range_empty_policy or "warn").strip().lower()
    if policy not in {"warn", "fail"}:
        raise ValueError(f"Unsupported range_empty_policy: {range_empty_policy}")

    summary: dict[str, Any] = {
        "tables": {},
        "start_date": str(start_date),
        "end_date": str(end_date),
        "range_window_days": int(max(1, int(range_window_days))),
        "enforce_ann_date_boundary": bool(enforce_ann_date_boundary),
        "prune_out_of_range": bool(prune_out_of_range),
        "empty_policy": policy,
    }
    if not selected_range_fact_tables:
        return summary

    for table in selected_range_fact_tables:
        vendor_spec = _FACT_VENDOR_SPECS[table]
        curated_spec = _FACT_CURATED_SPECS[table]
        pre_prune = _prune_range_table_out_of_bound_records(
            lake=lake,
            table=table,
            vendor_spec=vendor_spec,
            curated_spec=curated_spec,
            end_date=end_date,
            enabled=bool(prune_out_of_range),
        )
        raw_rows_total = 0
        table_rows = 0
        curated_rows_total = 0
        api_calls = 0
        window_rows: list[int] = []
        ann_filter_dropped = 0
        ann_filter_kept = 0
        ann_filter_has_ann_date = False
        last_trade_date = ""
        windows = _iter_date_windows(
            start_date=start_date,
            end_date=end_date,
            window_days=max(1, int(range_window_days)),
        )
        for idx, (win_start, win_end) in enumerate(windows, start=1):
            raw_df = _fetch_range_fact_frame(
                client=client,
                table=table,
                start_date=win_start,
                end_date=win_end,
            )
            api_calls += 1
            raw_rows = int(len(raw_df)) if isinstance(raw_df, pd.DataFrame) else 0
            raw_rows_total += raw_rows
            window_rows.append(raw_rows)

            raw_df_filtered, filter_meta = _enforce_ann_date_range(
                df=raw_df,
                start_date=start_date,
                end_date=end_date,
                enabled=bool(enforce_ann_date_boundary),
            )
            raw_df_filtered = _dedupe_range_rows(
                df=raw_df_filtered,
                key_cols=tuple(vendor_spec["key_cols"]),
            )
            ann_filter_dropped += int(filter_meta.get("dropped_rows", 0) or 0)
            ann_filter_kept += int(len(raw_df_filtered))
            ann_filter_has_ann_date = bool(ann_filter_has_ann_date or filter_meta.get("has_ann_date", False))

            if not raw_df_filtered.empty:
                curated_df = _curate_range_fact_table(table=table, raw_df=raw_df_filtered)
                lake.write_vendor_trade_table(
                    table=str(vendor_spec["table"]),
                    df=raw_df_filtered,
                    date_col=str(vendor_spec["date_col"]),
                    key_cols=tuple(vendor_spec["key_cols"]),
                    mode="upsert",
                )
                lake.write_curated_trade_table(
                    table=str(curated_spec["table"]),
                    df=curated_df,
                    date_col=str(curated_spec["date_col"]),
                    key_cols=tuple(curated_spec["key_cols"]),
                    mode="upsert",
                )
                table_rows += int(len(raw_df_filtered))
                curated_rows_total += int(len(curated_df))
                last_trade_date = _infer_last_date_from_frame(
                    raw_df_filtered,
                    date_col=str(vendor_spec["date_col"]),
                    fallback=win_end,
                )
                raw_df_filtered = pd.DataFrame()
                curated_df = pd.DataFrame()
                _release_process_memory()
            else:
                _release_process_memory()

            if idx % 10 == 0 or idx == len(windows):
                print(
                    f"  [update][range] table={table} window={idx}/{len(windows)} "
                    f"rows={raw_rows} span={win_start}->{win_end}"
                )

        if table_rows <= 0:
            lake.write_vendor_trade_table(
                table=str(vendor_spec["table"]),
                df=pd.DataFrame(),
                date_col=str(vendor_spec["date_col"]),
                key_cols=tuple(vendor_spec["key_cols"]),
                mode="upsert",
            )
            lake.write_curated_trade_table(
                table=str(curated_spec["table"]),
                df=pd.DataFrame(),
                date_col=str(curated_spec["date_col"]),
                key_cols=tuple(curated_spec["key_cols"]),
                mode="upsert",
            )

        state_updated = False
        table_extra = {
            "mode": "ann_date_range",
            "start_date": str(start_date),
            "end_date": str(end_date),
            "range_window_days": int(max(1, int(range_window_days))),
            "api_calls": int(api_calls),
            "range_windows": int(len(windows)),
            "enforce_ann_date_boundary": bool(enforce_ann_date_boundary),
            "pre_prune": dict(pre_prune),
        }
        if table_rows > 0:
            lake.update_ingestion_state(
                table=table,
                last_trade_date=last_trade_date,
                row_count=table_rows,
                extra=table_extra,
                allow_rewind=bool(enforce_ann_date_boundary),
            )
            state_updated = True
        else:
            last_trade_date = ""
            if policy == "fail":
                raise RuntimeError(
                    f"Range fact table '{table}' returned 0 rows in [{start_date}, {end_date}] "
                    f"after boundary filter; abort due to --range-empty-policy=fail"
                )
            print(
                f"[update][warn] range fact '{table}' is empty in [{start_date}, {end_date}] "
                f"(policy=warn, ingestion_state not advanced)"
            )
        summary["tables"][table] = {
            "raw_rows": int(raw_rows_total),
            "raw_rows_filtered": table_rows,
            "curated_rows": int(curated_rows_total),
            "last_trade_date": str(last_trade_date),
            "state_updated": bool(state_updated),
            "api_calls": int(api_calls),
            "range_windows": int(len(windows)),
            "window_rows": list(window_rows),
            "ann_date_filter": {
                "enabled": bool(enforce_ann_date_boundary),
                "has_ann_date": bool(ann_filter_has_ann_date),
                "dropped_rows": int(ann_filter_dropped),
                "kept_rows": int(ann_filter_kept),
            },
            "pre_prune": dict(pre_prune),
        }

    return summary


def _refresh_code_range_fact_tables(
    lake: ParquetLake,
    client: Any,
    selected_code_range_fact_tables: list[str],
    start_date: str,
    end_date: str,
    ts_codes_filter: list[str] | None = None,
    index_daily_ts_codes: list[str] | None = None,
    index_universe_config: str = "configs/index_universes.yaml",
    index_universe_names: list[str] | None = None,
    index_weight_index_codes: list[str] | None = None,
    index_universe_missing_policy: str = "warn",
    workers: int = 1,
) -> dict[str, Any]:
    stock_ts_codes: list[str] | None = None
    index_codes = _resolve_index_daily_ts_codes(index_daily_ts_codes or [])
    index_weight_codes: list[str] | None = None
    worker_count = max(1, int(workers))
    summary: dict[str, Any] = {
        "tables": {},
        "start_date": str(start_date),
        "end_date": str(end_date),
        "flush_codes": int(_CODE_RANGE_FLUSH_CODES),
        "workers": int(worker_count),
    }
    if not selected_code_range_fact_tables:
        return summary

    for table in selected_code_range_fact_tables:
        table_start_date = _effective_code_range_start_date(table=table, start_date=start_date)
        if str(table) == "index_daily":
            ts_codes = list(index_codes)
        elif str(table) == "index_weight":
            if index_weight_codes is None:
                index_weight_codes = _resolve_index_weight_codes_for_refresh(
                    lake=lake,
                    config_path=index_universe_config,
                    universe_names=index_universe_names,
                    explicit_codes=index_weight_index_codes,
                    missing_policy=index_universe_missing_policy,
                )
            ts_codes = list(index_weight_codes)
        else:
            if stock_ts_codes is None:
                stock_ts_codes = _resolve_code_range_ts_codes(
                    lake=lake,
                    client=client,
                    ts_codes_filter=ts_codes_filter,
                )
            ts_codes = list(stock_ts_codes)
        if not ts_codes:
            raise RuntimeError(
                "No ts_codes available for code-range fact table "
                f"'{table}'. Refresh stock_basic dimensions or pass explicit codes."
            )
        vendor_spec = _FACT_VENDOR_SPECS[table]
        curated_spec = _FACT_CURATED_SPECS[table]
        raw_rows = 0
        curated_rows = 0
        api_calls = 0
        batch_frames: list[pd.DataFrame] = []
        progress_interval = 20 if str(table) == "stk_factor_pro" else 100
        print(
            f"  [update][code-range] table={table} start codes=0/{len(ts_codes)} "
            f"span={table_start_date}->{end_date} workers={worker_count}"
        )
        if worker_count <= 1:
            for idx, ts_code in enumerate(ts_codes, start=1):
                frame = _fetch_code_range_fact_frame(
                    client=client,
                    table=table,
                    ts_code=ts_code,
                    start_date=table_start_date,
                    end_date=end_date,
                )
                api_calls += 1
                if isinstance(frame, pd.DataFrame) and not frame.empty:
                    batch_frames.append(frame)
                if len(batch_frames) >= _CODE_RANGE_FLUSH_CODES or idx == len(ts_codes):
                    flushed = _flush_code_range_batch(
                        lake=lake,
                        table=table,
                        raw_frames=batch_frames,
                        vendor_spec=vendor_spec,
                        curated_spec=curated_spec,
                    )
                    raw_rows += int(flushed.get("raw_rows", 0))
                    curated_rows += int(flushed.get("curated_rows", 0))
                    batch_frames = []
                if idx % progress_interval == 0 or idx == len(ts_codes):
                    print(f"  [update][code-range] table={table} codes={idx}/{len(ts_codes)} api_calls={api_calls}")
        else:
            in_flight_limit = min(len(ts_codes), max(worker_count, worker_count * 2))
            code_iter = iter(ts_codes)
            completed_codes = 0
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures: dict[Any, str] = {}

                def submit_next() -> None:
                    try:
                        code = next(code_iter)
                    except StopIteration:
                        return
                    future = executor.submit(
                        _fetch_code_range_fact_frame,
                        client,
                        table,
                        code,
                        table_start_date,
                        end_date,
                    )
                    futures[future] = code

                for _ in range(in_flight_limit):
                    submit_next()

                while futures:
                    done, _pending = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        _code = futures.pop(future)
                        frame = future.result()
                        api_calls += 1
                        completed_codes += 1
                        if isinstance(frame, pd.DataFrame) and not frame.empty:
                            batch_frames.append(frame)
                        if len(batch_frames) >= _CODE_RANGE_FLUSH_CODES:
                            flushed = _flush_code_range_batch(
                                lake=lake,
                                table=table,
                                raw_frames=batch_frames,
                                vendor_spec=vendor_spec,
                                curated_spec=curated_spec,
                            )
                            raw_rows += int(flushed.get("raw_rows", 0))
                            curated_rows += int(flushed.get("curated_rows", 0))
                            batch_frames = []
                        if completed_codes % progress_interval == 0 or completed_codes == len(ts_codes):
                            print(
                                f"  [update][code-range] table={table} "
                                f"codes={completed_codes}/{len(ts_codes)} api_calls={api_calls}"
                            )
                        submit_next()
            if batch_frames:
                flushed = _flush_code_range_batch(
                    lake=lake,
                    table=table,
                    raw_frames=batch_frames,
                    vendor_spec=vendor_spec,
                    curated_spec=curated_spec,
                )
                raw_rows += int(flushed.get("raw_rows", 0))
                curated_rows += int(flushed.get("curated_rows", 0))
                batch_frames = []

        lake.update_ingestion_state(
            table=table,
            last_trade_date=_fmt_trade_date(end_date),
            row_count=int(raw_rows),
            extra={
                "mode": "ts_code_range",
                "start_date": str(table_start_date),
                "end_date": str(end_date),
                "api_calls": int(api_calls),
                "ts_code_count": int(len(ts_codes)),
            },
            allow_rewind=True,
        )
        summary["tables"][table] = {
            "raw_rows": int(raw_rows),
            "curated_rows": int(curated_rows),
            "api_calls": int(api_calls),
            "ts_code_count": int(len(ts_codes)),
            "workers": int(worker_count),
        }
    return summary


def _refresh_sparse_stk_factor_pro_months(
    lake: ParquetLake,
    client: Any,
    sparse_repair_plan: dict[str, Any],
    ts_codes_filter: list[str] | None,
    workers: int,
) -> dict[str, Any]:
    windows = list(sparse_repair_plan.get("windows", []) or [])
    if not windows:
        return {
            "tables": {"stk_factor_pro": {"status": "skipped_no_sparse_dates"}},
            "sparse_summary": dict(sparse_repair_plan.get("summary", {})),
        }

    month_summaries: list[dict[str, Any]] = []
    raw_rows = 0
    curated_rows = 0
    api_calls = 0
    for idx, window in enumerate(windows, start=1):
        win_start = str(window.get("start_date", "") or "")
        win_end = str(window.get("end_date", "") or "")
        month = str(window.get("month", "") or "")
        print(
            f"[update][stk_factor_pro_sparse] repairing month={month} "
            f"window={win_start}->{win_end} ({idx}/{len(windows)})"
        )
        summary = _refresh_code_range_fact_tables(
            lake=lake,
            client=client,
            selected_code_range_fact_tables=["stk_factor_pro"],
            start_date=win_start,
            end_date=win_end,
            ts_codes_filter=ts_codes_filter,
            workers=workers,
        )
        table_summary = dict((summary.get("tables") or {}).get("stk_factor_pro", {}))
        raw_rows += int(table_summary.get("raw_rows", 0) or 0)
        curated_rows += int(table_summary.get("curated_rows", 0) or 0)
        api_calls += int(table_summary.get("api_calls", 0) or 0)
        month_summaries.append(
            {
                "month": month,
                "start_date": win_start,
                "end_date": win_end,
                **table_summary,
            }
        )

    return {
        "tables": {
            "stk_factor_pro": {
                "raw_rows": int(raw_rows),
                "curated_rows": int(curated_rows),
                "api_calls": int(api_calls),
                "repaired_months": int(len(windows)),
                "workers": max(1, int(workers)),
            }
        },
        "months": month_summaries,
        "sparse_summary": dict(sparse_repair_plan.get("summary", {})),
    }


def _build_stk_factor_pro_sparse_repair_plan(
    lake: ParquetLake,
    start_date: str,
    end_date: str,
    fields_raw: str,
    min_rows: int,
    max_finite_rate: float,
    output_csv: str,
) -> dict[str, Any]:
    report = _detect_stk_factor_pro_sparse_dates(
        lake=lake,
        start_date=start_date,
        end_date=end_date,
        fields_raw=fields_raw,
        min_rows=min_rows,
        max_finite_rate=max_finite_rate,
    )
    rows = list(report.get("rows", []) or [])
    output_path = str(output_csv or "").strip()
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)

    windows = _month_windows_for_sparse_rows(rows=rows, start_date=start_date, end_date=end_date)
    summary = {
        "sparse_dates": int(len(rows)),
        "sparse_months": int(len({str(row.get("month", "") or "") for row in rows if str(row.get("month", "") or "")})),
        "repair_windows": int(len(windows)),
        "fields_scanned": int(len(report.get("fields", []) or [])),
        "min_rows": int(min_rows),
        "max_finite_rate": float(max_finite_rate),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "output_csv": output_path,
        "windows": [f"{x.get('start_date', '')}->{x.get('end_date', '')}" for x in windows],
    }
    return {
        "rows": rows,
        "windows": windows,
        "summary": summary,
        "fields": list(report.get("fields", []) or []),
    }


def _detect_stk_factor_pro_sparse_dates(
    lake: ParquetLake,
    start_date: str,
    end_date: str,
    fields_raw: str = "",
    min_rows: int = 100,
    max_finite_rate: float = 0.01,
) -> dict[str, Any]:
    root = lake.curated_table_root("facts/stk_factor_pro")
    paths = _month_partition_parquet_paths(root=root, start_date=start_date, end_date=end_date)
    if not paths:
        return {"rows": [], "fields": [], "status": "missing_parquet"}

    try:
        import duckdb  # type: ignore
    except Exception as exc:
        raise RuntimeError("duckdb is required to detect sparse stk_factor_pro dates") from exc

    con = duckdb.connect(database=":memory:")
    try:
        columns = list(
            con.execute(
                "select * from read_parquet(?, union_by_name=true) limit 0",
                [paths],
            )
            .fetchdf()
            .columns
        )
        fields = _resolve_stk_factor_pro_sparse_fields(columns=columns, fields_raw=fields_raw)
        if not fields:
            return {"rows": [], "fields": [], "status": "missing_fields"}

        select_parts = ['cast("date" as DATE) as date', "count(*) as row_count"]
        aliases: dict[str, str] = {}
        for field in fields:
            alias = f"{field}__finite"
            aliases[field] = alias
            select_parts.append(
                f"sum(case when isfinite(try_cast({_quote_ident(field)} as DOUBLE)) then 1 else 0 end) as {_quote_ident(alias)}"
            )
        sql = (
            f"select {', '.join(select_parts)} "
            "from read_parquet(?, union_by_name=true) "
            'where cast("date" as DATE) between cast(? as DATE) and cast(? as DATE) '
            "group by 1 order by 1"
        )
        stats = con.execute(sql, [paths, _fmt_trade_date(start_date), _fmt_trade_date(end_date)]).fetchdf()
    finally:
        con.close()

    rows: list[dict[str, Any]] = []
    if stats.empty:
        return {"rows": rows, "fields": fields, "status": "empty_stats"}

    for _, record in stats.iterrows():
        row_count = int(record.get("row_count", 0) or 0)
        if row_count < int(min_rows):
            continue
        sparse_fields: list[str] = []
        sparse_rates: list[str] = []
        for field, alias in aliases.items():
            finite_count = int(record.get(alias, 0) or 0)
            finite_rate = finite_count / row_count if row_count > 0 else 0.0
            if finite_rate <= float(max_finite_rate):
                sparse_fields.append(field)
                sparse_rates.append(f"{field}:{finite_rate:.6f}")
        if not sparse_fields:
            continue
        dt = pd.to_datetime(record["date"], errors="coerce")
        if pd.isna(dt):
            continue
        rows.append(
            {
                "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                "month": pd.Timestamp(dt).strftime("%Y-%m"),
                "row_count": row_count,
                "sparse_field_count": int(len(sparse_fields)),
                "sparse_fields": ",".join(sparse_fields),
                "sparse_finite_rates": ",".join(sparse_rates),
            }
        )
    return {"rows": rows, "fields": fields, "status": "ok"}


def _flush_code_range_batch(
    lake: ParquetLake,
    table: str,
    raw_frames: list[pd.DataFrame],
    vendor_spec: dict[str, object],
    curated_spec: dict[str, object],
) -> dict[str, int]:
    raw_df = _concat_frames(raw_frames)
    if raw_df.empty:
        return {"raw_rows": 0, "curated_rows": 0}
    curated_df = _curate_code_range_fact_table(table=table, raw_df=raw_df)
    raw_rows = int(len(raw_df))
    lake.write_vendor_trade_table(
        table=str(vendor_spec["table"]),
        df=raw_df,
        date_col=str(vendor_spec["date_col"]),
        key_cols=tuple(vendor_spec["key_cols"]),
        mode="upsert",
    )
    raw_df = pd.DataFrame()
    _release_process_memory()
    if str(table) == "cyq_chips" and isinstance(curated_df, dict):
        long_df = curated_df.get("long", pd.DataFrame())
        daily_df = curated_df.get("daily", pd.DataFrame())
        lake.write_curated_trade_table(
            table=str(curated_spec["table"]),
            df=long_df,
            date_col=str(curated_spec["date_col"]),
            key_cols=tuple(curated_spec["key_cols"]),
            mode="upsert",
        )
        lake.write_curated_trade_table(
            table="facts/cyq_chips_daily",
            df=daily_df,
            date_col="date",
            key_cols=("code", "date"),
            mode="upsert",
        )
        curated_rows = int(len(long_df))
    else:
        curated_rows = int(len(curated_df))
        lake.write_curated_trade_table(
            table=str(curated_spec["table"]),
            df=curated_df,
            date_col=str(curated_spec["date_col"]),
            key_cols=tuple(curated_spec["key_cols"]),
            mode="upsert",
        )
    curated_df = pd.DataFrame()
    _release_process_memory()
    return {"raw_rows": raw_rows, "curated_rows": curated_rows}


def _effective_code_range_start_date(table: str, start_date: str) -> str:
    table_key = str(table or "").strip().lower()
    if table_key != "cyq_chips":
        return str(start_date)
    start_ts = pd.to_datetime(str(start_date), errors="coerce")
    min_ts = pd.to_datetime(_CYQ_CHIPS_START_DATE)
    if pd.isna(start_ts) or start_ts >= min_ts:
        return str(start_date)
    print(f"[update][warn] cyq_chips starts from {_CYQ_CHIPS_START_DATE}; clamping requested start_date={start_date}")
    return _CYQ_CHIPS_START_DATE


def _configure_stdio_for_live_logs() -> None:
    for stream in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:
            pass


def _resolve_code_range_ts_codes(
    lake: ParquetLake,
    client: Any,
    ts_codes_filter: list[str] | None = None,
) -> list[str]:
    filtered = _valid_ts_codes(list(ts_codes_filter or []))
    if filtered:
        return filtered
    codes = _load_lake_ts_codes(lake)
    if codes:
        return codes
    stock_basic = client.fetch_stock_basic()
    if isinstance(stock_basic, pd.DataFrame) and not stock_basic.empty:
        col = "ts_code" if "ts_code" in stock_basic.columns else ("code" if "code" in stock_basic.columns else "")
        if col:
            return _valid_ts_codes(_column_values(stock_basic, col))
    return []


def _resolve_index_daily_ts_codes(raw: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(raw, str):
        tokens = _parse_csv_tokens(raw)
    else:
        tokens = [str(x or "").strip().upper() for x in raw if str(x or "").strip()]
    if not tokens:
        tokens = list(_DEFAULT_INDEX_DAILY_TS_CODES)
    out: list[str] = []
    for token in tokens:
        code = str(token or "").strip().upper()
        if code and code not in out:
            out.append(code)
    return out


def _resolve_index_weight_index_codes(
    raw: str | list[str] | tuple[str, ...],
) -> list[str]:
    if isinstance(raw, str):
        tokens = _parse_csv_tokens(raw)
    else:
        tokens = [str(x or "").strip() for x in raw if str(x or "").strip()]
    out: list[str] = []
    for token in tokens:
        code = normalize_index_code(token)
        if code and code not in out:
            out.append(code)
    return out


def _resolve_index_weight_codes_for_refresh(
    lake: ParquetLake,
    config_path: str,
    universe_names: list[str] | None,
    explicit_codes: list[str] | None,
    missing_policy: str = "warn",
) -> list[str]:
    explicit = _resolve_index_weight_index_codes(explicit_codes or [])
    if explicit:
        return explicit

    dim_universe = _load_latest_snapshot_table(lake=lake, table="dims/index_universe", date_col="snapshot_date")
    if isinstance(dim_universe, pd.DataFrame) and not dim_universe.empty:
        names = {str(x or "").strip().lower() for x in (universe_names or []) if str(x or "").strip()}
        work = dim_universe.copy()
        if names and "universe_name" in work.columns:
            work = work[work["universe_name"].astype(str).str.lower().isin(names)]
        if "status" in work.columns:
            work = work[work["status"].astype(str).str.lower() == "active"]
        if "index_weight_code" in work.columns:
            return _resolve_index_weight_index_codes(_column_values(work, "index_weight_code"))

    specs = load_index_universe_config(config_path)
    index_basic = _load_latest_snapshot_table(lake=lake, table="dims/index_basic", date_col="snapshot_date")
    resolved = resolve_index_universes(
        specs,
        index_basic,
        universe_names=universe_names or list(specs),
        missing_policy=missing_policy,
    )
    active = resolved[resolved["status"].astype(str).str.lower() == "active"]
    return _resolve_index_weight_index_codes(_column_values(active, "index_weight_code"))


def _load_latest_snapshot_table(lake: ParquetLake, table: str, date_col: str = "snapshot_date") -> pd.DataFrame:
    try:
        df = lake.read_curated_table(table, date_col=date_col)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if date_col not in df.columns:
        return df
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    max_date = work[date_col].max()
    if pd.isna(max_date):
        return work
    return work[work[date_col] == max_date].reset_index(drop=True)


def _estimate_code_range_ts_code_count(
    lake: ParquetLake,
    selected_code_range_fact_tables: list[str],
    code_range_ts_codes: list[str],
    index_daily_ts_codes: list[str],
    index_weight_index_codes: list[str] | None = None,
) -> int:
    counts: list[int] = []
    selected = {str(x).strip() for x in selected_code_range_fact_tables}
    if "index_daily" in selected:
        counts.append(len(_resolve_index_daily_ts_codes(index_daily_ts_codes)))
    if "index_weight" in selected:
        counts.append(len(_resolve_index_weight_index_codes(index_weight_index_codes or [])))
    if selected - {"index_daily", "index_weight"}:
        counts.append(len(code_range_ts_codes) if code_range_ts_codes else len(_load_lake_ts_codes(lake)))
    return max(counts) if counts else 0


def _load_lake_ts_codes(lake: ParquetLake) -> list[str]:
    try:
        df = lake.read_curated_table("dims/security_master", date_col="snapshot_date")
    except Exception:
        return []
    if df is None or df.empty or "code" not in df.columns:
        return []
    return _valid_ts_codes(_column_values(df, "code"))


def _valid_ts_codes(codes: list[str]) -> list[str]:
    out: list[str] = []
    for code in codes:
        text = str(code or "").strip().upper()
        if not text.endswith((".SZ", ".SH", ".BJ")):
            continue
        if text not in out:
            out.append(text)
    return out


def _parse_csv_tokens(raw: str) -> list[str]:
    return [token.strip() for token in str(raw or "").split(",") if token.strip()]


def _resolve_stk_factor_pro_sparse_fields(columns: list[str], fields_raw: str) -> list[str]:
    available = {str(col) for col in columns}
    tokens = _parse_csv_tokens(fields_raw)
    if not tokens:
        return sorted(col for col in available if col.startswith("tech_"))
    out: list[str] = []
    for token in tokens:
        field = str(token).strip()
        if not field:
            continue
        if field not in available and not field.startswith("tech_"):
            field = f"tech_{field}"
        if field in available and field not in out:
            out.append(field)
    return out


def _month_partition_parquet_paths(root: Path, start_date: str, end_date: str) -> list[str]:
    if not root.exists():
        return []
    start_dt = _coerce_date(start_date)
    end_dt = _coerce_date(end_date)
    paths: list[str] = []
    cur = date(start_dt.year, start_dt.month, 1)
    end_month = date(end_dt.year, end_dt.month, 1)
    while cur <= end_month:
        part_dir = root / f"year={cur.year:04d}" / f"month={cur.month:02d}"
        if part_dir.exists():
            for path in sorted(part_dir.glob("*.parquet")):
                paths.append(str(path.as_posix()))
        next_month = pd.Timestamp(cur) + pd.DateOffset(months=1)
        cur = date(int(next_month.year), int(next_month.month), 1)
    return paths


def _month_windows_for_sparse_rows(
    rows: list[dict[str, Any]],
    start_date: str,
    end_date: str,
) -> list[dict[str, str]]:
    if not rows:
        return []
    start_dt = pd.Timestamp(_coerce_date(start_date))
    end_dt = pd.Timestamp(_coerce_date(end_date))
    month_values = sorted({str(row.get("month", "") or "") for row in rows if str(row.get("month", "") or "")})
    months = [pd.Timestamp(f"{month}-01") for month in month_values]
    windows: list[dict[str, str]] = []
    if not months:
        return windows

    groups: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    group_start = months[0]
    group_end = months[0]
    for month in months[1:]:
        expected_next = group_end + pd.DateOffset(months=1)
        if month == expected_next:
            group_end = month
            continue
        groups.append((group_start, group_end))
        group_start = month
        group_end = month
    groups.append((group_start, group_end))

    for month_start, last_month_start in groups:
        month_end = last_month_start + pd.offsets.MonthEnd(0)
        win_start = max(month_start, start_dt)
        win_end = min(month_end, end_dt)
        label = month_start.strftime("%Y-%m")
        if last_month_start != month_start:
            label = f"{month_start.strftime('%Y-%m')}..{last_month_start.strftime('%Y-%m')}"
        windows.append(
            {
                "month": label,
                "start_date": win_start.strftime("%Y-%m-%d"),
                "end_date": win_end.strftime("%Y-%m-%d"),
            }
        )
    return windows


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _refresh_selected_dimensions(
    lake: ParquetLake,
    client: Any,
    snapshot_date: str,
    selected_dim_tables: list[str],
    trade_cal: pd.DataFrame,
    index_basic_markets: list[str] | None = None,
    index_universe_config: str = "configs/index_universes.yaml",
    index_universe_names: list[str] | None = None,
    index_universe_missing_policy: str = "warn",
) -> None:
    if not selected_dim_tables:
        print("[update] refresh-dims enabled but selected_dim_tables is empty, skipping")
        return

    dim_data: dict[str, pd.DataFrame] = {}
    if "trade_cal" in selected_dim_tables:
        dim_data["trade_cal"] = trade_cal
    if "stock_basic" in selected_dim_tables:
        dim_data["stock_basic"] = client.fetch_stock_basic()
    if "index_classify" in selected_dim_tables:
        dim_data["index_classify"] = client.fetch_index_classify(src="SW2021")
    if "index_member_all" in selected_dim_tables:
        dim_data["index_member_all"] = client.fetch_index_member_all(src="SW2021")
    if "namechange" in selected_dim_tables:
        dim_data["namechange"] = client.fetch_namechange(end_date=snapshot_date)
    if "index_basic" in selected_dim_tables:
        dim_data["index_basic"] = _fetch_index_basic_markets(
            client=client,
            markets=list(index_basic_markets or []),
        )
    if "ths_index" in selected_dim_tables:
        dim_data["ths_index"] = client.fetch_ths_index()
    if "ths_member" in selected_dim_tables:
        ths_index_for_member = dim_data.get("ths_index", pd.DataFrame())
        if ths_index_for_member.empty:
            ths_index_for_member = client.fetch_ths_index()
            dim_data.setdefault("ths_index", ths_index_for_member)
        dim_data["ths_member"] = client.fetch_ths_member(ts_codes=_column_values(ths_index_for_member, "ts_code"))

    if "trade_cal" in selected_dim_tables:
        lake.write_vendor_snapshot(
            table="trade_cal",
            snapshot_date=snapshot_date,
            df=dim_data.get("trade_cal", pd.DataFrame()),
        )
        lake.write_curated_snapshot(
            table="dims/trade_calendar",
            snapshot_date=snapshot_date,
            df=curate_trade_cal(dim_data.get("trade_cal", pd.DataFrame())),
        )
    if "stock_basic" in selected_dim_tables:
        stock_basic = dim_data.get("stock_basic", pd.DataFrame())
        lake.write_vendor_snapshot(table="stock_basic", snapshot_date=snapshot_date, df=stock_basic)
        lake.write_curated_snapshot(
            table="dims/security_master",
            snapshot_date=snapshot_date,
            df=curate_stock_basic(stock_basic, snapshot_date=snapshot_date),
        )
    curated_index_classify = pd.DataFrame()
    if "index_classify" in selected_dim_tables:
        index_classify = dim_data.get("index_classify", pd.DataFrame())
        lake.write_vendor_snapshot(table="index_classify", snapshot_date=snapshot_date, df=index_classify)
        curated_index_classify = curate_index_classify(index_classify)
        lake.write_curated_snapshot(
            table="dims/sw_classify",
            snapshot_date=snapshot_date,
            df=curated_index_classify,
        )
    if "index_member_all" in selected_dim_tables:
        index_member_all = dim_data.get("index_member_all", pd.DataFrame())
        lake.write_vendor_snapshot(table="index_member_all", snapshot_date=snapshot_date, df=index_member_all)
        if curated_index_classify.empty:
            curated_index_classify = curate_index_classify(dim_data.get("index_classify", pd.DataFrame()))
        lake.write_curated_snapshot(
            table="dims/sw_membership_history",
            snapshot_date=snapshot_date,
            df=curate_index_member_all(index_member_all, index_classify_df=curated_index_classify),
        )
    if "namechange" in selected_dim_tables:
        namechange = dim_data.get("namechange", pd.DataFrame())
        lake.write_vendor_snapshot(table="namechange", snapshot_date=snapshot_date, df=namechange)
        lake.write_curated_snapshot(
            table="dims/security_namechange",
            snapshot_date=snapshot_date,
            df=curate_security_namechange(namechange),
        )
        lake.update_ingestion_state(
            table="namechange",
            last_trade_date=_fmt_trade_date(snapshot_date),
            row_count=len(namechange),
            extra={"mode": "snapshot"},
        )
    if "index_basic" in selected_dim_tables:
        index_basic = dim_data.get("index_basic", pd.DataFrame())
        lake.write_vendor_snapshot(table="index_basic", snapshot_date=snapshot_date, df=index_basic)
        curated_index_basic = curate_index_basic(index_basic, snapshot_date=snapshot_date)
        lake.write_curated_snapshot(
            table="dims/index_basic",
            snapshot_date=snapshot_date,
            df=curated_index_basic,
        )
        index_universe_df = resolve_index_universes(
            load_index_universe_config(index_universe_config),
            curated_index_basic,
            universe_names=index_universe_names,
            missing_policy=index_universe_missing_policy,
            snapshot_date=snapshot_date,
        )
        lake.write_curated_snapshot(
            table="dims/index_universe",
            snapshot_date=snapshot_date,
            df=index_universe_df,
        )
        lake.update_ingestion_state(
            table="index_basic",
            last_trade_date=_fmt_trade_date(snapshot_date),
            row_count=len(index_basic),
            extra={"mode": "snapshot", "markets": list(index_basic_markets or [])},
        )
    if "ths_index" in selected_dim_tables:
        ths_index = dim_data.get("ths_index", pd.DataFrame())
        lake.write_vendor_snapshot(table="ths_index", snapshot_date=snapshot_date, df=ths_index)
        lake.write_curated_snapshot(
            table="dims/ths_index",
            snapshot_date=snapshot_date,
            df=curate_ths_index(ths_index, snapshot_date=snapshot_date),
        )
        lake.update_ingestion_state(
            table="ths_index",
            last_trade_date=_fmt_trade_date(snapshot_date),
            row_count=len(ths_index),
            extra={"mode": "snapshot"},
        )
    if "ths_member" in selected_dim_tables:
        ths_member = dim_data.get("ths_member", pd.DataFrame())
        lake.write_vendor_snapshot(table="ths_member", snapshot_date=snapshot_date, df=ths_member)
        lake.write_curated_snapshot(
            table="dims/ths_member",
            snapshot_date=snapshot_date,
            df=curate_ths_member(ths_member),
        )
        lake.update_ingestion_state(
            table="ths_member",
            last_trade_date=_fmt_trade_date(snapshot_date),
            row_count=len(ths_member),
            extra={"mode": "snapshot"},
        )


def _flush_trade_batch(
    lake: ParquetLake,
    adjust_mode: str,
    batch_dates: list[str],
    selected_fact_tables: list[str],
    fact_frames: dict[str, list[pd.DataFrame]],
) -> dict[str, int | str]:
    raw_table_data: dict[str, pd.DataFrame] = {
        table: _concat_frames(fact_frames.get(table, [])) for table in selected_fact_tables
    }

    for table in selected_fact_tables:
        df = raw_table_data.get(table, pd.DataFrame())
        vendor_spec = _FACT_VENDOR_SPECS[table]
        lake.write_vendor_trade_table(
            table=str(vendor_spec["table"]),
            df=df,
            date_col=str(vendor_spec["date_col"]),
            key_cols=tuple(vendor_spec["key_cols"]),
            mode="upsert",
        )
        _release_process_memory()

    curated_table_data = _build_curated_fact_frames(
        selected_fact_tables=selected_fact_tables,
        raw_table_data=raw_table_data,
        adjust_mode=adjust_mode,
    )
    raw_row_counts = {table: int(len(raw_table_data.get(table, pd.DataFrame()))) for table in selected_fact_tables}
    raw_table_data.clear()
    _release_process_memory()

    for table in selected_fact_tables:
        if table not in curated_table_data:
            continue
        curated_df = curated_table_data[table]
        curated_spec = _FACT_CURATED_SPECS[table]
        lake.write_curated_trade_table(
            table=str(curated_spec["table"]),
            df=curated_df,
            date_col=str(curated_spec["date_col"]),
            key_cols=tuple(curated_spec["key_cols"]),
            mode="upsert",
        )
        curated_table_data[table] = pd.DataFrame()
        _release_process_memory()

    if batch_dates:
        last_trade_date = _fmt_trade_date(batch_dates[-1])
        for table in selected_fact_tables:
            lake.update_ingestion_state(
                table=table,
                last_trade_date=last_trade_date,
                row_count=raw_row_counts.get(table, 0),
            )

    out: dict[str, int | str] = {
        "batch_days": int(len(batch_dates)),
        "last_trade_date": _fmt_trade_date(batch_dates[-1]) if batch_dates else "",
    }
    for table in selected_fact_tables:
        out[f"{table}_rows"] = int(raw_row_counts.get(table, 0))
    return out


def _effective_flush_trade_days(
    requested_flush_trade_days: int,
    selected_trade_fact_tables: list[str],
) -> int:
    requested = max(1, int(requested_flush_trade_days))
    selected = {str(x).strip().lower() for x in selected_trade_fact_tables if str(x).strip()}
    if selected & _WIDE_TRADE_FACT_TABLES:
        return min(requested, _WIDE_TRADE_FACT_FLUSH_DAYS)
    return requested


def _release_process_memory() -> None:
    gc.collect()
    try:
        import pyarrow as pa  # type: ignore

        pa.default_memory_pool().release_unused()
    except Exception:
        pass


def _resolve_default_start_date(
    lake: ParquetLake,
    client: Any,
    end_date: str,
    exchange: str,
    lookback_trade_days: int = 5,
    anchor_tables: list[str] | tuple[str, ...] | None = None,
) -> str:
    state = lake.load_ingestion_state()
    tables = state.get("tables", {}) if isinstance(state, dict) else {}
    if not isinstance(tables, dict):
        tables = {}

    candidate_tables = [str(x).strip() for x in (anchor_tables or []) if str(x).strip()]
    if not candidate_tables:
        candidate_tables = ["daily"]

    anchor_dates: list[pd.Timestamp] = []
    for table in candidate_tables:
        table_state = tables.get(table, {})
        if not isinstance(table_state, dict):
            continue

        raw_last_trade = str(table_state.get("last_trade_date", "") or "").strip()
        parsed_state = pd.to_datetime(raw_last_trade, errors="coerce") if raw_last_trade else pd.NaT
        parsed_lake = pd.to_datetime(
            lake.infer_vendor_table_max_trade_date(table=table, date_col="trade_date"),
            errors="coerce",
        )
        anchor = _max_valid_trade_date(parsed_state, parsed_lake)
        if pd.notna(anchor):
            anchor_dates.append(pd.Timestamp(anchor))

    if not anchor_dates:
        daily_state = tables.get("daily", {})
        raw_daily_state = ""
        if isinstance(daily_state, dict):
            raw_daily_state = str(daily_state.get("last_trade_date", "") or "").strip()
        parsed_daily_state = pd.to_datetime(raw_daily_state, errors="coerce") if raw_daily_state else pd.NaT
        parsed_daily_lake = pd.to_datetime(
            lake.infer_vendor_table_max_trade_date(table="daily", date_col="trade_date"),
            errors="coerce",
        )
        daily_anchor = _max_valid_trade_date(parsed_daily_state, parsed_daily_lake)
        if pd.notna(daily_anchor):
            anchor_dates.append(pd.Timestamp(daily_anchor))

    if not anchor_dates:
        return "2020-01-01"

    # Use the earliest anchor to avoid skipping lagging selected tables.
    dt = min(anchor_dates)
    end_dt = pd.to_datetime(str(end_date), errors="coerce")
    probe_end_dt = dt if pd.isna(end_dt) else min(dt, end_dt)
    probe_end = probe_end_dt.strftime("%Y-%m-%d")

    lookback = max(0, int(lookback_trade_days))
    if lookback <= 0:
        return (dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    probe_start = (dt - pd.Timedelta(days=max(60, lookback * 7))).strftime("%Y-%m-%d")
    try:
        trade_days = client.fetch_open_trade_dates(
            start_date=probe_start,
            end_date=probe_end,
            exchange=str(exchange),
        )
    except Exception:
        trade_days = []

    if trade_days:
        idx = max(0, len(trade_days) - lookback)
        return _fmt_trade_date(str(trade_days[idx]))

    fallback = dt - pd.Timedelta(days=max(7, lookback * 2))
    return fallback.strftime("%Y-%m-%d")


def _max_valid_trade_date(*values: pd.Timestamp) -> pd.Timestamp:
    valid = [pd.Timestamp(v) for v in values if pd.notna(v)]
    if not valid:
        return pd.NaT
    return max(valid)


def _split_selected_fact_tables(
    selected_fact_tables: list[str],
    stk_factor_pro_fetch_mode: str = "trade_date",
) -> tuple[list[str], list[str], list[str]]:
    range_set = set(_RANGE_FACT_TABLES)
    code_range_set = set(_CODE_RANGE_FACT_TABLES)
    stk_factor_mode = str(stk_factor_pro_fetch_mode or "trade_date").strip().lower()
    trade_tables: list[str] = []
    code_range_tables: list[str] = []
    range_tables: list[str] = []
    for table in selected_fact_tables:
        key = str(table)
        if key in range_set:
            range_tables.append(key)
        elif key == "stk_factor_pro" and stk_factor_mode == "ts_code_range":
            code_range_tables.append(key)
        elif key in code_range_set:
            code_range_tables.append(key)
        else:
            trade_tables.append(key)
    return trade_tables, code_range_tables, range_tables


def _build_execution_plan(
    selected_trade_fact_tables: list[str],
    selected_range_fact_tables: list[str],
    selected_dim_tables: list[str],
    refresh_dims: bool,
    start_date: str,
    end_date: str,
    open_trade_dates: list[str],
    pending_trade_dates: list[str],
    flush_trade_days: int,
    range_window_days: int,
    prune_out_of_range: bool,
    skip_duckdb: bool,
    trade_calendar_source: str,
    need_trade_bundle: bool,
    selected_code_range_fact_tables: list[str] | None = None,
    code_range_ts_code_count: int = 0,
    index_daily_ts_code_count: int = 0,
) -> dict[str, Any]:
    selected_code_range_fact_tables = list(selected_code_range_fact_tables or [])
    if "index_daily" in set(selected_code_range_fact_tables) and int(code_range_ts_code_count) <= 0:
        code_range_ts_code_count = int(index_daily_ts_code_count)
    pending_days = int(len(pending_trade_dates))
    trade_fact_calls = {str(table): pending_days for table in selected_trade_fact_tables}
    code_count = max(0, int(code_range_ts_code_count))
    code_range_fact_calls: dict[str, int | str] = {
        str(table): (code_count if code_count > 0 else "unknown") for table in selected_code_range_fact_tables
    }
    range_window_count = len(
        _iter_date_windows(
            start_date=start_date,
            end_date=end_date,
            window_days=max(1, int(range_window_days)),
        )
    )
    range_fact_calls = {str(table): int(range_window_count) for table in selected_range_fact_tables}
    dim_calls_by_table: dict[str, int] = {}
    if refresh_dims:
        for table in selected_dim_tables:
            dim_calls_by_table[str(table)] = _estimate_dim_api_calls(table)

    calendar_calls_base = 1 if need_trade_bundle else 0
    calendar_calls_max = 3 if need_trade_bundle else 0
    known_code_range_calls = int(sum(v for v in code_range_fact_calls.values() if isinstance(v, int)))
    fact_calls_total = int(sum(trade_fact_calls.values()) + known_code_range_calls + sum(range_fact_calls.values()))
    dim_calls_total = int(sum(dim_calls_by_table.values()))
    total_calls_base = int(calendar_calls_base + fact_calls_total + dim_calls_total)
    total_calls_max = int(calendar_calls_max + fact_calls_total + dim_calls_total)
    batch_count_est = (pending_days + max(1, int(flush_trade_days)) - 1) // max(1, int(flush_trade_days))

    return {
        "selected_fact_tables": list(selected_trade_fact_tables)
        + list(selected_code_range_fact_tables)
        + list(selected_range_fact_tables),
        "selected_trade_fact_tables": list(selected_trade_fact_tables),
        "selected_code_range_fact_tables": list(selected_code_range_fact_tables),
        "selected_range_fact_tables": list(selected_range_fact_tables),
        "selected_dim_tables": list(selected_dim_tables) if refresh_dims else [],
        "open_trade_days": int(len(open_trade_dates)),
        "pending_trade_days": pending_days,
        "flush_trade_days": int(max(1, int(flush_trade_days))),
        "range_window_days": int(max(1, int(range_window_days))),
        "range_window_count": int(range_window_count),
        "prune_out_of_range": bool(prune_out_of_range),
        "estimated_batches": int(batch_count_est),
        "trade_fact_calls_by_table": trade_fact_calls,
        "code_range_fact_calls_by_table": code_range_fact_calls,
        "code_range_ts_code_count": int(code_count),
        "range_fact_calls_by_table": range_fact_calls,
        "dim_calls_by_table": dim_calls_by_table,
        "calendar_calls_base": int(calendar_calls_base),
        "calendar_calls_max": int(calendar_calls_max),
        "estimated_total_api_calls_base": int(total_calls_base),
        "estimated_total_api_calls_max": int(total_calls_max),
        "trade_day_source": str(trade_calendar_source),
        "rebuild_duckdb_catalog": not bool(skip_duckdb),
    }


def _print_execution_plan(plan: dict[str, Any]) -> None:
    print("[update][plan] ----------")
    print(f"[update][plan] trade_day_source={plan.get('trade_day_source', '')}")
    print(
        f"[update][plan] fact_tables={plan.get('selected_fact_tables', [])} "
        f"trade_fact_tables={plan.get('selected_trade_fact_tables', [])} "
        f"code_range_fact_tables={plan.get('selected_code_range_fact_tables', [])} "
        f"range_fact_tables={plan.get('selected_range_fact_tables', [])} "
        f"dim_tables={plan.get('selected_dim_tables', [])}"
    )
    print(
        f"[update][plan] open_trade_days={plan.get('open_trade_days', 0)} "
        f"pending_trade_days={plan.get('pending_trade_days', 0)} "
        f"flush_trade_days={plan.get('flush_trade_days', 0)} "
        f"range_window_days={plan.get('range_window_days', 0)} "
        f"range_window_count={plan.get('range_window_count', 0)} "
        f"prune_out_of_range={plan.get('prune_out_of_range', False)} "
        f"estimated_batches={plan.get('estimated_batches', 0)}"
    )
    print(f"[update][plan] trade_fact_api_calls={plan.get('trade_fact_calls_by_table', {})}")
    print(
        f"[update][plan] code_range_fact_api_calls={plan.get('code_range_fact_calls_by_table', {})} "
        f"ts_code_count={plan.get('code_range_ts_code_count', 0)}"
    )
    print(f"[update][plan] range_fact_api_calls={plan.get('range_fact_calls_by_table', {})}")
    print(f"[update][plan] dim_api_calls={plan.get('dim_calls_by_table', {})}")
    print(
        "[update][plan] total_api_calls_estimate="
        f"{plan.get('estimated_total_api_calls_base', 0)}~{plan.get('estimated_total_api_calls_max', 0)} "
        "(calendar fallback included in upper bound)"
    )
    print(f"[update][plan] rebuild_duckdb_catalog={plan.get('rebuild_duckdb_catalog', False)}")
    print("[update][plan] ----------")


def _estimate_dim_api_calls(table: str) -> int:
    table_key = str(table or "").strip().lower()
    if table_key == "stock_basic":
        return 3
    if table_key in {
        "index_classify",
        "index_member_all",
        "namechange",
        "index_basic",
        "ths_index",
        "ths_member",
    }:
        return 1
    # trade_cal uses the already-fetched bundle; no extra API call here.
    if table_key == "trade_cal":
        return 0
    return 1


def _fetch_index_basic_markets(client: Any, markets: list[str]) -> pd.DataFrame:
    market_list = [str(x or "").strip() for x in markets if str(x or "").strip()]
    if not market_list:
        return client.fetch_index_basic()
    frames: list[pd.DataFrame] = []
    for market in dict.fromkeys(market_list):
        frame = client.fetch_index_basic(market=market)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frames.append(frame)
    out = _concat_frames(frames)
    if out.empty:
        return out
    dedupe_cols = [col for col in ["ts_code"] if col in out.columns]
    if dedupe_cols:
        out = out.drop_duplicates(subset=dedupe_cols, keep="last")
    return out.reset_index(drop=True)


def _column_values(df: pd.DataFrame, column: str) -> list[str]:
    if df is None or df.empty or str(column) not in df.columns:
        return []
    return [str(x).strip() for x in df[str(column)].dropna().astype(str).tolist() if str(x).strip()]


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid = [f for f in frames if isinstance(f, pd.DataFrame) and not f.empty]
    if not valid:
        return pd.DataFrame()
    return pd.concat(valid, ignore_index=True)


def _fmt_trade_date(raw: str) -> str:
    dt = pd.to_datetime(str(raw), errors="coerce")
    if pd.isna(dt):
        return str(raw)
    return dt.strftime("%Y-%m-%d")


def _infer_last_date_from_frame(df: pd.DataFrame, date_col: str, fallback: str) -> str:
    if df is None or df.empty or str(date_col) not in df.columns:
        return _fmt_trade_date(fallback)
    series = pd.to_datetime(df[str(date_col)], errors="coerce")
    if series.notna().any():
        return pd.Timestamp(series.max()).strftime("%Y-%m-%d")
    return _fmt_trade_date(fallback)


if __name__ == "__main__":
    main()
