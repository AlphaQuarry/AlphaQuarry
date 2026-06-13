from __future__ import annotations

import argparse
import gc
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_mining.datasource import (
    ParquetLake,
    TIER_PRESETS,
    TushareApiError,
    TushareErrorCategory,
    build_checkpoint_signature,
    build_duckdb_catalog,
    build_tushare_client_from_settings,
    curate_finance_balancesheet_vip,
    curate_finance_cashflow_vip,
    curate_finance_fina_indicator_vip,
    curate_finance_income_vip,
    curate_index_classify,
    curate_index_member_all,
    curate_market_adj_factor,
    curate_market_daily,
    curate_market_daily_basic,
    curate_market_stk_limit,
    curate_market_suspend_d,
    curate_moneyflow_ths,
    curate_security_namechange,
    curate_stock_basic,
    curate_ths_index,
    curate_ths_member,
    curate_trade_cal,
    get_checkpoint_path,
    load_checkpoint,
    load_datasource_settings,
    reset_checkpoint,
    save_checkpoint,
)
from alpha_mining.datasource.duckdb_runtime import build_duckdb_runtime_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap Tushare -> Parquet lake -> DuckDB catalog")
    parser.add_argument("--config", default="", help="Datasource config yaml path")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default="", help="YYYY-MM-DD, default=today")
    parser.add_argument("--exchange", default="SSE")
    parser.add_argument("--adjust-mode", default="", choices=["", "qfq", "hfq"])
    parser.add_argument("--source-view", default="", help="Override source view name")
    parser.add_argument("--field-catalog-version", default="")
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
    parser.add_argument("--include-p2", action="store_true", help="Include P2 financial VIP tables")
    parser.add_argument("--include-p3", action="store_true", help="Include P3 moneyflow/theme tables")
    parser.add_argument(
        "--tier",
        default="",
        choices=["", "basic", "standard", "extended", "full"],
        help="Data tier preset. Overrides --include-p2/--include-p3.",
    )
    parser.add_argument(
        "--flush-trade-days",
        type=int,
        default=20,
        help="Flush batch size by trade days",
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
    args = parser.parse_args()

    # --tier 解析: 覆盖 --include-p2/--include-p3
    tier = str(args.tier or "").strip().lower()
    if tier and tier in TIER_PRESETS:
        _tier_fact, _tier_dim, _tier_points = TIER_PRESETS[tier]
        # 根据 tier 决定 include_p2 / include_p3
        args.include_p2 = "p2" in _tier_fact
        args.include_p3 = any(g in _tier_fact for g in ("p3", "p3_legacy"))
        print(f"[bootstrap] tier={tier}, include_p2={args.include_p2}, include_p3={args.include_p3}")

    end_date = str(args.end_date).strip() or datetime.now().strftime("%Y-%m-%d")

    settings = load_datasource_settings(str(args.config or "") or None)
    lake = ParquetLake(paths=settings.paths)
    client = build_tushare_client_from_settings(settings.tushare)

    adjust_mode = str(args.adjust_mode).strip().lower() or settings.adjust_mode
    source_view = str(args.source_view).strip() or settings.source_view
    field_catalog_version = str(args.field_catalog_version).strip() or settings.field_catalog_version
    flush_trade_days = max(1, int(args.flush_trade_days))

    checkpoint_task = "bootstrap_tushare_lake_daily"
    checkpoint_signature = build_checkpoint_signature(
        task_name=checkpoint_task,
        payload={
            "start_date": str(args.start_date),
            "end_date": str(end_date),
            "exchange": str(args.exchange),
            "adjust_mode": str(adjust_mode),
            "lake_root": str(settings.paths.lake_root_path.as_posix()),
            "http_url": str(settings.tushare.http_url or ""),
            "source_view": str(source_view),
            "include_bj": bool(settings.include_bj),
            "universe_min_days_since_listed": int(settings.universe_min_days_since_listed),
            "universe_exclude_st": bool(settings.universe_exclude_st),
            "include_p2": bool(args.include_p2),
            "include_p3": bool(args.include_p3),
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
        print(f"[bootstrap] checkpoint reset: {removed} -> {checkpoint_path.as_posix()}")

    print("[bootstrap] loading trade calendar...")
    trade_cal, open_trade_dates, trade_calendar_source = client.fetch_trade_calendar_bundle(
        start_date=args.start_date,
        end_date=end_date,
        exchange=str(args.exchange),
    )
    print(f"[bootstrap] open trade days: {len(open_trade_dates)} (source={trade_calendar_source})")

    completed_set: set[str] = set()
    checkpoint_payload: dict[str, object] = {
        "status": "running",
        "total_trade_dates": int(len(open_trade_dates)),
        "completed_trade_dates": [],
        "start_date": str(args.start_date),
        "end_date": str(end_date),
        "exchange": str(args.exchange),
        "adjust_mode": str(adjust_mode),
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
            f"[bootstrap] resume={args.resume} checkpoint={checkpoint_path.as_posix()} "
            f"completed={len(completed_set)}/{len(open_trade_dates)}"
        )
    else:
        print("[bootstrap] resume disabled; running from scratch")

    pending_trade_dates = [d for d in open_trade_dates if str(d) not in completed_set]
    print(f"[bootstrap] pending trade days: {len(pending_trade_dates)}")

    print("[bootstrap] loading stock_basic/index_classify/index_member_all/namechange...")
    stock_basic = client.fetch_stock_basic()
    index_classify = client.fetch_index_classify(src="SW2021")
    index_member_all = client.fetch_index_member_all(src="SW2021")
    namechange = client.fetch_namechange(end_date=end_date)
    ths_index = pd.DataFrame()
    ths_member = pd.DataFrame()
    if bool(args.include_p3):
        print("[bootstrap] loading ths_index/ths_member...")
        ths_index = client.fetch_ths_index()
        ths_member = client.fetch_ths_member(ts_codes=_column_values(ths_index, "ts_code"))

    print("[bootstrap] writing vendor snapshots...")
    lake.write_vendor_snapshot(table="trade_cal", snapshot_date=end_date, df=trade_cal)
    lake.write_vendor_snapshot(table="stock_basic", snapshot_date=end_date, df=stock_basic)
    lake.write_vendor_snapshot(table="index_classify", snapshot_date=end_date, df=index_classify)
    lake.write_vendor_snapshot(table="index_member_all", snapshot_date=end_date, df=index_member_all)
    lake.write_vendor_snapshot(table="namechange", snapshot_date=end_date, df=namechange)
    if bool(args.include_p3):
        lake.write_vendor_snapshot(table="ths_index", snapshot_date=end_date, df=ths_index)
        lake.write_vendor_snapshot(table="ths_member", snapshot_date=end_date, df=ths_member)

    print("[bootstrap] writing curated snapshots...")
    curated_trade_cal = curate_trade_cal(trade_cal)
    curated_stock_basic = curate_stock_basic(stock_basic, snapshot_date=end_date)
    curated_index_classify = curate_index_classify(index_classify)
    curated_index_member_all = curate_index_member_all(index_member_all, index_classify_df=curated_index_classify)
    curated_namechange = curate_security_namechange(namechange)
    curated_ths_index = curate_ths_index(ths_index, snapshot_date=end_date)
    curated_ths_member = curate_ths_member(ths_member)

    lake.write_curated_snapshot(table="dims/trade_calendar", snapshot_date=end_date, df=curated_trade_cal)
    lake.write_curated_snapshot(table="dims/security_master", snapshot_date=end_date, df=curated_stock_basic)
    lake.write_curated_snapshot(table="dims/sw_classify", snapshot_date=end_date, df=curated_index_classify)
    lake.write_curated_snapshot(
        table="dims/sw_membership_history",
        snapshot_date=end_date,
        df=curated_index_member_all,
    )
    lake.write_curated_snapshot(table="dims/security_namechange", snapshot_date=end_date, df=curated_namechange)
    if bool(args.include_p3):
        lake.write_curated_snapshot(table="dims/ths_index", snapshot_date=end_date, df=curated_ths_index)
        lake.write_curated_snapshot(table="dims/ths_member", snapshot_date=end_date, df=curated_ths_member)
        lake.update_ingestion_state(
            table="ths_index",
            last_trade_date=_fmt_trade_date(end_date),
            row_count=len(ths_index),
            extra={"mode": "snapshot"},
        )
        lake.update_ingestion_state(
            table="ths_member",
            last_trade_date=_fmt_trade_date(end_date),
            row_count=len(ths_member),
            extra={"mode": "snapshot"},
        )
    lake.update_ingestion_state(
        table="namechange",
        last_trade_date=_fmt_trade_date(end_date),
        row_count=len(namechange),
        extra={"mode": "snapshot"},
    )

    print("[bootstrap] loading daily/daily_basic/adj_factor by trade date...")
    batch_dates: list[str] = []
    daily_frames: list[pd.DataFrame] = []
    daily_basic_frames: list[pd.DataFrame] = []
    adj_factor_frames: list[pd.DataFrame] = []
    stk_limit_frames: list[pd.DataFrame] = []
    suspend_d_frames: list[pd.DataFrame] = []
    moneyflow_ths_frames: list[pd.DataFrame] = []
    _perm_denied: set[str] = set()

    for idx, trade_date in enumerate(pending_trade_dates, start=1):
        try:
            if "daily" not in _perm_denied:
                daily_frames.append(client.fetch_daily_by_trade_date(trade_date))
            if "daily_basic" not in _perm_denied:
                daily_basic_frames.append(client.fetch_daily_basic_by_trade_date(trade_date))
            if "adj_factor" not in _perm_denied:
                adj_factor_frames.append(client.fetch_adj_factor_by_trade_date(trade_date))
            if "stk_limit" not in _perm_denied:
                stk_limit_frames.append(client.fetch_stk_limit_by_trade_date(trade_date))
            if "suspend_d" not in _perm_denied:
                suspend_d_frames.append(client.fetch_suspend_d_by_trade_date(trade_date))
            if bool(args.include_p3) and "moneyflow_ths" not in _perm_denied:
                moneyflow_ths_frames.append(client.fetch_moneyflow_ths_by_trade_date(trade_date))
        except TushareApiError as exc:
            if exc.category == TushareErrorCategory.AUTH:
                _perm_denied.add(exc.api_name)
                print(f"[bootstrap] ❌ 跳过无权限接口: {exc.api_name}", file=sys.stderr)
            elif exc.category == TushareErrorCategory.RATE_LIMIT:
                print(
                    f"[bootstrap] ⏳ 频率限制，跳过本批次: {exc.api_name}",
                    file=sys.stderr,
                )
                # 不加入 _perm_denied，下个批次会重试
            else:
                raise
        batch_dates.append(str(trade_date))

        should_flush = (len(batch_dates) >= flush_trade_days) or (idx == len(pending_trade_dates))
        if not should_flush:
            continue

        flushed = _flush_trade_batch(
            lake=lake,
            adjust_mode=adjust_mode,
            batch_dates=batch_dates,
            daily_frames=daily_frames,
            daily_basic_frames=daily_basic_frames,
            adj_factor_frames=adj_factor_frames,
            stk_limit_frames=stk_limit_frames,
            suspend_d_frames=suspend_d_frames,
            moneyflow_ths_frames=moneyflow_ths_frames,
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
            f"  [bootstrap] flushed {len(batch_dates)} trade days, "
            f"completed={len(completed_set)}/{len(open_trade_dates)}"
        )

        batch_dates = []
        daily_frames = []
        daily_basic_frames = []
        adj_factor_frames = []
        stk_limit_frames = []
        suspend_d_frames = []
        moneyflow_ths_frames = []
        _release_process_memory()

    p2_summary: dict[str, dict[str, int | str]] = {}
    if bool(args.include_p2):
        print("[bootstrap] loading finance p2 facts by ann_date range...")
        try:
            p2_summary = _refresh_finance_p2(
                lake=lake,
                client=client,
                start_date=str(args.start_date),
                end_date=end_date,
            )
            print(f"[bootstrap] finance p2 summary: {p2_summary}")
        except TushareApiError as exc:
            if exc.category == TushareErrorCategory.AUTH:
                _perm_denied.add(exc.api_name)
                print(
                    f"[bootstrap] ⚠️ 跳过无权限的 P2 财务数据: {exc.api_name}",
                    file=sys.stderr,
                )
            else:
                raise
    else:
        print("[bootstrap] include-p2 disabled, skip finance p2 facts")

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
        print("[bootstrap] building duckdb catalog...")
        duckdb_runtime = build_duckdb_runtime_settings(
            duckdb_path=settings.paths.duckdb_path,
            temp_directory=str(args.duckdb_temp_directory or "").strip(),
            isolate_run=bool(args.duckdb_temp_isolate_run),
            run_id=str(args.duckdb_temp_run_id or "").strip(),
            run_prefix="run_bootstrap_duckdb_catalog",
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
        print("[bootstrap] duckdb catalog:", catalog_out)

    if _perm_denied:
        print(
            f"\n[bootstrap] ⚠️ 以下接口因权限不足被跳过: {sorted(_perm_denied)}\n"
            f"  如需这些数据，请升级 Tushare 积分后重新运行\n"
            f"  参考: docs/tushare_points_reference.md",
            file=sys.stderr,
        )

    print("[bootstrap] done")
    print(
        {
            "open_trade_days": int(len(open_trade_dates)),
            "completed_trade_days": int(len(completed_set)),
            "pending_trade_days": int(len(pending_trade_dates)),
            "flush_trade_days": int(flush_trade_days),
            "adjust_mode": adjust_mode,
            "checkpoint_path": str(checkpoint_path.as_posix()),
            "p2_summary": p2_summary,
            "include_p3": bool(args.include_p3),
        }
    )


def _flush_trade_batch(
    lake: ParquetLake,
    adjust_mode: str,
    batch_dates: list[str],
    daily_frames: list[pd.DataFrame],
    daily_basic_frames: list[pd.DataFrame],
    adj_factor_frames: list[pd.DataFrame],
    stk_limit_frames: list[pd.DataFrame],
    suspend_d_frames: list[pd.DataFrame],
    moneyflow_ths_frames: list[pd.DataFrame] | None = None,
) -> dict[str, int | str]:
    daily_df = _concat_frames(daily_frames)
    daily_basic_df = _concat_frames(daily_basic_frames)
    adj_factor_df = _concat_frames(adj_factor_frames)
    stk_limit_df = _concat_frames(stk_limit_frames)
    suspend_d_df = _concat_frames(suspend_d_frames)
    moneyflow_ths_df = _concat_frames(list(moneyflow_ths_frames or []))
    row_counts = {
        "daily_rows": int(len(daily_df)),
        "daily_basic_rows": int(len(daily_basic_df)),
        "adj_factor_rows": int(len(adj_factor_df)),
        "stk_limit_rows": int(len(stk_limit_df)),
        "suspend_d_rows": int(len(suspend_d_df)),
        "moneyflow_ths_rows": int(len(moneyflow_ths_df)),
    }

    lake.write_vendor_trade_table(
        table="daily",
        df=daily_df,
        date_col="trade_date",
        key_cols=("ts_code", "trade_date"),
        mode="upsert",
    )
    lake.write_vendor_trade_table(
        table="daily_basic",
        df=daily_basic_df,
        date_col="trade_date",
        key_cols=("ts_code", "trade_date"),
        mode="upsert",
    )
    lake.write_vendor_trade_table(
        table="adj_factor",
        df=adj_factor_df,
        date_col="trade_date",
        key_cols=("ts_code", "trade_date"),
        mode="upsert",
    )
    lake.write_vendor_trade_table(
        table="stk_limit",
        df=stk_limit_df,
        date_col="trade_date",
        key_cols=("ts_code", "trade_date"),
        mode="upsert",
    )
    lake.write_vendor_trade_table(
        table="suspend_d",
        df=suspend_d_df,
        date_col="trade_date",
        key_cols=("ts_code", "trade_date"),
        mode="upsert",
    )
    if moneyflow_ths_frames is not None:
        lake.write_vendor_trade_table(
            table="moneyflow_ths",
            df=moneyflow_ths_df,
            date_col="trade_date",
            key_cols=("ts_code", "trade_date"),
            mode="upsert",
        )
    _release_process_memory()

    curated_daily = curate_market_daily(daily_df, adj_factor_df, adjust_mode=adjust_mode)
    curated_daily_basic = curate_market_daily_basic(daily_basic_df)
    curated_adj_factor = curate_market_adj_factor(adj_factor_df)
    curated_stk_limit = curate_market_stk_limit(stk_limit_df)
    curated_suspend_d = curate_market_suspend_d(suspend_d_df)
    curated_moneyflow_ths = curate_moneyflow_ths(moneyflow_ths_df)

    lake.write_curated_trade_table(
        table="facts/market_daily",
        df=curated_daily,
        date_col="date",
        key_cols=("code", "date"),
        mode="upsert",
    )
    lake.write_curated_trade_table(
        table="facts/market_daily_basic",
        df=curated_daily_basic,
        date_col="date",
        key_cols=("code", "date"),
        mode="upsert",
    )
    lake.write_curated_trade_table(
        table="facts/market_adj_factor",
        df=curated_adj_factor,
        date_col="date",
        key_cols=("code", "date"),
        mode="upsert",
    )
    lake.write_curated_trade_table(
        table="facts/market_stk_limit",
        df=curated_stk_limit,
        date_col="date",
        key_cols=("code", "date"),
        mode="upsert",
    )
    lake.write_curated_trade_table(
        table="facts/market_suspend_d",
        df=curated_suspend_d,
        date_col="date",
        key_cols=("code", "date"),
        mode="upsert",
    )
    if moneyflow_ths_frames is not None:
        lake.write_curated_trade_table(
            table="facts/moneyflow_ths",
            df=curated_moneyflow_ths,
            date_col="date",
            key_cols=("code", "date"),
            mode="upsert",
        )

    if batch_dates:
        last_trade_date = _fmt_trade_date(batch_dates[-1])
        lake.update_ingestion_state(table="daily", last_trade_date=last_trade_date, row_count=len(daily_df))
        lake.update_ingestion_state(
            table="daily_basic",
            last_trade_date=last_trade_date,
            row_count=len(daily_basic_df),
        )
        lake.update_ingestion_state(
            table="adj_factor",
            last_trade_date=last_trade_date,
            row_count=len(adj_factor_df),
        )
        lake.update_ingestion_state(
            table="stk_limit",
            last_trade_date=last_trade_date,
            row_count=len(stk_limit_df),
        )
        lake.update_ingestion_state(
            table="suspend_d",
            last_trade_date=last_trade_date,
            row_count=len(suspend_d_df),
        )
        if moneyflow_ths_frames is not None:
            lake.update_ingestion_state(
                table="moneyflow_ths",
                last_trade_date=last_trade_date,
                row_count=len(moneyflow_ths_df),
            )

    daily_df = pd.DataFrame()
    daily_basic_df = pd.DataFrame()
    adj_factor_df = pd.DataFrame()
    stk_limit_df = pd.DataFrame()
    suspend_d_df = pd.DataFrame()
    moneyflow_ths_df = pd.DataFrame()
    curated_daily = pd.DataFrame()
    curated_daily_basic = pd.DataFrame()
    curated_adj_factor = pd.DataFrame()
    curated_stk_limit = pd.DataFrame()
    curated_suspend_d = pd.DataFrame()
    curated_moneyflow_ths = pd.DataFrame()
    _release_process_memory()

    return {
        **row_counts,
        "batch_days": int(len(batch_dates)),
        "last_trade_date": _fmt_trade_date(batch_dates[-1]) if batch_dates else "",
    }


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid = [f for f in frames if isinstance(f, pd.DataFrame) and not f.empty]
    if not valid:
        return pd.DataFrame()
    out = pd.concat(valid, ignore_index=True)
    return out


def _release_process_memory() -> None:
    gc.collect()
    try:
        import pyarrow as pa  # type: ignore

        pa.default_memory_pool().release_unused()
    except Exception:
        pass


def _fmt_trade_date(raw: str) -> str:
    dt = pd.to_datetime(str(raw), errors="coerce")
    if pd.isna(dt):
        return str(raw)
    return dt.strftime("%Y-%m-%d")


def _refresh_finance_p2(
    lake: ParquetLake,
    client: Any,
    start_date: str,
    end_date: str,
) -> dict[str, dict[str, int | str]]:
    raw_income = client.fetch_income_vip(start_date=start_date, end_date=end_date)
    raw_balance = client.fetch_balancesheet_vip(start_date=start_date, end_date=end_date)
    raw_cashflow = client.fetch_cashflow_vip(start_date=start_date, end_date=end_date)
    raw_indicator = client.fetch_fina_indicator_vip(start_date=start_date, end_date=end_date)

    curated_income = curate_finance_income_vip(raw_income)
    curated_balance = curate_finance_balancesheet_vip(raw_balance)
    curated_cashflow = curate_finance_cashflow_vip(raw_cashflow)
    curated_indicator = curate_finance_fina_indicator_vip(raw_indicator)

    specs = [
        ("income_vip", raw_income, curated_income, "facts/finance_income_q"),
        (
            "balancesheet_vip",
            raw_balance,
            curated_balance,
            "facts/finance_balancesheet_q",
        ),
        ("cashflow_vip", raw_cashflow, curated_cashflow, "facts/finance_cashflow_q"),
        (
            "fina_indicator_vip",
            raw_indicator,
            curated_indicator,
            "facts/finance_indicator_q",
        ),
    ]
    out: dict[str, dict[str, int | str]] = {}
    for table, raw_df, curated_df, curated_table in specs:
        lake.write_vendor_trade_table(
            table=table,
            df=raw_df,
            date_col="ann_date",
            key_cols=("ts_code", "ann_date", "end_date"),
            mode="upsert",
        )
        lake.write_curated_trade_table(
            table=curated_table,
            df=curated_df,
            date_col="ann_date",
            key_cols=("code", "ann_date", "end_date"),
            mode="upsert",
        )
        last_trade_date = _infer_last_date(raw_df, date_col="ann_date", fallback=end_date)
        lake.update_ingestion_state(
            table=table,
            last_trade_date=last_trade_date,
            row_count=len(raw_df),
            extra={
                "mode": "ann_date_range",
                "start_date": str(start_date),
                "end_date": str(end_date),
            },
        )
        out[table] = {
            "raw_rows": int(len(raw_df)),
            "curated_rows": int(len(curated_df)),
            "last_trade_date": str(last_trade_date),
        }
    return out


def _infer_last_date(df: pd.DataFrame, date_col: str, fallback: str) -> str:
    if df is None or df.empty or str(date_col) not in df.columns:
        return _fmt_trade_date(fallback)
    series = pd.to_datetime(df[str(date_col)], errors="coerce")
    if series.notna().any():
        return pd.Timestamp(series.max()).strftime("%Y-%m-%d")
    return _fmt_trade_date(fallback)


def _column_values(df: pd.DataFrame, column: str) -> list[str]:
    if df is None or df.empty or str(column) not in df.columns:
        return []
    return [str(x).strip() for x in df[str(column)].dropna().astype(str).tolist() if str(x).strip()]


if __name__ == "__main__":
    main()
