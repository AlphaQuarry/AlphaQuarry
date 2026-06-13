from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_mining.datasource.loader import load_panel_from_duckdb
from alpha_mining.live.artifacts import write_latest_index
from alpha_mining.live.calendar import resolve_execute_date
from alpha_mining.live.config import load_live_config
from alpha_mining.live.data_status import check_live_data_status
from alpha_mining.live.account import load_account_inputs
from alpha_mining.live.jobs import count_sa_runs_on_date, write_sa_job
from alpha_mining.live.locks import LiveLockError, live_lock
from alpha_mining.live.orders import build_rebalance_orders, validate_orders_scope
from alpha_mining.live.parity import run_live_signal_parity
from alpha_mining.live.portfolio import build_target_holdings
from alpha_mining.live.registry import load_active_snapshots
from alpha_mining.live.retention import apply_retention
from alpha_mining.live.signal import build_live_superalpha_signal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Live Superalpha target holdings generation.")
    parser.add_argument("--config", default="configs/live.example.yaml")
    parser.add_argument("--universe", default="")
    parser.add_argument("--date", default="auto", help="signal_date: auto or YYYY-MM-DD")
    parser.add_argument("--execute-date", default="auto")
    parser.add_argument("--superalpha-id", default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-reason", default="")
    parser.add_argument("--skip-orders", action="store_true")
    parser.add_argument("--position-path", default="")
    parser.add_argument("--account-snapshot-path", default="")
    parser.add_argument("--account-total-value", type=float, default=None)
    parser.add_argument("--cash", type=float, default=None)
    parser.add_argument("--position-market-value", type=float, default=None)
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--strict-parity", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    cfg = load_live_config(args.config)
    if args.universe:
        cfg.universe = str(args.universe)
    if args.strict_parity:
        cfg.parity.strict = True
    setattr(cfg, "_dry_run", bool(args.dry_run))
    run_id = _make_run_id(cfg.universe)
    snapshots = load_active_snapshots(base_dir=cfg.store_root, universe=cfg.universe, superalpha_id=args.superalpha_id)
    requested_date = None if args.date == "auto" else str(args.date)
    status = check_live_data_status(
        config=cfg,
        requested_date=requested_date,
        snapshots=snapshots,
        force=bool(args.force),
        force_reason=str(args.force_reason or "manual force"),
        write_artifact=not args.dry_run,
    )
    if status["status"] == "no_active_superalpha":
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    if status["status"] != "ready":
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 2
    signal_date = str(status["resolved_signal_date"])
    execute = resolve_execute_date(
        config=cfg,
        signal_date=signal_date,
        requested_execute_date=None if args.execute_date == "auto" else str(args.execute_date),
    )
    if not execute["valid"] and not args.force:
        print(
            json.dumps(
                {"status": "blocked", "execute_date": execute},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    execute_date = str(execute["execute_date"])
    sa_statuses: list[dict[str, Any]] = []
    active_ids = [str(row.get("superalpha_id") or "") for row in snapshots]
    orders_scope = validate_orders_scope(
        config=cfg,
        active_superalpha_ids=active_ids,
        requested_superalpha_id=args.superalpha_id,
    )
    for snapshot in snapshots:
        sid = str(snapshot.get("superalpha_id") or "")
        try:
            if not args.dry_run and count_sa_runs_on_date(
                config=cfg,
                superalpha_id=sid,
                run_date=datetime.now(timezone.utc).date().isoformat(),
            ) >= int(cfg.retention.max_runs_per_day_per_sa):
                raise RuntimeError("max_runs_per_day_per_sa_exceeded")
            with live_lock(config=cfg, name=sid, run_id=run_id) as lock_info:
                parity_result = {"status": "skipped", "reason": "parity_disabled"}
                if bool(cfg.parity.enabled) and not args.skip_parity:
                    try:
                        parity_result = run_live_signal_parity(config=cfg, snapshot=snapshot, signal_date=signal_date)
                    except Exception as exc:
                        parity_result = {
                            "status": "blocked" if bool(cfg.parity.strict) else "warning",
                            "reasons": [str(exc)],
                        }
                    if parity_result.get("status") == "blocked":
                        raise RuntimeError(
                            "parity_blocked:"
                            + ",".join(parity_result.get("reasons") or parity_result.get("blocking_reasons") or [])
                        )
                signal_result = build_live_superalpha_signal(
                    config=cfg,
                    snapshot=snapshot,
                    signal_date=signal_date,
                    dry_run=args.dry_run,
                )
                if signal_result.get("status") != "ok":
                    raise RuntimeError(";".join(signal_result.get("blocking_reasons", ["signal_failed"])))
                signal_df = (
                    signal_result.get("signal") if args.dry_run else pd.read_parquet(signal_result["signal_path"])
                )
                market = _load_market_frame(cfg, signal_date)
                holdings = build_target_holdings(
                    config=cfg,
                    superalpha_id=sid,
                    signal=signal_df,
                    market=market,
                    signal_date=signal_date,
                    execute_date=execute_date,
                    dry_run=args.dry_run,
                )
                if holdings.get("status") != "ok":
                    raise RuntimeError(";".join(holdings.get("blocking_reasons", ["holdings_failed"])))
                holdings_df = (
                    holdings.get("holdings")
                    if args.dry_run
                    else pd.read_parquet(holdings.get("holdings_path") or holdings.get("artifact_path"))
                )
                orders_result = _build_orders_if_requested(
                    cfg=cfg,
                    sid=sid,
                    args=args,
                    orders_scope=orders_scope,
                    holdings_df=holdings_df,
                    signal_date=signal_date,
                    execute_date=execute_date,
                )
                if bool(cfg.orders.required) and orders_result.get("status") != "ok":
                    raise RuntimeError(
                        "orders_required_failed:"
                        + str(orders_result.get("reason") or ";".join(orders_result.get("blocking_reasons", [])))
                    )
                if not args.dry_run:
                    write_sa_job(
                        config=cfg,
                        superalpha_id=sid,
                        job={
                            "job_id": f"{run_id}_{sid}",
                            "status": "done",
                            "signal_date": signal_date,
                            "execute_date": execute_date,
                            "lock": lock_info,
                            "parity": _compact_result(parity_result),
                            "orders": _compact_result(orders_result),
                        },
                    )
                sa_statuses.append(
                    {
                        "superalpha_id": sid,
                        "status": "ok",
                        "stale": False,
                        "signal_date": signal_date,
                        "execute_date": execute_date,
                        "orders": _compact_result(orders_result),
                        "parity": _compact_result(parity_result),
                    }
                )
        except Exception as exc:
            if not args.dry_run:
                write_sa_job(
                    config=cfg,
                    superalpha_id=sid,
                    job={
                        "job_id": f"{run_id}_{sid}",
                        "status": "failed",
                        "error": str(exc),
                    },
                    update_success_latest=False,
                )
            sa_statuses.append(
                {
                    "superalpha_id": sid,
                    "status": "failed_today",
                    "stale": True,
                    "error": str(exc),
                }
            )
    if not args.dry_run:
        try:
            with live_lock(config=cfg, name="global", run_id=run_id):
                write_latest_index(config=cfg, sa_statuses=sa_statuses)
                apply_retention(config=cfg)
        except LiveLockError as exc:
            print(
                json.dumps(
                    {
                        "status": "blocked",
                        "error": str(exc),
                        "superalphas": sa_statuses,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 4
    print(
        json.dumps(
            {
                "status": "ok",
                "data_status": status,
                "execute_date": execute,
                "superalphas": sa_statuses,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if all(row["status"] == "ok" for row in sa_statuses) else 3


def _load_market_frame(cfg: Any, signal_date: str) -> pd.DataFrame:
    fields = list(
        dict.fromkeys(
            [
                "close",
                *cfg.tradability.critical_fields,
                *cfg.tradability.market_value_any_of,
            ]
        )
    )
    available_fields = [f for f in fields if f]
    return load_panel_from_duckdb(
        duckdb_path=str(cfg.duckdb_path),
        source_view=str(cfg.source_view),
        required_fields=available_fields,
        start_date=signal_date,
        end_date=signal_date,
        date_col=cfg.data.date_col,
        code_col=cfg.data.code_col,
        base_fields=(),
        run_filters={"include_bj": bool(cfg.tradability.include_bj)},
    )


def _build_orders_if_requested(
    *,
    cfg: Any,
    sid: str,
    args: argparse.Namespace,
    orders_scope: dict[str, Any],
    holdings_df: pd.DataFrame,
    signal_date: str,
    execute_date: str,
) -> dict[str, Any]:
    if args.skip_orders or not bool(cfg.orders.enabled):
        return {"status": "skipped", "reason": "orders_disabled_or_skipped"}
    if orders_scope.get("status") != "ok":
        return orders_scope
    overrides = {
        key: value
        for key, value in {
            "account_total_value": args.account_total_value,
            "cash": args.cash,
            "position_market_value": args.position_market_value,
        }.items()
        if value is not None
    }
    account_result = load_account_inputs(
        config=cfg,
        execute_date=execute_date,
        position_path=args.position_path or None,
        account_snapshot_path=args.account_snapshot_path or None,
        account_overrides=overrides,
    )
    if account_result.get("status") != "ok":
        if account_result.get("status") == "skipped" and not bool(cfg.orders.required):
            return {
                "status": "skipped",
                "reason": account_result.get("reason") or "orders_skipped",
            }
        if not bool(cfg.orders.required):
            return {
                "status": "skipped",
                "reason": ";".join(account_result.get("blocking_reasons", ["orders_inputs_invalid"])),
            }
        return account_result
    tradability_result = _load_position_tradability_frame(
        cfg=cfg,
        positions=account_result["positions"],
        signal_date=signal_date,
        execute_date=execute_date,
    )
    orders = build_rebalance_orders(
        config=cfg,
        superalpha_id=sid,
        target_holdings=holdings_df,
        positions=account_result["positions"],
        account=account_result["account"],
        execute_date=execute_date,
        dry_run=bool(args.dry_run),
        position_tradability=tradability_result.get("frame"),
    )
    warnings = list(orders.get("warnings", []))
    warnings.extend(tradability_result.get("warnings", []))
    if warnings:
        orders["warnings"] = sorted(set(str(x) for x in warnings if str(x)))
    if isinstance(orders.get("summary"), dict):
        orders["account"] = account_result["account"]
    return orders


def _load_position_tradability_frame(
    *, cfg: Any, positions: pd.DataFrame, signal_date: str, execute_date: str
) -> dict[str, Any]:
    if positions.empty or "code" not in positions.columns:
        return {
            "frame": pd.DataFrame(),
            "warnings": ["position_tradability_positions_empty"],
        }
    codes = sorted({str(x) for x in positions["code"].dropna().astype(str) if str(x)})
    if not codes:
        return {
            "frame": pd.DataFrame(),
            "warnings": ["position_tradability_codes_empty"],
        }
    fields = ["can_sell", "is_suspended", "is_limit_down_close"]
    try:
        frame = load_panel_from_duckdb(
            duckdb_path=str(cfg.duckdb_path),
            source_view=str(cfg.source_view),
            required_fields=fields,
            start_date=str(signal_date),
            end_date=str(execute_date),
            date_col=cfg.data.date_col,
            code_col=cfg.data.code_col,
            base_fields=(),
            run_filters={
                "include_bj": bool(cfg.tradability.include_bj),
                "include_codes": codes,
            },
        )
    except Exception as exc:
        return {
            "frame": pd.DataFrame(),
            "warnings": [f"position_tradability_load_failed:{exc}"],
        }
    if frame.empty:
        return {"frame": frame, "warnings": ["position_tradability_missing"]}
    out = frame.copy()
    out["_date_str"] = pd.to_datetime(out[cfg.data.date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    selected = out[out["_date_str"] == str(execute_date)].copy()
    warnings: list[str] = []
    if selected.empty:
        selected = out[out["_date_str"] == str(signal_date)].copy()
        if not selected.empty:
            warnings.append("position_tradability_fallback_signal_date")
    if selected.empty:
        selected = (
            out[out["_date_str"] <= str(execute_date)]
            .sort_values("_date_str", kind="mergesort")
            .groupby(cfg.data.code_col, as_index=False)
            .tail(1)
        )
        if not selected.empty:
            warnings.append("position_tradability_fallback_latest_available")
    if selected.empty:
        return {
            "frame": pd.DataFrame(),
            "warnings": ["position_tradability_missing_for_dates"],
        }
    selected = selected.rename(columns={cfg.data.date_col: "date", cfg.data.code_col: "code"})
    keep = [c for c in ["date", "code", *fields] if c in selected.columns]
    return {"frame": selected[keep].copy(), "warnings": warnings}


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in dict(result or {}).items() if k not in {"orders", "holdings", "signal", "positions"}}
    return out


def _make_run_id(universe: str) -> str:
    return f"live_{universe}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


if __name__ == "__main__":
    raise SystemExit(main())
