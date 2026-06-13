from __future__ import annotations

from pathlib import Path
from typing import Any

from alpha_mining.atomic_io import atomic_write_json

from .account import load_account_inputs
from .artifacts import live_paths, read_json, utc_now_iso
from .calendar import resolve_execute_date
from .data_status import check_live_data_status
from .lookback import estimate_snapshot_lookback
from .registry import load_active_snapshots


def check_live_readiness(
    *,
    config: Any,
    requested_date: str | None = None,
    superalpha_id: str = "all",
    position_path: str | Path | None = None,
    account_overrides: dict[str, Any] | None = None,
    strict: bool = False,
    json_out: str | Path | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    blocking: list[str] = []
    paths = live_paths(config.store_root, config.universe)
    snapshots = load_active_snapshots(
        base_dir=config.store_root,
        universe=config.universe,
        superalpha_id=superalpha_id,
    )
    if not paths.universe_root.exists():
        blocking.append("universe_root_missing")
    if not snapshots:
        blocking.append("no_active_superalpha")
    snapshot_checks = [_snapshot_check(snap, config=config) for snap in snapshots]
    for row in snapshot_checks:
        warnings.extend(row.get("warnings", []))
        blocking.extend(row.get("blocking_reasons", []))
    try:
        data_status = check_live_data_status(config=config, requested_date=requested_date, snapshots=snapshots)
    except Exception as exc:
        data_status = {"status": "error", "error": str(exc)}
        blocking.append("data_status_error")
    if data_status.get("status") != "ready":
        blocking.extend(str(x) for x in data_status.get("blocking_fields", []) if str(x))
    warnings.extend(str(x) for x in data_status.get("catalog_warnings", []) if str(x))
    signal_date = str(data_status.get("resolved_signal_date") or requested_date or "")
    execute_result: dict[str, Any] = {
        "execute_date": "",
        "valid": False,
        "warnings": [],
    }
    execute_date = ""
    if data_status.get("status") == "ready" and signal_date:
        try:
            execute_result = resolve_execute_date(config=config, signal_date=signal_date)
            execute_date = str(execute_result.get("execute_date") or "")
            if not bool(execute_result.get("valid")):
                blocking.append("execute_date_unresolved")
            warnings.extend(str(x) for x in execute_result.get("warnings", []) if str(x))
        except Exception as exc:
            execute_result = {
                "execute_date": "",
                "valid": False,
                "warnings": [str(exc)],
            }
            blocking.append("execute_date_error")
    account_result = {"status": "skipped", "reason": "orders_disabled"}
    if bool(config.orders.enabled):
        account_result = load_account_inputs(
            config=config,
            execute_date=execute_date or signal_date or requested_date or "",
            position_path=position_path,
            account_overrides=account_overrides or {},
        )
        if account_result.get("status") == "skipped":
            warnings.append(str(account_result.get("reason") or "orders_skipped"))
        elif account_result.get("status") != "ok":
            reasons = [str(x) for x in account_result.get("blocking_reasons", [])]
            if bool(config.orders.required) or strict:
                blocking.extend(reasons)
            else:
                warnings.extend(reasons)
        elif not bool(account_result.get("money_reliable")):
            warnings.append("account_money_basis_missing")
    latest_state = _latest_state(paths=paths, snapshots=snapshots)
    if any(row.get("stale") for row in latest_state):
        warnings.append("latest_stale")
    if strict and warnings:
        blocking.extend(warnings)
    status = "BLOCKED" if blocking else "WARN" if warnings else "READY"
    payload = {
        "schema_version": 1,
        "status": status,
        "created_at_utc": utc_now_iso(),
        "universe": str(config.universe),
        "requested_date": str(requested_date or ""),
        "signal_date": signal_date,
        "execute_date": execute_date,
        "warnings": sorted(set(warnings)),
        "blocking_reasons": sorted(set(blocking)),
        "checks": {
            "config": {
                "status": "ok",
                "store_root": str(Path(config.store_root).as_posix()),
                "source_view": str(config.source_view),
            },
            "active_superalphas": {
                "status": "ok" if snapshots else "blocked",
                "count": len(snapshots),
                "snapshots": snapshot_checks,
            },
            "data_status": data_status,
            "execute_date": execute_result,
            "account": _compact_account(account_result),
            "latest": latest_state,
        },
    }
    if json_out:
        atomic_write_json(json_out, payload, backup=True)
    return payload


def _snapshot_check(snapshot: dict[str, Any], *, config: Any) -> dict[str, Any]:
    warnings: list[str] = []
    blocking: list[str] = []
    sid = str(snapshot.get("superalpha_id") or "")
    if not sid:
        blocking.append("snapshot_missing_superalpha_id")
    if not snapshot.get("components") and not snapshot.get("component_expressions"):
        blocking.append("snapshot_missing_components")
    lookback = estimate_snapshot_lookback(snapshot, buffer=int(config.superalpha.lookback_buffer_days))
    warnings.extend([f"lookback_parse_warning:{x}" for x in lookback.get("warnings", [])])
    return {
        "superalpha_id": sid,
        "status": "ok" if not blocking else "blocked",
        "lookback": lookback,
        "warnings": warnings,
        "blocking_reasons": blocking,
    }


def _latest_state(*, paths: Any, snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for snap in snapshots:
        sid = str(snap.get("superalpha_id") or "")
        holdings = read_json(paths.holdings_dir(sid) / "latest.json", None)
        orders = read_json(paths.live_root / "orders" / sid / "latest.json", None)
        rows.append(
            {
                "superalpha_id": sid,
                "holdings_status": holdings.get("status") if isinstance(holdings, dict) else "missing",
                "orders_status": orders.get("status") if isinstance(orders, dict) else "missing",
                "stale": False,
                "dry_run": bool((orders or {}).get("dry_run")) if isinstance(orders, dict) else False,
            }
        )
    return rows


def _compact_account(result: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in result.items() if k not in {"positions"}}
    return out
