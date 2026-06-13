from __future__ import annotations

from typing import Any

import pandas as pd

from alpha_mining.datasource.loader import get_duckdb_view_columns
from alpha_mining.workflow.reproduce import load_required_fields_from_expression

from .artifacts import live_paths, utc_now_iso, write_json
from .field_catalog import (
    catalog_field_blocks_live,
    enrich_field_row_from_catalog,
    load_live_field_catalog,
)
from .registry import load_active_snapshots


def collect_active_required_fields(*, config: Any, snapshots: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    snaps = (
        snapshots
        if snapshots is not None
        else load_active_snapshots(base_dir=config.store_root, universe=config.universe)
    )
    fields: dict[str, set[str]] = {}
    for snap in snaps:
        sid = str(snap.get("superalpha_id") or "")
        expressions = snap.get("component_expressions") or [c.get("expression", "") for c in snap.get("components", [])]
        for expr in expressions:
            try:
                for field in load_required_fields_from_expression(str(expr)):
                    fields.setdefault(str(field), set()).add(sid)
            except Exception:
                continue
    if bool(config.tradability.enabled):
        for field in config.tradability.critical_fields:
            fields.setdefault(str(field), set()).add("__tradability__")
        fields.setdefault("__market_value_any_of__", set()).add("__tradability__")
    return {"fields": {k: sorted(v) for k, v in fields.items()}, "snapshots": snaps}


def check_live_data_status(
    *,
    config: Any,
    requested_date: str | None = None,
    snapshots: list[dict[str, Any]] | None = None,
    force: bool = False,
    force_reason: str = "",
    write_artifact: bool = False,
) -> dict[str, Any]:
    active = (
        snapshots
        if snapshots is not None
        else load_active_snapshots(base_dir=config.store_root, universe=config.universe)
    )
    if not active:
        payload = _base_payload(config=config, requested_date=requested_date, status="no_active_superalpha")
        if write_artifact:
            _write_status(config, payload)
        return payload

    required = collect_active_required_fields(config=config, snapshots=active)["fields"]
    columns = set(get_duckdb_view_columns(str(config.duckdb_path), str(config.source_view)))
    catalog = load_live_field_catalog(config=config)
    missing: list[str] = []
    catalog_warnings: list[str] = []
    if catalog.get("status") != "ok":
        catalog_warnings.append("field_catalog_missing")
        if str(getattr(config.data, "catalog_missing_policy", "warn")).lower() == "block":
            missing.append("field_catalog_missing")
    market_value_fields = [f for f in config.tradability.market_value_any_of if f in columns]
    field_names = [f for f in required if f != "__market_value_any_of__"]
    for field in field_names:
        if field not in columns:
            missing.append(field)
    if "__market_value_any_of__" in required and not market_value_fields:
        missing.append("one_of:" + ",".join(config.tradability.market_value_any_of))

    # Pure tradability filter fields — these are operational filters available at
    # execution time, not alpha signals.  The catalog leakage-safe check does not
    # apply to them.  Fields like "close" that serve dual roles (signal + filter)
    # are NOT included here; they still undergo the full catalog check.
    _TRADABILITY_FILTER_ONLY: set[str] = {
        "can_buy",
        "can_sell",
        "is_st",
        "is_suspended",
        "up_limit",
        "down_limit",
        "is_limit_up_close",
        "is_limit_down_close",
    }
    tradability_filter_set: set[str] = (
        _TRADABILITY_FILTER_ONLY & set(config.tradability.critical_fields)
        if bool(config.tradability.enabled)
        else set()
    )

    rows: list[dict[str, Any]] = []
    latest_dates: list[str] = []
    for field in field_names:
        if field not in columns:
            row = _field_row(
                field,
                required.get(field, []),
                available=False,
                blocking_reason="missing_field",
            )
            rows.append(enrich_field_row_from_catalog(row, catalog))
            continue
        latest = _latest_non_null_date(config, field)
        if latest:
            latest_dates.append(latest)
        rows.append(
            enrich_field_row_from_catalog(
                _field_row(field, required.get(field, []), available=True, latest=latest),
                catalog,
            )
        )

    common_ready_date = min(latest_dates) if latest_dates else ""
    expected = str(requested_date or common_ready_date or "")
    blocking_fields: list[str] = list(missing)
    for row in rows:
        if not row["available_in_view"]:
            continue
        field_name = str(row["field"])
        rate = _non_null_rate(config, field_name, expected) if expected else 0.0
        row["field_non_null_rate_on_expected_date"] = rate
        if row.get("catalog_status") != "ok":
            catalog_warnings.append(f"catalog_missing:{field_name}")
        if rate < float(config.data.min_field_non_null_rate):
            row["is_ready"] = False
            row["blocking_reason"] = "low_non_null_rate"
            blocking_fields.append(field_name)
        elif field_name not in tradability_filter_set:
            # Tradability fields are operational filters (e.g. can_buy, is_st), not
            # alpha signals.  They are available at execution time, so the catalog
            # leakage-safe check does not apply to them.
            catalog_block = catalog_field_blocks_live(row, strict_available_at=bool(config.data.strict_available_at))
            if catalog_block:
                row["is_ready"] = False
                row["blocking_reason"] = catalog_block
                blocking_fields.append(field_name)
            else:
                row["is_ready"] = True
        else:
            row["is_ready"] = True
    selected_mv = _select_market_value_field(config, market_value_fields, expected)
    if "__market_value_any_of__" in required:
        if not selected_mv.get("selected_market_value_field"):
            reason = "market_value_any_of_low_non_null_rate" if market_value_fields else "missing_market_value_field"
            if str(getattr(config.data, "market_value_missing_policy", "block")).lower() == "block":
                blocking_fields.append(reason)
            else:
                catalog_warnings.append(reason)
    original_blocking = sorted(set(blocking_fields))
    status = "ready" if not original_blocking and expected else "data_not_ready"
    if force and status != "ready":
        status = "ready"
    payload = {
        **_base_payload(config=config, requested_date=requested_date, status=status),
        "active_superalpha_count": len(active),
        "requested_date": str(requested_date or ""),
        "common_ready_date": common_ready_date,
        "resolved_signal_date": expected if status == "ready" else "",
        "ready_field_count": int(sum(1 for r in rows if r.get("is_ready"))),
        "blocking_field_count": int(len(original_blocking)),
        "blocking_fields": original_blocking,
        "fields": rows,
        "forced": bool(force),
        "force_reason": str(force_reason or ""),
        "original_blocking_fields": original_blocking if force else [],
        "degradations": []
        if market_value_fields or "__market_value_any_of__" not in required
        else ["missing_market_value_field"],
        "catalog_status": catalog.get("status", "missing"),
        "catalog_source": catalog.get("source", ""),
        "catalog_warnings": sorted(set(catalog_warnings)),
        "selected_market_value_field": selected_mv.get("selected_market_value_field", ""),
        "selected_market_value_non_null_rate": selected_mv.get("selected_market_value_non_null_rate", 0.0),
        "market_value_candidates": selected_mv.get("market_value_candidates", []),
    }
    if write_artifact:
        _write_status(config, payload)
    return payload


def _base_payload(*, config: Any, requested_date: str | None, status: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "universe": str(config.universe),
        "requested_date": str(requested_date or ""),
        "status": status,
        "created_at_utc": utc_now_iso(),
    }


def _field_row(
    field: str,
    required_by: list[str],
    *,
    available: bool,
    latest: str = "",
    blocking_reason: str = "",
) -> dict[str, Any]:
    return {
        "field": str(field),
        "required_by": required_by,
        "available_in_view": available,
        "field_latest_non_null_date": str(latest or ""),
        "field_non_null_rate_on_expected_date": 0.0,
        "is_ready": False,
        "blocking_reason": blocking_reason,
    }


def _latest_non_null_date(config: Any, field: str) -> str:
    import duckdb  # type: ignore

    date_col = config.data.date_col
    conn = duckdb.connect(str(config.duckdb_path), read_only=True)
    try:
        row = conn.execute(
            f'SELECT MAX("{date_col}") FROM "{config.source_view}" WHERE "{field}" IS NOT NULL'
        ).fetchone()
    finally:
        conn.close()
    return str(row[0])[:10] if row and row[0] is not None else ""


def _non_null_rate(config: Any, field: str, date: str) -> float:
    import duckdb  # type: ignore

    date_col = config.data.date_col
    conn = duckdb.connect(str(config.duckdb_path), read_only=True)
    try:
        row = conn.execute(
            f'SELECT COUNT(*) AS n, SUM(CASE WHEN "{field}" IS NOT NULL THEN 1 ELSE 0 END) AS nn '
            f'FROM "{config.source_view}" WHERE "{date_col}" = ?',
            [str(date)],
        ).fetchone()
    finally:
        conn.close()
    total = float(row[0] or 0) if row else 0.0
    non_null = float(row[1] or 0) if row else 0.0
    return non_null / total if total > 0 else 0.0


def _select_market_value_field(config: Any, fields: list[str], expected_date: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    selected = ""
    selected_rate = 0.0
    for field in fields:
        rate = _non_null_rate(config, field, expected_date) if expected_date else 0.0
        row = {"field": str(field), "non_null_rate": float(rate)}
        rows.append(row)
        if not selected and rate >= float(config.data.min_field_non_null_rate):
            selected = str(field)
            selected_rate = float(rate)
    return {
        "selected_market_value_field": selected,
        "selected_market_value_non_null_rate": selected_rate,
        "market_value_candidates": rows,
    }


def _write_status(config: Any, payload: dict[str, Any]) -> None:
    paths = live_paths(config.store_root, config.universe)
    date = str(
        payload.get("resolved_signal_date")
        or payload.get("common_ready_date")
        or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    )
    write_json(paths.data_status_dir / f"{date}.json", payload)
    write_json(paths.data_status_dir / "latest.json", payload)
