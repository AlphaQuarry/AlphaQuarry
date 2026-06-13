from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .artifacts import live_paths, read_json, utc_now_iso, write_json


VALID_STATUSES = {"active", "paused", "retired"}


def activate_superalpha(
    *,
    base_dir: str | Path,
    universe: str,
    superalpha_id: str,
    activated_by: str = "dashboard",
    status: str = "active",
    max_active: int | None = None,
) -> dict[str, Any]:
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid live status: {status}")
    paths = live_paths(base_dir, universe)
    meta_path = paths.universe_root / "superalphas" / str(superalpha_id) / "meta.json"
    if not meta_path.exists():
        raise KeyError(f"Superalpha meta not found: {superalpha_id}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    snapshot = _snapshot_from_meta(meta=meta, meta_path=meta_path, activated_status=status)
    now = utc_now_iso()
    registry = _load_registry(paths.registry_path, universe=str(universe))
    active = list(registry.get("active") or [])
    existing = next(
        (row for row in active if str(row.get("superalpha_id")) == str(superalpha_id)),
        None,
    )
    if status == "active":
        active_count = sum(
            1
            for row in active
            if str(row.get("status") or "") == "active" and str(row.get("superalpha_id")) != str(superalpha_id)
        )
        if (
            max_active is not None
            and active_count >= int(max_active)
            and (existing is None or str(existing.get("status") or "") != "active")
        ):
            raise ValueError(f"max_active exceeded: {max_active}")
    if existing is None:
        existing = {
            "superalpha_id": str(superalpha_id),
            "display_name": str(meta.get("name") or superalpha_id),
            "status": status,
            "activated_at_utc": now,
            "activated_by": activated_by,
            "last_live_run_id": "",
            "last_signal_date": "",
            "last_execute_date": "",
            "snapshot": snapshot,
        }
        active.append(existing)
    else:
        existing["status"] = status
        existing["updated_at_utc"] = now
        existing.setdefault("activated_at_utc", now)
        existing["snapshot"] = existing.get("snapshot") or snapshot
    registry.update(
        {
            "schema_version": 1,
            "universe": str(universe),
            "updated_at_utc": now,
            "active": active,
        }
    )
    write_json(paths.registry_path, registry)
    return {
        "status": status,
        "record": existing,
        "active_count": sum(1 for row in active if str(row.get("status") or "") == "active"),
    }


def list_live_superalphas(
    *,
    base_dir: str | Path,
    universe: str,
    include_paused: bool = False,
    include_retired: bool = False,
) -> list[dict[str, Any]]:
    paths = live_paths(base_dir, universe)
    registry = _load_registry(paths.registry_path, universe=str(universe))
    rows = []
    for row in registry.get("active") or []:
        status = str(row.get("status") or "")
        if status == "paused" and not include_paused:
            continue
        if status == "retired" and not include_retired:
            continue
        if status != "active" and status not in {"paused", "retired"}:
            continue
        out = dict(row)
        source = str((out.get("snapshot") or {}).get("source_meta_path") or "")
        out["source_meta_exists"] = Path(source).exists() if source else False
        rows.append(out)
    return rows


def update_superalpha_status(*, base_dir: str | Path, universe: str, superalpha_id: str, status: str) -> dict[str, Any]:
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid live status: {status}")
    paths = live_paths(base_dir, universe)
    registry = _load_registry(paths.registry_path, universe=str(universe))
    for row in registry.get("active") or []:
        if str(row.get("superalpha_id")) == str(superalpha_id):
            row["status"] = status
            row["updated_at_utc"] = utc_now_iso()
            registry["updated_at_utc"] = row["updated_at_utc"]
            write_json(paths.registry_path, registry)
            return {"status": "ok", "record": row}
    raise KeyError(f"Unknown live superalpha: {superalpha_id}")


def load_active_snapshots(*, base_dir: str | Path, universe: str, superalpha_id: str = "all") -> list[dict[str, Any]]:
    rows = list_live_superalphas(base_dir=base_dir, universe=universe)
    if str(superalpha_id or "all") != "all":
        rows = [row for row in rows if str(row.get("superalpha_id")) == str(superalpha_id)]
    return [dict(row.get("snapshot") or {}) | {"registry_record": row} for row in rows]


def _load_registry(path: Path, *, universe: str) -> dict[str, Any]:
    return read_json(path, {"schema_version": 1, "universe": universe, "active": []}) or {
        "schema_version": 1,
        "universe": universe,
        "active": [],
    }


def _snapshot_from_meta(*, meta: dict[str, Any], meta_path: Path, activated_status: str) -> dict[str, Any]:
    components = meta.get("components") if isinstance(meta.get("components"), list) else []
    raw = meta_path.read_bytes()
    activated_at = utc_now_iso()
    return {
        "schema_version": 1,
        "superalpha_id": str(meta.get("superalpha_id") or meta_path.parent.name),
        "universe": str(meta.get("universe") or meta_path.parents[2].name),
        "activated_at_utc": activated_at,
        "status": activated_status,
        "source_meta_path": str(meta_path.as_posix()),
        "source_meta_mtime": meta_path.stat().st_mtime,
        "source_meta_hash": hashlib.sha256(raw).hexdigest(),
        "combo_expression": str(meta.get("combo_expression") or "1"),
        "component_count": int(meta.get("component_count") or len(components)),
        "components": components,
        "component_factor_ids": [str(c.get("factor") or "") for c in components],
        "component_expressions": [str(c.get("expression") or "") for c in components],
        "component_weights": [float(c.get("weight") or 1.0) for c in components],
        "component_normalization": str(meta.get("component_normalization") or "cs_zscore"),
        "final_normalization": str(meta.get("final_normalization") or "cs_zscore"),
        "component_join": str(meta.get("component_join") or "concat"),
        "direction_adjustment": bool(meta.get("direction_adjustment", True)),
        "direction_sources": [str(c.get("direction_status") or c.get("direction_source") or "") for c in components],
        "direction_signs": [float(c.get("direction_sign") or 1.0) for c in components],
        "period": int(meta.get("period") or 0),
        "layers": int(meta.get("layers") or 0),
        "summary_metrics": dict(meta.get("summary") or {}),
    }
