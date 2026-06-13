from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .artifacts import live_paths


def apply_retention(*, config: Any, now_date: str | None = None) -> dict[str, Any]:
    paths = live_paths(config.store_root, config.universe)
    root = paths.live_root
    if not root.exists():
        return {
            "deleted_files": 0,
            "kept_latest_refs": 0,
            "scanned": 0,
            "skipped_latest": 0,
            "protected_failed": 0,
        }
    keep_refs = _latest_references(root)
    cutoff = (
        date.fromisoformat(str(now_date)) - timedelta(days=int(config.retention.keep_daily_artifacts_days))
        if now_date
        else date.today() - timedelta(days=int(config.retention.keep_daily_artifacts_days))
    )
    failed_cutoff = (
        date.fromisoformat(str(now_date)) - timedelta(days=int(config.retention.keep_failed_jobs_days))
        if now_date
        else date.today() - timedelta(days=int(config.retention.keep_failed_jobs_days))
    )
    deleted = 0
    scanned = 0
    skipped_latest = 0
    protected_failed = 0
    for path in list(root.glob("**/*")):
        if not path.is_file() or path.name == "latest.json" or path.suffix.lower() not in {".parquet", ".csv", ".json"}:
            continue
        scanned += 1
        if path.resolve() in keep_refs:
            skipped_latest += 1
            continue
        is_job = _is_job_path(root, path)
        payload = _read_json(path) if is_job and path.suffix.lower() == ".json" else {}
        stem_date = _payload_date(payload) if payload else _date_from_name(path.name)
        effective_cutoff = failed_cutoff if is_job and str(payload.get("status") or "").lower() == "failed" else cutoff
        if stem_date is not None and stem_date < effective_cutoff:
            if is_job and str(payload.get("status") or "").lower() == "failed" and stem_date >= failed_cutoff:
                protected_failed += 1
                continue
            path.unlink(missing_ok=True)
            deleted += 1
    return {
        "deleted_files": deleted,
        "kept_latest_refs": len(keep_refs),
        "scanned": scanned,
        "skipped_latest": skipped_latest,
        "protected_failed": protected_failed,
    }


def _latest_references(root: Path) -> set[Path]:
    refs: set[Path] = set()
    for latest in root.glob("**/latest.json"):
        try:
            payload = json.loads(latest.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in (
            "artifact_path",
            "signal_path",
            "holdings_path",
            "orders_path",
            "orders_csv_path",
            "summary_path",
        ):
            value = str(payload.get(key) or "")
            if value:
                refs.add(Path(value).resolve())
    return refs


def _date_from_name(name: str) -> date | None:
    try:
        return date.fromisoformat(Path(name).stem[:10])
    except Exception:
        return None


def _payload_date(payload: dict[str, Any]) -> date | None:
    for key in ("created_at_utc", "updated_at_utc", "run_started_at"):
        raw = str(payload.get(key) or "").strip()
        if not raw:
            continue
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.date()
        except Exception:
            continue
    return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _is_job_path(root: Path, path: Path) -> bool:
    try:
        return "jobs" in path.relative_to(root).parts
    except Exception:
        return False
