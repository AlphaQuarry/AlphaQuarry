from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


FIELD_CATALOG_STALE_AFTER_DAYS = 7
LOW_COVERAGE_THRESHOLD = 0.50


def coverage_counts(frame: pd.DataFrame) -> dict[str, Any]:
    total = int(len(frame)) if not frame.empty else 0
    if frame.empty or "coverage_rate" not in frame.columns:
        available = 0
    else:
        available = int(pd.to_numeric(frame["coverage_rate"], errors="coerce").notna().sum())
    missing = max(0, total - available)
    if total <= 0 or available <= 0:
        status = "missing"
    elif missing > 0:
        status = "partial"
    else:
        status = "available"
    return {
        "coverage_available_count": available,
        "coverage_missing_count": missing,
        "coverage_status": status,
    }


def low_coverage_count(frame: pd.DataFrame, threshold: float = LOW_COVERAGE_THRESHOLD) -> int:
    if frame.empty or "coverage_rate" not in frame.columns:
        return 0
    values = pd.to_numeric(frame["coverage_rate"], errors="coerce")
    return int((values < float(threshold)).sum())


def data_health_families(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    work = frame.copy()
    work["factor_family"] = work.get("factor_family", "other").fillna("").astype(str).str.strip()
    work.loc[work["factor_family"] == "", "factor_family"] = "other"
    rows: list[dict[str, Any]] = []
    for family, group in work.groupby("factor_family", sort=True):
        coverage = pd.to_numeric(group.get("coverage_rate", pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "family": str(family),
                "field_count": int(len(group)),
                "searchable_count": int(
                    _bool_series(group.get("is_searchable", pd.Series(False, index=group.index))).sum()
                ),
                "avg_coverage_rate": _nullable_float(coverage.mean()),
                "low_coverage_count": low_coverage_count(group),
                "max_available_end": _max_text(group.get("available_end")),
            }
        )
    return rows


def base_frame_summary(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(path.as_posix()),
        "exists": bool(path.exists()),
        "bytes": 0,
        "mtime_utc": "",
        "rows": 0,
        "columns": 0,
        "date_field_detected": False,
        "code_field_detected": False,
    }
    if not path.exists():
        return out
    try:
        stat = path.stat()
        out["bytes"] = int(stat.st_size)
        out["mtime_utc"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    except Exception:
        pass
    try:
        import pyarrow.parquet as pq

        meta = pq.ParquetFile(path)
        cols = list(meta.schema_arrow.names)
        out["rows"] = int(meta.metadata.num_rows)
        out["columns"] = int(len(cols))
        lowered = {str(col).lower() for col in cols}
        out["date_field_detected"] = bool(lowered & {"trade_date", "date", "datetime"})
        out["code_field_detected"] = bool(lowered & {"ts_code", "code", "symbol", "ticker"})
    except Exception:
        try:
            frame = pd.read_parquet(path)
            out["rows"] = int(len(frame))
            out["columns"] = int(len(frame.columns))
            lowered = {str(col).lower() for col in frame.columns}
            out["date_field_detected"] = bool(lowered & {"trade_date", "date", "datetime"})
            out["code_field_detected"] = bool(lowered & {"ts_code", "code", "symbol", "ticker"})
        except Exception:
            pass
    return out


def run_health_summary(path: Path, limit: int = 20) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    recent = records[-max(1, int(limit)) :]
    status_counts = Counter(str(row.get("status", "")) for row in recent if str(row.get("status", "")))
    memory_warning_count = 0
    hard_limit_count = 0
    scoreboard_rows: list[int] = []
    for row in recent:
        memory = row.get("source_chunk_memory", {})
        if isinstance(memory, dict):
            memory_warning_count += int(memory.get("source_chunk_mem_warning_count", 0) or 0)
        if bool(row.get("source_chunk_hard_limit_triggered", False)):
            hard_limit_count += 1
        scoreboard_rows.append(_to_int(row.get("scoreboard_rows"), 0))
    return {
        "path": str(path.as_posix()),
        "exists": bool(path.exists()),
        "total_records": int(len(records)),
        "inspected_records": int(len(recent)),
        "status_counts": dict(status_counts),
        "hard_limit_count": int(hard_limit_count),
        "memory_warning_count": int(memory_warning_count),
        "scoreboard_rows_min": min(scoreboard_rows) if scoreboard_rows else 0,
        "scoreboard_rows_max": max(scoreboard_rows) if scoreboard_rows else 0,
        "latest_status": str(recent[-1].get("status", "")) if recent else "",
    }


def quality_artifact_summary(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(path.as_posix()),
        "exists": bool(path.exists()),
        "overall_status": "",
        "generated_at_utc": "",
        "warn_field_count": 0,
        "fail_field_count": 0,
    }
    if not path.exists():
        return out
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return out
    fields = payload.get("fields", []) if isinstance(payload, dict) else []
    statuses = [str(row.get("status", "")).lower() for row in fields if isinstance(row, dict)]
    out.update(
        {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "generated_at_utc": str(payload.get("generated_at_utc", "") or payload.get("created_at_utc", "") or ""),
            "warn_field_count": int(sum(1 for item in statuses if item == "warn")),
            "fail_field_count": int(sum(1 for item in statuses if item == "fail")),
        }
    )
    return out


def _bool_series(series: Any) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    if series.empty:
        return pd.Series(dtype=bool)
    return series.fillna(False).astype(bool)


def _max_text(series: Any) -> str | None:
    if series is None:
        return None
    values = pd.Series(series).dropna().astype(str)
    values = values[values.str.strip() != ""]
    if values.empty:
        return None
    return str(values.max())


def _nullable_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)
