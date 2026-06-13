from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


def build_duckdb_runtime_settings(
    duckdb_path: str | Path,
    temp_directory: str = "",
    isolate_run: bool = True,
    run_id: str = "",
    run_prefix: str = "run_duckdb",
    memory_limit: str = "",
    threads: int = 0,
    max_temp_directory_size: str = "",
) -> dict[str, Any]:
    db_path = Path(duckdb_path)
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path
    temp_root = Path(temp_directory) if str(temp_directory or "").strip() else Path(f"{str(db_path)}.tmp")
    if not temp_root.is_absolute():
        temp_root = Path.cwd() / temp_root

    temp_dir = temp_root
    if bool(isolate_run):
        suffix = _safe_run_id(run_id or f"{run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        temp_dir = temp_root / suffix

    out: dict[str, Any] = {
        "temp_directory": str(temp_dir.as_posix()),
    }
    if str(memory_limit or "").strip():
        out["memory_limit"] = str(memory_limit).strip()
    try:
        threads_int = int(threads)
    except Exception:
        threads_int = 0
    if threads_int > 0:
        out["threads"] = int(threads_int)
    if str(max_temp_directory_size or "").strip():
        out["max_temp_directory_size"] = str(max_temp_directory_size).strip()
    return out


def normalize_duckdb_connection_config(
    duckdb_path: str | Path,
    duckdb_settings: dict[str, Any] | None,
) -> dict[str, str]:
    if not isinstance(duckdb_settings, dict) or not duckdb_settings:
        return {}

    out: dict[str, str] = {}
    memory_limit = str(duckdb_settings.get("memory_limit", "") or "").strip()
    if memory_limit:
        out["memory_limit"] = memory_limit

    threads_raw = duckdb_settings.get("threads", "")
    threads_text = str(threads_raw or "").strip()
    if threads_text:
        try:
            threads_val = int(threads_text)
        except Exception:
            threads_val = 0
        if threads_val > 0:
            out["threads"] = str(threads_val)

    temp_directory = str(duckdb_settings.get("temp_directory", "") or "").strip()
    if temp_directory:
        temp_path = Path(temp_directory)
        if not temp_path.is_absolute():
            temp_path = Path(duckdb_path).resolve().parent / temp_path
        temp_path.mkdir(parents=True, exist_ok=True)
        out["temp_directory"] = str(temp_path.as_posix())

    max_temp_size = str(duckdb_settings.get("max_temp_directory_size", "") or "").strip()
    if max_temp_size:
        out["max_temp_directory_size"] = max_temp_size
    return out


def _safe_run_id(raw: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(raw or "").strip())
    return out or f"run_duckdb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
