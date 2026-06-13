from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


CATALOG_COLUMNS = (
    "field_name",
    "available_at",
    "leakage_safe",
    "field_role",
    "source_table",
)


def load_live_field_catalog(*, config: Any) -> dict[str, Any]:
    frame = _load_from_duckdb(config)
    source = "duckdb:v_project_field_catalog" if not frame.empty else ""
    if frame.empty:
        frame = _load_from_artifact(config)
        source = "artifact:field_catalog.parquet" if not frame.empty else ""
    if frame.empty:
        return {"status": "missing", "source": source, "fields": {}}
    out = _normalize_catalog(frame)
    fields = {str(row["field_name"]): row for row in out.to_dict(orient="records") if str(row.get("field_name") or "")}
    return {"status": "ok", "source": source, "fields": fields}


def enrich_field_row_from_catalog(row: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    field = str(row.get("field") or "")
    fields = catalog.get("fields") if isinstance(catalog, dict) else {}
    meta = fields.get(field) if isinstance(fields, dict) else None
    if not isinstance(meta, dict):
        return {
            **row,
            "available_at": "",
            "leakage_safe": None,
            "field_role": "",
            "source_table": "",
            "catalog_status": "catalog_missing" if catalog.get("status") != "ok" else "field_missing",
        }
    return {
        **row,
        "available_at": str(meta.get("available_at") or ""),
        "leakage_safe": _to_bool_or_none(meta.get("leakage_safe")),
        "field_role": str(meta.get("field_role") or ""),
        "source_table": str(meta.get("source_table") or ""),
        "catalog_status": "ok",
    }


def catalog_field_blocks_live(row: dict[str, Any], *, strict_available_at: bool) -> str:
    if row.get("leakage_safe") is False:
        return "catalog_leakage_unsafe"
    if strict_available_at and _available_after_live_signal(str(row.get("available_at") or "")):
        return "catalog_available_at_late"
    return ""


def _load_from_duckdb(config: Any) -> pd.DataFrame:
    try:
        import duckdb  # type: ignore

        conn = duckdb.connect(str(config.duckdb_path), read_only=True)
        try:
            return conn.execute("SELECT * FROM v_project_field_catalog").fetchdf()
        finally:
            conn.close()
    except Exception:
        return pd.DataFrame()


def _load_from_artifact(config: Any) -> pd.DataFrame:
    candidates = [
        Path(config.store_root) / "data" / "lake" / "meta" / "field_catalog.parquet",
        Path.cwd() / "data" / "lake" / "meta" / "field_catalog.parquet",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            return pd.read_parquet(path)
        except Exception:
            continue
    return pd.DataFrame()


def _normalize_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "field_name" not in out.columns and "name" in out.columns:
        out["field_name"] = out["name"]
    for col in CATALOG_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    for col in ("field_name", "available_at", "field_role", "source_table"):
        out[col] = out[col].fillna("").astype(str)
    return out[list(CATALOG_COLUMNS)]


def _to_bool_or_none(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _available_after_live_signal(value: str) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    late_tokens = ("next_day", "t+1", "after_execute", "post_execute", "report_after")
    return any(token in text for token in late_tokens)
