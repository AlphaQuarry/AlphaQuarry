from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .ingestion_scope import resolve_fact_table_selection


HIGH_COST_TUSHARE_TABLES: frozenset[str] = frozenset(
    {"cyq_perf", "stk_factor_pro", "stk_auction_o", "stk_auction_c", "report_rc"}
)


def build_panel_quality_report(
    df: pd.DataFrame,
    expected_fields: Iterable[str] = (),
    required_fields: Iterable[str] = (),
    date_col: str = "date",
    code_col: str = "code",
    max_missing_ratio: float = 0.30,
    max_inf_ratio: float = 0.0,
    min_dates: int = 1,
    min_codes: int = 1,
) -> dict[str, Any]:
    work = pd.DataFrame() if df is None else df.copy()
    expected = _ordered_tokens(expected_fields)
    required = set(_ordered_tokens(required_fields))
    if not expected:
        expected = [str(c) for c in work.columns if str(c) not in {str(date_col), str(code_col)}]

    row_count = int(len(work))
    date_count = _nunique_present(work, date_col)
    code_count = _nunique_present(work, code_col)
    failures: list[str] = []
    warnings: list[str] = []
    if row_count <= 0:
        failures.append("empty_panel")
    if date_count < int(min_dates):
        failures.append("insufficient_dates")
    if code_count < int(min_codes):
        failures.append("insufficient_codes")

    field_reports = [
        _field_quality(
            work=work,
            field=name,
            required=name in required,
            max_missing_ratio=max_missing_ratio,
            max_inf_ratio=max_inf_ratio,
        )
        for name in expected
    ]
    for item in field_reports:
        status = str(item.get("status", "pass"))
        reason = str(item.get("reason", ""))
        if status == "fail":
            failures.append(f"{item.get('field')}: {reason}")
        elif status == "warn":
            warnings.append(f"{item.get('field')}: {reason}")

    overall = "fail" if failures else ("warn" if warnings else "pass")
    return {
        "schema_version": "panel_quality_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "row_count": row_count,
        "column_count": int(len(work.columns)),
        "date_col": str(date_col),
        "code_col": str(code_col),
        "date_count": int(date_count),
        "code_count": int(code_count),
        "date_start": _date_bound(work, date_col, "min"),
        "date_end": _date_bound(work, date_col, "max"),
        "thresholds": {
            "max_missing_ratio": float(max_missing_ratio),
            "max_inf_ratio": float(max_inf_ratio),
            "min_dates": int(min_dates),
            "min_codes": int(min_codes),
        },
        "failures": failures,
        "warnings": warnings,
        "fields": field_reports,
    }


def build_tushare_smoke_plan(
    start_date: str,
    end_date: str,
    fact_groups: str = "p3",
    fact_tables: str = "",
    exclude_fact_tables: str = "",
) -> dict[str, Any]:
    selected = resolve_fact_table_selection(
        groups_raw=str(fact_groups or ""),
        include_raw=str(fact_tables or ""),
        exclude_raw=str(exclude_fact_tables or ""),
    )
    return {
        "schema_version": "tushare_smoke_plan_v1",
        "mode": "dry_run",
        "start_date": str(start_date or ""),
        "end_date": str(end_date or ""),
        "selected_fact_tables": selected,
        "table_notes": [
            {
                "table": table,
                "cost": "high" if table in HIGH_COST_TUSHARE_TABLES else "normal",
                "requires_explicit_selection": bool(table in HIGH_COST_TUSHARE_TABLES),
            }
            for table in selected
        ],
        "will_call_real_api": False,
    }


def render_panel_quality_markdown(report: dict[str, Any]) -> str:
    fields = list(report.get("fields", []) or [])
    lines = [
        "# Panel Quality Report",
        "",
        f"- overall_status: {report.get('overall_status', '')}",
        f"- rows: {report.get('row_count', 0)}",
        f"- dates: {report.get('date_count', 0)} ({report.get('date_start', '')} -> {report.get('date_end', '')})",
        f"- codes: {report.get('code_count', 0)}",
        "",
        "| field | status | missing_ratio | inf_ratio | present | reason |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for item in fields:
        lines.append(
            "| {field} | {status} | {missing:.6f} | {inf:.6f} | {present} | {reason} |".format(
                field=str(item.get("field", "")),
                status=str(item.get("status", "")),
                missing=float(item.get("missing_ratio", 0.0) or 0.0),
                inf=float(item.get("inf_ratio", 0.0) or 0.0),
                present=str(bool(item.get("present", False))).lower(),
                reason=str(item.get("reason", "")),
            )
        )
    return "\n".join(lines) + "\n"


def write_quality_artifacts(
    report: dict[str, Any],
    json_out: str | Path = "",
    markdown_out: str | Path = "",
) -> dict[str, str]:
    out: dict[str, str] = {}
    if str(json_out or "").strip():
        path = Path(json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_to_serializable(report), ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        out["json_out"] = str(path.as_posix())
    if str(markdown_out or "").strip():
        path = Path(markdown_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_panel_quality_markdown(report), encoding="utf-8")
        out["markdown_out"] = str(path.as_posix())
    return out


def _field_quality(
    work: pd.DataFrame,
    field: str,
    required: bool,
    max_missing_ratio: float,
    max_inf_ratio: float,
) -> dict[str, Any]:
    if field not in work.columns:
        return {
            "field": str(field),
            "present": False,
            "required": bool(required),
            "missing_ratio": 1.0,
            "inf_ratio": 0.0,
            "status": "fail" if required else "warn",
            "reason": "missing_required_field" if required else "missing_optional_field",
        }
    series = work[field]
    row_count = max(1, int(len(series)))
    missing_ratio = float(series.isna().sum()) / float(row_count)
    numeric = pd.to_numeric(series, errors="coerce")
    inf_ratio = float(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum()) / float(row_count)

    status = "pass"
    reason = ""
    if inf_ratio > float(max_inf_ratio):
        status = "fail" if required else "warn"
        reason = "inf_ratio_above_threshold"
    elif missing_ratio > float(max_missing_ratio):
        status = "fail" if required else "warn"
        reason = "missing_ratio_above_threshold"
    return {
        "field": str(field),
        "present": True,
        "required": bool(required),
        "missing_ratio": missing_ratio,
        "inf_ratio": inf_ratio,
        "status": status,
        "reason": reason,
    }


def _ordered_tokens(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        token = str(value or "").strip()
        if token and token not in out:
            out.append(token)
    return out


def _nunique_present(work: pd.DataFrame, col: str) -> int:
    if col not in work.columns:
        return 0
    return int(work[col].dropna().nunique())


def _date_bound(work: pd.DataFrame, col: str, fn: str) -> str:
    if col not in work.columns or work.empty:
        return ""
    series = pd.to_datetime(work[col], errors="coerce").dropna()
    if series.empty:
        return ""
    value = series.min() if str(fn) == "min" else series.max()
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_serializable(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value
