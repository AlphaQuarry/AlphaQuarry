from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from .config import LakePathSettings


COVERAGE_COLUMNS: tuple[str, ...] = (
    "coverage_scope",
    "coverage_start_date",
    "coverage_end_date",
    "coverage_row_count",
    "non_null_count",
    "coverage_rate",
    "missing_rate",
    "finite_count",
    "finite_rate",
    "coverage_status",
    "coverage_updated_at_utc",
)

HEAVY_ASOF_SOURCE_TABLES: tuple[str, ...] = ("v_project_financial_asof_daily",)
AUTO_LIGHT_CHUNK_SIZE = 16
AUTO_PANEL_CHUNK_SIZE = 8


def refresh_field_coverage(
    paths: LakePathSettings,
    source_view: str = "v_project_panel_cn_a",
    start_date: str = "",
    end_date: str = "",
    fields: Sequence[str] = (),
    run_filters: dict[str, Any] | None = None,
    update_field_catalog: bool = False,
    output_csv_path: str | Path = "",
    chunk_size: int = 24,
    duckdb_settings: dict[str, Any] | None = None,
    include_source_tables: Sequence[str] = (),
    exclude_source_tables: Sequence[str] = (),
    include_heavy_asof: bool = False,
    incremental_csv: bool = True,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    try:
        import duckdb  # type: ignore
    except Exception as exc:
        raise RuntimeError("duckdb is required but not installed") from exc

    db_path = paths.duckdb_path_obj
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")

    effective_duckdb_settings = _normalize_duckdb_settings(db_path=db_path, duckdb_settings=duckdb_settings or {})
    conn = duckdb.connect(
        str(db_path),
        read_only=not bool(update_field_catalog),
        config=effective_duckdb_settings,
    )
    try:
        catalog_df = _load_field_catalog(conn=conn, paths=paths)
        selected_fields = _filter_fields_by_source_table(
            fields=_resolve_fields(fields=fields, catalog_df=catalog_df),
            catalog_df=catalog_df,
            include_source_tables=include_source_tables,
            exclude_source_tables=exclude_source_tables,
        )
        skipped_heavy_fields: list[str] = []
        if not bool(include_heavy_asof):
            selected_fields, skipped_heavy_fields = _split_heavy_asof_fields(
                fields=selected_fields,
                catalog_df=catalog_df,
            )
        available = _view_column_types(conn=conn, source_view=source_view)
        row_count_source = _resolve_lightweight_source(
            conn=conn,
            source_view=source_view,
            required_fields=_filter_required_fields(run_filters=run_filters or {}),
        )
        row_available = _view_column_types(conn=conn, source_view=row_count_source)
        where_sql, params = _build_where_clause(
            available_fields=set(row_available.keys()),
            start_date=start_date,
            end_date=end_date,
            run_filters=run_filters or {},
        )
        row_count = int(
            conn.execute(f"SELECT COUNT(*) FROM {_qident(row_count_source)}{where_sql}", params).fetchone()[0]
        )
        output_path = Path(output_csv_path) if output_csv_path else None
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        coverage_df, field_source_counts, incremental_csv_written = _compute_coverage(
            conn=conn,
            source_view=source_view,
            selected_fields=selected_fields,
            available_types=available,
            row_count=row_count,
            where_sql=where_sql,
            params=params,
            start_date=start_date,
            end_date=end_date,
            chunk_size=chunk_size,
            run_filters=run_filters or {},
            incremental_csv_path=output_path if bool(incremental_csv) else None,
            progress_callback=progress_callback,
        )
        skipped_df = _build_skipped_heavy_coverage_df(
            fields=skipped_heavy_fields,
            row_count=row_count,
            start_date=start_date,
            end_date=end_date,
        )
        if not skipped_df.empty:
            coverage_df = pd.concat([coverage_df, skipped_df], ignore_index=True)
            if output_path is not None and bool(incremental_csv):
                _append_coverage_rows_to_csv(
                    skipped_df.to_dict("records"),
                    output_path,
                    overwrite=not bool(incremental_csv_written),
                )
                incremental_csv_written = True
        if output_path is not None and bool(incremental_csv) and not bool(incremental_csv_written):
            pd.DataFrame(columns=["field_name", *COVERAGE_COLUMNS]).to_csv(output_path, index=False)
        if output_path is not None and not bool(incremental_csv):
            coverage_df.to_csv(output_path, index=False)

        if bool(update_field_catalog):
            _merge_coverage_into_field_catalog(
                conn=conn,
                paths=paths,
                catalog_df=catalog_df,
                coverage_df=coverage_df,
            )

        return {
            "coverage_df": coverage_df,
            "row_count": row_count,
            "field_count": int(len(coverage_df)),
            "output_csv_path": str(output_path.as_posix()) if output_path is not None else "",
            "field_catalog_path": str((paths.meta_path / "field_catalog.parquet").as_posix()),
            "updated_field_catalog": bool(update_field_catalog),
            "row_count_source": str(row_count_source),
            "field_source_counts": field_source_counts,
            "duckdb_settings": dict(effective_duckdb_settings),
            "include_source_tables": list(include_source_tables),
            "exclude_source_tables": list(exclude_source_tables),
            "include_heavy_asof": bool(include_heavy_asof),
            "skipped_heavy_field_count": int(len(skipped_heavy_fields)),
        }
    finally:
        conn.close()


def _load_field_catalog(conn: Any, paths: LakePathSettings) -> pd.DataFrame:
    try:
        df = conn.execute("SELECT * FROM v_project_field_catalog").fetchdf()
        if isinstance(df, pd.DataFrame) and not df.empty and "field_name" in df.columns:
            return df
    except Exception:
        pass
    path = paths.meta_path / "field_catalog.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["field_name"])


def _resolve_fields(fields: Sequence[str], catalog_df: pd.DataFrame) -> list[str]:
    explicit = _ordered_tokens(fields)
    if explicit:
        return explicit
    if isinstance(catalog_df, pd.DataFrame) and "field_name" in catalog_df.columns:
        return _ordered_tokens(catalog_df["field_name"].astype(str).tolist())
    return []


def _compute_coverage(
    conn: Any,
    source_view: str,
    selected_fields: Sequence[str],
    available_types: dict[str, str],
    row_count: int,
    where_sql: str,
    params: Sequence[Any],
    start_date: str,
    end_date: str,
    chunk_size: int,
    run_filters: dict[str, Any],
    incremental_csv_path: Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, int], bool]:
    updated_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    present_fields = [field for field in selected_fields if field in available_types]
    missing_fields = [field for field in selected_fields if field not in available_types]
    stats: dict[str, dict[str, Any]] = {}
    field_source_counts: dict[str, int] = {}
    incremental_csv_written = False

    source_groups = _group_fields_by_query_source(
        conn=conn,
        source_view=source_view,
        fields=present_fields,
        run_filters=run_filters,
    )
    chunks: list[tuple[str, list[str]]] = []
    for query_source, source_fields in source_groups.items():
        per_source_chunk_size = _effective_chunk_size(
            requested_chunk_size=int(chunk_size),
            query_source=query_source,
            source_view=source_view,
        )
        chunks.extend((query_source, chunk) for chunk in _chunks(source_fields, per_source_chunk_size))
    total_chunks = len(chunks)
    for chunk_index, (query_source, chunk) in enumerate(chunks, start=1):
        source_available = _view_column_types(conn=conn, source_view=query_source)
        chunk_where_sql, chunk_params = _build_where_clause(
            available_fields=set(source_available.keys()),
            start_date=start_date,
            end_date=end_date,
            run_filters=run_filters,
        )
        field_source_counts[query_source] = int(field_source_counts.get(query_source, 0)) + int(len(chunk))
        select_parts: list[str] = []
        for field in chunk:
            quoted = _qident(field)
            select_parts.append(f"COUNT({quoted}) AS {_qident(field + '__non_null')}")
            if _is_numeric_type(available_types.get(field, "")):
                select_parts.append(
                    f"SUM(CASE WHEN {quoted} IS NOT NULL AND isfinite(CAST({quoted} AS DOUBLE)) "
                    f"THEN 1 ELSE 0 END) AS {_qident(field + '__finite')}"
                )
        if not select_parts:
            continue
        out = conn.execute(
            f"SELECT {', '.join(select_parts)} FROM {_qident(query_source)}{chunk_where_sql}",
            list(chunk_params),
        ).fetchdf()
        record = out.iloc[0].to_dict() if not out.empty else {}
        for field in chunk:
            non_null = int(record.get(field + "__non_null", 0) or 0)
            finite_key = field + "__finite"
            finite_value = record.get(finite_key, pd.NA)
            stats[field] = {
                "non_null_count": non_null,
                "finite_count": int(finite_value) if pd.notna(finite_value) else pd.NA,
            }
        if incremental_csv_path is not None:
            _append_coverage_rows_to_csv(
                [
                    _coverage_row(
                        field=field,
                        row_count=row_count,
                        non_null_count=int(stats[field]["non_null_count"]),
                        finite_count=stats[field]["finite_count"],
                        start_date=start_date,
                        end_date=end_date,
                        status="ok",
                        updated_at=updated_at,
                    )
                    for field in chunk
                ],
                incremental_csv_path,
                overwrite=not bool(incremental_csv_written),
            )
            incremental_csv_written = True
        if progress_callback is not None:
            progress_callback(
                {
                    "chunk_index": int(chunk_index),
                    "chunk_count": int(total_chunks),
                    "source_view": str(query_source),
                    "field_count": int(len(chunk)),
                    "fields": list(chunk),
                }
            )

    for field in selected_fields:
        item = stats.get(field, {})
        non_null_count = int(item.get("non_null_count", 0) or 0)
        finite_count = item.get("finite_count", pd.NA)
        is_missing = field in missing_fields
        rows.append(
            _coverage_row(
                field=field,
                row_count=row_count,
                non_null_count=non_null_count,
                finite_count=finite_count,
                start_date=start_date,
                end_date=end_date,
                status="missing_field" if is_missing else "ok",
                updated_at=updated_at,
            )
        )
    if incremental_csv_path is not None:
        missing_rows = [row for row in rows if str(row.get("coverage_status", "")) == "missing_field"]
        _append_coverage_rows_to_csv(
            missing_rows,
            incremental_csv_path,
            overwrite=not bool(incremental_csv_written),
        )
        incremental_csv_written = bool(incremental_csv_written or missing_rows)
    return pd.DataFrame(rows), field_source_counts, incremental_csv_written


def _group_fields_by_query_source(
    conn: Any,
    source_view: str,
    fields: Sequence[str],
    run_filters: dict[str, Any],
) -> dict[str, list[str]]:
    required_filters = _filter_required_fields(run_filters=run_filters)
    out: dict[str, list[str]] = {}
    for field in fields:
        query_source = _resolve_lightweight_source(
            conn=conn,
            source_view=source_view,
            required_fields=[str(field), *required_filters],
        )
        out.setdefault(query_source, []).append(str(field))
    return out


def _effective_chunk_size(requested_chunk_size: int, query_source: str, source_view: str) -> int:
    requested = int(requested_chunk_size)
    if requested > 0:
        return max(1, requested)
    if str(query_source) == str(source_view) and str(source_view) == "v_project_panel_cn_a":
        return AUTO_PANEL_CHUNK_SIZE
    return AUTO_LIGHT_CHUNK_SIZE


def _coverage_row(
    field: str,
    row_count: int,
    non_null_count: int,
    finite_count: Any,
    start_date: str,
    end_date: str,
    status: str,
    updated_at: str,
) -> dict[str, Any]:
    coverage_rate = _safe_ratio(non_null_count, row_count)
    return {
        "field_name": str(field),
        "coverage_scope": "closed_loop",
        "coverage_start_date": str(start_date or ""),
        "coverage_end_date": str(end_date or ""),
        "coverage_row_count": int(row_count),
        "non_null_count": int(non_null_count),
        "coverage_rate": coverage_rate,
        "missing_rate": 1.0 - coverage_rate if row_count > 0 else 1.0,
        "finite_count": finite_count,
        "finite_rate": _safe_ratio(int(finite_count), row_count) if pd.notna(finite_count) else pd.NA,
        "coverage_status": str(status),
        "coverage_updated_at_utc": updated_at,
    }


def _append_coverage_rows_to_csv(rows: Sequence[dict[str, Any]], output_path: Path, overwrite: bool = False) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    mode = "w" if bool(overwrite) else "a"
    header = bool(overwrite) or not output_path.exists()
    frame.to_csv(output_path, mode=mode, header=header, index=False)


def _filter_fields_by_source_table(
    fields: Sequence[str],
    catalog_df: pd.DataFrame,
    include_source_tables: Sequence[str],
    exclude_source_tables: Sequence[str],
) -> list[str]:
    include = {str(x).strip() for x in include_source_tables if str(x).strip()}
    exclude = {str(x).strip() for x in exclude_source_tables if str(x).strip()}
    if not include and not exclude:
        return list(fields)
    if (
        not isinstance(catalog_df, pd.DataFrame)
        or "field_name" not in catalog_df.columns
        or "source_table" not in catalog_df.columns
    ):
        return list(fields)
    source_map = {
        str(row["field_name"]): str(row.get("source_table", "") or "").strip() for _, row in catalog_df.iterrows()
    }
    out: list[str] = []
    for field in fields:
        source_table = source_map.get(str(field), "")
        if include and source_table not in include:
            continue
        if exclude and source_table in exclude:
            continue
        out.append(str(field))
    return out


def _split_heavy_asof_fields(fields: Sequence[str], catalog_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    if (
        not isinstance(catalog_df, pd.DataFrame)
        or "field_name" not in catalog_df.columns
        or "source_table" not in catalog_df.columns
    ):
        return list(fields), []
    heavy_sources = {str(x).strip() for x in HEAVY_ASOF_SOURCE_TABLES}
    source_map = {
        str(row["field_name"]): str(row.get("source_table", "") or "").strip() for _, row in catalog_df.iterrows()
    }
    selected: list[str] = []
    skipped: list[str] = []
    for field in fields:
        if source_map.get(str(field), "") in heavy_sources:
            skipped.append(str(field))
        else:
            selected.append(str(field))
    return selected, skipped


def _build_skipped_heavy_coverage_df(
    fields: Sequence[str],
    row_count: int,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    if not fields:
        return pd.DataFrame()
    updated_at = datetime.now(timezone.utc).isoformat()
    return pd.DataFrame(
        [
            {
                "field_name": str(field),
                "coverage_scope": "closed_loop",
                "coverage_start_date": str(start_date or ""),
                "coverage_end_date": str(end_date or ""),
                "coverage_row_count": int(row_count),
                "non_null_count": pd.NA,
                "coverage_rate": pd.NA,
                "missing_rate": pd.NA,
                "finite_count": pd.NA,
                "finite_rate": pd.NA,
                "coverage_status": "skipped_heavy_source",
                "coverage_updated_at_utc": updated_at,
            }
            for field in fields
        ]
    )


def _merge_coverage_into_field_catalog(
    conn: Any,
    paths: LakePathSettings,
    catalog_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
) -> None:
    field_catalog_path = paths.meta_path / "field_catalog.parquet"
    field_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    base = pd.DataFrame() if catalog_df is None else catalog_df.copy()
    if "field_name" not in base.columns:
        base = pd.DataFrame({"field_name": coverage_df["field_name"].astype(str).tolist()})
    keep_cols = [col for col in base.columns if col not in COVERAGE_COLUMNS]
    merged = pd.merge(
        base[keep_cols],
        coverage_df[["field_name", *COVERAGE_COLUMNS]],
        on="field_name",
        how="left",
    )
    merged.to_parquet(field_catalog_path, index=False)
    conn.execute(
        "CREATE OR REPLACE VIEW v_project_field_catalog AS "
        f"SELECT * FROM read_parquet('{_escape_sql_path(field_catalog_path)}')"
    )


def _normalize_duckdb_settings(db_path: Path, duckdb_settings: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    memory_limit = str(duckdb_settings.get("memory_limit", "") or "").strip()
    if memory_limit:
        out["memory_limit"] = memory_limit
    try:
        threads = int(duckdb_settings.get("threads", 0) or 0)
    except Exception:
        threads = 0
    if threads > 0:
        out["threads"] = threads
    temp_directory = str(duckdb_settings.get("temp_directory", "") or "").strip()
    if not temp_directory:
        temp_directory = f"{str(db_path)}.tmp"
    temp_path = Path(temp_directory)
    if not temp_path.is_absolute():
        temp_path = Path.cwd() / temp_path
    temp_path.mkdir(parents=True, exist_ok=True)
    out["temp_directory"] = str(temp_path.as_posix())
    max_temp_size = str(duckdb_settings.get("max_temp_directory_size", "") or "").strip()
    if max_temp_size:
        out["max_temp_directory_size"] = max_temp_size
    return out


def _filter_required_fields(run_filters: dict[str, Any]) -> list[str]:
    required = ["date", "code"]
    if _to_bool(run_filters.get("universe_only", True), default=True):
        required.append("universe")
    return required


def _resolve_lightweight_source(conn: Any, source_view: str, required_fields: Sequence[str]) -> str:
    source = str(source_view or "").strip()
    candidates = [source]
    if source == "v_project_panel_cn_a":
        candidates = [
            "v_project_market_daily_base_hot",
            "project_market_daily_base",
            "v_project_market_daily_base",
            source,
        ]
    required = {str(field).strip() for field in required_fields if str(field).strip()}
    for candidate in candidates:
        try:
            available = set(_view_column_types(conn=conn, source_view=candidate).keys())
        except Exception:
            continue
        if required <= available:
            return candidate
    return source


def _view_column_types(conn: Any, source_view: str) -> dict[str, str]:
    schema_name, table_name = _split_view_name(source_view)
    if schema_name:
        df = conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = ? AND table_name = ? ORDER BY ordinal_position",
            [schema_name, table_name],
        ).fetchdf()
    else:
        df = conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = ? ORDER BY ordinal_position",
            [table_name],
        ).fetchdf()
    return {str(row["column_name"]): str(row["data_type"]) for _, row in df.iterrows()}


def _build_where_clause(
    available_fields: set[str],
    start_date: str,
    end_date: str,
    run_filters: dict[str, Any],
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if start_date:
        clauses.append(f"{_qident('date')} >= ?")
        params.append(str(start_date))
    if end_date:
        clauses.append(f"{_qident('date')} <= ?")
        params.append(str(end_date))
    if _to_bool(run_filters.get("universe_only", True), default=True) and "universe" in available_fields:
        clauses.append("COALESCE(universe, 0) = 1")
    if not _to_bool(run_filters.get("include_bj", True), default=True) and "code" in available_fields:
        clauses.append(f"{_qident('code')} NOT LIKE ?")
        params.append("%.BJ")
    include_codes = run_filters.get("include_codes", [])
    if isinstance(include_codes, (list, tuple)) and include_codes and "code" in available_fields:
        placeholders = ", ".join(["?" for _ in include_codes])
        clauses.append(f"{_qident('code')} IN ({placeholders})")
        params.extend([str(x) for x in include_codes])
    code_prefix = str(run_filters.get("code_prefix", "") or "").strip()
    if code_prefix and "code" in available_fields:
        clauses.append(f"{_qident('code')} LIKE ?")
        params.append(f"{code_prefix}%")
    return (" WHERE " + " AND ".join(clauses), params) if clauses else ("", params)


def _is_numeric_type(dtype: str) -> bool:
    text = str(dtype or "").upper()
    return any(
        token in text
        for token in (
            "INT",
            "DOUBLE",
            "FLOAT",
            "REAL",
            "DECIMAL",
            "NUMERIC",
            "HUGEINT",
            "UBIGINT",
        )
    )


def _safe_ratio(value: int, denominator: int) -> float:
    if int(denominator) <= 0:
        return 0.0
    return float(value) / float(denominator)


def _ordered_tokens(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        token = str(value or "").strip()
        if token and token not in out:
            out.append(token)
    return out


def _chunks(values: Sequence[str], size: int) -> Iterable[list[str]]:
    items = list(values)
    for idx in range(0, len(items), max(1, int(size))):
        yield items[idx : idx + max(1, int(size))]


def _split_view_name(source_view: str) -> tuple[str | None, str]:
    parts = [p for p in str(source_view or "").split(".") if p]
    if not parts:
        raise ValueError("source_view is empty")
    if len(parts) == 1:
        return None, parts[0]
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError(f"Unsupported source_view format: {source_view}")


def _qident(name: str) -> str:
    parts = [p for p in str(name).split(".") if p]
    return ".".join('"' + part.replace('"', '""') + '"' for part in parts)


def _escape_sql_path(path: str | Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def _to_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if not text:
        return bool(default)
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)
