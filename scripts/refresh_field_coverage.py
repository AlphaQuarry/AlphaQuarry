from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_mining.datasource import load_datasource_settings, refresh_field_coverage


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    settings = load_datasource_settings(str(args.config or "") or None)
    paths = settings.paths
    if str(args.duckdb_path or "").strip():
        paths = replace(paths, duckdb_path=str(args.duckdb_path).strip())

    temp_runtime = _build_duckdb_runtime_settings(
        duckdb_path=Path(paths.duckdb_path),
        temp_directory=str(args.duckdb_temp_directory or "").strip(),
        isolate_run=bool(args.duckdb_temp_isolate_run),
        run_id=str(args.duckdb_temp_run_id or "").strip(),
        memory_limit=str(args.duckdb_memory_limit or "").strip(),
        threads=max(0, int(args.duckdb_threads)),
        max_temp_directory_size=str(args.duckdb_max_temp_directory_size or "").strip(),
    )

    cleanup_pre: dict[str, object] = {"status": "disabled"}
    cleanup_post: dict[str, object] = {"status": "disabled"}
    if bool(args.cleanup_duckdb_temp) and bool(temp_runtime["isolate_run"]):
        cleanup_pre = _safe_cleanup_duckdb_temp_dir(
            Path(str(temp_runtime["temp_directory"])),
            allowed_root=Path(str(temp_runtime["temp_root"])),
        )
        print("[refresh_field_coverage][cleanup-pre]", cleanup_pre, flush=True)
    try:
        out = refresh_field_coverage(
            paths=paths,
            source_view=str(args.source_view or settings.source_view).strip(),
            start_date=str(args.start_date or "").strip(),
            end_date=str(args.end_date or "").strip(),
            fields=tuple(_parse_csv(str(args.fields or ""))),
            run_filters=dict(settings.run_filters),
            update_field_catalog=bool(args.update_field_catalog),
            output_csv_path=str(args.output_csv or "").strip(),
            chunk_size=int(args.chunk_size),
            duckdb_settings=dict(temp_runtime["duckdb_settings"]),
            include_source_tables=tuple(_parse_csv(str(args.include_source_tables or ""))),
            exclude_source_tables=tuple(_parse_csv(str(args.exclude_source_tables or ""))),
            include_heavy_asof=bool(args.include_heavy_asof),
            progress_callback=_print_progress,
        )
    finally:
        if bool(args.cleanup_duckdb_temp) and bool(temp_runtime["isolate_run"]):
            cleanup_post = _safe_cleanup_duckdb_temp_dir(
                Path(str(temp_runtime["temp_directory"])),
                allowed_root=Path(str(temp_runtime["temp_root"])),
            )
            print("[refresh_field_coverage][cleanup-post]", cleanup_post, flush=True)

    print(
        "[refresh_field_coverage]",
        {
            "rows": int(out.get("row_count", 0)),
            "fields": int(out.get("field_count", 0)),
            "skipped_heavy_fields": int(out.get("skipped_heavy_field_count", 0)),
            "include_heavy_asof": bool(out.get("include_heavy_asof", False)),
            "output_csv_path": str(out.get("output_csv_path", "")),
            "updated_field_catalog": bool(out.get("updated_field_catalog", False)),
            "field_catalog_path": str(out.get("field_catalog_path", "")),
            "row_count_source": str(out.get("row_count_source", "")),
            "field_source_counts": dict(out.get("field_source_counts", {})),
            "duckdb_settings": dict(out.get("duckdb_settings", {})),
            "cleanup_pre": cleanup_pre,
            "cleanup_post": cleanup_post,
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh raw field coverage using the closed-loop datasource scope.")
    parser.add_argument("--config", default="", help="Datasource config yaml path")
    parser.add_argument("--duckdb-path", default="", help="Override DuckDB path")
    parser.add_argument("--source-view", default="", help="Override source view")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument(
        "--fields",
        default="",
        help="Comma-separated fields; default is all field catalog fields",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Fields per aggregation chunk. 0 means auto: 16 for light sources, 8 for panel fallback.",
    )
    parser.add_argument("--output-csv", default="artifacts/data_quality/field_coverage.csv")
    parser.add_argument("--update-field-catalog", action="store_true")
    parser.add_argument(
        "--include-source-tables",
        default="",
        help="Comma-separated catalog source_table whitelist",
    )
    parser.add_argument(
        "--exclude-source-tables",
        default="",
        help="Comma-separated catalog source_table blacklist",
    )
    parser.add_argument(
        "--include-heavy-asof",
        action="store_true",
        help="Also compute finance/as-of fields. By default they are marked skipped_heavy_source to avoid expensive as-of joins.",
    )
    parser.add_argument("--duckdb-memory-limit", default="4GB")
    parser.add_argument("--duckdb-threads", type=int, default=2)
    parser.add_argument(
        "--duckdb-temp-directory",
        default="",
        help="DuckDB temp directory; default is <duckdb-path>.tmp on the same drive as the database.",
    )
    parser.add_argument(
        "--duckdb-temp-isolate-run",
        dest="duckdb_temp_isolate_run",
        action="store_true",
        default=True,
        help="Use a per-run subdirectory inside the DuckDB temp directory.",
    )
    parser.add_argument(
        "--no-duckdb-temp-isolate-run",
        dest="duckdb_temp_isolate_run",
        action="store_false",
        help="Write DuckDB temporary files directly into --duckdb-temp-directory.",
    )
    parser.add_argument("--duckdb-temp-run-id", default="", help="Optional temp run directory suffix.")
    parser.add_argument(
        "--cleanup-duckdb-temp",
        dest="cleanup_duckdb_temp",
        action="store_true",
        default=True,
        help="Pre/post clean the isolated DuckDB temp directory.",
    )
    parser.add_argument(
        "--no-cleanup-duckdb-temp",
        dest="cleanup_duckdb_temp",
        action="store_false",
        help="Keep the isolated DuckDB temp directory after the run for debugging.",
    )
    parser.add_argument("--duckdb-max-temp-directory-size", default="12GB")
    return parser


def _parse_csv(raw: str) -> list[str]:
    out: list[str] = []
    for item in str(raw or "").split(","):
        token = item.strip()
        if token and token not in out:
            out.append(token)
    return out


def _print_progress(event: dict[str, object]) -> None:
    print(
        "[refresh_field_coverage][progress]",
        {
            "chunk": f"{event.get('chunk_index')}/{event.get('chunk_count')}",
            "source_view": str(event.get("source_view", "")),
            "fields": list(event.get("fields", []) or []),
        },
        flush=True,
    )


def _build_duckdb_runtime_settings(
    duckdb_path: Path,
    temp_directory: str,
    isolate_run: bool,
    run_id: str,
    memory_limit: str,
    threads: int,
    max_temp_directory_size: str,
) -> dict[str, object]:
    db_path = duckdb_path
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    temp_root = Path(temp_directory) if str(temp_directory or "").strip() else Path(f"{str(db_path)}.tmp")
    if not temp_root.is_absolute():
        temp_root = ROOT / temp_root
    temp_dir = temp_root
    if bool(isolate_run):
        suffix = _safe_run_id(run_id or f"run_field_coverage_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        temp_dir = temp_root / suffix
    return {
        "isolate_run": bool(isolate_run),
        "temp_root": temp_root,
        "temp_directory": temp_dir,
        "duckdb_settings": {
            "memory_limit": str(memory_limit or "").strip(),
            "threads": max(0, int(threads)),
            "temp_directory": str(temp_dir),
            "max_temp_directory_size": str(max_temp_directory_size or "").strip(),
        },
    }


def _safe_run_id(raw: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(raw or "").strip())
    return out or f"run_field_coverage_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _safe_cleanup_duckdb_temp_dir(temp_dir: Path, allowed_root: Path) -> dict[str, object]:
    root = allowed_root.resolve()
    target = temp_dir.resolve()
    if target == root:
        return {
            "status": "skipped_refuse_root",
            "path": str(target),
            "allowed_root": str(root),
        }
    if root not in target.parents:
        return {
            "status": "skipped_outside_allowed_root",
            "path": str(target),
            "allowed_root": str(root),
        }
    if not target.exists():
        return {"status": "missing", "path": str(target), "allowed_root": str(root)}
    last_error = ""
    for attempt in range(1, 6):
        try:
            shutil.rmtree(target)
            return {
                "status": "deleted",
                "path": str(target),
                "allowed_root": str(root),
                "attempts": attempt,
            }
        except Exception as exc:
            last_error = str(exc)
            time.sleep(0.25 * attempt)
    if target.exists() and not any(target.iterdir()):
        try:
            target.rmdir()
            return {
                "status": "deleted_empty_dir",
                "path": str(target),
                "allowed_root": str(root),
            }
        except Exception as exc:
            last_error = str(exc)
    return {
        "status": "failed",
        "path": str(target),
        "allowed_root": str(root),
        "error": last_error,
    }


if __name__ == "__main__":
    main()
