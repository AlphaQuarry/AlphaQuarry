from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_mining.datasource import build_duckdb_catalog, load_datasource_settings
from alpha_mining.datasource.duckdb_runtime import build_duckdb_runtime_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DuckDB catalog/views over Parquet lake")
    parser.add_argument("--config", default="", help="Datasource config yaml path")
    parser.add_argument("--source-view", default="")
    parser.add_argument("--field-catalog-version", default="")
    parser.add_argument("--no-materialize-project-base", action="store_true")
    parser.add_argument("--duckdb-memory-limit", default="4GB", help="DuckDB memory_limit, e.g. 4GB")
    parser.add_argument("--duckdb-threads", type=int, default=2)
    parser.add_argument(
        "--duckdb-temp-directory",
        default="",
        help="DuckDB temp directory; default is <duckdb-path>.tmp on the same drive as the database.",
    )
    parser.add_argument("--duckdb-temp-run-id", default="")
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
    parser.add_argument("--duckdb-max-temp-directory-size", default="12GB")
    args = parser.parse_args()

    settings = load_datasource_settings(str(args.config or "") or None)
    source_view = str(args.source_view).strip() or settings.source_view
    field_catalog_version = str(args.field_catalog_version).strip() or settings.field_catalog_version
    duckdb_runtime = build_duckdb_runtime_settings(
        duckdb_path=settings.paths.duckdb_path,
        temp_directory=str(args.duckdb_temp_directory or "").strip(),
        isolate_run=bool(args.duckdb_temp_isolate_run),
        run_id=str(args.duckdb_temp_run_id or "").strip(),
        run_prefix="run_build_duckdb_catalog",
        memory_limit=str(args.duckdb_memory_limit or "").strip(),
        threads=max(0, int(args.duckdb_threads)),
        max_temp_directory_size=str(args.duckdb_max_temp_directory_size or "").strip(),
    )

    out = build_duckdb_catalog(
        paths=settings.paths,
        source_view=source_view,
        field_catalog_version=field_catalog_version,
        adjust_mode=settings.adjust_mode,
        universe_min_days_since_listed=int(settings.universe_min_days_since_listed),
        universe_exclude_st=bool(settings.universe_exclude_st),
        include_bj=bool(settings.include_bj),
        tradable_require_close=bool(settings.tradable_require_close),
        tradable_require_positive_volume=bool(settings.tradable_require_positive_volume),
        tradable_require_positive_amount=bool(settings.tradable_require_positive_amount),
        materialize_project_base=not bool(args.no_materialize_project_base),
        duckdb_settings=duckdb_runtime,
        field_catalog_enabled_categories=tuple(settings.field_catalog_enabled_categories),
        field_catalog_non_searchable_fields=tuple(settings.field_catalog_non_searchable_fields),
    )
    print("[build_duckdb_catalog] done")
    print(out)


if __name__ == "__main__":
    main()
