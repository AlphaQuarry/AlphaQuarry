from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_mining.datasource import (
    load_datasource_settings,
    refresh_duckdb_field_catalog,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh field catalog from DuckDB source view")
    parser.add_argument("--config", default="", help="Datasource config yaml path")
    parser.add_argument("--source-view", default="")
    parser.add_argument("--field-catalog-version", default="")
    args = parser.parse_args()

    settings = load_datasource_settings(str(args.config or "") or None)
    source_view = str(args.source_view).strip() or settings.source_view
    field_catalog_version = str(args.field_catalog_version).strip() or settings.field_catalog_version

    out = refresh_duckdb_field_catalog(
        paths=settings.paths,
        source_view=source_view,
        field_catalog_version=field_catalog_version,
        field_catalog_enabled_categories=tuple(settings.field_catalog_enabled_categories),
        field_catalog_non_searchable_fields=tuple(settings.field_catalog_non_searchable_fields),
    )
    print("[refresh_field_catalog] done")
    print(out)


if __name__ == "__main__":
    main()
