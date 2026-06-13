from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_mining.datasource import (  # noqa: E402
    build_snapshot_run_id,
    load_datasource_settings,
    load_panel_from_duckdb,
    materialize_input_snapshot,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize frozen input snapshot for reproduce")
    parser.add_argument("--source-backend", default="duckdb", choices=["duckdb", "file"])
    parser.add_argument("--config", default="", help="Datasource config yaml path")
    parser.add_argument("--duckdb-path", default="")
    parser.add_argument("--source-view", default="")
    parser.add_argument("--data-path", default="", help="Input file path for --source-backend=file")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--required-fields", default="", help="Comma-separated extra required fields")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--code-col", default="code")
    parser.add_argument("--base-fields", default="pct_chg,circ_mv")
    parser.add_argument("--group-fields", default="industry,sector")
    parser.add_argument("--run-filters-json", default="", help="JSON object for source filters")
    parser.add_argument("--snapshot-root", default="")
    parser.add_argument("--universe", default="cn_all")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    settings = load_datasource_settings(str(args.config or "") or None)
    source_backend = str(args.source_backend or "duckdb").strip().lower()
    run_filters = _parse_json_obj(str(args.run_filters_json or ""))

    if source_backend == "duckdb":
        duckdb_path = str(args.duckdb_path or settings.paths.duckdb_path).strip()
        source_view = str(args.source_view or settings.source_view).strip() or "v_project_panel_cn_a"
        if not duckdb_path:
            raise ValueError("duckdb backend requires --duckdb-path or PROJECT_DUCKDB_PATH")

        required_fields = _csv_fields(args.required_fields)
        base_fields = tuple(_csv_fields(args.base_fields))
        group_fields = tuple(_csv_fields(args.group_fields))
        raw_df = load_panel_from_duckdb(
            duckdb_path=duckdb_path,
            source_view=source_view,
            required_fields=required_fields,
            start_date=str(args.start_date or "") or None,
            end_date=str(args.end_date or "") or None,
            date_col=str(args.date_col),
            code_col=str(args.code_col),
            base_fields=base_fields,
            group_fields=group_fields,
            run_filters=run_filters,
        )
        source_meta = {
            "source_backend": "duckdb",
            "duckdb_path": str(Path(duckdb_path).as_posix()),
            "source_view": str(source_view),
            "start_date": str(args.start_date or ""),
            "end_date": str(args.end_date or ""),
            "required_fields": required_fields,
            "base_fields": list(base_fields),
            "group_fields": list(group_fields),
            "run_filters": run_filters,
        }
    else:
        data_path = Path(str(args.data_path or "").strip())
        if not data_path.exists():
            raise FileNotFoundError(data_path)
        raw_df = _load_dataframe(path=data_path)
        source_meta = {
            "source_backend": "file",
            "source_path": str(data_path.as_posix()),
        }

    if raw_df.empty:
        raise ValueError("input frame is empty, snapshot aborted")

    snapshot_root = str(args.snapshot_root).strip() or str(settings.paths.snapshots_path.as_posix())
    run_id = str(args.run_id or "").strip()
    if not run_id:
        run_id = build_snapshot_run_id(
            universe_name=str(args.universe),
            source_view=str(source_meta.get("source_view", source_backend)),
            start_date=str(args.start_date or "") or None,
            end_date=str(args.end_date or "") or None,
            fields=[str(x) for x in raw_df.columns],
        )

    out = materialize_input_snapshot(
        raw_df=raw_df,
        snapshot_root=snapshot_root,
        universe_name=str(args.universe),
        run_id=run_id,
        metadata=source_meta,
    )
    print("[materialize_input_snapshot] done")
    print(out)


def _csv_fields(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    out: list[str] = []
    for item in text.split(","):
        token = str(item or "").strip()
        if token and token not in out:
            out.append(token)
    return out


def _parse_json_obj(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError("--run-filters-json must be a JSON object")
    return loaded


def _load_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".pkl":
        with path.open("rb") as f:
            return pickle.load(f)
    raise ValueError(f"Unsupported data file extension: {path}")


if __name__ == "__main__":
    main()
