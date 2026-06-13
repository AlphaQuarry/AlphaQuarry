from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_mining.datasource import load_datasource_settings, load_panel_from_duckdb
from alpha_mining.datasource.quality import (
    build_panel_quality_report,
    write_quality_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check panel data quality and write JSON/Markdown reports.")
    parser.add_argument("--source-backend", default="duckdb", choices=["duckdb", "file"])
    parser.add_argument("--data-path", default="", help="File backend input path: parquet/csv/pkl")
    parser.add_argument("--datasource-config", default="")
    parser.add_argument("--duckdb-path", default="")
    parser.add_argument("--source-view", default="")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--fields", default="", help="Comma-separated fields to check")
    parser.add_argument("--required-fields", default="", help="Comma-separated required fields")
    parser.add_argument("--json-out", default="artifacts/data_quality/panel_quality.json")
    parser.add_argument("--markdown-out", default="artifacts/data_quality/panel_quality.md")
    parser.add_argument("--max-missing-ratio", type=float, default=0.30)
    parser.add_argument("--max-inf-ratio", type=float, default=0.0)
    parser.add_argument("--fail-on-warn", action="store_true")
    args = parser.parse_args()

    raise SystemExit(
        run_quality_check(
            source_backend=str(args.source_backend),
            data_path=str(args.data_path),
            datasource_config=str(args.datasource_config),
            duckdb_path=str(args.duckdb_path),
            source_view=str(args.source_view),
            start_date=str(args.start_date),
            end_date=str(args.end_date),
            fields=str(args.fields),
            required_fields=str(args.required_fields),
            json_out=str(args.json_out),
            markdown_out=str(args.markdown_out),
            max_missing_ratio=float(args.max_missing_ratio),
            max_inf_ratio=float(args.max_inf_ratio),
            fail_on_warn=bool(args.fail_on_warn),
        )
    )


def run_quality_check(
    source_backend: str,
    data_path: str,
    datasource_config: str,
    duckdb_path: str,
    source_view: str,
    start_date: str,
    end_date: str,
    fields: str,
    required_fields: str,
    json_out: str,
    markdown_out: str,
    max_missing_ratio: float,
    max_inf_ratio: float,
    fail_on_warn: bool = False,
) -> int:
    settings = load_datasource_settings(str(datasource_config or "") or None)
    expected = _parse_csv(fields)
    required = _parse_csv(required_fields)
    backend = str(source_backend or settings.source_backend).strip().lower()
    if backend == "file":
        if not str(data_path or "").strip():
            raise ValueError("--data-path is required for file backend")
        df = _load_dataframe(Path(data_path))
    else:
        path = str(duckdb_path or settings.paths.duckdb_path)
        view = str(source_view or settings.source_view)
        required_fields_for_load = expected or required
        df = load_panel_from_duckdb(
            duckdb_path=path,
            source_view=view,
            required_fields=required_fields_for_load,
            start_date=str(start_date or "") or None,
            end_date=str(end_date or "") or None,
            run_filters=settings.run_filters,
        )

    report = build_panel_quality_report(
        df,
        expected_fields=expected,
        required_fields=required,
        max_missing_ratio=float(max_missing_ratio),
        max_inf_ratio=float(max_inf_ratio),
    )
    written = write_quality_artifacts(report=report, json_out=json_out, markdown_out=markdown_out)
    print("[check_panel_quality]", {"overall_status": report["overall_status"], **written})
    if report["overall_status"] == "fail":
        return 2
    if bool(fail_on_warn) and report["overall_status"] == "warn":
        return 1
    return 0


def _load_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported data file extension: {path.suffix}")


def _parse_csv(raw: str) -> list[str]:
    out: list[str] = []
    for item in str(raw or "").split(","):
        token = item.strip()
        if token and token not in out:
            out.append(token)
    return out


if __name__ == "__main__":
    main()
