from __future__ import annotations

from pathlib import Path
import importlib.util

import duckdb  # type: ignore
import pandas as pd

from alpha_mining.datasource.config import LakePathSettings
from alpha_mining.datasource.field_coverage import refresh_field_coverage


def test_refresh_field_coverage_applies_closed_loop_filters_and_missing_fields(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "market.duckdb"
    meta_dir = tmp_path / "lake" / "meta"
    meta_dir.mkdir(parents=True)
    field_catalog_path = meta_dir / "field_catalog.parquet"
    pd.DataFrame(
        {
            "field_name": ["close", "moneyflow_net_amount", "missing_raw"],
            "field_type": ["SCALAR", "SCALAR", "SCALAR"],
            "category": ["price", "moneyflow", "moneyflow"],
            "source_table": [
                "v_project_panel_cn_a",
                "v_project_panel_cn_a",
                "v_project_panel_cn_a",
            ],
            "dtype": ["DOUBLE", "DOUBLE", "DOUBLE"],
            "unit": ["", "", ""],
            "available_start": ["", "", ""],
            "available_end": ["", "", ""],
            "is_default_enabled": [True, False, False],
            "is_searchable": [True, True, True],
            "description": ["close", "moneyflow", "missing"],
            "factor_family": ["price_volume", "moneyflow", "moneyflow"],
            "field_catalog_version": ["v_test", "v_test", "v_test"],
        }
    ).to_parquet(field_catalog_path, index=False)

    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE panel AS
            SELECT * FROM (
                VALUES
                    (DATE '2024-01-02', '000001.SZ', 1, 10.0, 1.0),
                    (DATE '2024-01-03', '000001.SZ', 1, 11.0, NULL),
                    (DATE '2024-01-04', '000001.SZ', 0, 12.0, 3.0),
                    (DATE '2024-01-03', '830001.BJ', 1, 13.0, 4.0)
            ) AS t(date, code, universe, close, moneyflow_net_amount)
            """
        )
        conn.execute("CREATE VIEW v_project_panel_cn_a AS SELECT * FROM panel")
        catalog_sql_path = field_catalog_path.as_posix().replace("'", "''")
        conn.execute(f"CREATE VIEW v_project_field_catalog AS SELECT * FROM read_parquet('{catalog_sql_path}')")
    finally:
        conn.close()

    paths = LakePathSettings(
        lake_root=str((tmp_path / "lake").as_posix()),
        duckdb_path=str(db_path.as_posix()),
    )
    out = refresh_field_coverage(
        paths=paths,
        source_view="v_project_panel_cn_a",
        start_date="2024-01-02",
        end_date="2024-01-03",
        fields=("close", "moneyflow_net_amount", "missing_raw"),
        run_filters={"universe_only": True, "include_bj": False},
        update_field_catalog=True,
        output_csv_path=tmp_path / "coverage.csv",
        chunk_size=2,
    )

    coverage = out["coverage_df"]
    by_field = {str(row["field_name"]): row for _, row in coverage.iterrows()}
    assert int(out["row_count"]) == 2
    assert by_field["close"]["coverage_rate"] == 1.0
    assert by_field["moneyflow_net_amount"]["coverage_rate"] == 0.5
    assert by_field["moneyflow_net_amount"]["missing_rate"] == 0.5
    assert by_field["missing_raw"]["coverage_rate"] == 0.0
    assert str(by_field["missing_raw"]["coverage_status"]) == "missing_field"
    assert (tmp_path / "coverage.csv").exists()

    refreshed = pd.read_parquet(field_catalog_path)
    assert "coverage_rate" in refreshed.columns
    assert float(refreshed.loc[refreshed["field_name"] == "moneyflow_net_amount", "coverage_rate"].iloc[0]) == 0.5

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        catalog_cols = conn.execute("DESCRIBE v_project_field_catalog").fetchdf()["column_name"].astype(str).tolist()
        assert "coverage_rate" in catalog_cols
    finally:
        conn.close()


def test_refresh_field_coverage_uses_lightweight_source_when_field_is_in_hot_base(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "market.duckdb"
    meta_dir = tmp_path / "lake" / "meta"
    meta_dir.mkdir(parents=True)
    field_catalog_path = meta_dir / "field_catalog.parquet"
    pd.DataFrame(
        {
            "field_name": ["close", "fin_roe"],
            "field_type": ["SCALAR", "SCALAR"],
            "category": ["price", "finance"],
            "source_table": [
                "v_project_market_daily_base",
                "v_project_financial_asof_daily",
            ],
            "dtype": ["DOUBLE", "DOUBLE"],
            "unit": ["", ""],
            "available_start": ["", ""],
            "available_end": ["", ""],
            "is_default_enabled": [True, False],
            "is_searchable": [True, True],
            "description": ["close", "roe"],
            "factor_family": ["price_volume", "fundamental"],
            "field_catalog_version": ["v_test", "v_test"],
        }
    ).to_parquet(field_catalog_path, index=False)

    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE project_market_daily_base AS
            SELECT * FROM (
                VALUES
                    (DATE '2024-01-02', '000001.SZ', 1, 10.0),
                    (DATE '2024-01-03', '000001.SZ', 1, NULL)
            ) AS t(date, code, universe, close)
            """
        )
        conn.execute("CREATE VIEW v_project_market_daily_base_hot AS SELECT * FROM project_market_daily_base")
        conn.execute(
            "CREATE VIEW v_project_panel_cn_a AS SELECT date, code, universe, close, CAST(NULL AS DOUBLE) AS fin_roe FROM project_market_daily_base"
        )
        catalog_sql_path = field_catalog_path.as_posix().replace("'", "''")
        conn.execute(f"CREATE VIEW v_project_field_catalog AS SELECT * FROM read_parquet('{catalog_sql_path}')")
    finally:
        conn.close()

    paths = LakePathSettings(
        lake_root=str((tmp_path / "lake").as_posix()),
        duckdb_path=str(db_path.as_posix()),
    )
    out = refresh_field_coverage(
        paths=paths,
        source_view="v_project_panel_cn_a",
        fields=("close",),
        run_filters={"universe_only": True},
    )

    assert out["field_source_counts"] == {"v_project_market_daily_base_hot": 1}
    assert out["coverage_df"].iloc[0]["coverage_rate"] == 0.5


def test_refresh_field_coverage_groups_fields_by_resolved_source_before_chunking(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "market.duckdb"
    meta_dir = tmp_path / "lake" / "meta"
    meta_dir.mkdir(parents=True)
    field_catalog_path = meta_dir / "field_catalog.parquet"
    pd.DataFrame(
        {
            "field_name": ["close", "tech_asi_qfq", "volume"],
            "source_table": [
                "v_project_market_daily_base",
                "fact_stk_factor_pro",
                "v_project_market_daily_base",
            ],
            "dtype": ["DOUBLE", "DOUBLE", "DOUBLE"],
        }
    ).to_parquet(field_catalog_path, index=False)

    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE panel AS
            SELECT * FROM (
                VALUES
                    (DATE '2024-01-02', '000001.SZ', 1, 10.0, 100.0, 2.0),
                    (DATE '2024-01-03', '000001.SZ', 1, 11.0, NULL, NULL)
            ) AS t(date, code, universe, close, volume, tech_asi_qfq)
            """
        )
        conn.execute("CREATE TABLE project_market_daily_base AS SELECT date, code, universe, close, volume FROM panel")
        conn.execute(
            "CREATE VIEW v_project_market_daily_base_hot AS SELECT date, code, universe, close, volume FROM project_market_daily_base"
        )
        conn.execute("CREATE VIEW v_project_panel_cn_a AS SELECT * FROM panel")
        catalog_sql_path = field_catalog_path.as_posix().replace("'", "''")
        conn.execute(f"CREATE VIEW v_project_field_catalog AS SELECT * FROM read_parquet('{catalog_sql_path}')")
    finally:
        conn.close()

    progress: list[dict[str, object]] = []
    paths = LakePathSettings(
        lake_root=str((tmp_path / "lake").as_posix()),
        duckdb_path=str(db_path.as_posix()),
    )
    out = refresh_field_coverage(
        paths=paths,
        source_view="v_project_panel_cn_a",
        fields=("close", "tech_asi_qfq", "volume"),
        run_filters={"universe_only": True},
        chunk_size=8,
        progress_callback=lambda event: progress.append(dict(event)),
    )

    assert out["field_source_counts"] == {
        "v_project_market_daily_base_hot": 2,
        "v_project_panel_cn_a": 1,
    }
    assert [event["source_view"] for event in progress] == [
        "v_project_market_daily_base_hot",
        "v_project_panel_cn_a",
    ]
    assert progress[0]["fields"] == ["close", "volume"]
    by_field = {str(row["field_name"]): row for _, row in out["coverage_df"].iterrows()}
    assert by_field["volume"]["coverage_rate"] == 0.5
    assert by_field["tech_asi_qfq"]["coverage_rate"] == 0.5


def test_refresh_field_coverage_skips_heavy_asof_by_default_and_allows_opt_in(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "market.duckdb"
    meta_dir = tmp_path / "lake" / "meta"
    meta_dir.mkdir(parents=True)
    field_catalog_path = meta_dir / "field_catalog.parquet"
    pd.DataFrame(
        {
            "field_name": ["close", "fin_roe"],
            "source_table": [
                "v_project_market_daily_base",
                "v_project_financial_asof_daily",
            ],
            "dtype": ["DOUBLE", "DOUBLE"],
        }
    ).to_parquet(field_catalog_path, index=False)

    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE project_market_daily_base AS
            SELECT * FROM (
                VALUES
                    (DATE '2024-01-02', '000001.SZ', 1, 10.0, 1.0),
                    (DATE '2024-01-03', '000001.SZ', 1, 11.0, NULL)
            ) AS t(date, code, universe, close, fin_roe)
            """
        )
        conn.execute(
            "CREATE VIEW v_project_market_daily_base_hot AS SELECT date, code, universe, close FROM project_market_daily_base"
        )
        conn.execute("CREATE VIEW v_project_panel_cn_a AS SELECT * FROM project_market_daily_base")
        catalog_sql_path = field_catalog_path.as_posix().replace("'", "''")
        conn.execute(f"CREATE VIEW v_project_field_catalog AS SELECT * FROM read_parquet('{catalog_sql_path}')")
    finally:
        conn.close()

    paths = LakePathSettings(
        lake_root=str((tmp_path / "lake").as_posix()),
        duckdb_path=str(db_path.as_posix()),
    )
    skipped = refresh_field_coverage(
        paths=paths,
        source_view="v_project_panel_cn_a",
        fields=("close", "fin_roe"),
        run_filters={"universe_only": True},
        output_csv_path=tmp_path / "coverage_default.csv",
    )["coverage_df"]
    by_field = {str(row["field_name"]): row for _, row in skipped.iterrows()}

    assert by_field["close"]["coverage_status"] == "ok"
    assert by_field["fin_roe"]["coverage_status"] == "skipped_heavy_source"
    assert pd.isna(by_field["fin_roe"]["coverage_rate"])

    included = refresh_field_coverage(
        paths=paths,
        source_view="v_project_panel_cn_a",
        fields=("fin_roe",),
        run_filters={"universe_only": True},
        include_heavy_asof=True,
    )["coverage_df"]
    assert included.iloc[0]["coverage_status"] == "ok"
    assert included.iloc[0]["coverage_rate"] == 0.5


def test_refresh_field_coverage_writes_incremental_csv_before_progress_callback(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "market.duckdb"
    meta_dir = tmp_path / "lake" / "meta"
    meta_dir.mkdir(parents=True)
    field_catalog_path = meta_dir / "field_catalog.parquet"
    pd.DataFrame({"field_name": ["close", "volume"]}).to_parquet(field_catalog_path, index=False)

    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE v_project_panel_cn_a AS
            SELECT * FROM (
                VALUES
                    (DATE '2024-01-02', '000001.SZ', 1, 10.0, 100.0),
                    (DATE '2024-01-03', '000001.SZ', 1, 11.0, NULL)
            ) AS t(date, code, universe, close, volume)
            """
        )
        catalog_sql_path = field_catalog_path.as_posix().replace("'", "''")
        conn.execute(f"CREATE VIEW v_project_field_catalog AS SELECT * FROM read_parquet('{catalog_sql_path}')")
    finally:
        conn.close()

    output_csv = tmp_path / "incremental.csv"
    seen_mid_run: list[list[str]] = []

    def _progress(_: dict[str, object]) -> None:
        seen_mid_run.append(pd.read_csv(output_csv)["field_name"].astype(str).tolist())

    paths = LakePathSettings(
        lake_root=str((tmp_path / "lake").as_posix()),
        duckdb_path=str(db_path.as_posix()),
    )
    refresh_field_coverage(
        paths=paths,
        source_view="v_project_panel_cn_a",
        fields=("close", "volume"),
        run_filters={"universe_only": True},
        output_csv_path=output_csv,
        chunk_size=1,
        progress_callback=_progress,
    )

    assert seen_mid_run[0] == ["close"]
    assert pd.read_csv(output_csv)["field_name"].astype(str).tolist() == [
        "close",
        "volume",
    ]


def test_refresh_field_coverage_accepts_duckdb_runtime_settings(tmp_path: Path) -> None:
    db_path = tmp_path / "market.duckdb"
    temp_dir = tmp_path / "duckdb_tmp"
    meta_dir = tmp_path / "lake" / "meta"
    meta_dir.mkdir(parents=True)
    field_catalog_path = meta_dir / "field_catalog.parquet"
    pd.DataFrame({"field_name": ["close"]}).to_parquet(field_catalog_path, index=False)

    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(
            "CREATE TABLE v_project_panel_cn_a AS SELECT DATE '2024-01-02' AS date, '000001.SZ' AS code, 1 AS universe, 10.0 AS close"
        )
        catalog_sql_path = field_catalog_path.as_posix().replace("'", "''")
        conn.execute(f"CREATE VIEW v_project_field_catalog AS SELECT * FROM read_parquet('{catalog_sql_path}')")
    finally:
        conn.close()

    paths = LakePathSettings(
        lake_root=str((tmp_path / "lake").as_posix()),
        duckdb_path=str(db_path.as_posix()),
    )
    out = refresh_field_coverage(
        paths=paths,
        source_view="v_project_panel_cn_a",
        fields=("close",),
        run_filters={"universe_only": True},
        duckdb_settings={
            "temp_directory": str(temp_dir.as_posix()),
            "memory_limit": "512MB",
            "threads": 1,
        },
    )

    assert out["duckdb_settings"]["temp_directory"] == str(temp_dir.as_posix())


def test_refresh_field_coverage_cli_safe_cleanup_removes_isolated_temp_only(
    tmp_path: Path,
) -> None:
    module = _load_refresh_field_coverage_script()
    root = tmp_path / "market.duckdb.tmp"
    isolated = root / "run_field_coverage_test"
    isolated.mkdir(parents=True)
    (isolated / "spill.tmp").write_text("x", encoding="utf-8")

    result = module._safe_cleanup_duckdb_temp_dir(isolated, allowed_root=root)

    assert result["status"] == "deleted"
    assert not isolated.exists()
    assert root.exists()


def test_refresh_field_coverage_cli_default_chunk_size_is_auto() -> None:
    module = _load_refresh_field_coverage_script()
    parser = module.build_arg_parser()
    args = parser.parse_args([])
    assert int(args.chunk_size) == 0


def _load_refresh_field_coverage_script():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "refresh_field_coverage.py"
    spec = importlib.util.spec_from_file_location("refresh_field_coverage_script", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
