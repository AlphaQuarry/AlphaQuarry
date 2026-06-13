from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from alpha_mining.workflow.artifacts import save_dataframe_artifact
from alpha_mining.workflow.analysis_cycle import (
    AnalysisLevelConfig,
    BatchAnalysisConfig,
)
from alpha_mining.workflow.superalpha import (
    SuperalphaBusyError,
    SuperalphaConfig,
    SuperalphaError,
    build_superalpha_signal,
    find_existing_superalpha_run,
    list_superalpha_components,
    parse_combo_expression,
    run_superalpha_backtest,
    _component_cache_key,
    _prune_component_cache,
    _check_system_drive_free_space,
    _disk_snapshot,
    _process_memory_snapshot,
    _safe_dir_size,
    _resolve_component_signal,
    _superalpha_run_lock,
    _superalpha_runtime_dirs,
)
from alpha_mining.workflow.universe_store import save_universe_base_frame


def test_parse_combo_expression_equal_weight() -> None:
    components = [
        {"factor": "alpha_a", "score": 60.0},
        {"factor": "alpha_b", "score": 70.0},
        {"factor": "alpha_c", "score": 80.0},
    ]

    result = parse_combo_expression("1", components)

    assert result.basis == "equal_weight"
    assert result.weights == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_parse_combo_expression_fixed_weights_and_normalizes_abs_sum() -> None:
    components = [{"factor": "a"}, {"factor": "b"}, {"factor": "c"}]

    result = parse_combo_expression("[0.4, 0.3, -0.3]", components)

    assert result.basis == "fixed"
    assert result.weights == pytest.approx([0.4, 0.3, -0.3])


def test_parse_combo_expression_rejects_bad_weights() -> None:
    components = [{"factor": "a"}, {"factor": "b"}]

    with pytest.raises(SuperalphaError, match="length"):
        parse_combo_expression("0.4,0.3,0.3", components)

    with pytest.raises(SuperalphaError, match="all zero"):
        parse_combo_expression("[0, 0]", components)

    with pytest.raises(SuperalphaError, match="unsupported"):
        parse_combo_expression("__import__('os').system('echo bad')", components)


def test_parse_combo_expression_metadata_weight() -> None:
    components = [
        {"factor": "alpha_a", "score": 60.0},
        {"factor": "alpha_b", "score": 30.0},
    ]

    result = parse_combo_expression("score", components)

    assert result.basis == "score"
    assert result.weights == pytest.approx([2 / 3, 1 / 3])


def test_list_superalpha_components_returns_only_accepted_for_universe(
    tmp_path: Path,
) -> None:
    registry_dir = tmp_path / "u1" / "library"
    registry_dir.mkdir(parents=True)
    signal_path = save_dataframe_artifact(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01"]),
                "code": ["000001.SZ"],
                "alpha_a": [1.0],
            }
        ),
        tmp_path / "series" / "alpha_a_signal",
        preferred="parquet",
    )["path"]
    pd.DataFrame(
        {
            "universe": ["u1", "u1", "u2"],
            "factor": ["alpha_a", "alpha_b", "alpha_other"],
            "status": ["accepted", "staging", "accepted"],
            "score": [70.0, 55.0, 90.0],
            "signal_artifact_path": [signal_path, "b.parquet", "other.parquet"],
        }
    ).to_csv(registry_dir / "factor_library_registry.csv", index=False)

    rows = list_superalpha_components(base_dir=tmp_path, universe_name="u1")

    assert [row["factor"] for row in rows] == ["alpha_a"]
    assert rows[0]["score"] == 70.0
    assert rows[0]["signal_available"] is True


def test_list_superalpha_components_marks_missing_signal_unavailable(
    tmp_path: Path,
) -> None:
    registry_dir = tmp_path / "u1" / "library"
    registry_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "universe": ["u1"],
            "factor": ["alpha_a"],
            "status": ["accepted"],
            "score": [70.0],
            "signal_artifact_path": [""],
        }
    ).to_csv(registry_dir / "factor_library_registry.csv", index=False)

    rows = list_superalpha_components(base_dir=tmp_path, universe_name="u1")

    assert rows[0]["signal_available"] is False
    assert rows[0]["signal_status_reason"] == "missing_signal_artifact"


def test_build_superalpha_signal_reads_selected_compact_signals(tmp_path: Path) -> None:
    registry_dir = tmp_path / "u1" / "library"
    registry_dir.mkdir(parents=True)
    dates = pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"])
    codes = ["000001.SZ", "000002.SZ"] * 2
    sig_a = pd.DataFrame({"date": dates, "code": codes, "alpha_a": [1.0, 3.0, 2.0, 4.0]})
    sig_b = pd.DataFrame({"date": dates, "code": codes, "alpha_b": [10.0, 20.0, 4.0, 8.0]})
    path_a = save_dataframe_artifact(sig_a, tmp_path / "series" / "alpha_a_signal", preferred="parquet")["path"]
    path_b = save_dataframe_artifact(sig_b, tmp_path / "series" / "alpha_b_signal", preferred="parquet")["path"]
    pd.DataFrame(
        {
            "universe": ["u1", "u1"],
            "factor": ["alpha_a", "alpha_b"],
            "status": ["accepted", "accepted"],
            "score": [60.0, 80.0],
            "signal_artifact_path": [path_a, path_b],
        }
    ).to_csv(registry_dir / "factor_library_registry.csv", index=False)

    result = build_superalpha_signal(
        base_dir=tmp_path,
        universe_name="u1",
        selected_factor_ids=["alpha_a", "alpha_b"],
        combo_expression="1",
    )

    signal = result.signal.sort_values(["date", "code"]).reset_index(drop=True)
    assert list(signal.columns) == ["date", "code", "superalpha"]
    assert len(signal) == 4
    for _, part in signal.groupby("date"):
        assert float(part["superalpha"].mean()) == pytest.approx(0.0)


def test_build_superalpha_signal_missing_signal_artifact_is_clear(
    tmp_path: Path,
) -> None:
    registry_dir = tmp_path / "u1" / "library"
    registry_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "universe": ["u1"],
            "factor": ["alpha_missing"],
            "status": ["accepted"],
            "signal_artifact_path": [str(tmp_path / "missing.parquet")],
        }
    ).to_csv(registry_dir / "factor_library_registry.csv", index=False)

    with pytest.raises(SuperalphaError, match="unable to resolve signal"):
        build_superalpha_signal(
            base_dir=tmp_path,
            universe_name="u1",
            selected_factor_ids=["alpha_missing"],
            combo_expression="1",
        )


def test_build_superalpha_signal_limits_component_count(tmp_path: Path) -> None:
    with pytest.raises(SuperalphaError, match="at most 1"):
        build_superalpha_signal(
            base_dir=tmp_path,
            universe_name="u1",
            selected_factor_ids=["alpha_a", "alpha_b"],
            combo_expression="1",
            config=SuperalphaConfig(max_components=1),
        )


def test_run_superalpha_backtest_writes_independent_artifacts(tmp_path: Path) -> None:
    dates = pd.bdate_range("2024-01-02", periods=8)
    codes = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
    base = pd.DataFrame(
        [
            {
                "date": date,
                "code": code,
                "pct_chg": 0.001 * (idx + 1) * (code_rank + 1),
                "circ_mv": 1000.0 + idx * 10,
                "can_buy": True,
                "can_sell": True,
            }
            for idx, date in enumerate(dates)
            for code_rank, code in enumerate(codes)
        ]
    )
    save_universe_base_frame(base, base_dir=tmp_path, universe_name="u1")
    signal_dir = tmp_path / "signals"
    rows = []
    for factor, power in [("alpha_a", 1), ("alpha_b", 2)]:
        signal = pd.DataFrame(
            [
                {
                    "date": date,
                    "code": code,
                    factor: (idx + 1) * float((rank + 1) ** power),
                }
                for idx, date in enumerate(dates)
                for rank, code in enumerate(codes)
            ]
        )
        saved = save_dataframe_artifact(signal, signal_dir / f"{factor}_signal", preferred="parquet")["path"]
        rows.append(
            {
                "universe": "u1",
                "factor": factor,
                "status": "accepted",
                "score": 70.0,
                "signal_artifact_path": saved,
            }
        )
    registry_dir = tmp_path / "u1" / "library"
    registry_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(registry_dir / "factor_library_registry.csv", index=False)

    result = run_superalpha_backtest(
        base_dir=tmp_path,
        universe_name="u1",
        selected_factor_ids=["alpha_a", "alpha_b"],
        combo_expression="1",
        analysis_config=BatchAnalysisConfig(
            layers=2,
            benchmark_enabled=False,
            include_full_ic_lag_analysis=False,
            analysis_level=AnalysisLevelConfig(mode="light"),
        ),
    )

    run_dir = tmp_path / "u1" / "superalphas" / result["superalpha_id"]
    assert result["status"] == "ok"
    assert (run_dir / "meta.json").exists()
    assert (run_dir / "analysis_meta.json").exists()
    assert (run_dir / "superalpha_values.parquet").exists()
    assert (run_dir / "portfolio_pnl_df.parquet").exists()
    meta = pd.read_json(run_dir / "meta.json", typ="series")
    assert meta["component_count"] == 2
    assert meta["schema_version"] == 2


def test_list_superalpha_components_returns_new_fields(tmp_path: Path) -> None:
    """Test that list_superalpha_components returns new signal status fields."""
    registry_dir = tmp_path / "u1" / "library"
    registry_dir.mkdir(parents=True)
    signal_path = save_dataframe_artifact(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01"]),
                "code": ["000001.SZ"],
                "alpha_a": [1.0],
            }
        ),
        tmp_path / "series" / "alpha_a_signal",
        preferred="parquet",
    )["path"]
    pd.DataFrame(
        {
            "universe": ["u1"],
            "factor": ["alpha_a"],
            "status": ["accepted"],
            "score": [70.0],
            "signal_artifact_path": [signal_path],
        }
    ).to_csv(registry_dir / "factor_library_registry.csv", index=False)

    rows = list_superalpha_components(base_dir=tmp_path, universe_name="u1")

    assert len(rows) == 1
    row = rows[0]
    assert row["signal_status"] == "compact"
    assert row["signal_available"] is True
    assert row["can_backtest"] is True
    assert row["can_reproduce"] is False
    assert row["direction_sign"] == 1
    assert row["direction_status"] in (
        "registry",
        "missing_default_positive",
        "source_run",
    )


def test_list_superalpha_components_marks_unavailable(tmp_path: Path) -> None:
    """Test that factors without signal are marked unavailable."""
    registry_dir = tmp_path / "u1" / "library"
    registry_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "universe": ["u1"],
            "factor": ["alpha_missing"],
            "status": ["accepted"],
            "score": [70.0],
            "signal_artifact_path": [""],
        }
    ).to_csv(registry_dir / "factor_library_registry.csv", index=False)

    rows = list_superalpha_components(base_dir=tmp_path, universe_name="u1")

    assert len(rows) == 1
    row = rows[0]
    assert row["signal_status"] == "unavailable"
    assert row["signal_available"] is False
    assert row["can_backtest"] is False
    assert row["can_reproduce"] is False  # no expression registry


def test_inspect_signal_status_checks_expression_registry(tmp_path: Path) -> None:
    """Test that can_reproduce is based on expression registry, not just factor name."""
    registry_dir = tmp_path / "u1" / "library"
    registry_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "universe": ["u1", "u1"],
            "factor": ["alpha00001", "alpha00002"],
            "status": ["accepted", "accepted"],
            "score": [70.0, 60.0],
            "signal_artifact_path": ["", ""],
        }
    ).to_csv(registry_dir / "factor_library_registry.csv", index=False)

    # Create expression registry with only alpha00001
    catalog_dir = tmp_path / "u1" / "catalog"
    catalog_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "alpha_name": ["alpha00001"],
            "expression": ["close/open"],
        }
    ).to_csv(catalog_dir / "expressions.csv", index=False)

    rows = list_superalpha_components(base_dir=tmp_path, universe_name="u1")
    by_factor = {row["factor"]: row for row in rows}

    assert by_factor["alpha00001"]["can_reproduce"] is True
    assert by_factor["alpha00002"]["can_reproduce"] is False


def test_resolve_component_signal_compact_fallback(tmp_path: Path) -> None:
    """Test compact signal fallback when file is unreadable."""
    registry_dir = tmp_path / "u1" / "library"
    registry_dir.mkdir(parents=True)
    # Create a corrupt signal file
    corrupt_path = tmp_path / "signals" / "alpha_a_signal.parquet"
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_bytes(b"corrupt data")

    pd.DataFrame(
        {
            "universe": ["u1"],
            "factor": ["alpha_a"],
            "status": ["accepted"],
            "score": [70.0],
            "signal_artifact_path": [str(corrupt_path)],
        }
    ).to_csv(registry_dir / "factor_library_registry.csv", index=False)

    rows = list_superalpha_components(base_dir=tmp_path, universe_name="u1")
    assert rows[0]["signal_status"] == "read_error"
    assert rows[0]["signal_available"] is False


def test_metadata_weight_clamp_negative(tmp_path: Path) -> None:
    """Test that metadata weights clamp negative values to 0."""
    from alpha_mining.workflow.superalpha import parse_combo_expression

    components = [
        {"factor": "alpha_a", "score": 60.0},
        {"factor": "alpha_b", "score": -30.0},  # negative score
    ]

    result = parse_combo_expression("score", components)

    assert result.basis == "score"
    # Negative should be clamped to 0, so alpha_a gets all weight
    assert result.weights[0] == pytest.approx(1.0)
    assert result.weights[1] == pytest.approx(0.0)


def test_fixed_weight_negative_preserved() -> None:
    """Test that fixed weights preserve negative values."""
    from alpha_mining.workflow.superalpha import parse_combo_expression

    components = [{"factor": "a"}, {"factor": "b"}]

    result = parse_combo_expression("[0.6, -0.4]", components)

    assert result.basis == "fixed"
    # Fixed weights normalize by sum(abs)
    assert result.weights[0] == pytest.approx(0.6)
    assert result.weights[1] == pytest.approx(-0.4)


def test_hash_changes_with_schema_config() -> None:
    """Test that superalpha_id hash changes with schema/config."""
    from alpha_mining.workflow.superalpha import _stable_hash

    payload1 = {
        "schema_version": 1,
        "component_normalization": "cs_zscore",
        "final_normalization": "cs_zscore",
        "component_join": "inner",
        "direction_adjustment": True,
        "universe": "u1",
        "factors": ["alpha_a"],
        "combo_expression": "1",
        "weight_normalization": "sum_abs",
        "period": 1,
        "layers": 10,
        "component_fingerprints": ["abc"],
    }
    payload2 = {**payload1, "schema_version": 2}

    hash1 = _stable_hash(payload1)
    hash2 = _stable_hash(payload2)

    assert hash1 != hash2


def _setup_factor_with_expression(
    tmp_path: Path,
    factor: str,
    *,
    has_manifest: bool = True,
    manifest_has_snapshot: bool = True,
    snapshot_exists: bool = True,
) -> None:
    """Helper to set up a factor with expression registry and optional manifest."""
    registry_dir = tmp_path / "u1" / "library"
    registry_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "universe": ["u1"],
            "factor": [factor],
            "status": ["accepted"],
            "score": [70.0],
            "signal_artifact_path": [""],
        }
    ).to_csv(registry_dir / "factor_library_registry.csv", index=False)

    catalog_dir = tmp_path / "u1" / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)

    # Expression registry
    expr_rows = [{"alpha_name": factor, "expression": "close/open"}]
    if has_manifest:
        expr_rows[0]["input_manifest_id"] = "manifest_001"
    pd.DataFrame(expr_rows).to_csv(catalog_dir / "expressions.csv", index=False)

    # Input manifest
    if has_manifest:
        manifest_dir = catalog_dir / "input_manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = str(tmp_path / "snapshots" / "snapshot_001.parquet")
        if snapshot_exists:
            sp = Path(snapshot_path)
            sp.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"date": ["2025-01-01"], "code": ["000001.SZ"]}).to_parquet(sp, index=False)
        manifest = {
            "manifest_id": "manifest_001",
            "snapshot_path": snapshot_path if manifest_has_snapshot else "",
            "source_path": "",
            "duckdb_path": str(tmp_path / "duckdb" / "market.duckdb"),
            "source_view": "v_test",
        }
        (manifest_dir / "manifest_001.json").write_text(__import__("json").dumps(manifest), encoding="utf-8")


def test_inspect_reproducible_factor_can_backtest(tmp_path: Path) -> None:
    """expression + manifest + snapshot 存在 → can_backtest=True, signal_status='reproducible'."""
    _setup_factor_with_expression(
        tmp_path,
        "alpha00001",
        has_manifest=True,
        manifest_has_snapshot=True,
        snapshot_exists=True,
    )

    rows = list_superalpha_components(base_dir=tmp_path, universe_name="u1")

    assert len(rows) == 1
    row = rows[0]
    assert row["signal_status"] == "reproducible"
    assert row["can_backtest"] is True
    assert row["can_reproduce"] is True
    assert row["strict_reproducibility"] is True


def test_inspect_duckdb_fallback_no_manifest(tmp_path: Path) -> None:
    """expression 存在 + manifest 缺失 → can_backtest=True, signal_status='duckdb_fallback'."""
    _setup_factor_with_expression(tmp_path, "alpha00001", has_manifest=False)

    rows = list_superalpha_components(base_dir=tmp_path, universe_name="u1")

    assert len(rows) == 1
    row = rows[0]
    assert row["signal_status"] == "duckdb_fallback"
    assert row["can_backtest"] is True  # expression exists, reproduce may succeed
    assert row["can_reproduce"] is True
    assert row["strict_reproducibility"] is False
    assert "manifest" in (row.get("reproduce_warning") or "").lower()


def test_inspect_duckdb_fallback_no_snapshot(tmp_path: Path) -> None:
    """expression + manifest 有 duckdb 信息 + snapshot 缺失 → can_backtest=True."""
    _setup_factor_with_expression(
        tmp_path,
        "alpha00001",
        has_manifest=True,
        manifest_has_snapshot=True,
        snapshot_exists=False,
    )

    rows = list_superalpha_components(base_dir=tmp_path, universe_name="u1")

    assert len(rows) == 1
    row = rows[0]
    assert row["signal_status"] == "duckdb_fallback"
    assert row["can_backtest"] is True  # manifest has duckdb_path + source_view
    assert row["can_reproduce"] is True
    assert row["strict_reproducibility"] is False


def test_inspect_duckdb_fallback_no_duckdb_info(tmp_path: Path) -> None:
    """expression + manifest 存在但无 duckdb 信息 + snapshot 缺失 → can_backtest=True."""
    registry_dir = tmp_path / "u1" / "library"
    registry_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "universe": ["u1"],
            "factor": ["alpha00001"],
            "status": ["accepted"],
            "score": [70.0],
            "signal_artifact_path": [""],
        }
    ).to_csv(registry_dir / "factor_library_registry.csv", index=False)

    catalog_dir = tmp_path / "u1" / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "alpha_name": ["alpha00001"],
            "expression": ["close/open"],
            "input_manifest_id": ["m1"],
        }
    ).to_csv(catalog_dir / "expressions.csv", index=False)

    manifest_dir = catalog_dir / "input_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    # Manifest without duckdb_path/source_view
    (manifest_dir / "m1.json").write_text(
        __import__("json").dumps({"manifest_id": "m1", "snapshot_path": "", "source_path": ""}),
        encoding="utf-8",
    )

    rows = list_superalpha_components(base_dir=tmp_path, universe_name="u1")

    assert len(rows) == 1
    row = rows[0]
    assert row["signal_status"] == "duckdb_fallback"
    assert row["can_backtest"] is True  # expression exists, reproduce may succeed
    assert row["can_reproduce"] is True
    assert row["strict_reproducibility"] is False


def test_direction_sign_parsed_from_factor_metrics(tmp_path: Path) -> None:
    """factor_metrics 有 direction_sign 列时正确解析."""
    from alpha_mining.workflow.superalpha import _resolve_direction_sign

    # Create source run with direction_sign in factor_metrics
    source_run_dir = tmp_path / "u1" / "analysis" / "period_1" / "analysis_run_001"
    source_run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"factor": ["alpha00001"], "direction_sign": [-1]}).to_csv(
        source_run_dir / "factor_metrics.csv", index=False
    )

    sign, status, warning = _resolve_direction_sign(
        "alpha00001",
        {"analysis_run_id": "run_001"},
        base_dir=tmp_path,
        universe_name="u1",
    )

    assert sign == -1
    assert status == "source_factor_metrics"
    assert warning == ""


def test_direction_warning_on_default(tmp_path: Path) -> None:
    """默认 +1 时有 warning."""
    from alpha_mining.workflow.superalpha import _resolve_direction_sign

    sign, status, warning = _resolve_direction_sign(
        "alpha00001",
        {},  # no registry direction, no source run
        base_dir=tmp_path,
        universe_name="u1",
    )

    assert sign == 1
    assert status == "missing_default_positive"
    assert "defaulted to +1" in warning


def test_direction_label_parsing() -> None:
    """Test that direction labels like top/bottom/long/short are parsed."""
    from alpha_mining.workflow.superalpha import _parse_direction_value

    assert _parse_direction_value("top") == 1
    assert _parse_direction_value("long") == 1
    assert _parse_direction_value("positive") == 1
    assert _parse_direction_value("+1") == 1
    assert _parse_direction_value("bottom") == -1
    assert _parse_direction_value("short") == -1
    assert _parse_direction_value("negative") == -1
    assert _parse_direction_value("-1") == -1
    assert _parse_direction_value(1) == 1
    assert _parse_direction_value(-1) == -1
    assert _parse_direction_value(None) is None
    assert _parse_direction_value("") is None
    assert _parse_direction_value("unknown") is None


def test_join_coverage_diagnostics(tmp_path: Path) -> None:
    """coverage 记录正确."""
    registry_dir = tmp_path / "u1" / "library"
    registry_dir.mkdir(parents=True)
    dates_a = pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"])
    codes_a = ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"]
    # Component A has 4 rows (2 stocks x 2 dates)
    sig_a = pd.DataFrame({"date": dates_a, "code": codes_a, "alpha_a": [1.0, 2.0, 3.0, 4.0]})
    # Component B has 4 rows but only 000001.SZ on both dates + 000003.SZ on date2
    dates_b = pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"])
    codes_b = ["000001.SZ", "000003.SZ", "000001.SZ", "000003.SZ"]
    sig_b = pd.DataFrame({"date": dates_b, "code": codes_b, "alpha_b": [10.0, 30.0, 20.0, 40.0]})
    path_a = save_dataframe_artifact(sig_a, tmp_path / "series" / "alpha_a_signal", preferred="parquet")["path"]
    path_b = save_dataframe_artifact(sig_b, tmp_path / "series" / "alpha_b_signal", preferred="parquet")["path"]
    pd.DataFrame(
        {
            "universe": ["u1", "u1"],
            "factor": ["alpha_a", "alpha_b"],
            "status": ["accepted", "accepted"],
            "score": [60.0, 80.0],
            "signal_artifact_path": [path_a, path_b],
        }
    ).to_csv(registry_dir / "factor_library_registry.csv", index=False)

    result = build_superalpha_signal(
        base_dir=tmp_path,
        universe_name="u1",
        selected_factor_ids=["alpha_a", "alpha_b"],
        combo_expression="1",
    )

    # Inner join: only 000001.SZ on both dates (2 rows out of 6 union)
    assert result.extra_meta["component_rows"]["alpha_a"] == 4
    assert result.extra_meta["component_rows"]["alpha_b"] == 4
    assert result.extra_meta["post_join_rows"] == 2  # only 000001.SZ on both dates
    assert result.extra_meta["coverage_ratio"] == pytest.approx(2 / 6)  # 2 joined / 6 union
    assert result.extra_meta["join_method"] == "inner"


def test_hash_includes_direction() -> None:
    """hash 随 direction 变化."""
    from alpha_mining.workflow.superalpha import _stable_hash

    payload1 = {
        "factors": ["alpha_a"],
        "direction_signs": [1],
        "direction_statuses": ["registry"],
    }
    payload2 = {
        "factors": ["alpha_a"],
        "direction_signs": [-1],
        "direction_statuses": ["registry"],
    }

    assert _stable_hash(payload1) != _stable_hash(payload2)


def test_build_signal_error_includes_resolution_chain(tmp_path: Path) -> None:
    """错误信息包含 resolution_chain 和 hint."""
    registry_dir = tmp_path / "u1" / "library"
    registry_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "universe": ["u1"],
            "factor": ["alpha_missing"],
            "status": ["accepted"],
            "signal_artifact_path": [""],
        }
    ).to_csv(registry_dir / "factor_library_registry.csv", index=False)

    with pytest.raises(SuperalphaError, match="resolution_chain"):
        build_superalpha_signal(
            base_dir=tmp_path,
            universe_name="u1",
            selected_factor_ids=["alpha_missing"],
            combo_expression="1",
        )


# ---------------------------------------------------------------------------
# P0 tests: cache, lock, diagnostics, DuckDB temp override
# ---------------------------------------------------------------------------


def test_pre_signal_cache_hit(tmp_path: Path, monkeypatch) -> None:
    """Pre-signal cache should return cached result without calling build_superalpha_signal."""
    run_dir = tmp_path / "u1" / "superalphas" / "superalpha_test123"
    run_dir.mkdir(parents=True)
    cfg = SuperalphaConfig()
    analysis_cfg = BatchAnalysisConfig(analysis_level=AnalysisLevelConfig(mode="light"))
    meta = {
        "schema_version": 2,
        "superalpha_id": "superalpha_test123",
        "universe": "u1",
        "combo_expression": "1",
        "component_count": 1,
        "components": [{"factor": "alpha_a", "direction_sign": 1, "direction_status": "registry"}],
        "component_normalization": cfg.component_normalization,
        "final_normalization": cfg.final_normalization,
        "component_join": cfg.component_join,
        "direction_adjustment": cfg.direction_adjustment,
        "weight_normalization": cfg.weight_normalization,
        "weight_basis": "equal_weight",
        "period": int(analysis_cfg.period),
        "layers": int(analysis_cfg.layers),
        "summary": {"sharpe": 1.5},
    }
    (run_dir / "meta.json").write_text(__import__("json").dumps(meta), encoding="utf-8")

    result = find_existing_superalpha_run(
        base_dir=tmp_path,
        universe_name="u1",
        selected_factor_ids=["alpha_a"],
        combo_expression="1",
        config=cfg,
        analysis_config=analysis_cfg,
    )
    assert result is not None
    assert result["status"] == "cached"
    assert result["cache_stage"] == "pre_signal"
    assert result["superalpha_id"] == "superalpha_test123"


def test_runtime_dirs_helper(tmp_path: Path) -> None:
    """Runtime dirs should all be under superalphas/."""
    cfg = SuperalphaConfig()
    dirs = _superalpha_runtime_dirs(tmp_path, "u1", cfg)
    assert dirs["root"] == tmp_path / "u1" / "superalphas"
    assert dirs["python_tmp"] == tmp_path / "u1" / "superalphas" / "_tmp" / "python"
    assert dirs["duckdb_tmp"] == tmp_path / "u1" / "superalphas" / "_tmp" / "duckdb"
    assert dirs["component_tmp"] == tmp_path / "u1" / "superalphas" / "_tmp" / "components"
    assert dirs["locks"] == tmp_path / "u1" / "superalphas" / "_locks"


def test_file_lock_prevents_concurrent(tmp_path: Path) -> None:
    """File lock should raise SuperalphaBusyError when lock is held."""
    lock_dir = tmp_path / "lock_test"
    lock_dir.mkdir(parents=True)
    owner_path = lock_dir / "owner.json"
    owner_path.write_text(
        __import__("json").dumps(
            {
                "pid": 99999,
                "created_at_utc": "2026-01-01T00:00:00",
                "created_at_epoch": __import__("time").time(),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SuperalphaBusyError, match="another Superalpha"):
        with _superalpha_run_lock(lock_dir):
            pass


def test_file_lock_cleans_stale(tmp_path: Path) -> None:
    """Stale lock should be cleaned and allow proceeding."""
    lock_dir = tmp_path / "lock_stale"
    lock_dir.mkdir(parents=True)
    owner_path = lock_dir / "owner.json"
    owner_path.write_text(
        __import__("json").dumps(
            {
                "pid": 99999,
                "created_at_utc": "2020-01-01T00:00:00",
                "created_at_epoch": 0,
            }
        ),
        encoding="utf-8",
    )
    with _superalpha_run_lock(lock_dir):
        pass  # Should not raise


def test_disk_snapshot(tmp_path: Path) -> None:
    """Disk snapshot should return free/total for each path."""
    paths = {"test_dir": tmp_path}
    result = _disk_snapshot(paths)
    assert "test_dir" in result
    assert "free_gb" in result["test_dir"]
    assert result["test_dir"]["free_gb"] > 0


def test_process_memory_snapshot() -> None:
    """Process memory snapshot should return available or fallback."""
    result = _process_memory_snapshot()
    assert "available" in result


def test_safe_dir_size(tmp_path: Path) -> None:
    """Safe dir size should report file count and size."""
    (tmp_path / "test.txt").write_text("hello", encoding="utf-8")
    result = _safe_dir_size(tmp_path)
    assert result["exists"] is True
    assert result["file_count"] >= 1
    assert result["size_mb"] >= 0


def test_system_drive_preflight_does_not_raise_for_reasonable_limit() -> None:
    """System drive check with 0.001 GB should not raise on any system."""
    _check_system_drive_free_space(0.001)  # Should not raise


def test_system_drive_preflight_raises_for_absurd_limit() -> None:
    """System drive check with absurd limit should raise."""
    with pytest.raises(SuperalphaError, match="系统盘剩余空间不足"):
        _check_system_drive_free_space(999999.0)


def test_write_resource_diagnostics(tmp_path: Path) -> None:
    """Resource diagnostics should write JSON with expected fields."""
    from alpha_mining.workflow.superalpha import _write_resource_diagnostics

    cfg = SuperalphaConfig()
    runtime_dirs = _superalpha_runtime_dirs(tmp_path, "u1", cfg)
    output_path = tmp_path / "resource_meta.json"

    _write_resource_diagnostics(
        output_path,
        universe="u1",
        selected_count=3,
        cfg=cfg,
        runtime_dirs=runtime_dirs,
        snapshots=[{"stage": "test", "disk": {}, "process_memory": {}}],
        stage="done",
    )
    assert output_path.exists()
    data = __import__("json").loads(output_path.read_text(encoding="utf-8"))
    assert data["universe"] == "u1"
    assert data["selected_count"] == 3
    assert data["stage"] == "done"
    assert "runtime_dirs" in data
    assert "duckdb_settings" in data


def test_write_resource_diagnostics_skipped_when_disabled(tmp_path: Path) -> None:
    """Resource diagnostics should not write when enable_resource_diagnostics=False."""
    from alpha_mining.workflow.superalpha import _write_resource_diagnostics

    cfg = SuperalphaConfig(enable_resource_diagnostics=False)
    runtime_dirs = _superalpha_runtime_dirs(tmp_path, "u1", cfg)
    output_path = tmp_path / "resource_meta.json"

    _write_resource_diagnostics(
        output_path,
        universe="u1",
        selected_count=3,
        cfg=cfg,
        runtime_dirs=runtime_dirs,
    )
    assert not output_path.exists()


def test_superalpha_config_has_resource_fields() -> None:
    """SuperalphaConfig should have all new resource/diagnostic fields."""
    cfg = SuperalphaConfig()
    assert cfg.python_tmp_subdir == "_tmp/python"
    assert cfg.duckdb_tmp_subdir == "_tmp/duckdb"
    assert cfg.component_tmp_subdir == "_tmp/components"
    assert cfg.duckdb_memory_limit == "2GB"
    assert cfg.duckdb_max_temp_directory_size == "50GB"
    assert cfg.duckdb_threads == ""
    assert cfg.min_system_drive_free_space_gb == 8.0
    assert cfg.enable_resource_diagnostics is True
    assert cfg.component_cache_policy == "bounded"
    assert cfg.component_cache_max_size_gb == 2.0
    assert cfg.component_cache_max_files == 200
    assert cfg.component_cache_ttl_days == 14


def test_pre_signal_cache_metadata_weight_uses_component_metadata(
    tmp_path: Path,
) -> None:
    """Metadata weight cache lookup should use accepted component metadata, not dummy rows."""
    registry_dir = tmp_path / "u1" / "library"
    registry_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "universe": ["u1", "u1"],
            "factor": ["alpha_a", "alpha_b"],
            "status": ["accepted", "accepted"],
            "score": [75.0, 25.0],
        }
    ).to_csv(registry_dir / "factor_library_registry.csv", index=False)

    cfg = SuperalphaConfig()
    analysis_cfg = BatchAnalysisConfig(analysis_level=AnalysisLevelConfig(mode="light"))
    run_dir = tmp_path / "u1" / "superalphas" / "superalpha_score"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(
        __import__("json").dumps(
            {
                "schema_version": 2,
                "superalpha_id": "superalpha_score",
                "universe": "u1",
                "combo_expression": "score",
                "component_count": 2,
                "components": [{"factor": "alpha_a"}, {"factor": "alpha_b"}],
                "component_normalization": cfg.component_normalization,
                "final_normalization": cfg.final_normalization,
                "component_join": cfg.component_join,
                "direction_adjustment": cfg.direction_adjustment,
                "weight_normalization": cfg.weight_normalization,
                "period": int(analysis_cfg.period),
                "layers": int(analysis_cfg.layers),
                "summary": {"score_total": 1.0},
            }
        ),
        encoding="utf-8",
    )

    result = find_existing_superalpha_run(
        base_dir=tmp_path,
        universe_name="u1",
        selected_factor_ids=["alpha_a", "alpha_b"],
        combo_expression="score",
        config=cfg,
        analysis_config=analysis_cfg,
    )

    assert result is not None
    assert result["status"] == "cached"
    assert result["superalpha_id"] == "superalpha_score"


def test_list_components_marks_unreadable_signal_unavailable(tmp_path: Path) -> None:
    """Existing but unreadable signal artifacts should not be selectable for backtest."""
    corrupt_path = tmp_path / "signals" / "bad_signal.parquet"
    corrupt_path.parent.mkdir(parents=True)
    corrupt_path.write_bytes(b"not a parquet file")
    registry_dir = tmp_path / "u1" / "library"
    registry_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "universe": ["u1"],
            "factor": ["alpha_bad"],
            "status": ["accepted"],
            "signal_artifact_path": [str(corrupt_path)],
        }
    ).to_csv(registry_dir / "factor_library_registry.csv", index=False)

    rows = list_superalpha_components(base_dir=tmp_path, universe_name="u1")

    assert rows[0]["signal_status"] == "read_error"
    assert rows[0]["signal_available"] is False
    assert rows[0]["can_backtest"] is False


def test_component_cache_key_changes_with_expression_and_manifest() -> None:
    """Component cache keys should include expression and source lineage."""
    row = {
        "factor": "alpha_a",
        "expression": "close/open",
        "expression_hash": "expr1",
        "input_manifest_id": "m1",
        "simulation_config_hash": "sim1",
    }

    key1 = _component_cache_key("alpha_a", row)
    key2 = _component_cache_key("alpha_a", {**row, "expression_hash": "expr2"})
    key3 = _component_cache_key("alpha_a", {**row, "input_manifest_id": "m2"})

    assert key1.startswith("alpha_a_")
    assert key1 != key2
    assert key1 != key3


def test_prune_component_cache_removes_old_and_excess_files(tmp_path: Path) -> None:
    """Bounded component cache should prune stale and excessive files."""
    cache_dir = tmp_path / "_component_cache"
    cache_dir.mkdir()
    old_file = cache_dir / "old.parquet"
    new_file = cache_dir / "new.parquet"
    extra_file = cache_dir / "extra.parquet"
    old_file.write_bytes(b"old")
    new_file.write_bytes(b"new")
    extra_file.write_bytes(b"extra")
    old_time = __import__("time").time() - 20 * 86400
    __import__("os").utime(old_file, (old_time, old_time))

    summary = _prune_component_cache(
        cache_dir,
        max_size_gb=1.0,
        max_files=1,
        ttl_days=14,
    )

    assert summary["deleted_files"] >= 2
    assert old_file.exists() is False
    assert len(list(cache_dir.glob("*.parquet"))) == 1


def test_resolve_component_signal_passes_duckdb_settings_to_reproduce(tmp_path: Path, monkeypatch) -> None:
    """SA reproduce fallback should pass DuckDB temp/memory settings through to reproduce."""
    captured: dict[str, object] = {}

    def fake_reproduce_alpha_by_name(**kwargs):
        captured.update(kwargs)
        return {
            "output_df": pd.DataFrame(
                {
                    "date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
                    "code": ["000001.SZ", "000002.SZ"],
                    "alpha_a": [1.0, 2.0],
                }
            ),
            "saved": {},
            "reproduce_source_mode": "duckdb_fallback",
            "strict_reproducibility": False,
            "reproduce_warning": "",
        }

    import alpha_mining.workflow.reproduce as reproduce_module

    monkeypatch.setattr(reproduce_module, "reproduce_alpha_by_name", fake_reproduce_alpha_by_name)
    duckdb_settings = {
        "temp_directory": str(tmp_path / "duck_tmp"),
        "memory_limit": "2GB",
    }

    raw, meta = _resolve_component_signal(
        "alpha_a",
        {"factor": "alpha_a", "expression": "close/open"},
        base_dir=tmp_path,
        universe_name="u1",
        config=SuperalphaConfig(allow_reproduce_fallback=True, cache_reproduced_components=False),
        duckdb_settings_override=duckdb_settings,
    )

    assert raw.empty is False
    assert meta["signal_status"] == "reproduced"
    assert captured["duckdb_settings_override"] == duckdb_settings
