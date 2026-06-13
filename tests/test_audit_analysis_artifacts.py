from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_audit_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_analysis_artifacts.py"
    spec = importlib.util.spec_from_file_location("audit_analysis_artifacts_under_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_file(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def test_audit_analysis_artifacts_summarizes_dynamic_and_png_sizes(tmp_path: Path) -> None:
    audit = _load_audit_module()
    run_dir = tmp_path / "cn_all" / "analysis" / "period_1" / "analysis_alpha00001_l5_ts1"
    run_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "dashboard_factor_metrics": run_dir / "dashboard_factor_metrics.csv",
        "phase_metrics_df": run_dir / "phase_metrics_df.csv",
        "ic_df": run_dir / "ic_df.csv",
        "portfolio_pnl_df": run_dir / "portfolio_pnl_df.parquet",
        "analysis_distribution_histogram": run_dir / "analysis_distribution_histogram.csv",
        "analysis_ic_decay": run_dir / "analysis_ic_decay.csv",
        "visualization_manifest": run_dir / "visualization_manifest.csv",
    }
    sizes = {
        "dashboard_factor_metrics": 11,
        "phase_metrics_df": 13,
        "ic_df": 17,
        "portfolio_pnl_df": 19,
        "analysis_distribution_histogram": 23,
        "analysis_ic_decay": 29,
        "visualization_manifest": 31,
    }
    for key, path in artifacts.items():
        _write_file(path, sizes[key])
    _write_file(run_dir / "visualizations" / "alpha00001" / "distribution.png", 37)
    _write_file(run_dir / "visualizations" / "alpha00001" / "ic.png", 41)

    meta = {
        "analysis_run_id": "analysis_alpha00001_l5_ts1",
        "alpha_names": ["alpha00001"],
        "period": 1,
        "layers": 5,
        "created_at_utc": "2026-05-16T00:00:00+00:00",
        "analysis_dir": str(run_dir),
        "factor_metrics_path": str(run_dir / "factor_metrics.csv"),
        "table_paths": {key: str(path) for key, path in artifacts.items()},
        "extra_meta": {"include_visualization_png": True},
    }
    _write_file(run_dir / "factor_metrics.csv", 7)
    (run_dir / "analysis_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    report = audit.audit_analysis_artifacts(store_root=tmp_path)

    assert report["run_count"] == 1
    row = report["runs"][0]
    assert row["universe"] == "cn_all"
    assert row["run_id"] == "analysis_alpha00001_l5_ts1"
    assert row["factor_count"] == 1
    assert row["dynamic_analysis_bytes"] == 23 + 29
    assert row["visualization_png_bytes"] == 37 + 41
    assert row["visualization_png_present"] is True
    assert row["visualization_manifest_bytes"] == 31
    assert row["png_enabled"] is True
    assert row["total_bytes"] >= sum(sizes.values()) + 7 + 37 + 41


def test_audit_analysis_artifacts_filters_universe_and_run_id(tmp_path: Path) -> None:
    audit = _load_audit_module()
    for universe in ["cn_all", "cn_small"]:
        run_dir = tmp_path / universe / "analysis" / "period_1" / "analysis_alpha00001_l5_ts1"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "analysis_meta.json").write_text(
            json.dumps(
                {
                    "analysis_run_id": "analysis_alpha00001_l5_ts1",
                    "alpha_names": ["alpha00001"],
                    "period": 1,
                    "layers": 5,
                    "analysis_dir": str(run_dir),
                    "table_paths": {},
                    "extra_meta": {},
                }
            ),
            encoding="utf-8",
        )

    report = audit.audit_analysis_artifacts(
        store_root=tmp_path,
        universe="cn_small",
        run_id="analysis_alpha00001_l5_ts1",
    )

    assert report["run_count"] == 1
    assert report["runs"][0]["universe"] == "cn_small"


def test_audit_analysis_artifacts_is_stable_for_empty_and_missing_artifacts(tmp_path: Path) -> None:
    audit = _load_audit_module()

    empty = audit.audit_analysis_artifacts(store_root=tmp_path)

    run_dir = tmp_path / "cn_all" / "analysis" / "period_1" / "analysis_missing_artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis_meta.json").write_text(
        json.dumps(
            {
                "analysis_run_id": "analysis_missing_artifacts",
                "alpha_names": ["alpha00001"],
                "period": 1,
                "layers": 10,
                "analysis_dir": str(run_dir),
                "factor_metrics_path": str(run_dir / "missing_factor_metrics.csv"),
                "table_paths": {
                    "dashboard_factor_metrics": str(run_dir / "missing_dashboard.csv"),
                    "portfolio_pnl_df": str(run_dir / "missing_pnl.parquet"),
                },
                "extra_meta": {},
            }
        ),
        encoding="utf-8",
    )
    missing = audit.audit_analysis_artifacts(store_root=tmp_path)

    assert empty["run_count"] == 0
    assert empty["total_bytes"] == 0
    assert missing["run_count"] == 1
    row = missing["runs"][0]
    assert row["factor_count"] == 1
    assert row["factor_metrics_bytes"] == 0
    assert row["dashboard_metrics_bytes"] == 0
    assert row["portfolio_pnl_bytes"] == 0
    assert row["dynamic_analysis_bytes"] == 0
