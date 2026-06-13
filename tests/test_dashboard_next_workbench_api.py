from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from dashboard.api.app import create_app


def test_closed_loop_running_job_without_live_pid_becomes_interrupted(tmp_path: Path, monkeypatch) -> None:
    import dashboard.api.closed_loop_jobs as jobs_module

    job_dir = tmp_path / "_dashboard_jobs" / "closed_loop" / "cl_stale"
    job_dir.mkdir(parents=True)
    (job_dir / "stdout.log").write_text("", encoding="utf-8")
    (job_dir / "stderr.log").write_text("", encoding="utf-8")
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": "cl_stale",
                "status": "running",
                "pid": 999991,
                "universe": "cn_all",
                "created_at_utc": "2026-06-01T00:00:00+00:00",
                "started_at_utc": "2026-06-01T00:00:00+00:00",
                "ended_at_utc": "",
                "exit_code": None,
                "params": {"universe": "cn_all"},
                "command": ["python", "scripts\\run_closed_loop.py"],
                "stdout_path": str((job_dir / "stdout.log").as_posix()),
                "stderr_path": str((job_dir / "stderr.log").as_posix()),
                "result_summary": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(jobs_module, "_pid_exists", lambda pid: False)

    client = TestClient(create_app(store_root=tmp_path))
    response = client.get("/api/closed-loop/jobs/cl_stale")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "interrupted"
    assert payload["failure_category"] == "interrupted"
    assert payload["ended_at_utc"]


def test_closed_loop_external_running_job_and_failure_classification(tmp_path: Path, monkeypatch) -> None:
    import dashboard.api.closed_loop_jobs as jobs_module

    job_dir = tmp_path / "_dashboard_jobs" / "closed_loop" / "cl_external"
    job_dir.mkdir(parents=True)
    (job_dir / "stdout.log").write_text("source chunk hard limit triggered\n", encoding="utf-8")
    (job_dir / "stderr.log").write_text("", encoding="utf-8")
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": "cl_external",
                "status": "running",
                "pid": 12345,
                "universe": "cn_all",
                "created_at_utc": "2026-06-01T00:00:00+00:00",
                "started_at_utc": "2026-06-01T00:00:00+00:00",
                "ended_at_utc": "",
                "exit_code": None,
                "params": {"universe": "cn_all"},
                "command": ["python", "scripts\\run_closed_loop.py"],
                "stdout_path": str((job_dir / "stdout.log").as_posix()),
                "stderr_path": str((job_dir / "stderr.log").as_posix()),
                "result_summary": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(jobs_module, "_pid_exists", lambda pid: True)

    client = TestClient(create_app(store_root=tmp_path))
    response = client.get("/api/closed-loop/jobs/cl_external")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "running"
    assert payload["external_process"] is True
    assert payload["failure_category"] == "memory_protection"
    assert "memory" in payload["failure_hint"].lower()
    assert payload["status_label"] == "Running outside dashboard"
    assert payload["status_hint"]


def test_closed_loop_presets_and_conflict_include_lock_owner(tmp_path: Path, monkeypatch) -> None:
    from tests.test_dashboard_workbench_api import _FakePopen
    import dashboard.api.closed_loop_jobs as jobs_module

    _FakePopen.instances.clear()
    monkeypatch.setattr(jobs_module.subprocess, "Popen", _FakePopen)
    lock_dir = tmp_path / "u1" / ".closed_loop.lock"
    lock_dir.mkdir(parents=True)
    (lock_dir / "owner.json").write_text(
        json.dumps(
            {
                "owner_id": "secret-owner-id",
                "pid": 43210,
                "hostname": "local-host",
                "started_at_utc": "2026-06-01T00:00:00+00:00",
                "heartbeat_at_utc": "2026-06-01T00:01:00+00:00",
                "universe": "u1",
                "config_hash": "abc123",
            }
        ),
        encoding="utf-8",
    )
    client = TestClient(create_app(store_root=tmp_path))

    params = client.get("/api/closed-loop/params")
    created = client.post(
        "/api/closed-loop/jobs",
        json={"params": {"universe": "u1", "request_new": 2, "batch_size": 2, "max_eval": 5, "iterations": 1, "source_chunk_mem_hard_limit_mb": 4096}},
    )
    conflict = client.post(
        "/api/closed-loop/jobs",
        json={"params": {"universe": "u1", "request_new": 2, "batch_size": 2, "max_eval": 5, "iterations": 1, "source_chunk_mem_hard_limit_mb": 4096}},
    )

    assert params.status_code == 200
    smoke = {row["id"]: row for row in params.json()["presets"]}["smoke"]
    assert smoke["params"]["request_new"] == 5
    assert smoke["params"]["max_eval"] == 80
    assert created.status_code == 200
    assert created.json()["lock_owner"]["pid"] == 43210
    assert created.json()["lock_age_seconds"] is not None
    assert created.json()["lock_stale_hint"]
    assert "owner_id" not in json.dumps(created.json()["lock_owner"])
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["running_job"]["lock_owner"]["hostname"] == "local-host"


def test_run_compare_returns_summary_delta_and_overlap(tmp_path: Path) -> None:
    from tests.test_factor_dashboard_api import _write_analysis_run

    _write_analysis_run(tmp_path, "cn_all", "analysis_left", period=1, layers=10)
    _write_analysis_run(tmp_path, "cn_all", "analysis_right", period=5, layers=20)
    right_metrics = tmp_path / "cn_all" / "analysis" / "period_5" / "analysis_right" / "dashboard_factor_metrics.csv"
    frame = pd.read_csv(right_metrics)
    frame.loc[frame["factor"] == "alpha00002", "factor"] = "alpha00003"
    frame.loc[frame["factor"] == "alpha00003", "feedback_score"] = 90.0
    frame.to_csv(right_metrics, index=False)

    client = TestClient(create_app(store_root=tmp_path))
    response = client.get(
        "/api/runs/compare",
        params={"universe": "cn_all", "left_run_id": "analysis_left", "right_run_id": "analysis_right", "top_n": 2},
    )
    scoreboard = client.get(
        "/api/runs/compare",
        params={"universe": "cn_all", "left_run_id": "__scoreboard__", "right_run_id": "analysis_right"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["left"]["run_id"] == "analysis_left"
    assert payload["right"]["run_id"] == "analysis_right"
    assert payload["metrics"]["feedback_score_mean"]["delta"] is not None
    assert payload["left_artifact_status"] == "complete"
    assert payload["right_artifact_status"] == "complete"
    assert payload["overlap"]["overlap_count"] == 1
    assert payload["overlap"]["overlap_ratio"] == 0.5
    assert "alpha00001" in payload["overlap"]["shared_factors"]
    assert scoreboard.status_code == 400


def test_closed_loop_failure_patterns_return_actionable_categories(tmp_path: Path) -> None:
    cases = {
        "duckdb temporary directory is full": "duckdb_temp",
        "closed_loop lock exists and owner is active": "lock_conflict",
        "empty frame after applying run filters": "data_empty",
        "analysis artifact missing: portfolio_pnl_df": "analysis_error",
        "candidate parse error near ts_rank(": "candidate_generation",
        "invalid argument: batch_size": "config_error",
    }
    client = TestClient(create_app(store_root=tmp_path))
    for index, (message, category) in enumerate(cases.items()):
        job_dir = tmp_path / "_dashboard_jobs" / "closed_loop" / f"cl_failed_{index}"
        job_dir.mkdir(parents=True)
        (job_dir / "stdout.log").write_text("", encoding="utf-8")
        (job_dir / "stderr.log").write_text(message, encoding="utf-8")
        (job_dir / "job.json").write_text(
            json.dumps(
                {
                    "job_id": f"cl_failed_{index}",
                    "status": "failed",
                    "pid": 0,
                    "universe": "cn_all",
                    "created_at_utc": "2026-06-01T00:00:00+00:00",
                    "started_at_utc": "2026-06-01T00:00:00+00:00",
                    "ended_at_utc": "2026-06-01T00:01:00+00:00",
                    "exit_code": 1,
                    "params": {"universe": "cn_all"},
                    "command": ["python", "scripts\\run_closed_loop.py"],
                    "stdout_path": str((job_dir / "stdout.log").as_posix()),
                    "stderr_path": str((job_dir / "stderr.log").as_posix()),
                    "result_summary": {},
                }
            ),
            encoding="utf-8",
        )
        response = client.get(f"/api/closed-loop/jobs/cl_failed_{index}")
        assert response.status_code == 200
        payload = response.json()
        assert payload["failure_category"] == category
        assert payload["failure_title"]
        assert payload["failure_hint"]


def test_data_health_reports_missing_coverage_without_low_coverage_false_positive(tmp_path: Path) -> None:
    catalog_dir = tmp_path / "data" / "lake" / "meta"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "field_name": ["close", "volume"],
            "factor_family": ["price", "price"],
            "available_end": ["2026-05-30", "2026-05-30"],
            "coverage_rate": [None, None],
            "is_searchable": [True, True],
        }
    ).to_parquet(catalog_dir / "field_catalog.parquet", index=False)

    client = TestClient(create_app(store_root=tmp_path))
    response = client.get("/api/data/health", params={"universe": "cn_all"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["thresholds"]["low_coverage_threshold"] == 0.5
    assert payload["catalog"]["coverage_status"] == "missing"
    assert payload["catalog"]["coverage_available_count"] == 0
    assert payload["catalog"]["coverage_missing_count"] == 2
    assert payload["catalog"]["low_coverage_count"] == 0
    assert any(warning["code"] == "coverage_not_refreshed" for warning in payload["warnings"])


def test_run_compare_reports_missing_and_partial_artifact_status(tmp_path: Path) -> None:
    from tests.test_factor_dashboard_api import _write_analysis_run

    _write_analysis_run(tmp_path, "cn_all", "analysis_complete", period=1, layers=10)
    _write_analysis_run(tmp_path, "cn_all", "analysis_partial", period=5, layers=10)
    partial_path = tmp_path / "cn_all" / "analysis" / "period_5" / "analysis_partial" / "dashboard_factor_metrics.csv"
    pd.read_csv(partial_path)[["factor", "feedback_score"]].to_csv(partial_path, index=False)
    _write_analysis_run(tmp_path, "cn_all", "analysis_missing", period=10, layers=10)
    missing_path = tmp_path / "cn_all" / "analysis" / "period_10" / "analysis_missing" / "dashboard_factor_metrics.csv"
    missing_path.unlink()

    client = TestClient(create_app(store_root=tmp_path))
    partial = client.get(
        "/api/runs/compare",
        params={"universe": "cn_all", "left_run_id": "analysis_complete", "right_run_id": "analysis_partial"},
    )
    missing = client.get(
        "/api/runs/compare",
        params={"universe": "cn_all", "left_run_id": "analysis_complete", "right_run_id": "analysis_missing"},
    )

    assert partial.status_code == 200
    assert partial.json()["right_artifact_status"] == "partial_metrics"
    assert any("Right run metrics are partial" in warning for warning in partial.json()["warnings"])
    assert missing.status_code == 200
    assert missing.json()["right_artifact_status"] == "missing_metrics"
    assert any("Right run metrics artifact is missing" in warning for warning in missing.json()["warnings"])


def test_data_health_reads_catalog_base_run_health_and_quality_artifact(tmp_path: Path) -> None:
    catalog_dir = tmp_path / "data" / "lake" / "meta"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "field_name": ["close", "volume", "moneyflow"],
            "factor_family": ["price", "price", "moneyflow"],
            "available_end": ["2026-05-30", "2026-05-20", "2026-05-10"],
            "coverage_rate": [0.95, 0.40, 0.20],
            "is_searchable": [True, True, False],
        }
    ).to_parquet(catalog_dir / "field_catalog.parquet", index=False)
    base_dir = tmp_path / "cn_all" / "base"
    base_dir.mkdir(parents=True)
    pd.DataFrame({"trade_date": ["2026-05-30"], "ts_code": ["000001.SZ"], "close": [10.0]}).to_parquet(
        base_dir / "base_frame.parquet",
        index=False,
    )
    feedback_dir = tmp_path / "cn_all" / "feedback"
    feedback_dir.mkdir(parents=True)
    (feedback_dir / "run_health.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"status": "ok", "scoreboard_rows": 5, "source_chunk_memory": {"source_chunk_mem_warning_count": 1}}),
                json.dumps({"status": "failed", "scoreboard_rows": 2, "source_chunk_hard_limit_triggered": True}),
            ]
        ),
        encoding="utf-8",
    )
    quality_dir = Path("artifacts") / "data_quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    quality_path = quality_dir / "panel_quality.json"
    original = quality_path.read_text(encoding="utf-8") if quality_path.exists() else None
    quality_path.write_text(
        json.dumps(
            {
                "overall_status": "warn",
                "generated_at_utc": "2026-06-01T00:00:00+00:00",
                "fields": [
                    {"field": "close", "status": "pass"},
                    {"field": "volume", "status": "warn"},
                    {"field": "moneyflow", "status": "fail"},
                ],
            }
        ),
        encoding="utf-8",
    )
    try:
        client = TestClient(create_app(store_root=tmp_path))
        response = client.get("/api/data/health", params={"universe": "cn_all"})
    finally:
        if original is None:
            quality_path.unlink(missing_ok=True)
        else:
            quality_path.write_text(original, encoding="utf-8")

    assert response.status_code == 200
    payload = response.json()
    assert payload["catalog"]["row_count"] == 3
    assert payload["catalog"]["low_coverage_count"] == 2
    assert payload["universe_base"]["exists"] is True
    assert payload["universe_base"]["rows"] == 1
    assert payload["closed_loop_health"]["hard_limit_count"] == 1
    assert payload["quality_artifact"]["overall_status"] == "warn"
    assert any(row["family"] == "price" for row in payload["families"])
