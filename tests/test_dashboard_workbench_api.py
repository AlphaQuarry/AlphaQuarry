from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from dashboard.api.app import create_app


def test_dashboard_overview_reports_freshness_without_secret_paths(tmp_path: Path) -> None:
    catalog_dir = tmp_path / "data" / "lake" / "meta"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "field_name": ["close", "volume"],
            "factor_family": ["price_volume", "price_volume"],
            "available_end": ["2026-05-29", "2026-05-30"],
            "is_searchable": [True, True],
        }
    ).to_parquet(catalog_dir / "field_catalog.parquet", index=False)
    feedback_dir = tmp_path / "u1" / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"alpha_name": ["alpha1"], "feedback_score": [61.0]}).to_csv(
        feedback_dir / "expression_scoreboard.csv",
        index=False,
    )

    client = TestClient(create_app(store_root=tmp_path))
    response = client.get("/api/dashboard/overview")

    assert response.status_code == 200
    payload = response.json()
    assert payload["field_catalog_status"] == "ok"
    assert payload["field_catalog_max_available_end"].startswith("2026-05-30")
    assert payload["universe_count"] == 1
    encoded = json.dumps(payload, ensure_ascii=False)
    assert "datasource.local.yaml" not in encoded
    assert "token" not in encoded.lower()


def test_preflight_api_reports_token_risk_without_exposing_secret(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("configs/datasource.local.yaml\n.env\n", encoding="utf-8")
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "datasource.local.yaml").write_text(
        "tushare:\n  token: super-secret-token\n  http_url: https://private.local\n",
        encoding="utf-8",
    )

    client = TestClient(create_app(store_root=tmp_path))
    response = client.get("/api/preflight", params={"config": str(cfg_dir / "datasource.local.yaml")})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "warn"
    assert payload["strict_exit_code"] == 2
    encoded = json.dumps(payload, ensure_ascii=False)
    assert "super-secret-token" not in encoded
    assert "datasource.local.yaml" not in encoded
    assert any("TUSHARE_TOKEN" in item for item in payload["remediations"])
    assert any("custom tushare.http_url" in item for item in payload["infos"])


def test_preflight_example_config_is_strict_clean_without_secret_leak() -> None:
    from dashboard.api.preflight import run_preflight_checks

    payload = run_preflight_checks(config="configs/datasource.example.yaml", root=Path.cwd())

    assert payload["status"] == "ok"
    assert payload["strict_exit_code"] == 0
    encoded = json.dumps(payload, ensure_ascii=False).lower()
    assert "token:" not in encoded
    assert "datasource.local.yaml" not in encoded


def test_library_check_returns_thresholds_and_legacy_status(tmp_path: Path) -> None:
    from tests.test_factor_dashboard_api import _write_analysis_run

    run_id = "analysis_alpha00001-alpha00002_l10_ts1"
    _write_analysis_run(tmp_path, "cn_all", run_id, include_phase_data=True)
    library_dir = tmp_path / "cn_all" / "library"
    library_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "factor": ["alpha00002"],
            "analysis_run_id": [run_id],
            "status": ["accepted"],
            "score": [55.0],
            "score_basis": ["effective"],
        }
    ).to_csv(library_dir / "factor_library_registry.csv", index=False)

    client = TestClient(create_app(store_root=tmp_path))
    check = client.post(
        "/api/factors/alpha00002/library/check",
        json={"universe": "cn_all", "run_id": run_id},
    )
    status = client.get(
        "/api/factors/alpha00002/library/status",
        params={"universe": "cn_all", "run_id": run_id},
    )

    assert check.status_code == 200
    assert check.json()["thresholds"]["min_score"] == 60.0
    assert check.json()["decision"] in {"reject", "staging"}
    assert status.status_code == 200
    assert status.json()["library_status"] == "legacy_accepted"
    assert status.json()["registry_row"]["legacy_status_warning"]


def test_closed_loop_params_include_safe_defaults(tmp_path: Path) -> None:
    client = TestClient(create_app(store_root=tmp_path))

    response = client.get("/api/closed-loop/params")

    assert response.status_code == 200
    payload = response.json()
    names = {param["name"]: param for group in payload["groups"] for param in group["params"]}
    assert names["source_backend"]["default"] == "duckdb"
    assert names["source_chunk_loading"]["default"] is True
    assert names["source_chunk_mem_hard_limit_mb"]["default"] >= 4096
    assert names["max_eval"]["max"] <= 5000
    assert {"basic", "data", "generation", "analysis", "resources", "library"}.issubset(
        {group["id"] for group in payload["groups"]}
    )


class _FakePopen:
    next_pid = 43210
    instances: list["_FakePopen"] = []

    def __init__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        self.pid = _FakePopen.next_pid
        _FakePopen.next_pid += 1
        self.returncode = None
        self.terminated = False
        _FakePopen.instances.append(self)

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15


def test_closed_loop_job_create_conflict_and_cancel(tmp_path: Path, monkeypatch) -> None:
    import dashboard.api.closed_loop_jobs as jobs_module

    _FakePopen.instances.clear()
    monkeypatch.setattr(jobs_module.subprocess, "Popen", _FakePopen)

    client = TestClient(create_app(store_root=tmp_path))
    request = {
        "params": {
            "universe": "u1",
            "start_date": "2026-01-01",
            "end_date": "2026-01-31",
            "request_new": 2,
            "batch_size": 2,
            "max_eval": 5,
            "iterations": 1,
            "source_chunk_mem_hard_limit_mb": 4096,
        }
    }

    created = client.post("/api/closed-loop/jobs", json=request)
    conflict = client.post("/api/closed-loop/jobs", json=request)
    listed = client.get("/api/closed-loop/jobs")
    detail = client.get(f"/api/closed-loop/jobs/{created.json()['job_id']}")
    cancelled = client.post(f"/api/closed-loop/jobs/{created.json()['job_id']}/cancel")

    assert created.status_code == 200
    assert created.json()["status"] == "running"
    assert created.json()["pid"] == 43210
    assert conflict.status_code == 409
    assert listed.status_code == 200
    assert listed.json()["jobs"][0]["job_id"] == created.json()["job_id"]
    assert detail.status_code == 200
    assert "run_closed_loop.py" in " ".join(detail.json()["command"])
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert _FakePopen.instances[0].terminated is True
