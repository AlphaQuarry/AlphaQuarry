from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from dashboard.api.app import create_app


def _write_registry(root: Path, universe: str) -> None:
    library_dir = root / universe / "library"
    library_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "universe": [universe, universe, universe],
            "factor": ["alpha_a", "alpha_b", "alpha_c"],
            "expression": ["a", "b", "c"],
            "status": ["accepted", "staging", "rejected"],
            "score": [70.0, 55.0, 80.0],
            "acceptance_mode": ["standard", "", ""],
            "signal_artifact_path": ["a.parquet", "b.parquet", "c.parquet"],
        }
    ).to_csv(library_dir / "factor_library_registry.csv", index=False)


def _write_superalpha_run(root: Path, universe: str, superalpha_id: str = "superalpha_test") -> Path:
    run_dir = root / universe / "superalphas" / superalpha_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(
        {
            "factor": ["superalpha"],
            "period": [1],
            "layers": [10],
            "expression": ["superalpha_combo"],
            "ic_mean": [0.02],
            "ir": [0.4],
            "long_short_total_return": [0.05],
            "long_short_annualized_return": [0.10],
            "long_short_volatility": [0.20],
            "long_short_sharpe_ratio": [0.5],
            "long_short_max_drawdown": [0.04],
            "long_short_fitness_ratio": [0.3],
            "best_layer_total_return": [0.06],
            "best_layer_annualized_return": [0.11],
            "best_layer_volatility": [0.18],
            "best_layer_sharpe": [0.6],
            "best_layer_max_drawdown": [0.03],
            "best_layer_fitness_ratio": [0.4],
            "turnover_long_only_mean": [0.2],
            "margin_long_only": [0.001],
            "score_predictive_power": [60.0],
            "score_long_only_performance": [50.0],
            "score_stability": [55.0],
            "score_tradeability": [90.0],
            "score_total": [62.0],
            "feedback_score": [62.0],
            "effectiveness_tier": ["B"],
            "train_score_total": [62.0],
            "train_ic_mean": [0.02],
            "train_ir": [0.4],
        }
    )
    metrics_path = run_dir / "dashboard_factor_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    factor_metrics_path = run_dir / "factor_metrics.csv"
    metrics.to_csv(factor_metrics_path, index=False)
    pnl = pd.DataFrame(
        {
            "factor": ["superalpha", "superalpha"],
            "trade_date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "portfolio": ["long_only", "long_only"],
            "return": [0.01, 0.02],
            "cum_return": [0.01, 0.0302],
            "return_gross": [0.01, 0.02],
            "cum_return_gross": [0.01, 0.0302],
            "has_net_pnl": [False, False],
            "holding_count": [2, 2],
            "turnover": [0.0, 0.2],
        }
    )
    pnl_path = run_dir / "portfolio_pnl_df.parquet"
    pnl.to_parquet(pnl_path, index=False)
    ic = pd.DataFrame({"trade_date": pd.to_datetime(["2025-01-01", "2025-01-02"]), "superalpha_ic": [0.1, 0.2]})
    ic_path = run_dir / "ic_df.csv"
    ic.to_csv(ic_path, index=False)
    meta = {
        "schema_version": 1,
        "superalpha_id": superalpha_id,
        "analysis_run_id": superalpha_id,
        "name": "Unit Superalpha",
        "universe": universe,
        "created_at_utc": "2026-05-22T00:00:00+00:00",
        "combo_expression": "1",
        "component_count": 1,
        "components": [
            {
                "factor": "alpha_a",
                "weight": 1.0,
                "expression": "a",
                "signal_status": "compact",
                "acceptance_mode": "standard",
                "score": 70.0,
                "candidate_long_only_sharpe": 0.7,
                "candidate_long_short_sharpe": 0.6,
            }
        ],
        "summary": {"score_total": 62.0},
        "component_join": "concat",
        "cache_summary": {"deleted_count": 1, "remaining_bytes": 1234},
        "cleanup_summary": {"removed_paths": 3, "failed_paths": []},
        "period": 1,
        "layers": 10,
        "analysis_dir": str(run_dir.as_posix()),
        "factor_metrics_path": str(factor_metrics_path.as_posix()),
        "table_paths": {
            "dashboard_factor_metrics": str(metrics_path.as_posix()),
            "portfolio_pnl_df": str(pnl_path.as_posix()),
            "ic_df": str(ic_path.as_posix()),
        },
        "alpha_names": ["superalpha"],
        "extra_meta": {"superalpha": True},
        "status": "ok",
    }
    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (run_dir / "analysis_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (run_dir / "resource_meta.json").write_text(
        json.dumps(
            {
                "runtime_dirs": {"duckdb_tmp": str((run_dir / "_tmp" / "duckdb").as_posix())},
                "duckdb_settings": {
                    "temp_directory": str((run_dir / "_tmp" / "duckdb").as_posix()),
                    "memory_limit": "2GB",
                    "max_temp_directory_size": "50GB",
                },
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_superalpha_components_api_returns_only_accepted(tmp_path: Path) -> None:
    _write_registry(tmp_path, "u1")
    client = TestClient(create_app(store_root=tmp_path))

    response = client.get("/api/superalphas/components", params={"universe": "u1"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert [row["factor"] for row in payload["components"]] == ["alpha_a"]


def test_superalpha_backtest_validates_request(tmp_path: Path) -> None:
    client = TestClient(create_app(store_root=tmp_path))

    response = client.post("/api/superalphas/backtest", json={"universe": "u1", "factor_ids": [], "combo_expression": "1"})

    assert response.status_code == 400
    assert "factor_ids" in response.text


def test_superalpha_backtest_rejects_invalid_safety_params(tmp_path: Path) -> None:
    client = TestClient(create_app(store_root=tmp_path))

    bad_join = client.post(
        "/api/superalphas/backtest",
        json={"universe": "u1", "factor_ids": ["alpha_a"], "component_join": "outer"},
    )
    bad_components = client.post(
        "/api/superalphas/backtest",
        json={"universe": "u1", "factor_ids": ["alpha_a"], "max_components": 999},
    )
    bad_threads = client.post(
        "/api/superalphas/backtest",
        json={"universe": "u1", "factor_ids": ["alpha_a"], "duckdb_threads": "many"},
    )

    assert bad_join.status_code == 400
    assert "component_join" in bad_join.text
    assert bad_components.status_code == 400
    assert "max_components" in bad_components.text
    assert bad_threads.status_code == 400
    assert "duckdb_threads" in bad_threads.text


def test_superalpha_backtest_runs_list_and_detail(tmp_path: Path, monkeypatch) -> None:
    _write_registry(tmp_path, "u1")

    def fake_backtest(**kwargs):
        fake_backtest.last_config = kwargs["config"]
        run_dir = _write_superalpha_run(Path(kwargs["base_dir"]), kwargs["universe_name"])
        return {
            "status": "ok",
            "superalpha_id": "superalpha_test",
            "summary": {"score_total": 62.0},
            "artifact_path": str(run_dir.as_posix()),
        }

    import dashboard.api.service as service_module

    monkeypatch.setattr(service_module, "workflow_run_superalpha_backtest", fake_backtest)
    client = TestClient(create_app(store_root=tmp_path))

    response = client.post(
        "/api/superalphas/backtest",
        json={"universe": "u1", "factor_ids": ["alpha_a"], "combo_expression": "1"},
    )
    assert response.status_code == 200
    assert response.json()["superalpha_id"] == "superalpha_test"
    assert getattr(fake_backtest, "last_config").allow_reproduce_fallback is True

    runs = client.get("/api/superalphas/runs", params={"universe": "u1"}).json()
    assert [row["superalpha_id"] for row in runs["runs"]] == ["superalpha_test"]
    assert runs["runs"][0]["display_name"] == "Unit Superalpha"
    assert runs["runs"][0]["components"][0]["factor"] == "alpha_a"
    assert runs["runs"][0]["resource_summary"]["duckdb_memory_limit"] == "2GB"
    assert runs["runs"][0]["cache_summary"]["deleted_count"] == 1
    assert runs["runs"][0]["cleanup_summary"]["removed_paths"] == 3

    detail = client.get("/api/superalphas/superalpha_test/detail").json()
    assert detail["status"] == "ok"
    assert detail["factor"]["factor"] == "superalpha"
    assert detail["pnl"]["status"] == "ok"
    assert detail["analysis_data"]["status"] == "ok"


def test_superalpha_components_api_returns_new_fields(tmp_path: Path) -> None:
    """Test that components API returns new signal status fields."""
    _write_registry(tmp_path, "u1")
    client = TestClient(create_app(store_root=tmp_path))

    response = client.get("/api/superalphas/components", params={"universe": "u1"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert len(payload["components"]) > 0
    # Check new fields exist (even if None/False for missing signals)
    component = payload["components"][0]
    assert "signal_status" in component or "signal_available" in component


def test_superalpha_components_api_reproducible_requires_expression(tmp_path: Path) -> None:
    """Test that can_reproduce is false when expression registry is missing."""
    _write_registry(tmp_path, "u1")
    client = TestClient(create_app(store_root=tmp_path))

    response = client.get("/api/superalphas/components", params={"universe": "u1"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    # No expression registry exists, so can_reproduce should be false
    for component in payload["components"]:
        assert component.get("can_reproduce") is False


def test_superalpha_detail_v1_still_readable(tmp_path: Path) -> None:
    """Test that schema_version=1 runs are still readable."""
    _write_registry(tmp_path, "u1")
    run_dir = _write_superalpha_run(tmp_path, "u1", superalpha_id="superalpha_v1")

    # Read and verify the meta has schema_version=1
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["schema_version"] == 1

    # Detail should still be loadable
    client = TestClient(create_app(store_root=tmp_path))
    detail = client.get("/api/superalphas/superalpha_v1/detail").json()
    assert detail["status"] == "ok"
    assert detail["meta"]["schema_version"] == 1


def test_superalpha_run_rename_updates_display_name_only(tmp_path: Path) -> None:
    run_dir = _write_superalpha_run(tmp_path, "u1", superalpha_id="superalpha_rename")
    client = TestClient(create_app(store_root=tmp_path))

    response = client.patch("/api/superalphas/superalpha_rename", json={"name": "My SA Blend"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["run"]["superalpha_id"] == "superalpha_rename"
    assert payload["run"]["run_id"] == "superalpha_rename"
    assert payload["run"]["name"] == "My SA Blend"
    assert payload["run"]["display_name"] == "My SA Blend"
    assert payload["run"]["artifact_path"] == str(run_dir.as_posix())
    assert (run_dir / "meta.json").exists()
    assert json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))["name"] == "My SA Blend"
    assert json.loads((run_dir / "analysis_meta.json").read_text(encoding="utf-8"))["name"] == "My SA Blend"

    runs = client.get("/api/superalphas/runs", params={"universe": "u1"}).json()
    assert runs["runs"][0]["display_name"] == "My SA Blend"


def test_superalpha_run_rename_rejects_invalid_name(tmp_path: Path) -> None:
    _write_superalpha_run(tmp_path, "u1", superalpha_id="superalpha_rename")
    client = TestClient(create_app(store_root=tmp_path))

    empty = client.patch("/api/superalphas/superalpha_rename", json={"name": "   "})
    control = client.patch("/api/superalphas/superalpha_rename", json={"name": "bad\nname"})
    too_long = client.patch("/api/superalphas/superalpha_rename", json={"name": "x" * 81})

    assert empty.status_code == 400
    assert control.status_code == 400
    assert too_long.status_code == 400


def test_components_api_reproducible_can_be_selected(tmp_path: Path) -> None:
    """Test that reproducible factors have can_backtest=True in API response."""
    library_dir = tmp_path / "u1" / "library"
    library_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "universe": ["u1"],
            "factor": ["alpha00001"],
            "status": ["accepted"],
            "score": [70.0],
            "signal_artifact_path": [""],
        }
    ).to_csv(library_dir / "factor_library_registry.csv", index=False)

    catalog_dir = tmp_path / "u1" / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"alpha_name": ["alpha00001"], "expression": ["close/open"], "input_manifest_id": ["m1"]}
    ).to_csv(catalog_dir / "expressions.csv", index=False)

    manifest_dir = catalog_dir / "input_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = tmp_path / "snapshots" / "s1.parquet"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": ["2025-01-01"], "code": ["000001.SZ"]}).to_parquet(snapshot_path, index=False)
    (manifest_dir / "m1.json").write_text(
        json.dumps({"manifest_id": "m1", "snapshot_path": str(snapshot_path), "source_path": ""}),
        encoding="utf-8",
    )

    client = TestClient(create_app(store_root=tmp_path))
    response = client.get("/api/superalphas/components", params={"universe": "u1"})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["components"]) == 1
    comp = payload["components"][0]
    assert comp["signal_status"] == "reproducible"
    assert comp["can_backtest"] is True
    assert comp["can_reproduce"] is True
