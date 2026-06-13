from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dashboard.api.app import create_app
from alpha_mining.workflow.universe_store import save_universe_alpha_values


def _write_analysis_run(
    root: Path,
    universe: str,
    run_id: str,
    *,
    period: int = 1,
    layers: int = 10,
    include_pnl: bool = True,
    include_dashboard_metrics: bool = True,
    include_visualizations: bool = False,
    include_phase_data: bool = False,
    include_benchmark: bool = False,
) -> None:
    run_dir = root / universe / "analysis" / f"period_{period}" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame(
        {
            "factor": ["alpha00001", "alpha00002"],
            "period": [period, period],
            "layers": [layers, layers],
            "expression": ["rank(close)", "ts_rank(volume, 5)"],
            "ic_mean": [0.03, -0.02],
            "ir": [0.6, -0.3],
            "long_short_total_return": [0.10, 0.04],
            "long_short_annualized_return": [0.12, 0.05],
            "long_short_volatility": [0.20, 0.25],
            "long_short_sharpe_ratio": [1.2, 0.5],
            "long_short_max_drawdown": [0.08, 0.10],
            "long_short_fitness_ratio": [0.7, 0.2],
            "best_layer_total_return": [0.11, 0.03],
            "best_layer_annualized_return": [0.13, 0.04],
            "best_layer_volatility": [0.18, 0.22],
            "best_layer_sharpe": [1.4, 0.4],
            "best_layer_max_drawdown": [0.06, 0.11],
            "best_layer_fitness_ratio": [0.8, 0.1],
            "best_minus_universe_annualized_return": [0.05, -0.01],
            "turnover_long_only_mean": [0.30, 0.40],
            "margin_long_only": [0.002, 0.001],
            "score_predictive_power": [80.0, 40.0],
            "score_long_only_performance": [90.0, 35.0],
            "score_stability": [70.0, 30.0],
            "score_tradeability": [85.0, 50.0],
            "score_total": [82.0, 38.0],
            "effectiveness_tier": ["A", "C"],
            "feedback_phase": ["train", "train"],
            "feedback_score": [80.0, 30.0],
            "train_obs": [2, 2],
            "train_ic_mean": [0.02, -0.01],
            "train_ir": [0.5, -0.2],
            "train_positive_ic_ratio": [1.0, 0.0],
            "train_long_short_total_return": [0.10, 0.02],
            "train_long_short_annualized_return": [0.12, 0.03],
            "train_long_short_volatility": [0.20, 0.24],
            "train_long_short_sharpe_ratio": [1.2, 0.4],
            "train_long_short_max_drawdown": [0.08, 0.11],
            "train_long_short_fitness_ratio": [0.7, 0.2],
            "train_turnover_long_short_mean": [0.25, 0.35],
            "train_margin_long_short": [0.004, 0.001],
            "train_margin_long_short_bp": [40.0, 10.0],
            "train_score_total": [80.0, 30.0],
            "val_obs": [1, 1],
            "val_ic_mean": [0.01, -0.02],
            "val_ir": [0.4, -0.3],
            "val_positive_ic_ratio": [1.0, 0.0],
            "val_long_short_total_return": [0.04, 0.01],
            "val_long_short_annualized_return": [0.05, 0.02],
            "val_long_short_volatility": [0.18, 0.22],
            "val_long_short_sharpe_ratio": [0.5, 0.1],
            "val_long_short_max_drawdown": [0.04, 0.09],
            "val_long_short_fitness_ratio": [0.3, 0.1],
            "val_turnover_long_short_mean": [0.30, 0.40],
            "val_margin_long_short": [0.002, 0.001],
            "val_margin_long_short_bp": [20.0, 10.0],
            "val_score_total": [70.0, 25.0],
            "test_obs": [1, 1],
            "test_ic_mean": [0.03, -0.03],
            "test_ir": [0.6, -0.4],
            "test_positive_ic_ratio": [1.0, 0.0],
            "test_long_short_total_return": [0.02, -0.01],
            "test_long_short_annualized_return": [0.03, -0.02],
            "test_long_short_volatility": [0.16, 0.20],
            "test_long_short_sharpe_ratio": [0.6, -0.2],
            "test_long_short_max_drawdown": [0.03, 0.08],
            "test_long_short_fitness_ratio": [0.2, -0.1],
            "test_turnover_long_short_mean": [0.20, 0.30],
            "test_margin_long_short": [0.001, -0.001],
            "test_margin_long_short_bp": [10.0, -10.0],
            "test_score_total": [75.0, 20.0],
        }
    )
    metrics_path = run_dir / ("dashboard_factor_metrics.csv" if include_dashboard_metrics else "factor_metrics.csv")
    metrics.to_csv(metrics_path, index=False)

    table_paths: dict[str, str] = {}
    if include_dashboard_metrics:
        table_paths["dashboard_factor_metrics"] = str(metrics_path.as_posix())
    if include_pnl:
        if include_phase_data:
            pnl = pd.DataFrame(
                {
                    "factor": ["alpha00001"] * 12,
                    "trade_date": pd.to_datetime(
                        [
                            "2024-12-30",
                            "2024-12-31",
                            "2025-01-02",
                            "2025-01-03",
                            "2026-01-02",
                            "2026-01-05",
                            "2024-12-30",
                            "2024-12-31",
                            "2025-01-02",
                            "2025-01-03",
                            "2026-01-02",
                            "2026-01-05",
                        ]
                    ),
                    "portfolio": [
                        "long_only",
                        "long_short",
                        "long_only",
                        "long_short",
                        "long_only",
                        "long_short",
                        "layer_1",
                        "layer_2",
                        "layer_1",
                        "layer_2",
                        "layer_1",
                        "layer_2",
                    ],
                    "return": [0.01, 0.02, 0.01, 0.02, 0.03, 0.04, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
                    "cum_return": [0.01, 0.02, 0.0201, 0.0404, 0.0507, 0.0820, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
                    "holding_count": [20, None, 20, None, 20, None, 10, 10, 10, 10, 10, 10],
                    "turnover": [0.0, 0.25, 0.2, 0.30, 0.3, 0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "blocked_buy_ratio": [0.0, None, 0.0, None, 0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "blocked_sell_ratio": [0.0, None, 0.0, None, 0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "tradability_return_drag": [0.0, None, 0.0, None, 0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                }
            )
        else:
            pnl = pd.DataFrame(
                {
                    "factor": ["alpha00001", "alpha00001", "alpha00001", "alpha00002"],
                    "trade_date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-01"]),
                    "portfolio": ["long_only", "long_only", "long_10", "long_short"],
                    "return": [0.01, 0.02, 0.015, -0.01],
                    "cum_return": [0.01, 0.0302, 0.015, -0.01],
                    "holding_count": [20, 20, 10, None],
                    "turnover": [0.0, 0.2, 0.1, None],
                    "blocked_buy_ratio": [0.0, 0.0, 0.0, None],
                    "blocked_sell_ratio": [0.0, 0.0, 0.0, None],
                    "tradability_return_drag": [0.0, 0.0, 0.0, None],
                }
            )
        pnl_path = run_dir / "portfolio_pnl_df.parquet"
        pnl.to_parquet(pnl_path, index=False)
        table_paths["portfolio_pnl_df"] = str(pnl_path.as_posix())
    if include_benchmark:
        benchmark = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03", "2026-01-02"]),
                "portfolio": ["benchmark"] * 5,
                "return": [0.005, 0.005, 0.004, 0.004, 0.003],
                "cum_return": [0.005, 0.010025, 0.014065, 0.018121, 0.021175],
                "holding_count": [None] * 5,
                "turnover": [None] * 5,
                "blocked_buy_ratio": [None] * 5,
                "blocked_sell_ratio": [None] * 5,
                "tradability_return_drag": [None] * 5,
            }
        )
        benchmark_path = run_dir / "benchmark_pnl_df.parquet"
        benchmark.to_parquet(benchmark_path, index=False)
        table_paths["benchmark_pnl_df"] = str(benchmark_path.as_posix())
    if include_phase_data:
        phase_metrics = metrics[
            [
                "factor",
                "feedback_phase",
                "feedback_score",
                "train_obs",
                "train_ic_mean",
                "train_ir",
                "train_positive_ic_ratio",
                "train_long_short_total_return",
                "train_long_short_annualized_return",
                "train_long_short_volatility",
                "train_long_short_sharpe_ratio",
                "train_long_short_max_drawdown",
                "train_long_short_fitness_ratio",
                "train_turnover_long_short_mean",
                "train_margin_long_short",
                "train_margin_long_short_bp",
                "train_score_total",
                "val_obs",
                "val_ic_mean",
                "val_ir",
                "val_positive_ic_ratio",
                "val_long_short_total_return",
                "val_long_short_annualized_return",
                "val_long_short_volatility",
                "val_long_short_sharpe_ratio",
                "val_long_short_max_drawdown",
                "val_long_short_fitness_ratio",
                "val_turnover_long_short_mean",
                "val_margin_long_short",
                "val_margin_long_short_bp",
                "val_score_total",
                "test_obs",
                "test_ic_mean",
                "test_ir",
                "test_positive_ic_ratio",
                "test_long_short_total_return",
                "test_long_short_annualized_return",
                "test_long_short_volatility",
                "test_long_short_sharpe_ratio",
                "test_long_short_max_drawdown",
                "test_long_short_fitness_ratio",
                "test_turnover_long_short_mean",
                "test_margin_long_short",
                "test_margin_long_short_bp",
                "test_score_total",
            ]
        ].copy()
        phase_path = run_dir / "phase_metrics_df.csv"
        phase_metrics.to_csv(phase_path, index=False)
        table_paths["phase_metrics_df"] = str(phase_path.as_posix())
        ic_df = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-02", "2026-01-02"]),
                "alpha00001_ic": [0.02, 0.03, 0.01, 0.04],
            }
        )
        ic_path = run_dir / "ic_df.csv"
        ic_df.to_csv(ic_path, index=False)
        table_paths["ic_df"] = str(ic_path.as_posix())
        hist = pd.DataFrame(
            {
                "factor": ["alpha00001"] * 6,
                "phase": ["train", "train", "val", "val", "test", "test"],
                "bin_index": [0, 1, 0, 1, 0, 1],
                "bin_left": [-1.0, 0.0, -1.0, 0.0, -1.0, 0.0],
                "bin_right": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                "bin_mid": [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5],
                "count": [3, 5, 2, 4, 1, 6],
                "total_count": [8, 8, 6, 6, 7, 7],
            }
        )
        hist_path = run_dir / "analysis_distribution_histogram.csv"
        hist.to_csv(hist_path, index=False)
        table_paths["analysis_distribution_histogram"] = str(hist_path.as_posix())
        decay = pd.DataFrame(
            {
                "factor": ["alpha00001"] * 6,
                "phase": ["train", "train", "val", "val", "test", "test"],
                "lag": [0, 1, 0, 1, 0, 1],
                "ic": [0.02, 0.01, 0.03, 0.02, 0.04, 0.01],
                "half_life": [1, 1, 1, 1, 1, 1],
                "ic_decay_rank_corr": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            }
        )
        decay_path = run_dir / "analysis_ic_decay.csv"
        decay.to_csv(decay_path, index=False)
        table_paths["analysis_ic_decay"] = str(decay_path.as_posix())
        coverage_by_date = pd.DataFrame(
            {
                "factor": ["alpha00001"] * 4,
                "trade_date": pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-02", "2026-01-02"]),
                "non_missing_obs": [80, 82, 75, 70],
                "total_obs": [100, 100, 100, 100],
                "coverage_rate": [0.80, 0.82, 0.75, 0.70],
            }
        )
        coverage_path = run_dir / "analysis_factor_coverage_by_date.parquet"
        coverage_by_date.to_parquet(coverage_path, index=False)
        table_paths["analysis_factor_coverage_by_date"] = str(coverage_path.as_posix())
    if include_visualizations:
        image_dir = run_dir / "visualizations" / "alpha00001"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / "distribution.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        manifest = pd.DataFrame(
            {
                "plot_id": ["alpha00001__distribution", "alpha00001__escape"],
                "scope": ["factor", "factor"],
                "factor": ["alpha00001", "alpha00001"],
                "category": ["distribution", "distribution"],
                "title": ["alpha00001 Distribution", "Escape"],
                "relative_path": ["visualizations/alpha00001/distribution.png", "../outside.png"],
                "width": [1200, 1200],
                "height": [600, 600],
                "sort_order": [10, 20],
                "created_at_utc": ["2026-05-09T00:00:00+00:00", "2026-05-09T00:00:00+00:00"],
                "source": ["test", "test"],
            }
        )
        manifest_path = run_dir / "visualization_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        table_paths["visualization_manifest"] = str(manifest_path.as_posix())

    meta = {
        "analysis_run_id": run_id,
        "alpha_names": ["alpha00001", "alpha00002"],
        "period": period,
        "layers": layers,
        "analysis_dir": str(run_dir.as_posix()),
        "factor_metrics_path": str(metrics_path.as_posix()),
        "table_paths": table_paths,
        "created_at_utc": "2026-05-09T00:00:00+00:00",
        "extra_meta": {
            "closed_loop": True,
            "phase_config": {
                "available_phases": ["train", "val", "test"],
                "feedback_phase": "train",
                "test_default_visible": False,
                "windows": [
                    {"key": "train", "label": "Train", "start": "2016-01-01", "end": "2024-12-31", "available": True, "visible_default": True},
                    {"key": "val", "label": "Val", "start": "2025-01-01", "end": "2025-12-31", "available": True, "visible_default": True},
                    {"key": "test", "label": "Test", "start": "2026-01-01", "end": "2026-01-05", "available": True, "visible_default": False},
                ],
            }
            if include_phase_data
            else {},
            "benchmark_status": {"status": "ok", "code": "000300.SH", "row_count": 5}
            if include_benchmark
            else {"status": "missing", "reason": "index data not found"},
        },
    }
    (run_dir / "analysis_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_dashboard_api_discovers_universes_runs_factors_and_pnl(tmp_path: Path) -> None:
    _write_analysis_run(tmp_path, "cn_all", "analysis_alpha00001-alpha00002_l10_ts1")
    _write_analysis_run(tmp_path, "cn_small", "analysis_alpha00001-alpha00002_l5_ts1", period=5, layers=5)

    client = TestClient(create_app(store_root=tmp_path))

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["universe_count"] == 2

    universes = client.get("/api/universes")
    assert universes.status_code == 200
    assert {row["name"] for row in universes.json()["universes"]} == {"cn_all", "cn_small"}

    runs = client.get("/api/runs", params={"universe": "cn_all"})
    assert runs.status_code == 200
    run = runs.json()["runs"][0]
    assert run["period"] == 1
    assert run["layers"] == 10
    assert run["factor_count"] == 2
    assert run["has_dashboard_metrics"] is True
    assert run["has_portfolio_pnl"] is True
    assert run["has_visualizations"] is False

    factors = client.get(
        "/api/factors",
        params={
            "universe": "cn_all",
            "run_id": "analysis_alpha00001-alpha00002_l10_ts1",
            "q": "alpha00001",
            "sort_by": "score_total",
            "sort_dir": "desc",
        },
    )
    assert factors.status_code == 200
    payload = factors.json()
    assert payload["total"] == 1
    assert payload["factors"][0]["factor"] == "alpha00001"
    assert payload["factors"][0]["expression"] == "rank(close)"

    pnl = client.get(
        "/api/factors/alpha00001/pnl",
        params={"universe": "cn_all", "run_id": "analysis_alpha00001-alpha00002_l10_ts1"},
    )
    assert pnl.status_code == 200
    pnl_payload = pnl.json()
    assert pnl_payload["status"] == "ok"
    assert {row["portfolio"] for row in pnl_payload["rows"]} == {"long_only", "long_10"}
    first_row = pnl_payload["rows"][0]
    assert first_row["return_gross"] == first_row["return"]
    assert first_row["cum_return_gross"] == first_row["cum_return"]
    assert first_row["has_net_pnl"] is False
    assert first_row["return_net"] is None
    assert first_row["cum_return_net"] is None


def test_dashboard_api_exposes_net_pnl_without_filling_old_or_benchmark_rows(tmp_path: Path) -> None:
    run_id = "analysis_alpha00001-alpha00002_l10_ts1"
    _write_analysis_run(tmp_path, "cn_all", run_id, include_benchmark=True)
    run_dir = tmp_path / "cn_all" / "analysis" / "period_1" / run_id
    pnl_path = run_dir / "portfolio_pnl_df.parquet"
    pnl = pd.read_parquet(pnl_path)
    pnl["return_gross"] = pnl["return"]
    pnl["cum_return_gross"] = pnl["cum_return"]
    pnl["transaction_cost"] = [0.0, 0.001, 0.0005, None]
    pnl["return_net"] = [0.01, 0.019, 0.0145, None]
    pnl["has_net_pnl"] = [True, True, True, False]
    pnl["cost_model"] = ["unit_test", "unit_test", "unit_test", None]
    pnl["buy_turnover"] = [0.0, 0.1, 0.1, None]
    pnl["sell_turnover"] = [0.0, 0.1, 0.1, None]
    pnl["cum_return_net"] = (
        pnl.groupby(["factor", "portfolio"], sort=False)["return_net"]
        .transform(lambda s: (1.0 + pd.to_numeric(s, errors="coerce").fillna(0.0)).cumprod() - 1.0)
    )
    pnl.loc[~pnl["has_net_pnl"], "cum_return_net"] = pd.NA
    pnl.to_parquet(pnl_path, index=False)

    client = TestClient(create_app(store_root=tmp_path))
    response = client.get("/api/factors/alpha00001/pnl", params={"universe": "cn_all", "run_id": run_id})

    assert response.status_code == 200
    payload = response.json()
    long_only_rows = [row for row in payload["rows"] if row["portfolio"] == "long_only"]
    assert long_only_rows
    assert all(row["has_net_pnl"] is True for row in long_only_rows)
    assert long_only_rows[1]["return_net"] == pytest.approx(0.019)
    benchmark_rows = [row for row in payload["rows"] if row["portfolio"] == "benchmark"]
    assert benchmark_rows
    assert benchmark_rows[0]["has_net_pnl"] is False
    assert benchmark_rows[0]["return_net"] is None
    assert payload["portfolio_metrics"]["rows_net"]
    assert payload["portfolio_metrics"]["net_available"] is True


def test_dashboard_scoreboard_run_lists_global_factors_and_resolves_details(tmp_path: Path) -> None:
    _write_analysis_run(tmp_path, "cn_all", "analysis_alpha00001-alpha00002_l10_ts1", include_phase_data=True)
    feedback_dir = tmp_path / "cn_all" / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "alpha_name": ["alpha00001", "alpha00002"],
            "expression": ["rank(close)", "ts_rank(volume, 5)"],
            "score_total": [82.0, 38.0],
            "train_score_total": [80.0, 30.0],
            "scoreboard_score": [79.0, 20.0],
            "analysis_run_id": ["analysis_alpha00001-alpha00002_l10_ts1", "analysis_alpha00001-alpha00002_l10_ts1"],
        }
    ).to_csv(feedback_dir / "expression_scoreboard.csv", index=False)

    client = TestClient(create_app(store_root=tmp_path))
    runs = client.get("/api/runs", params={"universe": "cn_all"})
    assert runs.status_code == 200
    run_rows = runs.json()["runs"]
    assert run_rows[0]["run_id"] == "__scoreboard__"
    assert run_rows[0]["is_scoreboard"] is True
    assert run_rows[0]["factor_count"] == 2

    factors = client.get("/api/factors", params={"universe": "cn_all", "run_id": "__scoreboard__"})
    assert factors.status_code == 200
    rows = factors.json()["factors"]
    assert rows[0]["factor"] == "alpha00001"
    assert rows[0]["feedback_score"] == 80.0
    assert rows[0]["analysis_run_id"] == "analysis_alpha00001-alpha00002_l10_ts1"

    pnl = client.get("/api/factors/alpha00001/pnl", params={"universe": "cn_all", "run_id": "__scoreboard__"})
    assert pnl.status_code == 200
    assert pnl.json()["status"] == "ok"
    assert pnl.json()["benchmark_status"]["status"] == "missing"


def test_dashboard_api_handles_missing_pnl_without_error(tmp_path: Path) -> None:
    _write_analysis_run(
        tmp_path,
        "cn_all",
        "analysis_alpha00001-alpha00002_l10_ts1",
        include_pnl=False,
    )

    client = TestClient(create_app(store_root=tmp_path))
    pnl = client.get(
        "/api/factors/alpha00001/pnl",
        params={"universe": "cn_all", "run_id": "analysis_alpha00001-alpha00002_l10_ts1"},
    )
    assert pnl.status_code == 200
    assert pnl.json()["status"] == "missing"
    assert pnl.json()["rows"] == []


def test_dashboard_api_returns_phase_metadata_and_hides_test_by_default(tmp_path: Path) -> None:
    _write_analysis_run(
        tmp_path,
        "cn_all",
        "analysis_alpha00001-alpha00002_l10_ts1",
        include_phase_data=True,
    )

    client = TestClient(create_app(store_root=tmp_path))
    run = client.get("/api/runs", params={"universe": "cn_all"}).json()["runs"][0]
    assert run["has_phase_metrics"] is True
    assert run["has_ic_rows"] is True
    assert run["available_phases"] == ["train", "val", "test"]
    assert run["phase_config"]["feedback_phase"] == "train"

    pnl = client.get(
        "/api/factors/alpha00001/pnl",
        params={"universe": "cn_all", "run_id": "analysis_alpha00001-alpha00002_l10_ts1"},
    )
    assert pnl.status_code == 200
    payload = pnl.json()
    assert payload["status"] == "ok"
    assert payload["phase_config"]["available_phases"] == ["train", "val", "test"]
    assert payload["phase_metrics"]["train"]["score_total"] == 80.0
    assert payload["phase_metrics"]["train"]["long_short_total_return"] == 0.10
    assert payload["phase_metrics"]["train"]["long_short_max_drawdown"] == 0.08
    assert payload["phase_metrics"]["train"]["turnover_long_short_mean"] == 0.25
    assert payload["phase_metrics"]["train"]["margin_long_short"] == 0.004
    assert {row["phase"] for row in payload["rows"]} == {"train", "val"}
    breakdown = payload["portfolio_metrics"]
    assert breakdown["scope_phase"] == "train"
    assert breakdown["benchmark_available"] is False
    breakdown_rows = {row["portfolio"]: row for row in breakdown["rows"]}
    assert {"layer_1", "layer_2", "long_short"}.issubset(set(breakdown_rows))
    assert breakdown_rows["long_short"]["total_return"] == pytest.approx(0.02)
    assert breakdown_rows["long_short"]["turnover"] == pytest.approx(0.25)
    assert breakdown_rows["long_short"]["excess_annualized_return"] is None

    pnl_with_test = client.get(
        "/api/factors/alpha00001/pnl",
        params={
            "universe": "cn_all",
            "run_id": "analysis_alpha00001-alpha00002_l10_ts1",
            "include_test": "true",
        },
    )
    assert pnl_with_test.status_code == 200
    assert "test" in {row["phase"] for row in pnl_with_test.json()["rows"]}

    analysis = client.get(
        "/api/factors/alpha00001/analysis-data",
        params={"universe": "cn_all", "run_id": "analysis_alpha00001-alpha00002_l10_ts1"},
    )
    assert analysis.status_code == 200
    analysis_payload = analysis.json()
    assert analysis_payload["status"] == "ok"
    assert {row["phase"] for row in analysis_payload["ic_series"]} == {"train", "val"}
    assert analysis_payload["ic_series"][0]["cumulative_ic"] == 0.02
    assert analysis_payload["ic_series"][1]["cumulative_ic"] == 0.05
    assert analysis_payload["ic_series"][2]["phase"] == "val"
    assert analysis_payload["ic_series"][2]["cumulative_ic"] == 0.01
    assert {row["phase"] for row in analysis_payload["coverage_series"]} == {"train", "val"}
    assert analysis_payload["coverage_series"][0]["coverage_rate"] == 0.80
    assert {row["phase"] for row in analysis_payload["distribution"]} == {"train", "val"}
    assert {row["phase"] for row in analysis_payload["ic_distribution"]} == {"train", "val"}
    assert {row["phase"] for row in analysis_payload["ic_decay"]} == {"train", "val"}
    assert {row["phase"] for row in analysis_payload["yearly_ic"]} == {"train", "val"}
    assert {row["phase"] for row in analysis_payload["monthly_ic"]} == {"train", "val"}
    layer_returns = {
        (row["phase"], row["portfolio"]): row["terminal_return"]
        for row in analysis_payload["layer_terminal_return"]
    }
    assert layer_returns[("train", "layer_1")] == pytest.approx(0.10)
    assert layer_returns[("val", "layer_1")] == pytest.approx(0.30)
    assert "test" not in analysis_payload["phase_metrics"]

    analysis_with_test = client.get(
        "/api/factors/alpha00001/analysis-data",
        params={
            "universe": "cn_all",
            "run_id": "analysis_alpha00001-alpha00002_l10_ts1",
            "include_test": "true",
        },
    )
    assert analysis_with_test.status_code == 200
    analysis_with_test_payload = analysis_with_test.json()
    assert "test" in {row["phase"] for row in analysis_with_test_payload["ic_series"]}
    assert "test" in {row["phase"] for row in analysis_with_test_payload["coverage_series"]}
    assert "test" in {row["phase"] for row in analysis_with_test_payload["distribution"]}
    assert "test" in {row["phase"] for row in analysis_with_test_payload["ic_distribution"]}
    assert "test" in analysis_with_test_payload["phase_metrics"]
    test_ic_rows = [row for row in analysis_with_test_payload["ic_series"] if row["phase"] == "test"]
    assert test_ic_rows[0]["cumulative_ic"] == 0.04


def test_dashboard_api_phase_feedback_score_only_applies_to_feedback_phase(tmp_path: Path) -> None:
    _write_analysis_run(
        tmp_path,
        "cn_all",
        "analysis_alpha00001-alpha00002_l10_ts1",
        include_phase_data=True,
    )
    phase_path = (
        tmp_path
        / "cn_all"
        / "analysis"
        / "period_1"
        / "analysis_alpha00001-alpha00002_l10_ts1"
        / "phase_metrics_df.csv"
    )
    phase_metrics = pd.read_csv(phase_path)
    phase_metrics = phase_metrics.drop(columns=["feedback_score"])
    phase_metrics.to_csv(phase_path, index=False)

    client = TestClient(create_app(store_root=tmp_path))
    pnl = client.get(
        "/api/factors/alpha00001/pnl",
        params={"universe": "cn_all", "run_id": "analysis_alpha00001-alpha00002_l10_ts1"},
    )

    assert pnl.status_code == 200
    assert pnl.json()["phase_metrics"]["train"]["feedback_score"] == 80.0
    assert "feedback_score" not in pnl.json()["phase_metrics"]["val"]
    assert pnl.json()["phase_metrics"]["val"]["score_total"] == 70.0


def test_dashboard_api_phase_metrics_fill_display_values_from_gross_net_suffixes(tmp_path: Path) -> None:
    run_id = "analysis_alpha00001-alpha00002_l10_ts1"
    _write_analysis_run(tmp_path, "cn_all", run_id, include_phase_data=True)
    phase_path = tmp_path / "cn_all" / "analysis" / "period_1" / run_id / "phase_metrics_df.csv"
    phase_metrics = pd.read_csv(phase_path)
    phase_metrics = phase_metrics.drop(
        columns=[
            "train_long_short_total_return",
            "train_long_short_sharpe_ratio",
            "train_turnover_long_short_mean",
            "train_margin_long_short",
        ]
    )
    phase_metrics["train_long_short_total_return_gross"] = 0.11
    phase_metrics["train_long_short_sharpe_ratio_gross"] = 1.3
    phase_metrics["train_turnover_long_short_mean_gross"] = 0.27
    phase_metrics["train_margin_long_short_gross"] = 0.005
    phase_metrics["train_long_only_total_return_gross"] = 0.20
    phase_metrics["train_long_only_total_return_net"] = 0.12
    phase_metrics["train_score_total_basis"] = "net"
    phase_metrics.to_csv(phase_path, index=False)

    client = TestClient(create_app(store_root=tmp_path))
    response = client.get("/api/factors/alpha00001/pnl", params={"universe": "cn_all", "run_id": run_id})

    assert response.status_code == 200
    train = response.json()["phase_metrics"]["train"]
    assert train["long_short_total_return"] == pytest.approx(0.11)
    assert train["long_short_sharpe_ratio"] == pytest.approx(1.3)
    assert train["turnover_long_short_mean"] == pytest.approx(0.27)
    assert train["margin_long_short"] == pytest.approx(0.005)
    assert train["long_only_total_return"] == pytest.approx(0.12)


def test_dashboard_data_catalog_empty_and_field_catalog_reads_without_recompute(tmp_path: Path) -> None:
    client = TestClient(create_app(store_root=tmp_path))

    empty_families = client.get("/api/data/families")
    empty_fields = client.get("/api/data/fields")
    assert empty_families.status_code == 200
    assert empty_fields.status_code == 200
    assert empty_families.json()["families"] == []
    assert empty_fields.json()["fields"] == []
    assert empty_fields.json()["total"] == 0

    catalog_dir = tmp_path / "data" / "lake" / "meta"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "field_name": ["close", "roe"],
            "factor_family": ["price", "fundamental"],
            "category": ["market", "finance"],
            "source_table": ["market_daily", "finance_indicator"],
            "dtype": ["float64", "float64"],
            "available_start": ["2020-01-01", "2021-01-01"],
            "available_end": ["2026-01-01", "2026-01-01"],
            "coverage_rate": [0.95, 0.50],
            "finite_rate": [0.95, 0.48],
            "is_searchable": [True, False],
        }
    ).to_parquet(catalog_dir / "field_catalog.parquet", index=False)

    client = TestClient(create_app(store_root=tmp_path))
    families = client.get("/api/data/families")
    fields = client.get("/api/data/fields", params={"family": "price", "searchable_only": "true"})

    assert families.status_code == 200
    family_rows = {row["family"]: row for row in families.json()["families"]}
    assert family_rows["price"]["field_count"] == 1
    assert family_rows["price"]["searchable_count"] == 1
    assert fields.status_code == 200
    assert fields.json()["total"] == 1
    assert fields.json()["fields"][0]["field_name"] == "close"


def test_dashboard_data_catalog_does_not_expose_datasource_local_path(tmp_path: Path) -> None:
    catalog_dir = tmp_path / "data" / "lake" / "meta"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "field_name": ["close"],
            "factor_family": ["price"],
            "category": ["price"],
            "source_table": ["market_daily"],
            "dtype": ["float64"],
            "is_searchable": [True],
        }
    ).to_parquet(catalog_dir / "field_catalog.parquet", index=False)

    client = TestClient(create_app(store_root=tmp_path))
    fields = client.get("/api/data/fields")
    families = client.get("/api/data/families")

    assert fields.status_code == 200
    assert families.status_code == 200
    for payload in (fields.json(), families.json()):
        encoded = json.dumps(payload, ensure_ascii=False)
        assert "datasource_config" not in payload
        assert "datasource.local.yaml" not in encoded


def test_dashboard_data_catalog_returns_field_role_metadata_and_compat_defaults(tmp_path: Path) -> None:
    catalog_dir = tmp_path / "data" / "lake" / "meta"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "field_name": ["close", "old_factor"],
            "factor_family": ["price", "other"],
            "category": ["price", "technical"],
            "source_table": ["market_daily", "legacy_table"],
            "dtype": ["float64", "float64"],
            "field_role": ["signal_input", None],
            "available_at": ["same_day_close_available", None],
            "preprocessing_policy": ["expression_wrapper:ts_backfill+winsorize", None],
            "leakage_safe": [True, None],
            "is_searchable": [True, True],
        }
    ).to_parquet(catalog_dir / "field_catalog.parquet", index=False)

    client = TestClient(create_app(store_root=tmp_path))
    response = client.get("/api/data/fields")

    assert response.status_code == 200
    rows = {row["field_name"]: row for row in response.json()["fields"]}
    assert rows["close"]["field_role"] == "signal_input"
    assert rows["close"]["available_at"] == "same_day_close_available"
    assert rows["close"]["preprocessing_policy"] == "expression_wrapper:ts_backfill+winsorize"
    assert rows["close"]["leakage_safe"] is True
    assert rows["old_factor"]["field_role"] == ""
    assert rows["old_factor"]["available_at"] == ""
    assert rows["old_factor"]["preprocessing_policy"] == ""
    assert rows["old_factor"]["leakage_safe"] is False
    assert "rule-inferred" in response.json()["metadata_note"]


def test_dashboard_library_api_handles_missing_and_registry_statuses(tmp_path: Path) -> None:
    client = TestClient(create_app(store_root=tmp_path))
    missing = client.get("/api/library", params={"universe": "cn_all"})
    assert missing.status_code == 200
    assert missing.json()["status"] == "missing"
    assert missing.json()["factors"] == []

    library_dir = tmp_path / "cn_all" / "library"
    library_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "factor": ["alpha_a", "alpha_b", "alpha_c"],
            "analysis_run_id": ["r1", "r2", "r3"],
            "status": ["accepted", "staging", "rejected"],
            "score": [65.0, 55.0, 70.0],
            "score_basis": ["net", "net", "net"],
            "signal_corr": [0.10, 0.20, 0.90],
            "ic_corr": [0.20, 0.30, 0.95],
            "long_only_corr": [0.30, 0.40, 0.10],
            "long_short_corr": [0.25, 0.45, 0.15],
            "max_pnl_corr": [0.30, 0.45, 0.15],
            "nearest_factor_id": ["", "alpha_a", "alpha_a"],
            "rejection_reason": ["", "score_below_min", "signal_corr;ic_corr"],
        }
    ).to_csv(library_dir / "factor_library_registry.csv", index=False)

    response = client.get("/api/library", params={"universe": "cn_all"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["total"] == 3
    rows = {row["factor"]: row for row in payload["factors"]}
    assert rows["alpha_a"]["status"] == "accepted"
    assert rows["alpha_b"]["status"] == "staging"
    assert rows["alpha_c"]["rejection_reason"] == "signal_corr;ic_corr"
    assert rows["alpha_c"]["ic_corr"] == pytest.approx(0.95)


def test_dashboard_library_api_tolerates_legacy_registry_columns(tmp_path: Path) -> None:
    library_dir = tmp_path / "cn_all" / "library"
    library_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "factor": ["alpha_legacy"],
            "status": ["accepted"],
            "score": [62.0],
        }
    ).to_csv(library_dir / "factor_library_registry.csv", index=False)

    client = TestClient(create_app(store_root=tmp_path))
    response = client.get("/api/library", params={"universe": "cn_all"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["factors"][0]["factor"] == "alpha_legacy"
    assert payload["factors"][0]["status"] == "accepted"


def test_dashboard_manual_library_check_and_submit_support_scoreboard_run(tmp_path: Path) -> None:
    run_id = "analysis_alpha00001-alpha00002_l10_ts1"
    _write_analysis_run(tmp_path, "cn_all", run_id, include_phase_data=True)
    save_universe_alpha_values(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-02", "2026-01-02"]),
                "code": ["000001.SZ"] * 4,
                "alpha00001": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        alpha_name="alpha00001",
        base_dir=tmp_path,
        universe_name="cn_all",
    )
    feedback_dir = tmp_path / "cn_all" / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "alpha_name": ["alpha00001"],
            "expression": ["rank(close)"],
            "feedback_score": [80.0],
            "score_total": [82.0],
            "analysis_run_id": [run_id],
        }
    ).to_csv(feedback_dir / "expression_scoreboard.csv", index=False)

    client = TestClient(create_app(store_root=tmp_path))
    missing_status = client.get(
        "/api/factors/alpha00001/library/status",
        params={"universe": "cn_all", "run_id": "__scoreboard__"},
    )
    assert missing_status.status_code == 200
    assert missing_status.json()["library_status"] == "none"

    check = client.post(
        "/api/factors/alpha00001/library/check",
        json={"universe": "cn_all", "run_id": "__scoreboard__"},
    )
    assert check.status_code == 200
    check_payload = check.json()
    assert check_payload["status"] == "ok"
    assert check_payload["can_submit"] is True
    assert check_payload["decision"] == "pass"

    submit = client.post(
        "/api/factors/alpha00001/library/submit",
        json={"universe": "cn_all", "run_id": "__scoreboard__", "submitted_by": "dashboard_test"},
    )
    assert submit.status_code == 200
    submit_payload = submit.json()
    assert submit_payload["submitted"] is True
    assert submit_payload["library_status"] == "accepted"
    assert submit_payload["acceptance_mode"] == "standard"

    accepted_status = client.get(
        "/api/factors/alpha00001/library/status",
        params={"universe": "cn_all", "run_id": "__scoreboard__"},
    )
    assert accepted_status.status_code == 200
    assert accepted_status.json()["library_status"] == "accepted"
    library = client.get("/api/library", params={"universe": "cn_all"})
    row = library.json()["factors"][0]
    assert row["factor"] == "alpha00001"
    assert row["submitted_by"] == "dashboard_test"
    assert row["pnl_artifact_path"]


def test_dashboard_factor_scores_support_old_and_net_score_artifacts(tmp_path: Path) -> None:
    run_id = "analysis_alpha00001-alpha00002_l10_ts1"
    _write_analysis_run(tmp_path, "cn_all", run_id, include_pnl=False)
    run_dir = tmp_path / "cn_all" / "analysis" / "period_1" / run_id
    metrics_path = run_dir / "dashboard_factor_metrics.csv"

    old_metrics = pd.DataFrame(
        {
            "factor": ["alpha_old"],
            "score_total": [52.0],
            "feedback_score": [51.0],
            "expression": ["rank(close)"],
        }
    )
    old_metrics.to_csv(metrics_path, index=False)
    client = TestClient(create_app(store_root=tmp_path))
    old_response = client.get("/api/factors", params={"universe": "cn_all", "run_id": run_id})
    assert old_response.status_code == 200
    assert old_response.json()["factors"][0]["feedback_score"] == 51.0

    new_metrics = pd.DataFrame(
        {
            "factor": ["alpha_net", "alpha_gross"],
            "score_total": [61.0, 58.0],
            "score_total_gross": [80.0, 58.0],
            "score_total_net": [61.0, None],
            "feedback_score_net": [61.0, None],
            "feedback_score_gross": [80.0, 58.0],
            "score_total_basis": ["net", "gross"],
            "expression": ["rank(close)", "rank(volume)"],
        }
    )
    new_metrics.to_csv(metrics_path, index=False)
    client = TestClient(create_app(store_root=tmp_path))
    response = client.get(
        "/api/factors",
        params={"universe": "cn_all", "run_id": run_id, "sort_by": "feedback_score", "sort_dir": "desc"},
    )

    assert response.status_code == 200
    rows = response.json()["factors"]
    assert rows[0]["factor"] == "alpha_net"
    assert rows[0]["feedback_score"] == 61.0
    assert rows[1]["feedback_score"] == 58.0


def test_dashboard_api_merges_shared_benchmark_for_pnl_and_excess(tmp_path: Path) -> None:
    _write_analysis_run(
        tmp_path,
        "cn_all",
        "analysis_alpha00001-alpha00002_l10_ts1",
        include_phase_data=True,
        include_benchmark=True,
    )

    client = TestClient(create_app(store_root=tmp_path))
    run = client.get("/api/runs", params={"universe": "cn_all"}).json()["runs"][0]
    assert run["benchmark_status"]["status"] == "ok"
    assert run["has_benchmark_pnl"] is True

    pnl = client.get(
        "/api/factors/alpha00001/pnl",
        params={"universe": "cn_all", "run_id": "analysis_alpha00001-alpha00002_l10_ts1"},
    )
    assert pnl.status_code == 200
    payload = pnl.json()
    assert "benchmark" in {row["portfolio"] for row in payload["rows"]}
    breakdown = {row["portfolio"]: row for row in payload["portfolio_metrics"]["rows"]}
    assert payload["portfolio_metrics"]["benchmark_available"] is True
    assert "benchmark" in breakdown
    assert breakdown["long_short"]["excess_annualized_return"] is not None


def test_dashboard_api_falls_back_to_factor_metrics_for_historical_runs(tmp_path: Path) -> None:
    _write_analysis_run(
        tmp_path,
        "cn_all",
        "analysis_alpha00001-alpha00002_l10_ts1",
        include_pnl=False,
        include_dashboard_metrics=False,
    )

    client = TestClient(create_app(store_root=tmp_path))
    runs = client.get("/api/runs", params={"universe": "cn_all"}).json()["runs"]
    assert runs[0]["has_dashboard_metrics"] is False
    assert runs[0]["has_factor_metrics"] is True

    factors = client.get(
        "/api/factors",
        params={"universe": "cn_all", "run_id": "analysis_alpha00001-alpha00002_l10_ts1"},
    )
    assert factors.status_code == 200
    assert factors.json()["status"] == "ok"
    assert factors.json()["total"] == 2


def test_dashboard_api_serves_visualization_manifest_and_image(tmp_path: Path) -> None:
    _write_analysis_run(
        tmp_path,
        "cn_all",
        "analysis_alpha00001-alpha00002_l10_ts1",
        include_visualizations=True,
    )

    client = TestClient(create_app(store_root=tmp_path))

    runs = client.get("/api/runs", params={"universe": "cn_all"})
    assert runs.status_code == 200
    assert runs.json()["runs"][0]["has_visualizations"] is True

    visuals = client.get(
        "/api/factors/alpha00001/visualizations",
        params={"universe": "cn_all", "run_id": "analysis_alpha00001-alpha00002_l10_ts1"},
    )
    assert visuals.status_code == 200
    payload = visuals.json()
    assert payload["status"] == "ok"
    assert [image["plot_id"] for image in payload["images"]] == ["alpha00001__distribution"]
    assert payload["images"][0]["url"].startswith("/api/factors/alpha00001/visualizations/")

    image = client.get(
        "/api/factors/alpha00001/visualizations/alpha00001__distribution/image",
        params={"universe": "cn_all", "run_id": "analysis_alpha00001-alpha00002_l10_ts1"},
    )
    assert image.status_code == 200
    assert image.content.startswith(b"\x89PNG")

    missing = client.get(
        "/api/factors/alpha00001/visualizations/does_not_exist/image",
        params={"universe": "cn_all", "run_id": "analysis_alpha00001-alpha00002_l10_ts1"},
    )
    assert missing.status_code == 404

    escape = client.get(
        "/api/factors/alpha00001/visualizations/alpha00001__escape/image",
        params={"universe": "cn_all", "run_id": "analysis_alpha00001-alpha00002_l10_ts1"},
    )
    assert escape.status_code == 404


def test_dashboard_api_reports_missing_visualizations(tmp_path: Path) -> None:
    _write_analysis_run(tmp_path, "cn_all", "analysis_alpha00001-alpha00002_l10_ts1")

    client = TestClient(create_app(store_root=tmp_path))
    visuals = client.get(
        "/api/factors/alpha00001/visualizations",
        params={"universe": "cn_all", "run_id": "analysis_alpha00001-alpha00002_l10_ts1"},
    )

    assert visuals.status_code == 200
    assert visuals.json()["status"] == "missing"
    assert visuals.json()["images"] == []
