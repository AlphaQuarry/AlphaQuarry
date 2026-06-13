from __future__ import annotations

import pandas as pd

from alpha_mining.workflow.factor_library import (
    FactorLibraryConfig,
    check_factor_library_candidate,
    load_factor_library_registry,
    submit_factor_library_candidate,
    submit_factor_library_candidates,
)


def test_factor_library_defaults_accept_only_sixty_score_and_records_staging(
    tmp_path,
) -> None:
    metrics = pd.DataFrame(
        {
            "factor": ["alpha_a", "alpha_mid", "alpha_low"],
            "feedback_score": [60.0, 55.0, 10.0],
            "score_total_basis": ["net", "net", "net"],
        }
    )
    ic = pd.DataFrame(
        {
            "trade_date": pd.date_range("2025-01-01", periods=4),
            "alpha_a_ic": [0.1, 0.2, 0.3, 0.4],
            "alpha_mid_ic": [0.1, -0.1, 0.2, 0.0],
            "alpha_low_ic": [0.3, 0.2, 0.1, 0.0],
        }
    )

    result = submit_factor_library_candidates(
        base_dir=tmp_path,
        universe_name="u1",
        run_id="run1",
        factor_metrics_df=metrics,
        ic_df=ic,
        config=FactorLibraryConfig(enabled=True, min_score=60.0, staging_min_score=50.0),
    )

    registry = load_factor_library_registry(base_dir=tmp_path, universe_name="u1")
    assert result["candidate_count"] == 2
    assert result["accepted_factors"] == ["alpha_a"]
    assert registry["factor"].tolist() == ["alpha_a", "alpha_mid"]
    assert registry["status"].tolist() == ["accepted", "staging"]
    assert "score_below_min" in str(registry.iloc[1]["rejection_reason"])
    assert {
        "signal_corr",
        "ic_corr",
        "long_only_corr",
        "long_short_corr",
        "nearest_factor_id",
    }.issubset(registry.columns)


def test_factor_library_registry_falls_back_to_backup_when_current_csv_is_corrupt(
    tmp_path,
) -> None:
    first = pd.DataFrame({"factor": ["alpha_a"], "feedback_score": [65.0]})
    second = pd.DataFrame({"factor": ["alpha_b"], "feedback_score": [66.0]})

    submit_factor_library_candidates(
        base_dir=tmp_path,
        universe_name="u1",
        run_id="run1",
        factor_metrics_df=first,
        config=FactorLibraryConfig(enabled=True),
    )
    submit_factor_library_candidates(
        base_dir=tmp_path,
        universe_name="u1",
        run_id="run2",
        factor_metrics_df=second,
        config=FactorLibraryConfig(enabled=True),
    )

    registry_path = tmp_path / "u1" / "library" / "factor_library_registry.csv"
    backup_path = registry_path.with_suffix(registry_path.suffix + ".bak")
    assert backup_path.exists()
    registry_path.write_text('"unterminated\n1,2,3', encoding="utf-8")

    recovered = load_factor_library_registry(base_dir=tmp_path, universe_name="u1")

    assert not recovered.empty
    assert "alpha_a" in set(recovered["factor"].astype(str))


def test_factor_library_rejects_or_stages_correlated_new_candidate(tmp_path) -> None:
    first = pd.DataFrame({"factor": ["alpha_a"], "feedback_score": [60.0]})
    second = pd.DataFrame({"factor": ["alpha_c"], "feedback_score": [70.0]})
    ic = pd.DataFrame(
        {
            "trade_date": pd.date_range("2025-01-01", periods=4),
            "alpha_a_ic": [0.1, 0.2, 0.3, 0.4],
            "alpha_c_ic": [0.2, 0.4, 0.6, 0.8],
        }
    )

    submit_factor_library_candidates(
        base_dir=tmp_path,
        universe_name="u1",
        run_id="run1",
        factor_metrics_df=first,
        ic_df=ic,
        config=FactorLibraryConfig(enabled=True),
    )
    submit_factor_library_candidates(
        base_dir=tmp_path,
        universe_name="u1",
        run_id="run2",
        factor_metrics_df=second,
        ic_df=ic,
        config=FactorLibraryConfig(enabled=True),
    )

    registry = load_factor_library_registry(base_dir=tmp_path, universe_name="u1").sort_values(
        ["analysis_run_id", "factor"]
    )
    assert registry["status"].tolist() == ["accepted", "rejected"]
    assert "ic_corr" in str(registry.iloc[1]["rejection_reason"])
    assert registry.iloc[1]["ic_corr"] >= 0.80
    assert registry.iloc[1]["nearest_factor_id"] == "alpha_a"


def test_factor_library_uses_net_score_when_transaction_cost_enabled(tmp_path) -> None:
    metrics = pd.DataFrame(
        {
            "factor": ["alpha_net_bad"],
            "feedback_score": [70.0],
            "feedback_score_net": [45.0],
            "score_total": [70.0],
            "score_total_net": [45.0],
            "score_total_basis": ["net"],
        }
    )

    result = submit_factor_library_candidates(
        base_dir=tmp_path,
        universe_name="u1",
        run_id="run1",
        factor_metrics_df=metrics,
        config=FactorLibraryConfig(
            enabled=True,
            min_score=60.0,
            staging_min_score=50.0,
            transaction_cost_enabled=True,
        ),
    )

    registry = load_factor_library_registry(base_dir=tmp_path, universe_name="u1")
    assert result["candidate_count"] == 0
    assert registry.empty


def test_factor_library_records_long_only_and_long_short_pnl_correlations(
    tmp_path,
) -> None:
    first = pd.DataFrame({"factor": ["alpha_a"], "feedback_score_net": [65.0]})
    second = pd.DataFrame({"factor": ["alpha_b"], "feedback_score_net": [70.0]})
    dates = pd.date_range("2025-01-01", periods=4)
    pnl = pd.DataFrame(
        {
            "factor": ["alpha_a"] * 8 + ["alpha_b"] * 8,
            "portfolio": (["long_only"] * 4 + ["long_short"] * 4) * 2,
            "trade_date": list(dates) * 4,
            "return": [0.01, 0.02, 0.03, 0.04, 0.04, 0.03, 0.02, 0.01] * 2,
            "return_net": [0.009, 0.019, 0.029, 0.039, None, None, None, None] * 2,
            "has_net_pnl": [True, True, True, True, False, False, False, False] * 2,
        }
    )

    submit_factor_library_candidates(
        base_dir=tmp_path,
        universe_name="u1",
        run_id="run1",
        factor_metrics_df=first,
        portfolio_pnl_df=pnl,
        config=FactorLibraryConfig(enabled=True),
    )
    submit_factor_library_candidates(
        base_dir=tmp_path,
        universe_name="u1",
        run_id="run2",
        factor_metrics_df=second,
        portfolio_pnl_df=pnl,
        config=FactorLibraryConfig(enabled=True),
    )

    registry = load_factor_library_registry(base_dir=tmp_path, universe_name="u1").sort_values(
        ["analysis_run_id", "factor"]
    )
    row = registry.iloc[1]
    assert row["long_only_corr"] == 1.0
    assert row["long_short_corr"] == 1.0
    assert row["max_pnl_corr"] == 1.0
    assert row["nearest_factor_id"] == "alpha_a"
    assert row["status"] == "rejected"


def test_factor_library_noops_when_disabled(tmp_path) -> None:
    metrics = pd.DataFrame({"factor": ["alpha_a"], "feedback_score": [90.0]})

    result = submit_factor_library_candidates(
        base_dir=tmp_path,
        universe_name="u1",
        run_id="run1",
        factor_metrics_df=metrics,
        config=FactorLibraryConfig(enabled=False),
    )

    assert result["enabled"] is False
    assert result["accepted_factors"] == []
    assert load_factor_library_registry(base_dir=tmp_path, universe_name="u1").empty


def test_factor_library_stages_when_existing_assets_require_missing_corr_series(
    tmp_path,
) -> None:
    first = pd.DataFrame({"factor": ["alpha_a"], "feedback_score": [65.0]})
    second = pd.DataFrame({"factor": ["alpha_b"], "feedback_score": [70.0]})

    submit_factor_library_candidates(
        base_dir=tmp_path,
        universe_name="u1",
        run_id="run1",
        factor_metrics_df=first,
        config=FactorLibraryConfig(enabled=True),
    )
    submit_factor_library_candidates(
        base_dir=tmp_path,
        universe_name="u1",
        run_id="run2",
        factor_metrics_df=second,
        config=FactorLibraryConfig(enabled=True),
    )

    registry = load_factor_library_registry(base_dir=tmp_path, universe_name="u1").sort_values(
        ["analysis_run_id", "factor"]
    )
    row = registry.iloc[1]
    assert row["status"] == "staging"
    assert "missing_ic_corr" in str(row["library_status_reason"])


def test_manual_check_submit_records_compact_series_and_schema(tmp_path) -> None:
    metrics = pd.DataFrame(
        {
            "factor": ["alpha_a"],
            "feedback_score_net": [66.0],
            "score_total_net": [66.0],
            "feedback_score_gross": [70.0],
            "score_total_gross": [70.0],
            "long_only_sharpe_ratio": [1.1],
            "long_short_sharpe_ratio": [0.7],
            "expression": ["rank(close)"],
        }
    )
    dates = pd.date_range("2025-01-01", periods=4)
    signal = pd.DataFrame({"date": dates, "code": ["000001.SZ"] * 4, "alpha_a": [1.0, 2.0, 3.0, 4.0]})
    ic = pd.DataFrame({"trade_date": dates, "alpha_a_ic": [0.1, 0.2, 0.3, 0.4]})
    pnl = pd.DataFrame(
        {
            "factor": ["alpha_a"] * 8,
            "portfolio": ["long_only"] * 4 + ["long_short"] * 4,
            "trade_date": list(dates) * 2,
            "return": [0.01, 0.02, 0.03, 0.04, 0.02, 0.01, 0.02, 0.03],
            "return_gross": [0.011, 0.021, 0.031, 0.041, 0.02, 0.01, 0.02, 0.03],
            "return_net": [0.009, 0.019, 0.029, 0.039, None, None, None, None],
            "has_net_pnl": [True, True, True, True, False, False, False, False],
        }
    )

    check = check_factor_library_candidate(
        base_dir=tmp_path,
        universe_name="u1",
        run_id="run1",
        factor="alpha_a",
        factor_metrics_df=metrics,
        ic_df=ic,
        portfolio_pnl_df=pnl,
        signal_df=signal,
        config=FactorLibraryConfig(enabled=True),
    )
    assert check["status"] == "ok"
    assert check["can_submit"] is True
    assert check["decision"] == "pass"
    assert check["score_basis"] == "net"

    result = submit_factor_library_candidate(
        base_dir=tmp_path,
        universe_name="u1",
        run_id="run1",
        factor="alpha_a",
        factor_metrics_df=metrics,
        ic_df=ic,
        portfolio_pnl_df=pnl,
        signal_df=signal,
        config=FactorLibraryConfig(enabled=True),
        submitted_by="unit_test",
    )

    assert result["status"] == "ok"
    assert result["submitted"] is True
    registry = load_factor_library_registry(base_dir=tmp_path, universe_name="u1")
    row = registry.iloc[0]
    assert row["status"] == "accepted"
    assert row["acceptance_mode"] == "standard"
    assert row["submitted_by"] == "unit_test"
    assert row["source_score_total_net"] == 66.0
    pnl_series = pd.read_parquet(row["pnl_artifact_path"])
    assert "portfolio" in pnl_series.columns
    assert set(pnl_series["portfolio"]) == {"long_only", "long_short"}


def test_manual_submit_blocks_when_signal_artifact_cannot_be_saved(tmp_path) -> None:
    metrics = pd.DataFrame({"factor": ["alpha_a"], "feedback_score": [66.0]})

    result = submit_factor_library_candidate(
        base_dir=tmp_path,
        universe_name="u1",
        run_id="run1",
        factor="alpha_a",
        factor_metrics_df=metrics,
        config=FactorLibraryConfig(enabled=True),
        submitted_by="unit_test",
    )

    assert result["status"] == "blocked"
    assert result["submitted"] is False
    assert "missing_signal_artifact" in str(result["reason"])


def test_manual_submit_allows_high_corr_sharpe_override(tmp_path) -> None:
    dates = pd.date_range("2025-01-01", periods=6)
    base_pnl = pd.DataFrame(
        {
            "factor": ["alpha_peer"] * 6,
            "portfolio": ["long_only"] * 6,
            "trade_date": dates,
            "return": [0.01, -0.01, 0.01, -0.01, 0.01, -0.01],
            "return_gross": [0.01, -0.01, 0.01, -0.01, 0.01, -0.01],
        }
    )
    submit_factor_library_candidate(
        base_dir=tmp_path,
        universe_name="u1",
        run_id="run_peer",
        factor="alpha_peer",
        factor_metrics_df=pd.DataFrame({"factor": ["alpha_peer"], "feedback_score": [66.0]}),
        ic_df=pd.DataFrame({"trade_date": dates, "alpha_peer_ic": [1, 2, 3, 4, 5, 6]}),
        portfolio_pnl_df=base_pnl,
        signal_df=pd.DataFrame({"date": dates, "code": ["000001.SZ"] * 6, "alpha_peer": [1, 2, 3, 4, 5, 6]}),
        config=FactorLibraryConfig(enabled=True),
    )

    candidate_pnl = pd.DataFrame(
        {
            "factor": ["alpha_new"] * 6,
            "portfolio": ["long_only"] * 6,
            "trade_date": dates,
            "return": [0.01, 0.012, 0.011, 0.013, 0.012, 0.014],
            "return_gross": [0.01, 0.012, 0.011, 0.013, 0.012, 0.014],
        }
    )
    result = submit_factor_library_candidate(
        base_dir=tmp_path,
        universe_name="u1",
        run_id="run_new",
        factor="alpha_new",
        factor_metrics_df=pd.DataFrame({"factor": ["alpha_new"], "feedback_score": [70.0]}),
        ic_df=pd.DataFrame({"trade_date": dates, "alpha_new_ic": [1, 2, 3, 4, 5, 6]}),
        portfolio_pnl_df=candidate_pnl,
        signal_df=pd.DataFrame({"date": dates, "code": ["000001.SZ"] * 6, "alpha_new": [1, 2, 3, 4, 5, 6]}),
        config=FactorLibraryConfig(enabled=True, sharpe_override_threshold=0.15),
    )

    assert result["submitted"] is True
    assert result["acceptance_mode"] == "sharpe_override"
    registry = load_factor_library_registry(base_dir=tmp_path, universe_name="u1")
    row = registry[registry["factor"] == "alpha_new"].iloc[0]
    assert row["status"] == "accepted"
    assert row["high_corr_peer_count"] == 1
    assert row["override_reason"] == "high_corr_but_sharpe_improved"


def test_submit_reproduce_fallback(tmp_path, monkeypatch) -> None:
    """Test that Submit attempts reproduce when signal_df is missing."""
    dates = pd.date_range("2025-01-01", periods=5)
    metrics = pd.DataFrame({"factor": ["alpha_a"], "feedback_score": [70.0]})

    # Mock reproduce to return a valid signal
    def fake_reproduce(**kwargs):
        return {
            "output_df": pd.DataFrame({"date": dates, "code": ["000001.SZ"] * 5, "alpha_a": [1, 2, 3, 4, 5]}),
            "strict_reproducibility": False,
            "reproduce_source_mode": "duckdb_fallback",
            "reproduce_warning": "",
            "saved": {"path": ""},
        }

    import alpha_mining.workflow.reproduce as reproduce_module

    monkeypatch.setattr(reproduce_module, "reproduce_alpha_by_name", fake_reproduce)

    result = submit_factor_library_candidate(
        base_dir=tmp_path,
        universe_name="u1",
        run_id="run1",
        factor="alpha_a",
        factor_metrics_df=metrics,
        signal_df=None,
        ic_df=pd.DataFrame({"trade_date": dates, "alpha_a_ic": [0.1, 0.2, 0.3, 0.4, 0.5]}),
        config=FactorLibraryConfig(enabled=True),
        submitted_by="unit_test",
    )

    # Should attempt reproduce and succeed
    assert result["submitted"] is True
    assert result["row"].get("signal_source") == "reproduced"


def test_submit_reproduce_failure_stays_staging(tmp_path, monkeypatch) -> None:
    """Test that Submit stays staging when reproduce fails."""
    metrics = pd.DataFrame({"factor": ["alpha_a"], "feedback_score": [70.0]})

    # Mock reproduce to fail
    def fake_reproduce(**kwargs):
        raise ValueError("expression not found")

    import alpha_mining.workflow.reproduce as reproduce_module

    monkeypatch.setattr(reproduce_module, "reproduce_alpha_by_name", fake_reproduce)

    result = submit_factor_library_candidate(
        base_dir=tmp_path,
        universe_name="u1",
        run_id="run1",
        factor="alpha_a",
        factor_metrics_df=metrics,
        signal_df=None,
        config=FactorLibraryConfig(enabled=True),
        submitted_by="unit_test",
    )

    assert result["status"] == "blocked"
    assert result["submitted"] is False
    assert "missing_signal_artifact" in str(result["reason"])
