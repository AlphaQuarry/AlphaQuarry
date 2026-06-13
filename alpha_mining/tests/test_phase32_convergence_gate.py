from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_gate_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "phase32_convergence_gate.py"
    spec = importlib.util.spec_from_file_location("phase32_convergence_gate_under_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_phase32_gate_metrics_and_status() -> None:
    mod = _load_gate_module()
    df = pd.DataFrame(
        [
            {
                "grid": "bal",
                "overall_win": True,
                "quality_win": True,
                "mechanism_pass": True,
                "delta_topn_score_mean": 0.01,
                "delta_topn_score_median": 0.01,
                "delta_topn_positive_ratio": 0.05,
                "delta_topn_turnover_mean": 0.01,
                "mutation_ratio": 0.08,
                "fragment_cooldown_ratio": 0.20,
            },
            {
                "grid": "bal",
                "overall_win": False,
                "quality_win": True,
                "mechanism_pass": True,
                "delta_topn_score_mean": 0.00,
                "delta_topn_score_median": 0.00,
                "delta_topn_positive_ratio": 0.00,
                "delta_topn_turnover_mean": 0.02,
                "mutation_ratio": 0.09,
                "fragment_cooldown_ratio": 0.30,
            },
        ]
    )
    overall = mod._build_overall_metrics(df)
    args = mod._build_parser().parse_args([])
    args.min_overall_win_rate = 0.40
    args.min_quality_win_rate = 0.80
    args.min_mechanism_pass_rate = 1.00
    args.min_delta_score_mean = 0.0
    args.min_delta_positive_ratio = 0.0
    args.max_delta_turnover_mean = 0.03
    args.min_mutation_ratio = 0.02
    args.max_mutation_ratio = 0.18
    gates = mod._evaluate_gates(overall=overall, args=args)
    assert gates["phase32_converged"] is True


def test_phase32_grid_rank_has_gate_pass_column() -> None:
    mod = _load_gate_module()
    rank_df = pd.DataFrame(
        [
            {
                "grid": "cons",
                "overall_win_rate": 0.0,
                "quality_win_rate": 0.0,
                "mechanism_pass_rate": 1.0,
                "avg_delta_topn_score_mean": 0.0,
                "avg_delta_topn_positive_ratio": 0.0,
                "avg_delta_topn_turnover_mean": 0.0,
                "avg_mutation_ratio": 0.04,
                "avg_fragment_cooldown_ratio": 0.3,
            },
            {
                "grid": "bal",
                "overall_win_rate": 0.7,
                "quality_win_rate": 0.7,
                "mechanism_pass_rate": 1.0,
                "avg_delta_topn_score_mean": 0.01,
                "avg_delta_topn_positive_ratio": 0.03,
                "avg_delta_topn_turnover_mean": 0.01,
                "avg_mutation_ratio": 0.08,
                "avg_fragment_cooldown_ratio": 0.2,
            },
        ]
    )
    args = mod._build_parser().parse_args([])
    ranked = mod._build_grid_rank_with_pass_flags(rank_df=rank_df, args=args)
    assert "gate_pass" in ranked.columns
    assert "rank_score" in ranked.columns
    assert bool(ranked.iloc[0]["gate_pass"]) is True
