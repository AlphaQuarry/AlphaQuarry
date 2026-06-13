from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pandas as pd

from alpha_mining.config import AlphaMiningConfig, AlphaSimulationConfig
from alpha_mining.mining.explore import DeepExploreConfig
from alpha_mining.mining.candidate_planner import _reserve_feedback_mutation_candidates
from alpha_mining.panel_store import PanelStore
from alpha_mining.workflow.closed_loop import ClosedLoopConfig, _generate_candidates


def _panel_store() -> PanelStore:
    rows = []
    for date in pd.date_range("2024-01-01", periods=12):
        for code, industry, sector, subindustry in [
            ("A", "bank", "financial", "bank_large"),
            ("B", "tech", "growth", "software"),
            ("C", "tech", "growth", "hardware"),
        ]:
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "close": 10.0 + float(len(code)),
                    "open": 9.8 + float(len(code)),
                    "amount": 100.0 + float(len(code)),
                    "volume": 1000.0 + float(len(code)),
                    "circ_mv": 1e9 + 1e7 * float(len(code)),
                    "industry": industry,
                    "sector": sector,
                    "subindustry": subindustry,
                    "pct_chg": 0.01,
                    "target": 1.0,
                    "universe": 1.0,
                }
            )
    return PanelStore.from_long_frame(
        pd.DataFrame(rows),
        group_fields=["industry", "sector", "subindustry"],
    )


def test_generate_candidates_includes_feedback_mutation_v2_when_enabled() -> None:
    base_dir = Path("data") / f"_feedback_mutation_v2_{uuid.uuid4().hex}"
    universe = "ut_mutation_v2"
    feedback_dir = base_dir / universe / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "factor": ["alpha_0001", "alpha_0002"],
            "expression": ["ts_rank(close, 22)", "group_rank(amount, industry)"],
            "score_total": [1.0, 0.4],
            "batch_id": ["batch_1", "batch_1"],
        }
    ).to_csv(feedback_dir / "expression_scoreboard.csv", index=False)

    cfg = ClosedLoopConfig(
        universe_base_dir=str(base_dir),
        universe_name=universe,
        group_fields=("industry", "sector", "subindustry"),
        include_fields=("close", "open", "amount", "volume", "circ_mv"),
        search_mode="operator_only",
        max_eval_expressions=12,
        enable_feedback_mutation=True,
        mutation_budget_ratio=0.15,
        mutation_max_children_per_parent=3,
        deep_explore_config=DeepExploreConfig(max_candidates=40, random_seed=7),
        mining_config=AlphaMiningConfig(simulation=AlphaSimulationConfig(delay=1, universe="universe")),
    )
    try:
        expressions, meta = _generate_candidates(_panel_store(), cfg)
        assert expressions
        candidates = pd.read_csv(meta["candidates_path"])
        assert {
            "mutation_type",
            "fragment_hash",
            "feedback_source",
            "parent_hash",
            "fragment_score",
            "parent_score",
            "mutation_type_score",
        } <= set(candidates.columns)
        mutated = candidates[candidates["source"].astype(str) == "feedback_mutation_v2"]
        assert not mutated.empty
        assert mutated["mutation_type"].astype(str).str.len().gt(0).all()
        assert mutated["feedback_source"].astype(str).eq("feedback_mutation_v2").all()
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_generate_candidates_keeps_old_mode_when_feedback_mutation_disabled() -> None:
    base_dir = Path("data") / f"_feedback_mutation_v2_off_{uuid.uuid4().hex}"
    universe = "ut_mutation_v2_off"
    feedback_dir = base_dir / universe / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "factor": ["alpha_0001"],
            "expression": ["ts_rank(close, 22)"],
            "score_total": [1.0],
            "batch_id": ["batch_1"],
        }
    ).to_csv(feedback_dir / "expression_scoreboard.csv", index=False)

    cfg = ClosedLoopConfig(
        universe_base_dir=str(base_dir),
        universe_name=universe,
        group_fields=("industry", "sector", "subindustry"),
        include_fields=("close", "open", "amount", "volume", "circ_mv"),
        search_mode="operator_only",
        max_eval_expressions=12,
        enable_feedback_mutation=False,
        deep_explore_config=DeepExploreConfig(max_candidates=40, random_seed=7),
        mining_config=AlphaMiningConfig(simulation=AlphaSimulationConfig(delay=1, universe="universe")),
    )
    try:
        expressions, meta = _generate_candidates(_panel_store(), cfg)
        assert expressions
        candidates = pd.read_csv(meta["candidates_path"])
        assert "feedback_mutation_v2" not in set(candidates["source"].astype(str))
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_generate_candidates_enforces_mutation_min_selected_count_when_enabled() -> None:
    base_dir = Path("data") / f"_feedback_mutation_v2_min_{uuid.uuid4().hex}"
    universe = "ut_mutation_v2_min"
    feedback_dir = base_dir / universe / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "factor": ["alpha_0001", "alpha_0002"],
            "expression": ["ts_rank(close, 22)", "group_rank(amount, industry)"],
            "score_total": [1.0, 0.8],
            "batch_id": ["batch_1", "batch_1"],
        }
    ).to_csv(feedback_dir / "expression_scoreboard.csv", index=False)

    cfg = ClosedLoopConfig(
        universe_base_dir=str(base_dir),
        universe_name=universe,
        group_fields=("industry", "sector", "subindustry"),
        include_fields=("close", "open", "amount", "volume", "circ_mv"),
        search_mode="operator_only",
        max_eval_expressions=16,
        enable_feedback_mutation=True,
        mutation_budget_ratio=0.20,
        mutation_max_children_per_parent=4,
        mutation_min_selected_count=2,
        deep_explore_config=DeepExploreConfig(max_candidates=40, random_seed=7),
        mining_config=AlphaMiningConfig(simulation=AlphaSimulationConfig(delay=1, universe="universe")),
    )
    try:
        expressions, meta = _generate_candidates(_panel_store(), cfg)
        assert expressions
        candidates = pd.read_csv(meta["candidates_path"])
        selected = candidates[candidates["selection_bucket"].astype(str).str.len().gt(0)]
        mutation_selected = int((selected["source"].astype(str) == "feedback_mutation_v2").sum())
        available_mutation = int(
            (
                (candidates["source"].astype(str) == "feedback_mutation_v2")
                & (candidates["prefilter_status"].astype(str) == "pass")
            ).sum()
        )
        expected_min = min(2, available_mutation, int(cfg.max_eval_expressions))
        assert mutation_selected >= expected_min
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_reserve_feedback_mutation_candidates_respects_min_count() -> None:
    ranked = pd.DataFrame(
        [
            {"candidate_id": "c1", "source": "op_signature", "candidate_score": 1.0},
            {"candidate_id": "c2", "source": "op_signature", "candidate_score": 0.9},
            {
                "candidate_id": "c3",
                "source": "feedback_mutation_v2",
                "candidate_score": 0.8,
            },
            {"candidate_id": "c4", "source": "op_signature", "candidate_score": 0.7},
            {
                "candidate_id": "c5",
                "source": "feedback_mutation_v2",
                "candidate_score": 0.6,
            },
        ]
    )
    selected = _reserve_feedback_mutation_candidates(
        ranked_df=ranked,
        max_eval=4,
        min_count=2,
        min_ratio=0.0,
    )
    mutation_selected = int((selected["source"].astype(str) == "feedback_mutation_v2").sum())
    assert mutation_selected >= 2
    assert len(selected) == 4
