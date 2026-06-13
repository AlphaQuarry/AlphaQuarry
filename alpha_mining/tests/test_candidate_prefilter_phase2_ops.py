from __future__ import annotations

import pandas as pd

from alpha_mining.mining.candidate_prefilter import CandidatePrefilter
from alpha_mining.mining.explore import DeepExploreConfig, build_operator_search_space
from alpha_mining.mining.expression_layers import (
    LayeredBuilderConfig,
    LayeredExpressionBuilder,
)
from alpha_mining.mining.field_universe import build_field_universe
from alpha_mining.panel_store import PanelStore


def _panel_store() -> PanelStore:
    rows = []
    for date in pd.date_range("2024-01-01", periods=6):
        for code, industry in [("A", "bank"), ("B", "tech"), ("C", "tech")]:
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "close": 10.0 + float(len(code)),
                    "amount": 100.0,
                    "circ_mv": 1e9,
                    "industry": industry,
                    "pct_chg": 0.01,
                    "target": 1.0,
                }
            )
    return PanelStore.from_long_frame(pd.DataFrame(rows), group_fields=["industry"])


def test_prefilter_accepts_new_phase2_signatures_and_still_rejects_leakage() -> None:
    kinds = {"close": "scalar", "amount": "scalar", "industry": "group"}
    prefilter = CandidatePrefilter(field_kinds=kinds, max_operator_count=12, max_field_count=4, max_depth=8)

    assert prefilter.check("quantile(close, 'gaussian', 1.0)").passed
    assert prefilter.check("truncate(close, 0.02)").passed
    assert prefilter.check("ts_covariance(close, amount, 5)").passed
    assert prefilter.check("ts_count_nans(close, 5)").passed
    assert prefilter.check("group_median(close, industry)").passed
    assert prefilter.check("group_scale(close, industry)").passed
    assert prefilter.check("trade_when_hold(greater(close, 0), close, less(close, 0))").passed

    leak = prefilter.check("rank(pct_chg)")
    assert not leak.passed
    assert leak.reject_reason.startswith("unknown_field")


def test_operator_only_generation_excludes_stateful_ops_by_default() -> None:
    exprs = build_operator_search_space(
        available_fields={"close", "amount"},
        available_groups={"industry"},
        config=DeepExploreConfig(max_candidates=120, random_seed=7, enable_stateful_phase2_ops=False),
    )
    text = "\n".join(expr for _, expr in exprs)
    assert "hump(" not in text
    assert "trade_when_hold(" not in text


def test_operator_only_generation_includes_stateful_ops_when_enabled() -> None:
    exprs = build_operator_search_space(
        available_fields={"close", "amount"},
        available_groups={"industry"},
        config=DeepExploreConfig(max_candidates=120, random_seed=7, enable_stateful_phase2_ops=True),
    )
    text = "\n".join(expr for _, expr in exprs)
    assert "hump(" in text
    assert "trade_when_hold(" in text


def test_layered_builder_respects_stateful_phase2_switch() -> None:
    universe = build_field_universe(
        _panel_store(),
        explicit_include_fields=["close", "amount", "circ_mv"],
        group_fields=["industry"],
    )
    off_cfg = LayeredBuilderConfig(
        max_order=4,
        max_candidates=120,
        layer_budgets={"L0": 3, "L1": 6, "L2": 6, "L3": 12, "L4": 0},
        windows=(5,),
        enable_stateful_phase2_ops=False,
        random_seed=3,
    )
    on_cfg = LayeredBuilderConfig(
        max_order=4,
        max_candidates=120,
        layer_budgets={"L0": 3, "L1": 6, "L2": 6, "L3": 12, "L4": 0},
        windows=(5,),
        enable_stateful_phase2_ops=True,
        random_seed=3,
    )

    off_exprs = [x.expression for x in LayeredExpressionBuilder().build(universe, feedback_hints={}, config=off_cfg)]
    on_exprs = [x.expression for x in LayeredExpressionBuilder().build(universe, feedback_hints={}, config=on_cfg)]

    assert all("hump(" not in expr and "trade_when_hold(" not in expr for expr in off_exprs)
    assert any(("hump(" in expr or "trade_when_hold(" in expr) for expr in on_exprs)
