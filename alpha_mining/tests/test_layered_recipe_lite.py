from __future__ import annotations

import pandas as pd

from alpha_mining.mining.candidate_prefilter import CandidatePrefilter
from alpha_mining.mining.expression_layers import (
    LayeredBuilderConfig,
    LayeredExpressionBuilder,
)
from alpha_mining.mining.field_universe import build_field_universe
from alpha_mining.mining.layered_recipes import (
    build_layered_recipe_candidates,
    build_role_pair_candidates,
)
from alpha_mining.panel_store import PanelStore


def _store() -> PanelStore:
    rows = []
    for date in pd.date_range("2024-01-01", periods=6):
        for code, industry in [("A", "bank"), ("B", "tech")]:
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "close": 10.0,
                    "amount": 100.0,
                    "volume": 1000.0,
                    "turnover_rate": 0.04,
                    "circ_mv": 1e9,
                    "moneyflow_buy_lg_amount": 12.0,
                    "moneyflow_sell_lg_amount": 7.0,
                    "pe_ttm": 13.0,
                    "cyq_winner_rate": 0.6,
                    "industry": industry,
                }
            )
    return PanelStore.from_long_frame(pd.DataFrame(rows), group_fields=["industry"])


def test_recipe_lite_generates_moneyflow_imbalance_when_buy_sell_exist() -> None:
    universe = build_field_universe(_store(), group_fields=["industry"])

    recipes = build_layered_recipe_candidates(universe, windows=(5, 22), max_total=20, max_per_family=8)

    moneyflow = [item for item in recipes if item.metadata.get("recipe_family") == "moneyflow_imbalance"]
    assert moneyflow
    assert any(
        "moneyflow_buy_lg_amount" in item.expression and "moneyflow_sell_lg_amount" in item.expression
        for item in moneyflow
    )
    assert all(item.metadata.get("recipe_id") for item in moneyflow)


def test_recipe_lite_skips_missing_family_without_error() -> None:
    store = PanelStore.from_long_frame(
        pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3),
                "code": ["A", "A", "A"],
                "close": [1.0, 2.0, 3.0],
                "amount": [10.0, 20.0, 30.0],
            }
        )
    )
    universe = build_field_universe(store)

    recipes = build_layered_recipe_candidates(universe, windows=(5,), max_total=20, max_per_family=8)

    assert recipes
    assert not [item for item in recipes if item.metadata.get("recipe_family") == "moneyflow_imbalance"]


def test_role_aware_pair_candidates_have_metadata_and_respect_cap() -> None:
    universe = build_field_universe(_store(), group_fields=["industry"])

    pairs = build_role_pair_candidates(universe, windows=(5,), max_total=3, cross_family_pair_ratio=0.5)

    assert 1 <= len(pairs) <= 3
    assert all(item.metadata.get("role_pair_type") for item in pairs)
    assert any(item.metadata.get("role_pair_type") == "moneyflow_buy_sell" for item in pairs)


def test_recipe_and_role_pair_candidates_are_prefilter_compatible() -> None:
    universe = build_field_universe(_store(), group_fields=["industry"])
    prefilter = CandidatePrefilter(
        field_kinds=universe.kind_map(),
        max_operator_count=12,
        max_field_count=4,
        max_depth=6,
        reject_naked_division=True,
    )

    candidates = build_layered_recipe_candidates(universe, windows=(5,), max_total=12, max_per_family=4)
    candidates += build_role_pair_candidates(universe, windows=(5,), max_total=6, cross_family_pair_ratio=0.5)

    assert candidates
    assert all(prefilter.check(item.expression).passed for item in candidates)


def test_layered_builder_includes_recipe_and_role_pair_metadata() -> None:
    universe = build_field_universe(_store(), group_fields=["industry"])
    cfg = LayeredBuilderConfig(
        max_order=2,
        max_candidates=120,
        layer_budgets={"L0": 12, "L1": 56, "L2": 48, "L3": 0, "L4": 0},
        windows=(5,),
        layer_recipe_max_total=10,
        layer_recipe_max_per_family=4,
        layer_role_pair_max_total=6,
    )

    out = LayeredExpressionBuilder().build(universe, feedback_hints={}, config=cfg)

    assert any(item.metadata.get("recipe_family") for item in out)
    assert any(item.metadata.get("role_pair_type") for item in out)
