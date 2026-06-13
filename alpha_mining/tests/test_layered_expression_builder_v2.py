from __future__ import annotations

import json

import pandas as pd

from alpha_mining.mining.candidate_prefilter import CandidatePrefilter
from alpha_mining.mining.explore import DeepExploreConfig
from alpha_mining.mining.fragment_mutation import MutationConfig
from alpha_mining.mining.expression_layers import (
    LayeredBuilderConfig,
    LayeredExpressionBuilder,
)
from alpha_mining.mining.field_universe import build_field_universe
from alpha_mining.panel_store import PanelStore
from alpha_mining.workflow.closed_loop import ClosedLoopConfig


def _panel_store() -> PanelStore:
    rows = []
    for date in pd.date_range("2024-01-01", periods=8):
        for code, industry in [("A", "bank"), ("B", "tech"), ("C", "steel")]:
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "open": 9.8,
                    "high": 10.5,
                    "low": 9.5,
                    "close": 10.0,
                    "amount": 100.0,
                    "volume": 1000.0,
                    "turnover_rate": 0.03,
                    "volume_ratio": 1.2,
                    "circ_mv": 1e9,
                    "total_mv": 2e9,
                    "moneyflow_net_amount": 12.0,
                    "moneyflow_buy_lg_amount": 7.0,
                    "moneyflow_sell_lg_amount": 3.0,
                    "pe": 12.0,
                    "pe_ttm": 13.0,
                    "pb": 1.5,
                    "dv_ttm": 0.02,
                    "cyq_winner_rate": 0.6,
                    "tech_rsi_qfq_6": 55.0,
                    "pct_chg": 0.01,
                    "industry": industry,
                    "universe": 1,
                }
            )
    return PanelStore.from_long_frame(pd.DataFrame(rows), group_fields=["industry"])


def test_layered_builder_generates_l0_to_l4_with_lineage_and_budgets() -> None:
    universe = build_field_universe(
        _panel_store(),
        explicit_include_fields=[
            "moneyflow_net_amount",
            "moneyflow_buy_lg_amount",
            "amount",
            "circ_mv",
        ],
        group_fields=["industry"],
    )
    cfg = LayeredBuilderConfig(
        max_order=4,
        max_candidates=48,
        layer_budgets={"L0": 4, "L1": 10, "L2": 12, "L3": 8, "L4": 8},
        windows=(5,),
        random_seed=3,
    )

    out = LayeredExpressionBuilder().build(universe, feedback_hints={"layer_weights": {"L3": 1.0}}, config=cfg)

    layers = {item.layer for item in out}
    assert {"L0", "L1", "L2", "L3", "L4"} <= layers
    assert len(out) <= cfg.max_candidates
    for layer, budget in cfg.layer_budgets.items():
        assert sum(1 for item in out if item.layer == layer) <= budget
    assert all(item.builder_source == "layered_v2" for item in out)
    assert all(item.parent_hash for item in out if item.layer_order > 0)
    assert any(item.layer == "L3" and item.expression.startswith("if_else(") for item in out)
    assert any(item.layer == "L4" and "neutralize" in item.expression for item in out)


def test_layered_builder_respects_max_order() -> None:
    universe = build_field_universe(
        _panel_store(),
        explicit_include_fields=["moneyflow_net_amount", "amount"],
        group_fields=["industry"],
    )
    cfg = LayeredBuilderConfig(
        max_order=2,
        max_candidates=40,
        layer_budgets={"L0": 4, "L1": 8, "L2": 8, "L3": 8, "L4": 8},
        windows=(5,),
    )

    out = LayeredExpressionBuilder().build(universe, feedback_hints={}, config=cfg)

    assert out
    assert {item.layer for item in out} <= {"L0", "L1", "L2"}


def test_layered_builder_outputs_parser_and_signature_compatible_expressions() -> None:
    universe = build_field_universe(
        _panel_store(),
        explicit_include_fields=[
            "moneyflow_net_amount",
            "moneyflow_buy_lg_amount",
            "amount",
            "circ_mv",
        ],
        group_fields=["industry"],
    )
    cfg = LayeredBuilderConfig(
        max_order=4,
        max_candidates=64,
        layer_budgets={"L0": 4, "L1": 12, "L2": 16, "L3": 8, "L4": 8},
        windows=(5,),
    )
    prefilter = CandidatePrefilter(
        field_kinds=universe.kind_map(),
        max_operator_count=12,
        max_field_count=4,
        max_depth=6,
    )

    out = LayeredExpressionBuilder().build(universe, feedback_hints={}, config=cfg)
    passed_layers = {item.layer for item in out if prefilter.check(item.expression).passed}

    assert {"L0", "L1", "L2", "L3", "L4"} <= passed_layers


def test_layered_defaults_use_layered_v2_and_include_132_window() -> None:
    assert ClosedLoopConfig().search_mode == "layered_v2"
    assert 132 in DeepExploreConfig().windows
    assert 132 in LayeredBuilderConfig().windows
    assert 132 in MutationConfig().windows


def test_layered_builder_generates_multi_family_gates_with_metadata() -> None:
    universe = build_field_universe(
        _panel_store(),
        explicit_include_fields=[
            "amount",
            "volume",
            "turnover_rate",
            "moneyflow_net_amount",
            "moneyflow_buy_lg_amount",
            "moneyflow_sell_lg_amount",
            "close",
        ],
        group_fields=["industry"],
    )
    cfg = LayeredBuilderConfig(
        max_order=3,
        max_candidates=96,
        layer_budgets={"L0": 8, "L1": 24, "L2": 0, "L3": 32, "L4": 0},
        windows=(5, 10),
        layer_gate_max_total=12,
        layer_gate_max_per_family=4,
        layer_gate_seed_max=8,
        random_seed=7,
    )

    out = LayeredExpressionBuilder().build(universe, feedback_hints={}, config=cfg)
    gate_items = [item for item in out if item.layer == "L3"]
    gate_families = {str(item.metadata.get("gate_family", "")) for item in gate_items}

    assert {
        "liquidity_activity",
        "moneyflow_pressure",
        "price_trend",
        "industry_activity",
    } <= gate_families
    assert all(str(item.metadata.get("gate_expression", "")).startswith(("greater(", "less(")) for item in gate_items)
    price_gates = [
        str(item.metadata.get("gate_expression", ""))
        for item in gate_items
        if item.metadata.get("gate_family") == "price_trend"
    ]
    moneyflow_gates = [
        str(item.metadata.get("gate_expression", ""))
        for item in gate_items
        if item.metadata.get("gate_family") == "moneyflow_pressure"
    ]
    assert price_gates
    assert moneyflow_gates
    assert all("moneyflow_" not in expr for expr in price_gates)
    assert any("moneyflow_" in expr for expr in moneyflow_gates)


def test_layered_builder_generates_bucket_groups_and_composites() -> None:
    universe = build_field_universe(
        _panel_store(),
        explicit_include_fields=["amount", "circ_mv", "pe", "close"],
        group_fields=["industry"],
    )
    cfg = LayeredBuilderConfig(
        max_order=4,
        max_candidates=120,
        layer_budgets={"L0": 6, "L1": 24, "L2": 8, "L3": 0, "L4": 40},
        windows=(5,),
        layer_enable_bucket_groups=True,
        layer_bucket_max_groups=8,
        layer_bucket_max_composite_groups=4,
        random_seed=5,
    )

    out = LayeredExpressionBuilder().build(universe, feedback_hints={}, config=cfg)
    bucket_items = [item for item in out if item.metadata.get("bucket_expression")]

    assert any("bucket(rank(" in item.expression and "group_neutralize(" in item.expression for item in bucket_items)
    assert any("group_cartesian_product(industry, bucket(" in item.expression for item in bucket_items)
    assert {"size", "liquidity", "valuation"} & {str(item.metadata.get("bucket_family", "")) for item in bucket_items}
    assert any(item.metadata.get("bucket_source_field") for item in bucket_items)
    assert any(item.metadata.get("bucket_source_family") for item in bucket_items)
    assert any(item.metadata.get("bucket_range") for item in bucket_items)
    assert all("group_complexity" in item.metadata for item in bucket_items)


def test_layered_stable_operator_expansion_is_capped_and_prefilter_compatible() -> None:
    universe = build_field_universe(
        _panel_store(),
        explicit_include_fields=["amount", "close", "circ_mv"],
        group_fields=["industry"],
    )
    cfg = LayeredBuilderConfig(
        max_order=1,
        max_candidates=80,
        layer_budgets={"L0": 4, "L1": 64, "L2": 0, "L3": 0, "L4": 0},
        windows=(5,),
        layer_operator_tier="stable",
        layer_operator_expansion_max_total=6,
    )
    prefilter = CandidatePrefilter(
        field_kinds=universe.kind_map(),
        max_operator_count=12,
        max_field_count=4,
        max_depth=6,
    )

    out = LayeredExpressionBuilder().build(universe, feedback_hints={}, config=cfg)
    stable_items = [item for item in out if item.metadata.get("operator_tier") == "stable"]
    stable_expressions = [item.expression for item in stable_items]

    assert 1 <= len(stable_items) <= 6
    assert any(
        any(
            op in item.expression
            for op in [
                "s_log_1p",
                "ts_decay_linear",
                "ts_ir",
                "ts_arg_max",
                "ts_arg_min",
                "signed_power",
            ]
        )
        for item in stable_items
    )
    assert len(stable_expressions) == len(set(stable_expressions))
    assert any(expr.startswith("signed_power(zscore(") for expr in stable_expressions)
    assert all(prefilter.check(item.expression).passed for item in stable_items)


def test_layered_bucket_l1_peer_comparison_is_capped() -> None:
    universe = build_field_universe(
        _panel_store(),
        explicit_include_fields=[
            "amount",
            "close",
            "circ_mv",
            "total_mv",
            "turnover_rate",
        ],
        group_fields=["industry"],
    )
    cfg = LayeredBuilderConfig(
        max_order=1,
        max_candidates=120,
        layer_budgets={"L0": 5, "L1": 80, "L2": 0, "L3": 0, "L4": 0},
        windows=(5,),
        layer_bucket_l1_max_total=4,
    )

    out = LayeredExpressionBuilder().build(universe, feedback_hints={}, config=cfg)
    bucket_l1 = [
        item for item in out if item.layer == "L1" and item.metadata.get("generated_group_type") == "bucket_l1_peer"
    ]

    assert 1 <= len(bucket_l1) <= 4
    assert all(item.expression.startswith(("group_rank(", "group_zscore(")) for item in bucket_l1)
    assert all("bucket(rank(" in item.expression for item in bucket_l1)
    assert all(item.metadata.get("bucket_source_family") in {"size", "liquidity"} for item in bucket_l1)


def test_layered_bucket_l2_pair_wrapper_respects_field_count_gate() -> None:
    universe = build_field_universe(
        _panel_store(),
        explicit_include_fields=[
            "amount",
            "close",
            "circ_mv",
            "total_mv",
            "turnover_rate",
        ],
        group_fields=["industry"],
    )
    base_kwargs = dict(
        max_order=2,
        max_candidates=160,
        layer_budgets={"L0": 5, "L1": 20, "L2": 80, "L3": 0, "L4": 0},
        windows=(5,),
        layer_bucket_l2_max_total=6,
        random_seed=9,
    )

    disabled = LayeredExpressionBuilder().build(
        universe,
        feedback_hints={},
        config=LayeredBuilderConfig(**base_kwargs, max_field_count_for_bucket_l2=2),
    )
    enabled = LayeredExpressionBuilder().build(
        universe,
        feedback_hints={},
        config=LayeredBuilderConfig(**base_kwargs, max_field_count_for_bucket_l2=3),
    )

    assert not [item for item in disabled if item.metadata.get("generated_group_type") == "bucket_l2_pair"]
    bucket_l2 = [item for item in enabled if item.metadata.get("generated_group_type") == "bucket_l2_pair"]
    assert 1 <= len(bucket_l2) <= 6
    assert all(item.layer == "L2" for item in bucket_l2)
    assert all("bucket(rank(" in item.expression for item in bucket_l2)


def test_candidate_metadata_json_tracks_layered_sources() -> None:
    from alpha_mining.mining.candidate_planner import plan_candidates

    class Config:
        include_fields = ("amount", "circ_mv", "close", "moneyflow_net_amount")
        exclude_fields = ()
        group_fields = ("industry",)
        vector_fields = ()
        search_field_universe = ()
        include_factor_families = ()
        exclude_factor_families = ()
        deep_explore_config = DeepExploreConfig(windows=(5,), max_candidates=50, random_seed=1)
        field_preprocessing_config = None
        search_mode = "layered_v2"
        use_signature_generator = True
        layer_max_order = 4
        layer_max_candidates = 120
        layer_budgets = {"L0": 4, "L1": 18, "L2": 8, "L3": 12, "L4": 24}
        layer_include_gates = True
        enable_stateful_phase2_ops = False
        layer_enable_bucket_groups = True
        layer_bucket_max_groups = 4
        layer_bucket_max_composite_groups = 2
        layer_enable_recipe_lite = True
        layer_recipe_max_total = 8
        layer_recipe_max_per_family = 4
        layer_role_pair_max_total = 6
        layer_cross_family_pair_ratio = 0.15
        field_profile_lite_enabled = True
        feedback_policy_lite_enabled = True
        layer_operator_tier = "stable"
        layer_operator_expansion_max_total = 4
        max_eval_expressions = 30
        enable_feedback_mutation = False
        enable_candidate_ranking = False
        enable_sample_prefilter = False
        template_include_families = ()
        template_pool_override = {}
        feedback_min_explore_ratio = 0.3
        enable_family_quota = False
        family_max_selected_ratio = 0.45
        family_min_explore_ratio = 0.25
        mining_config = type(
            "Mining",
            (),
            {
                "max_operator_count": 12,
                "max_field_count": 5,
                "skip_templates_with_missing_group": True,
            },
        )()
        search_field_source = "panel_store"

    _, candidate_df, _, meta = plan_candidates(_panel_store(), Config(), existing_hashes=set(), batch_id="t")
    metadata_rows = [json.loads(str(value)) for value in candidate_df["metadata_json"].dropna().astype(str)]

    assert meta["field_diagnostics"]["available"]["liquidity"] is True
    assert any(row.get("gate_family") for row in metadata_rows)
    assert any(row.get("bucket_expression") for row in metadata_rows)
    assert any(row.get("operator_tier") == "stable" for row in metadata_rows)
    assert any(row.get("recipe_family") for row in metadata_rows)
    assert any(row.get("role_pair_type") for row in metadata_rows)
    assert any("field_profile_score" in row for row in metadata_rows)


def test_builder_skips_existing_hashes() -> None:
    """已评估表达式应被 builder 跳过，不占用预算。"""
    from alpha_mining.hashing import expression_hash

    universe = build_field_universe(
        _panel_store(),
        explicit_include_fields=["amount", "circ_mv", "close"],
        group_fields=["industry"],
    )
    cfg = LayeredBuilderConfig(
        max_order=1,
        max_candidates=100,
        layer_budgets={"L0": 10, "L1": 90},
        windows=(5,),
        random_seed=3,
    )

    # 第一次生成
    out1 = LayeredExpressionBuilder().build(universe, config=cfg)
    assert len(out1) > 0

    # 收集前 5 个表达式的 hash 作为"已评估"集合
    existing = {expression_hash(item.expression) for item in out1[:5]}

    # 第二次生成，传入已评估集合
    builder2 = LayeredExpressionBuilder()
    out2 = builder2.build(universe, config=cfg, existing_hashes=existing)

    # 验证：out2 不包含已评估的表达式
    out2_hashes = {expression_hash(item.expression) for item in out2}
    assert not out2_hashes.intersection(existing)

    # 验证：dedup_count 被正确记录
    assert builder2.dedup_count == len(existing)


def test_builder_budget_not_wasted_on_existing() -> None:
    """已评估表达式不应占用预算名额，且 dedup_count 正确统计。"""
    from alpha_mining.hashing import expression_hash

    universe = build_field_universe(
        _panel_store(),
        explicit_include_fields=["amount", "circ_mv", "close"],
        group_fields=["industry"],
    )
    cfg = LayeredBuilderConfig(
        max_order=1,
        max_candidates=50,
        layer_budgets={"L0": 5, "L1": 45},
        windows=(5,),
        random_seed=3,
    )

    # 第一次生成
    out1 = LayeredExpressionBuilder().build(universe, config=cfg)
    l0_hashes = {expression_hash(item.expression) for item in out1 if item.layer == "L0"}
    assert len(l0_hashes) > 0

    # 第二次生成，传入 L0 已评估集合
    builder2 = LayeredExpressionBuilder()
    out2 = builder2.build(universe, config=cfg, existing_hashes=l0_hashes)

    # 验证：L0 全部被跳过
    l0_count_2 = sum(1 for item in out2 if item.layer == "L0")
    assert l0_count_2 == 0

    # 验证：dedup_count 等于已评估的 L0 数量
    assert builder2.dedup_count == len(l0_hashes)

    # 验证：总候选数不超过 max_candidates
    assert len(out2) <= cfg.max_candidates


def test_builder_backward_compatible_without_existing_hashes() -> None:
    """不传 existing_hashes 时行为与现有一致。"""
    universe = build_field_universe(
        _panel_store(),
        explicit_include_fields=["amount", "circ_mv", "close"],
        group_fields=["industry"],
    )
    cfg = LayeredBuilderConfig(
        max_order=1,
        max_candidates=50,
        windows=(5,),
        random_seed=3,
    )

    # 分别用 None 和 set() 调用
    out_none = LayeredExpressionBuilder().build(universe, config=cfg, existing_hashes=None)
    out_empty = LayeredExpressionBuilder().build(universe, config=cfg, existing_hashes=set())

    # 验证结果完全相同
    assert len(out_none) == len(out_empty)
    assert [item.expression for item in out_none] == [item.expression for item in out_empty]
