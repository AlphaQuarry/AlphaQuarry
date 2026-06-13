from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from ..hashing import expression_hash, normalize_expression
from .field_universe import FieldUniverse
from .field_preprocessing import FieldExpressionFactory, FieldPreprocessConfig
from .field_semantics import infer_field_semantic
from .field_profile_lite import (
    FieldProfile,
    aggregate_field_profile_score,
    aggregate_recommended_windows,
    build_field_profiles,
)
from .layered_recipes import build_layered_recipe_candidates, build_role_pair_candidates


DEFAULT_LAYER_BUDGETS: dict[str, int] = {
    "L0": 32,
    "L1": 160,
    "L2": 160,
    "L3": 100,
    "L4": 80,
}
DEFAULT_WINDOWS: tuple[int, ...] = (5, 10, 22, 66, 132)


@dataclass(frozen=True)
class LayeredBuilderConfig:
    max_order: int = 4
    max_candidates: int = 400
    layer_budgets: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_LAYER_BUDGETS))
    windows: tuple[int, ...] = DEFAULT_WINDOWS
    include_gates: bool = True
    enable_stateful_phase2_ops: bool = False
    field_preprocess_config: FieldPreprocessConfig = field(default_factory=FieldPreprocessConfig)
    random_seed: int = 42
    field_rotation_focus_count: int = 0  # 0 = 不轮转 (兼容旧行为), >0 = 每轮聚焦 N 个字段
    field_rotation_iteration: int = 0  # 当前轮次 (用于种子变化)
    budget_rotation_mode: str = "none"  # none / round_robin
    layer_gate_families: tuple[str, ...] = (
        "liquidity_activity",
        "moneyflow_pressure",
        "price_trend",
        "industry_activity",
    )
    layer_gate_max_total: int = 24
    layer_gate_max_per_family: int = 6
    layer_gate_seed_max: int = 18
    layer_gate_templates: tuple[str, ...] = ("if_else_zero",)
    layer_enable_event_gates: bool = False
    layer_enable_bucket_groups: bool = True
    layer_bucket_max_groups: int = 12
    layer_bucket_max_composite_groups: int = 6
    layer_bucket_ranges: tuple[str, ...] = ("0,1,0.2",)
    layer_bucket_field_families: tuple[str, ...] = (
        "size",
        "liquidity",
        "valuation",
        "chip",
        "technical",
    )
    layer_bucket_use_composite_industry: bool = True
    layer_bucket_l1_max_total: int = 24
    layer_bucket_l2_max_total: int = 20
    max_field_count_for_bucket_l2: int = 4
    layer_enable_recipe_lite: bool = True
    layer_recipe_max_total: int = 80
    layer_recipe_max_per_family: int = 16
    layer_role_pair_max_total: int = 80
    layer_cross_family_pair_ratio: float = 0.15
    field_profile_lite_enabled: bool = True
    field_profile_lite_min_coverage: float = 0.20
    field_profile_lite_min_finite_rate: float = 0.80
    field_profile_lite_top_fields_per_family: int = 50
    feedback_policy_lite_enabled: bool = True
    layer_operator_tier: str = "stable"
    layer_operator_expansion_max_total: int = 100


@dataclass(frozen=True)
class CandidateExpression:
    expression: str
    layer: str
    layer_order: int
    layer_family: str
    source: str
    parent_expression: str = ""
    parent_hash: str = ""
    builder_source: str = "layered_v2"
    windows: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateExpression:
    expression: str
    family: str
    source_fields: tuple[str, ...] = ()
    source_groups: tuple[str, ...] = ()
    windows: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GroupExpression:
    expression: str
    family: str
    source_fields: tuple[str, ...] = ()
    source_groups: tuple[str, ...] = ()
    range_expression: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class LayeredExpressionBuilder:
    """
    Deterministic L0-L4 candidate source.

    This builder deliberately emits parser/signature-compatible expressions and
    relies on the existing canonicalizer, prefilter, sample evaluator and ranker
    for final acceptance.
    """

    def __init__(self) -> None:
        self.dedup_count: int = 0

    def build(
        self,
        field_universe: FieldUniverse,
        feedback_hints: dict[str, Any] | None = None,
        config: LayeredBuilderConfig | None = None,
        existing_hashes: set[str] | None = None,
    ) -> list[CandidateExpression]:
        cfg = config or LayeredBuilderConfig()
        hints = feedback_hints or {}
        max_order = max(0, min(4, int(cfg.max_order)))
        windows = _sanitize_windows(cfg.windows)
        groups = sorted(field_universe.group_fields)
        field_categories = {str(spec.name): tuple(str(x) for x in spec.categories) for spec in field_universe.specs}
        field_profiles: dict[str, FieldProfile] = {}
        if bool(cfg.field_profile_lite_enabled):
            raw_profiles = hints.get("field_profiles", {}) if isinstance(hints, dict) else {}
            if isinstance(raw_profiles, dict) and raw_profiles:
                field_profiles = {
                    str(name): profile for name, profile in raw_profiles.items() if isinstance(profile, FieldProfile)
                }
            if not field_profiles:
                field_profiles = build_field_profiles(
                    field_universe,
                    panel_store=None,
                    feedback_hints=hints,
                    min_coverage=float(cfg.field_profile_lite_min_coverage),
                    min_finite_rate=float(cfg.field_profile_lite_min_finite_rate),
                    top_fields_per_family=int(cfg.field_profile_lite_top_fields_per_family),
                )
        fields = _ordered_fields(field_universe.scalar_fields, field_categories, hints, field_profiles)
        field_expr_map = FieldExpressionFactory(cfg.field_preprocess_config).expression_map(fields)
        if not fields or max_order < 0:
            return []

        budgets = _rotated_budgets(
            _normalized_budgets(cfg.layer_budgets),
            str(cfg.budget_rotation_mode),
            int(cfg.field_rotation_iteration),
        )
        max_candidates = max(1, int(cfg.max_candidates))
        rng = random.Random(int(cfg.random_seed))
        by_layer: dict[str, list[CandidateExpression]] = {f"L{i}": [] for i in range(5)}
        out: list[CandidateExpression] = []
        seen: set[str] = set()
        existing = {str(x) for x in (existing_hashes or set()) if str(x)}
        self.dedup_count = 0

        # 字段轮转焦点: 每轮聚焦不同的字段子集
        focus_fields: set[str] | None = None
        if int(cfg.field_rotation_focus_count) > 0:
            rotation_rng = random.Random(int(cfg.random_seed) + int(cfg.field_rotation_iteration) * 1000)
            rotated = list(fields)
            rotation_rng.shuffle(rotated)
            focus_fields = set(rotated[: int(cfg.field_rotation_focus_count)])

        def add(
            layer: str,
            expr: str,
            family: str,
            parent: str = "",
            expr_windows: tuple[int, ...] = (),
            metadata: dict[str, Any] | None = None,
        ) -> CandidateExpression | None:
            order = _layer_order(layer)
            if order > max_order:
                return None
            if len(out) >= max_candidates:
                return None
            if len(by_layer[layer]) >= budgets.get(layer, 0):
                return None
            text = str(expr or "").strip()
            if not text:
                return None
            key = normalize_expression(text)
            if key in seen:
                return None
            if expression_hash(text) in existing:
                self.dedup_count += 1
                return None
            seen.add(key)
            item_metadata = dict(metadata or {})
            if field_profiles:
                source_fields = item_metadata.get("source_fields", ())
                if isinstance(source_fields, str):
                    profile_fields = [x.strip() for x in source_fields.split(",") if x.strip()]
                else:
                    try:
                        profile_fields = [str(x) for x in source_fields if str(x)]
                    except Exception:
                        profile_fields = []
                if not profile_fields:
                    profile_fields = sorted(_fields_in_text(text).intersection(set(field_profiles.keys())))
                profile_score = aggregate_field_profile_score(profile_fields, field_profiles)
                profile_windows = aggregate_recommended_windows(profile_fields, field_profiles)
                item_metadata.setdefault("field_profile_score", float(profile_score))
                item_metadata.setdefault("profile_recommended_windows", list(profile_windows))
            item = CandidateExpression(
                expression=text,
                layer=layer,
                layer_order=order,
                layer_family=str(family or "layered"),
                source=f"layered_v2:{layer}:{family or 'layered'}",
                parent_expression=str(parent or ""),
                parent_hash=expression_hash(parent) if parent else "",
                windows=tuple(int(w) for w in expr_windows if int(w) > 0),
                metadata=item_metadata,
            )
            by_layer[layer].append(item)
            out.append(item)
            return item

        for field_name in fields:
            add(
                "L0",
                field_expr_map.get(field_name, field_name),
                _field_family(field_name, field_categories.get(field_name, ())),
            )

        if max_order >= 1:
            if bool(cfg.enable_stateful_phase2_ops):
                for base in list(by_layer["L0"]):
                    if focus_fields is not None and not _fields_in_text(base.expression).intersection(focus_fields):
                        continue
                    f = base.expression
                    add(
                        "L1",
                        f"hump({f})",
                        "stateful",
                        parent=f,
                        metadata={"operator_tier": "stateful"},
                    )
                    add(
                        "L1",
                        f"trade_when_hold(greater({f}, 0.0), {f}, less({f}, 0.0))",
                        "stateful",
                        parent=f,
                        metadata={"operator_tier": "stateful"},
                    )
                    if len(by_layer["L1"]) >= budgets.get("L1", 0):
                        break
            if bool(cfg.layer_enable_recipe_lite):
                recipe_items = build_layered_recipe_candidates(
                    field_universe,
                    windows=windows,
                    max_total=max(0, int(cfg.layer_recipe_max_total)),
                    max_per_family=max(1, int(cfg.layer_recipe_max_per_family)),
                    field_expression_map=field_expr_map,
                    feedback_hints=hints,
                )
                for recipe in recipe_items:
                    if focus_fields is not None:
                        recipe_src = recipe.metadata.get("source_fields", ()) if recipe.metadata else ()
                        if isinstance(recipe_src, str):
                            recipe_src_fields = {x.strip() for x in recipe_src.split(",") if x.strip()}
                        else:
                            recipe_src_fields = {str(x) for x in recipe_src if str(x)}
                        if recipe_src_fields and not recipe_src_fields.intersection(focus_fields):
                            continue
                    add(
                        recipe.layer,
                        recipe.expression,
                        recipe.layer_family,
                        parent=recipe.parent_expression,
                        expr_windows=recipe.windows,
                        metadata=recipe.metadata,
                    )
            bucket_l1_groups = (
                _build_bucket_peer_groups(
                    fields=fields,
                    field_expr_map=field_expr_map,
                    field_categories=field_categories,
                    cfg=cfg,
                    allowed_families={"size", "liquidity"},
                )
                if bool(cfg.layer_enable_bucket_groups)
                else []
            )
            bucket_l1_count = 0
            for base in list(by_layer["L0"]):
                if focus_fields is not None and not _fields_in_text(base.expression).intersection(focus_fields):
                    continue
                parent = base.expression
                parent_fields = _fields_in_text(parent)
                for group_item in bucket_l1_groups:
                    if bucket_l1_count >= max(0, int(cfg.layer_bucket_l1_max_total)):
                        break
                    if set(group_item.source_fields).intersection(parent_fields):
                        continue
                    for op in ("group_rank", "group_zscore"):
                        item = add(
                            "L1",
                            f"{op}({parent}, {group_item.expression})",
                            "bucket_peer",
                            parent=parent,
                            expr_windows=base.windows,
                            metadata=_bucket_metadata(group_item, generated_group_type="bucket_l1_peer"),
                        )
                        if item is not None:
                            bucket_l1_count += 1
                        if bucket_l1_count >= max(0, int(cfg.layer_bucket_l1_max_total)):
                            break
                if bucket_l1_count >= max(0, int(cfg.layer_bucket_l1_max_total)):
                    break
            stable_expansion_count = 0
            for base in list(by_layer["L0"]):
                if focus_fields is not None and not _fields_in_text(base.expression).intersection(focus_fields):
                    continue
                f = base.expression
                if str(cfg.layer_operator_tier or "").strip().lower() == "stable":
                    for expr, expr_windows in _stable_operator_expansions(f, windows):
                        if stable_expansion_count >= max(0, int(cfg.layer_operator_expansion_max_total)):
                            break
                        item = add(
                            "L1",
                            expr,
                            "operator_expansion",
                            parent=f,
                            expr_windows=expr_windows,
                            metadata={"operator_tier": "stable"},
                        )
                        if item is not None:
                            stable_expansion_count += 1
                add("L1", f"rank({f})", "cross_sectional", parent=f)
                add("L1", f"zscore({f})", "cross_sectional", parent=f)
                add("L1", f"normalize({f})", "cross_sectional", parent=f)
                add("L1", f"reverse({f})", "cross_sectional", parent=f)
                add("L1", f"quantile({f}, 'gaussian', 1.0)", "cross_sectional", parent=f)
                add("L1", f"truncate({f}, 0.02)", "cross_sectional", parent=f)
                add("L1", f"left_tail({f}, 0.0)", "cross_sectional", parent=f)
                add("L1", f"right_tail({f}, 0.0)", "cross_sectional", parent=f)
                for d in windows:
                    add(
                        "L1",
                        f"ts_rank({f}, {d})",
                        "time_series",
                        parent=f,
                        expr_windows=(d,),
                    )
                    add(
                        "L1",
                        f"ts_zscore({f}, {d})",
                        "time_series",
                        parent=f,
                        expr_windows=(d,),
                    )
                    add(
                        "L1",
                        f"ts_mean({f}, {d})",
                        "time_series",
                        parent=f,
                        expr_windows=(d,),
                    )
                    add(
                        "L1",
                        f"ts_std_dev({f}, {d})",
                        "time_series",
                        parent=f,
                        expr_windows=(d,),
                    )
                    add(
                        "L1",
                        f"ts_delta({f}, {d})",
                        "time_series",
                        parent=f,
                        expr_windows=(d,),
                    )
                    add(
                        "L1",
                        f"ts_min({f}, {d})",
                        "time_series",
                        parent=f,
                        expr_windows=(d,),
                    )
                    add(
                        "L1",
                        f"ts_max({f}, {d})",
                        "time_series",
                        parent=f,
                        expr_windows=(d,),
                    )
                    add(
                        "L1",
                        f"ts_median({f}, {d})",
                        "time_series",
                        parent=f,
                        expr_windows=(d,),
                    )
                    add(
                        "L1",
                        f"ts_av_diff({f}, {d})",
                        "time_series",
                        parent=f,
                        expr_windows=(d,),
                    )
                    add(
                        "L1",
                        f"ts_count_nans({f}, {d})",
                        "time_series",
                        parent=f,
                        expr_windows=(d,),
                    )
                for group in groups:
                    add("L1", f"group_rank({f}, {group})", "group", parent=f)
                    add("L1", f"group_zscore({f}, {group})", "group", parent=f)
                    add("L1", f"group_neutralize({f}, {group})", "group", parent=f)
                    add("L1", f"group_median({f}, {group})", "group", parent=f)
                    add("L1", f"group_scale({f}, {group})", "group", parent=f)
        if max_order >= 2:
            if bool(cfg.layer_enable_recipe_lite):
                role_pairs = build_role_pair_candidates(
                    field_universe,
                    windows=windows,
                    max_total=max(0, int(cfg.layer_role_pair_max_total)),
                    cross_family_pair_ratio=max(0.0, min(1.0, float(cfg.layer_cross_family_pair_ratio))),
                    field_expression_map=field_expr_map,
                    feedback_hints=hints,
                )
                for pair in role_pairs:
                    if focus_fields is not None:
                        pair_src = pair.metadata.get("source_fields", ()) if pair.metadata else ()
                        if isinstance(pair_src, str):
                            pair_src_fields = {x.strip() for x in pair_src.split(",") if x.strip()}
                        else:
                            pair_src_fields = {str(x) for x in pair_src if str(x)}
                        if pair_src_fields and not pair_src_fields.intersection(focus_fields):
                            continue
                    add(
                        pair.layer,
                        pair.expression,
                        pair.layer_family,
                        parent=pair.parent_expression,
                        expr_windows=pair.windows,
                        metadata=pair.metadata,
                    )
            field_pairs = [(a, b) for idx, a in enumerate(fields) for b in fields[idx + 1 :]]
            if focus_fields is not None:
                field_pairs = [(a, b) for a, b in field_pairs if a in focus_fields or b in focus_fields]
            rng.shuffle(field_pairs)
            bucket_l2_groups = (
                _build_bucket_peer_groups(
                    fields=fields,
                    field_expr_map=field_expr_map,
                    field_categories=field_categories,
                    cfg=cfg,
                    allowed_families={"size", "liquidity"},
                )
                if bool(cfg.layer_enable_bucket_groups) and int(cfg.max_field_count_for_bucket_l2) >= 3
                else []
            )
            bucket_l2_count = 0
            for a, b in field_pairs:
                d = windows[0]
                a_expr = field_expr_map.get(a, a)
                b_expr = field_expr_map.get(b, b)
                parent = f"{a},{b}"
                add(
                    "L2",
                    f"ts_corr({a_expr}, {b_expr}, {d})",
                    "pair_time_series",
                    parent=parent,
                    expr_windows=(d,),
                )
                add(
                    "L2",
                    f"ts_covariance({a_expr}, {b_expr}, {d})",
                    "pair_time_series",
                    parent=parent,
                    expr_windows=(d,),
                )
                add("L2", f"rank(div({a_expr}, {b_expr}))", "pair_ratio", parent=parent)
                add(
                    "L2",
                    f"zscore(sub({a_expr}, {b_expr}))",
                    "pair_spread",
                    parent=parent,
                )
                if groups:
                    add(
                        "L2",
                        f"group_zscore(sub({a_expr}, {b_expr}), {groups[0]})",
                        "pair_group",
                        parent=parent,
                    )
                if bucket_l2_groups and bucket_l2_count < max(0, int(cfg.layer_bucket_l2_max_total)):
                    for group_item in bucket_l2_groups:
                        if set(group_item.source_fields).intersection({a, b}):
                            continue
                        for expr in (
                            f"group_zscore(sub({a_expr}, {b_expr}), {group_item.expression})",
                            f"group_rank(div({a_expr}, {b_expr}), {group_item.expression})",
                        ):
                            item = add(
                                "L2",
                                expr,
                                "bucket_pair",
                                parent=parent,
                                metadata=_bucket_metadata(group_item, generated_group_type="bucket_l2_pair"),
                            )
                            if item is not None:
                                bucket_l2_count += 1
                            if bucket_l2_count >= max(0, int(cfg.layer_bucket_l2_max_total)):
                                break
                        if bucket_l2_count >= max(0, int(cfg.layer_bucket_l2_max_total)):
                            break

            for seed in list(by_layer["L1"]):
                parent = seed.expression
                d = windows[0]
                add(
                    "L2",
                    f"ts_zscore({parent}, {d})",
                    "time_series_wrap",
                    parent=parent,
                    expr_windows=(d,),
                )
                add(
                    "L2",
                    f"ts_rank({parent}, {d})",
                    "time_series_wrap",
                    parent=parent,
                    expr_windows=(d,),
                )
                add(
                    "L2",
                    f"ts_av_diff({parent}, {d})",
                    "time_series_wrap",
                    parent=parent,
                    expr_windows=(d,),
                )
                if groups:
                    add(
                        "L2",
                        f"group_rank({parent}, {groups[0]})",
                        "group_wrap",
                        parent=parent,
                    )
                    add(
                        "L2",
                        f"group_zscore({parent}, {groups[0]})",
                        "group_wrap",
                        parent=parent,
                    )

        if max_order >= 3 and bool(cfg.include_gates):
            gate_expressions = _build_gate_expressions(
                fields=fields,
                groups=groups,
                field_expr_map=field_expr_map,
                field_categories=field_categories,
                cfg=cfg,
                windows=windows,
            )
            simple_alpha_seeds = _select_gate_alpha_seeds(by_layer, cfg)
            for seed in simple_alpha_seeds:
                alpha = seed.expression
                for gate_item in gate_expressions:
                    add(
                        "L3",
                        f"if_else({gate_item.expression}, {alpha}, zero_like({alpha}))",
                        "gate",
                        parent=alpha,
                        expr_windows=gate_item.windows,
                        metadata={
                            "gate_family": gate_item.family,
                            "gate_expression": gate_item.expression,
                            "context_fields": list(gate_item.metadata.get("context_fields", [])),
                        },
                    )
                    if bool(cfg.layer_enable_event_gates):
                        add(
                            "L3",
                            f"event_active({gate_item.expression}, {alpha})",
                            "gate",
                            parent=alpha,
                            expr_windows=gate_item.windows,
                            metadata={
                                "gate_family": gate_item.family,
                                "gate_expression": gate_item.expression,
                            },
                        )
                    if len(by_layer["L3"]) >= budgets.get("L3", 0):
                        break
                if len(by_layer["L3"]) >= budgets.get("L3", 0):
                    break

        if max_order >= 4:
            bucket_groups = (
                _build_group_expressions(
                    fields=fields,
                    groups=groups,
                    field_expr_map=field_expr_map,
                    field_categories=field_categories,
                    cfg=cfg,
                )
                if bool(cfg.layer_enable_bucket_groups)
                else []
            )
            risk_field = "circ_mv" if "circ_mv" in set(fields) else ""
            for base in list(by_layer["L0"]):
                parent = base.expression
                for group in groups:
                    add(
                        "L4",
                        f"group_neutralize({parent}, {group})",
                        "risk",
                        parent=parent,
                        expr_windows=base.windows,
                    )
                    add(
                        "L4",
                        f"group_zscore({parent}, {group})",
                        "risk",
                        parent=parent,
                        expr_windows=base.windows,
                    )
            bucket_wrap_seeds = list(by_layer["L1"]) + list(by_layer["L2"]) + list(by_layer["L3"])
            for seed in bucket_wrap_seeds:
                parent = seed.expression
                for group_item in bucket_groups:
                    for op in ("group_neutralize", "group_zscore", "group_rank"):
                        add(
                            "L4",
                            f"{op}({parent}, {group_item.expression})",
                            "bucket_risk",
                            parent=parent,
                            expr_windows=seed.windows,
                            metadata=_bucket_metadata(group_item),
                        )
                        if len(by_layer["L4"]) >= budgets.get("L4", 0):
                            break
                    if len(by_layer["L4"]) >= budgets.get("L4", 0):
                        break
                if len(by_layer["L4"]) >= budgets.get("L4", 0):
                    break
            wrap_seeds = list(by_layer["L3"]) + list(by_layer["L2"]) + list(by_layer["L1"])
            for seed in wrap_seeds:
                parent = seed.expression
                d = windows[0]
                for group in groups:
                    add(
                        "L4",
                        f"group_neutralize({parent}, {group})",
                        "risk",
                        parent=parent,
                        expr_windows=seed.windows,
                    )
                    add(
                        "L4",
                        f"group_zscore({parent}, {group})",
                        "risk",
                        parent=parent,
                        expr_windows=seed.windows,
                    )
                if risk_field and risk_field not in _fields_in_text(parent):
                    add(
                        "L4",
                        f"regression_neut({parent}, {risk_field})",
                        "risk",
                        parent=parent,
                        expr_windows=seed.windows,
                    )
                add(
                    "L4",
                    f"ts_zscore({parent}, {d})",
                    "rescale",
                    parent=parent,
                    expr_windows=tuple(sorted(set(seed.windows + (d,)))),
                )

        return out[:max_candidates]


def _stable_operator_expansions(field_expression: str, windows: tuple[int, ...]) -> list[tuple[str, tuple[int, ...]]]:
    f = str(field_expression)
    z = f"zscore({f})"
    d = int(windows[0]) if windows else 5
    return [
        (f"s_log_1p({f})", ()),
        (f"signed_power({z}, 2.0)", ()),
        (f"signed_power({z}, 0.5)", ()),
        (f"ts_decay_linear({f}, {d})", (d,)),
        (f"ts_ir({f}, {d})", (d,)),
        (f"ts_arg_max({f}, {d})", (d,)),
        (f"ts_arg_min({f}, {d})", (d,)),
    ]


def _select_gate_alpha_seeds(
    by_layer: dict[str, list[CandidateExpression]], cfg: LayeredBuilderConfig
) -> list[CandidateExpression]:
    max_count = max(1, int(cfg.layer_gate_seed_max))
    seeds = list(by_layer.get("L0", ()))
    seeds.extend(
        item
        for item in by_layer.get("L1", ())
        if item.expression.startswith(("rank(", "zscore(", "ts_mean(", "ts_rank(", "ts_zscore("))
    )
    return seeds[:max_count]


def _build_gate_expressions(
    *,
    fields: list[str],
    groups: list[str],
    field_expr_map: dict[str, str],
    field_categories: dict[str, tuple[str, ...]],
    cfg: LayeredBuilderConfig,
    windows: tuple[int, ...],
) -> list[GateExpression]:
    requested = {str(x).strip() for x in cfg.layer_gate_families if str(x).strip()}
    per_family = max(1, int(cfg.layer_gate_max_per_family))
    total = max(0, int(cfg.layer_gate_max_total))
    d_short = int(windows[0]) if windows else 5
    d_mid = int(windows[1]) if len(windows) > 1 else d_short
    by_family: dict[str, list[GateExpression]] = {}

    def add(
        family: str,
        expr: str,
        source_fields: tuple[str, ...],
        source_groups: tuple[str, ...] = (),
        win: tuple[int, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if family not in requested:
            return
        items = by_family.setdefault(family, [])
        if len(items) >= per_family:
            return
        items.append(
            GateExpression(
                expr,
                family,
                source_fields=source_fields,
                source_groups=source_groups,
                windows=win,
                metadata=dict(metadata or {}),
            )
        )

    semantics = {f: infer_field_semantic(f, field_categories.get(f, ())) for f in fields}
    liquidity = [f for f, semantic in semantics.items() if semantic.role == "liquidity"]
    moneyflow = [f for f, semantic in semantics.items() if semantic.role == "moneyflow"]
    price = [f for f, semantic in semantics.items() if semantic.role == "price"]
    context = [f for f in fields if f.lower().startswith("ctx_")]

    for field in liquidity[:per_family]:
        x = field_expr_map.get(field, field)
        add(
            "liquidity_activity",
            f"greater(ts_zscore({x}, {d_mid}), 0.0)",
            (field,),
            win=(d_mid,),
        )
        add(
            "liquidity_activity",
            f"greater(ts_rank({x}, {d_short}), 0.5)",
            (field,),
            win=(d_short,),
        )

    buy_fields = [f for f in moneyflow if "buy" in f.lower()]
    sell_fields = [f for f in moneyflow if "sell" in f.lower()]
    if buy_fields and sell_fields:
        buy = field_expr_map.get(buy_fields[0], buy_fields[0])
        sell = field_expr_map.get(sell_fields[0], sell_fields[0])
        add(
            "moneyflow_pressure",
            f"greater(ts_zscore(sub({buy}, {sell}), {d_mid}), 0.0)",
            (buy_fields[0], sell_fields[0]),
            win=(d_mid,),
        )
    for field in moneyflow[:per_family]:
        x = field_expr_map.get(field, field)
        add(
            "moneyflow_pressure",
            f"greater(ts_zscore({x}, {d_mid}), 0.0)",
            (field,),
            win=(d_mid,),
        )
        add(
            "moneyflow_pressure",
            f"greater(ts_rank({x}, {d_short}), 0.5)",
            (field,),
            win=(d_short,),
        )

    for field in price[:per_family]:
        x = field_expr_map.get(field, field)
        add(
            "price_trend",
            f"greater(ts_delta({x}, {d_mid}), 0.0)",
            (field,),
            win=(d_mid,),
        )
        add(
            "price_trend",
            f"greater(ts_rank({x}, {d_mid}), 0.5)",
            (field,),
            win=(d_mid,),
        )

    activity_fields = liquidity[:2] + moneyflow[:2]
    for group in [g for g in groups if g in {"industry", "sector", "subindustry"}][:1]:
        for field in activity_fields:
            x = field_expr_map.get(field, field)
            add(
                "industry_activity",
                f"greater(group_mean(ts_zscore({x}, {d_mid}), {group}), 0.0)",
                (field,),
                source_groups=(group,),
                win=(d_mid,),
            )

    for field in context[:per_family]:
        x = field_expr_map.get(field, field)
        add(
            "market_regime",
            f"greater(ts_mean({x}, {d_short}), 0.0)",
            (field,),
            win=(d_short,),
            metadata={"context_fields": [field]},
        )
    ordered_families = [str(x).strip() for x in cfg.layer_gate_families if str(x).strip()]
    out: list[GateExpression] = []
    while len(out) < total:
        progressed = False
        for family in ordered_families:
            items = by_family.get(family, [])
            if not items:
                continue
            out.append(items.pop(0))
            progressed = True
            if len(out) >= total:
                break
        if not progressed:
            break
    return out[:total]


def _build_group_expressions(
    *,
    fields: list[str],
    groups: list[str],
    field_expr_map: dict[str, str],
    field_categories: dict[str, tuple[str, ...]],
    cfg: LayeredBuilderConfig,
) -> list[GroupExpression]:
    families = {str(x).strip() for x in cfg.layer_bucket_field_families if str(x).strip()}
    ranges = tuple(str(x).strip() for x in cfg.layer_bucket_ranges if str(x).strip()) or ("0,1,0.2",)
    selected = _select_bucket_fields(
        fields,
        field_categories,
        families,
        max_count=max(1, int(cfg.layer_bucket_max_groups)),
    )
    out: list[GroupExpression] = []
    for field in selected:
        family = _bucket_family(field, field_categories.get(field, ()))
        x = field_expr_map.get(field, field)
        for range_expr in ranges:
            bucket_expr = f"bucket(rank({x}), '{range_expr}')"
            out.append(
                GroupExpression(
                    expression=bucket_expr,
                    family=family,
                    source_fields=(field,),
                    range_expression=range_expr,
                    metadata={
                        "generated_group_type": "bucket",
                        "bucket_source_field": field,
                        "bucket_source_family": family,
                        "bucket_range": range_expr,
                        "base_group": "",
                        "group_complexity": 1,
                    },
                )
            )
            if len(out) >= max(0, int(cfg.layer_bucket_max_groups)):
                break
        if len(out) >= max(0, int(cfg.layer_bucket_max_groups)):
            break

    if bool(cfg.layer_bucket_use_composite_industry) and out:
        composite_groups = [g for g in groups if g in {"industry", "sector", "subindustry"}]
        composite_limit = max(0, int(cfg.layer_bucket_max_composite_groups))
        composite_count = 0
        for base_group in composite_groups[:1]:
            for bucket_group in list(out):
                if composite_count >= composite_limit:
                    break
                out.append(
                    GroupExpression(
                        expression=f"group_cartesian_product({base_group}, {bucket_group.expression})",
                        family=bucket_group.family,
                        source_fields=bucket_group.source_fields,
                        source_groups=(base_group,),
                        range_expression=bucket_group.range_expression,
                        metadata={
                            **dict(bucket_group.metadata),
                            "generated_group_type": f"{base_group}_x_bucket",
                            "base_group": base_group,
                            "group_complexity": 2,
                        },
                    )
                )
                composite_count += 1
            if composite_count >= composite_limit:
                break
    return out


def _build_bucket_peer_groups(
    *,
    fields: list[str],
    field_expr_map: dict[str, str],
    field_categories: dict[str, tuple[str, ...]],
    cfg: LayeredBuilderConfig,
    allowed_families: set[str],
) -> list[GroupExpression]:
    ranges = tuple(str(x).strip() for x in cfg.layer_bucket_ranges if str(x).strip()) or ("0,1,0.2",)
    selected = _select_bucket_fields(
        fields,
        field_categories,
        {str(x).strip() for x in allowed_families if str(x).strip()},
        max_count=max(1, int(cfg.layer_bucket_max_groups)),
    )
    out: list[GroupExpression] = []
    for field in selected:
        family = _bucket_family(field, field_categories.get(field, ()))
        if family not in allowed_families:
            continue
        x = field_expr_map.get(field, field)
        for range_expr in ranges:
            out.append(
                GroupExpression(
                    expression=f"bucket(rank({x}), '{range_expr}')",
                    family=family,
                    source_fields=(field,),
                    range_expression=range_expr,
                    metadata={
                        "generated_group_type": "bucket",
                        "bucket_source_field": field,
                        "bucket_source_family": family,
                        "bucket_range": range_expr,
                        "base_group": "",
                        "group_complexity": 1,
                    },
                )
            )
            break
    return out


def _bucket_metadata(group_item: GroupExpression, generated_group_type: str | None = None) -> dict[str, Any]:
    meta = dict(group_item.metadata or {})
    group_type = str(generated_group_type or meta.get("generated_group_type", "bucket") or "bucket")
    return {
        "bucket_family": group_item.family,
        "bucket_expression": group_item.expression,
        "generated_group_type": group_type,
        "bucket_source_field": str(
            meta.get("bucket_source_field", "") or (group_item.source_fields[0] if group_item.source_fields else "")
        ),
        "bucket_source_family": str(meta.get("bucket_source_family", "") or group_item.family),
        "bucket_range": str(meta.get("bucket_range", "") or group_item.range_expression),
        "base_group": str(
            meta.get("base_group", "") or (group_item.source_groups[0] if group_item.source_groups else "")
        ),
        "group_complexity": int(meta.get("group_complexity", 1) or 1),
    }


BUCKET_FIELD_PRIORITY: dict[str, tuple[str, ...]] = {
    "size": ("circ_mv", "total_mv"),
    "liquidity": ("amount", "turnover_rate", "volume", "volume_ratio"),
    "valuation": ("pb", "pe_ttm", "ps_ttm", "dv_ttm"),
    "chip": ("cyq_winner_rate", "cyq_weight_avg", "cyq_cost_50pct"),
    "technical": ("tech_rsi", "tech_mfi", "volume_ratio"),
}


def _select_bucket_fields(
    fields: list[str],
    field_categories: dict[str, tuple[str, ...]],
    families: set[str],
    *,
    max_count: int,
) -> list[str]:
    selected: list[str] = []
    field_set = {str(f) for f in fields}

    def append(field: str) -> None:
        if field in field_set and field not in selected:
            selected.append(field)

    for family in ("size", "liquidity", "valuation", "chip", "technical"):
        if family not in families:
            continue
        for candidate in BUCKET_FIELD_PRIORITY.get(family, ()):
            exact_matches = [candidate] if candidate in field_set else []
            prefix_matches = [f for f in fields if f.startswith(candidate)]
            for field in exact_matches + prefix_matches:
                if _bucket_family(field, field_categories.get(field, ())) == family:
                    append(field)
                    break
            if selected and _bucket_family(selected[-1], field_categories.get(selected[-1], ())) == family:
                break
        if len(selected) >= max_count:
            return selected[:max_count]

    for field in fields:
        if len(selected) >= max_count:
            break
        family = _bucket_family(field, field_categories.get(field, ()))
        if family in families:
            append(field)
    return selected[:max_count]


def _bucket_family(name: str, categories: tuple[str, ...]) -> str:
    semantic = infer_field_semantic(name, categories)
    return semantic.bucket_family or semantic.role


def _sanitize_windows(values: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    out = sorted({int(v) for v in values if int(v) > 0})
    return tuple(out or [5, 10, 22])


def _normalized_budgets(raw: dict[str, int] | None) -> dict[str, int]:
    out = dict(DEFAULT_LAYER_BUDGETS)
    for key, value in dict(raw or {}).items():
        layer = str(key).upper()
        if not layer.startswith("L"):
            layer = f"L{layer}"
        if layer in out:
            out[layer] = max(0, int(value))
    return out


# Budget 轮转预设: 每轮侧重不同层, 总 Budget 保持与默认一致 (532)
# 策略: 侧重层 +60, 其余三层各 -20, 基准 120
_ROTATION_PRESETS: list[dict[str, int]] = [
    {"L0": 32, "L1": 180, "L2": 120, "L3": 120, "L4": 80},  # 侧重 L1 (单层变换)
    {"L0": 32, "L1": 120, "L2": 180, "L3": 120, "L4": 80},  # 侧重 L2 (双层组合)
    {"L0": 32, "L1": 120, "L2": 120, "L3": 180, "L4": 80},  # 侧重 L3 (门控)
    {"L0": 32, "L1": 120, "L2": 120, "L3": 80, "L4": 180},  # 侧重 L4 (中性化)
]


def _rotated_budgets(base: dict[str, int], mode: str, iteration: int) -> dict[str, int]:
    """根据轮转模式和轮次调整 Budget 分配。"""
    if mode != "round_robin" or iteration <= 0:
        return base
    preset = _ROTATION_PRESETS[iteration % len(_ROTATION_PRESETS)]
    out = dict(base)
    for key, value in preset.items():
        if key in out:
            out[key] = value
    return out


def _ordered_fields(
    fields: list[str],
    field_categories: dict[str, tuple[str, ...]],
    hints: dict[str, Any],
    field_profiles: dict[str, FieldProfile] | None = None,
) -> list[str]:
    weights = hints.get("field_weights", {}) if isinstance(hints, dict) else {}
    negative = hints.get("negative_field_weights", {}) if isinstance(hints, dict) else {}
    profiles = dict(field_profiles or {})

    def score(name: str) -> tuple[float, int, str]:
        positive = float(weights.get(name, 0.0)) if isinstance(weights, dict) else 0.0
        penalty = float(negative.get(name, 0.0)) if isinstance(negative, dict) else 0.0
        profile_score = float(profiles[name].field_profile_score) if name in profiles else 0.0
        category = _field_family(name, field_categories.get(name, ()))
        category_rank = {
            "moneyflow": 0,
            "liquidity": 1,
            "price": 2,
            "size": 3,
            "valuation": 4,
            "chip": 5,
            "technical": 6,
            "finance": 7,
            "analyst": 8,
        }.get(category, 9)
        return (profile_score + positive - penalty, -category_rank, name)

    return sorted([str(f) for f in fields if str(f)], key=score, reverse=True)


def _field_family(name: str, categories: tuple[str, ...]) -> str:
    return infer_field_semantic(name, categories).role


def _choose_gate_field(fields: list[str]) -> str:
    preferred = [
        f for f in fields if any(token in f.lower() for token in ["amount", "moneyflow", "volume", "turnover"])
    ]
    return preferred[0] if preferred else fields[0]


def _layer_order(layer: str) -> int:
    try:
        return int(str(layer).upper().replace("L", ""))
    except Exception:
        return 0


def _fields_in_text(expression: str) -> set[str]:
    raw = str(expression or "")
    tokens = raw.replace("(", " ").replace(")", " ").replace(",", " ").replace("+", " ").replace("-", " ").split()
    return {t.strip() for t in tokens if t.strip().isidentifier()}
