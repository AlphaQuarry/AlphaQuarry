from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .field_preprocessing import FieldExpressionFactory, FieldPreprocessConfig
from .field_semantics import infer_field_semantic
from .field_universe import FieldUniverse


@dataclass(frozen=True)
class LayeredRecipeCandidate:
    expression: str
    layer: str
    layer_family: str
    parent_expression: str = ""
    windows: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def build_layered_recipe_candidates(
    field_universe: FieldUniverse,
    *,
    windows: tuple[int, ...],
    max_total: int,
    max_per_family: int,
    field_expression_map: dict[str, str] | None = None,
    feedback_hints: dict[str, Any] | None = None,
) -> list[LayeredRecipeCandidate]:
    fields = list(field_universe.scalar_fields)
    groups = list(field_universe.group_fields)
    categories = {spec.name: tuple(spec.categories) for spec in field_universe.specs}
    expr_map = dict(field_expression_map or FieldExpressionFactory(FieldPreprocessConfig()).expression_map(fields))
    d_short, d_mid = _window_pair(windows)
    families: dict[str, list[LayeredRecipeCandidate]] = {
        "moneyflow_imbalance": [],
        "liquidity_shock": [],
        "valuation_peer": [],
        "chip_pressure": [],
        "trend_liquidity_confirmation": [],
    }

    by_role = _fields_by_role(fields, categories)
    buy_fields = [f for f in by_role.get("moneyflow", []) if "buy" in f.lower()]
    sell_fields = [f for f in by_role.get("moneyflow", []) if "sell" in f.lower()]
    if buy_fields and sell_fields:
        buy = buy_fields[0]
        sell = sell_fields[0]
        buy_expr = expr_map.get(buy, buy)
        sell_expr = expr_map.get(sell, sell)
        _append_recipe(
            families,
            "moneyflow_imbalance",
            f"ts_zscore(sub({buy_expr}, {sell_expr}), {d_mid})",
            (d_mid,),
            fields=(buy, sell),
            recipe_id="moneyflow_imbalance:buy_sell_ts_zscore",
        )
        _append_recipe(
            families,
            "moneyflow_imbalance",
            f"rank(sub({buy_expr}, {sell_expr}))",
            (),
            fields=(buy, sell),
            recipe_id="moneyflow_imbalance:buy_sell_rank",
        )

    for field in by_role.get("liquidity", [])[:3]:
        x = expr_map.get(field, field)
        _append_recipe(
            families,
            "liquidity_shock",
            f"ts_zscore({x}, {d_short})",
            (d_short,),
            fields=(field,),
            recipe_id=f"liquidity_shock:{field}:ts_zscore",
        )
        _append_recipe(
            families,
            "liquidity_shock",
            f"ts_rank({x}, {d_short})",
            (d_short,),
            fields=(field,),
            recipe_id=f"liquidity_shock:{field}:ts_rank",
        )

    peer_groups = [g for g in groups if g in {"industry", "sector", "subindustry"}]
    if peer_groups:
        group = peer_groups[0]
        for field in by_role.get("valuation", [])[:3]:
            x = expr_map.get(field, field)
            _append_recipe(
                families,
                "valuation_peer",
                f"group_zscore(rank({x}), {group})",
                (),
                fields=(field,),
                groups=(group,),
                recipe_id=f"valuation_peer:{field}:{group}",
            )

    for field in by_role.get("chip", [])[:3]:
        x = expr_map.get(field, field)
        _append_recipe(
            families,
            "chip_pressure",
            f"ts_rank({x}, {d_mid})",
            (d_mid,),
            fields=(field,),
            recipe_id=f"chip_pressure:{field}:ts_rank",
        )
        _append_recipe(
            families,
            "chip_pressure",
            f"zscore({x})",
            (),
            fields=(field,),
            recipe_id=f"chip_pressure:{field}:zscore",
        )

    price_fields = by_role.get("price", [])
    liquidity_fields = by_role.get("liquidity", [])
    if price_fields and liquidity_fields:
        price = price_fields[0]
        liq = liquidity_fields[0]
        price_expr = expr_map.get(price, price)
        liq_expr = expr_map.get(liq, liq)
        liq_rank = f"ts_rank({liq_expr}, {d_short})"
        _append_recipe(
            families,
            "trend_liquidity_confirmation",
            f"if_else(greater(ts_delta({price_expr}, {d_short}), 0.0), {liq_rank}, zero_like({liq_rank}))",
            (d_short,),
            fields=(price, liq),
            recipe_id="trend_liquidity_confirmation:price_delta_liquidity_rank",
        )

    ordered = _order_recipe_families(families, feedback_hints or {})
    out: list[LayeredRecipeCandidate] = []
    per_family = max(0, int(max_per_family))
    total = max(0, int(max_total))
    while len(out) < total:
        progressed = False
        for family in ordered:
            items = families.get(family, [])
            if not items:
                continue
            used_family = sum(1 for item in out if item.metadata.get("recipe_family") == family)
            if used_family >= per_family:
                items.clear()
                continue
            out.append(items.pop(0))
            progressed = True
            if len(out) >= total:
                break
        if not progressed:
            break
    return out[:total]


def build_role_pair_candidates(
    field_universe: FieldUniverse,
    *,
    windows: tuple[int, ...],
    max_total: int,
    cross_family_pair_ratio: float = 0.15,
    field_expression_map: dict[str, str] | None = None,
    feedback_hints: dict[str, Any] | None = None,
) -> list[LayeredRecipeCandidate]:
    fields = list(field_universe.scalar_fields)
    categories = {spec.name: tuple(spec.categories) for spec in field_universe.specs}
    expr_map = dict(field_expression_map or FieldExpressionFactory(FieldPreprocessConfig()).expression_map(fields))
    by_role = _fields_by_role(fields, categories)
    d_short, _ = _window_pair(windows)
    out: list[LayeredRecipeCandidate] = []
    cross_limit = max(
        0,
        int(max(0.0, min(1.0, float(cross_family_pair_ratio))) * max(0, int(max_total))),
    )
    if cross_family_pair_ratio > 0 and max_total > 0:
        cross_limit = max(1, cross_limit)
    cross_count = 0

    def add(
        expr: str,
        pair_type: str,
        source_fields: tuple[str, ...],
        family: str,
        *,
        is_cross_family: bool = False,
    ) -> None:
        nonlocal cross_count
        if len(out) >= max(0, int(max_total)):
            return
        if is_cross_family and cross_count >= cross_limit:
            return
        out.append(
            LayeredRecipeCandidate(
                expression=expr,
                layer="L2",
                layer_family="role_pair",
                parent_expression=",".join(source_fields),
                windows=(d_short,),
                metadata={
                    "role_pair_type": pair_type,
                    "recipe_family": family,
                    "recipe_id": f"{family}:{pair_type}",
                    "source_fields": list(source_fields),
                },
            )
        )
        if is_cross_family:
            cross_count += 1

    buy_fields = [f for f in by_role.get("moneyflow", []) if "buy" in f.lower()]
    sell_fields = [f for f in by_role.get("moneyflow", []) if "sell" in f.lower()]
    if buy_fields and sell_fields:
        buy, sell = buy_fields[0], sell_fields[0]
        add(
            f"rank(sub({expr_map.get(buy, buy)}, {expr_map.get(sell, sell)}))",
            "moneyflow_buy_sell",
            (buy, sell),
            "moneyflow_imbalance",
        )

    if by_role.get("price") and by_role.get("liquidity"):
        price, liq = by_role["price"][0], by_role["liquidity"][0]
        add(
            f"ts_corr({expr_map.get(price, price)}, {expr_map.get(liq, liq)}, {d_short})",
            "price_x_liquidity",
            (price, liq),
            "trend_liquidity_confirmation",
            is_cross_family=True,
        )

    if by_role.get("valuation") and by_role.get("size"):
        val, size = by_role["valuation"][0], by_role["size"][0]
        add(
            f"rank(div({expr_map.get(val, val)}, {expr_map.get(size, size)}))",
            "valuation_x_size",
            (val, size),
            "valuation_peer",
            is_cross_family=True,
        )

    if by_role.get("chip") and by_role.get("price"):
        chip, price = by_role["chip"][0], by_role["price"][0]
        add(
            f"ts_corr({expr_map.get(chip, chip)}, {expr_map.get(price, price)}, {d_short})",
            "chip_x_price",
            (chip, price),
            "chip_pressure",
            is_cross_family=True,
        )

    return _order_pair_candidates(out, feedback_hints or {})[: max(0, int(max_total))]


def _append_recipe(
    families: dict[str, list[LayeredRecipeCandidate]],
    family: str,
    expression: str,
    windows: tuple[int, ...],
    *,
    fields: tuple[str, ...],
    recipe_id: str,
    groups: tuple[str, ...] = (),
) -> None:
    families.setdefault(family, []).append(
        LayeredRecipeCandidate(
            expression=expression,
            layer="L1",
            layer_family="recipe_lite",
            parent_expression=",".join(fields),
            windows=windows,
            metadata={
                "recipe_id": recipe_id,
                "recipe_family": family,
                "source_fields": list(fields),
                "source_groups": list(groups),
            },
        )
    )


def _fields_by_role(fields: list[str], categories: dict[str, tuple[str, ...]]) -> dict[str, list[str]]:
    by_role: dict[str, list[str]] = {}
    for field in fields:
        semantic = infer_field_semantic(field, categories.get(field, ()))
        by_role.setdefault(semantic.role, []).append(field)
    return by_role


def _window_pair(windows: tuple[int, ...]) -> tuple[int, int]:
    values = [int(w) for w in windows if int(w) > 0]
    if not values:
        return 5, 22
    if len(values) == 1:
        return values[0], values[0]
    return values[0], values[1]


def _order_recipe_families(families: dict[str, list[LayeredRecipeCandidate]], hints: dict[str, Any]) -> list[str]:
    weights = hints.get("recipe_weights", {}) if isinstance(hints, dict) else {}
    negative = hints.get("negative_recipe_weights", {}) if isinstance(hints, dict) else {}

    def score(family: str) -> tuple[float, str]:
        pos = float(weights.get(family, 0.0)) if isinstance(weights, dict) else 0.0
        neg = float(negative.get(family, 0.0)) if isinstance(negative, dict) else 0.0
        return (pos - neg, family)

    return sorted([family for family, items in families.items() if items], key=score, reverse=True)


def _order_pair_candidates(items: list[LayeredRecipeCandidate], hints: dict[str, Any]) -> list[LayeredRecipeCandidate]:
    weights = hints.get("role_pair_type_weights", {}) if isinstance(hints, dict) else {}
    negative = hints.get("negative_role_pair_type_weights", {}) if isinstance(hints, dict) else {}

    def score(item: LayeredRecipeCandidate) -> tuple[float, str]:
        pair_type = str(item.metadata.get("role_pair_type", ""))
        pos = float(weights.get(pair_type, 0.0)) if isinstance(weights, dict) else 0.0
        neg = float(negative.get(pair_type, 0.0)) if isinstance(negative, dict) else 0.0
        return (pos - neg, pair_type)

    return sorted(items, key=score, reverse=True)
