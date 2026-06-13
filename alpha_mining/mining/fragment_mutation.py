from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..ast_nodes import (
    BinaryOpNode,
    ExpressionNode,
    FieldNode,
    FunctionCallNode,
    LiteralNode,
    UnaryOpNode,
)
from ..hashing import expression_hash
from ..parser import ExpressionParseError, parse_expression
from .expression_canonicalizer import canonicalize_expression
from .field_preprocessing import FieldExpressionFactory


DEFAULT_REJECTED_PATTERNS: tuple[str, ...] = (
    "rank(rank(",
    "zscore(zscore(",
    "group_neutralize(group_neutralize(",
    "trade_when_hold(trade_when_hold(",
)

# ts_ 运算符分组：同组内可互换
TS_OPERATOR_GROUPS: dict[str, tuple[str, ...]] = {
    "central": ("ts_mean", "ts_median"),
    "spread": ("ts_std_dev", "ts_av_diff"),
    "extreme": ("ts_min", "ts_max"),
    "rank": ("ts_rank", "ts_zscore"),
    "delta": ("ts_delta", "ts_delay"),
}


@dataclass(frozen=True)
class MutationConfig:
    windows: tuple[int, ...] = (5, 10, 22, 66, 132)
    max_mutations: int = 30
    max_children_per_parent: int = 3
    enable_stateful: bool = False
    stateful_ratio_cap: float = 0.10
    random_seed: int = 42
    enable_operator_swap: bool = True
    enable_crossover: bool = True
    rejected_patterns: tuple[str, ...] = ()  # 空表示使用默认规则


def generate_mutation_candidates(
    fragments_df: pd.DataFrame,
    field_roles: dict[str, str],
    group_fields: list[str],
    existing_hashes: set[str] | None = None,
    config: MutationConfig | None = None,
) -> list[dict[str, Any]]:
    cfg = config or MutationConfig()
    existing = {str(x) for x in (existing_hashes or set()) if str(x)}
    group_pool = [str(g) for g in group_fields if str(g)]
    out: list[dict[str, Any]] = []
    seen_hashes: set[str] = set(existing)
    per_parent: dict[str, int] = {}
    stateful_count = 0
    stateful_cap = max(
        0,
        int(round(float(cfg.max_mutations) * max(0.0, float(cfg.stateful_ratio_cap)))),
    )
    rng = random.Random(int(cfg.random_seed))

    if fragments_df is None or fragments_df.empty:
        return []
    work = fragments_df.copy()
    if "fragment_priority" in work.columns:
        work = work.sort_values("fragment_priority", ascending=False)
    work = work.head(max(1, int(cfg.max_mutations) * 3))

    for row in work.to_dict(orient="records"):
        if len(out) >= int(cfg.max_mutations):
            break
        fragment_expr = str(row.get("fragment_expression", "") or "").strip()
        if not fragment_expr:
            continue
        parent_expression = str(row.get("source_expression", "") or fragment_expr)
        parent_hash = str(row.get("source_alpha_hash", "") or expression_hash(parent_expression))
        parent_key = parent_hash or expression_hash(parent_expression)
        if per_parent.get(parent_key, 0) >= int(cfg.max_children_per_parent):
            continue

        base_candidates = _mutate_from_fragment(
            fragment_expr=fragment_expr,
            field_roles=field_roles,
            group_fields=group_pool,
            windows=tuple(int(x) for x in cfg.windows if int(x) > 0),
            enable_stateful=bool(cfg.enable_stateful),
            rng=rng,
            enable_operator_swap=bool(cfg.enable_operator_swap),
        )
        for item in base_candidates:
            if len(out) >= int(cfg.max_mutations):
                break
            mutation_type = str(item.get("mutation_type", ""))
            if mutation_type == "stateful_wrapper_add":
                if not bool(cfg.enable_stateful):
                    continue
                if stateful_count >= stateful_cap:
                    continue
            expr = str(item.get("expression", "") or "").strip()
            if not expr:
                continue
            if _is_rejected_pattern(expr, custom_patterns=cfg.rejected_patterns):
                continue
            canon = canonicalize_expression(expr)
            if not canon.passed or not canon.canonical_hash:
                continue
            expr_hash = expression_hash(expr)
            if canon.canonical_hash in seen_hashes or expr_hash in seen_hashes:
                continue
            seen_hashes.add(canon.canonical_hash)
            seen_hashes.add(expr_hash)
            payload = {
                "expression": expr,
                "canonical_expression": canon.canonical_expression,
                "canonical_hash": canon.canonical_hash,
                "parent_expression": parent_expression,
                "parent_hash": parent_key,
                "fragment_hash": str(row.get("fragment_hash", "")),
                "feedback_source": "feedback_mutation_v2",
                "mutation_type": mutation_type,
                "windows": ",".join(str(x) for x in sorted(set(item.get("windows", ()) or ()))),
            }
            out.append(payload)
            per_parent[parent_key] = per_parent.get(parent_key, 0) + 1
            if mutation_type == "stateful_wrapper_add":
                stateful_count += 1
            if per_parent[parent_key] >= int(cfg.max_children_per_parent):
                break

    return out


def generate_crossover_candidates(
    top_expressions: list[str],
    existing_hashes: set[str] | None = None,
    max_crossovers: int = 10,
    random_seed: int = 42,
) -> list[dict[str, Any]]:
    """从两个高分因子中各取一个子表达式，组合成新表达式。"""
    rng = random.Random(random_seed)
    out: list[dict[str, Any]] = []
    seen = {str(x) for x in (existing_hashes or set()) if str(x)}

    for i, expr_a in enumerate(top_expressions):
        for expr_b in top_expressions[i + 1 :]:
            if len(out) >= max_crossovers:
                break
            try:
                node_a = parse_expression(expr_a)
                node_b = parse_expression(expr_b)
            except ExpressionParseError:
                continue

            # 提取 node_b 的子表达式
            subtrees_b = _extract_subtrees(node_b, max_depth=2)
            if not subtrees_b:
                continue

            # 随机选择 node_a 的一个字段位置替换为 node_b 的子表达式
            field_paths_a = _collect_field_paths(node_a)
            if not field_paths_a:
                continue

            sub_b = rng.choice(subtrees_b)
            target_path, _ = rng.choice(field_paths_a)

            crossover_expr = _replace_subtree_at(node_a, target_path, sub_b)
            if not crossover_expr:
                continue

            canon = canonicalize_expression(crossover_expr)
            if not canon.passed or not canon.canonical_hash:
                continue
            if canon.canonical_hash in seen or expression_hash(crossover_expr) in seen:
                continue
            seen.add(canon.canonical_hash)

            out.append(
                {
                    "expression": crossover_expr,
                    "canonical_expression": canon.canonical_expression,
                    "canonical_hash": canon.canonical_hash,
                    "parent_expression": expr_a,
                    "parent_hash": expression_hash(expr_a),
                    "fragment_hash": "",
                    "feedback_source": "crossover",
                    "mutation_type": "crossover",
                    "windows": "",
                }
            )
    return out


def _extract_subtrees(node: ExpressionNode, max_depth: int = 2, depth: int = 0) -> list[ExpressionNode]:
    """提取 AST 中的子表达式（深度 <= max_depth 的非叶节点）"""
    out: list[ExpressionNode] = []
    if depth > 0 and depth <= max_depth:
        out.append(node)
    if depth >= max_depth:
        return out
    if isinstance(node, FunctionCallNode):
        for arg in node.args:
            out.extend(_extract_subtrees(arg, max_depth, depth + 1))
    elif isinstance(node, UnaryOpNode):
        out.extend(_extract_subtrees(node.operand, max_depth, depth + 1))
    elif isinstance(node, BinaryOpNode):
        out.extend(_extract_subtrees(node.left, max_depth, depth + 1))
        out.extend(_extract_subtrees(node.right, max_depth, depth + 1))
    return out


def _replace_subtree_at(node: ExpressionNode, path: tuple[int, ...], replacement: ExpressionNode) -> str:
    """将指定路径的子表达式替换为 replacement"""
    if not path:
        return _node_to_expression(replacement)
    head, *tail = list(path)
    if isinstance(node, FunctionCallNode):
        if head >= len(node.args):
            return ""
        child = _replace_subtree_at(node.args[head], tuple(tail), replacement)
        if not child:
            return ""
        try:
            child_node = parse_expression(child)
        except ExpressionParseError:
            return ""
        new_args = list(node.args)
        new_args[head] = child_node
        return _node_to_expression(
            FunctionCallNode(name=node.name, args=tuple(new_args), named_args=dict(node.named_args))
        )
    elif isinstance(node, UnaryOpNode):
        if head != 0:
            return ""
        child = _replace_subtree_at(node.operand, tuple(tail), replacement)
        if not child:
            return ""
        try:
            child_node = parse_expression(child)
        except ExpressionParseError:
            return ""
        return _node_to_expression(UnaryOpNode(op=node.op, operand=child_node))
    elif isinstance(node, BinaryOpNode):
        if head == 0:
            child = _replace_subtree_at(node.left, tuple(tail), replacement)
            if not child:
                return ""
            try:
                child_node = parse_expression(child)
            except ExpressionParseError:
                return ""
            return _node_to_expression(BinaryOpNode(op=node.op, left=child_node, right=node.right))
        if head == 1:
            child = _replace_subtree_at(node.right, tuple(tail), replacement)
            if not child:
                return ""
            try:
                child_node = parse_expression(child)
            except ExpressionParseError:
                return ""
            return _node_to_expression(BinaryOpNode(op=node.op, left=node.left, right=child_node))
    return ""


def _mutate_from_fragment(
    fragment_expr: str,
    field_roles: dict[str, str],
    group_fields: list[str],
    windows: tuple[int, ...],
    enable_stateful: bool,
    rng: random.Random,
    enable_operator_swap: bool = True,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        node = parse_expression(fragment_expr)
    except ExpressionParseError:
        return out
    field_factory = FieldExpressionFactory()

    # 1) window_shift
    for source_window in _collect_windows(node):
        for target in windows:
            if int(target) == int(source_window):
                continue
            replaced = _replace_window_once(node=node, source_window=int(source_window), target_window=int(target))
            if replaced:
                out.append(
                    {
                        "mutation_type": "window_shift",
                        "expression": replaced,
                        "windows": (int(target),),
                    }
                )

    # 2) same_role_field_swap
    field_paths = _collect_field_paths(node)
    for path, field_name in field_paths:
        role = str(field_roles.get(field_name, ""))
        if not role:
            continue
        candidates = [f for f, r in field_roles.items() if str(r) == role and str(f) != field_name]
        rng.shuffle(candidates)
        for replacement in candidates[:2]:
            repl_expr = _replace_field_once(
                node=node,
                path=path,
                replacement_field=str(replacement),
                replacement_expression=field_factory.expression_for(str(replacement), kind="scalar"),
            )
            if repl_expr:
                out.append(
                    {
                        "mutation_type": "same_role_field_swap",
                        "expression": repl_expr,
                        "windows": tuple(),
                    }
                )

    # 3) group_swap
    lower_groups = [g for g in group_fields if str(g).strip()]
    for path, field_name in field_paths:
        if field_name not in set(lower_groups):
            continue
        for replacement in [x for x in lower_groups if x != field_name][:2]:
            repl_expr = _replace_field_once(node=node, path=path, replacement_field=str(replacement))
            if repl_expr:
                out.append(
                    {
                        "mutation_type": "group_swap",
                        "expression": repl_expr,
                        "windows": tuple(),
                    }
                )

    # 4) operator_swap — 将 ts_ 运算符替换为同组的其他运算符
    if bool(enable_operator_swap):
        for op_path, op_name in _collect_ts_operator_paths(node):
            for _group_name, group_ops in TS_OPERATOR_GROUPS.items():
                if op_name in group_ops:
                    alternatives = [op for op in group_ops if op != op_name]
                    for alt in alternatives:
                        replaced = _replace_operator_once(node, op_path, alt)
                        if replaced:
                            out.append(
                                {
                                    "mutation_type": "operator_swap",
                                    "expression": replaced,
                                    "windows": tuple(),
                                }
                            )

    base = fragment_expr
    # 5) wrapper_add
    out.extend(
        [
            {
                "mutation_type": "wrapper_add",
                "expression": f"rank({base})",
                "windows": tuple(),
            },
            {
                "mutation_type": "wrapper_add",
                "expression": f"zscore({base})",
                "windows": tuple(),
            },
            {
                "mutation_type": "wrapper_add",
                "expression": f"normalize({base})",
                "windows": tuple(),
            },
        ]
    )

    # 5) group_wrapper_add
    for group in lower_groups[:2]:
        out.extend(
            [
                {
                    "mutation_type": "group_wrapper_add",
                    "expression": f"group_neutralize({base}, {group})",
                    "windows": tuple(),
                },
                {
                    "mutation_type": "group_wrapper_add",
                    "expression": f"group_rank({base}, {group})",
                    "windows": tuple(),
                },
            ]
        )

    # 6) stateful_wrapper_add
    if bool(enable_stateful):
        gate_field = _choose_gate_field(_collect_fields(node), field_roles)
        entry = f"greater(rank({gate_field}), ts_mean(rank({gate_field}), 22))"
        exit_ = f"less(rank({gate_field}), ts_mean(rank({gate_field}), 22))"
        out.extend(
            [
                {
                    "mutation_type": "stateful_wrapper_add",
                    "expression": f"hump({base}, 0.01)",
                    "windows": (22,),
                },
                {
                    "mutation_type": "stateful_wrapper_add",
                    "expression": f"trade_when_hold({entry}, {base}, {exit_})",
                    "windows": (22,),
                },
            ]
        )
    return out


def _collect_ts_operator_paths(node: ExpressionNode, path: tuple[int, ...] = ()) -> list[tuple[tuple[int, ...], str]]:
    """收集所有 ts_ 运算符的路径和名称"""
    out: list[tuple[tuple[int, ...], str]] = []
    if isinstance(node, FunctionCallNode):
        if node.name.startswith("ts_"):
            out.append((path, node.name))
        for idx, arg in enumerate(node.args):
            out.extend(_collect_ts_operator_paths(arg, path + (idx,)))
    elif isinstance(node, UnaryOpNode):
        out.extend(_collect_ts_operator_paths(node.operand, path + (0,)))
    elif isinstance(node, BinaryOpNode):
        out.extend(_collect_ts_operator_paths(node.left, path + (0,)))
        out.extend(_collect_ts_operator_paths(node.right, path + (1,)))
    return out


def _replace_operator_once(node: ExpressionNode, path: tuple[int, ...], new_name: str) -> str:
    """将指定路径的运算符替换为 new_name"""
    if not path:
        if isinstance(node, FunctionCallNode):
            return _node_to_expression(
                FunctionCallNode(name=new_name, args=node.args, named_args=dict(node.named_args))
            )
        return ""
    head, *tail = list(path)
    if isinstance(node, FunctionCallNode):
        if head >= len(node.args):
            return ""
        child = _replace_operator_once(node.args[head], tuple(tail), new_name)
        if not child:
            return ""
        try:
            child_node = parse_expression(child)
        except ExpressionParseError:
            return ""
        new_args = list(node.args)
        new_args[head] = child_node
        return _node_to_expression(
            FunctionCallNode(name=node.name, args=tuple(new_args), named_args=dict(node.named_args))
        )
    elif isinstance(node, UnaryOpNode):
        if head != 0:
            return ""
        child = _replace_operator_once(node.operand, tuple(tail), new_name)
        if not child:
            return ""
        try:
            child_node = parse_expression(child)
        except ExpressionParseError:
            return ""
        return _node_to_expression(UnaryOpNode(op=node.op, operand=child_node))
    elif isinstance(node, BinaryOpNode):
        if head == 0:
            child = _replace_operator_once(node.left, tuple(tail), new_name)
            if not child:
                return ""
            try:
                child_node = parse_expression(child)
            except ExpressionParseError:
                return ""
            return _node_to_expression(BinaryOpNode(op=node.op, left=child_node, right=node.right))
        if head == 1:
            child = _replace_operator_once(node.right, tuple(tail), new_name)
            if not child:
                return ""
            try:
                child_node = parse_expression(child)
            except ExpressionParseError:
                return ""
            return _node_to_expression(BinaryOpNode(op=node.op, left=node.left, right=child_node))
    return ""


def _replace_window_once(node: ExpressionNode, source_window: int, target_window: int) -> str:
    if isinstance(node, FunctionCallNode):
        changed = False
        new_args: list[ExpressionNode] = []
        for arg in node.args:
            if (
                not changed
                and node.name.startswith("ts_")
                and isinstance(arg, LiteralNode)
                and _is_int_like(arg.value)
                and int(arg.value) == int(source_window)
            ):
                new_args.append(LiteralNode(value=int(target_window)))
                changed = True
            else:
                new_args.append(arg)
        if changed:
            replaced = FunctionCallNode(name=node.name, args=tuple(new_args), named_args=dict(node.named_args))
            return _node_to_expression(replaced)
        for idx, arg in enumerate(node.args):
            child_expr = _replace_window_once(arg, source_window, target_window)
            if child_expr:
                try:
                    child_node = parse_expression(child_expr)
                except ExpressionParseError:
                    return ""
                new_args = list(node.args)
                new_args[idx] = child_node
                replaced = FunctionCallNode(
                    name=node.name,
                    args=tuple(new_args),
                    named_args=dict(node.named_args),
                )
                return _node_to_expression(replaced)
    elif isinstance(node, UnaryOpNode):
        child_expr = _replace_window_once(node.operand, source_window, target_window)
        if child_expr:
            try:
                child_node = parse_expression(child_expr)
            except ExpressionParseError:
                return ""
            return _node_to_expression(UnaryOpNode(op=node.op, operand=child_node))
    elif isinstance(node, BinaryOpNode):
        left = _replace_window_once(node.left, source_window, target_window)
        if left:
            try:
                left_node = parse_expression(left)
            except ExpressionParseError:
                return ""
            return _node_to_expression(BinaryOpNode(op=node.op, left=left_node, right=node.right))
        right = _replace_window_once(node.right, source_window, target_window)
        if right:
            try:
                right_node = parse_expression(right)
            except ExpressionParseError:
                return ""
            return _node_to_expression(BinaryOpNode(op=node.op, left=node.left, right=right_node))
    return ""


def _replace_field_once(
    node: ExpressionNode,
    path: tuple[int, ...],
    replacement_field: str,
    replacement_expression: str | None = None,
) -> str:
    if replacement_expression and replacement_expression != replacement_field:
        try:
            replacement_node = parse_expression(replacement_expression)
        except ExpressionParseError:
            replacement_node = FieldNode(name=str(replacement_field))
    else:
        replacement_node = FieldNode(name=str(replacement_field))
    replaced = _replace_field_node(node=node, path=path, replacement=replacement_node)
    if replaced is None:
        return ""
    return _node_to_expression(replaced)


def _replace_field_node(
    node: ExpressionNode, path: tuple[int, ...], replacement: ExpressionNode
) -> ExpressionNode | None:
    if not path:
        if isinstance(node, FieldNode):
            return replacement
        return None
    head, *tail = list(path)
    if isinstance(node, UnaryOpNode):
        if head != 0:
            return None
        child = _replace_field_node(node.operand, tuple(tail), replacement)
        return UnaryOpNode(op=node.op, operand=child) if child is not None else None
    if isinstance(node, BinaryOpNode):
        if head == 0:
            child = _replace_field_node(node.left, tuple(tail), replacement)
            return BinaryOpNode(op=node.op, left=child, right=node.right) if child is not None else None
        if head == 1:
            child = _replace_field_node(node.right, tuple(tail), replacement)
            return BinaryOpNode(op=node.op, left=node.left, right=child) if child is not None else None
        return None
    if isinstance(node, FunctionCallNode):
        if head >= len(node.args):
            return None
        child = _replace_field_node(node.args[head], tuple(tail), replacement)
        if child is None:
            return None
        new_args = list(node.args)
        new_args[head] = child
        return FunctionCallNode(name=node.name, args=tuple(new_args), named_args=dict(node.named_args))
    return None


def _collect_field_paths(node: ExpressionNode, path: tuple[int, ...] = ()) -> list[tuple[tuple[int, ...], str]]:
    if isinstance(node, FieldNode):
        return [(path, str(node.name))]
    if isinstance(node, UnaryOpNode):
        return _collect_field_paths(node.operand, path + (0,))
    if isinstance(node, BinaryOpNode):
        return _collect_field_paths(node.left, path + (0,)) + _collect_field_paths(node.right, path + (1,))
    if isinstance(node, FunctionCallNode):
        out: list[tuple[tuple[int, ...], str]] = []
        for idx, arg in enumerate(node.args):
            out.extend(_collect_field_paths(arg, path + (idx,)))
        return out
    return []


def _collect_windows(node: ExpressionNode) -> list[int]:
    if isinstance(node, FunctionCallNode):
        windows = []
        if node.name.startswith("ts_"):
            for arg in node.args:
                if isinstance(arg, LiteralNode) and _is_int_like(arg.value) and int(arg.value) > 0:
                    windows.append(int(arg.value))
        for arg in node.args:
            windows.extend(_collect_windows(arg))
        return windows
    if isinstance(node, UnaryOpNode):
        return _collect_windows(node.operand)
    if isinstance(node, BinaryOpNode):
        return _collect_windows(node.left) + _collect_windows(node.right)
    return []


def _collect_fields(node: ExpressionNode) -> list[str]:
    if isinstance(node, FieldNode):
        return [str(node.name)]
    if isinstance(node, UnaryOpNode):
        return _collect_fields(node.operand)
    if isinstance(node, BinaryOpNode):
        return _collect_fields(node.left) + _collect_fields(node.right)
    if isinstance(node, FunctionCallNode):
        out: list[str] = []
        for arg in node.args:
            out.extend(_collect_fields(arg))
        return out
    return []


def _choose_gate_field(fields: list[str], field_roles: dict[str, str]) -> str:
    ordered = [str(f) for f in fields if str(f)]
    preferred = [f for f in ordered if str(field_roles.get(f, "")) in {"liquidity", "price"}]
    return preferred[0] if preferred else (ordered[0] if ordered else "close")


def _node_to_expression(node: ExpressionNode) -> str:
    if isinstance(node, LiteralNode):
        return repr(node.value)
    if isinstance(node, FieldNode):
        return str(node.name)
    if isinstance(node, UnaryOpNode):
        return f"{node.op}{_node_to_expression(node.operand)}"
    if isinstance(node, BinaryOpNode):
        return f"{_node_to_expression(node.left)} {node.op} {_node_to_expression(node.right)}"
    if isinstance(node, FunctionCallNode):
        args = [_node_to_expression(arg) for arg in node.args]
        args.extend([f"{k}={_node_to_expression(v)}" for k, v in sorted(node.named_args.items())])
        return f"{node.name}({', '.join(args)})"
    return ""


def _is_rejected_pattern(expression: str, custom_patterns: tuple[str, ...] = ()) -> bool:
    low = "".join(str(expression or "").lower().split())
    patterns = custom_patterns if custom_patterns else DEFAULT_REJECTED_PATTERNS
    for pattern in patterns:
        if pattern.lower().replace(" ", "") in low:
            return True
    # 额外检查：嵌套次数限制
    if low.count("group_neutralize(") > 1:
        return True
    if low.count("trade_when_hold(") > 1:
        return True
    return False


def _is_int_like(value: Any) -> bool:
    try:
        _ = int(value)
        return True
    except Exception:
        return False
