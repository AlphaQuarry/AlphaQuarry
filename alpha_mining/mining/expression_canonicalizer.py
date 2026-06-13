from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from ..ast_nodes import (
    BinaryOpNode,
    ExpressionNode,
    FieldNode,
    FunctionCallNode,
    LiteralNode,
    UnaryOpNode,
)
from ..parser import parse_expression


_ALIASES = {
    "subtract": "sub",
    "multiply": "mul",
    "divide": "div",
    "cs_quantile": "quantile",
}
_BINARY_TO_FUNC = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
    "**": "power",
}
_COMMUTATIVE = {"add", "mul", "max", "min"}
_IDEMPOTENT_REJECT = {"rank", "zscore", "normalize"}
_SELF_COMPARISON_REJECT = {
    "greater",
    "less",
    "greater_equal",
    "less_equal",
    "equal",
    "not_equal",
}


@dataclass(frozen=True)
class CanonicalExpressionResult:
    original_expression: str
    canonical_expression: str
    canonical_hash: str
    reject_reason: str = ""

    @property
    def passed(self) -> bool:
        return not bool(self.reject_reason)


@dataclass(frozen=True)
class _CanonNode:
    expr: str
    kind: str
    name: str = ""
    args: tuple["_CanonNode", ...] = ()
    value: Any = None
    reject_reason: str = ""


def canonicalize_expression(expression: str) -> CanonicalExpressionResult:
    text = str(expression or "").strip()
    if not text:
        return CanonicalExpressionResult(text, "", "", "empty_expression")
    node = parse_expression(text)
    canon = _canonicalize_node(node)
    expr = canon.expr
    return CanonicalExpressionResult(
        original_expression=text,
        canonical_expression=expr,
        canonical_hash=_hash(expr) if expr else "",
        reject_reason=canon.reject_reason,
    )


def _canonicalize_node(node: ExpressionNode) -> _CanonNode:
    if isinstance(node, LiteralNode):
        return _CanonNode(expr=_format_literal(node.value), kind="literal", value=node.value)
    if isinstance(node, FieldNode):
        return _CanonNode(expr=str(node.name), kind="field", name=str(node.name))
    if isinstance(node, UnaryOpNode):
        child = _canonicalize_node(node.operand)
        if child.reject_reason:
            return child
        if node.op == "+":
            return child
        if node.op == "-":
            return _CanonNode(
                expr=f"mul(-1,{child.expr})",
                kind="func",
                name="mul",
                args=(_literal_node(-1), child),
            )
        return _CanonNode("", "invalid", reject_reason=f"unsupported_unary:{node.op}")
    if isinstance(node, BinaryOpNode):
        name = _BINARY_TO_FUNC.get(node.op)
        if not name:
            return _CanonNode("", "invalid", reject_reason=f"unsupported_binary:{node.op}")
        return _canonicalize_call(name, (_canonicalize_node(node.left), _canonicalize_node(node.right)))
    if isinstance(node, FunctionCallNode):
        if node.named_args:
            args = tuple(_canonicalize_node(arg) for arg in node.args) + tuple(
                _canonicalize_node(arg) for _, arg in sorted(node.named_args.items())
            )
        else:
            args = tuple(_canonicalize_node(arg) for arg in node.args)
        return _canonicalize_call(_ALIASES.get(node.name, node.name), args)
    return _CanonNode("", "invalid", reject_reason=f"unsupported_node:{type(node).__name__}")


def _canonicalize_call(name: str, args: tuple[_CanonNode, ...]) -> _CanonNode:
    for arg in args:
        if arg.reject_reason:
            return arg

    if name in _IDEMPOTENT_REJECT and len(args) == 1 and args[0].kind == "func" and args[0].name == name:
        return _CanonNode("", "invalid", reject_reason=f"nested_idempotent:{name}")

    if name == "reverse" and len(args) == 1 and args[0].kind == "func" and args[0].name == "reverse" and args[0].args:
        return args[0].args[0]

    if name == "sub" and len(args) == 2 and args[0].expr == args[1].expr:
        return _CanonNode("", "invalid", reject_reason="self_subtraction")
    if name in {"add", "sub"} and len(args) == 2 and _is_literal(args[1], 0):
        return args[0]
    if name == "add" and len(args) == 2 and _is_literal(args[0], 0):
        return args[1]
    if name == "mul" and len(args) == 2:
        if _is_literal(args[0], 0) or _is_literal(args[1], 0):
            return _CanonNode("", "invalid", reject_reason="zero_multiplication")
        if _is_literal(args[0], 1):
            return args[1]
        if _is_literal(args[1], 1):
            return args[0]
    if name == "div" and len(args) == 2:
        if _is_literal(args[1], 0):
            return _CanonNode("", "invalid", reject_reason="division_by_zero")
        if args[0].expr == args[1].expr:
            return _CanonNode("", "invalid", reject_reason="self_division")
        if _is_literal(args[1], 1):
            return args[0]
    if name in {"power", "signed_power"} and len(args) == 2:
        if _is_literal(args[1], 1):
            return args[0]
        if _is_literal(args[1], 0):
            return _literal_node(1)
    if name in {"max", "min"} and len(args) == 2 and args[0].expr == args[1].expr:
        return args[0]
    if name == "ts_delay" and len(args) == 2 and _is_literal(args[1], 0):
        return args[0]
    if name in {"ts_mean", "ts_min", "ts_max", "ts_median"} and len(args) == 2 and _is_literal(args[1], 1):
        return args[0]
    if name == "ts_delta" and len(args) == 2 and _is_literal(args[1], 0):
        return _literal_node(0)
    if name == "regression_neut" and len(args) == 2 and args[0].expr == args[1].expr:
        return _CanonNode("", "invalid", reject_reason="self_regression_neut")
    if name == "ts_corr" and len(args) >= 2 and args[0].expr == args[1].expr:
        return _CanonNode("", "invalid", reject_reason="self_ts_corr")
    if name == "ts_regression" and len(args) >= 2 and args[0].expr == args[1].expr:
        return _CanonNode("", "invalid", reject_reason="self_ts_regression")
    if name in _SELF_COMPARISON_REJECT and len(args) == 2 and args[0].expr == args[1].expr:
        return _CanonNode("", "invalid", reject_reason="self_comparison")
    if name in {"if_else", "trade_when"} and len(args) == 3 and args[1].expr == args[2].expr:
        return args[1]

    ordered = tuple(sorted(args, key=lambda x: x.expr)) if name in _COMMUTATIVE else args
    expr = f"{name}({','.join(arg.expr for arg in ordered)})"
    return _CanonNode(expr=expr, kind="func", name=name, args=ordered)


def _literal_node(value: Any) -> _CanonNode:
    return _CanonNode(expr=_format_literal(value), kind="literal", value=value)


def _is_literal(node: _CanonNode, value: float) -> bool:
    if node.kind != "literal":
        return False
    try:
        return float(node.value) == float(value)
    except Exception:
        return False


def _format_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return repr(value)
    return repr(value)


def _hash(expression: str) -> str:
    return hashlib.sha1(str(expression).encode("utf-8")).hexdigest()
