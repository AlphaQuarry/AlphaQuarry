from __future__ import annotations

from dataclasses import dataclass

from .ast_nodes import (
    BinaryOpNode,
    ExpressionNode,
    FieldNode,
    FunctionCallNode,
    LiteralNode,
    UnaryOpNode,
)
from .parser import parse_expression


@dataclass(frozen=True)
class ExpressionStats:
    operator_count: int
    field_count: int
    unique_fields: tuple[str, ...]


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    reasons: tuple[str, ...]
    stats: ExpressionStats


def expression_stats(expression: str | ExpressionNode) -> ExpressionStats:
    node = parse_expression(expression) if isinstance(expression, str) else expression
    operator_count, fields = _walk(node)
    unique = tuple(sorted(set(fields)))
    return ExpressionStats(operator_count=operator_count, field_count=len(unique), unique_fields=unique)


def validate_portability(
    expression: str | ExpressionNode,
    max_operator_count: int = 8,
    max_field_count: int = 3,
) -> ValidationResult:
    stats = expression_stats(expression)
    reasons: list[str] = []
    if stats.operator_count > max_operator_count:
        reasons.append(f"operator_count>{max_operator_count}")
    if stats.field_count > max_field_count:
        reasons.append(f"field_count>{max_field_count}")
    return ValidationResult(is_valid=(len(reasons) == 0), reasons=tuple(reasons), stats=stats)


def _walk(node: ExpressionNode) -> tuple[int, list[str]]:
    if isinstance(node, LiteralNode):
        return 0, []
    if isinstance(node, FieldNode):
        return 0, [node.name]
    if isinstance(node, UnaryOpNode):
        n, f = _walk(node.operand)
        return n + 1, f
    if isinstance(node, BinaryOpNode):
        nl, fl = _walk(node.left)
        nr, fr = _walk(node.right)
        return nl + nr + 1, fl + fr
    if isinstance(node, FunctionCallNode):
        total_ops = 1
        fields: list[str] = []
        for arg in node.args:
            n, f = _walk(arg)
            total_ops += n
            fields.extend(f)
        for arg in node.named_args.values():
            n, f = _walk(arg)
            total_ops += n
            fields.extend(f)
        return total_ops, fields
    return 0, []
