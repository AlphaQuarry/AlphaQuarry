from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExpressionNode:
    """Base AST node for alpha expressions."""


@dataclass(frozen=True)
class LiteralNode(ExpressionNode):
    value: Any


@dataclass(frozen=True)
class FieldNode(ExpressionNode):
    name: str


@dataclass(frozen=True)
class UnaryOpNode(ExpressionNode):
    op: str
    operand: ExpressionNode


@dataclass(frozen=True)
class BinaryOpNode(ExpressionNode):
    op: str
    left: ExpressionNode
    right: ExpressionNode


@dataclass(frozen=True)
class FunctionCallNode(ExpressionNode):
    name: str
    args: tuple[ExpressionNode, ...] = field(default_factory=tuple)
    named_args: dict[str, ExpressionNode] = field(default_factory=dict)
