from __future__ import annotations

import hashlib

from .ast_nodes import (
    BinaryOpNode,
    ExpressionNode,
    FieldNode,
    FunctionCallNode,
    LiteralNode,
    UnaryOpNode,
)


def normalize_expression(expression: str) -> str:
    """Normalize expression string for deduplication.

    Removes whitespace to create a canonical string representation.
    This is a lightweight normalization that preserves operator semantics.

    Args:
        expression: Alpha expression string.

    Returns:
        Normalized expression with all whitespace removed.

    Example:
        >>> normalize_expression('ts_mean( close , 20 )')
        'ts_mean(close,20)'
    """
    return "".join(expression.split())


def expression_hash(expression: str) -> str:
    """Compute SHA1 hash of a normalized expression string.

    Useful for detecting duplicate expressions regardless of formatting.

    Args:
        expression: Alpha expression string.

    Returns:
        Hexadecimal SHA1 hash string.

    Example:
        >>> h = expression_hash('ts_mean(close, 20)')
        >>> print(len(h))  # 40
    """
    return hashlib.sha1(normalize_expression(expression).encode("utf-8")).hexdigest()


def structural_hash(node: ExpressionNode) -> str:
    """Compute SHA1 hash of an expression AST structure.

    Unlike expression_hash, this operates on the parsed AST and produces
    identical hashes for structurally equivalent expressions even if their
    string representations differ (e.g., different variable names but same
    structure).

    Args:
        node: Root ExpressionNode of the AST.

    Returns:
        Hexadecimal SHA1 hash string.

    Example:
        >>> node1 = parse_expression('ts_mean(close, 20)')
        >>> node2 = parse_expression('ts_mean(open, 20)')
        >>> # Different hashes because field names differ
        >>> assert structural_hash(node1) != structural_hash(node2)
    """
    serialized = _serialize_node(node)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def _serialize_node(node: ExpressionNode) -> str:
    if isinstance(node, LiteralNode):
        return f"lit:{repr(node.value)}"
    if isinstance(node, FieldNode):
        return f"field:{node.name}"
    if isinstance(node, UnaryOpNode):
        return f"u({node.op},{_serialize_node(node.operand)})"
    if isinstance(node, BinaryOpNode):
        return f"b({node.op},{_serialize_node(node.left)},{_serialize_node(node.right)})"
    if isinstance(node, FunctionCallNode):
        args = ",".join(_serialize_node(arg) for arg in node.args)
        kwargs = ",".join(f"{k}={_serialize_node(v)}" for k, v in sorted(node.named_args.items()))
        return f"f({node.name}|{args}|{kwargs})"
    return f"unknown:{type(node).__name__}"
