from __future__ import annotations

import ast
from functools import lru_cache

from .ast_nodes import (
    BinaryOpNode,
    ExpressionNode,
    FieldNode,
    FunctionCallNode,
    LiteralNode,
    UnaryOpNode,
)


class ExpressionParseError(ValueError):
    """Raised when expression string cannot be parsed into project AST.

    This error indicates a syntax error in the alpha expression string.
    Common causes include:
    - Invalid operator names
    - Mismatched parentheses
    - Unsupported syntax (e.g., chained comparisons)
    - Invalid literal values
    """


_BINARY_OPS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Pow: "**",
    ast.Mod: "%",
}

_UNARY_OPS = {
    ast.UAdd: "+",
    ast.USub: "-",
}

_COMPARE_TO_FUNC = {
    ast.Gt: "greater",
    ast.GtE: "greater_equal",
    ast.Lt: "less",
    ast.LtE: "less_equal",
    ast.Eq: "equal",
    ast.NotEq: "not_equal",
}


@lru_cache(maxsize=4096)
def parse_expression(expression: str) -> ExpressionNode:
    """Parse function-style alpha expression into custom AST.

    Supports function calls, binary/unary operators, and comparisons.
    Results are cached for performance.

    Args:
        expression: Alpha expression string (e.g., 'ts_mean(close, 20)').

    Returns:
        Root ExpressionNode of the parsed AST.

    Raises:
        ExpressionParseError: If the expression has invalid syntax.

    Examples:
        >>> node = parse_expression('ts_mean(close, 20)')
        >>> node = parse_expression('cs_rank(close / open - 1)')
        >>> node = parse_expression('where(volume > 1000000, close, nan)')
    """
    try:
        module = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionParseError(f"Invalid expression syntax: {expression}") from exc
    return _convert_node(module.body)


def _convert_node(node: ast.AST) -> ExpressionNode:
    if isinstance(node, ast.Constant):
        return LiteralNode(value=node.value)

    # Keep a small fallback for very old Python ASTs without touching deprecated
    # ast.Num/Str/NameConstant aliases on modern runtimes.
    legacy_type = type(node).__name__
    if legacy_type == "Num":  # py<3.8 compatibility
        return LiteralNode(value=getattr(node, "n"))
    if legacy_type == "Str":  # py<3.8 compatibility
        return LiteralNode(value=getattr(node, "s"))
    if legacy_type == "NameConstant":  # py<3.8 compatibility
        return LiteralNode(value=getattr(node, "value"))

    if isinstance(node, ast.Name):
        return FieldNode(name=node.id)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ExpressionParseError(f"Unsupported unary operator: {op_type.__name__}")
        return UnaryOpNode(op=_UNARY_OPS[op_type], operand=_convert_node(node.operand))

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINARY_OPS:
            raise ExpressionParseError(f"Unsupported binary operator: {op_type.__name__}")
        return BinaryOpNode(
            op=_BINARY_OPS[op_type],
            left=_convert_node(node.left),
            right=_convert_node(node.right),
        )

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ExpressionParseError("Chained comparisons are not supported in MVP parser")
        op_type = type(node.ops[0])
        if op_type not in _COMPARE_TO_FUNC:
            raise ExpressionParseError(f"Unsupported comparison operator: {op_type.__name__}")
        return FunctionCallNode(
            name=_COMPARE_TO_FUNC[op_type],
            args=(_convert_node(node.left), _convert_node(node.comparators[0])),
            named_args={},
        )

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExpressionParseError("Only direct function calls are supported in MVP parser")
        positional = tuple(_convert_node(arg) for arg in node.args)
        named = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise ExpressionParseError("kwargs unpacking is not supported")
            named[kw.arg] = _convert_node(kw.value)
        return FunctionCallNode(name=node.func.id, args=positional, named_args=named)

    raise ExpressionParseError(f"Unsupported AST node type: {type(node).__name__}")
