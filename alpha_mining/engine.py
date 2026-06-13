from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .ast_nodes import (
    BinaryOpNode,
    ExpressionNode,
    FieldNode,
    FunctionCallNode,
    LiteralNode,
    UnaryOpNode,
)
from .hashing import structural_hash
from .panel_store import PanelStore
from .parser import parse_expression
from .registry import OperatorRegistry, build_default_registry


@dataclass
class ExpressionEngine:
    """Evaluate expression strings/AST against PanelStore using OperatorRegistry.

    The engine parses alpha expressions into an AST and evaluates them against
    a PanelStore containing date x code panels. It supports operator profiling,
    result caching via structural hashing, and safe division handling.

    Attributes:
        panel_store: Container for date x code panels used during evaluation.
        registry: Operator name -> callable registry for expression operators.
        node_cache: Cache mapping structural hashes to evaluated results.

    Example:
        >>> from alpha_mining import ExpressionEngine, PanelStore
        >>> store = PanelStore.from_long_frame(df, date_col='date', code_col='code')
        >>> engine = ExpressionEngine(panel_store=store)
        >>> result = engine.eval('ts_mean(close, 20)')
    """

    panel_store: PanelStore
    registry: OperatorRegistry = field(default_factory=build_default_registry)
    node_cache: dict[str, Any] = field(default_factory=dict)
    _operator_profile: dict[str, dict[str, float]] = field(default_factory=dict)
    _profile_enabled: bool = False

    def eval(self, expression: str | ExpressionNode, use_cache: bool = True) -> Any:
        """Evaluate an alpha expression against the panel store.

        Args:
            expression: Alpha expression string (e.g., 'ts_mean(close, 20)')
                or pre-parsed ExpressionNode.
            use_cache: Whether to use structural hash caching. Defaults to True.

        Returns:
            Evaluated result, typically a pd.DataFrame or pd.Series with
            date x code dimensions.

        Raises:
            ExpressionParseError: If the expression string cannot be parsed.
            KeyError: If an operator is not found in the registry.
        """
        node = parse_expression(expression) if isinstance(expression, str) else expression
        cache_key = None
        if use_cache:
            cache_key = structural_hash(node)
            if cache_key in self.node_cache:
                return self.node_cache[cache_key]

        result = self._eval_node(node)
        if use_cache and cache_key is not None:
            self.node_cache[cache_key] = result
        return result

    def clear_cache(self) -> None:
        """Clear the expression evaluation cache.

        Removes all cached results, forcing re-evaluation on next call.
        """
        self.node_cache.clear()

    def enable_operator_profiling(self, reset: bool = True) -> None:
        """Enable operator performance profiling.

        When enabled, the engine tracks execution time for each operator call.

        Args:
            reset: Whether to clear existing profile data. Defaults to True.
        """
        self._profile_enabled = True
        if reset:
            self._operator_profile.clear()

    def disable_operator_profiling(self) -> None:
        """Disable operator performance profiling.

        Profile data is preserved; use reset_operator_profile() to clear it.
        """
        self._profile_enabled = False

    def reset_operator_profile(self) -> None:
        """Clear all operator profiling data."""
        self._operator_profile.clear()

    def get_operator_profile(self) -> pd.DataFrame:
        """Get operator performance profiling results.

        Returns:
            DataFrame with columns: operator, count, total_sec, avg_sec.
            Sorted by total_sec descending.

        Example:
            >>> engine.enable_operator_profiling()
            >>> engine.eval('ts_mean(close, 20)')
            >>> profile = engine.get_operator_profile()
            >>> print(profile.head())
        """
        if not self._operator_profile:
            return pd.DataFrame(columns=["operator", "count", "total_sec", "avg_sec"])
        rows: list[dict[str, float | int | str]] = []
        for op, stats in self._operator_profile.items():
            count = int(stats.get("count", 0))
            total = float(stats.get("total_sec", 0.0))
            rows.append(
                {
                    "operator": op,
                    "count": count,
                    "total_sec": total,
                    "avg_sec": (total / count) if count > 0 else np.nan,
                }
            )
        out = pd.DataFrame(rows)
        out = out.sort_values(["total_sec", "count"], ascending=[False, False]).reset_index(drop=True)
        return out

    def _eval_node(self, node: ExpressionNode) -> Any:
        if isinstance(node, LiteralNode):
            return node.value

        if isinstance(node, FieldNode):
            return self.panel_store.get_field(node.name)

        if isinstance(node, UnaryOpNode):
            value = self._eval_node(node.operand)
            if node.op == "+":
                return value
            if node.op == "-":
                return -value
            raise ValueError(f"Unsupported unary op: {node.op}")

        if isinstance(node, BinaryOpNode):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return _binary_op(node.op, left, right)

        if isinstance(node, FunctionCallNode):
            fn = self.registry.get(node.name)
            args = [self._eval_node(arg) for arg in node.args]
            kwargs = {k: self._eval_node(v) for k, v in node.named_args.items()}
            if not self._profile_enabled:
                return _clean_nonfinite(fn(*args, **kwargs))

            t0 = time.perf_counter()
            out = _clean_nonfinite(fn(*args, **kwargs))
            dt = time.perf_counter() - t0
            stats = self._operator_profile.setdefault(node.name, {"count": 0, "total_sec": 0.0})
            stats["count"] = int(stats["count"]) + 1
            stats["total_sec"] = float(stats["total_sec"]) + dt
            return out

        raise TypeError(f"Unsupported expression node type: {type(node).__name__}")


def _clean_nonfinite(result: Any) -> Any:
    """Replace inf/-inf with nan in the result of a binary operation."""
    if isinstance(result, pd.DataFrame):
        return result.replace([np.inf, -np.inf], np.nan)
    if isinstance(result, pd.Series):
        return result.replace([np.inf, -np.inf], np.nan)
    if isinstance(result, np.ndarray):
        return np.where(np.isfinite(result), result, np.nan)
    if isinstance(result, (int, float)):
        return result if np.isfinite(result) else np.nan
    return result


def _binary_op(op: str, left: Any, right: Any) -> Any:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/":
        if isinstance(right, (pd.DataFrame, pd.Series)):
            right = right.replace(0, np.nan)
        elif isinstance(right, np.ndarray):
            right = np.where(right == 0, np.nan, right)
        elif isinstance(right, (int, float)) and right == 0:
            right = np.nan
        return _clean_nonfinite(left / right)
    if op == "**":
        try:
            return _clean_nonfinite(left**right)
        except (ZeroDivisionError, OverflowError, ValueError):
            if isinstance(left, pd.DataFrame):
                return pd.DataFrame(np.nan, index=left.index, columns=left.columns)
            if isinstance(left, pd.Series):
                return pd.Series(np.nan, index=left.index)
            if isinstance(left, np.ndarray):
                return np.full_like(left, np.nan, dtype=float)
            return np.nan
    if op == "%":
        if isinstance(right, (pd.DataFrame, pd.Series)):
            right = right.replace(0, np.nan)
        elif isinstance(right, np.ndarray):
            right = np.where(right == 0, np.nan, right)
        elif isinstance(right, (int, float)) and right == 0:
            right = np.nan
        return _clean_nonfinite(left % right)
    raise ValueError(f"Unsupported binary op: {op}")
