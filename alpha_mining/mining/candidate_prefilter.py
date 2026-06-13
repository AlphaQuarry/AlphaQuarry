from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..ast_nodes import (
    BinaryOpNode,
    ExpressionNode,
    FieldNode,
    FunctionCallNode,
    LiteralNode,
    UnaryOpNode,
)
from ..hashing import expression_hash, normalize_expression
from ..parser import ExpressionParseError, parse_expression
from ..registry import build_default_registry
from ..validators import expression_stats
from .expression_canonicalizer import canonicalize_expression
from .field_universe import is_leakage_field
from .operator_signatures import (
    LITERAL,
    SCALAR,
    WINDOW,
    OperatorSignatureRegistry,
    build_default_operator_signature_registry,
)


@dataclass(frozen=True)
class CandidatePrefilterResult:
    expression: str
    passed: bool
    reject_stage: str = ""
    reject_reason: str = ""
    normalized_expression: str = ""
    expression_hash: str = ""
    canonical_expression: str = ""
    canonical_hash: str = ""
    lint_status: str = ""
    lint_warning_reason: str = ""
    fields: tuple[str, ...] = ()
    operators: tuple[str, ...] = ()
    operator_count: int = 0
    field_count: int = 0
    depth: int = 0


class CandidatePrefilter:
    def __init__(
        self,
        field_kinds: dict[str, str],
        max_operator_count: int = 8,
        max_field_count: int = 3,
        max_depth: int = 3,
        existing_hashes: Iterable[str] = (),
        reject_naked_division: bool = True,
        signature_registry: OperatorSignatureRegistry | None = None,
        preprocess_operator_exemptions: set[str] | None = None,
    ) -> None:
        self.field_kinds = dict(field_kinds)
        self.max_operator_count = int(max_operator_count)
        self.max_field_count = int(max_field_count)
        self.max_depth = int(max_depth)
        self.existing_hashes = {str(x) for x in existing_hashes if str(x)}
        self.reject_naked_division = bool(reject_naked_division)
        self.signature_registry = signature_registry or build_default_operator_signature_registry()
        self.operator_registry = build_default_registry()
        self.preprocess_operator_exemptions = set(preprocess_operator_exemptions or set())

    def check(self, expression: str) -> CandidatePrefilterResult:
        expr = str(expression or "").strip()
        normalized = normalize_expression(expr) if expr else ""
        expr_hash = expression_hash(expr) if expr else ""
        if not expr:
            return self._reject(expr, "parse", "empty_expression", normalized, expr_hash)
        try:
            original_node = parse_expression(expr)
        except ExpressionParseError as exc:
            return self._reject(expr, "parse", str(exc), normalized, expr_hash)
        if self.reject_naked_division and _has_binary_operator(original_node, "/"):
            return self._reject(expr, "type", "naked_division", normalized, expr_hash)
        try:
            canonical = canonicalize_expression(expr)
        except ExpressionParseError as exc:
            return self._reject(expr, "parse", str(exc), normalized, expr_hash)
        canonical_expr = canonical.canonical_expression
        canonical_hash = canonical.canonical_hash
        if canonical.reject_reason:
            return self._reject(
                expr,
                "canonical",
                canonical.reject_reason,
                normalized,
                expr_hash,
                canonical_expr=canonical_expr,
                canonical_hash=canonical_hash,
            )
        if expr_hash in self.existing_hashes:
            return self._reject(
                expr,
                "dedup",
                "duplicate_expression_hash",
                normalized,
                expr_hash,
                canonical_expr=canonical_expr,
                canonical_hash=canonical_hash,
            )
        if canonical_hash in self.existing_hashes:
            return self._reject(
                expr,
                "dedup",
                "duplicate_canonical_hash",
                normalized,
                expr_hash,
                canonical_expr=canonical_expr,
                canonical_hash=canonical_hash,
            )
        try:
            node = parse_expression(canonical_expr)
        except ExpressionParseError as exc:
            return self._reject(
                expr,
                "parse",
                str(exc),
                normalized,
                expr_hash,
                canonical_expr=canonical_expr,
                canonical_hash=canonical_hash,
            )
        try:
            stats = expression_stats(node)
        except Exception as exc:
            return self._reject(
                expr,
                "stats",
                str(exc),
                normalized,
                expr_hash,
                canonical_expr=canonical_expr,
                canonical_hash=canonical_hash,
            )
        operators = tuple(_collect_operators(node))
        fields = stats.unique_fields
        if not fields:
            return self._reject(
                expr,
                "canonical",
                "constant_canonical_expression",
                normalized,
                expr_hash,
                fields,
                operators,
                int(stats.operator_count),
                stats.field_count,
                _depth(node),
                canonical_expr,
                canonical_hash,
            )
        depth = _depth(node)
        preprocess_wrapper_count = _count_strict_preprocess_wrappers(node)
        adjusted_operator_count = int(stats.operator_count)
        adjusted_depth = int(depth)
        if self.preprocess_operator_exemptions and {
            "ts_backfill",
            "winsorize",
        }.issubset(self.preprocess_operator_exemptions):
            adjusted_operator_count = max(0, adjusted_operator_count - 2 * preprocess_wrapper_count)
            if preprocess_wrapper_count > 0:
                adjusted_depth = max(0, adjusted_depth - 2)
        if adjusted_operator_count > self.max_operator_count:
            return self._reject(
                expr,
                "limits",
                f"operator_count>{self.max_operator_count}",
                normalized,
                expr_hash,
                fields,
                operators,
                adjusted_operator_count,
                stats.field_count,
                adjusted_depth,
                canonical_expr,
                canonical_hash,
            )
        if stats.field_count > self.max_field_count:
            return self._reject(
                expr,
                "limits",
                f"field_count>{self.max_field_count}",
                normalized,
                expr_hash,
                fields,
                operators,
                adjusted_operator_count,
                stats.field_count,
                adjusted_depth,
                canonical_expr,
                canonical_hash,
            )
        if adjusted_depth > self.max_depth:
            return self._reject(
                expr,
                "limits",
                f"depth>{self.max_depth}",
                normalized,
                expr_hash,
                fields,
                operators,
                adjusted_operator_count,
                stats.field_count,
                adjusted_depth,
                canonical_expr,
                canonical_hash,
            )
        for field in fields:
            if field not in self.field_kinds:
                return self._reject(
                    expr,
                    "fields",
                    f"unknown_field:{field}",
                    normalized,
                    expr_hash,
                    fields,
                    operators,
                    adjusted_operator_count,
                    stats.field_count,
                    adjusted_depth,
                    canonical_expr,
                    canonical_hash,
                )
            if is_leakage_field(field):
                return self._reject(
                    expr,
                    "leakage",
                    f"leakage_field:{field}",
                    normalized,
                    expr_hash,
                    fields,
                    operators,
                    adjusted_operator_count,
                    stats.field_count,
                    adjusted_depth,
                    canonical_expr,
                    canonical_hash,
                )
        try:
            out_type = self._infer_type(node)
        except ValueError as exc:
            return self._reject(
                expr,
                "type",
                str(exc),
                normalized,
                expr_hash,
                fields,
                operators,
                adjusted_operator_count,
                stats.field_count,
                adjusted_depth,
                canonical_expr,
                canonical_hash,
            )
        if out_type != SCALAR:
            return self._reject(
                expr,
                "type",
                f"output_type:{out_type}",
                normalized,
                expr_hash,
                fields,
                operators,
                adjusted_operator_count,
                stats.field_count,
                adjusted_depth,
                canonical_expr,
                canonical_hash,
            )
        lint_status = "simplified" if canonical_expr and canonical_expr != normalized else "passed"
        lint_warning = "canonical_simplified" if lint_status == "simplified" else ""
        return CandidatePrefilterResult(
            expr,
            True,
            "",
            "",
            normalized,
            expr_hash,
            canonical_expr,
            canonical_hash,
            lint_status,
            lint_warning,
            fields,
            operators,
            adjusted_operator_count,
            stats.field_count,
            adjusted_depth,
        )

    def _infer_type(self, node: ExpressionNode) -> str:
        if isinstance(node, LiteralNode):
            return LITERAL
        if isinstance(node, FieldNode):
            return self.field_kinds.get(node.name, "")
        if isinstance(node, UnaryOpNode):
            typ = self._infer_type(node.operand)
            if typ != SCALAR:
                raise ValueError(f"unary_{node.op}_requires_scalar")
            return SCALAR
        if isinstance(node, BinaryOpNode):
            if self.reject_naked_division and node.op == "/":
                raise ValueError("naked_division")
            left = self._infer_type(node.left)
            right = self._infer_type(node.right)
            if left != SCALAR or right != SCALAR:
                raise ValueError(f"binary_{node.op}_requires_scalar")
            return SCALAR
        if isinstance(node, FunctionCallNode):
            if not self.operator_registry.has(node.name):
                raise ValueError(f"unknown_operator:{node.name}")
            arg_types = tuple(self._infer_type(arg) for arg in node.args)
            sig = self.signature_registry.match(node.name, arg_types)
            if sig is None:
                if not self.signature_registry.has(node.name):
                    raise ValueError(f"missing_signature:{node.name}")
                expected_arities = sorted(
                    {len(spec.input_types) for spec in self.signature_registry.get_all(node.name)}
                )
                if len(arg_types) not in expected_arities:
                    raise ValueError(f"arity_mismatch:{node.name}")
                expected = next(
                    spec.input_types
                    for spec in self.signature_registry.get_all(node.name)
                    if len(spec.input_types) == len(arg_types)
                )
                mismatch = next(
                    (
                        f"{got}!={want}"
                        for got, want in zip(arg_types, expected)
                        if got != want and not (want == WINDOW and got == LITERAL)
                    ),
                    "",
                )
                raise ValueError(f"type_mismatch:{node.name}:{mismatch}")
            return sig.output_type
        raise ValueError(f"unsupported_node:{type(node).__name__}")

    def _reject(
        self,
        expr: str,
        stage: str,
        reason: str,
        normalized: str,
        expr_hash: str,
        fields: tuple[str, ...] = (),
        operators: tuple[str, ...] = (),
        operator_count: int = 0,
        field_count: int = 0,
        depth: int = 0,
        canonical_expr: str = "",
        canonical_hash: str = "",
    ) -> CandidatePrefilterResult:
        return CandidatePrefilterResult(
            expr,
            False,
            stage,
            reason,
            normalized,
            expr_hash,
            canonical_expr,
            canonical_hash,
            "rejected",
            str(reason or ""),
            fields,
            operators,
            operator_count,
            field_count,
            depth,
        )


def _collect_operators(node: ExpressionNode) -> list[str]:
    if isinstance(node, FunctionCallNode):
        out = [node.name]
        for arg in node.args:
            out.extend(_collect_operators(arg))
        for arg in node.named_args.values():
            out.extend(_collect_operators(arg))
        return out
    if isinstance(node, BinaryOpNode):
        return [node.op] + _collect_operators(node.left) + _collect_operators(node.right)
    if isinstance(node, UnaryOpNode):
        return [node.op] + _collect_operators(node.operand)
    return []


def _has_binary_operator(node: ExpressionNode, operator: str) -> bool:
    if isinstance(node, BinaryOpNode):
        return (
            node.op == operator
            or _has_binary_operator(node.left, operator)
            or _has_binary_operator(node.right, operator)
        )
    if isinstance(node, UnaryOpNode):
        return _has_binary_operator(node.operand, operator)
    if isinstance(node, FunctionCallNode):
        return any(_has_binary_operator(arg, operator) for arg in node.args) or any(
            _has_binary_operator(arg, operator) for arg in node.named_args.values()
        )
    return False


def _depth(node: ExpressionNode) -> int:
    if isinstance(node, (LiteralNode, FieldNode)):
        return 0
    if isinstance(node, UnaryOpNode):
        return 1 + _depth(node.operand)
    if isinstance(node, BinaryOpNode):
        return 1 + max(_depth(node.left), _depth(node.right))
    if isinstance(node, FunctionCallNode):
        child_depths = [_depth(arg) for arg in node.args] + [_depth(arg) for arg in node.named_args.values()]
        return 1 + (max(child_depths) if child_depths else 0)
    return 0


def _count_strict_preprocess_wrappers(node: ExpressionNode) -> int:
    count = 0
    if _is_strict_preprocess_wrapper(node):
        count += 1
    if isinstance(node, UnaryOpNode):
        count += _count_strict_preprocess_wrappers(node.operand)
    elif isinstance(node, BinaryOpNode):
        count += _count_strict_preprocess_wrappers(node.left)
        count += _count_strict_preprocess_wrappers(node.right)
    elif isinstance(node, FunctionCallNode):
        for arg in node.args:
            count += _count_strict_preprocess_wrappers(arg)
        for arg in node.named_args.values():
            count += _count_strict_preprocess_wrappers(arg)
    return count


def _is_strict_preprocess_wrapper(node: ExpressionNode) -> bool:
    if not isinstance(node, FunctionCallNode):
        return False
    if node.name != "winsorize" or len(node.args) < 2:
        return False
    inner = node.args[0]
    if not isinstance(inner, FunctionCallNode):
        return False
    if inner.name != "ts_backfill" or len(inner.args) < 2:
        return False
    return (
        isinstance(inner.args[0], FieldNode)
        and isinstance(inner.args[1], LiteralNode)
        and isinstance(node.args[1], LiteralNode)
    )
