from __future__ import annotations

from ..validators import validate_portability


def prefilter_candidates(
    expressions: list[str],
    max_operator_count: int = 8,
    max_field_count: int = 3,
) -> tuple[list[str], dict[str, tuple[str, ...]]]:
    passed: list[str] = []
    failed: dict[str, tuple[str, ...]] = {}
    for expr in expressions:
        result = validate_portability(
            expr,
            max_operator_count=max_operator_count,
            max_field_count=max_field_count,
        )
        if result.is_valid:
            passed.append(expr)
        else:
            failed[expr] = result.reasons
    return passed, failed
