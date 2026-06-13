from __future__ import annotations

from ..hashing import expression_hash, normalize_expression


def deduplicate_expressions(expressions: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for expr in expressions:
        key = expression_hash(normalize_expression(expr))
        if key in seen:
            continue
        seen.add(key)
        out.append(expr)
    return out
