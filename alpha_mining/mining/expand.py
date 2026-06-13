from __future__ import annotations

from itertools import product
from typing import Any

from ..schema import AlphaTemplate


def expand_template(template: AlphaTemplate, value_pool: dict[str, list[Any]]) -> list[str]:
    """
    Expand template expression with placeholder values.

    Placeholder convention:
    - expression: "ts_mean({field}, {d})"
    - value_pool: {"field": ["close", "volume"], "d": [5, 22]}
    """
    merged_pool: dict[str, list[Any]] = {}
    if template.placeholders:
        merged_pool.update({k: list(v) for k, v in template.placeholders.items()})
    if value_pool:
        merged_pool.update({k: list(v) for k, v in value_pool.items()})

    keys = sorted(merged_pool.keys())
    if not keys:
        return [template.expression]
    if any(len(merged_pool[k]) == 0 for k in keys):
        return []

    results: list[str] = []
    for vals in product(*(merged_pool[k] for k in keys)):
        kwargs = {k: v for k, v in zip(keys, vals)}
        results.append(template.expression.format(**kwargs))
    return results
