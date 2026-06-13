from __future__ import annotations

import random


def build_pair_expression_space(
    scalar_fields: list[str],
    group_fields: list[str],
    windows: list[int],
    max_pairs: int = 200,
    random_seed: int = 42,
    field_expression_map: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    fields = [str(x) for x in scalar_fields if str(x)]
    if len(fields) < 2:
        return []
    rng = random.Random(int(random_seed))
    pairs = [(a, b) for i, a in enumerate(fields) for b in fields[i + 1 :]]
    rng.shuffle(pairs)
    pairs = pairs[: max(0, int(max_pairs))]
    wins = [int(w) for w in windows if int(w) > 0] or [5, 10, 22]
    out: list[tuple[str, str]] = []
    expr_map = dict(field_expression_map or {})
    for a, b in pairs:
        pair_key = f"{a}|{b}"
        a_expr = expr_map.get(a, a)
        b_expr = expr_map.get(b, b)
        d = rng.choice(wins)
        out.extend(
            [
                (f"pair_spread:{pair_key}", f"rank(sub({a_expr}, {b_expr}))"),
                (f"pair_zspread:{pair_key}", f"zscore(sub({a_expr}, {b_expr}))"),
                (
                    f"pair_ts_zspread:{pair_key}",
                    f"ts_zscore(sub({a_expr}, {b_expr}), {d})",
                ),
                (
                    f"pair_ts_rank_spread:{pair_key}",
                    f"ts_rank(sub({a_expr}, {b_expr}), {d})",
                ),
                (f"pair_ratio:{pair_key}", f"rank(div({a_expr}, {b_expr}))"),
                (f"pair_corr:{pair_key}", f"ts_corr({a_expr}, {b_expr}, {d})"),
            ]
        )
        if group_fields:
            out.append(
                (
                    f"pair_group_spread:{pair_key}",
                    f"group_zscore(sub({a_expr}, {b_expr}), {group_fields[0]})",
                )
            )
    return out
