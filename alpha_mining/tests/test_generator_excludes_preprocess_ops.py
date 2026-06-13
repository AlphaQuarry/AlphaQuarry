from __future__ import annotations

from alpha_mining.mining.explore import (
    DeepExploreConfig,
    RandomExpressionGenerator,
    build_operator_search_space,
)


def test_random_expression_generator_excludes_winsorize_and_ts_backfill_from_random_ops() -> None:
    gen = RandomExpressionGenerator(
        scalar_fields=["close"],
        group_fields=[],
        vector_fields=[],
        windows=[5],
        max_depth=2,
        random_seed=1,
    )

    assert "winsorize" not in {sig.name for sig in gen.unary_ops}
    assert "ts_backfill" not in {sig.name for sig in gen.ts_ops}


def test_operator_search_space_uses_preprocessed_base_without_depth1_winsorize() -> None:
    items = build_operator_search_space(
        available_fields=["close"],
        available_groups=[],
        config=DeepExploreConfig(max_depth=1, max_candidates=20),
    )
    expressions = [expr for _source, expr in items]

    assert "winsorize(ts_backfill(close, 120), 4.0)" in expressions
    assert "winsorize(close, 4.0)" not in expressions
