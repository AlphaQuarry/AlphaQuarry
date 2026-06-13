from __future__ import annotations

import pandas as pd

from alpha_mining.engine import ExpressionEngine
from alpha_mining.mining.candidate_prefilter import CandidatePrefilter
from alpha_mining.mining.expression_layers import (
    LayeredBuilderConfig,
    LayeredExpressionBuilder,
)
from alpha_mining.mining.field_preprocessing import (
    FieldPreprocessConfig,
    FieldExpressionFactory,
)
from alpha_mining.mining.field_universe import FieldUniverse
from alpha_mining.mining.explore import FieldSpec
from alpha_mining.mining.pair_generator import build_pair_expression_space
from alpha_mining.panel_store import PanelStore


def test_field_expression_factory_wraps_scalar_and_skips_excluded_fields() -> None:
    factory = FieldExpressionFactory(
        FieldPreprocessConfig(
            enabled=True,
            ts_backfill_window=120,
            winsorize_std=4.0,
            exclude_fields=("pct_chg", "industry"),
            exclude_roles=("group",),
        )
    )

    assert factory.expression_for("close", kind="scalar") == "winsorize(ts_backfill(close, 120), 4.0)"
    assert factory.expression_for("pct_chg", kind="scalar") == "pct_chg"
    assert factory.expression_for("industry", kind="group") == "industry"


def test_layered_builder_uses_preprocessed_l0_and_does_not_add_winsorize_l1() -> None:
    universe = FieldUniverse(
        specs=(
            FieldSpec(name="close", field_kind="scalar", categories=("price",)),
            FieldSpec(name="volume", field_kind="scalar", categories=("liquidity",)),
            FieldSpec(name="industry", field_kind="group", categories=("group",)),
        ),
        excluded_fields=(),
    )
    cfg = LayeredBuilderConfig(max_order=1, max_candidates=40)
    candidates = LayeredExpressionBuilder().build(universe, config=cfg)
    expressions = [c.expression for c in candidates]

    assert "winsorize(ts_backfill(close, 120), 4.0)" in expressions
    assert "close" not in expressions
    assert not any(expr.startswith("winsorize(close") for expr in expressions)
    assert not any("winsorize(winsorize(" in expr for expr in expressions)


def test_pair_generator_uses_preprocessed_expressions_but_keeps_raw_pair_key() -> None:
    factory = FieldExpressionFactory()
    pairs = build_pair_expression_space(
        scalar_fields=["close", "volume"],
        group_fields=["industry"],
        windows=[5],
        field_expression_map=factory.expression_map(["close", "volume"]),
        max_pairs=1,
    )

    keys = [key for key, _expr in pairs]
    expressions = [expr for _key, expr in pairs]
    assert any(key.endswith("close|volume") for key in keys)
    assert all("winsorize(ts_backfill(close, 120), 4.0)" in expr or "group_zscore" in expr for expr in expressions)
    assert any("winsorize(ts_backfill(volume, 120), 4.0)" in expr for expr in expressions)


def test_candidate_prefilter_exempts_strict_preprocess_wrapper_from_limits() -> None:
    expr = "rank(winsorize(ts_backfill(close, 120), 4.0))"
    prefilter = CandidatePrefilter(
        field_kinds={"close": "scalar"},
        max_operator_count=1,
        max_depth=1,
        preprocess_operator_exemptions={"ts_backfill", "winsorize"},
    )

    result = prefilter.check(expr)

    assert result.passed, result.reject_reason
    assert result.operator_count == 1
    assert result.depth == 1


def test_preprocessed_expression_evaluates_with_real_engine() -> None:
    rows = []
    for dt, value in zip(pd.date_range("2024-04-01", periods=3), [1.0, None, 3.0]):
        rows.append({"date": dt, "code": "A", "close": value})
        rows.append({"date": dt, "code": "B", "close": value * 2 if value is not None else None})
    engine = ExpressionEngine(PanelStore.from_long_frame(pd.DataFrame(rows)))
    out = engine.eval("winsorize(ts_backfill(close, 120), 4.0)")

    assert out.loc[pd.Timestamp("2024-04-02"), "A"] == 1.0
    assert out.loc[pd.Timestamp("2024-04-02"), "B"] == 2.0
