from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pandas as pd

from alpha_mining.mining.candidate_prefilter import CandidatePrefilter
from alpha_mining.mining.expression_canonicalizer import canonicalize_expression
from alpha_mining.mining.expression_layers import (
    LayeredBuilderConfig,
    LayeredExpressionBuilder,
)
from alpha_mining.mining.field_universe import build_field_universe
from alpha_mining.panel_store import PanelStore
from alpha_mining.workflow.universe_store import (
    append_universe_expressions,
    load_seen_expression_hashes_for_universe,
    load_universe_expression_registry,
)


def test_canonicalizer_rejects_static_trivial_expressions() -> None:
    cases = {
        "sub(close, close)": "self_subtraction",
        "mul(close, 0)": "zero_multiplication",
        "mul(0, close)": "zero_multiplication",
        "div(close, 0)": "division_by_zero",
        "ts_corr(close, close, 5)": "self_ts_corr",
        "regression_neut(close, close)": "self_regression_neut",
        "greater(close, close)": "self_comparison",
    }
    for expr, reason in cases.items():
        result = canonicalize_expression(expr)
        assert not result.passed
        assert result.reject_reason == reason


def test_canonicalizer_simplifies_safe_identities_and_zero_like_passes_prefilter() -> None:
    assert canonicalize_expression("div(close, 1)").canonical_expression == "close"
    assert canonicalize_expression("power(close, 1)").canonical_expression == "close"
    assert canonicalize_expression("signed_power(close, 1)").canonical_expression == "close"
    assert canonicalize_expression("power(close, 0)").canonical_expression == "1"
    assert canonicalize_expression("signed_power(close, 0)").canonical_expression == "1"
    assert canonicalize_expression("max(close, close)").canonical_expression == "close"
    assert canonicalize_expression("min(close, close)").canonical_expression == "close"
    assert canonicalize_expression("ts_delta(close, 0)").canonical_expression == "0"

    prefilter = CandidatePrefilter(field_kinds={"close": "scalar", "open": "scalar"}, max_depth=6)
    assert prefilter.check("if_else(greater(close, open), close, zero_like(close))").passed


def test_canonicalizer_simplifies_high_value_trivial_windows_and_redundant_branches() -> None:
    assert canonicalize_expression("ts_delay(close, 0)").canonical_expression == "close"
    assert canonicalize_expression("ts_mean(close, 1)").canonical_expression == "close"
    assert canonicalize_expression("ts_min(close, 1)").canonical_expression == "close"
    assert canonicalize_expression("ts_max(close, 1)").canonical_expression == "close"
    assert canonicalize_expression("ts_median(close, 1)").canonical_expression == "close"
    assert canonicalize_expression("if_else(greater(close, open), close, close)").canonical_expression == "close"
    assert canonicalize_expression("trade_when(greater(close, open), close, close)").canonical_expression == "close"

    result = canonicalize_expression("ts_regression(close, close, 5)")
    assert not result.passed
    assert result.reject_reason == "self_ts_regression"


def test_prefilter_rejects_canonical_constant_outputs() -> None:
    prefilter = CandidatePrefilter(field_kinds={"close": "scalar"}, max_depth=6)
    result = prefilter.check("power(close, 0)")
    assert not result.passed
    assert result.reject_stage == "canonical"
    assert result.reject_reason == "constant_canonical_expression"
    assert result.canonical_expression == "1"


def test_layered_v2_gate_uses_zero_like_instead_of_self_subtraction() -> None:
    rows = []
    for date in pd.date_range("2024-01-01", periods=8):
        for code in ["A", "B", "C"]:
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "close": 10.0,
                    "amount": 100.0,
                    "circ_mv": 1e9,
                    "pct_chg": 0.01,
                    "industry": "bank",
                    "universe": 1,
                }
            )
    panel = PanelStore.from_long_frame(pd.DataFrame(rows), group_fields=["industry"])
    universe = build_field_universe(panel, explicit_include_fields=["close", "amount"], group_fields=["industry"])
    out = LayeredExpressionBuilder().build(
        universe,
        config=LayeredBuilderConfig(
            max_order=3,
            max_candidates=32,
            layer_budgets={"L0": 2, "L1": 8, "L2": 4, "L3": 8, "L4": 0},
            windows=(5,),
        ),
    )
    l3 = [item.expression for item in out if item.layer == "L3"]
    assert l3
    assert all("zero_like(" in expr for expr in l3)
    assert all("sub(" not in expr for expr in l3)


def test_universe_registry_dedupes_by_computed_canonical_hash() -> None:
    base_dir = Path("data") / f"_canonical_dedupe_test_{uuid.uuid4().hex}"
    universe_name = "cn_test"
    try:
        incoming = pd.DataFrame(
            {
                "expression": ["add(close, 0)", "close"],
                "source": ["unit", "unit"],
            }
        )
        added = append_universe_expressions(incoming, base_dir=base_dir, universe_name=universe_name)
        assert len(added) == 1
        assert added.iloc[0]["expression"] == "close"
        assert added.iloc[0]["original_expression"] == "add(close, 0)"
        assert added.iloc[0]["simplified_expression"] == "close"
        assert added.iloc[0]["canonical_hash"]

        registry = load_universe_expression_registry(base_dir=base_dir, universe_name=universe_name)
        seen = load_seen_expression_hashes_for_universe(base_dir=base_dir, universe_name=universe_name)
        assert len(registry) == 1
        assert registry.iloc[0]["canonical_hash"] in seen
        assert registry.iloc[0]["lint_status"] == "simplified"
        assert "lint_warning_reason" in registry.columns
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)
