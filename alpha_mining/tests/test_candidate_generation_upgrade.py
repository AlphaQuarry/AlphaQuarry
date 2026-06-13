from __future__ import annotations

import json
import pandas as pd
import shutil
import uuid
from pathlib import Path

from alpha_mining.config import AlphaMiningConfig, AlphaSimulationConfig
from alpha_mining.mining.candidate_prefilter import CandidatePrefilter
from alpha_mining.mining.explore import (
    DeepExploreConfig,
    FieldSpec,
    build_signature_aware_search_space,
)
from alpha_mining.mining.field_universe import build_field_universe
from alpha_mining.panel_store import PanelStore
from alpha_mining.workflow.closed_loop import ClosedLoopConfig, _generate_candidates


def _panel_store() -> PanelStore:
    rows = []
    for date in pd.date_range("2024-01-01", periods=5):
        for code, industry in [("A", "bank"), ("B", "tech")]:
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "close": 10.0,
                    "amount": 100.0,
                    "pct_chg": 0.01,
                    "ret_exec_cc_audit_1d": 0.02,
                    "target": 1.0,
                    "industry": industry,
                    "universe": 1,
                }
            )
    df = pd.DataFrame(rows)
    return PanelStore.from_long_frame(df, date_col="date", code_col="code", group_fields=["industry"])


def test_field_universe_excludes_leakage_fields() -> None:
    universe = build_field_universe(_panel_store(), group_fields=["industry"])
    assert "close" in universe.scalar_fields
    assert "industry" in universe.group_fields
    assert "pct_chg" not in universe.scalar_fields
    assert "ret_exec_cc_audit_1d" not in universe.scalar_fields
    assert "target" not in universe.scalar_fields
    assert "universe" not in universe.scalar_fields


def test_field_universe_include_fields_limits_scalar_pool_but_keeps_groups() -> None:
    universe = build_field_universe(
        _panel_store(),
        explicit_include_fields=["amount"],
        group_fields=["industry"],
        search_field_universe=["amount", "close"],
    )
    assert universe.scalar_fields == ["amount"]
    assert universe.group_fields == ["industry"]
    assert "close" in universe.excluded_fields


def test_candidate_prefilter_rejects_leakage_naked_division_and_type_mismatch() -> None:
    field_kinds = {
        "close": "scalar",
        "amount": "scalar",
        "industry": "group",
        "vec": "vector",
    }
    prefilter = CandidatePrefilter(field_kinds=field_kinds, max_operator_count=8, max_field_count=4, max_depth=4)
    assert prefilter.check("rank(close)").passed
    assert prefilter.check("(close) / (amount)").reject_reason == "naked_division"
    assert prefilter.check("rank(industry)").reject_reason.startswith("type_mismatch")
    assert prefilter.check("ts_rank(vec, 22)").reject_reason.startswith("type_mismatch")
    assert prefilter.check("rank(pct_chg)").reject_reason.startswith("unknown_field")


def test_signature_generator_does_not_emit_vec_norm_or_naked_division() -> None:
    exprs = build_signature_aware_search_space(
        available_fields={"close", "vec"},
        available_groups={"industry"},
        field_specs=[
            FieldSpec("close", "scalar"),
            FieldSpec("vec", "vector"),
            FieldSpec("industry", "group"),
        ],
        config=DeepExploreConfig(max_candidates=50, random_seed=7),
    )
    text = "\n".join(expr for _, expr in exprs)
    assert "vec_norm" not in text
    assert " / " not in text


def test_closed_loop_generate_candidates_writes_candidate_artifacts() -> None:
    base_dir = Path("data") / f"_candidate_upgrade_{uuid.uuid4().hex}"
    cfg = ClosedLoopConfig(
        universe_base_dir=str(base_dir),
        universe_name="ut",
        group_fields=("industry",),
        search_mode="operator_only",
        max_eval_expressions=10,
        deep_explore_config=DeepExploreConfig(max_candidates=20, random_seed=3),
        mining_config=AlphaMiningConfig(simulation=AlphaSimulationConfig(delay=1, universe="universe")),
    )
    try:
        expressions, meta = _generate_candidates(_panel_store(), cfg)
        assert expressions
        assert int(meta["passed_candidate_count"]) == len(expressions)
        assert meta["candidates_path"]
        assert pd.read_csv(meta["candidates_path"]).shape[0] >= len(expressions)
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_closed_loop_include_fields_limits_generated_candidate_fields() -> None:
    base_dir = Path("data") / f"_candidate_include_{uuid.uuid4().hex}"
    cfg = ClosedLoopConfig(
        universe_base_dir=str(base_dir),
        universe_name="ut",
        group_fields=("industry",),
        include_fields=("amount",),
        search_field_universe=("amount", "close"),
        search_mode="operator_only",
        max_eval_expressions=10,
        deep_explore_config=DeepExploreConfig(max_candidates=20, random_seed=3),
        mining_config=AlphaMiningConfig(simulation=AlphaSimulationConfig(delay=1, universe="universe")),
    )
    try:
        expressions, meta = _generate_candidates(_panel_store(), cfg)
        assert expressions
        passed = meta["candidate_df"][meta["candidate_df"]["prefilter_status"] == "pass"]
        assert not passed.empty
        for fields in passed["fields"].dropna().astype(str):
            used = {x.strip() for x in fields.split(",") if x.strip()}
            assert used <= {"amount", "industry"}
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_closed_loop_layered_v2_writes_layer_metadata_artifacts() -> None:
    base_dir = Path("data") / f"_candidate_layered_v2_{uuid.uuid4().hex}"
    cfg = ClosedLoopConfig(
        universe_base_dir=str(base_dir),
        universe_name="ut",
        group_fields=("industry",),
        include_fields=("amount", "close", "circ_mv"),
        search_mode="layered_v2",
        layer_max_order=4,
        layer_max_candidates=48,
        layer_budgets={"L0": 4, "L1": 10, "L2": 12, "L3": 8, "L4": 8},
        max_eval_expressions=16,
        deep_explore_config=DeepExploreConfig(windows=(3,), max_candidates=20, random_seed=3),
        mining_config=AlphaMiningConfig(
            max_operator_count=12,
            max_field_count=4,
            simulation=AlphaSimulationConfig(delay=1, universe="universe"),
        ),
    )
    try:
        expressions, meta = _generate_candidates(_panel_store(), cfg)
        assert expressions
        candidates = pd.read_csv(meta["candidates_path"])
        expected_cols = {
            "layer",
            "layer_family",
            "parent_expression",
            "parent_hash",
            "mutation_type",
            "fragment_hash",
            "feedback_source",
            "builder_source",
            "layer_order",
        }
        assert expected_cols <= set(candidates.columns)
        selected = candidates[candidates["expression"].isin(expressions)]
        assert not selected.empty
        assert selected["layer"].fillna("").astype(str).str.startswith("L").any()
        assert selected["layer_order"].fillna(0).astype(int).max() >= 2
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_candidate_artifacts_include_reproducible_scoring_and_lineage_fields() -> None:
    base_dir = Path("data") / f"_candidate_artifacts_{uuid.uuid4().hex}"
    cfg = ClosedLoopConfig(
        universe_base_dir=str(base_dir),
        universe_name="ut",
        group_fields=("industry",),
        include_fields=("amount", "close", "circ_mv"),
        search_mode="layered_v2",
        layer_max_order=4,
        layer_max_candidates=48,
        layer_budgets={"L0": 4, "L1": 10, "L2": 12, "L3": 8, "L4": 8},
        max_eval_expressions=16,
        deep_explore_config=DeepExploreConfig(windows=(3,), max_candidates=20, random_seed=3),
        mining_config=AlphaMiningConfig(
            max_operator_count=12,
            max_field_count=4,
            simulation=AlphaSimulationConfig(delay=1, universe="universe"),
        ),
    )
    try:
        expressions, meta = _generate_candidates(_panel_store(), cfg)
        assert expressions
        candidates = pd.read_csv(meta["candidates_path"])
        required_cols = {
            "candidate_id",
            "expression",
            "canonical_expression",
            "canonical_hash",
            "layer",
            "layer_family",
            "parent_expression",
            "parent_hash",
            "mutation_type",
            "fragment_hash",
            "feedback_source",
            "builder_source",
            "layer_order",
            "candidate_score",
            "feedback_score",
            "fragment_score",
            "parent_score",
            "mutation_type_score",
            "novelty_score",
            "selection_bucket",
        }
        assert required_cols <= set(candidates.columns)

        selected = candidates[candidates["expression"].isin(expressions)]
        assert not selected.empty
        assert selected["canonical_hash"].fillna("").astype(str).str.len().gt(0).all()
        assert selected["candidate_score"].notna().any()
        metadata_rows = [json.loads(str(value)) for value in candidates["metadata_json"].dropna().astype(str)]
        bucket_rows = [row for row in metadata_rows if row.get("bucket_expression")]
        assert bucket_rows
        assert any("bucket_source_field" in row for row in bucket_rows)
        assert any("bucket_source_family" in row for row in bucket_rows)
        assert any("bucket_range" in row for row in bucket_rows)
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)
