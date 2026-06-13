from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from alpha_mining.config import AlphaMiningConfig, AlphaSimulationConfig
from alpha_mining.mining.candidate_planner import (
    _apply_sample_prefilter,
    _field_role_map,
)
from alpha_mining.mining.candidate_ranker import CandidateRanker, CandidateRankerConfig
from alpha_mining.mining.feedback_sampler import FeedbackSampler, FeedbackSamplerConfig
from alpha_mining.mining.explore import DeepExploreConfig
from alpha_mining.mining.field_universe import build_field_universe
from alpha_mining.panel_store import PanelStore
from alpha_mining.workflow.closed_loop import ClosedLoopConfig, _generate_candidates


def test_candidate_ranker_uses_feedback_but_keeps_exploration_rows() -> None:
    candidates = pd.DataFrame(
        [
            {
                "expression": "rank(close)",
                "family": "operator",
                "fields": "close",
                "operators": "rank",
                "operator_count": 1,
                "field_count": 1,
                "depth": 1,
                "prefilter_status": "pass",
            },
            {
                "expression": "rank(volume)",
                "family": "operator",
                "fields": "volume",
                "operators": "rank",
                "operator_count": 1,
                "field_count": 1,
                "depth": 1,
                "prefilter_status": "pass",
            },
            {
                "expression": "ts_mean(amount, 5)",
                "family": "operator",
                "fields": "amount",
                "operators": "ts_mean",
                "operator_count": 1,
                "field_count": 1,
                "depth": 1,
                "prefilter_status": "pass",
            },
        ]
    )
    hints = {
        "enabled": True,
        "field_weights": {"volume": 1.0},
        "operator_weights": {"rank": 1.0},
        "family_weights": {},
    }

    ranked = CandidateRanker(CandidateRankerConfig(min_explore_ratio=0.34)).rank(candidates, hints, max_eval=2)

    assert ranked.iloc[0]["expression"] == "rank(volume)"
    assert "explore" in set(ranked["selection_bucket"])
    assert "candidate_score" in ranked.columns
    assert ranked["feedback_score"].max() > 0


def test_candidate_ranker_uses_optional_quality_signals_when_present() -> None:
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "low_quality",
                "expression": "rank(close)",
                "family": "operator",
                "fields": "close",
                "operators": "rank",
                "operator_count": 1,
                "field_count": 1,
                "depth": 1,
                "prefilter_status": "pass",
                "field_profile_score": 0.0,
                "recipe_score": 0.0,
                "role_pair_score": 0.0,
                "bucket_quality_score": 0.0,
                "gate_quality_score": 0.0,
                "sample_quality_score": 0.0,
                "cost_score": 0.0,
            },
            {
                "candidate_id": "high_quality",
                "expression": "rank(amount)",
                "family": "operator",
                "fields": "amount",
                "operators": "rank",
                "operator_count": 1,
                "field_count": 1,
                "depth": 1,
                "prefilter_status": "pass",
                "field_profile_score": 1.0,
                "recipe_score": 1.0,
                "role_pair_score": 1.0,
                "bucket_quality_score": 1.0,
                "gate_quality_score": 1.0,
                "sample_quality_score": 1.0,
                "cost_score": 0.0,
            },
        ]
    )

    ranked = CandidateRanker(CandidateRankerConfig(min_explore_ratio=0.0)).rank(candidates, {}, max_eval=2)

    assert ranked.iloc[0]["candidate_id"] == "high_quality"
    assert {
        "field_profile_score",
        "recipe_score",
        "sample_quality_score",
        "cost_score",
    } <= set(ranked.columns)


def test_candidate_ranker_missing_optional_quality_signals_stays_compatible() -> None:
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "a",
                "expression": "rank(close)",
                "family": "operator",
                "fields": "close",
                "operators": "rank",
                "operator_count": 1,
                "field_count": 1,
                "depth": 1,
                "prefilter_status": "pass",
            },
            {
                "candidate_id": "b",
                "expression": "rank(volume)",
                "family": "operator",
                "fields": "volume",
                "operators": "rank",
                "operator_count": 1,
                "field_count": 1,
                "depth": 1,
                "prefilter_status": "pass",
            },
        ]
    )

    ranked = CandidateRanker(CandidateRankerConfig(min_explore_ratio=0.0)).rank(candidates, {}, max_eval=2)

    assert len(ranked) == 2
    assert "candidate_score" in ranked.columns


def test_feedback_sampler_builds_positive_and_negative_hints() -> None:
    scoreboard = pd.DataFrame(
        {
            "expression": ["rank(volume)", "rank(close)"],
            "fields": ["volume", "close"],
            "operators": ["rank", "rank"],
            "family": ["operator", "operator"],
            "score_total": [2.0, -1.0],
            "turnover_long_only_mean": [0.2, 1.2],
        }
    )

    hints = FeedbackSampler(FeedbackSamplerConfig(enabled=True)).build_weight_hints(scoreboard)

    assert hints["field_weights"]["volume"] > 0
    assert hints["negative_field_weights"]["close"] > 0
    assert hints["negative_operator_weights"]["rank"] > 0


def test_feedback_sampler_builds_layer_window_and_group_hints() -> None:
    scoreboard = pd.DataFrame(
        {
            "expression": ["group_neutralize(rank(volume), industry)", "rank(close)"],
            "fields": ["volume,industry", "close"],
            "operators": ["group_neutralize,rank", "rank"],
            "family": ["risk", "cross_sectional"],
            "layer": ["L4", "L1"],
            "windows": ["5,22", ""],
            "groups": ["industry", ""],
            "score_total": [2.0, -1.0],
        }
    )

    hints = FeedbackSampler(FeedbackSamplerConfig(enabled=True)).build_weight_hints(scoreboard)

    assert hints["layer_weights"]["L4"] > 0
    assert hints["window_weights"]["5"] > 0
    assert hints["group_weights"]["industry"] > 0
    assert hints["negative_layer_weights"]["L1"] > 0


def test_feedback_sampler_builds_fragment_parent_and_mutation_hints() -> None:
    scoreboard = pd.DataFrame(
        {
            "expression": ["rank(close)", "rank(volume)"],
            "fields": ["close", "volume"],
            "operators": ["rank", "rank"],
            "family": ["operator", "operator"],
            "fragment_hash": ["frag_a", "frag_b"],
            "parent_hash": ["parent_a", "parent_b"],
            "mutation_type": ["window_shift", "group_swap"],
            "score_total": [1.5, -0.2],
        }
    )
    hints = FeedbackSampler(FeedbackSamplerConfig(enabled=True)).build_weight_hints(scoreboard)
    assert hints["fragment_weights"]["frag_a"] > 0
    assert hints["parent_weights"]["parent_a"] > 0
    assert hints["mutation_type_weights"]["window_shift"] > 0
    assert hints["negative_fragment_weights"]["frag_b"] > 0
    assert hints["negative_parent_weights"]["parent_b"] > 0
    assert hints["negative_mutation_type_weights"]["group_swap"] > 0


def test_candidate_ranker_uses_layer_feedback_and_exposes_layer_balance() -> None:
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "expression": "rank(close)",
                "family": "cross_sectional",
                "layer": "L1",
                "fields": "close",
                "operators": "rank",
                "operator_count": 1,
                "field_count": 1,
                "depth": 1,
                "prefilter_status": "pass",
            },
            {
                "candidate_id": "c2",
                "expression": "group_neutralize(rank(volume), industry)",
                "family": "risk",
                "layer": "L4",
                "fields": "volume,industry",
                "groups": "industry",
                "operators": "group_neutralize,rank",
                "windows": "5",
                "operator_count": 2,
                "field_count": 2,
                "depth": 2,
                "prefilter_status": "pass",
            },
            {
                "candidate_id": "c3",
                "expression": "ts_zscore(rank(amount), 5)",
                "family": "time_series",
                "layer": "L2",
                "fields": "amount",
                "operators": "ts_zscore,rank",
                "windows": "5",
                "operator_count": 2,
                "field_count": 1,
                "depth": 2,
                "prefilter_status": "pass",
            },
        ]
    )
    hints = {
        "enabled": True,
        "layer_weights": {"L4": 1.0},
        "group_weights": {"industry": 1.0},
        "window_weights": {"5": 0.5},
    }

    ranked = CandidateRanker(CandidateRankerConfig(min_explore_ratio=0.34)).rank(candidates, hints, max_eval=2)

    assert ranked.iloc[0]["candidate_id"] == "c2"
    assert "layer_balance_score" in ranked.columns
    assert "explore" in set(ranked["selection_bucket"])


def test_candidate_ranker_layer_quota_preserves_explore_bucket() -> None:
    rows = []
    for idx in range(20):
        layer = "L3" if idx < 2 else "L1"
        rows.append(
            {
                "candidate_id": f"c{idx}",
                "expression": f"rank(field_{idx})",
                "family": "cross_sectional",
                "factor_family": "price_volume",
                "layer": layer,
                "fields": f"field_{idx}",
                "operators": "rank",
                "windows": "5",
                "operator_count": 1,
                "field_count": 1,
                "depth": 1,
                "prefilter_status": "pass",
            }
        )
    df = pd.DataFrame(rows)
    ranked = CandidateRanker(
        CandidateRankerConfig(
            min_explore_ratio=0.25,
            use_layer_quota=True,
            layer_selection_min_ratio={"L3": 0.25},
            use_factor_family_quota=False,
        )
    ).rank(df, feedback_hints={}, max_eval=8)

    assert (ranked["layer"].astype(str) == "L3").sum() >= 2
    assert "explore" in set(ranked["selection_bucket"])


def test_candidate_ranker_applies_l0_max_and_structure_floor() -> None:
    rows = []
    for idx in range(10):
        rows.append(
            {
                "candidate_id": f"l0_{idx}",
                "expression": f"field_{idx}",
                "family": "raw",
                "factor_family": "raw",
                "layer": "L0",
                "fields": f"field_{idx}",
                "operators": "",
                "operator_count": 0,
                "field_count": 1,
                "depth": 0,
                "prefilter_status": "pass",
            }
        )
    rows.extend(
        [
            {
                "candidate_id": "bucket_l4",
                "expression": "group_rank(close, bucket(rank(circ_mv), '0,1,0.2'))",
                "family": "layered",
                "factor_family": "layered",
                "layer": "L4",
                "fields": "close,circ_mv",
                "operators": "group_rank,bucket,rank",
                "operator_count": 3,
                "field_count": 2,
                "depth": 3,
                "prefilter_status": "pass",
                "metadata_json": json.dumps({"bucket_expression": "bucket(rank(circ_mv), '0,1,0.2')"}),
            },
            {
                "candidate_id": "gate_l3",
                "expression": "if_else(amount, rank(close), 0)",
                "family": "layered",
                "factor_family": "layered",
                "layer": "L3",
                "fields": "amount,close",
                "operators": "if_else,rank",
                "operator_count": 2,
                "field_count": 2,
                "depth": 2,
                "prefilter_status": "pass",
                "metadata_json": json.dumps({"gate_family": "liquidity_activity"}),
            },
        ]
    )

    ranked = CandidateRanker(
        CandidateRankerConfig(
            min_explore_ratio=0.0,
            use_factor_family_quota=False,
            use_layer_quota=True,
            layer_selection_max_ratio={"L0": 0.10},
            structure_selection_min_ratio={"bucket": 0.10, "gate": 0.10},
        )
    ).rank(pd.DataFrame(rows), feedback_hints={}, max_eval=10)

    assert (ranked["layer"].astype(str) == "L0").sum() <= 1
    assert "bucket_l4" in set(ranked["candidate_id"])
    assert "gate_l3" in set(ranked["candidate_id"])


def test_candidate_ranker_uses_fragment_parent_and_mutation_scores() -> None:
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "m1",
                "expression": "rank(close)",
                "family": "feedback_mutation",
                "layer": "M1",
                "fields": "close",
                "operators": "rank",
                "fragment_hash": "frag_good",
                "parent_hash": "parent_good",
                "mutation_type": "window_shift",
                "operator_count": 1,
                "field_count": 1,
                "depth": 1,
                "prefilter_status": "pass",
            },
            {
                "candidate_id": "m2",
                "expression": "rank(volume)",
                "family": "feedback_mutation",
                "layer": "M1",
                "fields": "volume",
                "operators": "rank",
                "fragment_hash": "frag_bad",
                "parent_hash": "parent_bad",
                "mutation_type": "group_swap",
                "operator_count": 1,
                "field_count": 1,
                "depth": 1,
                "prefilter_status": "pass",
            },
        ]
    )
    hints = {
        "enabled": True,
        "fragment_weights": {"frag_good": 1.0},
        "parent_weights": {"parent_good": 1.0},
        "mutation_type_weights": {"window_shift": 1.0},
        "negative_fragment_weights": {"frag_bad": 1.0},
        "negative_parent_weights": {"parent_bad": 1.0},
        "negative_mutation_type_weights": {"group_swap": 1.0},
    }
    ranked = CandidateRanker(CandidateRankerConfig(min_explore_ratio=0.0)).rank(candidates, hints, max_eval=2)
    assert ranked.iloc[0]["candidate_id"] == "m1"
    assert {"fragment_score", "parent_score", "mutation_type_score"} <= set(ranked.columns)


def test_candidate_ranker_interleaves_layers_when_feedback_is_empty() -> None:
    rows = []
    for i in range(8):
        rows.append(
            {
                "candidate_id": f"l0_{i}",
                "expression": f"field_{i}",
                "family": "moneyflow",
                "layer": "L0",
                "fields": f"field_{i}",
                "operators": "",
                "operator_count": 0,
                "field_count": 1,
                "depth": 0,
                "prefilter_status": "pass",
            }
        )
    for layer in ["L1", "L2", "L3", "L4"]:
        rows.append(
            {
                "candidate_id": layer.lower(),
                "expression": f"expr_{layer}",
                "family": "layered",
                "layer": layer,
                "fields": "field_x",
                "operators": "rank",
                "operator_count": 2,
                "field_count": 1,
                "depth": 2,
                "prefilter_status": "pass",
            }
        )

    ranked = CandidateRanker(CandidateRankerConfig(min_explore_ratio=0.20)).rank(pd.DataFrame(rows), {}, max_eval=5)

    assert len(set(ranked["layer"].astype(str))) >= 4
    assert ranked.head(5)["layer"].astype(str).tolist().count("L0") <= 2


def test_sample_prefilter_skips_missing_fields_without_rejecting_candidate() -> None:
    rows = []
    for date in pd.date_range("2024-01-01", periods=3):
        for code in ["A", "B"]:
            rows.append({"date": date, "code": code, "circ_mv": 1e9})
    store = PanelStore.from_long_frame(pd.DataFrame(rows))
    candidate_df = pd.DataFrame(
        [
            {
                "candidate_id": "candidate_1",
                "expression": "rank(close)",
                "prefilter_status": "pass",
                "reject_stage": "",
                "reject_reason": "",
            }
        ]
    )
    cfg = ClosedLoopConfig(enable_sample_prefilter=True)

    out, sample_df = _apply_sample_prefilter(candidate_df, store, cfg)

    assert out.iloc[0]["sample_status"] == "skipped"
    assert out.iloc[0]["sample_reject_reason"] == "missing_sample_fields:close"
    assert out.iloc[0]["prefilter_status"] == "pass"
    assert sample_df.iloc[0]["sample_status"] == "skipped"


def test_sample_prefilter_respects_max_evaluation_budget() -> None:
    rows = []
    for date in pd.date_range("2024-01-01", periods=3):
        for code in ["A", "B"]:
            rows.append({"date": date, "code": code, "close": 10.0, "volume": 100.0})
    store = PanelStore.from_long_frame(pd.DataFrame(rows))
    candidate_df = pd.DataFrame(
        [
            {
                "candidate_id": "candidate_1",
                "expression": "rank(close)",
                "prefilter_status": "pass",
            },
            {
                "candidate_id": "candidate_2",
                "expression": "rank(volume)",
                "prefilter_status": "pass",
            },
        ]
    )
    cfg = ClosedLoopConfig(enable_sample_prefilter=True, sample_prefilter_max_evaluations=1)

    out, sample_df = _apply_sample_prefilter(candidate_df, store, cfg)

    assert out.iloc[0]["sample_status"] in {"pass", "reject"}
    assert out.iloc[1]["sample_status"] == "skipped_budget"
    assert out.iloc[1]["sample_reject_reason"] == "sample_prefilter_budget_not_selected"
    assert out.iloc[1]["prefilter_status"] == "pass"
    assert sample_df["sample_status"].tolist()[-1] == "skipped_budget"


def test_sample_prefilter_stratifies_budget_across_layers_and_features() -> None:
    rows = []
    for date in pd.date_range("2024-01-01", periods=4):
        for idx, code in enumerate(["A", "B", "C", "D"], start=1):
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "close": float(idx),
                    "amount": float(idx * 10),
                    "circ_mv": float(idx),
                }
            )
    store = PanelStore.from_long_frame(pd.DataFrame(rows))
    candidates = []
    for idx in range(5):
        candidates.append(
            {
                "candidate_id": f"l1_{idx}",
                "expression": "rank(close)",
                "prefilter_status": "pass",
                "layer": "L1",
                "metadata_json": "{}",
            }
        )
    candidates.extend(
        [
            {
                "candidate_id": "l4_bucket",
                "expression": "group_rank(close, bucket(rank(circ_mv), '0,1,0.5'))",
                "prefilter_status": "pass",
                "layer": "L4",
                "metadata_json": json.dumps({"bucket_expression": "bucket(rank(circ_mv), '0,1,0.5')"}),
            },
            {
                "candidate_id": "l3_gate",
                "expression": "rank(amount)",
                "prefilter_status": "pass",
                "layer": "L3",
                "metadata_json": json.dumps({"gate_family": "liquidity_activity"}),
            },
            {
                "candidate_id": "recipe",
                "expression": "rank(close)",
                "prefilter_status": "pass",
                "layer": "L2",
                "metadata_json": json.dumps({"recipe_family": "liquidity_shock"}),
            },
        ]
    )
    cfg = ClosedLoopConfig(
        enable_sample_prefilter=True,
        sample_prefilter_max_evaluations=3,
        sample_prefilter_stratified=True,
        bucket_quality_min_median_group_size=2,
        bucket_quality_min_group_count=2,
    )

    out, sample_df = _apply_sample_prefilter(pd.DataFrame(candidates), store, cfg)

    evaluated_ids = set(sample_df[sample_df["sample_status"].isin(["pass", "reject"])]["candidate_id"])
    assert {"l4_bucket", "l3_gate", "recipe"} <= evaluated_ids
    assert "skipped_budget" in set(out["sample_status"].astype(str))


def test_sample_prefilter_writes_sample_and_bucket_quality_scores() -> None:
    rows = []
    for date in pd.date_range("2024-01-01", periods=5):
        for idx, code in enumerate(["A", "B", "C", "D"], start=1):
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "close": float(idx),
                    "circ_mv": float(idx),
                }
            )
    store = PanelStore.from_long_frame(pd.DataFrame(rows))
    candidate_df = pd.DataFrame(
        [
            {
                "candidate_id": "candidate_1",
                "expression": "rank(close)",
                "prefilter_status": "pass",
                "metadata_json": "{}",
            },
            {
                "candidate_id": "candidate_2",
                "expression": "group_rank(close, bucket(rank(circ_mv), '0,1,0.5'))",
                "prefilter_status": "pass",
                "metadata_json": json.dumps({"bucket_expression": "bucket(rank(circ_mv), '0,1,0.5')"}),
            },
        ]
    )
    cfg = ClosedLoopConfig(
        enable_sample_prefilter=True,
        bucket_quality_lite_enabled=True,
        bucket_quality_min_median_group_size=2,
        bucket_quality_min_group_count=2,
    )

    out, sample_df = _apply_sample_prefilter(candidate_df, store, cfg)

    assert {
        "sample_coverage",
        "sample_quality_score",
        "bucket_sample_quality_score",
    } <= set(out.columns)
    assert out.loc[out["candidate_id"] == "candidate_1", "sample_quality_score"].iloc[0] > 0
    bucket_row = out[out["candidate_id"] == "candidate_2"].iloc[0]
    assert bucket_row["bucket_sample_status"] == "pass"
    assert bucket_row["bucket_sample_quality_score"] > 0
    assert {
        "sample_quality_score",
        "bucket_sample_quality_score",
        "bucket_sample_status",
    } <= set(sample_df.columns)


def test_sample_prefilter_rejects_low_quality_composite_bucket_but_not_plain_bucket() -> None:
    rows = []
    for date in pd.date_range("2024-01-01", periods=4):
        for idx, code in enumerate(["A", "B", "C", "D"], start=1):
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "close": float(idx),
                    "circ_mv": float(idx),
                    "industry": "only_one",
                }
            )
    store = PanelStore.from_long_frame(pd.DataFrame(rows), group_fields=["industry"])
    bucket_expr = "group_cartesian_product(industry, bucket(rank(circ_mv), '0,1,0.5'))"
    candidate_df = pd.DataFrame(
        [
            {
                "candidate_id": "composite",
                "expression": f"group_rank(close, {bucket_expr})",
                "prefilter_status": "pass",
                "metadata_json": json.dumps({"bucket_expression": bucket_expr, "group_complexity": 2}),
            },
            {
                "candidate_id": "plain",
                "expression": "group_rank(close, bucket(rank(circ_mv), '0,1,0.5'))",
                "prefilter_status": "pass",
                "metadata_json": json.dumps({"bucket_expression": "bucket(rank(circ_mv), '0,1,0.5')"}),
            },
        ]
    )
    cfg = ClosedLoopConfig(
        enable_sample_prefilter=True,
        bucket_quality_lite_enabled=True,
        bucket_quality_min_median_group_size=99,
        bucket_quality_min_group_count=2,
        bucket_quality_reject_low_quality_composite=True,
        bucket_quality_reject_low_quality_plain=False,
    )

    out, sample_df = _apply_sample_prefilter(candidate_df, store, cfg)

    composite = out[out["candidate_id"] == "composite"].iloc[0]
    plain = out[out["candidate_id"] == "plain"].iloc[0]
    assert composite["prefilter_status"] == "reject"
    assert composite["reject_stage"] == "bucket_quality"
    assert plain["prefilter_status"] == "pass"
    assert {
        "bucket_sample_quality_status",
        "bucket_sample_nan_group_ratio",
        "bucket_sample_is_composite",
    } <= set(sample_df.columns)


def test_sample_prefilter_reuses_bucket_quality_cache_within_batch() -> None:
    rows = []
    for date in pd.date_range("2024-01-01", periods=4):
        for idx, code in enumerate(["A", "B", "C", "D"], start=1):
            rows.append({"date": date, "code": code, "close": float(idx), "circ_mv": float(idx)})
    store = PanelStore.from_long_frame(pd.DataFrame(rows))
    metadata = json.dumps({"bucket_expression": "bucket(rank(circ_mv), '0,1,0.5')"})
    candidate_df = pd.DataFrame(
        [
            {
                "candidate_id": "a",
                "expression": "group_rank(close, bucket(rank(circ_mv), '0,1,0.5'))",
                "prefilter_status": "pass",
                "metadata_json": metadata,
            },
            {
                "candidate_id": "b",
                "expression": "group_zscore(close, bucket(rank(circ_mv), '0,1,0.5'))",
                "prefilter_status": "pass",
                "metadata_json": metadata,
            },
        ]
    )
    cfg = ClosedLoopConfig(
        enable_sample_prefilter=True,
        bucket_quality_lite_enabled=True,
        bucket_quality_max_evaluations=1,
        bucket_quality_min_median_group_size=2,
        bucket_quality_min_group_count=2,
    )

    out, _ = _apply_sample_prefilter(candidate_df, store, cfg)

    assert out["bucket_sample_status"].tolist() == ["pass", "pass"]
    assert out["bucket_sample_cache_hit"].tolist() == [False, True]


def test_field_role_map_uses_semantic_roles_not_first_catalog_category() -> None:
    store = PanelStore.from_long_frame(
        pd.DataFrame(
            [
                {
                    "date": "2024-01-01",
                    "code": "A",
                    "moneyflow_buy_lg_amount": 1.0,
                    "close": 2.0,
                },
                {
                    "date": "2024-01-01",
                    "code": "B",
                    "moneyflow_buy_lg_amount": 2.0,
                    "close": 3.0,
                },
            ]
        )
    )

    role_map = _field_role_map(build_field_universe(store))

    assert role_map["moneyflow_buy_lg_amount"] == "moneyflow"
    assert role_map["close"] == "price"


def test_generate_candidates_writes_feedback_hints_and_ranked_columns() -> None:
    rows = []
    for date in pd.date_range("2024-01-01", periods=8):
        for code, industry in [("A", "bank"), ("B", "tech"), ("C", "tech")]:
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "close": 10.0,
                    "volume": 100.0,
                    "pct_chg": 0.01,
                    "circ_mv": 1e9,
                    "industry": industry,
                    "universe": 1,
                }
            )
    store = PanelStore.from_long_frame(pd.DataFrame(rows), group_fields=["industry"])
    base_dir = Path("data") / f"_feedback_v1_{uuid.uuid4().hex}"
    universe = "ut"
    feedback_dir = base_dir / universe / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "expression": ["rank(volume)"],
            "fields": ["volume"],
            "operators": ["rank"],
            "family": ["operator"],
            "score_total": [1.0],
        }
    ).to_csv(feedback_dir / "expression_scoreboard.csv", index=False)

    cfg = ClosedLoopConfig(
        universe_base_dir=str(base_dir),
        universe_name=universe,
        group_fields=("industry",),
        search_mode="operator_only",
        max_eval_expressions=5,
        deep_explore_config=DeepExploreConfig(max_candidates=20, random_seed=5),
        mining_config=AlphaMiningConfig(simulation=AlphaSimulationConfig(delay=1, universe="universe")),
    )
    try:
        expressions, meta = _generate_candidates(store, cfg)
        assert expressions
        candidates = pd.read_csv(meta["candidates_path"])
        assert {
            "canonical_hash",
            "candidate_score",
            "feedback_score",
            "novelty_score",
        } <= set(candidates.columns)
        assert Path(meta["generation_diagnostics_path"]).exists()

        hints_path = Path(meta["feedback_hints_path"])
        assert hints_path.exists()
        hints = json.loads(hints_path.read_text(encoding="utf-8"))
        assert hints["field_weights"].get("volume", 0) > 0
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)
