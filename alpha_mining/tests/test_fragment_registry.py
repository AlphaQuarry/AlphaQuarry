from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_mining.mining.fragment_registry import (
    apply_candidate_feedback_to_registry,
    FragmentRegistryConfig,
    extract_fragments_from_expression,
    refresh_fragment_registry,
    select_active_fragments,
)


def test_extract_fragments_filters_leakage_and_captures_types() -> None:
    fragments = extract_fragments_from_expression(
        "group_rank(ts_mean(close, 22), industry)",
        source_alpha_name="alpha001",
        source_score=1.2,
    )
    assert fragments
    types = {str(x.get("fragment_type", "")) for x in fragments}
    assert "field" in types
    assert "ts_wrapper" in types
    assert "group_wrapper" in types

    leaked = extract_fragments_from_expression("rank(pct_chg)", source_alpha_name="alpha_bad", source_score=1.0)
    assert leaked == []


def test_refresh_fragment_registry_dedup_and_cooldown(tmp_path: Path) -> None:
    registry_path = tmp_path / "fragment_registry.parquet"
    cfg = FragmentRegistryConfig(cooldown_batches=3, max_age_batches=50, top_k=128)

    negative_scoreboard = pd.DataFrame(
        {
            "factor": ["alpha_n1"],
            "expression": ["rank(close)"],
            "score_total": [-0.6],
            "batch_id": ["batch_n1"],
        }
    )
    registry_df, batch_no, _ = refresh_fragment_registry(
        scoreboard_df=negative_scoreboard,
        registry_path=registry_path,
        config=cfg,
        current_batch=1,
    )
    assert batch_no == 1
    rank_rows = registry_df[registry_df["fragment_expression"].astype(str) == "rank(close)"]
    assert not rank_rows.empty
    assert int(rank_rows.iloc[0]["cooldown_until"]) >= 4
    blocked = select_active_fragments(registry_df, current_batch=2, max_age_batches=50, limit=128)
    assert "rank(close)" not in set(blocked["fragment_expression"].astype(str))
    unblocked = select_active_fragments(registry_df, current_batch=5, max_age_batches=50, limit=128)
    assert "rank(close)" in set(unblocked["fragment_expression"].astype(str))

    dedup_scoreboard = pd.DataFrame(
        {
            "factor": ["alpha_p1", "alpha_p2"],
            "expression": ["add(close, volume)", "add(volume, close)"],
            "score_total": [0.7, 0.9],
            "batch_id": ["batch_p1", "batch_p2"],
        }
    )
    registry_df2, _, _ = refresh_fragment_registry(
        scoreboard_df=dedup_scoreboard,
        registry_path=registry_path,
        config=cfg,
        current_batch=2,
    )
    add_rows = registry_df2[registry_df2["fragment_expression"].astype(str) == "add(close,volume)"]
    assert len(add_rows) == 1
    assert int(add_rows.iloc[0]["positive_count"]) >= 2


def test_candidate_feedback_writeback_updates_cooldown_and_positive_paths() -> None:
    registry = pd.DataFrame(
        [
            {
                "fragment_hash": "frag_1",
                "fragment_expression": "rank(close)",
                "fragment_type": "unary",
                "operators": "rank",
                "fields": "close",
                "windows": "",
                "groups": "",
                "output_type": "scalar",
                "depth": 1,
                "complexity": 2,
                "source_alpha_name": "alpha_x",
                "source_alpha_hash": "parent_x",
                "source_expression": "rank(close)",
                "source_batch_id": "b1",
                "source_score": 1.0,
                "oos_score": float("nan"),
                "positive_count": 0,
                "negative_count": 0,
                "rejected_count": 0,
                "mean_child_score": float("nan"),
                "best_child_score": float("nan"),
                "last_seen_batch": 1,
                "cooldown_until": 0,
                "status": "active",
            }
        ]
    )
    candidate_df = pd.DataFrame(
        [
            {
                "source": "feedback_mutation_v2",
                "fragment_hash": "frag_1",
                "expression": "rank(close)",
                "prefilter_status": "pass",
                "sample_status": "pass",
            },
            {
                "source": "feedback_mutation_v2",
                "fragment_hash": "frag_1",
                "expression": "rank(close)",
                "prefilter_status": "reject",
                "sample_status": "",
            },
        ]
    )
    updated, summary = apply_candidate_feedback_to_registry(
        registry_df=registry,
        candidate_df=candidate_df,
        current_batch=2,
        cooldown_batches=3,
        evaluated_expressions={"rank(close)"},
    )
    row = updated.iloc[0]
    assert int(row["positive_count"]) == 1
    assert int(row["negative_count"]) == 1
    assert int(row["rejected_count"]) == 1
    assert int(row["cooldown_until"]) >= 5
    assert str(row["status"]) == "cooldown"
    assert summary["positive_updates"] == 1
    assert summary["negative_updates"] == 1
