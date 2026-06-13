from __future__ import annotations

import json

import pandas as pd

from alpha_mining.mining.feedback_policy_lite import (
    build_feedback_policy_hints,
    merge_feedback_policy_hints,
)
from alpha_mining.mining.feedback_sampler import FeedbackSampler, FeedbackSamplerConfig


def test_feedback_policy_lite_returns_empty_hints_for_missing_history() -> None:
    hints = build_feedback_policy_hints(pd.DataFrame())

    assert hints["recipe_weights"] == {}
    assert hints["gate_family_weights"] == {}
    assert hints["bucket_family_weights"] == {}
    assert hints["role_pair_type_weights"] == {}


def test_feedback_policy_lite_extracts_positive_and_negative_metadata_weights() -> None:
    history = pd.DataFrame(
        {
            "score_total": [2.0, -1.0],
            "metadata_json": [
                json.dumps(
                    {
                        "recipe_family": "moneyflow_imbalance",
                        "recipe_id": "moneyflow_imbalance:buy_sell",
                        "gate_family": "moneyflow_pressure",
                        "bucket_family": "size",
                        "role_pair_type": "moneyflow_buy_sell",
                        "operator_tier": "stable",
                    }
                ),
                json.dumps(
                    {
                        "recipe_family": "valuation_peer",
                        "recipe_id": "valuation_peer:industry",
                        "gate_family": "price_trend",
                        "bucket_family": "valuation",
                        "role_pair_type": "valuation_x_size",
                        "operator_tier": "experimental",
                    }
                ),
            ],
        }
    )

    hints = build_feedback_policy_hints(history)

    assert hints["recipe_weights"]["moneyflow_imbalance"] > 0
    assert hints["negative_recipe_weights"]["valuation_peer"] > 0
    assert hints["gate_family_weights"]["moneyflow_pressure"] > 0
    assert hints["negative_gate_family_weights"]["price_trend"] > 0
    assert hints["bucket_family_weights"]["size"] > 0
    assert hints["negative_bucket_family_weights"]["valuation"] > 0
    assert hints["role_pair_type_weights"]["moneyflow_buy_sell"] > 0
    assert hints["operator_tier_weights"]["stable"] > 0


def test_feedback_policy_lite_prefers_net_score_basis() -> None:
    history = pd.DataFrame(
        {
            "feedback_score": [2.0, -2.0],
            "feedback_score_net": [-1.0, 1.0],
            "metadata_json": [
                json.dumps({"recipe_family": "gross_winner"}),
                json.dumps({"recipe_family": "net_winner"}),
            ],
        }
    )

    hints = build_feedback_policy_hints(history)

    assert hints["score_column"] == "feedback_score_net"
    assert hints["score_basis"] == "net"
    assert hints["recipe_weights"]["net_winner"] > 0
    assert hints["negative_recipe_weights"]["gross_winner"] > 0


def test_feedback_sampler_prefers_net_score_basis() -> None:
    scoreboard = pd.DataFrame(
        {
            "fields": ["close", "amount"],
            "operators": ["rank", "rank"],
            "family": ["operator", "operator"],
            "feedback_score": [2.0, -2.0],
            "feedback_score_net": [-1.0, 1.0],
        }
    )

    hints = FeedbackSampler(FeedbackSamplerConfig(enabled=True)).build_weight_hints(scoreboard)

    assert hints["score_col"] == "feedback_score_net"
    assert hints["score_basis"] == "net"
    assert hints["field_weights"]["amount"] > 0
    assert hints["negative_field_weights"]["close"] > 0


def test_feedback_policy_lite_merges_without_overwriting_existing_feedback() -> None:
    base = {"field_weights": {"close": 1.0}, "recipe_weights": {"old": 0.5}}
    policy = {"recipe_weights": {"new": 1.2}, "field_weights": {"amount": 2.0}}

    merged = merge_feedback_policy_hints(base, policy)

    assert merged["field_weights"]["close"] == 1.0
    assert merged["field_weights"]["amount"] == 2.0
    assert merged["recipe_weights"]["old"] == 0.5
    assert merged["recipe_weights"]["new"] == 1.2
