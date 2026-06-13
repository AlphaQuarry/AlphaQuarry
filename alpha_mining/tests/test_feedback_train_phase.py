from __future__ import annotations

import pandas as pd

from alpha_mining.mining.feedback_sampler import FeedbackSampler
from alpha_mining.mining.fragment_registry import refresh_fragment_registry
from alpha_mining.workflow.closed_loop import _scoreboard_score


def test_scoreboard_score_prefers_train_over_val_and_full_period() -> None:
    scores = _scoreboard_score(
        pd.DataFrame(
            {
                "train_score_total": [10.0, 90.0],
                "val_score_total": [99.0, 1.0],
                "score_total": [50.0, 50.0],
            }
        )
    )

    assert list(scores) == [10.0, 90.0]


def test_scoreboard_score_fallback_order() -> None:
    assert list(_scoreboard_score(pd.DataFrame({"train_score": [7.0]}))) == [7.0]
    assert list(_scoreboard_score(pd.DataFrame({"score_total": [8.0]}))) == [8.0]
    assert list(_scoreboard_score(pd.DataFrame({"scoreboard_score": [9.0]}))) == [9.0]


def test_feedback_sampler_uses_feedback_score_first() -> None:
    hints = FeedbackSampler().build_weight_hints(
        pd.DataFrame(
            {
                "expression": ["rank(close)", "rank(volume)"],
                "fields": ["close", "volume"],
                "feedback_score": [1.0, 100.0],
                "score_total": [100.0, 1.0],
            }
        )
    )

    assert hints["score_col"] == "feedback_score"
    assert hints["field_weights"] == {"volume": 1.0}


def test_fragment_registry_uses_feedback_score_first(tmp_path) -> None:
    registry_df, _, _ = refresh_fragment_registry(
        pd.DataFrame(
            {
                "factor": ["alpha_a"],
                "expression": ["rank(close)"],
                "feedback_score": [88.0],
                "score_total": [1.0],
            }
        ),
        tmp_path / "fragment_registry.parquet",
    )

    assert not registry_df.empty
    assert float(registry_df["source_score"].max()) == 88.0
