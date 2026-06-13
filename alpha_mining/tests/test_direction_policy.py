from __future__ import annotations

import pandas as pd

from alpha_mining.workflow.direction_policy import build_direction_policy_tables
from factor_research import SampleSplitConfig, calculate_best_layer_metrics


def test_direction_policy_train_locked_ignores_val_and_test_signs() -> None:
    ic_df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                [
                    "2024-12-30",
                    "2024-12-31",
                    "2025-01-02",
                    "2025-01-03",
                    "2026-01-02",
                    "2026-01-05",
                ]
            ),
            "alpha_a_ic": [0.10, 0.12, -0.20, -0.21, -0.30, -0.31],
        }
    )

    policy, phase_local = build_direction_policy_tables(
        ic_df=ic_df,
        factors=["alpha_a"],
        sample_split_config=SampleSplitConfig(),
    )

    row = policy.iloc[0]
    assert row["factor"] == "alpha_a"
    assert row["direction_policy"] == "train_locked"
    assert row["direction_source_phase"] == "train"
    assert row["direction_sign"] == 1.0
    assert row["best_layer_direction_train_locked"] == "top"

    val_row = phase_local[(phase_local["factor"] == "alpha_a") & (phase_local["phase"] == "val")].iloc[0]
    assert val_row["direction_sign"] == -1.0
    assert val_row["best_layer_direction_phase_local"] == "bottom"


def test_best_layer_metrics_accepts_train_locked_sign_override() -> None:
    layer_results = {
        "alpha_a": pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2024-12-30", "2024-12-30", "2024-12-31", "2024-12-31"]),
                "znz_code": ["A", "B", "A", "B"],
                "layer": [1, 2, 1, 2],
                "alpha_a": [0.1, 0.9, 0.2, 0.8],
                "pct_chg_1d": [0.00, 0.02, 0.00, 0.02],
            }
        )
    }
    full_sample_summary = pd.DataFrame({"factor": ["alpha_a"], "ic_mean": [-0.10]})

    metrics = calculate_best_layer_metrics(
        layer_results,
        ic_summary_df=full_sample_summary,
        ic_signs_override={"alpha_a": 1.0},
        period=1,
    )

    assert metrics.iloc[0]["best_layer_label"] == 2
    assert metrics.iloc[0]["best_layer_direction"] == "top"
