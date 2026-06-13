from __future__ import annotations

import pandas as pd

from alpha_mining.workflow.analysis_data_artifacts import (
    build_distribution_histogram_table,
    build_phase_ic_decay_table,
)
from factor_research import SampleSplitConfig


def _sample_frame() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2024-12-30",
            "2024-12-31",
            "2025-01-02",
            "2025-01-03",
            "2026-01-02",
            "2026-01-05",
        ]
    )
    rows = []
    for date_idx, date in enumerate(dates):
        for code_idx, code in enumerate(["A", "B", "C", "D"]):
            rows.append(
                {
                    "trade_date": date,
                    "znz_code": code,
                    "pct_chg": 0.01 * (code_idx - 1) + 0.001 * date_idx,
                    "pct_chg_1d": 0.008 * (code_idx - 1) + 0.001 * date_idx,
                    "alpha00001": float(code_idx + date_idx),
                }
            )
    return pd.DataFrame(rows)


def test_distribution_histogram_is_compact_and_phase_aware() -> None:
    hist = build_distribution_histogram_table(
        _sample_frame(),
        ["alpha00001"],
        sample_split_config=SampleSplitConfig(),
        bins=8,
    )

    assert set(hist["phase"]) == {"train", "val", "test"}
    assert set(hist.columns) == {
        "factor",
        "phase",
        "bin_index",
        "bin_left",
        "bin_right",
        "bin_mid",
        "count",
        "total_count",
    }
    assert "alpha00001" in hist["factor"].astype(str).unique()
    assert "value" not in hist.columns
    assert int(hist.groupby("phase")["count"].sum().loc["test"]) > 0


def test_phase_ic_decay_skips_missing_phase_without_failing() -> None:
    frame = _sample_frame()
    frame = frame[frame["trade_date"] < pd.Timestamp("2026-01-01")].copy()

    decay = build_phase_ic_decay_table(
        frame,
        ["alpha00001"],
        return_col="pct_chg",
        period=1,
        max_lag=3,
        sample_split_config=SampleSplitConfig(),
    )

    assert set(decay["phase"]) == {"train", "val"}
    assert "test" not in set(decay["phase"])
    assert set(decay["lag"].dropna().astype(int)).issubset({0, 1, 2, 3})
