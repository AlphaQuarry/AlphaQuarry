from __future__ import annotations

import pandas as pd

from factor_research import (
    SampleSplitConfig,
    assign_phase,
    assign_sample_split,
    build_phase_windows,
)


def test_assign_phase_uses_train_val_test_boundaries() -> None:
    df = pd.DataFrame({"trade_date": pd.to_datetime(["2024-12-31", "2025-01-01", "2025-12-31", "2026-01-01"])})

    out = assign_phase(df, config=SampleSplitConfig())

    assert out["sample_phase"].tolist() == ["train", "val", "val", "test"]
    assert out["sample_split"].tolist() == ["train", "validation", "validation", "oos"]


def test_assign_sample_split_keeps_legacy_names() -> None:
    df = pd.DataFrame({"trade_date": pd.to_datetime(["2025-06-01", "2026-02-01"])})

    out = assign_sample_split(df, config=SampleSplitConfig())

    assert out["sample_split"].tolist() == ["validation", "oos"]


def test_phase_windows_follow_available_max_date() -> None:
    cfg = SampleSplitConfig()

    train_only = build_phase_windows(cfg, max_date="2024-12-31")
    train_val = build_phase_windows(cfg, max_date="2025-05-01")
    all_phases = build_phase_windows(cfg, max_date="2026-02-01")

    assert [phase.key for phase in train_only] == ["train"]
    assert [phase.key for phase in train_val] == ["train", "val"]
    assert [phase.key for phase in all_phases] == ["train", "val", "test"]
    assert all_phases[-1].visible_default is False
    assert all_phases[-1].end == "2026-02-01"
