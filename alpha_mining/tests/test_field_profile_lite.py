from __future__ import annotations

import pandas as pd

from alpha_mining.mining.field_profile_lite import (
    build_field_profiles,
    recommended_windows_for_role,
)
from alpha_mining.mining.field_universe import build_field_universe
from alpha_mining.panel_store import PanelStore


def _store() -> PanelStore:
    rows = []
    for i, date in enumerate(pd.date_range("2024-01-01", periods=5)):
        for code in ["A", "B"]:
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "close": float(i + 1),
                    "amount": 100.0,
                    "pe_ttm": None if i < 4 else 12.0,
                    "moneyflow_buy_lg_amount": 3.0,
                }
            )
    return PanelStore.from_long_frame(pd.DataFrame(rows))


def test_field_profile_lite_is_deterministic_without_panel_sample() -> None:
    universe = build_field_universe(_store())

    profiles = build_field_profiles(universe, panel_store=None, feedback_hints={})

    assert profiles["moneyflow_buy_lg_amount"].field_profile_score > profiles["pe_ttm"].field_profile_score
    assert profiles["close"].coverage_score == 0.5
    assert profiles["close"].finite_score == 0.5


def test_field_profile_lite_uses_panel_coverage_when_available() -> None:
    store = _store()
    universe = build_field_universe(store)

    profiles = build_field_profiles(universe, panel_store=store, feedback_hints={})

    assert profiles["close"].coverage_score > profiles["pe_ttm"].coverage_score
    assert profiles["close"].finite_score > profiles["pe_ttm"].finite_score
    assert profiles["close"].field_profile_score > profiles["pe_ttm"].field_profile_score


def test_field_profile_lite_marks_low_quality_status_without_dropping_field() -> None:
    store = _store()
    universe = build_field_universe(store)

    profiles = build_field_profiles(
        universe,
        panel_store=store,
        feedback_hints={},
        min_coverage=0.20,
        min_finite_rate=0.80,
    )

    assert "pe_ttm" in profiles
    assert profiles["pe_ttm"].field_profile_status in {
        "low_coverage",
        "low_finite",
        "low_coverage,low_finite",
    }
    assert profiles["close"].field_profile_status == "pass"


def test_field_profile_lite_uses_positive_and_negative_feedback() -> None:
    store = _store()
    universe = build_field_universe(store)

    profiles = build_field_profiles(
        universe,
        panel_store=None,
        feedback_hints={
            "field_weights": {"pe_ttm": 2.0},
            "negative_field_weights": {"close": 2.0},
        },
    )

    assert profiles["pe_ttm"].feedback_score > 0
    assert profiles["close"].negative_feedback_score > 0
    assert profiles["pe_ttm"].field_profile_score > profiles["close"].field_profile_score


def test_recommended_windows_follow_field_roles() -> None:
    assert recommended_windows_for_role("price") == (5, 10, 22, 66)
    assert recommended_windows_for_role("liquidity") == (5, 10, 22, 66)
    assert recommended_windows_for_role("moneyflow") == (5, 10, 22, 66)
    assert recommended_windows_for_role("valuation") == (22, 66, 132)
    assert recommended_windows_for_role("size") == (22, 66, 132)
    assert recommended_windows_for_role("chip") == (10, 22, 66)
