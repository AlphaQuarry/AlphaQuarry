from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_mining.simulation.neutralization import (
    apply_neutralization,
    neutralization_group_field,
    normalize_neutralization_mode,
)


def _alpha_panel() -> pd.DataFrame:
    return pd.DataFrame(
        [[1.0, 3.0, np.nan, 8.0], [2.0, 4.0, 6.0, 10.0]],
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        columns=["A", "B", "C", "D"],
    )


def test_none_returns_original_panel() -> None:
    alpha = _alpha_panel()

    out = apply_neutralization(alpha, "none")

    assert out is alpha


def test_market_demeans_each_cross_section_and_preserves_nan() -> None:
    alpha = _alpha_panel()

    out = apply_neutralization(alpha, "MARKET")

    assert np.isnan(out.loc[pd.Timestamp("2024-01-01"), "C"])
    assert abs(float(out.loc[pd.Timestamp("2024-01-01")].sum(skipna=True))) <= 1.0e-12
    assert abs(float(out.loc[pd.Timestamp("2024-01-02")].sum(skipna=True))) <= 1.0e-12


@pytest.mark.parametrize(
    ("mode", "field"),
    [
        ("SECTOR", "sector"),
        ("industry", "industry"),
        ("GROUP:subindustry", "subindustry"),
    ],
)
def test_group_modes_demean_within_group(mode: str, field: str) -> None:
    alpha = _alpha_panel()
    groups = pd.DataFrame(
        [["G1", "G1", "G2", "G2"], ["G1", "G1", "G2", "G2"]],
        index=alpha.index,
        columns=alpha.columns,
    )

    out = apply_neutralization(alpha, mode, group_panel=groups)
    normalized = normalize_neutralization_mode(mode)

    assert neutralization_group_field(mode) == field
    assert normalized in {"SECTOR", "INDUSTRY", "SUBINDUSTRY"}
    for dt in alpha.index:
        for group_name in ["G1", "G2"]:
            members = groups.loc[dt] == group_name
            values = out.loc[dt, members]
            if values.notna().any():
                assert abs(float(values.sum(skipna=True))) <= 1.0e-12


def test_group_neutralization_missing_group_values_emit_nan() -> None:
    alpha = _alpha_panel()
    groups = pd.DataFrame(
        [["G1", None, "G2", "G2"], ["G1", "G1", None, "G2"]],
        index=alpha.index,
        columns=alpha.columns,
    )

    out = apply_neutralization(alpha, "SECTOR", group_panel=groups)

    assert np.isnan(out.loc[pd.Timestamp("2024-01-01"), "B"])
    assert np.isnan(out.loc[pd.Timestamp("2024-01-02"), "C"])


def test_group_neutralization_requires_group_panel() -> None:
    with pytest.raises(ValueError, match="requires group_panel"):
        apply_neutralization(_alpha_panel(), "INDUSTRY")


@pytest.mark.parametrize("mode", ["STATISTICAL", "RAM", "FAST"])
def test_unsupported_neutralization_mode_fails_fast(mode: str) -> None:
    with pytest.raises(ValueError, match="Unsupported neutralization mode"):
        normalize_neutralization_mode(mode)
