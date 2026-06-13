from __future__ import annotations

import numpy as np
import pandas as pd


def apply_portfolio_scale(
    alpha_panel: pd.DataFrame,
    scale_value: float = 1.0,
    longscale: float | None = None,
    shortscale: float | None = None,
) -> pd.DataFrame:
    """Scale each date row into portfolio weights."""
    scale_value = float(scale_value)
    if longscale is not None or shortscale is not None:
        long_target = float(longscale if longscale is not None else scale_value / 2.0)
        short_target = float(shortscale if shortscale is not None else scale_value / 2.0)
        return alpha_panel.apply(
            lambda row: _scale_row_long_short(row, long_target=long_target, short_target=short_target),
            axis=1,
        )
    return alpha_panel.apply(lambda row: _scale_row_total(row, scale_value=scale_value), axis=1)


def _scale_row_total(row: pd.Series, scale_value: float) -> pd.Series:
    values = pd.to_numeric(row, errors="coerce").replace([np.inf, -np.inf], np.nan)
    denom = float(values.abs().sum(skipna=True))
    out = pd.Series(0.0, index=row.index, dtype="float64")
    if denom <= 0 or not np.isfinite(denom):
        out[values.isna()] = np.nan
        return out
    out.loc[values.notna()] = values.loc[values.notna()] / denom * float(scale_value)
    out[values.isna()] = np.nan
    return out


def _scale_row_long_short(row: pd.Series, long_target: float, short_target: float) -> pd.Series:
    values = pd.to_numeric(row, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = pd.Series(0.0, index=row.index, dtype="float64")
    pos = values > 0
    neg = values < 0
    pos_sum = float(values[pos].sum(skipna=True))
    neg_sum = float(values[neg].abs().sum(skipna=True))
    if pos_sum > 0 and np.isfinite(pos_sum):
        out.loc[pos] = values.loc[pos] / pos_sum * max(0.0, float(long_target))
    if neg_sum > 0 and np.isfinite(neg_sum):
        out.loc[neg] = -values.loc[neg].abs() / neg_sum * max(0.0, float(short_target))
    out[values.isna()] = np.nan
    return out
