from __future__ import annotations

import pandas as pd
import numpy as np


def apply_truncation(alpha_panel: pd.DataFrame, truncation: float | None) -> pd.DataFrame:
    if truncation is None or truncation <= 0:
        return alpha_panel
    t = float(truncation)
    return alpha_panel.clip(lower=-t, upper=t)


def capped_rescale_positive(v: pd.Series, target_sum: float, cap: float) -> pd.Series:
    """Allocate non-negative values to target_sum with per-name cap when feasible."""
    values = pd.to_numeric(v, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    target = max(0.0, float(target_sum))
    cap_value = float(cap)
    if target <= 0:
        return pd.Series(0.0, index=v.index, dtype="float64")
    if cap_value <= 0:
        raise ValueError("cap must be positive")
    if values.empty:
        return pd.Series(dtype="float64", index=v.index)

    feasible_target = min(target, float(len(values)) * cap_value)
    out = pd.Series(0.0, index=values.index, dtype="float64")
    remaining = values.copy()
    remaining_target = feasible_target

    while len(remaining) > 0 and remaining_target > 1.0e-15:
        denom = float(remaining.sum(skipna=True))
        if denom <= 0 or not np.isfinite(denom):
            proposed = pd.Series(
                remaining_target / len(remaining),
                index=remaining.index,
                dtype="float64",
            )
        else:
            proposed = remaining / denom * remaining_target
        over = proposed > cap_value + 1.0e-12
        if not bool(over.any()):
            out.loc[remaining.index] = proposed.clip(lower=0.0, upper=cap_value)
            return out
        capped_idx = proposed[over].index
        out.loc[capped_idx] = cap_value
        remaining_target -= cap_value * len(capped_idx)
        remaining = remaining.drop(capped_idx)

    return out.clip(lower=0.0, upper=cap_value)


def apply_truncation_capped_rescale(weights: pd.DataFrame, cap: float | None) -> pd.DataFrame:
    if cap is None or cap <= 0:
        return weights
    cap_value = float(cap)
    return weights.apply(lambda row: _truncate_row_capped(row, cap_value), axis=1)


def apply_truncation_long_short(weights: pd.DataFrame, cap: float | None) -> pd.DataFrame:
    if cap is None or cap <= 0:
        return weights
    cap_value = float(cap)
    if cap_value > 1:
        raise ValueError("truncation cap must be <= 1")
    return weights.apply(lambda row: _truncate_row_long_short(row, cap_value), axis=1)


def apply_portfolio_truncation(weights: pd.DataFrame, cap: float | None, mode: str = "clip") -> pd.DataFrame:
    mode_norm = str(mode or "clip").strip().lower()
    if mode_norm == "clip":
        return apply_truncation(weights, cap)
    if mode_norm == "capped_rescale":
        return apply_truncation_capped_rescale(weights, cap)
    if mode_norm == "long_short_capped_rescale":
        return apply_truncation_long_short(weights, cap)
    raise ValueError(f"Unsupported truncation_mode: {mode}")


def _truncate_row_capped(row: pd.Series, cap: float) -> pd.Series:
    values = pd.to_numeric(row, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = values.dropna()
    target = float(valid.abs().sum(skipna=True))
    if valid.empty or target <= 0:
        return values * 0.0
    allocated = capped_rescale_positive(valid.abs(), target_sum=target, cap=cap)
    out = pd.Series(np.nan, index=row.index, dtype="float64")
    out.loc[allocated.index] = np.sign(valid.loc[allocated.index]) * allocated
    return out


def _truncate_row_long_short(row: pd.Series, cap: float) -> pd.Series:
    values = pd.to_numeric(row, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = pd.Series(np.nan, index=row.index, dtype="float64")
    valid = values.dropna()
    out.loc[valid.index] = 0.0
    pos = valid[valid > 0]
    neg_abs = valid[valid < 0].abs()
    if not pos.empty:
        out.loc[pos.index] = capped_rescale_positive(pos, target_sum=float(pos.sum()), cap=cap)
    if not neg_abs.empty:
        out.loc[neg_abs.index] = -capped_rescale_positive(neg_abs, target_sum=float(neg_abs.sum()), cap=cap)
    return out
