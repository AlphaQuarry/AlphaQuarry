from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import OperatorRegistry


def register_operators(registry: OperatorRegistry) -> None:
    registry.register("regression_neut", _regression_neut)
    registry.register("ts_regression", _ts_regression)


def _regression_neut(y: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional OLS residual by date, vectorized for one regressor + intercept."""
    common_idx = y.index.intersection(x.index)
    common_cols = y.columns.intersection(x.columns)
    yv = y.reindex(index=common_idx, columns=common_cols).astype(float)
    xv = x.reindex(index=common_idx, columns=common_cols).astype(float)

    valid = np.isfinite(yv.values) & np.isfinite(xv.values)
    n = valid.sum(axis=1).astype(float)
    if len(n) == 0:
        return pd.DataFrame(index=y.index, columns=y.columns, dtype=float)

    x_arr = xv.values
    y_arr = yv.values

    sum_x = np.where(valid, x_arr, 0.0).sum(axis=1)
    sum_y = np.where(valid, y_arr, 0.0).sum(axis=1)
    sum_xy = np.where(valid, x_arr * y_arr, 0.0).sum(axis=1)
    sum_x2 = np.where(valid, x_arr * x_arr, 0.0).sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        sxy = sum_xy - (sum_x * sum_y) / n
        sxx = sum_x2 - (sum_x * sum_x) / n
        beta = sxy / sxx
        beta = np.where((n >= 2.0) & np.isfinite(beta), beta, np.nan)
        alpha = (sum_y / n) - beta * (sum_x / n)

    pred = alpha[:, None] + beta[:, None] * x_arr
    resid = np.where(valid, y_arr - pred, np.nan)

    out_common = pd.DataFrame(resid, index=common_idx, columns=common_cols)
    out = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)
    out.loc[common_idx, common_cols] = out_common.values
    return out


def _ts_regression(y: pd.DataFrame, x: pd.DataFrame, d: int, rettype: int = 0) -> pd.DataFrame:
    """
    Simplified rolling ts_regression:
    - rettype=0: beta
    - rettype=2: residual of latest observation
    """
    window = max(int(d), 1)
    common_idx = y.index.intersection(x.index)
    common_cols = y.columns.intersection(x.columns)
    yv = y.reindex(index=common_idx, columns=common_cols).astype(float)
    xv = x.reindex(index=common_idx, columns=common_cols).astype(float)

    valid = xv.notna() & yv.notna()
    xv_pair = xv.where(valid)
    yv_pair = yv.where(valid)

    n = valid.rolling(window, min_periods=2).sum()
    sum_x = xv_pair.rolling(window, min_periods=2).sum()
    sum_y = yv_pair.rolling(window, min_periods=2).sum()
    sum_xy = (xv_pair * yv_pair).rolling(window, min_periods=2).sum()
    sum_x2 = (xv_pair * xv_pair).rolling(window, min_periods=2).sum()

    with np.errstate(invalid="ignore", divide="ignore"):
        sxy = sum_xy - (sum_x * sum_y) / n
        sxx = sum_x2 - (sum_x * sum_x) / n
        beta = sxy / sxx
        beta = beta.where((n >= 2.0) & sxx.ne(0.0))

    if int(rettype) == 2:
        mean_x = sum_x / n
        mean_y = sum_y / n
        alpha = mean_y - beta * mean_x
        out_common = yv - (alpha + beta * xv)
        out_common = out_common.where(valid)
    else:
        out_common = beta

    out = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)
    out.loc[common_idx, common_cols] = out_common.values
    return out
