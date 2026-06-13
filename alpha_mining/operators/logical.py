from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import OperatorRegistry


def register_operators(registry: OperatorRegistry) -> None:
    registry.register("if_else", lambda cond, a, b: a.where(cond, b))
    registry.register("trade_when", _trade_when)
    registry.register("trade_when_hold", _trade_when_hold)
    registry.register("is_nan", lambda x: x.isna())
    registry.register("is_not_nan", lambda x: x.notna())
    registry.register("greater", lambda x, y: x > y)
    registry.register("less", lambda x, y: x < y)
    registry.register("greater_equal", lambda x, y: x >= y)
    registry.register("less_equal", lambda x, y: x <= y)
    registry.register("equal", lambda x, y: x == y)
    registry.register("not_equal", lambda x, y: x != y)


def _trade_when(cond, a, b):
    """
    Simplified MVP trade_when:
    - cond True -> a
    - cond False -> b
    """
    if hasattr(a, "where"):
        return a.where(cond, b)
    return np.where(cond, a, b)


def _trade_when_hold(entry, alpha, exit):
    """
    Stateful gate:
    - exit=True: close position and output NaN
    - entry=True: open/update position with current alpha
    - otherwise: hold last opened alpha until exit
    """
    if isinstance(alpha, pd.DataFrame):
        return _trade_when_hold_dataframe(entry, alpha, exit)
    if isinstance(alpha, pd.Series):
        return _trade_when_hold_series(entry, alpha, exit)
    return _trade_when_hold_array(entry, alpha, exit)


def _trade_when_hold_dataframe(entry, alpha: pd.DataFrame, exit):
    base = alpha.astype(float).replace([np.inf, -np.inf], np.nan)
    entry_df = _to_bool_dataframe(entry, index=base.index, columns=base.columns)
    exit_df = _to_bool_dataframe(exit, index=base.index, columns=base.columns)

    arr = np.asarray(base.values, dtype=float)
    ent = np.asarray(entry_df.values, dtype=bool)
    ext = np.asarray(exit_df.values, dtype=bool)
    out = np.full(arr.shape, np.nan, dtype=float)
    n, m = arr.shape
    for j in range(m):
        is_open = False
        held = np.nan
        for i in range(n):
            if ext[i, j]:
                out[i, j] = np.nan
                is_open = False
                held = np.nan
            elif ent[i, j]:
                held = arr[i, j] if np.isfinite(arr[i, j]) else np.nan
                out[i, j] = held
                is_open = True
            elif is_open:
                out[i, j] = held
            else:
                out[i, j] = np.nan
    return pd.DataFrame(out, index=base.index, columns=base.columns)


def _trade_when_hold_series(entry, alpha: pd.Series, exit):
    base = alpha.astype(float).replace([np.inf, -np.inf], np.nan)
    entry_series = _to_bool_series(entry, index=base.index)
    exit_series = _to_bool_series(exit, index=base.index)
    arr = np.asarray(base.values, dtype=float)
    ent = np.asarray(entry_series.values, dtype=bool)
    ext = np.asarray(exit_series.values, dtype=bool)
    out = np.full(arr.shape, np.nan, dtype=float)
    is_open = False
    held = np.nan
    for i in range(arr.shape[0]):
        if ext[i]:
            out[i] = np.nan
            is_open = False
            held = np.nan
        elif ent[i]:
            held = arr[i] if np.isfinite(arr[i]) else np.nan
            out[i] = held
            is_open = True
        elif is_open:
            out[i] = held
        else:
            out[i] = np.nan
    return pd.Series(out, index=base.index, name=base.name)


def _trade_when_hold_array(entry, alpha, exit):
    arr = np.asarray(alpha, dtype=float)
    if arr.ndim != 1:
        raise ValueError("trade_when_hold_array_requires_1d_alpha")
    arr = np.where(np.isfinite(arr), arr, np.nan)
    ent = np.asarray(entry, dtype=bool)
    ext = np.asarray(exit, dtype=bool)
    if ent.shape != arr.shape or ext.shape != arr.shape:
        raise ValueError("trade_when_hold_shape_mismatch")
    out = np.full(arr.shape, np.nan, dtype=float)
    is_open = False
    held = np.nan
    for i in range(arr.shape[0]):
        if ext[i]:
            out[i] = np.nan
            is_open = False
            held = np.nan
        elif ent[i]:
            held = arr[i] if np.isfinite(arr[i]) else np.nan
            out[i] = held
            is_open = True
        elif is_open:
            out[i] = held
        else:
            out[i] = np.nan
    return out


def _to_bool_dataframe(value, index, columns) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.reindex(index=index, columns=columns).fillna(False).astype(bool)
    if isinstance(value, pd.Series):
        expanded = pd.DataFrame({col: value for col in columns}, index=index)
        return expanded.fillna(False).astype(bool)
    arr = np.asarray(value)
    if arr.ndim == 0:
        return pd.DataFrame(bool(arr), index=index, columns=columns)
    if arr.ndim == 1 and arr.shape[0] == len(index):
        expanded = np.broadcast_to(arr.reshape(-1, 1), (len(index), len(columns)))
        return pd.DataFrame(expanded, index=index, columns=columns).fillna(False).astype(bool)
    if arr.shape == (len(index), len(columns)):
        return pd.DataFrame(arr, index=index, columns=columns).fillna(False).astype(bool)
    raise ValueError("trade_when_hold_bool_shape_mismatch")


def _to_bool_series(value, index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.reindex(index=index).fillna(False).astype(bool)
    if isinstance(value, pd.DataFrame):
        if value.shape[1] != 1:
            raise ValueError("trade_when_hold_series_requires_single_column_condition")
        return value.iloc[:, 0].reindex(index=index).fillna(False).astype(bool)
    arr = np.asarray(value)
    if arr.ndim == 0:
        return pd.Series(bool(arr), index=index)
    if arr.ndim == 1 and arr.shape[0] == len(index):
        return pd.Series(arr, index=index).fillna(False).astype(bool)
    raise ValueError("trade_when_hold_series_bool_shape_mismatch")
