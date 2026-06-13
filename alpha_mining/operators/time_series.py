from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import OperatorRegistry

try:
    import bottleneck as bn
except Exception:  # pragma: no cover - optional fast path
    bn = None


def register_operators(registry: OperatorRegistry) -> None:
    registry.register("ts_delay", lambda x, d: x.shift(int(d)))
    registry.register("ts_delta", lambda x, d: x - x.shift(int(d)))
    registry.register("ts_mean", lambda x, d: x.rolling(window=int(d), min_periods=1).mean())
    registry.register("ts_min", _ts_min)
    registry.register("ts_max", _ts_max)
    registry.register("ts_median", _ts_median)
    registry.register("ts_sum", lambda x, d: x.rolling(window=int(d), min_periods=1).sum())
    registry.register("ts_std_dev", lambda x, d: x.rolling(window=int(d), min_periods=1).std())
    registry.register(
        "ts_product",
        lambda x, d: (1 + x).rolling(window=int(d), min_periods=1).apply(np.prod, raw=True) - 1,
    )
    registry.register("ts_av_diff", _ts_av_diff)
    registry.register("ts_rank", _ts_rank)
    registry.register("ts_zscore", _ts_zscore)
    registry.register("ts_ir", _ts_ir)
    registry.register("ts_corr", _ts_corr)
    registry.register("ts_covariance", _ts_covariance)
    registry.register("ts_count_nans", _ts_count_nans)
    registry.register("ts_backfill", _ts_backfill)
    registry.register("ts_arg_max", _ts_arg_max)
    registry.register("ts_arg_min", _ts_arg_min)
    registry.register("ts_decay_linear", _ts_decay_linear)
    registry.register("ts_decay_exp_window", _ts_decay_exp_window)
    registry.register("days_from_last_change", _days_from_last_change)
    registry.register("hump", _hump)


def _ts_min(x, d):
    """Rolling minimum over the last d observations (inclusive)."""
    return _clean_nonfinite(_to_numeric_panel(x).rolling(int(d), min_periods=1).min())


def _ts_max(x, d):
    """Rolling maximum over the last d observations (inclusive)."""
    return _clean_nonfinite(_to_numeric_panel(x).rolling(int(d), min_periods=1).max())


def _ts_median(x, d):
    """Rolling median over the last d observations (inclusive)."""
    return _clean_nonfinite(_to_numeric_panel(x).rolling(int(d), min_periods=1).median())


def _ts_av_diff(x, d):
    """Difference between current value and rolling average: x - ts_mean(x, d)."""
    base = _to_numeric_panel(x)
    return _clean_nonfinite(base - base.rolling(int(d), min_periods=1).mean())


def _ts_rank(x, d):
    d = int(d)
    if isinstance(x, pd.Series):
        arr = np.asarray(x.values, dtype=float)
        out = _ts_rank_array(arr, d)
        return pd.Series(out, index=x.index, name=x.name)

    if isinstance(x, pd.DataFrame):
        arr2 = np.asarray(x.values, dtype=float)
        out2 = _ts_rank_array(arr2, d)
        return pd.DataFrame(out2, index=x.index, columns=x.columns)

    arr = np.asarray(x, dtype=float)
    return _ts_rank_array(arr, d)


def _ts_zscore(x, d):
    d = int(d)
    base = _to_numeric_panel(x)
    mean = base.rolling(d, min_periods=1).mean()
    std = base.rolling(d, min_periods=1).std().replace(0, np.nan)
    return _clean_nonfinite((base - mean) / std)


def _ts_ir(x, d):
    d = int(d)
    base = _to_numeric_panel(x)
    mean = base.rolling(d, min_periods=1).mean()
    std = base.rolling(d, min_periods=1).std().replace(0, np.nan)
    return _clean_nonfinite(mean / std)


def _ts_corr(x, y, d):
    base_x = _to_numeric_panel(x)
    base_y = _to_numeric_panel(y)
    return _clean_nonfinite(base_x.rolling(int(d), min_periods=2).corr(base_y))


def _ts_covariance(x, y, d):
    """Rolling covariance with min sample size 2."""
    base_x = _to_numeric_panel(x)
    base_y = _to_numeric_panel(y)
    return _clean_nonfinite(base_x.rolling(int(d), min_periods=2).cov(base_y))


def _ts_count_nans(x, d):
    """Rolling count of NaN/non-finite values within the last d observations."""
    base = _to_numeric_panel(x)
    return base.isna().rolling(int(d), min_periods=1).sum()


def _ts_backfill(x, d):
    """Fill current NaN values from prior observations, limited to d rows."""
    limit = max(0, int(d))
    base = _to_numeric_panel(x)
    if limit <= 0:
        return base
    if hasattr(base, "ffill"):
        return _clean_nonfinite(base.ffill(limit=limit))
    return _clean_nonfinite(base)


def _ts_arg_max(x, d):
    return x.rolling(int(d), min_periods=1).apply(_argmax_relative_lag, raw=True)


def _ts_arg_min(x, d):
    return x.rolling(int(d), min_periods=1).apply(_argmin_relative_lag, raw=True)


def _ts_decay_linear(x, d):
    d = int(d)
    base = _to_numeric_panel(x)
    weights = np.arange(1, d + 1, dtype=float)
    weights = weights / weights.sum()

    def _weighted(arr):
        w = weights[-len(arr) :]
        finite = np.isfinite(arr)
        if not bool(finite.any()):
            return np.nan
        arr_use = np.where(finite, arr, 0.0)
        w_use = np.where(finite, w, 0.0)
        denom = w_use.sum()
        if denom <= 0:
            return np.nan
        return float(np.dot(arr_use, w_use / denom))

    return _clean_nonfinite(base.rolling(d, min_periods=1).apply(_weighted, raw=True))


def _ts_decay_exp_window(x, d, factor=0.5):
    d = int(d)
    factor = float(factor)
    base = _to_numeric_panel(x)
    out = (
        base.ewm(alpha=max(min(factor, 1.0), 1e-9), min_periods=1, adjust=False).mean().rolling(d, min_periods=1).mean()
    )
    return _clean_nonfinite(out)


def _days_from_last_change(x):
    if isinstance(x, pd.Series):
        arr = np.asarray(x.values)
        out = np.zeros(len(arr), dtype=float)
        for i in range(1, len(arr)):
            out[i] = out[i - 1] + 1.0 if arr[i] == arr[i - 1] else 0.0
        return pd.Series(out, index=x.index, name=x.name)

    if isinstance(x, pd.DataFrame):
        arr = np.asarray(x.values)
        if arr.size == 0:
            return x.astype(float)
        n, m = arr.shape
        out = np.zeros((n, m), dtype=float)
        for i in range(1, n):
            same = arr[i] == arr[i - 1]
            out[i] = np.where(same, out[i - 1] + 1.0, 0.0)
        return pd.DataFrame(out, index=x.index, columns=x.columns)

    arr = np.asarray(x)
    out = np.zeros(len(arr), dtype=float)
    for i in range(1, len(arr)):
        out[i] = out[i - 1] + 1.0 if arr[i] == arr[i - 1] else 0.0
    return out


def _hump(x, hump=0.01):
    """
    Stateful delta limiter:
    keep previous output if today's change is within hump; otherwise move by hump step.
    """
    step = max(abs(float(hump)), 0.0)
    if isinstance(x, pd.Series):
        arr = _to_float_array(x.values)
        out = _hump_array(arr, step=step)
        return pd.Series(out, index=x.index, name=x.name)

    if isinstance(x, pd.DataFrame):
        arr = np.asarray(x.values, dtype=float)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        out = np.full(arr.shape, np.nan, dtype=float)
        for j in range(arr.shape[1]):
            out[:, j] = _hump_array(arr[:, j], step=step)
        return pd.DataFrame(out, index=x.index, columns=x.columns)

    arr = _to_float_array(x)
    return _hump_array(arr, step=step)


def _argmax_relative_lag(arr) -> float:
    values = np.asarray(arr, dtype=float)
    finite = np.isfinite(values)
    if not bool(finite.any()):
        return np.nan
    # Reverse to make index 0 = current day, matching WQ relative index semantics.
    safe = np.where(finite[::-1], values[::-1], -np.inf)
    return float(np.argmax(safe))


def _argmin_relative_lag(arr) -> float:
    values = np.asarray(arr, dtype=float)
    finite = np.isfinite(values)
    if not bool(finite.any()):
        return np.nan
    # Reverse to make index 0 = current day, matching WQ relative index semantics.
    safe = np.where(finite[::-1], values[::-1], np.inf)
    return float(np.argmin(safe))


def _ts_rank_array(arr: np.ndarray, window: int) -> np.ndarray:
    if bn is None:
        return _ts_rank_array_no_bn(arr, window)

    n = int(arr.shape[0]) if arr.ndim >= 1 else 0
    if n <= 0:
        return np.asarray(arr, dtype=float)
    window = max(1, min(int(window), n))

    if arr.ndim == 1:
        mr = bn.move_rank(arr, window=window, min_count=1)
        cnt = bn.move_sum(np.isfinite(arr).astype(float), window=window, min_count=1)
    else:
        mr = bn.move_rank(arr, window=window, min_count=1, axis=0)
        cnt = bn.move_sum(np.isfinite(arr).astype(float), window=window, min_count=1, axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        # bn.move_rank scale:
        #   mr = 2 * (rank - 1) / (n - 1) - 1
        # Recover pandas pct rank = rank / n.
        pct = (((mr + 1.0) * 0.5) * (cnt - 1.0) + 1.0) / cnt
        pct = np.where(cnt > 1.0, pct, np.where(cnt == 1.0, 1.0, np.nan))
        pct = np.where(np.isfinite(arr), pct, np.nan)
    return pct


def _ts_rank_array_no_bn(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Fast fallback for rolling rank(pct=True) of the latest value in each window.

    This avoids pandas rolling.apply (Python callback per window), and uses
    vectorized lag-wise comparisons. Complexity is O(window * rows * cols).
    """
    values = np.asarray(arr, dtype=float)
    if values.ndim == 0:
        return np.asarray(values, dtype=float)

    squeeze = values.ndim == 1
    if squeeze:
        values2d = values.reshape(-1, 1)
    else:
        values2d = values

    n, m = values2d.shape
    if n == 0:
        return values if squeeze else values2d

    w = max(1, min(int(window), n))
    finite_cur = np.isfinite(values2d)

    less = np.zeros((n, m), dtype=np.uint32)
    equal = np.zeros((n, m), dtype=np.uint32)
    count = np.zeros((n, m), dtype=np.uint32)

    for lag in range(w):
        if lag == 0:
            lhs = values2d
            rhs = values2d
            tgt = slice(None)
        else:
            lhs = values2d[lag:]
            rhs = values2d[:-lag]
            tgt = slice(lag, None)

        valid = finite_cur[tgt] & np.isfinite(rhs)
        count[tgt] += valid
        less[tgt] += valid & (rhs < lhs)
        equal[tgt] += valid & (rhs == lhs)

    rank_avg = less.astype(float) + (equal.astype(float) + 1.0) * 0.5
    out = np.full((n, m), np.nan, dtype=float)
    valid_out = (count > 0) & finite_cur
    out[valid_out] = rank_avg[valid_out] / count[valid_out]

    if squeeze:
        return out[:, 0]
    return out


def _to_numeric_panel(x):
    if hasattr(x, "astype"):
        out = x.astype(float)
        return _clean_nonfinite(out)
    arr = _to_float_array(x)
    return _clean_nonfinite(arr)


def _to_float_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.where(np.isfinite(arr), arr, np.nan)


def _hump_array(arr: np.ndarray, step: float) -> np.ndarray:
    values = np.asarray(arr, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    prev = np.nan
    for i, value in enumerate(values):
        if not np.isfinite(prev):
            out[i] = value if np.isfinite(value) else np.nan
            prev = out[i]
            continue
        if not np.isfinite(value):
            out[i] = prev
            prev = out[i]
            continue
        delta = value - prev
        if abs(delta) <= step:
            out[i] = prev
        else:
            out[i] = prev + np.sign(delta) * step
        prev = out[i]
    return out


def _clean_nonfinite(value):
    if hasattr(value, "replace"):
        return value.replace([np.inf, -np.inf], np.nan)
    arr = np.asarray(value)
    if arr.ndim == 0:
        scalar = float(arr)
        return scalar if np.isfinite(scalar) else np.nan
    return np.where(np.isfinite(arr), arr, np.nan)
