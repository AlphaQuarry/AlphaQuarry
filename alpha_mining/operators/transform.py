from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from ..registry import OperatorRegistry


def register_operators(registry: OperatorRegistry) -> None:
    registry.register("abs", lambda x: x.abs() if hasattr(x, "abs") else abs(x))
    registry.register("sign", lambda x: np.sign(x))
    registry.register("log", _log)
    registry.register("sqrt", _sqrt)
    registry.register("s_log_1p", _s_log_1p)
    registry.register("reverse", lambda x: -x)
    registry.register("scale", _scale)
    registry.register("zscore", _zscore)
    registry.register("normalize", _normalize)
    registry.register("rank", _rank)
    registry.register("winsorize", _winsorize)
    registry.register("quantile", _quantile)
    registry.register("cs_quantile", _quantile)
    registry.register("truncate", _truncate)
    registry.register("left_tail", _left_tail)
    registry.register("right_tail", _right_tail)
    registry.register("zero_like", _zero_like)


def _rank(x):
    return x.rank(axis=1, pct=True) if hasattr(x, "rank") else x


def _zscore(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1).replace(0, np.nan), axis=0)


def _normalize(x):
    return x.sub(x.mean(axis=1), axis=0)


def _scale(x, scale=1.0, longscale=None, shortscale=None):
    if not hasattr(x, "apply"):
        arr = np.asarray(x, dtype=float)
        denom = np.nansum(np.abs(arr))
        if denom <= 0 or not np.isfinite(denom):
            return np.where(np.isfinite(arr), 0.0, np.nan)
        return arr / denom * float(scale)
    if longscale is not None or shortscale is not None:
        long_target = float(longscale if longscale is not None else float(scale) / 2.0)
        short_target = float(shortscale if shortscale is not None else float(scale) / 2.0)
        return x.apply(lambda row: _scale_row_long_short(row, long_target, short_target), axis=1)
    scale_value = float(scale)
    row_sum = x.abs().sum(axis=1).replace(0, np.nan)
    return _clean_nonfinite(x.div(row_sum, axis=0) * scale_value)


def _scale_row_long_short(row, long_target: float, short_target: float):
    values = pd.to_numeric(row, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = values * 0.0
    pos = values > 0
    neg = values < 0
    pos_sum = values[pos].sum(skipna=True)
    neg_sum = values[neg].abs().sum(skipna=True)
    if pos_sum > 0:
        out.loc[pos] = values.loc[pos] / pos_sum * max(0.0, float(long_target))
    if neg_sum > 0:
        out.loc[neg] = -values.loc[neg].abs() / neg_sum * max(0.0, float(short_target))
    return out


def _winsorize(x, std=4.0):
    mean = x.mean(axis=1)
    sigma = x.std(axis=1).replace(0, np.nan)
    upper = mean + std * sigma
    lower = mean - std * sigma
    return x.clip(lower=lower, upper=upper, axis=0)


def _quantile(x, driver: str = "gaussian", sigma: float = 1.0):
    """
    Cross-sectional quantile mapping.
    - uniform: return pct-rank in [0, 1]
    - gaussian: map pct-rank through inverse normal CDF and multiply by sigma
    """
    ranked = _rank(x)
    mode = str(driver or "gaussian").strip().lower()
    if mode in {"uniform", "u"}:
        return _clean_nonfinite(ranked)
    if mode in {"gaussian", "normal", "n"}:
        scale = max(abs(float(sigma)), 1.0e-12)
        return _clean_nonfinite(_inverse_normal(ranked) * scale)
    raise ValueError(f"unsupported_quantile_driver:{driver}")


def _truncate(x, max_percent: float = 0.01):
    """Clip value magnitude to +/- max_percent."""
    bound = abs(float(max_percent))
    if hasattr(x, "clip"):
        return _clean_nonfinite(x.clip(lower=-bound, upper=bound))
    arr = np.asarray(x, dtype=float)
    return _clean_nonfinite(np.clip(arr, -bound, bound))


def _left_tail(x, maximum: float = 0.0):
    """Keep values <= maximum and mask others as NaN."""
    maximum = float(maximum)
    if hasattr(x, "where"):
        return _clean_nonfinite(x.where(x <= maximum, np.nan))
    arr = np.asarray(x, dtype=float)
    return _clean_nonfinite(np.where(arr <= maximum, arr, np.nan))


def _right_tail(x, minimum: float = 0.0):
    """Keep values >= minimum and mask others as NaN."""
    minimum = float(minimum)
    if hasattr(x, "where"):
        return _clean_nonfinite(x.where(x >= minimum, np.nan))
    arr = np.asarray(x, dtype=float)
    return _clean_nonfinite(np.where(arr >= minimum, arr, np.nan))


def _zero_like(x):
    if hasattr(x, "where"):
        return x * 0.0
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return 0.0
    return np.zeros_like(arr, dtype=float)


def _log(x):
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        if hasattr(x, "where"):
            safe = x.where(x > 0, np.nan)
        else:
            arr = np.asarray(x, dtype=float)
            safe = np.where(arr > 0, arr, np.nan)
        return _clean_nonfinite(np.log(safe))


def _sqrt(x):
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        if hasattr(x, "where"):
            safe = x.where(x >= 0, np.nan)
        else:
            arr = np.asarray(x, dtype=float)
            safe = np.where(arr >= 0, arr, np.nan)
        return _clean_nonfinite(np.sqrt(safe))


def _s_log_1p(x):
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return _clean_nonfinite(np.sign(x) * np.log1p(np.abs(x)))


def _inverse_normal(value, eps: float = 1.0e-6):
    dist = NormalDist()
    if hasattr(value, "to_numpy"):
        arr = np.asarray(value.to_numpy(dtype=float), dtype=float)
        out = np.full(arr.shape, np.nan, dtype=float)
        valid = np.isfinite(arr)
        clipped = np.clip(arr[valid], eps, 1.0 - eps)
        mapped = np.fromiter((dist.inv_cdf(float(v)) for v in clipped), dtype=float, count=clipped.size)
        out[valid] = mapped
        if hasattr(value, "columns"):
            import pandas as pd

            return pd.DataFrame(out, index=value.index, columns=value.columns)
        import pandas as pd

        return pd.Series(out, index=value.index, name=getattr(value, "name", None))
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        if not np.isfinite(float(arr)):
            return np.nan
        return dist.inv_cdf(float(np.clip(arr, eps, 1.0 - eps)))
    out = np.full(arr.shape, np.nan, dtype=float)
    valid = np.isfinite(arr)
    clipped = np.clip(arr[valid], eps, 1.0 - eps)
    mapped = np.fromiter((dist.inv_cdf(float(v)) for v in clipped), dtype=float, count=clipped.size)
    out[valid] = mapped
    return out


def _clean_nonfinite(value):
    if hasattr(value, "replace"):
        return value.replace([np.inf, -np.inf], np.nan)
    arr = np.asarray(value)
    if arr.ndim == 0:
        scalar = float(arr)
        return scalar if np.isfinite(scalar) else np.nan
    return np.where(np.isfinite(arr), arr, np.nan)
