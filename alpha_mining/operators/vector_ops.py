from __future__ import annotations

import numpy as np
import pandas as pd

from ..panel_store import VectorPanel
from ..registry import OperatorRegistry


def register_operators(registry: OperatorRegistry) -> None:
    registry.register("vec_avg", _vec_avg)
    registry.register("vec_sum", _vec_sum)
    registry.register("vec_stddev", _vec_stddev)
    registry.register("vec_max", _vec_max)
    registry.register("vec_min", _vec_min)
    registry.register("vec_count", _vec_count)


def _to_array(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float)
        return arr[np.isfinite(arr)]
    return np.asarray([], dtype=float)


def _is_vector_panel(x) -> bool:
    return isinstance(x, VectorPanel)


def _vector_panel_reduce(v: VectorPanel, mode: str) -> pd.DataFrame:
    if not v.components:
        raise ValueError(f"Vector field '{v.name}' has no components")

    index = v.index
    columns = v.columns
    shape = (len(index), len(columns))
    sum_arr = np.zeros(shape, dtype=float)
    count_arr = np.zeros(shape, dtype=np.int32)
    min_arr = np.full(shape, np.nan, dtype=float)
    max_arr = np.full(shape, np.nan, dtype=float)
    sumsq_arr = np.zeros(shape, dtype=float)
    min_initialized = False
    max_initialized = False

    needs_sum = mode in {"avg", "sum", "std"}
    needs_count = mode in {"avg", "count", "std"}
    needs_min = mode == "min"
    needs_max = mode == "max"
    needs_sumsq = mode == "std"

    for comp in v.components:
        arr = np.asarray(comp.values, dtype=float)
        finite = np.isfinite(arr)

        if needs_sum:
            sum_arr += np.where(finite, arr, 0.0)
        if needs_count:
            count_arr += finite.astype(np.int32)
        if needs_min:
            if not min_initialized:
                min_arr = np.where(finite, arr, np.nan)
                min_initialized = True
            else:
                min_arr = np.where(
                    finite,
                    np.where(np.isfinite(min_arr), np.minimum(min_arr, arr), arr),
                    min_arr,
                )
        if needs_max:
            if not max_initialized:
                max_arr = np.where(finite, arr, np.nan)
                max_initialized = True
            else:
                max_arr = np.where(
                    finite,
                    np.where(np.isfinite(max_arr), np.maximum(max_arr, arr), arr),
                    max_arr,
                )
        if needs_sumsq:
            sumsq_arr += np.where(finite, arr * arr, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        if mode == "sum":
            out = np.where(count_arr > 0, sum_arr, np.nan)
        elif mode == "avg":
            out = np.where(count_arr > 0, sum_arr / count_arr, np.nan)
        elif mode == "count":
            out = count_arr
        elif mode == "min":
            out = min_arr
        elif mode == "max":
            out = max_arr
        elif mode == "std":
            mean = np.where(count_arr > 0, sum_arr / count_arr, np.nan)
            second = np.where(count_arr > 0, sumsq_arr / count_arr, np.nan)
            variance = second - mean * mean
            variance = np.where(variance < 0, 0.0, variance)
            out = np.sqrt(variance)
            out = np.where(count_arr > 0, out, np.nan)
        else:
            raise ValueError(f"Unsupported vector reduction mode: {mode}")

    return pd.DataFrame(out, index=index, columns=columns)


def _vec_reduce(x, fn, default=np.nan):
    if _is_vector_panel(x):
        raise TypeError("VectorPanel must be reduced via _vector_panel_reduce")
    if hasattr(x, "map"):

        def _apply(v):
            arr = _to_array(v)
            return fn(arr) if arr.size else default

        return x.map(_apply)
    arr = _to_array(x)
    return fn(arr) if arr.size else default


def _vec_avg(x):
    if _is_vector_panel(x):
        return _vector_panel_reduce(x, mode="avg")
    return _vec_reduce(x, np.mean)


def _vec_sum(x):
    if _is_vector_panel(x):
        return _vector_panel_reduce(x, mode="sum")
    return _vec_reduce(x, np.sum, default=0.0)


def _vec_stddev(x):
    if _is_vector_panel(x):
        return _vector_panel_reduce(x, mode="std")
    return _vec_reduce(x, np.std)


def _vec_max(x):
    if _is_vector_panel(x):
        return _vector_panel_reduce(x, mode="max")
    return _vec_reduce(x, np.max)


def _vec_min(x):
    if _is_vector_panel(x):
        return _vector_panel_reduce(x, mode="min")
    return _vec_reduce(x, np.min)


def _vec_count(x):
    if _is_vector_panel(x):
        return _vector_panel_reduce(x, mode="count")
    return _vec_reduce(x, len, default=0.0)
