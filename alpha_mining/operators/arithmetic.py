from __future__ import annotations

import numpy as np

from ..registry import OperatorRegistry

_DIV_EPS = 1.0e-12


def register_operators(registry: OperatorRegistry) -> None:
    registry.register("add", lambda x, y: x + y)
    registry.register("sub", lambda x, y: x - y)
    registry.register("mul", lambda x, y: x * y)

    def _safe_div(x, y):
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            if hasattr(y, "where"):
                abs_y = np.abs(y)
                # Treat zero and near-zero denominators as NaN to avoid overflow spikes.
                denom = y.where(abs_y >= _DIV_EPS, np.nan)
            elif isinstance(y, (int, float, np.number)) and abs(float(y)) < _DIV_EPS:
                denom = np.nan
            else:
                arr = np.asarray(y)
                if arr.ndim == 0:
                    denom = np.nan if abs(float(arr)) < _DIV_EPS else y
                else:
                    denom = np.where(np.abs(arr) < _DIV_EPS, np.nan, arr)
            return _clean_nonfinite(x / denom)

    registry.register("div", _safe_div)
    registry.register("divide", _safe_div)
    registry.register("subtract", lambda x, y: x - y)
    registry.register("multiply", lambda x, y: x * y)
    registry.register("inverse", lambda x: _safe_div(1.0, x))
    registry.register("power", _power)
    registry.register("signed_power", _signed_power)
    registry.register("max", lambda x, y: _clean_nonfinite(np.maximum(x, y)))
    registry.register("min", lambda x, y: _clean_nonfinite(np.minimum(x, y)))


def _power(x, y):
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return _clean_nonfinite(np.power(x, y))


def _signed_power(x, y):
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return _clean_nonfinite(np.sign(x) * np.power(np.abs(x), y))


def _clean_nonfinite(value):
    if hasattr(value, "replace"):
        return value.replace([np.inf, -np.inf], np.nan)
    arr = np.asarray(value)
    if arr.ndim == 0:
        scalar = float(arr)
        return scalar if np.isfinite(scalar) else np.nan
    return np.where(np.isfinite(arr), arr, np.nan)
