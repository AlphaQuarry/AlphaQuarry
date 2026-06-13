from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..engine import ExpressionEngine
from ..panel_store import PanelStore
from ..validators import expression_stats


@dataclass(frozen=True)
class SampleEvaluatorConfig:
    enabled: bool = False
    min_coverage: float = 0.30
    max_inf_ratio: float = 0.01
    max_abs_value: float = 1.0e12
    reject_constant: bool = True
    skip_missing_fields: bool = True


@dataclass(frozen=True)
class SampleEvaluationResult:
    expression: str
    passed: bool
    reject_reason: str = ""
    coverage: float = 0.0
    inf_ratio: float = 0.0
    extreme_ratio: float = 0.0
    unique_count: int = 0
    error: str = ""
    status: str = ""


class SampleEvaluator:
    def __init__(self, config: SampleEvaluatorConfig | None = None) -> None:
        self.config = config or SampleEvaluatorConfig()

    def evaluate(self, expression: str, panel_store: PanelStore) -> SampleEvaluationResult:
        expr = str(expression or "").strip()
        missing_fields = _missing_expression_fields(expr, panel_store)
        if missing_fields and bool(self.config.skip_missing_fields):
            return SampleEvaluationResult(
                expr,
                True,
                f"missing_sample_fields:{','.join(missing_fields)}",
                status="skipped",
            )
        try:
            value = ExpressionEngine(panel_store).eval(expr, use_cache=False)
            frame = _as_frame(value)
        except Exception as exc:
            if bool(self.config.skip_missing_fields) and _is_missing_field_error(exc):
                return SampleEvaluationResult(expr, True, "missing_sample_fields", status="skipped")
            return SampleEvaluationResult(expr, False, "eval_error", error=f"{type(exc).__name__}: {exc}")

        if frame.empty:
            return SampleEvaluationResult(expr, False, "empty_output")
        arr = frame.to_numpy(dtype=float, copy=False)
        total = int(arr.size)
        if total <= 0:
            return SampleEvaluationResult(expr, False, "empty_output")

        finite = np.isfinite(arr)
        inf_ratio = float(np.isinf(arr).sum()) / float(total)
        coverage = float(finite.sum()) / float(total)
        finite_values = arr[finite]
        unique_count = int(pd.Series(finite_values).nunique(dropna=True)) if finite_values.size else 0
        extreme_ratio = float((np.abs(np.where(finite, arr, 0.0)) > float(self.config.max_abs_value)).sum()) / float(
            total
        )

        if inf_ratio > float(self.config.max_inf_ratio):
            reason = "inf_ratio_above_max"
        elif coverage <= 0.0:
            reason = "all_nan_output"
        elif coverage < float(self.config.min_coverage):
            reason = "coverage_below_min"
        elif extreme_ratio > 0.0:
            reason = "extreme_ratio_above_max"
        elif bool(self.config.reject_constant) and unique_count <= 1:
            reason = "constant_output"
        else:
            reason = ""

        return SampleEvaluationResult(
            expr,
            not bool(reason),
            reason,
            coverage=coverage,
            inf_ratio=inf_ratio,
            extreme_ratio=extreme_ratio,
            unique_count=unique_count,
            status="reject" if reason else "pass",
        )


def _missing_expression_fields(expression: str, panel_store: PanelStore) -> list[str]:
    try:
        stats = expression_stats(str(expression or ""))
    except Exception:
        return []
    available = set(panel_store.available_scalar_fields())
    available.update(panel_store.available_vector_fields())
    available.update(panel_store.available_group_fields())
    missing = [str(field) for field in stats.unique_fields if str(field) not in available]
    return sorted(dict.fromkeys(missing))


def _is_missing_field_error(exc: Exception) -> bool:
    text = str(exc)
    return isinstance(exc, KeyError) and "not found in PanelStore" in text


def _as_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, pd.Series):
        return value.to_frame()
    return pd.DataFrame([[value]])
