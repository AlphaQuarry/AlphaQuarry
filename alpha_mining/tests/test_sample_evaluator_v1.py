from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_mining.mining.sample_evaluator import SampleEvaluator, SampleEvaluatorConfig
from alpha_mining.panel_store import PanelStore


def _panel_store() -> PanelStore:
    rows = []
    for idx, date in enumerate(pd.date_range("2024-01-01", periods=6)):
        for col_idx, code in enumerate(["A", "B", "C"], start=1):
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "close": float(idx + col_idx),
                    "constant": 1.0,
                    "zero": 0.0,
                    "bad": np.inf if code == "A" else 1.0,
                    "sparse": np.nan if code != "A" else float(idx),
                }
            )
    return PanelStore.from_long_frame(pd.DataFrame(rows))


def test_sample_evaluator_rejects_constant_low_coverage_and_high_inf_outputs() -> None:
    evaluator = SampleEvaluator(SampleEvaluatorConfig(min_coverage=0.5, max_inf_ratio=0.0))
    store = _panel_store()

    assert evaluator.evaluate("rank(close)", store).passed
    assert evaluator.evaluate("constant", store).reject_reason == "constant_output"
    assert evaluator.evaluate("sparse", store).reject_reason == "coverage_below_min"
    assert evaluator.evaluate("bad", store).reject_reason == "inf_ratio_above_max"


def test_sample_evaluator_skips_missing_sample_fields_instead_of_rejecting() -> None:
    evaluator = SampleEvaluator(SampleEvaluatorConfig(min_coverage=0.5, max_inf_ratio=0.0))
    store = _panel_store()

    result = evaluator.evaluate("rank(missing_field)", store)

    assert result.passed
    assert result.status == "skipped"
    assert result.reject_reason == "missing_sample_fields:missing_field"
    assert result.error == ""
