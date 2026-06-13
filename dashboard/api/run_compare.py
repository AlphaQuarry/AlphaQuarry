from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def compare_metric_rows(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
    left_summary = _run_metric_summary(left)
    right_summary = _run_metric_summary(right)
    keys = [
        "factor_count",
        "effective_count",
        "accepted_like_count",
        "feedback_score_mean",
        "feedback_score_median",
        "ic_mean",
        "long_short_sharpe_mean",
        "top_score",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        left_value = left_summary.get(key)
        right_value = right_summary.get(key)
        out[key] = {
            "left": left_value,
            "right": right_value,
            "delta": _delta(right_value, left_value),
        }
    return out


def compare_artifact_status(run: Any, frame: pd.DataFrame) -> str:
    path = getattr(run, "metrics_path", None)
    if path is None or not Path(path).exists():
        return "missing_metrics"
    if frame.empty or "factor" not in frame.columns:
        return "invalid_metrics"
    required = {"feedback_score", "ic_mean", "long_short_sharpe_ratio"}
    if not required.issubset(set(frame.columns)):
        return "partial_metrics"
    return "complete"


def top_overlap(left: pd.DataFrame, right: pd.DataFrame, *, top_n: int) -> dict[str, Any]:
    left_top = _top_factors(left, top_n=top_n)
    right_top = _top_factors(right, top_n=top_n)
    shared = sorted(set(left_top) & set(right_top))
    denominator = max(1, min(len(left_top), len(right_top), int(top_n)))
    return {
        "left_top": left_top,
        "right_top": right_top,
        "shared_factors": shared,
        "left_only": [factor for factor in left_top if factor not in set(right_top)],
        "right_only": [factor for factor in right_top if factor not in set(left_top)],
        "overlap_count": int(len(shared)),
        "overlap_ratio": float(len(shared) / denominator),
    }


def _run_metric_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "factor" not in frame.columns:
        return {
            "factor_count": 0,
            "effective_count": 0,
            "accepted_like_count": 0,
            "feedback_score_mean": None,
            "feedback_score_median": None,
            "ic_mean": None,
            "long_short_sharpe_mean": None,
            "top_score": None,
        }
    work = frame.copy()
    score = pd.to_numeric(work.get("feedback_score", pd.Series(dtype=float)), errors="coerce")
    tier = work.get("effectiveness_tier", pd.Series("", index=work.index)).fillna("").astype(str).str.upper()
    effective = tier.isin({"S", "A", "B"}) | (score >= 60)
    return {
        "factor_count": int(len(work)),
        "effective_count": int(effective.sum()),
        "accepted_like_count": int(effective.sum()),
        "feedback_score_mean": _nullable_float(score.mean()),
        "feedback_score_median": _nullable_float(score.median()),
        "ic_mean": _nullable_float(pd.to_numeric(work.get("ic_mean", pd.Series(dtype=float)), errors="coerce").mean()),
        "long_short_sharpe_mean": _nullable_float(
            pd.to_numeric(
                work.get("long_short_sharpe_ratio", pd.Series(dtype=float)),
                errors="coerce",
            ).mean()
        ),
        "top_score": _nullable_float(score.max()),
    }


def _delta(right: Any, left: Any) -> float | None:
    try:
        if right is None or left is None:
            return None
        return float(right) - float(left)
    except Exception:
        return None


def _top_factors(frame: pd.DataFrame, *, top_n: int) -> list[str]:
    if frame.empty or "factor" not in frame.columns:
        return []
    work = frame.copy()
    score = pd.to_numeric(work.get("feedback_score", pd.Series(dtype=float)), errors="coerce")
    work = work.assign(_score=score)
    work = work.sort_values(
        ["_score", "factor"],
        ascending=[False, True],
        na_position="last",
        kind="mergesort",
    )
    return [str(x) for x in work["factor"].head(max(1, int(top_n))).tolist()]


def _nullable_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None
