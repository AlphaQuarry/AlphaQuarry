from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..engine import ExpressionEngine
from ..panel_store import PanelStore
from ..validators import expression_stats


@dataclass(frozen=True)
class BucketQualityConfig:
    min_coverage: float = 0.50
    min_median_group_size: int = 5
    min_group_count: int = 3
    max_nan_group_ratio: float = 0.30


@dataclass(frozen=True)
class BucketQualityResult:
    expression: str
    status: str
    quality_status: str = ""
    reject_reason: str = ""
    coverage: float = 0.0
    group_count_median: float = 0.0
    group_size_median: float = 0.0
    group_size_min: float = 0.0
    nan_group_ratio: float = 0.0
    is_composite: bool = False
    quality_score: float = 0.0
    error: str = ""


def evaluate_bucket_quality(
    bucket_expression: str,
    panel_store: PanelStore,
    config: BucketQualityConfig | None = None,
) -> BucketQualityResult:
    cfg = config or BucketQualityConfig()
    expr = str(bucket_expression or "").strip()
    is_composite = _is_composite_bucket_expression(expr)
    if not expr:
        return BucketQualityResult(
            expr,
            "skipped",
            "skipped",
            "empty_bucket_expression",
            is_composite=is_composite,
        )

    missing = _missing_expression_fields(expr, panel_store)
    if missing:
        return BucketQualityResult(
            expr,
            "skipped",
            "skipped",
            f"missing_bucket_fields:{','.join(missing)}",
            is_composite=is_composite,
        )

    try:
        value = ExpressionEngine(panel_store).eval(expr, use_cache=False)
        frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    except Exception as exc:
        return BucketQualityResult(
            expr,
            "skipped",
            "error",
            "bucket_eval_error",
            is_composite=is_composite,
            error=f"{type(exc).__name__}: {exc}",
        )

    if frame.empty:
        return BucketQualityResult(expr, "skipped", "skipped", "empty_bucket_output", is_composite=is_composite)

    total = int(frame.size)
    if total <= 0:
        return BucketQualityResult(expr, "skipped", "skipped", "empty_bucket_output", is_composite=is_composite)

    valid = frame.notna()
    coverage = float(valid.to_numpy().sum()) / float(total)
    nan_group_ratio = float(1.0 - coverage)
    group_counts: list[int] = []
    group_sizes: list[int] = []
    for _, row in frame.iterrows():
        counts = row.dropna().astype(str).value_counts()
        if counts.empty:
            continue
        group_counts.append(int(len(counts)))
        group_sizes.extend([int(x) for x in counts.tolist()])

    group_count_median = float(np.median(group_counts)) if group_counts else 0.0
    group_size_median = float(np.median(group_sizes)) if group_sizes else 0.0
    group_size_min = float(min(group_sizes)) if group_sizes else 0.0
    quality_score = _quality_score(
        coverage=coverage,
        group_count_median=group_count_median,
        group_size_median=group_size_median,
        cfg=cfg,
    )

    reasons: list[str] = []
    if coverage < float(cfg.min_coverage):
        reasons.append("coverage_below_min")
    if group_count_median < float(cfg.min_group_count):
        reasons.append("group_count_below_min")
    if group_size_median < float(cfg.min_median_group_size):
        reasons.append("group_size_below_min")
    if nan_group_ratio > float(cfg.max_nan_group_ratio):
        reasons.append("nan_group_ratio_above_max")
    quality_status = "low_quality" if reasons else "pass"

    return BucketQualityResult(
        expression=expr,
        status=quality_status,
        quality_status=quality_status,
        reject_reason=",".join(reasons),
        coverage=coverage,
        group_count_median=group_count_median,
        group_size_median=group_size_median,
        group_size_min=group_size_min,
        nan_group_ratio=nan_group_ratio,
        is_composite=is_composite,
        quality_score=quality_score,
    )


def _quality_score(
    *,
    coverage: float,
    group_count_median: float,
    group_size_median: float,
    cfg: BucketQualityConfig,
) -> float:
    coverage_score = _clip01(coverage / max(1.0e-12, float(cfg.min_coverage)))
    group_count_score = _clip01(group_count_median / max(1.0, float(cfg.min_group_count)))
    group_size_score = _clip01(group_size_median / max(1.0, float(cfg.min_median_group_size)))
    return float(0.40 * coverage_score + 0.30 * group_count_score + 0.30 * group_size_score)


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


def _clip01(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    if not np.isfinite(out):
        return 0.0
    return max(0.0, min(1.0, out))


def _is_composite_bucket_expression(expression: str) -> bool:
    text = str(expression or "").lower()
    return "group_cartesian_product" in text
