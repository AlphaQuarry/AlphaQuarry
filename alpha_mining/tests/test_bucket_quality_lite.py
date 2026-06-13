from __future__ import annotations

import pandas as pd

from alpha_mining.mining.bucket_quality_lite import (
    BucketQualityConfig,
    evaluate_bucket_quality,
)
from alpha_mining.panel_store import PanelStore


def _store() -> PanelStore:
    rows = []
    for date in pd.date_range("2024-01-01", periods=4):
        for idx, code in enumerate(["A", "B", "C", "D", "E", "F"], start=1):
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "circ_mv": float(idx),
                    "sparse": float(idx) if idx == 1 else None,
                }
            )
    return PanelStore.from_long_frame(pd.DataFrame(rows))


def test_bucket_quality_lite_scores_valid_bucket_group() -> None:
    result = evaluate_bucket_quality(
        "bucket(rank(circ_mv), '0,1,0.5')",
        _store(),
        BucketQualityConfig(min_coverage=0.30, min_median_group_size=2, min_group_count=2),
    )

    assert result.status == "pass"
    assert result.coverage == 1.0
    assert result.group_count_median >= 2
    assert result.group_size_median >= 2
    assert result.quality_score > 0
    assert result.quality_status == "pass"
    assert result.nan_group_ratio == 0.0
    assert result.is_composite is False


def test_bucket_quality_lite_marks_sparse_bucket_low_quality_without_error() -> None:
    result = evaluate_bucket_quality(
        "bucket(rank(sparse), '0,1,0.5')",
        _store(),
        BucketQualityConfig(min_coverage=0.80, min_median_group_size=2, min_group_count=2),
    )

    assert result.status == "low_quality"
    assert result.coverage < 0.80
    assert result.quality_score < 1.0
    assert result.error == ""
    assert result.quality_status == "low_quality"


def test_bucket_quality_lite_skips_missing_fields() -> None:
    result = evaluate_bucket_quality(
        "bucket(rank(missing_field), '0,1,0.5')",
        _store(),
        BucketQualityConfig(),
    )

    assert result.status == "skipped"
    assert result.reject_reason == "missing_bucket_fields:missing_field"
    assert result.quality_score == 0.0


def test_bucket_quality_lite_identifies_composite_and_nan_group_ratio() -> None:
    result = evaluate_bucket_quality(
        "group_cartesian_product(industry, bucket(rank(circ_mv), '0,1,0.5'))",
        PanelStore.from_long_frame(
            pd.DataFrame(
                [
                    {
                        "date": "2024-01-01",
                        "code": "A",
                        "circ_mv": 1.0,
                        "industry": "bank",
                    },
                    {
                        "date": "2024-01-01",
                        "code": "B",
                        "circ_mv": 2.0,
                        "industry": None,
                    },
                    {
                        "date": "2024-01-02",
                        "code": "A",
                        "circ_mv": 1.0,
                        "industry": "bank",
                    },
                    {
                        "date": "2024-01-02",
                        "code": "B",
                        "circ_mv": 2.0,
                        "industry": None,
                    },
                ]
            ),
            group_fields=["industry"],
        ),
        BucketQualityConfig(max_nan_group_ratio=0.25),
    )

    assert result.is_composite is True
    assert result.nan_group_ratio > 0.25
    assert result.quality_status == "low_quality"
    assert "nan_group_ratio_above_max" in result.reject_reason
