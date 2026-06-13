from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dashboard.api.data_health import (
    base_frame_summary,
    coverage_counts,
    data_health_families,
    low_coverage_count,
    quality_artifact_summary,
    run_health_summary,
)


def test_coverage_helpers_handle_missing_partial_and_available() -> None:
    missing = pd.DataFrame({"field_name": ["close", "volume"]})
    partial = pd.DataFrame({"coverage_rate": [0.9, None, 0.2]})
    available = pd.DataFrame({"coverage_rate": [0.9, 0.7, 0.2]})

    assert coverage_counts(missing) == {
        "coverage_available_count": 0,
        "coverage_missing_count": 2,
        "coverage_status": "missing",
    }
    assert coverage_counts(partial)["coverage_status"] == "partial"
    assert coverage_counts(available)["coverage_status"] == "available"
    assert low_coverage_count(missing) == 0
    assert low_coverage_count(available, threshold=0.50) == 1


def test_data_health_families_normalizes_family_and_coverage() -> None:
    frame = pd.DataFrame(
        {
            "field_name": ["close", "volume", "moneyflow"],
            "factor_family": ["price", "", None],
            "coverage_rate": [0.9, 0.4, None],
            "is_searchable": [True, False, True],
            "available_end": ["2026-05-30", "2026-05-20", "2026-05-10"],
        }
    )

    rows = data_health_families(frame)

    assert {row["family"] for row in rows} == {"other", "price"}
    price = next(row for row in rows if row["family"] == "price")
    assert price["field_count"] == 1
    assert price["searchable_count"] == 1
    assert price["low_coverage_count"] == 0


def test_base_frame_run_health_and_quality_missing_payloads_are_stable(tmp_path: Path) -> None:
    missing_base = base_frame_summary(tmp_path / "missing.parquet")
    missing_health = run_health_summary(tmp_path / "missing.jsonl")
    missing_quality = quality_artifact_summary(tmp_path / "missing_quality.json")

    assert missing_base["exists"] is False
    assert missing_base["rows"] == 0
    assert missing_health["exists"] is False
    assert missing_health["total_records"] == 0
    assert missing_quality["exists"] is False
    assert missing_quality["overall_status"] == ""


def test_run_health_and_quality_summaries_read_existing_artifacts(tmp_path: Path) -> None:
    health_path = tmp_path / "run_health.jsonl"
    health_path.write_text(
        "\n".join(
            [
                json.dumps({"status": "ok", "scoreboard_rows": 5, "source_chunk_memory": {"source_chunk_mem_warning_count": 2}}),
                json.dumps({"status": "failed", "scoreboard_rows": 1, "source_chunk_hard_limit_triggered": True}),
            ]
        ),
        encoding="utf-8",
    )
    quality_path = tmp_path / "panel_quality.json"
    quality_path.write_text(
        json.dumps(
            {
                "overall_status": "warn",
                "generated_at_utc": "2026-06-01T00:00:00+00:00",
                "fields": [{"status": "warn"}, {"status": "fail"}, {"status": "pass"}],
            }
        ),
        encoding="utf-8",
    )

    health = run_health_summary(health_path)
    quality = quality_artifact_summary(quality_path)

    assert health["status_counts"] == {"ok": 1, "failed": 1}
    assert health["hard_limit_count"] == 1
    assert health["memory_warning_count"] == 2
    assert health["scoreboard_rows_min"] == 1
    assert quality["overall_status"] == "warn"
    assert quality["warn_field_count"] == 1
    assert quality["fail_field_count"] == 1
