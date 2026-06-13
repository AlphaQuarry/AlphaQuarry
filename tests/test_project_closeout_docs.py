from __future__ import annotations

from pathlib import Path


def test_readme_contains_concept_map_and_dashboard_security_boundary() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    assert "Data Lake -> DuckDB -> Universe Store -> Closed Loop -> Analysis Run -> Factor Library -> Superalpha -> Live" in text
    assert "127.0.0.1" in text
    assert "do not expose" in text.lower()


def test_quickstart_keeps_dashboard_local_only_boundary() -> None:
    text = Path("docs/quickstart.md").read_text(encoding="utf-8")

    assert "127.0.0.1" in text
    assert "不要暴露" in text


def test_quickstart_contains_status_playbook_and_resource_notes() -> None:
    text = Path("docs/quickstart.md").read_text(encoding="utf-8")

    for phrase in [
        "状态处理手册",
        "preflight warning",
        "coverage not refreshed",
        "missing artifact",
        "memory hard limit",
        "stale lock",
        "data/lake",
        "data/duckdb",
        "data/alpha_universe_store",
        "scripts/audit_analysis_artifacts.py",
    ]:
        assert phrase in text
