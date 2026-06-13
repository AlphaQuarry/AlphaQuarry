from __future__ import annotations

from pathlib import Path


def test_maintenance_checklist_covers_daily_weekly_and_failure_triage() -> None:
    text = Path("docs/maintenance.md").read_text(encoding="utf-8")

    for phrase in [
        "每日检查",
        "每周检查",
        "Overview freshness",
        "preflight",
        "Closed Loop",
        "Live readiness",
        "Data Health coverage",
        "run_health",
        "audit_analysis_artifacts",
        "data/lake",
        "data/duckdb",
        "data/alpha_universe_store",
        "stderr",
        "锁所有者",
        "Dashboard API 测试",
        "alpha_mining/tests",
    ]:
        assert phrase in text


def test_readme_and_quickstart_link_to_maintenance_checklist() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    quickstart = Path("docs/quickstart.md").read_text(encoding="utf-8")

    assert "docs/maintenance.md" in readme
    assert "docs/maintenance.md" in quickstart


def test_user_visible_status_terms_stay_consistent() -> None:
    paths = [
        Path("docs/quickstart.md"),
        Path("docs/maintenance.md"),
        Path("dashboard/frontend/src/utils/statusCopy.ts"),
        Path("dashboard/frontend/src/components/ClosedLoopPage.tsx"),
        Path("dashboard/frontend/src/components/ComparePage.tsx"),
        Path("dashboard/frontend/src/components/DataPage.tsx"),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    for phrase in [
        "Missing artifact",
        "Partial metrics",
        "Not refreshed",
        "Interrupted",
        "Running outside dashboard",
    ]:
        assert phrase in combined
