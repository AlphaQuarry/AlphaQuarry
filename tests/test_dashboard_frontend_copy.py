from __future__ import annotations

from pathlib import Path


def test_dashboard_workbench_pages_use_shared_state_copy() -> None:
    helper = Path("dashboard/frontend/src/utils/statusCopy.ts").read_text(encoding="utf-8")

    for expected in [
        "Loading data health",
        "No dashboard-launched closed-loop jobs yet.",
        "Missing artifact",
        "API error",
    ]:
        assert expected in helper

    for path in [
        Path("dashboard/frontend/src/components/ClosedLoopPage.tsx"),
        Path("dashboard/frontend/src/components/ComparePage.tsx"),
        Path("dashboard/frontend/src/components/DataPage.tsx"),
    ]:
        assert "STATUS_COPY" in path.read_text(encoding="utf-8")
