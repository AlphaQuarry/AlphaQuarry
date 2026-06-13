from __future__ import annotations

from pathlib import Path


TEXT_FILES = (
    Path("README.md"),
    Path("docs/quickstart.md"),
    Path("docs/live_superalpha_runbook.md"),
    Path("dashboard/frontend/src/components/ClosedLoopPage.tsx"),
    Path("dashboard/frontend/src/components/ComparePage.tsx"),
    Path("dashboard/frontend/src/components/DataPage.tsx"),
    Path("dashboard/frontend/src/components/LivePage.tsx"),
    Path("dashboard/frontend/src/components/OverviewPage.tsx"),
    Path("dashboard/frontend/src/components/SuperalphaPage.tsx"),
    Path("alpha_mining/workflow/superalpha.py"),
)

FORBIDDEN_TEXT = (
    "\ufffd",
    "瀹炵洏",
    "浜哄伐",
    "方案 §",
)


def test_operator_facing_text_files_are_utf8_without_mojibake() -> None:
    for path in TEXT_FILES:
        text = path.read_text(encoding="utf-8")
        for bad in FORBIDDEN_TEXT:
            assert bad not in text, f"{path.as_posix()} contains suspicious text {bad!r}"


def test_live_superalpha_runbook_has_expected_content() -> None:
    text = Path("docs/live_superalpha_runbook.md").read_text(encoding="utf-8")

    assert "# Live SuperAlpha" in text
