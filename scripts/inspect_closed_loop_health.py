from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_mining.workflow.universe_store import (
    DEFAULT_UNIVERSE_BASE_DIR,
    get_universe_paths,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect recent closed-loop run_health.jsonl status.")
    parser.add_argument("--base-dir", default=DEFAULT_UNIVERSE_BASE_DIR)
    parser.add_argument("--universe", default="cn_all")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    summary = inspect_closed_loop_health(
        base_dir=args.base_dir,
        universe_name=args.universe,
        limit=max(1, int(args.limit)),
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))


def inspect_closed_loop_health(*, base_dir: str | Path, universe_name: str, limit: int = 20) -> dict[str, Any]:
    paths = get_universe_paths(base_dir=base_dir, universe_name=universe_name)
    path = paths["feedback_dir"] / "run_health.jsonl"
    records = _load_health_records(path)
    recent = records[-max(1, int(limit)) :]
    status_counts = Counter(str(row.get("status", "")) for row in recent if str(row.get("status", "")))
    error_counts = Counter(str(row.get("error", "")) for row in recent if str(row.get("error", "")))
    retention_deleted = 0
    memory_warning_count = 0
    hard_limit_count = 0
    scoreboard_rows: list[int] = []
    for row in recent:
        retention = row.get("artifact_retention_summary", {})
        if isinstance(retention, dict):
            retention_deleted += _deleted_count(retention)
        memory = row.get("source_chunk_memory", {})
        if isinstance(memory, dict):
            memory_warning_count += int(memory.get("source_chunk_mem_warning_count", 0) or 0)
        if bool(row.get("source_chunk_hard_limit_triggered", False)):
            hard_limit_count += 1
        scoreboard_rows.append(int(row.get("scoreboard_rows", 0) or 0))
    return {
        "path": str(path.as_posix()),
        "exists": bool(path.exists()),
        "total_records": int(len(records)),
        "inspected_records": int(len(recent)),
        "status_counts": dict(status_counts),
        "error_counts": dict(error_counts),
        "retention_deleted_items": int(retention_deleted),
        "memory_warning_count": int(memory_warning_count),
        "hard_limit_count": int(hard_limit_count),
        "scoreboard_rows_min": min(scoreboard_rows) if scoreboard_rows else 0,
        "scoreboard_rows_max": max(scoreboard_rows) if scoreboard_rows else 0,
        "latest_status": str(recent[-1].get("status", "")) if recent else "",
        "latest_created_at_utc": str(recent[-1].get("created_at_utc", "")) if recent else "",
    }


def _load_health_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _deleted_count(retention: dict[str, Any]) -> int:
    total = 0
    for value in retention.values():
        if isinstance(value, dict):
            total += int(value.get("deleted_files", 0) or 0)
            total += int(value.get("deleted_dirs", 0) or 0)
    total += int(retention.get("deleted_files", 0) or 0)
    total += int(retention.get("deleted_dirs", 0) or 0)
    return total


if __name__ == "__main__":
    main()
