from __future__ import annotations

import json
from typing import Any

from .artifacts import live_paths, utc_now_iso, write_json


def write_sa_job(
    *,
    config: Any,
    superalpha_id: str,
    job: dict[str, Any],
    update_success_latest: bool = True,
) -> dict[str, Any]:
    paths = live_paths(config.store_root, config.universe)
    payload = {
        "schema_version": 1,
        "superalpha_id": str(superalpha_id),
        "updated_at_utc": utc_now_iso(),
        **dict(job),
    }
    job_id = str(
        payload.get("job_id")
        or f"live_{config.universe}_{superalpha_id}_{payload['updated_at_utc'].replace(':', '').replace('-', '')}"
    )
    payload["job_id"] = job_id
    job_dir = paths.jobs_dir(superalpha_id)
    write_json(job_dir / f"{job_id}.json", payload)
    write_json(job_dir / "latest.json", payload)
    return payload


def count_sa_runs_on_date(*, config: Any, superalpha_id: str, run_date: str) -> int:
    paths = live_paths(config.store_root, config.universe)
    job_dir = paths.jobs_dir(superalpha_id)
    if not job_dir.exists():
        return 0
    count = 0
    for path in job_dir.glob("*.json"):
        if path.name == "latest.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        updated = str(payload.get("updated_at_utc") or payload.get("created_at_utc") or "")
        if updated[:10] == str(run_date):
            count += 1
    return count
