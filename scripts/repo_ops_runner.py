from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class OpsJob:
    name: str
    command: list[str]
    interval_seconds: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight repo-internal ops runner")
    parser.add_argument("--config", default="", help="Datasource config yaml path")
    parser.add_argument("--jobs", default="update_lake,rebuild_duckdb_catalog,monthly_compaction")
    parser.add_argument("--once", action="store_true", help="Run selected jobs once and exit")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--update-interval-seconds", type=int, default=86400)
    parser.add_argument("--catalog-interval-seconds", type=int, default=86400)
    parser.add_argument("--compaction-interval-seconds", type=int, default=2592000)
    parser.add_argument("--update-args", default="", help="Extra args passed to update_tushare_lake.py")
    parser.add_argument(
        "--catalog-args",
        default="",
        help="Extra args passed to build_duckdb_catalog.py",
    )
    parser.add_argument(
        "--compaction-args",
        default="",
        help="Extra args passed to compact_parquet_lake.py",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    args = parser.parse_args()

    jobs = build_jobs(
        config=str(args.config or ""),
        selected_jobs=_parse_csv(str(args.jobs or "")),
        update_interval_seconds=int(args.update_interval_seconds),
        catalog_interval_seconds=int(args.catalog_interval_seconds),
        compaction_interval_seconds=int(args.compaction_interval_seconds),
        update_args=_split_extra_args(str(args.update_args or "")),
        catalog_args=_split_extra_args(str(args.catalog_args or "")),
        compaction_args=_split_extra_args(str(args.compaction_args or "")),
    )
    run_jobs(
        jobs=jobs,
        once=bool(args.once),
        poll_seconds=max(1, int(args.poll_seconds)),
        dry_run=bool(args.dry_run),
    )


def build_jobs(
    config: str,
    selected_jobs: Sequence[str],
    update_interval_seconds: int = 86400,
    catalog_interval_seconds: int = 86400,
    compaction_interval_seconds: int = 2592000,
    update_args: Sequence[str] = (),
    catalog_args: Sequence[str] = (),
    compaction_args: Sequence[str] = (),
) -> list[OpsJob]:
    selected = {str(x).strip().lower() for x in selected_jobs if str(x).strip()}
    config_args = ["--config", str(config)] if str(config or "").strip() else []
    registry = {
        "update_lake": OpsJob(
            name="update_lake",
            command=[
                sys.executable,
                str(ROOT / "scripts" / "update_tushare_lake.py"),
                *config_args,
                *list(update_args),
            ],
            interval_seconds=max(1, int(update_interval_seconds)),
        ),
        "rebuild_duckdb_catalog": OpsJob(
            name="rebuild_duckdb_catalog",
            command=[
                sys.executable,
                str(ROOT / "scripts" / "build_duckdb_catalog.py"),
                *config_args,
                *list(catalog_args),
            ],
            interval_seconds=max(1, int(catalog_interval_seconds)),
        ),
        "monthly_compaction": OpsJob(
            name="monthly_compaction",
            command=[
                sys.executable,
                str(ROOT / "scripts" / "compact_parquet_lake.py"),
                *config_args,
                *list(compaction_args),
            ],
            interval_seconds=max(1, int(compaction_interval_seconds)),
        ),
    }
    unknown = sorted(selected.difference(registry.keys()))
    if unknown:
        raise ValueError(f"Unknown jobs: {unknown}. Allowed jobs: {sorted(registry.keys())}")
    return [registry[name] for name in registry.keys() if name in selected]


def run_jobs(jobs: Sequence[OpsJob], once: bool, poll_seconds: int, dry_run: bool = False) -> None:
    if not jobs:
        print("[ops] no selected jobs")
        return

    next_run = {job.name: datetime.now() for job in jobs}
    while True:
        now = datetime.now()
        for job in jobs:
            if now < next_run[job.name]:
                continue
            _run_job(job=job, dry_run=bool(dry_run))
            next_run[job.name] = datetime.now() + timedelta(seconds=max(1, int(job.interval_seconds)))
        if bool(once):
            return
        time.sleep(max(1, int(poll_seconds)))


def _run_job(job: OpsJob, dry_run: bool = False) -> None:
    print(f"[ops] job={job.name} command={_format_command(job.command)}")
    if bool(dry_run):
        return
    completed = subprocess.run(job.command, cwd=str(ROOT), check=False)
    if int(completed.returncode) != 0:
        print(f"[ops][warn] job={job.name} exit_code={completed.returncode}")


def _format_command(command: Sequence[str]) -> str:
    return " ".join(_quote_arg(str(x)) for x in command)


def _quote_arg(arg: str) -> str:
    if not arg or any(ch.isspace() for ch in arg):
        return '"' + arg.replace('"', '\\"') + '"'
    return arg


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def _split_extra_args(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    return text.split()


if __name__ == "__main__":
    main()
