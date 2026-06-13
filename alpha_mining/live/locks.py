from __future__ import annotations

import json
import os
import socket
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .artifacts import live_paths


class LiveLockError(RuntimeError):
    pass


@contextmanager
def live_lock(*, config: Any, name: str, run_id: str) -> Iterator[dict[str, Any]]:
    paths = live_paths(config.store_root, config.universe)
    lock_dir = paths.live_root / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / ("global.lock" if str(name) == "global" else f"{name}.lock")
    acquired = acquire_live_lock(config=config, lock_path=lock_path, run_id=run_id)
    try:
        yield acquired
    finally:
        release_live_lock(lock_path=lock_path, run_id=run_id)


def acquire_live_lock(*, config: Any, lock_path: Path, run_id: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    stale_recovered = False
    if lock_path.exists():
        existing = _read_lock(lock_path)
        created = _parse_time(existing.get("created_at_utc"))
        age = (now - created).total_seconds() if created else 0.0
        if created and age > int(config.runtime.stale_lock_seconds):
            stale_recovered = True
            lock_path.unlink(missing_ok=True)
        else:
            raise LiveLockError(f"live lock is already held: {lock_path}")
    payload = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "created_at_utc": now.isoformat(),
        "run_id": str(run_id),
        "command": " ".join(sys.argv),
        "stale_recovered": stale_recovered,
    }
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise LiveLockError(f"live lock is already held: {lock_path}") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload


def release_live_lock(*, lock_path: Path, run_id: str) -> None:
    payload = _read_lock(lock_path)
    if not payload or str(payload.get("run_id")) == str(run_id):
        lock_path.unlink(missing_ok=True)


def _read_lock(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_time(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None
