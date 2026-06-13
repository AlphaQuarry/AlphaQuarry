from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from alpha_mining.live.config import LiveConfig
from alpha_mining.live.locks import LiveLockError, live_lock


def test_per_sa_lock_blocks_concurrent_holder(tmp_path: Path) -> None:
    cfg = LiveConfig(universe="u1", store_root=tmp_path)

    with live_lock(config=cfg, name="sa1", run_id="run1"):
        with pytest.raises(LiveLockError):
            with live_lock(config=cfg, name="sa1", run_id="run2"):
                pass


def test_stale_lock_is_recovered_and_removed_on_exit(tmp_path: Path) -> None:
    cfg = LiveConfig(universe="u1", store_root=tmp_path)
    cfg.runtime.stale_lock_seconds = 1
    lock_path = tmp_path / "u1" / "live" / "locks" / "global.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    old = datetime.now(timezone.utc) - timedelta(seconds=60)
    lock_path.write_text(
        json.dumps({"created_at_utc": old.isoformat(), "run_id": "old"}),
        encoding="utf-8",
    )

    with live_lock(config=cfg, name="global", run_id="new") as acquired:
        assert acquired["stale_recovered"] is True
        assert json.loads(lock_path.read_text(encoding="utf-8"))["run_id"] == "new"

    assert not lock_path.exists()
