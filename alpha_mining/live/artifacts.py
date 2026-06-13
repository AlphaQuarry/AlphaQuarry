from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from alpha_mining.atomic_io import atomic_write_json
from alpha_mining.workflow.artifacts import save_dataframe_artifact


@dataclass(frozen=True)
class LivePaths:
    store_root: Path
    universe: str

    @property
    def universe_root(self) -> Path:
        return self.store_root / self.universe

    @property
    def live_root(self) -> Path:
        return self.universe_root / "live"

    @property
    def registry_path(self) -> Path:
        return self.live_root / "active_superalphas.json"

    @property
    def data_status_dir(self) -> Path:
        return self.live_root / "data_status"

    def signals_dir(self, superalpha_id: str) -> Path:
        return self.live_root / "signals" / str(superalpha_id)

    def holdings_dir(self, superalpha_id: str) -> Path:
        return self.live_root / "holdings" / str(superalpha_id)

    def jobs_dir(self, superalpha_id: str) -> Path:
        return self.live_root / "jobs" / str(superalpha_id)


def live_paths(store_root: str | Path, universe: str) -> LivePaths:
    return LivePaths(store_root=Path(store_root), universe=str(universe))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, payload: Any) -> str:
    return atomic_write_json(path, _jsonable(payload), backup=True)


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_frame(df: pd.DataFrame, stem: str | Path, *, preferred: str = "parquet") -> dict[str, str]:
    return save_dataframe_artifact(df, stem, index=False, preferred=preferred)


def write_latest_index(*, config: Any, sa_statuses: list[dict[str, Any]]) -> dict[str, Any]:
    paths = live_paths(config.store_root, config.universe)
    payload = {
        "schema_version": 1,
        "universe": str(config.universe),
        "status": "ok",
        "updated_at_utc": utc_now_iso(),
        "superalphas": _jsonable(sa_statuses),
    }
    write_json(paths.live_root / "latest.json", payload)
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
