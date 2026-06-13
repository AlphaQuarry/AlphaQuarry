from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ..parquet_utils import write_parquet_compat


def build_snapshot_run_id(
    universe_name: str,
    source_view: str,
    start_date: str | None,
    end_date: str | None,
    fields: Sequence[str],
) -> str:
    payload = {
        "universe_name": str(universe_name),
        "source_view": str(source_view),
        "start_date": str(start_date or ""),
        "end_date": str(end_date or ""),
        "fields": [str(x) for x in fields],
    }
    digest = hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"snapshot_{ts}_{digest}"


def materialize_input_snapshot(
    raw_df: pd.DataFrame,
    snapshot_root: str | Path,
    universe_name: str,
    run_id: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if raw_df is None:
        raw_df = pd.DataFrame()

    root = Path(snapshot_root) / f"universe={_normalize_key(universe_name)}" / f"run_id={_normalize_key(run_id)}"
    root.mkdir(parents=True, exist_ok=True)

    snapshot_path = root / "input_snapshot.parquet"
    write_parquet_compat(raw_df, snapshot_path, index=False)

    meta_payload = {
        "run_id": str(run_id),
        "universe_name": str(universe_name),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_path": str(snapshot_path.as_posix()),
        "row_count": int(len(raw_df)),
        "columns": [str(c) for c in raw_df.columns],
        "dtypes": {str(c): str(raw_df[c].dtype) for c in raw_df.columns},
        "extra": dict(metadata or {}),
    }
    meta_path = root / "snapshot_meta.json"
    meta_path.write_text(
        json.dumps(meta_payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )

    return {
        "snapshot_path": str(snapshot_path.as_posix()),
        "meta_path": str(meta_path.as_posix()),
        "row_count": int(len(raw_df)),
        "run_id": str(run_id),
    }


def load_input_snapshot(snapshot_path: str | Path) -> pd.DataFrame:
    p = Path(snapshot_path)
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_parquet(p)


def _normalize_key(value: str) -> str:
    cleaned = str(value or "").strip().replace(" ", "_").replace(":", "").replace("/", "_")
    return cleaned or "default"
