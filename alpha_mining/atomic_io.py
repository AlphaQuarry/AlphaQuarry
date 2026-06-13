from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8", backup: bool = False) -> str:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as handle:
            handle.write(str(text))
            handle.flush()
            os.fsync(handle.fileno())
        _backup_existing(target, backup=backup)
        os.replace(tmp_path, target)
        return str(target.as_posix())
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def atomic_write_json(path: str | Path, payload: Any, *, backup: bool = False) -> str:
    return atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
        backup=backup,
    )


def atomic_write_dataframe_csv(path: str | Path, df: pd.DataFrame, *, index: bool = False, backup: bool = False) -> str:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            df.to_csv(handle, index=index)
            handle.flush()
            os.fsync(handle.fileno())
        _backup_existing(target, backup=backup)
        os.replace(tmp_path, target)
        return str(target.as_posix())
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def read_csv_with_backup(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    target = Path(path).resolve()
    try:
        return pd.read_csv(target, **kwargs)
    except Exception:
        backup = _backup_path(target)
        if backup.exists():
            return pd.read_csv(backup, **kwargs)
        raise


def read_text_with_backup(path: str | Path, *, encoding: str = "utf-8") -> str:
    target = Path(path).resolve()
    try:
        return target.read_text(encoding=encoding)
    except Exception:
        backup = _backup_path(target)
        if backup.exists():
            return backup.read_text(encoding=encoding)
        raise


def _backup_existing(target: Path, *, backup: bool) -> None:
    if not backup or not target.exists() or not target.is_file():
        return
    shutil.copy2(target, _backup_path(target))


def _backup_path(target: Path) -> Path:
    return target.with_suffix(target.suffix + ".bak")
