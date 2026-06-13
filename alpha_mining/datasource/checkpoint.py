from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import LakePathSettings


def build_checkpoint_signature(task_name: str, payload: dict[str, Any]) -> str:
    body = {
        "task_name": str(task_name or "").strip(),
        "payload": _to_serializable(payload),
    }
    raw = json.dumps(body, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def get_checkpoint_path(
    paths: LakePathSettings,
    task_name: str,
    signature: str,
) -> Path:
    task_key = _normalize_key(task_name)
    sig_key = _normalize_key(signature)
    return paths.meta_path / "checkpoints" / f"{task_key}_{sig_key}.json"


def load_checkpoint(
    paths: LakePathSettings,
    task_name: str,
    signature: str,
) -> dict[str, Any]:
    path = get_checkpoint_path(paths=paths, task_name=task_name, signature=signature)
    if not path.exists():
        return {
            "task_name": str(task_name),
            "signature": str(signature),
            "status": "new",
            "completed_trade_dates": [],
            "updated_at_utc": "",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {
            "task_name": str(task_name),
            "signature": str(signature),
            "status": "new",
            "completed_trade_dates": [],
            "updated_at_utc": "",
        }
    payload.setdefault("task_name", str(task_name))
    payload.setdefault("signature", str(signature))
    payload.setdefault("status", "new")
    payload.setdefault("completed_trade_dates", [])
    payload.setdefault("updated_at_utc", "")
    return payload


def save_checkpoint(
    paths: LakePathSettings,
    task_name: str,
    signature: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    out = dict(payload or {})
    out["task_name"] = str(task_name)
    out["signature"] = str(signature)
    out["updated_at_utc"] = _utc_now_iso()

    path = get_checkpoint_path(paths=paths, task_name=task_name, signature=signature)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_serializable(out), ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return out


def reset_checkpoint(
    paths: LakePathSettings,
    task_name: str,
    signature: str,
) -> bool:
    path = get_checkpoint_path(paths=paths, task_name=task_name, signature=signature)
    if not path.exists():
        return False
    path.unlink()
    return True


def _normalize_key(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "default"
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "default"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(x) for x in value]
    if isinstance(value, tuple):
        return [_to_serializable(x) for x in value]
    return value
