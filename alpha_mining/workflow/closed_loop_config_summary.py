from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any


def config_summary(config: Any) -> dict[str, Any]:
    try:
        payload = asdict(config)
    except Exception:
        payload = dict(getattr(config, "__dict__", {}) or {})
    payload.pop("benchmark_returns", None)
    return payload


def closed_loop_config_hash(config: Any) -> str:
    payload = config_summary(config)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
