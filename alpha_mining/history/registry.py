from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..hashing import expression_hash


def compute_config_hash(config_snapshot: dict[str, Any]) -> str:
    canonical = _canonical_json(config_snapshot)
    # reuse normalized expression hash helper for stable sha1 string hashing
    return expression_hash(canonical)


def load_seen_hashes_for_config(base_dir: str | Path, config_hash: str) -> set[str]:
    idx_path = _config_index_path(base_dir, config_hash)
    if not idx_path.exists():
        return set()
    return {line.strip() for line in idx_path.read_text(encoding="utf-8").splitlines() if line.strip()}


def filter_new_expressions(
    expressions: list[str],
    seen_hashes: set[str],
) -> tuple[list[str], int]:
    out: list[str] = []
    local_seen: set[str] = set()
    skipped = 0
    for expr in expressions:
        h = expression_hash(expr)
        if h in seen_hashes or h in local_seen:
            skipped += 1
            continue
        out.append(expr)
        local_seen.add(h)
    return out, skipped


def save_run_registry(
    base_dir: str | Path,
    config_snapshot: dict[str, Any],
    expression_df: pd.DataFrame,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = Path(base_dir)
    runs_dir = base / "runs"
    configs_dir = runs_dir / "configs"
    expr_dir = runs_dir / "expressions"
    hash_dir = runs_dir / "expression_hashes"
    index_dir = base / "indexes"
    manifest_path = base / "manifest.jsonl"

    for p in [base, runs_dir, configs_dir, expr_dir, hash_dir, index_dir]:
        p.mkdir(parents=True, exist_ok=True)

    snapshot = to_serializable(config_snapshot)
    cfg_hash = compute_config_hash(snapshot)
    ts = datetime.now(timezone.utc)
    run_id = f"{ts.strftime('%Y%m%dT%H%M%SZ')}_{cfg_hash[:8]}"

    expr_df = _normalize_expression_df(expression_df)
    expr_hashes = sorted({expression_hash(e) for e in expr_df["expression"].tolist()})

    config_path = configs_dir / f"{run_id}.json"
    expr_path = expr_dir / f"{run_id}.csv"
    hash_path = hash_dir / f"{run_id}.txt"

    config_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    expr_df.to_csv(expr_path, index=False)
    hash_path.write_text("\n".join(expr_hashes), encoding="utf-8")

    idx_path = _config_index_path(base, cfg_hash)
    existing = load_seen_hashes_for_config(base, cfg_hash)
    merged = sorted(existing | set(expr_hashes))
    idx_path.write_text("\n".join(merged), encoding="utf-8")

    record = {
        "run_id": run_id,
        "created_at_utc": ts.isoformat(),
        "config_hash": cfg_hash,
        "config_path": str(config_path.as_posix()),
        "expression_path": str(expr_path.as_posix()),
        "expression_hash_path": str(hash_path.as_posix()),
        "expression_count": int(len(expr_df)),
        "expression_hash_count": int(len(expr_hashes)),
        "config_total_unique_expressions": int(len(merged)),
        "extra_meta": to_serializable(extra_meta or {}),
    }
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return record


def to_serializable(obj: Any) -> Any:
    if is_dataclass(obj):
        return to_serializable(asdict(obj))
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in sorted(obj.items(), key=lambda x: str(x[0]))}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(to_serializable(v) for v in obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _canonical_json(payload: dict[str, Any]) -> str:
    serializable = to_serializable(payload)
    return json.dumps(serializable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_expression_df(expression_df: pd.DataFrame) -> pd.DataFrame:
    if expression_df is None or len(expression_df) == 0:
        return pd.DataFrame(columns=["alpha_name", "expression", "source"])
    cols = [c for c in ["alpha_name", "expression", "source"] if c in expression_df.columns]
    if "expression" not in cols:
        raise ValueError("expression_df must include 'expression' column")
    out = expression_df[cols].copy()
    if "alpha_name" not in out.columns:
        out["alpha_name"] = [f"alpha_{i:04d}" for i in range(1, len(out) + 1)]
    if "source" not in out.columns:
        out["source"] = "unknown"
    out["expression"] = out["expression"].astype(str)
    return out[["alpha_name", "expression", "source"]]


def _config_index_path(base_dir: str | Path, config_hash: str) -> Path:
    return Path(base_dir) / "indexes" / f"{config_hash}.txt"
