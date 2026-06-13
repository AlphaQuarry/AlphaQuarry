from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from alpha_mining.workflow.artifacts import (
    load_dataframe_from_stem,
    load_saved_dataframe,
)
from alpha_mining.workflow.superalpha import SUPERALPHA_FACTOR

from .artifacts import live_paths, utc_now_iso, write_json
from .signal import build_live_superalpha_signal


def compare_signal_cross_sections(
    live_signal: pd.DataFrame,
    reference_signal: pd.DataFrame,
    *,
    signal_date: str,
    top_n: int = 50,
) -> dict[str, Any]:
    live = _slice(live_signal, signal_date).rename(columns={SUPERALPHA_FACTOR: "live_signal"})
    ref = _slice(reference_signal, signal_date).rename(columns={SUPERALPHA_FACTOR: "reference_signal"})
    merged = pd.merge(
        live[["code", "live_signal"]],
        ref[["code", "reference_signal"]],
        on="code",
        how="outer",
    )
    merged["live_signal"] = pd.to_numeric(merged["live_signal"], errors="coerce")
    merged["reference_signal"] = pd.to_numeric(merged["reference_signal"], errors="coerce")
    valid = merged.dropna(subset=["live_signal", "reference_signal"])
    rank_corr = valid["live_signal"].corr(valid["reference_signal"], method="spearman") if len(valid) >= 2 else None
    live_missing = float(merged["live_signal"].isna().mean()) if len(merged) else 1.0
    ref_missing = float(merged["reference_signal"].isna().mean()) if len(merged) else 1.0
    k = max(1, min(int(top_n), len(valid))) if len(valid) else 0
    top_overlap = _overlap(valid, "live_signal", "reference_signal", k, ascending=False)
    bottom_overlap = _overlap(valid, "live_signal", "reference_signal", k, ascending=True)
    if rank_corr is not None and not pd.isna(rank_corr) and abs(float(rank_corr) - 1.0) < 1e-12:
        rank_corr = 1.0
    return {
        "status": "ok" if len(valid) else "empty",
        "signal_date": str(signal_date),
        "valid_sample_count": int(len(valid)),
        "rank_correlation": None if pd.isna(rank_corr) else float(rank_corr),
        "missing_ratio_live": live_missing,
        "missing_ratio_reference": ref_missing,
        "missing_ratio_delta": abs(live_missing - ref_missing),
        "top_overlap": top_overlap,
        "bottom_overlap": bottom_overlap,
    }


def evaluate_parity_thresholds(*, config: Any, metrics: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if metrics.get("status") != "ok":
        reasons.append("parity_reference_empty")
    if _lt(metrics.get("rank_correlation"), config.parity.min_rank_corr):
        reasons.append("rank_correlation_below_threshold")
    if _lt(metrics.get("top_overlap"), config.parity.min_top_overlap):
        reasons.append("top_overlap_below_threshold")
    if _lt(metrics.get("bottom_overlap"), config.parity.min_bottom_overlap):
        reasons.append("bottom_overlap_below_threshold")
    if _gt(metrics.get("missing_ratio_delta"), config.parity.max_missing_ratio_delta):
        reasons.append("missing_ratio_delta_above_threshold")
    if not reasons:
        return {"status": "ok", "reasons": []}
    return {
        "status": "blocked" if bool(config.parity.strict) else "warning",
        "reasons": reasons,
    }


def run_live_signal_parity(
    *,
    config: Any,
    snapshot: dict[str, Any],
    signal_date: str,
    reference_path: str | Path | None = None,
) -> dict[str, Any]:
    live_result = build_live_superalpha_signal(config=config, snapshot=snapshot, signal_date=signal_date, dry_run=True)
    if live_result.get("status") != "ok":
        return {
            "status": "blocked",
            "blocking_reasons": live_result.get("blocking_reasons", ["live_signal_failed"]),
        }
    ref = load_reference_signal(config=config, snapshot=snapshot, reference_path=reference_path)
    metrics = compare_signal_cross_sections(live_result["signal"], ref, signal_date=signal_date)
    decision = evaluate_parity_thresholds(config=config, metrics=metrics)
    payload = {
        "schema_version": 1,
        "superalpha_id": str(snapshot.get("superalpha_id")),
        "created_at_utc": utc_now_iso(),
        "metrics": metrics,
        **decision,
    }
    paths = live_paths(config.store_root, config.universe)
    if not getattr(config, "_dry_run", False):
        write_json(
            paths.live_root / "parity" / str(snapshot.get("superalpha_id")) / f"{signal_date}.json",
            payload,
        )
    return payload


def load_reference_signal(
    *, config: Any, snapshot: dict[str, Any], reference_path: str | Path | None = None
) -> pd.DataFrame:
    if reference_path:
        return load_saved_dataframe(reference_path)
    sid = str(snapshot.get("superalpha_id") or "")
    stem = Path(config.store_root) / str(config.universe) / "superalphas" / sid / "superalpha_values"
    return load_dataframe_from_stem(stem)


def _slice(frame: pd.DataFrame, signal_date: str) -> pd.DataFrame:
    out = frame.copy()
    if "date" in out.columns:
        out = out[pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d") == str(signal_date)]
    return out[["code", SUPERALPHA_FACTOR]].copy()


def _overlap(frame: pd.DataFrame, left: str, right: str, k: int, *, ascending: bool) -> float | None:
    if k <= 0:
        return None
    a = set(frame.sort_values(left, ascending=ascending, kind="mergesort").head(k)["code"].astype(str))
    b = set(frame.sort_values(right, ascending=ascending, kind="mergesort").head(k)["code"].astype(str))
    return float(len(a & b) / k)


def _lt(value: Any, threshold: float) -> bool:
    try:
        return float(value) < float(threshold)
    except Exception:
        return True


def _gt(value: Any, threshold: float) -> bool:
    try:
        return float(value) > float(threshold)
    except Exception:
        return True
