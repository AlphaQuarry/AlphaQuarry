from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from alpha_mining.workflow.superalpha import SUPERALPHA_FACTOR

from .artifacts import live_paths, utc_now_iso, write_frame, write_json


def build_target_holdings(
    *,
    config: Any,
    superalpha_id: str,
    signal: pd.DataFrame,
    market: pd.DataFrame,
    signal_date: str,
    execute_date: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    if signal.empty or SUPERALPHA_FACTOR not in signal.columns:
        return _blocked(["signal_empty"])
    work = signal.copy()
    work[SUPERALPHA_FACTOR] = pd.to_numeric(work[SUPERALPHA_FACTOR], errors="coerce")
    work = work.dropna(subset=[SUPERALPHA_FACTOR])
    if work.empty:
        return _blocked(["signal_all_null"])
    if not market.empty:
        work = work.merge(market, on=["date", "code"], how="left")
    if bool(config.tradability.enabled):
        work["blocked"] = False
        work["block_reason"] = ""
        for col in config.tradability.critical_fields:
            if col not in work.columns:
                return _blocked([f"missing_tradability_field:{col}"])
        bad = (
            (pd.to_numeric(work["can_buy"], errors="coerce").fillna(0) <= 0)
            | (pd.to_numeric(work["is_st"], errors="coerce").fillna(0) > 0)
            | (pd.to_numeric(work["is_suspended"], errors="coerce").fillna(0) > 0)
            | (pd.to_numeric(work["is_limit_up_close"], errors="coerce").fillna(0) > 0)
        )
        work.loc[bad, "blocked"] = True
        work.loc[
            pd.to_numeric(work["can_buy"], errors="coerce").fillna(0) <= 0,
            "block_reason",
        ] = "cannot_buy"
        work.loc[pd.to_numeric(work["is_st"], errors="coerce").fillna(0) > 0, "block_reason"] = "is_st"
        work.loc[
            pd.to_numeric(work["is_suspended"], errors="coerce").fillna(0) > 0,
            "block_reason",
        ] = "is_suspended"
        buyable = work[~work["blocked"]].copy()
    else:
        work["blocked"] = False
        work["block_reason"] = ""
        buyable = work.copy()
    if buyable.empty:
        return _blocked(["zero_valid_names"])
    buyable = buyable.sort_values(SUPERALPHA_FACTOR, ascending=False, kind="mergesort")
    target = buyable.head(int(config.portfolio.target_count)).copy()
    fill_ratio = len(target) / max(1, int(config.portfolio.target_count))
    reasons: list[str] = []
    if len(target) < int(config.portfolio.min_target_count):
        reasons.append("too_few_buyable_names")
    if fill_ratio < float(config.portfolio.min_target_fill_ratio):
        reasons.append("target_fill_ratio_low")
    if reasons:
        return _blocked(reasons)
    weight_sum = 1.0 - float(config.portfolio.cash_buffer_ratio)
    equal = weight_sum / max(1, len(target))
    max_w = float(config.portfolio.max_single_name_weight)
    if equal > max_w + 1e-12:
        return _blocked(["single_name_weight_exceeds_limit"])
    target["target_weight"] = equal
    total_weight = float(target["target_weight"].sum())
    if not np.isfinite(total_weight) or total_weight <= 0 or total_weight > weight_sum + 1e-9:
        return _blocked(["weight_sum_abnormal"])
    target["schema_version"] = 1
    target["universe"] = str(config.universe)
    target["superalpha_id"] = str(superalpha_id)
    target["signal_date"] = str(signal_date)
    target["execute_date"] = str(execute_date)
    target["signal"] = target[SUPERALPHA_FACTOR]
    target["signal_rank"] = range(1, len(target) + 1)
    target["layer"] = (
        pd.qcut(
            target["signal_rank"],
            q=min(10, len(target)),
            labels=False,
            duplicates="drop",
        )
        + 1
    )
    target["target_notional"] = np.nan
    target["target_shares"] = np.nan
    target["created_at_utc"] = utc_now_iso()
    cols = [
        c
        for c in [
            "schema_version",
            "universe",
            "superalpha_id",
            "signal_date",
            "execute_date",
            "code",
            "signal",
            "signal_rank",
            "layer",
            "target_weight",
            "target_notional",
            "target_shares",
            "close",
            "circ_mv",
            "can_buy",
            "can_sell",
            "is_suspended",
            "is_st",
            "up_limit",
            "down_limit",
            "blocked",
            "block_reason",
            "created_at_utc",
        ]
        if c in target.columns
    ]
    out = target[cols].copy()
    if dry_run:
        return {"status": "ok", "holdings": out, "row_count": len(out)}
    paths = live_paths(config.store_root, config.universe)
    saved = write_frame(out, paths.holdings_dir(superalpha_id) / str(execute_date), preferred="parquet")
    latest = {
        "schema_version": 1,
        "status": "ok",
        "superalpha_id": str(superalpha_id),
        "signal_date": str(signal_date),
        "execute_date": str(execute_date),
        "row_count": int(len(out)),
        "artifact_path": saved["path"],
        "created_at_utc": utc_now_iso(),
    }
    write_json(paths.holdings_dir(superalpha_id) / "latest.json", latest)
    return {**latest, "holdings_path": saved["path"]}


def _blocked(reasons: list[str]) -> dict[str, Any]:
    return {"status": "blocked", "blocking_reasons": reasons}
