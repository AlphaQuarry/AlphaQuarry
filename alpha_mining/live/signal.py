from __future__ import annotations

import gc
from typing import Any

import numpy as np
import pandas as pd

from alpha_mining.datasource.loader import load_panel_from_duckdb
from alpha_mining.engine import ExpressionEngine
from alpha_mining.panel_store import PanelStore
from alpha_mining.workflow.reproduce import load_required_fields_from_expression
from alpha_mining.workflow.superalpha import SUPERALPHA_FACTOR, parse_combo_expression

from .artifacts import live_paths, utc_now_iso, write_frame, write_json
from .calendar import previous_trading_window
from .lookback import estimate_snapshot_lookback


def build_live_superalpha_signal(
    *, config: Any, snapshot: dict[str, Any], signal_date: str, dry_run: bool = False
) -> dict[str, Any]:
    lookback = estimate_snapshot_lookback(snapshot, buffer=int(config.superalpha.lookback_buffer_days))
    window_days = max(int(config.superalpha.live_window_trade_days), int(lookback["max_lookback"]))
    if window_days > int(config.superalpha.max_expression_lookback_days):
        return {
            "status": "blocked",
            "blocking_reasons": ["lookback_exceeds_limit"],
            "lookback": lookback,
        }
    window_start, _dates = previous_trading_window(config=config, signal_date=signal_date, window_days=window_days)
    components = list(snapshot.get("components") or [])
    expressions = [str(c.get("expression") or "") for c in components]
    required_fields: list[str] = []
    for expr in expressions:
        for field in load_required_fields_from_expression(expr):
            if field not in required_fields:
                required_fields.append(field)
    raw = load_panel_from_duckdb(
        duckdb_path=str(config.duckdb_path),
        source_view=str(config.source_view),
        expressions=expressions,
        required_fields=required_fields,
        start_date=window_start,
        end_date=str(signal_date),
        date_col=config.data.date_col,
        code_col=config.data.code_col,
        base_fields=(),
        run_filters={"include_bj": bool(config.tradability.include_bj)},
        duckdb_settings=_duckdb_settings(config),
        sort=True,
    )
    if raw.empty:
        return {
            "status": "blocked",
            "blocking_reasons": ["raw_window_empty"],
            "window_start_date": window_start,
        }
    frames: list[pd.DataFrame] = []
    weighted_concat: pd.DataFrame | None = None
    weights = _resolve_weights(snapshot, components)
    for component, weight in zip(components, weights, strict=True):
        frame = _evaluate_component(config, raw, component)
        if frame.empty:
            return {
                "status": "blocked",
                "blocking_reasons": ["component_signal_empty"],
                "component": component.get("factor"),
            }
        if bool(snapshot.get("direction_adjustment", True)):
            frame["value"] = pd.to_numeric(frame["value"], errors="coerce") * float(
                component.get("direction_sign") or 1.0
            )
        if str(snapshot.get("component_normalization") or "cs_zscore") == "cs_zscore":
            frame["value"] = _cross_sectional_zscore(frame["value"], frame["date"])
        frame["weighted_value"] = pd.to_numeric(frame["value"], errors="coerce") * float(weight)
        part = frame[["date", "code", "weighted_value"]].dropna(subset=["weighted_value"])
        if str(snapshot.get("component_join") or "concat") == "inner":
            frames.append(part)
        else:
            weighted_concat = part if weighted_concat is None else pd.concat([weighted_concat, part], ignore_index=True)
            weighted_concat = weighted_concat.groupby(["date", "code"], as_index=False)["weighted_value"].sum(
                min_count=1
            )
    if str(snapshot.get("component_join") or "concat") == "inner" and frames:
        merged = frames[0].rename(columns={"weighted_value": "w0"})
        for idx, part in enumerate(frames[1:], start=1):
            merged = merged.merge(
                part.rename(columns={"weighted_value": f"w{idx}"}),
                on=["date", "code"],
                how="inner",
            )
        wcols = [c for c in merged.columns if c.startswith("w")]
        out = merged[["date", "code"]].copy()
        out[SUPERALPHA_FACTOR] = merged[wcols].sum(axis=1, min_count=1)
    else:
        if weighted_concat is None:
            weighted_concat = pd.DataFrame(columns=["date", "code", "weighted_value"])
        out = weighted_concat.rename(columns={"weighted_value": SUPERALPHA_FACTOR})
    if str(snapshot.get("final_normalization") or "cs_zscore") == "cs_zscore" and not out.empty:
        out[SUPERALPHA_FACTOR] = _cross_sectional_zscore(out[SUPERALPHA_FACTOR], out["date"])
    out = out[pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d") == str(signal_date)].copy()
    if out.empty:
        return {
            "status": "blocked",
            "blocking_reasons": ["signal_date_empty"],
            "window_start_date": window_start,
        }
    out = out.sort_values(["date", "code"]).reset_index(drop=True)
    if dry_run:
        return {
            "status": "ok",
            "signal": out,
            "window_start_date": window_start,
            "lookback": lookback,
        }
    paths = live_paths(config.store_root, config.universe)
    sid = str(snapshot.get("superalpha_id"))
    saved = write_frame(out, paths.signals_dir(sid) / str(signal_date), preferred="parquet")
    latest = {
        "schema_version": 1,
        "status": "ok",
        "superalpha_id": sid,
        "signal_date": str(signal_date),
        "window_start_date": window_start,
        "lookback": lookback,
        "row_count": int(len(out)),
        "signal_path": saved["path"],
        "created_at_utc": utc_now_iso(),
    }
    write_json(paths.signals_dir(sid) / "latest.json", latest)
    gc.collect()
    return {**latest, "signal_path": saved["path"]}


def _evaluate_component(config: Any, raw: pd.DataFrame, component: dict[str, Any]) -> pd.DataFrame:
    expr = str(component.get("expression") or "")
    selected = [config.data.date_col, config.data.code_col]
    for field in load_required_fields_from_expression(expr):
        if field in raw.columns and field not in selected:
            selected.append(field)
    frame = raw[selected].copy()
    store = PanelStore.from_long_frame(frame, date_col=config.data.date_col, code_col=config.data.code_col)
    values = ExpressionEngine(panel_store=store).eval(expr, use_cache=False)
    out = values.stack().rename("value").reset_index()
    out.columns = ["date", "code", "value"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


def _resolve_weights(snapshot: dict[str, Any], components: list[dict[str, Any]]) -> list[float]:
    try:
        parsed = parse_combo_expression(str(snapshot.get("combo_expression") or "1"), components)
        return [float(x) for x in parsed.weights]
    except Exception:
        raw = snapshot.get("component_weights") or [c.get("weight", 1.0) for c in components]
        return [float(x or 1.0) for x in raw]


def _cross_sectional_zscore(values: pd.Series, dates: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    mean = numeric.groupby(dates).transform("mean")
    std = numeric.groupby(dates).transform("std").replace(0.0, np.nan)
    return (numeric - mean) / std


def _duckdb_settings(config: Any) -> dict[str, Any]:
    return {
        "memory_limit": str(config.runtime.duckdb_memory_limit),
        "threads": str(config.runtime.duckdb_threads),
        "temp_directory": str(config.runtime.duckdb_temp_directory),
        "max_temp_directory_size": str(config.runtime.duckdb_max_temp_directory_size),
    }
