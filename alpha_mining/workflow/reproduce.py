from __future__ import annotations

import gzip
import json
import pickle
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import AlphaSimulationConfig
from ..engine import ExpressionEngine
from ..panel_store import PanelStore
from ..simulation import apply_simulation_settings
from ..simulation.neutralization import (
    neutralization_group_field,
    normalize_neutralization_mode,
)
from ..validators import expression_stats
from .artifacts import load_saved_dataframe, save_dataframe_artifact
from .lifecycle import mark_reproduced
from .universe_store import (
    DEFAULT_UNIVERSE_BASE_DIR,
    load_universe_alpha_values,
    load_universe_expression_registry,
    load_universe_input_manifest,
)


def load_required_fields_from_expression(expression: str) -> list[str]:
    stats = expression_stats(expression)
    return [str(x) for x in stats.unique_fields]


def reproduce_alpha_by_expression(
    expression: str,
    base_dir: str | Path = DEFAULT_UNIVERSE_BASE_DIR,
    universe_name: str = "default",
    raw_df: pd.DataFrame | None = None,
    alpha_name: str = "alpha_reproduced",
    simulation_config: AlphaSimulationConfig | dict[str, Any] | None = None,
    manifest_id: str | None = None,
    save_path_stem: str | Path | None = None,
    duckdb_settings_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = load_universe_input_manifest(
        base_dir=base_dir,
        universe_name=universe_name,
        manifest_id=manifest_id,
    )
    payload = manifest.get("payload", manifest) if isinstance(manifest, dict) else {}
    date_col = str(payload.get("date_col", "date"))
    code_col = str(payload.get("code_col", "code"))
    group_fields = [str(x) for x in payload.get("group_fields", []) if str(x)]
    vector_fields = [str(x) for x in payload.get("vector_fields", []) if str(x)]
    required_fields = load_required_fields_from_expression(expression)

    source_meta = {
        "mode": "provided_raw_df",
        "strict_reproducibility": True,
        "warning": "",
    }
    if raw_df is None:
        raw_df, source_meta = _load_raw_df_from_manifest_payload(
            payload=payload,
            date_col=date_col,
            code_col=code_col,
            required_fields=required_fields,
            group_fields=group_fields,
            duckdb_settings_override=duckdb_settings_override,
        )
    if raw_df is None or raw_df.empty:
        raise ValueError("raw_df is empty and no valid manifest source could be loaded")

    sim_cfg = _normalize_simulation_config(simulation_config)
    selected_cols, inferred_vector_bases = _select_required_columns(
        df_columns=list(raw_df.columns),
        required_fields=required_fields,
        date_col=date_col,
        code_col=code_col,
    )
    for field in [sim_cfg.universe]:
        if field and field in raw_df.columns and field not in selected_cols:
            selected_cols.append(str(field))
    neutral_group = neutralization_group_field(sim_cfg.neutralization)
    if neutral_group:
        if neutral_group not in raw_df.columns:
            raise ValueError(
                f"neutralization={sim_cfg.neutralization} requires group field '{neutral_group}' in raw_df"
            )
        if neutral_group not in selected_cols:
            selected_cols.append(neutral_group)
        if neutral_group not in group_fields:
            group_fields.append(neutral_group)
    frame = raw_df[selected_cols].copy()
    panel_store = PanelStore.from_long_frame(
        frame,
        date_col=date_col,
        code_col=code_col,
        group_fields=[g for g in group_fields if g in frame.columns],
        vector_fields=list(dict.fromkeys(vector_fields + inferred_vector_bases)),
    )
    engine = ExpressionEngine(panel_store=panel_store)
    raw_panel = engine.eval(expression, use_cache=False)
    group_panel = panel_store.get_group_like(neutral_group) if neutral_group else None
    adjusted = apply_simulation_settings(raw_panel, config=sim_cfg, group_panel=group_panel)
    stacked = _stack_panel_compat(adjusted)
    out = stacked.rename(alpha_name).reset_index()
    out.columns = [date_col, code_col, alpha_name]
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.sort_values([code_col, date_col], kind="mergesort").reset_index(drop=True)

    saved = None
    if save_path_stem is not None:
        saved = save_dataframe_artifact(out, save_path_stem, index=False)

    return {
        "expression": str(expression),
        "alpha_name": str(alpha_name),
        "required_fields": required_fields,
        "output_df": out,
        "saved": saved,
        "manifest_id": str(manifest.get("manifest_id", "")) if isinstance(manifest, dict) else "",
        "manifest_schema_version": str(payload.get("manifest_schema_version", "")) if isinstance(payload, dict) else "",
        "reproduce_source_mode": str(source_meta.get("mode", "provided_raw_df")),
        "strict_reproducibility": bool(source_meta.get("strict_reproducibility", True)),
        "reproduce_warning": str(source_meta.get("warning", "")),
    }


def reproduce_alpha_by_name(
    alpha_name: str,
    base_dir: str | Path = DEFAULT_UNIVERSE_BASE_DIR,
    universe_name: str = "default",
    raw_df: pd.DataFrame | None = None,
    manifest_id: str | None = None,
    compare_with_saved: bool = True,
    save_path_stem: str | Path | None = None,
    mark_lifecycle: bool = True,
    duckdb_settings_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    name = str(alpha_name or "").strip()
    if not name:
        raise ValueError("alpha_name must not be empty")

    registry = load_universe_expression_registry(base_dir=base_dir, universe_name=universe_name)
    if registry.empty or "alpha_name" not in registry.columns:
        raise ValueError("Expression registry is empty")
    row_df = registry[registry["alpha_name"].astype(str) == name]
    if row_df.empty:
        raise ValueError(f"alpha_name not found in expression registry: {name}")
    row = row_df.iloc[-1]
    expression = str(row.get("expression", "")).strip()
    if not expression:
        raise ValueError(f"Empty expression for {name}")

    simulation_config = _load_simulation_config_from_registry_row(row)
    manifest_from_registry = str(row.get("input_manifest_id", "") or "").strip()
    result = reproduce_alpha_by_expression(
        expression=expression,
        base_dir=base_dir,
        universe_name=universe_name,
        raw_df=raw_df,
        alpha_name=name,
        simulation_config=simulation_config,
        manifest_id=(manifest_id or manifest_from_registry or None),
        save_path_stem=save_path_stem,
        duckdb_settings_override=duckdb_settings_override,
    )

    compare_summary: dict[str, Any] | None = None
    if compare_with_saved:
        compare_summary = _compare_with_existing_alpha(
            reproduced_df=result["output_df"],
            alpha_name=name,
            base_dir=base_dir,
            universe_name=universe_name,
        )
    result["compare_summary"] = compare_summary

    if mark_lifecycle:
        mark_reproduced(
            alpha_names=[name],
            base_dir=str(base_dir),
            universe_name=universe_name,
        )
    return result


def _load_raw_df_from_manifest_payload(
    payload: dict[str, Any],
    date_col: str,
    code_col: str,
    required_fields: list[str] | None = None,
    group_fields: list[str] | None = None,
    duckdb_settings_override: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _required = set(required_fields or [])

    def _has_all_required(df: pd.DataFrame) -> bool:
        if not _required:
            return True
        cols = set(str(c) for c in df.columns)
        return _required.issubset(cols)

    snapshot_path = str(payload.get("snapshot_path", "") or "").strip()
    if snapshot_path:
        p = Path(snapshot_path)
        if p.exists():
            try:
                out = load_saved_dataframe(p)
                out = _ensure_identifier_alias_columns(out, date_col=date_col, code_col=code_col)
                if _has_all_required(out):
                    return out, {
                        "mode": "snapshot",
                        "strict_reproducibility": True,
                        "warning": "",
                    }
                print(f"[reproduce][warn] snapshot missing required fields, trying fallback: {snapshot_path}")
            except Exception:
                pass
            suffix = p.suffix.lower()
            if suffix == ".pkl":
                try:
                    with p.open("rb") as f:
                        out = pickle.load(f)
                except Exception:
                    with gzip.open(p, "rb") as f:
                        out = pickle.load(f)
                out = _ensure_identifier_alias_columns(out, date_col=date_col, code_col=code_col)
                if _has_all_required(out):
                    return out, {
                        "mode": "snapshot",
                        "strict_reproducibility": True,
                        "warning": "",
                    }
                print(f"[reproduce][warn] snapshot missing required fields, trying fallback: {snapshot_path}")
            if suffix in {".parquet"}:
                out = pd.read_parquet(p)
                out = _ensure_identifier_alias_columns(out, date_col=date_col, code_col=code_col)
                if _has_all_required(out):
                    return out, {
                        "mode": "snapshot",
                        "strict_reproducibility": True,
                        "warning": "",
                    }
                print(f"[reproduce][warn] snapshot missing required fields, trying fallback: {snapshot_path}")
            if suffix in {".csv"}:
                out = pd.read_csv(p)
                out = _ensure_identifier_alias_columns(out, date_col=date_col, code_col=code_col)
                if _has_all_required(out):
                    return out, {
                        "mode": "snapshot",
                        "strict_reproducibility": True,
                        "warning": "",
                    }
                print(f"[reproduce][warn] snapshot missing required fields, trying fallback: {snapshot_path}")
        else:
            print(f"[reproduce][warn] snapshot_path not found, will try fallback source: {snapshot_path}")

    source_path = str(payload.get("source_path", "") or "").strip()
    if source_path:
        p = Path(source_path)
        if p.exists():
            try:
                out = load_saved_dataframe(p)
                out = _ensure_identifier_alias_columns(out, date_col=date_col, code_col=code_col)
                if _has_all_required(out):
                    return out, {
                        "mode": "source_path",
                        "strict_reproducibility": True,
                        "warning": "",
                    }
            except Exception:
                pass
            suffix = p.suffix.lower()
            if suffix == ".pkl":
                try:
                    with p.open("rb") as f:
                        out = pickle.load(f)
                except Exception:
                    with gzip.open(p, "rb") as f:
                        out = pickle.load(f)
                out = _ensure_identifier_alias_columns(out, date_col=date_col, code_col=code_col)
                if _has_all_required(out):
                    return out, {
                        "mode": "source_path",
                        "strict_reproducibility": True,
                        "warning": "",
                    }
            if suffix in {".parquet"}:
                out = pd.read_parquet(p)
                out = _ensure_identifier_alias_columns(out, date_col=date_col, code_col=code_col)
                if _has_all_required(out):
                    return out, {
                        "mode": "source_path",
                        "strict_reproducibility": True,
                        "warning": "",
                    }
            if suffix in {".csv"}:
                out = pd.read_csv(p)
                out = _ensure_identifier_alias_columns(out, date_col=date_col, code_col=code_col)
                if _has_all_required(out):
                    return out, {
                        "mode": "source_path",
                        "strict_reproducibility": True,
                        "warning": "",
                    }

    source_backend = str(payload.get("source_backend", "") or "").strip().lower()
    duckdb_path = str(payload.get("duckdb_path", "") or "").strip()
    source_view = str(payload.get("source_view", "") or "").strip()
    if duckdb_path and source_view and source_backend.startswith("duckdb"):
        from ..datasource.loader import load_panel_from_duckdb

        date_range = payload.get("date_range", {}) if isinstance(payload.get("date_range", {}), dict) else {}
        start_date = str(date_range.get("start", "") or "").strip() or None
        end_date = str(date_range.get("end", "") or "").strip() or None
        base_fields = [str(x) for x in payload.get("base_frame_cols", []) if str(x)]
        if not base_fields:
            base_fields = ["pct_chg", "circ_mv"]
        run_filters = payload.get("run_filters", {})
        if not isinstance(run_filters, dict):
            run_filters = {}
        duckdb_settings: dict[str, Any] = {}
        for key in (
            "duckdb_temp_directory",
            "duckdb_max_temp_directory_size",
            "duckdb_memory_limit",
            "duckdb_threads",
        ):
            val = str(payload.get(key, "") or "").strip()
            if val:
                settings_key = key.replace("duckdb_", "")
                duckdb_settings[settings_key] = val
        if duckdb_settings_override:
            duckdb_settings.update(duckdb_settings_override)
        warning_msg = "snapshot unavailable; fallback to duckdb source view may reduce strict reproducibility"
        print(f"[reproduce][warn] {warning_msg}")
        out = load_panel_from_duckdb(
            duckdb_path=duckdb_path,
            source_view=source_view,
            required_fields=list(required_fields or []),
            start_date=start_date,
            end_date=end_date,
            date_col=date_col,
            code_col=code_col,
            base_fields=base_fields,
            group_fields=list(group_fields or []),
            run_filters=run_filters,
            duckdb_settings=duckdb_settings or None,
            sort=False,
        )
        return (
            _ensure_identifier_alias_columns(out, date_col=date_col, code_col=code_col),
            {
                "mode": "duckdb_fallback",
                "strict_reproducibility": False,
                "warning": warning_msg,
            },
        )

    raise ValueError(
        "Manifest could not load source data. "
        "Tried snapshot_path, source_path, and duckdb fallback. Please pass raw_df explicitly."
    )


def _ensure_identifier_alias_columns(
    raw_df: pd.DataFrame,
    date_col: str,
    code_col: str,
) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return raw_df
    work = raw_df
    changed = False
    cols = set(str(c) for c in raw_df.columns)

    date_name = str(date_col)
    code_name = str(code_col)
    date_source = _resolve_col_alias(cols, date_name, ("date", "trade_date"))
    code_source = _resolve_col_alias(cols, code_name, ("code", "znz_code"))
    if date_source and date_name not in work.columns:
        work = work.copy(deep=False)
        changed = True
        work[date_name] = work[date_source]
    if code_source and code_name not in work.columns:
        if not changed:
            work = work.copy(deep=False)
            changed = True
        work[code_name] = work[code_source]
    return work


def _resolve_col_alias(columns: set[str], preferred: str, aliases: tuple[str, ...]) -> str | None:
    pref = str(preferred or "").strip()
    if pref and pref in columns:
        return pref
    for name in aliases:
        if str(name) in columns:
            return str(name)
    return None


def _normalize_simulation_config(
    cfg: AlphaSimulationConfig | dict[str, Any] | None,
) -> AlphaSimulationConfig:
    if isinstance(cfg, AlphaSimulationConfig):
        return cfg
    if is_dataclass(cfg):
        cfg = asdict(cfg)
    if isinstance(cfg, dict):
        return AlphaSimulationConfig(
            delay=int(cfg.get("delay", 1)),
            decay=int(cfg.get("decay", 0)),
            neutralization=normalize_neutralization_mode(str(cfg.get("neutralization", "NONE"))),
            truncation=cfg.get("truncation", None),
            pasteurization=bool(cfg.get("pasteurization", True)),
            universe=str(cfg.get("universe", "")) or None,
        )
    return AlphaSimulationConfig()


def _load_simulation_config_from_registry_row(row: pd.Series) -> AlphaSimulationConfig:
    raw = str(row.get("simulation_config_json", "") or "").strip()
    if not raw:
        return AlphaSimulationConfig()
    try:
        payload = json.loads(raw)
    except Exception:
        return AlphaSimulationConfig()
    return _normalize_simulation_config(payload)


def _select_required_columns(
    df_columns: list[str],
    required_fields: list[str],
    date_col: str,
    code_col: str,
) -> tuple[list[str], list[str]]:
    selected = [date_col, code_col]
    inferred_vector_bases: list[str] = []
    col_set = set(df_columns)
    for field in required_fields:
        if field in col_set:
            if field not in selected:
                selected.append(field)
            continue
        exploded = sorted([c for c in df_columns if c.startswith(f"{field}__")])
        if exploded:
            inferred_vector_bases.append(field)
            for c in exploded:
                if c not in selected:
                    selected.append(c)
            continue
        raise ValueError(f"Required field '{field}' not found in raw data columns")
    return selected, inferred_vector_bases


def _stack_panel_compat(panel: pd.DataFrame) -> pd.Series:
    try:
        return panel.stack(dropna=False)
    except ValueError as exc:
        if "dropna must be unspecified" in str(exc):
            return panel.stack(future_stack=True)
        raise
    except TypeError:
        return panel.stack()


def _compare_with_existing_alpha(
    reproduced_df: pd.DataFrame,
    alpha_name: str,
    base_dir: str | Path,
    universe_name: str,
) -> dict[str, Any]:
    try:
        existing = load_universe_alpha_values(
            alpha_name=alpha_name,
            base_dir=base_dir,
            universe_name=universe_name,
        )
    except Exception:
        return {"matched": False, "reason": "existing_alpha_not_found"}

    if existing.empty or reproduced_df.empty:
        return {"matched": False, "reason": "empty_frame"}

    date_col = "date" if "date" in reproduced_df.columns else reproduced_df.columns[0]
    code_col = "code" if "code" in reproduced_df.columns else reproduced_df.columns[1]
    merge_cols = [date_col, code_col]
    left = reproduced_df[merge_cols + [alpha_name]].copy()
    right = existing[merge_cols + [alpha_name]].copy()
    merged = pd.merge(left, right, on=merge_cols, how="inner", suffixes=("_new", "_old"))
    if merged.empty:
        return {"matched": False, "reason": "no_overlap"}

    new_v = pd.to_numeric(merged[f"{alpha_name}_new"], errors="coerce")
    old_v = pd.to_numeric(merged[f"{alpha_name}_old"], errors="coerce")
    diff = new_v - old_v
    mae = float(diff.abs().mean())
    max_abs = float(diff.abs().max())
    corr = float(new_v.corr(old_v)) if new_v.notna().any() and old_v.notna().any() else np.nan
    return {
        "matched": bool(mae <= 1e-8 or max_abs <= 1e-6),
        "n_overlap": int(len(merged)),
        "mae": mae,
        "max_abs_diff": max_abs,
        "corr": corr,
    }
