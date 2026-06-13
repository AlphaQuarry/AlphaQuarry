from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from ..hashing import expression_hash
from ..history.registry import to_serializable
from ..mining.expression_canonicalizer import canonicalize_expression
from ..parser import parse_expression
from ..simulation.neutralization import normalize_neutralization_mode
from ..validators import expression_stats
from ..atomic_io import (
    atomic_write_dataframe_csv,
    atomic_write_json,
    atomic_write_text,
    read_csv_with_backup,
)
from .artifacts import (
    load_dataframe_from_stem,
    load_saved_dataframe,
    save_dataframe_artifact,
)


DEFAULT_UNIVERSE_BASE_DIR = "data/alpha_universe_store"
_ALPHA_NAME_RE = re.compile(r"^alpha(?P<idx>\d+)$")


def normalize_universe_name(name: str) -> str:
    value = str(name or "").strip().lower()
    if not value:
        raise ValueError("universe_name must not be empty")
    sanitized = re.sub(r"[^a-z0-9._-]+", "_", value).strip("._-")
    if not sanitized:
        raise ValueError(f"Invalid universe_name: {name!r}")
    return sanitized


def format_alpha_name(alpha_index: int, width: int = 5) -> str:
    idx = int(alpha_index)
    if idx <= 0:
        raise ValueError("alpha_index must be positive")
    return f"alpha{idx:0{int(width)}d}"


def parse_alpha_index(alpha_name: str) -> int | None:
    m = _ALPHA_NAME_RE.match(str(alpha_name or "").strip())
    if not m:
        return None
    return int(m.group("idx"))


def get_universe_paths(
    base_dir: str | Path = DEFAULT_UNIVERSE_BASE_DIR,
    universe_name: str = "default",
) -> dict[str, Path]:
    universe_key = normalize_universe_name(universe_name)
    root = Path(base_dir) / universe_key
    return {
        "base_dir": Path(base_dir),
        "universe_key": Path(universe_key),
        "root": root,
        "base_frame_dir": root / "base",
        "catalog_dir": root / "catalog",
        "input_manifest_dir": root / "catalog" / "input_manifests",
        "alphas_dir": root / "alphas",
        "analysis_dir": root / "analysis",
        "feedback_dir": root / "feedback",
        "meta_path": root / "universe_meta.json",
        "base_frame_stem": root / "base" / "base_frame",
        "expression_csv": root / "catalog" / "expressions.csv",
        "expression_txt": root / "catalog" / "expressions.txt",
        "lifecycle_csv": root / "catalog" / "alpha_lifecycle.csv",
        "analysis_registry_csv": root / "analysis" / "analysis_registry.csv",
        "factor_metrics_registry_csv": root / "analysis" / "factor_metrics_registry.csv",
        "feedback_scoreboard_csv": root / "feedback" / "expression_scoreboard.csv",
    }


def init_universe_workspace(
    base_dir: str | Path = DEFAULT_UNIVERSE_BASE_DIR,
    universe_name: str = "default",
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, str]:
    paths = get_universe_paths(base_dir=base_dir, universe_name=universe_name)
    for key in [
        "root",
        "base_frame_dir",
        "catalog_dir",
        "input_manifest_dir",
        "alphas_dir",
        "analysis_dir",
        "feedback_dir",
    ]:
        paths[key].mkdir(parents=True, exist_ok=True)

    meta_path = paths["meta_path"]
    existing: dict[str, Any] = {}
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    now = _utc_now().isoformat()
    meta = {
        "universe_name": str(universe_name),
        "universe_key": str(paths["universe_key"]),
        "created_at_utc": existing.get("created_at_utc", now),
        "updated_at_utc": now,
        "extra_meta": _merge_dict(existing.get("extra_meta", {}), to_serializable(extra_meta or {})),
    }
    atomic_write_json(meta_path, meta, backup=True)
    return {
        "universe_name": str(universe_name),
        "universe_key": str(paths["universe_key"]),
        "universe_dir": str(paths["root"].as_posix()),
        "meta_path": str(meta_path.as_posix()),
    }


def load_universe_meta(
    base_dir: str | Path = DEFAULT_UNIVERSE_BASE_DIR,
    universe_name: str = "default",
) -> dict[str, Any]:
    paths = get_universe_paths(base_dir=base_dir, universe_name=universe_name)
    path = paths["meta_path"]
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_universe_base_frame(
    base_df: pd.DataFrame,
    base_dir: str | Path = DEFAULT_UNIVERSE_BASE_DIR,
    universe_name: str = "default",
) -> dict[str, str]:
    paths = get_universe_paths(base_dir=base_dir, universe_name=universe_name)
    paths["base_frame_dir"].mkdir(parents=True, exist_ok=True)
    saved = save_dataframe_artifact(base_df, paths["base_frame_stem"], index=False)
    _update_universe_meta(
        base_dir=base_dir,
        universe_name=universe_name,
        patch={
            "base_frame_path": saved["path"],
            "base_row_count": int(len(base_df)),
            "base_columns": [str(c) for c in base_df.columns],
        },
    )
    return saved


def load_universe_base_frame(
    base_dir: str | Path = DEFAULT_UNIVERSE_BASE_DIR,
    universe_name: str = "default",
) -> pd.DataFrame:
    meta = load_universe_meta(base_dir=base_dir, universe_name=universe_name)
    base_path = str(meta.get("base_frame_path", "") or "").strip()
    if base_path:
        return load_saved_dataframe(base_path)

    paths = get_universe_paths(base_dir=base_dir, universe_name=universe_name)
    return load_dataframe_from_stem(paths["base_frame_stem"])


def load_universe_expression_registry(
    base_dir: str | Path = DEFAULT_UNIVERSE_BASE_DIR,
    universe_name: str = "default",
) -> pd.DataFrame:
    paths = get_universe_paths(base_dir=base_dir, universe_name=universe_name)
    csv_path = paths["expression_csv"]
    txt_path = paths["expression_txt"]

    if csv_path.exists():
        out = read_csv_with_backup(csv_path)
    elif txt_path.exists():
        rows: list[dict[str, Any]] = []
        for line in txt_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            alpha_name = ""
            expression = line
            if "\t" in line:
                alpha_name, expression = line.split("\t", 1)
            rows.append({"alpha_name": alpha_name.strip(), "expression": expression.strip()})
        out = pd.DataFrame(rows)
    else:
        out = pd.DataFrame()

    required_cols = _expression_registry_columns()
    if out.empty:
        return pd.DataFrame(columns=required_cols)

    out = out.copy()
    if "alpha_name" not in out.columns:
        out["alpha_name"] = [format_alpha_name(i) for i in range(1, len(out) + 1)]
    if "expression" not in out.columns:
        out["expression"] = ""
    if "original_expression" not in out.columns:
        out["original_expression"] = out["expression"]
    if "simplified_expression" not in out.columns:
        out["simplified_expression"] = ""
    if "canonical_expression" not in out.columns:
        out["canonical_expression"] = ""
    if "canonical_hash" not in out.columns:
        out["canonical_hash"] = ""
    if "lint_passed" not in out.columns:
        out["lint_passed"] = True
    if "lint_reject_reason" not in out.columns:
        out["lint_reject_reason"] = ""
    if "lint_status" not in out.columns:
        out["lint_status"] = ""
    if "lint_warning_reason" not in out.columns:
        out["lint_warning_reason"] = ""
    if "source" not in out.columns:
        out["source"] = "unknown"
    if "factor_family" not in out.columns:
        out["factor_family"] = ""
    if "factor_family_mix_json" not in out.columns:
        out["factor_family_mix_json"] = "{}"
    if "primary_factor_family" not in out.columns:
        out["primary_factor_family"] = ""
    if "created_at_utc" not in out.columns:
        out["created_at_utc"] = ""
    if "simulation_config_json" not in out.columns:
        out["simulation_config_json"] = ""
    if "simulation_hash" not in out.columns:
        out["simulation_hash"] = ""
    if "signal_hash" not in out.columns:
        out["signal_hash"] = ""
    if "neutralization" not in out.columns:
        out["neutralization"] = ""
    if "input_manifest_id" not in out.columns:
        out["input_manifest_id"] = ""
    if "input_source_path" not in out.columns:
        out["input_source_path"] = ""
    if "panel_signature_hash" not in out.columns:
        out["panel_signature_hash"] = ""
    if "search_mode" not in out.columns:
        out["search_mode"] = ""

    out["alpha_name"] = out["alpha_name"].astype(str).str.strip()
    out["expression"] = out["expression"].astype(str)
    out["original_expression"] = out["original_expression"].fillna(out["expression"]).astype(str)
    out["source"] = out["source"].astype(str)
    out["expression_hash"] = out["expression"].map(expression_hash)
    out = _ensure_expression_canonical_columns(out)
    out["simulation_config_json"] = (
        out["simulation_config_json"].fillna("").astype(str).map(canonical_simulation_config_json)
    )
    out["simulation_hash"] = out["simulation_config_json"].map(simulation_hash_from_json)
    out["signal_hash"] = out.apply(
        lambda row: signal_hash_for_expression(
            str(row.get("canonical_expression", "") or row.get("expression", "")),
            str(row.get("simulation_config_json", "")),
        ),
        axis=1,
    )
    out["neutralization"] = out["simulation_config_json"].map(_neutralization_from_simulation_json)
    out["lint_status"] = out.apply(_derive_lint_status, axis=1)
    out["lint_warning_reason"] = out["lint_warning_reason"].fillna("").astype(str)
    out["input_manifest_id"] = out["input_manifest_id"].fillna("").astype(str)
    out["input_source_path"] = out["input_source_path"].fillna("").astype(str)
    out["panel_signature_hash"] = out["panel_signature_hash"].fillna("").astype(str)
    out["search_mode"] = out["search_mode"].fillna("").astype(str)

    out = out[out["alpha_name"].map(parse_alpha_index).notna()].copy()
    out["_alpha_idx"] = out["alpha_name"].map(lambda x: int(parse_alpha_index(x) or 0))
    out = out.sort_values(["_alpha_idx", "alpha_name"]).drop_duplicates(subset=["alpha_name"], keep="last")
    out = out.drop(columns=["_alpha_idx"]).reset_index(drop=True)
    extra_cols = [c for c in out.columns if c not in required_cols]
    return out[required_cols + extra_cols]


def load_seen_expression_hashes_for_universe(
    base_dir: str | Path = DEFAULT_UNIVERSE_BASE_DIR,
    universe_name: str = "default",
    simulation_config_json: str | None = None,
) -> set[str]:
    reg = load_universe_expression_registry(base_dir=base_dir, universe_name=universe_name)
    if reg.empty:
        return set()
    work = reg
    if simulation_config_json is not None and "simulation_hash" in work.columns:
        sim_hash = simulation_hash_from_json(canonical_simulation_config_json(simulation_config_json))
        work = work[work["simulation_hash"].astype(str) == sim_hash]
    seen: set[str] = set()
    for col in ["expression_hash", "canonical_hash"]:
        if col in work.columns:
            seen.update({str(x) for x in work[col].dropna().astype(str).tolist() if str(x)})
    return seen


def load_seen_signal_hashes_for_universe(
    base_dir: str | Path = DEFAULT_UNIVERSE_BASE_DIR,
    universe_name: str = "default",
    simulation_config_json: str | None = None,
) -> set[str]:
    reg = load_universe_expression_registry(base_dir=base_dir, universe_name=universe_name)
    if reg.empty or "signal_hash" not in reg.columns:
        return set()
    work = reg
    if simulation_config_json is not None and "simulation_hash" in work.columns:
        sim_hash = simulation_hash_from_json(canonical_simulation_config_json(simulation_config_json))
        work = work[work["simulation_hash"].astype(str) == sim_hash]
    return {str(x) for x in work["signal_hash"].dropna().astype(str).tolist() if str(x)}


def canonical_simulation_config_json(config: Any) -> str:
    if isinstance(config, str):
        raw = config.strip()
        if raw:
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {"neutralization": raw}
        else:
            payload = {}
    else:
        payload = to_serializable(config or {})
    if not isinstance(payload, dict):
        payload = {}
    payload = dict(payload)
    payload["neutralization"] = normalize_neutralization_mode(str(payload.get("neutralization", "NONE")))
    return json.dumps(
        to_serializable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def simulation_hash_from_json(simulation_config_json: str) -> str:
    return expression_hash(canonical_simulation_config_json(simulation_config_json))


def signal_hash_for_expression(expression: str, simulation_config_json: str) -> str:
    expr = str(expression or "").strip()
    try:
        canonical = canonicalize_expression(expr)
        if canonical.passed and canonical.canonical_expression:
            expr = str(canonical.canonical_expression)
    except Exception:
        pass
    return expression_hash(f"{expr}|{canonical_simulation_config_json(simulation_config_json)}")


def _neutralization_from_simulation_json(simulation_config_json: str) -> str:
    try:
        payload = json.loads(canonical_simulation_config_json(simulation_config_json))
    except Exception:
        payload = {}
    return normalize_neutralization_mode(str(payload.get("neutralization", "NONE")))


def next_universe_alpha_index(
    base_dir: str | Path = DEFAULT_UNIVERSE_BASE_DIR,
    universe_name: str = "default",
) -> int:
    reg = load_universe_expression_registry(base_dir=base_dir, universe_name=universe_name)
    if reg.empty:
        return 1
    max_idx = max(int(parse_alpha_index(x) or 0) for x in reg["alpha_name"].tolist())
    return int(max_idx) + 1


def save_universe_input_manifest(
    manifest: dict[str, Any],
    base_dir: str | Path = DEFAULT_UNIVERSE_BASE_DIR,
    universe_name: str = "default",
    manifest_id: str | None = None,
) -> dict[str, Any]:
    paths = get_universe_paths(base_dir=base_dir, universe_name=universe_name)
    paths["input_manifest_dir"].mkdir(parents=True, exist_ok=True)
    now_iso = _utc_now().isoformat()
    manifest_key = str(manifest_id or f"manifest_{now_iso.replace(':', '').replace('-', '')}").strip()
    if not manifest_key:
        raise ValueError("manifest_id resolves to empty string")

    record = {
        "manifest_id": manifest_key,
        "created_at_utc": now_iso,
        "payload": to_serializable(manifest or {}),
    }
    path = paths["input_manifest_dir"] / f"{manifest_key}.json"
    atomic_write_json(path, record, backup=True)
    _update_universe_meta(
        base_dir=base_dir,
        universe_name=universe_name,
        patch={"last_input_manifest_id": manifest_key},
    )
    return {"manifest_id": manifest_key, "path": str(path.as_posix())}


def load_universe_input_manifest(
    base_dir: str | Path = DEFAULT_UNIVERSE_BASE_DIR,
    universe_name: str = "default",
    manifest_id: str | None = None,
) -> dict[str, Any]:
    paths = get_universe_paths(base_dir=base_dir, universe_name=universe_name)
    manifest_key = str(manifest_id or "").strip()
    if not manifest_key:
        meta = load_universe_meta(base_dir=base_dir, universe_name=universe_name)
        manifest_key = str(meta.get("last_input_manifest_id", "") or "").strip()
    if not manifest_key:
        return {}
    path = paths["input_manifest_dir"] / f"{manifest_key}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return {}


def load_alpha_lifecycle_registry(
    base_dir: str | Path = DEFAULT_UNIVERSE_BASE_DIR,
    universe_name: str = "default",
) -> pd.DataFrame:
    paths = get_universe_paths(base_dir=base_dir, universe_name=universe_name)
    path = paths["lifecycle_csv"]
    cols = _lifecycle_columns()
    if not path.exists():
        return pd.DataFrame(columns=cols)
    out = read_csv_with_backup(path)
    for col in cols:
        if col not in out.columns:
            out[col] = ""
    return out[cols]


def dashboard_factor_metric_columns() -> list[str]:
    columns = [
        "factor",
        "period",
        "layers",
        "expression",
        "ic_mean",
        "ir",
        "long_short_total_return",
        "long_short_annualized_return",
        "long_short_volatility",
        "long_short_sharpe_ratio",
        "long_short_max_drawdown",
        "long_short_fitness_ratio",
        "best_layer_total_return",
        "best_layer_annualized_return",
        "best_layer_volatility",
        "best_layer_sharpe",
        "best_layer_max_drawdown",
        "best_layer_fitness_ratio",
        "best_minus_universe_annualized_return",
        "turnover_long_only_mean",
        "margin_long_only",
        "score_predictive_power",
        "score_long_only_performance",
        "score_stability",
        "score_tradeability",
        "score_total",
        "effectiveness_tier",
        "ic_decay_spearman",
        "feedback_phase",
        "feedback_score",
        "benchmark_annualized_return",
        "long_short_excess_annualized_return_vs_benchmark",
        "long_only_excess_annualized_return_vs_benchmark",
        "best_minus_benchmark_annualized_return",
    ]
    phase_metric_suffixes = [
        "obs",
        "ic_mean",
        "ic_std",
        "ir",
        "positive_ic_ratio",
        "long_short_total_return",
        "long_short_annualized_return",
        "long_short_volatility",
        "long_short_sharpe_ratio",
        "long_short_max_drawdown",
        "long_short_fitness_ratio",
        "long_short_excess_annualized_return_vs_benchmark",
        "long_only_total_return",
        "long_only_annualized_return",
        "long_only_volatility",
        "long_only_sharpe_ratio",
        "long_only_max_drawdown",
        "long_only_fitness_ratio",
        "long_only_excess_annualized_return_vs_benchmark",
        "benchmark_annualized_return",
        "best_minus_benchmark_annualized_return",
        "turnover_long_short_mean",
        "margin_long_short",
        "margin_long_short_bp",
        "turnover_long_only_mean",
        "margin_long_only",
        "margin_long_only_bp",
        "score_total",
        "feedback_score",
    ]
    for phase in ["train", "val", "test"]:
        columns.extend(f"{phase}_{suffix}" for suffix in phase_metric_suffixes)
    return columns


def build_dashboard_factor_metrics(
    factor_metrics_df: pd.DataFrame,
    expression_registry_df: pd.DataFrame | None = None,
    period: int = 1,
    layers: int = 10,
) -> pd.DataFrame:
    columns = dashboard_factor_metric_columns()
    if factor_metrics_df is None or factor_metrics_df.empty or "factor" not in factor_metrics_df.columns:
        return pd.DataFrame(columns=columns)

    out = factor_metrics_df.copy()
    out["factor"] = out["factor"].astype(str)
    out["period"] = int(period)
    out["layers"] = int(layers)

    expr_map: dict[str, str] = {}
    if (
        expression_registry_df is not None
        and not expression_registry_df.empty
        and "alpha_name" in expression_registry_df.columns
    ):
        reg = expression_registry_df.copy()
        reg["alpha_name"] = reg["alpha_name"].astype(str)
        for _, row in reg.iterrows():
            canonical = str(row.get("canonical_expression", "") or "").strip()
            expression = str(row.get("expression", "") or "").strip()
            expr_map[str(row.get("alpha_name"))] = canonical or expression

    if "expression" not in out.columns:
        out["expression"] = ""
    out["expression"] = [
        expr_map.get(str(row.get("factor")), str(row.get("expression", "") or "")) for _, row in out.iterrows()
    ]

    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out[columns].sort_values("factor", kind="mergesort").reset_index(drop=True)


def append_alpha_lifecycle_records(
    lifecycle_df: pd.DataFrame,
    base_dir: str | Path = DEFAULT_UNIVERSE_BASE_DIR,
    universe_name: str = "default",
) -> pd.DataFrame:
    if lifecycle_df is None or len(lifecycle_df) == 0:
        return pd.DataFrame(columns=_lifecycle_columns())
    incoming = lifecycle_df.copy()
    if "alpha_name" not in incoming.columns:
        raise ValueError("lifecycle_df must include 'alpha_name'")
    for col in _lifecycle_columns():
        if col not in incoming.columns:
            incoming[col] = ""
    incoming["alpha_name"] = incoming["alpha_name"].astype(str).str.strip()
    incoming = incoming[incoming["alpha_name"] != ""].copy()
    if incoming.empty:
        return pd.DataFrame(columns=_lifecycle_columns())

    paths = get_universe_paths(base_dir=base_dir, universe_name=universe_name)
    existing = load_alpha_lifecycle_registry(base_dir=base_dir, universe_name=universe_name)
    merged = pd.concat([existing, incoming[_lifecycle_columns()]], ignore_index=True)
    merged = merged.drop_duplicates(subset=["alpha_name"], keep="last").reset_index(drop=True)
    paths["catalog_dir"].mkdir(parents=True, exist_ok=True)
    atomic_write_dataframe_csv(paths["lifecycle_csv"], merged, index=False, backup=True)
    return incoming[_lifecycle_columns()].reset_index(drop=True)


def update_alpha_lifecycle_status(
    alpha_names: Sequence[str],
    status: str,
    base_dir: str | Path = DEFAULT_UNIVERSE_BASE_DIR,
    universe_name: str = "default",
    error_message: str | None = None,
    failure_kind: str | None = None,
    last_error_stage: str | None = None,
    analysis_run_id: str | None = None,
    alpha_value_path: str | None = None,
    simulation_config_json: str | None = None,
    input_manifest_id: str | None = None,
) -> pd.DataFrame:
    names = [str(x).strip() for x in alpha_names if str(x).strip()]
    if not names:
        return pd.DataFrame(columns=_lifecycle_columns())

    status_norm = str(status or "").strip().upper()
    if not status_norm:
        raise ValueError("status must not be empty")

    now_iso = _utc_now().isoformat()
    existing = load_alpha_lifecycle_registry(base_dir=base_dir, universe_name=universe_name)
    if existing.empty:
        existing = pd.DataFrame(columns=_lifecycle_columns())

    rec_map = {str(r["alpha_name"]): r.to_dict() for _, r in existing.iterrows()}
    for name in names:
        rec = rec_map.get(name, {c: "" for c in _lifecycle_columns()})
        rec["alpha_name"] = name
        rec["status"] = status_norm
        if not rec.get("created_at_utc"):
            rec["created_at_utc"] = now_iso
        if status_norm == "MATERIALIZED":
            rec["materialized_at_utc"] = now_iso
        if status_norm == "ANALYZED":
            rec["analyzed_at_utc"] = now_iso
        if status_norm == "PURGED":
            rec["purged_at_utc"] = now_iso
        if status_norm == "REPRODUCED":
            rec["reproduced_at_utc"] = now_iso
        if status_norm == "FAILED":
            rec["last_error"] = str(error_message or rec.get("last_error", "") or "")
            if failure_kind is not None:
                rec["failure_kind"] = str(failure_kind)
            if last_error_stage is not None:
                rec["last_error_stage"] = str(last_error_stage)
            retry_num = pd.to_numeric(rec.get("retry_count", 0), errors="coerce")
            retry_base = 0 if pd.isna(retry_num) else int(retry_num)
            rec["retry_count"] = retry_base + 1
        if status_norm == "PERMANENT_FAILED":
            rec["last_error"] = str(error_message or rec.get("last_error", "") or "")
            rec["failure_kind"] = str(failure_kind or "permanent")
            rec["last_error_stage"] = str(last_error_stage or "")
        if analysis_run_id is not None:
            rec["analysis_run_id"] = str(analysis_run_id)
        if alpha_value_path is not None:
            rec["alpha_value_path"] = str(alpha_value_path)
        if simulation_config_json is not None:
            rec["simulation_config_json"] = str(simulation_config_json)
        if input_manifest_id is not None:
            rec["input_manifest_id"] = str(input_manifest_id)
        rec_map[name] = rec

    updated = pd.DataFrame(list(rec_map.values()))
    for col in _lifecycle_columns():
        if col not in updated.columns:
            updated[col] = ""
    updated = updated[_lifecycle_columns()].copy()
    paths = get_universe_paths(base_dir=base_dir, universe_name=universe_name)
    paths["catalog_dir"].mkdir(parents=True, exist_ok=True)
    atomic_write_dataframe_csv(paths["lifecycle_csv"], updated, index=False, backup=True)
    return updated[updated["alpha_name"].isin(names)].reset_index(drop=True)


def delete_universe_alpha_values(
    alpha_names: Sequence[str],
    base_dir: str | Path = DEFAULT_UNIVERSE_BASE_DIR,
    universe_name: str = "default",
) -> dict[str, Any]:
    names = [str(x).strip() for x in alpha_names if str(x).strip()]
    if not names:
        return {
            "requested": 0,
            "deleted": 0,
            "missing": 0,
            "failed": 0,
            "deleted_paths": [],
            "failed_paths": [],
        }
    paths = get_universe_paths(base_dir=base_dir, universe_name=universe_name)
    deleted_paths: list[str] = []
    failed_paths: list[str] = []
    missing = 0
    failed = 0
    for name in names:
        removed = False
        found_any = False
        for ext in [".parquet", ".pkl", ".csv"]:
            p = paths["alphas_dir"] / f"{name}{ext}"
            if p.exists():
                found_any = True
                # Windows/Jupyter 可能会短暂持有文件句柄，做小步重试避免整批失败。
                deleted_this_file = False
                for attempt in range(3):
                    try:
                        p.unlink()
                        deleted_paths.append(str(p.as_posix()))
                        removed = True
                        deleted_this_file = True
                        break
                    except PermissionError:
                        if attempt < 2:
                            time.sleep(0.1 * (attempt + 1))
                        else:
                            failed += 1
                            failed_paths.append(str(p.as_posix()))
                if deleted_this_file:
                    continue
        if (not removed) and (not found_any):
            missing += 1
    return {
        "requested": len(names),
        "deleted": len(deleted_paths),
        "missing": int(missing),
        "failed": int(failed),
        "deleted_paths": deleted_paths,
        "failed_paths": failed_paths,
    }


def load_factor_metrics_registry(
    base_dir: str | Path = DEFAULT_UNIVERSE_BASE_DIR,
    universe_name: str = "default",
) -> pd.DataFrame:
    paths = get_universe_paths(base_dir=base_dir, universe_name=universe_name)
    path = paths["factor_metrics_registry_csv"]
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "factor",
                "period",
                "layers",
                "is_timeseries",
                "analysis_run_id",
                "analysis_dir",
                "factor_metrics_path",
                "analyzed_at_utc",
            ]
        )
    out = read_csv_with_backup(path)
    if "factor" in out.columns:
        out["factor"] = out["factor"].astype(str)
    return out


def append_factor_metrics_registry(
    factor_metrics_df: pd.DataFrame,
    base_dir: str | Path,
    universe_name: str,
    period: int,
    layers: int,
    is_timeseries: bool,
    analysis_run_id: str,
    analysis_dir: str,
    factor_metrics_path: str,
    analyzed_at_utc: str,
) -> pd.DataFrame:
    if factor_metrics_df is None or factor_metrics_df.empty:
        return pd.DataFrame()
    if "factor" not in factor_metrics_df.columns:
        raise ValueError("factor_metrics_df must include 'factor'")

    payload = factor_metrics_df.copy()
    payload["factor"] = payload["factor"].astype(str)
    payload["period"] = int(period)
    payload["layers"] = int(layers)
    payload["is_timeseries"] = bool(is_timeseries)
    payload["analysis_run_id"] = str(analysis_run_id)
    payload["analysis_dir"] = str(analysis_dir)
    payload["factor_metrics_path"] = str(factor_metrics_path)
    payload["analyzed_at_utc"] = str(analyzed_at_utc)

    paths = get_universe_paths(base_dir=base_dir, universe_name=universe_name)
    existing = load_factor_metrics_registry(base_dir=base_dir, universe_name=universe_name)
    merged = pd.concat([existing, payload], ignore_index=True)
    dedupe_keys = [k for k in ["factor", "period", "layers", "is_timeseries"] if k in merged.columns]
    if dedupe_keys:
        merged = merged.drop_duplicates(subset=dedupe_keys, keep="last")
    paths["analysis_dir"].mkdir(parents=True, exist_ok=True)
    atomic_write_dataframe_csv(paths["factor_metrics_registry_csv"], merged, index=False, backup=True)
    return payload.reset_index(drop=True)


def append_universe_expressions(
    expression_df: pd.DataFrame,
    base_dir: str | Path = DEFAULT_UNIVERSE_BASE_DIR,
    universe_name: str = "default",
) -> pd.DataFrame:
    if expression_df is None or len(expression_df) == 0:
        return pd.DataFrame(columns=_expression_registry_columns())
    if "expression" not in expression_df.columns:
        raise ValueError("expression_df must include 'expression' column")

    existing = load_universe_expression_registry(base_dir=base_dir, universe_name=universe_name)
    seen_hashes = set()
    seen_signal_hashes = set()
    if len(existing):
        for col in ["expression_hash", "canonical_hash"]:
            if col in existing.columns:
                seen_hashes.update({str(x) for x in existing[col].dropna().astype(str).tolist() if str(x)})
        if "signal_hash" in existing.columns:
            seen_signal_hashes.update({str(x) for x in existing["signal_hash"].dropna().astype(str).tolist() if str(x)})

    rows: list[dict[str, Any]] = []
    current_idx = next_universe_alpha_index(base_dir=base_dir, universe_name=universe_name)
    now_iso = _utc_now().isoformat()
    for _, row in expression_df.iterrows():
        raw_expr = str(row.get("original_expression", row.get("expression", "")) or "").strip()
        expr = str(row.get("expression", "") or "").strip()
        if not expr:
            continue
        canonical = canonicalize_expression(expr)
        if not canonical.passed:
            continue
        exec_expr = str(canonical.canonical_expression or expr).strip()
        if _is_constant_expression(exec_expr):
            continue
        expr_hash = expression_hash(exec_expr)
        canonical_hash = str(canonical.canonical_hash or expr_hash)
        simulation_config_json = canonical_simulation_config_json(str(row.get("simulation_config_json", "") or ""))
        simulation_hash = simulation_hash_from_json(simulation_config_json)
        signal_hash = signal_hash_for_expression(exec_expr, simulation_config_json)
        neutralization = _neutralization_from_simulation_json(simulation_config_json)
        if signal_hash in seen_signal_hashes:
            continue
        if not str(row.get("simulation_config_json", "") or "").strip() and (
            expr_hash in seen_hashes or canonical_hash in seen_hashes
        ):
            continue
        alpha_name = format_alpha_name(current_idx)
        payload_row = {
            "alpha_name": alpha_name,
            "expression": exec_expr,
            "original_expression": raw_expr or expr,
            "simplified_expression": exec_expr,
            "source": str(row.get("source", "unknown") or "unknown"),
            "factor_family": str(row.get("factor_family", "") or ""),
            "factor_family_mix_json": str(row.get("factor_family_mix_json", "{}") or "{}"),
            "primary_factor_family": str(row.get("primary_factor_family", "") or ""),
            "expression_hash": expr_hash,
            "canonical_expression": exec_expr,
            "canonical_hash": canonical_hash,
            "lint_passed": True,
            "lint_reject_reason": "",
            "lint_status": "simplified" if exec_expr != expr else "passed",
            "lint_warning_reason": "canonical_simplified" if exec_expr != expr else "",
            "created_at_utc": now_iso,
            "simulation_config_json": simulation_config_json,
            "simulation_hash": simulation_hash,
            "signal_hash": signal_hash,
            "neutralization": neutralization,
            "input_manifest_id": str(row.get("input_manifest_id", "") or ""),
            "input_source_path": str(row.get("input_source_path", "") or ""),
            "panel_signature_hash": str(row.get("panel_signature_hash", "") or ""),
            "search_mode": str(row.get("search_mode", "") or ""),
        }
        for col, value in row.items():
            if col not in payload_row and col != "alpha_name":
                payload_row[str(col)] = value
        rows.append(payload_row)
        seen_hashes.add(expr_hash)
        seen_hashes.add(canonical_hash)
        seen_signal_hashes.add(signal_hash)
        current_idx += 1

    if not rows:
        return pd.DataFrame(columns=_expression_registry_columns())

    new_df = pd.DataFrame(rows)
    merged = pd.concat([existing, new_df], ignore_index=True)
    merged["_alpha_idx"] = merged["alpha_name"].map(lambda x: int(parse_alpha_index(x) or 0))
    merged = merged.sort_values(["_alpha_idx", "alpha_name"]).drop(columns=["_alpha_idx"]).reset_index(drop=True)

    _save_universe_expression_registry(
        merged,
        base_dir=base_dir,
        universe_name=universe_name,
    )
    _update_universe_meta(
        base_dir=base_dir,
        universe_name=universe_name,
        patch={
            "expression_count": int(len(merged)),
            "last_expression_alpha": str(merged["alpha_name"].iloc[-1]),
        },
    )
    return new_df


def _save_universe_expression_registry(
    expression_df: pd.DataFrame,
    base_dir: str | Path = DEFAULT_UNIVERSE_BASE_DIR,
    universe_name: str = "default",
) -> None:
    paths = get_universe_paths(base_dir=base_dir, universe_name=universe_name)
    paths["catalog_dir"].mkdir(parents=True, exist_ok=True)
    out = expression_df.copy()
    atomic_write_dataframe_csv(paths["expression_csv"], out, index=False, backup=True)

    lines = [f"{row['alpha_name']}\t{row['expression']}" for _, row in out.iterrows()]
    atomic_write_text(paths["expression_txt"], "\n".join(lines), encoding="utf-8", backup=True)


def save_universe_alpha_values(
    alpha_df: pd.DataFrame,
    alpha_name: str,
    base_dir: str | Path = DEFAULT_UNIVERSE_BASE_DIR,
    universe_name: str = "default",
    date_col: str = "date",
    code_col: str = "code",
) -> dict[str, str]:
    if alpha_name not in alpha_df.columns:
        raise ValueError(f"alpha_name column not found in alpha_df: {alpha_name}")
    required = [date_col, code_col, alpha_name]
    missing = [c for c in required if c not in alpha_df.columns]
    if missing:
        raise ValueError(f"Missing columns in alpha_df for save_universe_alpha_values: {missing}")

    paths = get_universe_paths(base_dir=base_dir, universe_name=universe_name)
    paths["alphas_dir"].mkdir(parents=True, exist_ok=True)
    save_df = alpha_df[required].copy()
    saved = save_dataframe_artifact(save_df, paths["alphas_dir"] / alpha_name, index=False)
    return saved


def load_universe_alpha_values(
    alpha_name: str,
    base_dir: str | Path = DEFAULT_UNIVERSE_BASE_DIR,
    universe_name: str = "default",
) -> pd.DataFrame:
    paths = get_universe_paths(base_dir=base_dir, universe_name=universe_name)
    return load_dataframe_from_stem(paths["alphas_dir"] / alpha_name)


def load_universe_alpha_batch(
    alpha_names: Sequence[str],
    base_dir: str | Path = DEFAULT_UNIVERSE_BASE_DIR,
    universe_name: str = "default",
    date_col: str = "date",
    code_col: str = "code",
) -> pd.DataFrame:
    names = [str(x) for x in alpha_names if str(x)]
    if not names:
        return pd.DataFrame(columns=[date_col, code_col])

    merged: pd.DataFrame | None = None
    for alpha_name in names:
        frame = load_universe_alpha_values(alpha_name=alpha_name, base_dir=base_dir, universe_name=universe_name)
        work = frame.copy()
        if alpha_name not in work.columns:
            value_cols = [c for c in work.columns if c not in [date_col, code_col]]
            if len(value_cols) == 1:
                work = work.rename(columns={value_cols[0]: alpha_name})
            else:
                raise ValueError(f"Could not resolve value column for {alpha_name}")
        work = work[[date_col, code_col, alpha_name]].copy()
        if merged is None:
            merged = work
        else:
            merged = pd.merge(merged, work, on=[date_col, code_col], how="outer")

    if merged is None:
        return pd.DataFrame(columns=[date_col, code_col])
    merged[date_col] = pd.to_datetime(merged[date_col], errors="coerce")
    merged = merged.sort_values([code_col, date_col], kind="mergesort").reset_index(drop=True)
    return merged


def load_universe_analysis_registry(
    base_dir: str | Path = DEFAULT_UNIVERSE_BASE_DIR,
    universe_name: str = "default",
) -> pd.DataFrame:
    paths = get_universe_paths(base_dir=base_dir, universe_name=universe_name)
    path = paths["analysis_registry_csv"]
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "alpha_name",
                "period",
                "layers",
                "is_timeseries",
                "analysis_run_id",
                "analysis_dir",
                "factor_metrics_path",
                "analyzed_at_utc",
            ]
        )
    out = read_csv_with_backup(path)
    for col in ["period", "layers"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    if "is_timeseries" in out.columns:
        out["is_timeseries"] = out["is_timeseries"].astype(str).str.lower().isin({"1", "true", "yes"})
    return out


def select_alpha_names_for_analysis(
    base_dir: str | Path,
    universe_name: str,
    mode: str = "next_pending",
    batch_size: int = 5,
    period: int = 1,
    layers: int = 10,
    is_timeseries: bool = True,
    alpha_names: Iterable[str] | None = None,
    alpha_start: int | None = None,
    alpha_end: int | None = None,
    force_reanalyze: bool = False,
) -> list[str]:
    reg = load_universe_expression_registry(base_dir=base_dir, universe_name=universe_name)
    if reg.empty:
        return []

    requested_batch = max(1, int(batch_size))
    requested_batch = min(requested_batch, 5)
    all_names = reg["alpha_name"].astype(str).tolist()

    mode_norm = str(mode or "next_pending").strip().lower()
    if mode_norm == "alpha_list":
        selected = [str(x) for x in (alpha_names or []) if str(x)]
    elif mode_norm == "alpha_range":
        if alpha_start is None or alpha_end is None:
            raise ValueError("alpha_range mode requires alpha_start and alpha_end")
        lo = int(alpha_start)
        hi = int(alpha_end)
        if hi < lo:
            lo, hi = hi, lo
        selected = []
        for name in all_names:
            idx = parse_alpha_index(name)
            if idx is None:
                continue
            if lo <= idx <= hi:
                selected.append(name)
    else:
        selected = list(all_names)

    if not force_reanalyze:
        analyzed = load_universe_analysis_registry(base_dir=base_dir, universe_name=universe_name)
        if not analyzed.empty:
            period_series = (
                pd.to_numeric(analyzed["period"], errors="coerce")
                if "period" in analyzed.columns
                else pd.Series([pd.NA] * len(analyzed), index=analyzed.index)
            )
            layers_series = (
                pd.to_numeric(analyzed["layers"], errors="coerce")
                if "layers" in analyzed.columns
                else pd.Series([pd.NA] * len(analyzed), index=analyzed.index)
            )
            ts_series = (
                analyzed["is_timeseries"].astype(str).str.lower().isin(["true", "1", "yes"])
                if "is_timeseries" in analyzed.columns
                else pd.Series([False] * len(analyzed), index=analyzed.index)
            )
            mask = (period_series == int(period)) & (layers_series == int(layers)) & (ts_series == bool(is_timeseries))
            analyzed_names = (
                set(analyzed.loc[mask, "alpha_name"].astype(str).tolist())
                if "alpha_name" in analyzed.columns
                else set()
            )
            selected = [name for name in selected if name not in analyzed_names]

    return selected[:requested_batch]


def save_universe_analysis_run(
    base_dir: str | Path,
    universe_name: str,
    alpha_names: Sequence[str],
    period: int,
    layers: int,
    is_timeseries: bool,
    factor_metrics_df: pd.DataFrame,
    tables: dict[str, pd.DataFrame] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    names = [str(x) for x in alpha_names if str(x)]
    if not names:
        raise ValueError("alpha_names must not be empty")

    paths = get_universe_paths(base_dir=base_dir, universe_name=universe_name)
    alpha_span = names[0] if len(names) == 1 else f"{names[0]}-{names[-1]}"
    run_id = f"analysis_{alpha_span}_l{int(layers)}_ts{1 if bool(is_timeseries) else 0}"
    run_dir = paths["analysis_dir"] / f"period_{int(period)}" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    saved_factor_metrics = save_dataframe_artifact(
        factor_metrics_df,
        run_dir / "factor_metrics",
        index=False,
        preferred="csv",
    )

    table_paths: dict[str, str] = {}
    for name, frame in (tables or {}).items():
        if not isinstance(frame, pd.DataFrame):
            continue
        preferred = (
            "parquet"
            if str(name)
            in {
                "portfolio_pnl_df",
                "benchmark_pnl_df",
                "analysis_factor_coverage_by_date",
            }
            else "csv"
        )
        parquet_kwargs: dict[str, Any] = {}
        if str(name) == "portfolio_pnl_df":
            parquet_kwargs["row_group_size"] = int(os.environ.get("ALPHA_MINING_PNL_ROW_GROUP_SIZE", "100000"))
        saved = save_dataframe_artifact(
            frame,
            run_dir / str(name),
            index=False,
            preferred=preferred,
            parquet_kwargs=parquet_kwargs or None,
        )
        table_paths[str(name)] = saved["path"]

    record = {
        "analysis_run_id": run_id,
        "alpha_names": names,
        "period": int(period),
        "layers": int(layers),
        "is_timeseries": bool(is_timeseries),
        "analysis_dir": str(run_dir.as_posix()),
        "factor_metrics_path": saved_factor_metrics["path"],
        "table_paths": table_paths,
        "created_at_utc": _utc_now().isoformat(),
        "extra_meta": to_serializable(extra_meta or {}),
    }
    atomic_write_json(run_dir / "analysis_meta.json", record, backup=True)

    registry_rows = pd.DataFrame(
        [
            {
                "alpha_name": alpha_name,
                "period": int(period),
                "layers": int(layers),
                "is_timeseries": bool(is_timeseries),
                "analysis_run_id": run_id,
                "analysis_dir": str(run_dir.as_posix()),
                "factor_metrics_path": saved_factor_metrics["path"],
                "analyzed_at_utc": record["created_at_utc"],
            }
            for alpha_name in names
        ]
    )
    _append_analysis_registry(
        registry_rows=registry_rows,
        base_dir=base_dir,
        universe_name=universe_name,
    )
    append_factor_metrics_registry(
        factor_metrics_df=factor_metrics_df,
        base_dir=base_dir,
        universe_name=universe_name,
        period=int(period),
        layers=int(layers),
        is_timeseries=bool(is_timeseries),
        analysis_run_id=run_id,
        analysis_dir=str(run_dir.as_posix()),
        factor_metrics_path=saved_factor_metrics["path"],
        analyzed_at_utc=record["created_at_utc"],
    )
    _update_universe_meta(
        base_dir=base_dir,
        universe_name=universe_name,
        patch={
            "last_analysis_run_id": run_id,
            "last_analysis_period": int(period),
            "last_analysis_alpha_count": int(len(names)),
        },
    )
    return record


def _append_analysis_registry(
    registry_rows: pd.DataFrame,
    base_dir: str | Path,
    universe_name: str,
) -> None:
    paths = get_universe_paths(base_dir=base_dir, universe_name=universe_name)
    registry_path = paths["analysis_registry_csv"]
    paths["analysis_dir"].mkdir(parents=True, exist_ok=True)
    existing = load_universe_analysis_registry(base_dir=base_dir, universe_name=universe_name)
    merged = pd.concat([existing, registry_rows], ignore_index=True)
    dedupe_keys = ["alpha_name", "period", "layers", "is_timeseries"]
    valid_keys = [k for k in dedupe_keys if k in merged.columns]
    if valid_keys:
        merged = merged.drop_duplicates(subset=valid_keys, keep="last")
    atomic_write_dataframe_csv(registry_path, merged, index=False, backup=True)


def _expression_registry_columns() -> list[str]:
    return [
        "alpha_name",
        "expression",
        "original_expression",
        "simplified_expression",
        "source",
        "factor_family",
        "factor_family_mix_json",
        "primary_factor_family",
        "expression_hash",
        "canonical_expression",
        "canonical_hash",
        "lint_passed",
        "lint_reject_reason",
        "lint_status",
        "lint_warning_reason",
        "created_at_utc",
        "simulation_config_json",
        "simulation_hash",
        "signal_hash",
        "neutralization",
        "input_manifest_id",
        "input_source_path",
        "panel_signature_hash",
        "search_mode",
    ]


def _ensure_expression_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    for idx, row in out.iterrows():
        expr = str(row.get("expression", "") or "").strip()
        if not expr:
            continue
        current_canon = str(row.get("canonical_expression", "") or "").strip()
        current_hash = str(row.get("canonical_hash", "") or "").strip()
        if current_canon and current_hash:
            continue
        try:
            canonical = canonicalize_expression(expr)
        except Exception:
            canonical = None
        if canonical is not None and canonical.passed:
            out.at[idx, "canonical_expression"] = canonical.canonical_expression
            out.at[idx, "canonical_hash"] = canonical.canonical_hash
            if not str(row.get("simplified_expression", "") or "").strip():
                out.at[idx, "simplified_expression"] = canonical.canonical_expression
            if "lint_passed" in out.columns:
                out.at[idx, "lint_passed"] = True
            if "lint_reject_reason" in out.columns:
                out.at[idx, "lint_reject_reason"] = ""
            if "lint_status" in out.columns:
                out.at[idx, "lint_status"] = "simplified" if canonical.canonical_expression != expr else "passed"
            if "lint_warning_reason" in out.columns:
                out.at[idx, "lint_warning_reason"] = (
                    "canonical_simplified" if canonical.canonical_expression != expr else ""
                )
        else:
            out.at[idx, "canonical_expression"] = current_canon or expr
            out.at[idx, "canonical_hash"] = current_hash or expression_hash(expr)
            if "lint_passed" in out.columns:
                out.at[idx, "lint_passed"] = False
            if "lint_reject_reason" in out.columns:
                out.at[idx, "lint_reject_reason"] = getattr(canonical, "reject_reason", "canonicalize_failed")
            if "lint_status" in out.columns:
                out.at[idx, "lint_status"] = "rejected"
            if "lint_warning_reason" in out.columns:
                out.at[idx, "lint_warning_reason"] = getattr(canonical, "reject_reason", "canonicalize_failed")
    return out


def _derive_lint_status(row: pd.Series) -> str:
    current = str(row.get("lint_status", "") or "").strip()
    if current in {"passed", "simplified", "rejected"}:
        return current
    if not bool(row.get("lint_passed", True)):
        return "rejected"
    expr = str(row.get("expression", "") or "").strip()
    original = str(row.get("original_expression", expr) or "").strip()
    simplified = str(row.get("simplified_expression", "") or "").strip()
    canonical = str(row.get("canonical_expression", "") or "").strip()
    if (
        (original and expr and original != expr)
        or (simplified and expr and simplified != original)
        or (canonical and original and canonical != original)
    ):
        return "simplified"
    return "passed"


def _is_constant_expression(expression: str) -> bool:
    try:
        stats = expression_stats(parse_expression(str(expression or "").strip()))
    except Exception:
        return False
    return len(tuple(stats.unique_fields)) == 0


def _update_universe_meta(
    base_dir: str | Path,
    universe_name: str,
    patch: dict[str, Any],
) -> dict[str, Any]:
    paths = get_universe_paths(base_dir=base_dir, universe_name=universe_name)
    current = load_universe_meta(base_dir=base_dir, universe_name=universe_name)
    if not current:
        init_universe_workspace(base_dir=base_dir, universe_name=universe_name)
        current = load_universe_meta(base_dir=base_dir, universe_name=universe_name)
    merged = _merge_dict(current, to_serializable(patch))
    merged["updated_at_utc"] = _utc_now().isoformat()
    atomic_write_json(paths["meta_path"], merged, backup=True)
    return merged


def _merge_dict(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _lifecycle_columns() -> list[str]:
    return [
        "alpha_name",
        "expression",
        "expression_hash",
        "source",
        "status",
        "alpha_value_path",
        "analysis_run_id",
        "created_at_utc",
        "materialized_at_utc",
        "analyzed_at_utc",
        "purged_at_utc",
        "reproduced_at_utc",
        "last_error",
        "failure_kind",
        "last_error_stage",
        "retry_count",
        "simulation_config_json",
        "input_manifest_id",
    ]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
