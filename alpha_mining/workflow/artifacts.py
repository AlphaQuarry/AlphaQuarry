from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..atomic_io import (
    atomic_write_dataframe_csv,
    atomic_write_json,
    read_csv_with_backup,
)
from ..history.registry import compute_config_hash, to_serializable
from ..parquet_utils import write_parquet_compat


DEFAULT_RUNS_BASE_DIR = "data/alpha_pipeline_runs"


def init_run_workspace(
    base_dir: str | Path = DEFAULT_RUNS_BASE_DIR,
    run_id: str | None = None,
    config_snapshot: dict[str, Any] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    Initialize a run workspace.

    Directory layout:
    - <run>/base
    - <run>/mining/batches
    - <run>/analysis/batches
    - <run>/feedback
    """
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)

    snapshot = to_serializable(config_snapshot or {})
    config_hash = compute_config_hash(snapshot) if snapshot else "nohash"
    if run_id is None:
        ts = _utc_now().strftime("%Y%m%dT%H%M%SZ")
        run_id = f"run_{ts}_{config_hash[:8]}"

    run_dir = root / run_id
    (run_dir / "base").mkdir(parents=True, exist_ok=True)
    (run_dir / "mining" / "batches").mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis" / "batches").mkdir(parents=True, exist_ok=True)
    (run_dir / "feedback").mkdir(parents=True, exist_ok=True)

    meta_path = run_dir / "run_meta.json"
    existing_meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            existing_meta = {}

    now_iso = _utc_now().isoformat()
    merged_meta = {
        "run_id": run_id,
        "created_at_utc": existing_meta.get("created_at_utc", now_iso),
        "updated_at_utc": now_iso,
        "config_hash": existing_meta.get("config_hash", config_hash),
        "config_snapshot": snapshot if snapshot else existing_meta.get("config_snapshot", {}),
        "extra_meta": _merge_dict(existing_meta.get("extra_meta", {}), to_serializable(extra_meta or {})),
    }
    atomic_write_json(meta_path, merged_meta, backup=True)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir.as_posix()),
        "run_meta_path": str(meta_path.as_posix()),
    }


def load_run_meta(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "run_meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def update_run_meta(run_dir: str | Path, patch: dict[str, Any]) -> dict[str, Any]:
    meta = load_run_meta(run_dir)
    merged = _merge_dict(meta, to_serializable(patch))
    merged["updated_at_utc"] = _utc_now().isoformat()
    path = Path(run_dir) / "run_meta.json"
    atomic_write_json(path, merged, backup=True)
    return merged


def format_batch_id(batch_index: int, alpha_start: int, alpha_end: int) -> str:
    return f"batch_{int(batch_index):03d}__alpha_{int(alpha_start):04d}-{int(alpha_end):04d}"


def chunk_list(values: list[Any], chunk_size: int) -> list[list[Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    return [values[i : i + chunk_size] for i in range(0, len(values), chunk_size)]


def save_base_frame(
    run_dir: str | Path,
    base_df: pd.DataFrame,
    stem_name: str = "base_frame",
) -> dict[str, str]:
    base_dir = Path(run_dir) / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    saved = save_dataframe_artifact(base_df, base_dir / stem_name, index=False)
    update_run_meta(
        run_dir,
        {
            "base_frame_path": saved["path"],
            "base_row_count": int(len(base_df)),
            "base_columns": [str(c) for c in base_df.columns],
        },
    )
    return saved


def load_base_frame(run_dir: str | Path) -> pd.DataFrame:
    meta = load_run_meta(run_dir)
    base_path = meta.get("base_frame_path")
    if base_path:
        return load_saved_dataframe(base_path)
    return load_dataframe_from_stem(Path(run_dir) / "base" / "base_frame")


def save_mining_batch(
    run_dir: str | Path,
    batch_index: int,
    alpha_start: int,
    alpha_end: int,
    alpha_df: pd.DataFrame,
    expression_df: pd.DataFrame,
    expr_timing_df: pd.DataFrame | None = None,
    operator_timing_df: pd.DataFrame | None = None,
    diagnostics_df: pd.DataFrame | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    batch_id = format_batch_id(batch_index=batch_index, alpha_start=alpha_start, alpha_end=alpha_end)
    batch_dir = Path(run_dir) / "mining" / "batches" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    saved_alpha = save_dataframe_artifact(alpha_df, batch_dir / "alpha_values", index=False)
    saved_expr = save_dataframe_artifact(expression_df, batch_dir / "expression_map", index=False, preferred="csv")
    saved_expr_timing = (
        save_dataframe_artifact(
            expr_timing_df,
            batch_dir / "expression_timing",
            index=False,
            preferred="csv",
        )
        if isinstance(expr_timing_df, pd.DataFrame)
        else None
    )
    saved_operator_timing = (
        save_dataframe_artifact(
            operator_timing_df,
            batch_dir / "operator_timing",
            index=False,
            preferred="csv",
        )
        if isinstance(operator_timing_df, pd.DataFrame)
        else None
    )
    saved_diag = (
        save_dataframe_artifact(diagnostics_df, batch_dir / "diagnostics", index=False, preferred="csv")
        if isinstance(diagnostics_df, pd.DataFrame)
        else None
    )

    created_at = _utc_now().isoformat()
    record = {
        "batch_index": int(batch_index),
        "batch_id": batch_id,
        "alpha_start": int(alpha_start),
        "alpha_end": int(alpha_end),
        "alpha_count": int(max(0, alpha_end - alpha_start + 1)),
        "row_count": int(len(alpha_df)),
        "alpha_values_path": saved_alpha["path"],
        "expression_map_path": saved_expr["path"],
        "expression_timing_path": saved_expr_timing["path"] if saved_expr_timing else "",
        "operator_timing_path": saved_operator_timing["path"] if saved_operator_timing else "",
        "diagnostics_path": saved_diag["path"] if saved_diag else "",
        "created_at_utc": created_at,
    }

    batch_meta = {
        **record,
        "extra_meta": to_serializable(extra_meta or {}),
    }
    atomic_write_json(batch_dir / "batch_meta.json", batch_meta, backup=True)

    _append_manifest_row(
        Path(run_dir) / "mining" / "batch_manifest.csv",
        row=record,
        dedupe_keys=["batch_id"],
    )
    return batch_meta


def load_mining_manifest(run_dir: str | Path) -> pd.DataFrame:
    manifest_path = Path(run_dir) / "mining" / "batch_manifest.csv"
    if not manifest_path.exists():
        return pd.DataFrame(
            columns=[
                "batch_index",
                "batch_id",
                "alpha_start",
                "alpha_end",
                "alpha_count",
                "row_count",
                "alpha_values_path",
                "expression_map_path",
                "expression_timing_path",
                "operator_timing_path",
                "diagnostics_path",
                "created_at_utc",
            ]
        )
    out = read_csv_with_backup(manifest_path)
    if "batch_index" in out.columns:
        out = out.sort_values("batch_index").reset_index(drop=True)
    return out


def save_analysis_batch(
    run_dir: str | Path,
    batch_id: str,
    period: int,
    factor_metrics_df: pd.DataFrame,
    tables: dict[str, pd.DataFrame] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    analysis_dir = Path(run_dir) / "analysis" / "batches" / batch_id / f"period_{int(period)}"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    saved_factor_metrics = save_dataframe_artifact(
        factor_metrics_df,
        analysis_dir / "factor_metrics",
        index=False,
        preferred="csv",
    )

    table_paths: dict[str, str] = {}
    for name, frame in (tables or {}).items():
        if not isinstance(frame, pd.DataFrame):
            continue
        saved = save_dataframe_artifact(frame, analysis_dir / name, index=False, preferred="csv")
        table_paths[str(name)] = saved["path"]

    created_at = _utc_now().isoformat()
    record = {
        "batch_id": batch_id,
        "period": int(period),
        "factor_count": int(len(factor_metrics_df)),
        "factor_metrics_path": saved_factor_metrics["path"],
        "table_paths_json": json.dumps(table_paths, ensure_ascii=False, sort_keys=True),
        "created_at_utc": created_at,
    }

    batch_meta = {**record, "extra_meta": to_serializable(extra_meta or {})}
    atomic_write_json(analysis_dir / "analysis_meta.json", batch_meta, backup=True)

    _append_manifest_row(
        Path(run_dir) / "analysis" / "batch_manifest.csv",
        row=record,
        dedupe_keys=["batch_id", "period"],
    )
    return batch_meta


def load_analysis_manifest(run_dir: str | Path) -> pd.DataFrame:
    manifest_path = Path(run_dir) / "analysis" / "batch_manifest.csv"
    if not manifest_path.exists():
        return pd.DataFrame(
            columns=[
                "batch_id",
                "period",
                "factor_count",
                "factor_metrics_path",
                "table_paths_json",
                "created_at_utc",
            ]
        )
    out = read_csv_with_backup(manifest_path)
    sort_cols = [c for c in ["period", "batch_id"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def compile_feedback_scoreboard(
    run_dir: str | Path,
    period: int | None = None,
    score_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    mining_manifest = load_mining_manifest(run_dir)
    analysis_manifest = load_analysis_manifest(run_dir)
    if mining_manifest.empty or analysis_manifest.empty:
        return pd.DataFrame()

    if period is not None and "period" in analysis_manifest.columns:
        analysis_manifest = analysis_manifest[analysis_manifest["period"] == int(period)].copy()
    if analysis_manifest.empty:
        return pd.DataFrame()

    expression_frames: list[pd.DataFrame] = []
    for row in mining_manifest.to_dict(orient="records"):
        path = str(row.get("expression_map_path", "") or "").strip()
        if not path:
            continue
        try:
            expr = load_saved_dataframe(path)
        except Exception:
            continue
        if expr.empty:
            continue
        expr = expr.copy()
        if "alpha_name" not in expr.columns:
            continue
        expr["batch_id"] = str(row.get("batch_id", ""))
        expression_frames.append(expr)

    metrics_frames: list[pd.DataFrame] = []
    for row in analysis_manifest.to_dict(orient="records"):
        path = str(row.get("factor_metrics_path", "") or "").strip()
        if not path:
            continue
        try:
            metrics = load_saved_dataframe(path)
        except Exception:
            continue
        if metrics.empty:
            continue
        metrics = metrics.copy()
        if "factor" in metrics.columns and "alpha_name" not in metrics.columns:
            metrics["alpha_name"] = metrics["factor"].astype(str)
        if "alpha_name" not in metrics.columns:
            continue
        metrics["batch_id"] = str(row.get("batch_id", ""))
        metrics["period"] = int(row.get("period", np.nan))
        metrics_frames.append(metrics)

    if not expression_frames or not metrics_frames:
        return pd.DataFrame()

    expr_df = pd.concat(expression_frames, ignore_index=True)
    metrics_df = pd.concat(metrics_frames, ignore_index=True)

    merged = pd.merge(
        metrics_df,
        expr_df[["batch_id", "alpha_name", "expression", "source"]],
        on=["batch_id", "alpha_name"],
        how="left",
    )
    merged = _add_composite_score(merged, weights=score_weights)
    merged = merged.sort_values(["composite_score", "ir", "ic_mean"], ascending=[False, False, False]).reset_index(
        drop=True
    )

    feedback_dir = Path(run_dir) / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    out_path = feedback_dir / "expression_scoreboard.csv"
    atomic_write_dataframe_csv(out_path, merged, index=False, backup=True)

    update_run_meta(
        run_dir,
        {
            "feedback_scoreboard_path": str(out_path.as_posix()),
            "feedback_last_updated_utc": _utc_now().isoformat(),
        },
    )
    return merged


def save_dataframe_artifact(
    df: pd.DataFrame,
    stem_path: str | Path,
    index: bool = False,
    preferred: str = "parquet",
    parquet_kwargs: dict[str, Any] | None = None,
) -> dict[str, str]:
    stem = Path(stem_path)
    if stem.suffix:
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)

    preferred = str(preferred or "parquet").lower()
    if preferred == "parquet":
        parquet_path = stem.with_suffix(".parquet")
        tmp_path: Path | None = None
        try:
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{parquet_path.name}.",
                suffix=".tmp.parquet",
                dir=str(parquet_path.parent),
            )
            os.close(fd)
            tmp_path = Path(tmp_name)
            write_parquet_compat(df, tmp_path, index=index, **dict(parquet_kwargs or {}))
            os.replace(tmp_path, parquet_path)
            return {"path": str(parquet_path.as_posix()), "format": "parquet"}
        except Exception:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
            pass

    if preferred in {"parquet", "pickle", "pkl"}:
        pkl_path = stem.with_suffix(".pkl")
        df.to_pickle(pkl_path)
        return {"path": str(pkl_path.as_posix()), "format": "pickle"}

    csv_path = stem.with_suffix(".csv")
    atomic_write_dataframe_csv(csv_path, df, index=index, backup=True)
    return {"path": str(csv_path.as_posix()), "format": "csv"}


def load_saved_dataframe(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(p)
    if ext == ".pkl":
        return pd.read_pickle(p)
    if ext == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported file extension for dataframe load: {p}")


def load_dataframe_from_stem(stem_path: str | Path) -> pd.DataFrame:
    stem = Path(stem_path)
    if stem.suffix:
        return load_saved_dataframe(stem)
    for ext in [".parquet", ".pkl", ".csv"]:
        p = stem.with_suffix(ext)
        if p.exists():
            return load_saved_dataframe(p)
    raise FileNotFoundError(stem)


def _append_manifest_row(manifest_path: Path, row: dict[str, Any], dedupe_keys: Iterable[str]) -> None:
    dedupe_keys = [str(k) for k in dedupe_keys]
    if manifest_path.exists():
        existing = read_csv_with_backup(manifest_path)
    else:
        existing = pd.DataFrame()

    appended = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    if dedupe_keys and not appended.empty:
        valid_keys = [k for k in dedupe_keys if k in appended.columns]
        if valid_keys:
            appended = appended.drop_duplicates(subset=valid_keys, keep="last")
    atomic_write_dataframe_csv(manifest_path, appended, index=False, backup=True)


def _add_composite_score(df: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    out = df.copy()
    default_weights = {
        "ir": 0.35,
        "ic_mean": 0.30,
        "long_short_sharpe_ratio": 0.20,
        "long_short_total_return": 0.10,
        "avg_turnover": -0.05,
    }
    active_weights = dict(default_weights)
    if weights:
        active_weights.update({str(k): float(v) for k, v in weights.items()})

    if "avg_turnover" not in out.columns:
        min_col = "avg_min_layer_turnover"
        max_col = "avg_max_layer_turnover"
        if min_col in out.columns or max_col in out.columns:
            out["avg_turnover"] = np.nanmean(
                np.column_stack(
                    [
                        pd.to_numeric(out[min_col], errors="coerce")
                        if min_col in out.columns
                        else np.full(len(out), np.nan),
                        pd.to_numeric(out[max_col], errors="coerce")
                        if max_col in out.columns
                        else np.full(len(out), np.nan),
                    ]
                ),
                axis=1,
            )

    score = np.zeros(len(out), dtype=np.float64)
    for col, w in active_weights.items():
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        std = float(values.std(skipna=True))
        if std > 0:
            z = (values - values.mean(skipna=True)) / std
        else:
            z = values * 0.0
        score = score + float(w) * np.asarray(z.fillna(0.0), dtype=np.float64)

    out["composite_score"] = score
    out["composite_rank"] = out["composite_score"].rank(method="dense", ascending=False).astype("Int64")
    return out


def _merge_dict(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
