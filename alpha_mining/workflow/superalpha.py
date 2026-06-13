from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

from ..history.registry import to_serializable
from .analysis_cycle import (
    BatchAnalysisConfig,
    build_factor_research_input,
    run_factor_analysis_batch,
)
from .artifacts import load_saved_dataframe, save_dataframe_artifact
from .factor_library import load_factor_library_registry
from .universe_store import (
    build_dashboard_factor_metrics,
    get_universe_paths,
    load_universe_base_frame,
    load_universe_expression_registry,
    load_universe_input_manifest,
)


SUPERALPHA_FACTOR = "superalpha"
ALLOWED_METADATA_WEIGHTS = {
    "score",
    "feedback_score",
    "long_only_sharpe",
    "long_short_sharpe",
}


class SuperalphaError(ValueError):
    """Raised for user-correctable Superalpha request errors."""


class SuperalphaBusyError(SuperalphaError):
    """Raised when another SA backtest is already running."""


@dataclass(frozen=True)
class SuperalphaConfig:
    max_components: int = 50
    component_normalization: str = "cs_zscore"
    final_normalization: str = "cs_zscore"
    weight_normalization: str = "sum_abs"
    component_join: str = "inner"
    direction_adjustment: bool = True
    allow_reproduce_fallback: bool = True
    cache_reproduced_components: bool = True
    schema_version: int = 2
    min_free_space_gb: float = 5.0
    # Runtime / resource config.
    python_tmp_subdir: str = "_tmp/python"
    duckdb_tmp_subdir: str = "_tmp/duckdb"
    component_tmp_subdir: str = "_tmp/components"
    duckdb_memory_limit: str = "2GB"
    duckdb_max_temp_directory_size: str = "50GB"
    duckdb_threads: str = ""
    min_system_drive_free_space_gb: float = 8.0
    enable_resource_diagnostics: bool = True
    component_cache_policy: str = "bounded"
    component_cache_max_size_gb: float = 2.0
    component_cache_max_files: int = 200
    component_cache_ttl_days: int = 14


@dataclass(frozen=True)
class ComboWeights:
    weights: list[float]
    basis: str
    expression: str


@dataclass(frozen=True)
class SuperalphaSignalResult:
    signal: pd.DataFrame
    components: list[dict[str, Any]]
    weights: ComboWeights
    coverage_before_join: float | None = None
    coverage_after_join: float | None = None
    extra_meta: dict[str, Any] = field(default_factory=dict)


def _check_disk_space(path: Path, min_gb: float) -> None:
    """Raise SuperalphaError if the disk containing `path` has less than `min_gb` free."""
    try:
        usage = shutil.disk_usage(str(path))
        free_gb = usage.free / (1024**3)
        if free_gb < min_gb:
            raise SuperalphaError(
                f"insufficient disk space for Superalpha backtest: "
                f"free={free_gb:.1f}GB, required={min_gb:.1f}GB, path={path}"
            )
    except OSError:
        pass


@contextmanager
def _sa_tmpdir_context(tmp_dir: Path) -> Iterator[None]:
    """Redirect Python TMP/TEMP/TMPDIR to *tmp_dir* during SA backtest.

    This ensures pyarrow, pandas, and other libraries that use tempfile
    write to the project directory instead of the system C: drive.
    The original environment variables are restored on exit.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env_keys = ("TMP", "TEMP", "TMPDIR")
    old_values: dict[str, str | None] = {}
    for key in env_keys:
        old_values[key] = os.environ.get(key)
        os.environ[key] = str(tmp_dir)
    try:
        yield
    finally:
        for key, old in old_values.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


# ---------------------------------------------------------------------------
# Runtime dirs, disk/memory diagnostics, and file lock.
# ---------------------------------------------------------------------------


def _superalpha_runtime_dirs(base_path: Path, universe_name: str, cfg: SuperalphaConfig) -> dict[str, Path]:
    root = (base_path / str(universe_name) / "superalphas").resolve()
    return {
        "root": root,
        "tmp_root": root / "_tmp",
        "python_tmp": root / cfg.python_tmp_subdir,
        "duckdb_tmp": root / cfg.duckdb_tmp_subdir,
        "component_tmp": root / cfg.component_tmp_subdir,
        "jobs": root / "_jobs",
        "locks": root / "_locks",
    }


def _check_system_drive_free_space(min_gb: float) -> None:
    """Check system drive (C: on Windows) free space. Warns about pagefile risk."""
    if min_gb <= 0:
        return
    try:
        system_drive = os.environ.get("SystemDrive", "C:") if os.name == "nt" else "/"
        usage = shutil.disk_usage(system_drive + "\\") if os.name == "nt" else shutil.disk_usage("/")
        free_gb = usage.free / (1024**3)
        if free_gb < min_gb:
            raise SuperalphaError(
                f"系统盘剩余空间不足 ({free_gb:.1f}GB < {min_gb:.1f}GB)。"
                f" SA backtest 内存峰值可能导致 Windows pagefile.sys 扩张并撑满系统盘。"
                f" 建议：关闭其他程序、扩大系统盘、或将 pagefile 移到非系统盘。"
            )
    except SuperalphaError:
        raise
    except Exception:
        pass


def _disk_snapshot(paths: dict[str, Path]) -> dict[str, Any]:
    """Return free/total/used GB for each logical path."""
    seen: dict[str, dict[str, Any]] = {}
    result: dict[str, Any] = {}
    for label, path in paths.items():
        try:
            resolved = path.resolve()
            anchor = str(resolved.anchor) or str(resolved)
            usage = shutil.disk_usage(str(resolved))
            info = {
                "path": str(path),
                "anchor": anchor,
                "total_gb": round(usage.total / (1024**3), 2),
                "used_gb": round(usage.used / (1024**3), 2),
                "free_gb": round(usage.free / (1024**3), 2),
            }
            result[label] = info
            if anchor not in seen:
                seen[anchor] = info
        except Exception:
            result[label] = {"path": str(path), "error": "unavailable"}
    return result


def _process_memory_snapshot() -> dict[str, Any]:
    """Return RSS/VMS if psutil is available; lightweight fallback otherwise."""
    try:
        import psutil

        proc = psutil.Process()
        mem = proc.memory_info()
        return {
            "available": True,
            "rss_mb": round(mem.rss / (1024**2), 1),
            "vms_mb": round(mem.vms / (1024**2), 1),
        }
    except ImportError:
        return {"available": False, "reason": "psutil_not_installed"}
    except Exception:
        return {"available": False, "reason": "error"}


def _safe_dir_size(path: Path, *, max_files: int = 200000) -> dict[str, Any]:
    """Return approximate dir size and file count."""
    if not path.exists():
        return {"exists": False, "size_mb": 0, "file_count": 0}
    total_bytes = 0
    file_count = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                try:
                    total_bytes += f.stat().st_size
                    file_count += 1
                    if file_count >= max_files:
                        break
                except Exception:
                    pass
    except Exception:
        pass
    return {
        "exists": True,
        "size_mb": round(total_bytes / (1024**2), 2),
        "file_count": file_count,
    }


def _write_resource_diagnostics(
    output_path: Path,
    *,
    universe: str,
    selected_count: int,
    cfg: SuperalphaConfig,
    runtime_dirs: dict[str, Path],
    snapshots: list[dict[str, Any]] | None = None,
    stage: str = "",
    error: str = "",
) -> None:
    """Write resource diagnostics JSON."""
    if not cfg.enable_resource_diagnostics:
        return
    try:
        data: dict[str, Any] = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "universe": universe,
            "selected_count": selected_count,
            "component_join": cfg.component_join,
            "allow_reproduce_fallback": cfg.allow_reproduce_fallback,
            "runtime_dirs": {k: str(v) for k, v in runtime_dirs.items()},
            "env_tmp": {k: os.environ.get(k, "") for k in ("TMP", "TEMP", "TMPDIR")},
            "duckdb_settings": {
                "temp_directory": str(runtime_dirs.get("duckdb_tmp", "")),
                "memory_limit": cfg.duckdb_memory_limit,
                "max_temp_directory_size": cfg.duckdb_max_temp_directory_size,
            },
            "stage": stage,
            "snapshots": snapshots or [],
        }
        if error:
            data["error"] = error
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(to_serializable(data), ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _is_pid_alive(pid: int) -> bool:
    """Check if a process is still running."""
    if os.name == "nt":
        import subprocess

        try:
            out = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {pid}"],
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return str(pid) in out.decode(errors="ignore")
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


@contextmanager
def _superalpha_run_lock(lock_dir: Path, *, stale_after_seconds: int = 43200) -> Iterator[None]:
    """File-based lock to prevent concurrent SA backtest runs.

    Uses os.mkdir() for atomic creation. Stale locks older than
    *stale_after_seconds* (default 12h) are automatically cleaned.
    Also checks if the owning PID is still alive.
    """
    lock_dir = Path(lock_dir)
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    owner_path = lock_dir / "owner.json"
    acquired = False
    try:
        if lock_dir.exists():
            stale = True
            if owner_path.exists():
                try:
                    owner = json.loads(owner_path.read_text(encoding="utf-8"))
                    created = float(owner.get("created_at_epoch", 0))
                    owner_pid = int(owner.get("pid", 0))
                    elapsed = datetime.now(timezone.utc).timestamp() - created if created > 0 else float("inf")
                    if elapsed < stale_after_seconds or (owner_pid > 0 and _is_pid_alive(owner_pid)):
                        raise SuperalphaBusyError(
                            f"another Superalpha backtest is running (pid={owner_pid}, "
                            f"started={owner.get('created_at_utc', '?')})"
                        )
                except SuperalphaBusyError:
                    raise
                except Exception:
                    stale = True
            else:
                stale = time.time() - lock_dir.stat().st_mtime >= stale_after_seconds
                if not stale:
                    raise SuperalphaBusyError("another Superalpha backtest is running")
            if stale:
                shutil.rmtree(lock_dir, ignore_errors=True)

        os.mkdir(lock_dir)
        acquired = True
        owner_path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "created_at_epoch": datetime.now(timezone.utc).timestamp(),
                }
            ),
            encoding="utf-8",
        )
        yield
    finally:
        if acquired:
            shutil.rmtree(lock_dir, ignore_errors=True)
    return

    lock_dir.mkdir(parents=True, exist_ok=True)
    owner_path = lock_dir / "owner.json"
    try:
        # Check for existing lock
        if owner_path.exists():
            try:
                owner = json.loads(owner_path.read_text(encoding="utf-8"))
                created = float(owner.get("created_at_epoch", 0))
                owner_pid = int(owner.get("pid", 0))
                elapsed = datetime.now(timezone.utc).timestamp() - created if created > 0 else float("inf")
                if elapsed < stale_after_seconds:
                    # Lock is within timeout — treat as active (raise busy)
                    raise SuperalphaBusyError(
                        f"另一个 Superalpha backtest 正在运行 (pid={owner_pid}, "
                        f"started={owner.get('created_at_utc', '?')})。请等待其完成后重试。"
                    )
                # Lock has exceeded timeout — check if PID is still alive
                if owner_pid > 0 and _is_pid_alive(owner_pid):
                    raise SuperalphaBusyError(
                        f"另一个 Superalpha backtest 正在运行 (pid={owner_pid}, "
                        f"started={owner.get('created_at_utc', '?')})。请等待其完成后重试。"
                    )
                # Stale lock with dead process — clean up
                for f in lock_dir.iterdir():
                    try:
                        f.unlink()
                    except Exception:
                        pass
            except (json.JSONDecodeError, KeyError):
                # Corrupt owner.json — treat as stale
                for f in lock_dir.iterdir():
                    try:
                        f.unlink()
                    except Exception:
                        pass
        # Create lock
        owner_path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "created_at_epoch": datetime.now(timezone.utc).timestamp(),
                }
            ),
            encoding="utf-8",
        )
        yield
    finally:
        try:
            for f in lock_dir.iterdir():
                try:
                    f.unlink()
                except Exception:
                    pass
            lock_dir.rmdir()
        except Exception:
            pass


def _dataframe_artifact_readable(path: Path) -> bool:
    """Return True if a dataframe artifact can be opened without loading it fully."""
    try:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            pd.read_parquet(path, columns=[])
            return True
        if suffix == ".csv":
            pd.read_csv(path, nrows=1)
            return True
        if suffix == ".pkl":
            pd.read_pickle(path)
            return True
        return path.exists()
    except Exception:
        return False


def _read_error_status(path: Path) -> dict[str, Any]:
    return {
        "signal_status": "read_error",
        "signal_available": False,
        "can_reproduce": False,
        "can_backtest": False,
        "signal_status_reason": f"signal artifact is not readable: {path}",
        "signal_source": str(path.as_posix()),
        "reproduce_source_mode": "",
        "strict_reproducibility": False,
        "reproduce_warning": "",
        "cache_path": "",
        "direction_warning": "",
    }


def _clean_optional_text(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value or "").strip()


def _component_cache_key(factor: str, row: dict[str, Any]) -> str:
    """Build a versioned component cache stem for reproduced signals."""
    safe_factor = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(factor or "").strip()) or "factor"
    payload = {
        "factor": str(factor or ""),
        "expression": str(row.get("expression", "") or ""),
        "expression_hash": str(row.get("expression_hash", "") or ""),
        "input_manifest_id": str(row.get("input_manifest_id", "") or ""),
        "manifest_id": str(row.get("manifest_id", "") or ""),
        "simulation_config_hash": str(row.get("simulation_config_hash", "") or ""),
        "simulation_config": row.get("simulation_config", None),
        "source_view": str(row.get("source_view", "") or ""),
        "source_path": str(row.get("source_path", "") or ""),
        "duckdb_path": str(row.get("duckdb_path", "") or ""),
    }
    return f"{safe_factor}_{_stable_hash(payload)[:16]}"


def _component_cache_candidates(base: Path, universe: str, factor: str, row: dict[str, Any]) -> list[Path]:
    cache_dir = base / universe / "superalphas" / "_component_cache"
    return [
        cache_dir / f"{_component_cache_key(factor, row)}.parquet",
        cache_dir / f"{factor}.parquet",
    ]


def _prune_component_cache(
    cache_dir: Path,
    *,
    max_size_gb: float,
    max_files: int,
    ttl_days: int,
) -> dict[str, Any]:
    """Prune component cache by TTL, file count, and total size."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return {
            "policy": "bounded",
            "deleted_files": 0,
            "deleted_bytes": 0,
            "remaining_files": 0,
            "remaining_bytes": 0,
        }

    now = time.time()
    ttl_seconds = max(0, int(ttl_days)) * 86400
    max_bytes = int(max(0.0, float(max_size_gb)) * (1024**3))
    deleted_files = 0
    deleted_bytes = 0

    def _delete(path: Path) -> None:
        nonlocal deleted_files, deleted_bytes
        try:
            size = path.stat().st_size
            path.unlink()
            deleted_files += 1
            deleted_bytes += size
        except Exception:
            pass

    if ttl_seconds > 0:
        for path in [p for p in cache_dir.glob("*") if p.is_file()]:
            try:
                if now - path.stat().st_mtime > ttl_seconds:
                    _delete(path)
            except Exception:
                pass

    files = [p for p in cache_dir.glob("*") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
    while max_files > 0 and len(files) > int(max_files):
        _delete(files.pop(0))

    files = [p for p in cache_dir.glob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files if p.exists())
    files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
    while max_bytes > 0 and total > max_bytes and files:
        path = files.pop(0)
        try:
            size = path.stat().st_size
        except Exception:
            size = 0
        _delete(path)
        total -= size

    remaining = [p for p in cache_dir.glob("*") if p.is_file()]
    remaining_bytes = sum(p.stat().st_size for p in remaining if p.exists())
    return {
        "policy": "bounded",
        "deleted_files": int(deleted_files),
        "deleted_bytes": int(deleted_bytes),
        "remaining_files": int(len(remaining)),
        "remaining_bytes": int(remaining_bytes),
    }


def _cleanup_superalpha_tmp_dirs(runtime_dirs: dict[str, Path]) -> dict[str, Any]:
    deleted_dirs: list[str] = []
    errors: list[str] = []
    for label in ("python_tmp", "duckdb_tmp", "component_tmp"):
        path = runtime_dirs.get(label)
        if not path:
            continue
        try:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
                deleted_dirs.append(label)
            path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    return {"deleted_dirs": deleted_dirs, "errors": errors}


def list_superalpha_components(
    *,
    base_dir: str | Path,
    universe_name: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Lightweight inspect: check paths, registry, manifest metadata without loading panels or reproducing."""
    registry = load_factor_library_registry(base_dir=base_dir, universe_name=universe_name)
    if registry.empty:
        return []
    work = registry.copy()
    work = work[work.get("status", pd.Series(dtype=str)).fillna("").astype(str).str.lower() == "accepted"].copy()
    if work.empty:
        return []
    work["universe"] = work.get("universe", str(universe_name)).fillna(str(universe_name))
    work = work[work["universe"].astype(str) == str(universe_name)].copy()
    if "score" in work.columns:
        work["_score_sort"] = pd.to_numeric(work["score"], errors="coerce")
    else:
        work["_score_sort"] = np.nan
    if "submitted_at_utc" in work.columns:
        work["_submitted_sort"] = pd.to_datetime(work["submitted_at_utc"], errors="coerce")
    else:
        work["_submitted_sort"] = pd.NaT
    work = work.sort_values(
        ["_score_sort", "_submitted_sort", "factor"],
        ascending=[False, False, True],
        na_position="last",
    )
    if limit is not None:
        work = work.head(max(0, int(limit)))

    base = Path(base_dir)
    universe = str(universe_name)

    # Load expression registry for can_reproduce check
    try:
        expr_registry = load_universe_expression_registry(base_dir=base, universe_name=universe)
    except Exception:
        expr_registry = pd.DataFrame()

    # Lightweight signal status inspection
    signal_statuses: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        factor = str(row.get("factor") or "")
        signal_path = _clean_optional_text(row.get("signal_artifact_path"))
        status = _inspect_signal_status(
            factor,
            signal_path,
            base_dir=base,
            universe_name=universe,
            expression_registry=expr_registry,
            row_metadata=row.to_dict(),
        )
        signal_statuses.append(status)

    work["signal_available"] = [s["signal_available"] for s in signal_statuses]
    work["signal_status"] = [s["signal_status"] for s in signal_statuses]
    work["signal_status_reason"] = [s["signal_status_reason"] for s in signal_statuses]
    work["can_reproduce"] = [s["can_reproduce"] for s in signal_statuses]
    work["can_backtest"] = [s["can_backtest"] for s in signal_statuses]
    work["signal_source"] = [s["signal_source"] for s in signal_statuses]
    work["reproduce_source_mode"] = [s["reproduce_source_mode"] for s in signal_statuses]
    work["strict_reproducibility"] = [s["strict_reproducibility"] for s in signal_statuses]
    work["reproduce_warning"] = [s["reproduce_warning"] for s in signal_statuses]
    work["cache_path"] = [s["cache_path"] for s in signal_statuses]

    # Lightweight direction inspection
    direction_results = []
    for _, row in work.iterrows():
        factor = str(row.get("factor") or "")
        dir_sign, dir_status, dir_warning = _resolve_direction_sign(
            factor, row.to_dict(), base_dir=base, universe_name=universe
        )
        direction_results.append(
            {
                "direction_sign": dir_sign,
                "direction_status": dir_status,
                "direction_warning": dir_warning,
            }
        )
    work["direction_sign"] = [d["direction_sign"] for d in direction_results]
    work["direction_status"] = [d["direction_status"] for d in direction_results]
    work["direction_warning"] = [d["direction_warning"] for d in direction_results]

    drop_cols = [col for col in ["_score_sort", "_submitted_sort"] if col in work.columns]
    return [_json_record(row) for row in work.drop(columns=drop_cols).to_dict(orient="records")]


def _inspect_signal_status(
    factor: str,
    signal_path: str,
    *,
    base_dir: Path,
    universe_name: str,
    expression_registry: pd.DataFrame | None = None,
    row_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Lightweight signal status check without loading data."""
    # Check compact signal
    if signal_path and signal_path.lower() not in {"nan", "none", "<na>"}:
        p = Path(signal_path)
        if p.exists():
            if not _dataframe_artifact_readable(p):
                return _read_error_status(p)
            return {
                "signal_status": "compact",
                "signal_available": True,
                "can_reproduce": False,
                "can_backtest": True,
                "signal_status_reason": "",
                "signal_source": str(p.as_posix()),
                "reproduce_source_mode": "compact",
                "strict_reproducibility": True,
                "reproduce_warning": "",
                "cache_path": "",
                "direction_warning": "",
            }

    # Check raw alpha
    raw_alpha_path = base_dir / universe_name / "alphas" / f"{factor}.parquet"
    if raw_alpha_path.exists():
        if not _dataframe_artifact_readable(raw_alpha_path):
            return _read_error_status(raw_alpha_path)
        return {
            "signal_status": "raw",
            "signal_available": True,
            "can_reproduce": False,
            "can_backtest": True,
            "signal_status_reason": "",
            "signal_source": str(raw_alpha_path.as_posix()),
            "reproduce_source_mode": "raw_alpha",
            "strict_reproducibility": True,
            "reproduce_warning": "",
            "cache_path": "",
            "direction_warning": "",
        }

    # Check component cache
    cache_path = next(
        (p for p in _component_cache_candidates(base_dir, universe_name, factor, row_metadata or {}) if p.exists()),
        None,
    )
    if cache_path is not None:
        if not _dataframe_artifact_readable(cache_path):
            return _read_error_status(cache_path)
        return {
            "signal_status": "cached",
            "signal_available": True,
            "can_reproduce": False,
            "can_backtest": True,
            "signal_status_reason": "",
            "signal_source": str(cache_path.as_posix()),
            "reproduce_source_mode": "cache",
            "strict_reproducibility": True,
            "reproduce_warning": "",
            "cache_path": str(cache_path.as_posix()),
            "direction_warning": "",
        }

    # Check expression registry for reproduce capability
    has_expression = False
    manifest_id = ""
    if (
        expression_registry is not None
        and not expression_registry.empty
        and "alpha_name" in expression_registry.columns
    ):
        expr_row = expression_registry[expression_registry["alpha_name"].astype(str) == factor]
        if not expr_row.empty:
            has_expression = True
            manifest_id = str(expr_row.iloc[-1].get("input_manifest_id", "") or "").strip()

    if not has_expression:
        return {
            "signal_status": "unavailable",
            "signal_available": False,
            "can_reproduce": False,
            "can_backtest": False,
            "signal_status_reason": _signal_artifact_reason(signal_path),
            "signal_source": "",
            "reproduce_source_mode": "",
            "strict_reproducibility": False,
            "reproduce_warning": "",
            "cache_path": "",
            "direction_warning": "",
        }

    # Expression exists — check manifest and source for strict vs fallback
    strict = False
    warning = ""
    status = "reproducible"
    can_backtest = True

    if manifest_id:
        paths = get_universe_paths(base_dir=base_dir, universe_name=universe_name)
        manifest_path = paths["input_manifest_dir"] / f"{manifest_id}.json"
        if manifest_path.exists():
            try:
                manifest = load_universe_input_manifest(
                    base_dir=base_dir,
                    universe_name=universe_name,
                    manifest_id=manifest_id,
                )
                snapshot_path = str(manifest.get("snapshot_path", "") or "").strip()
                source_path = str(manifest.get("source_path", "") or "").strip()
                if (snapshot_path and Path(snapshot_path).exists()) or (source_path and Path(source_path).exists()):
                    strict = True
                    can_backtest = True
                else:
                    # Check if manifest has DuckDB fallback info (reproduce can use it)
                    duckdb_path = str(manifest.get("duckdb_path", "") or "").strip()
                    source_view = str(manifest.get("source_view", "") or "").strip()
                    has_duckdb_info = bool(duckdb_path and source_view)
                    status = "duckdb_fallback"
                    warning = (
                        "snapshot/source missing; will use DuckDB fallback"
                        if has_duckdb_info
                        else "snapshot/source missing; no DuckDB fallback info"
                    )
            except Exception:
                status = "duckdb_fallback"
                warning = "manifest unreadable; will use DuckDB fallback"
        else:
            status = "duckdb_fallback"
            warning = "input_manifest file missing; using DuckDB fallback"
    else:
        status = "duckdb_fallback"
        warning = "input_manifest_id missing; using DuckDB fallback"

    return {
        "signal_status": status,
        "signal_available": False,
        "can_reproduce": True,
        "can_backtest": can_backtest,
        "signal_status_reason": "",
        "signal_source": "",
        "reproduce_source_mode": "",
        "strict_reproducibility": strict,
        "reproduce_warning": warning,
        "cache_path": "",
        "direction_warning": "",
    }


def parse_combo_expression(combo_expression: str, components: Sequence[dict[str, Any]]) -> ComboWeights:
    rows = list(components)
    if not rows:
        raise SuperalphaError("selected factors must not be empty")
    expr = str(combo_expression or "1").strip() or "1"
    if expr == "1":
        return ComboWeights(weights=[1.0 / len(rows)] * len(rows), basis="equal_weight", expression=expr)
    if expr in ALLOWED_METADATA_WEIGHTS:
        values = [_metadata_weight_value(row, expr) for row in rows]
        return ComboWeights(
            weights=_normalize_weights(values, clamp_negative=True),
            basis=expr,
            expression=expr,
        )
    if _looks_like_fixed_weights(expr):
        values = _parse_fixed_weights(expr)
        if len(values) != len(rows):
            raise SuperalphaError(
                f"fixed weight length must match selected factors length: {len(values)} != {len(rows)}"
            )
        return ComboWeights(weights=_normalize_weights(values), basis="fixed", expression=expr)
    raise SuperalphaError("unsupported combo expression; use 1, a fixed weight vector, or an allowed metadata weight")


def find_existing_superalpha_run(
    *,
    base_dir: str | Path,
    universe_name: str,
    selected_factor_ids: Sequence[str],
    combo_expression: str,
    config: SuperalphaConfig,
    analysis_config: BatchAnalysisConfig,
) -> dict[str, Any] | None:
    """Scan existing superalphas/*/meta.json and return matching meta before loading signals.

    This avoids expensive signal construction when the same combination has already been run.
    """
    base_path = Path(base_dir)
    superalphas_dir = base_path / str(universe_name) / "superalphas"
    if not superalphas_dir.exists():
        return None

    factor_ids = [str(x).strip() for x in selected_factor_ids if str(x).strip()]
    expr = str(combo_expression or "1").strip() or "1"
    if expr in ALLOWED_METADATA_WEIGHTS:
        components = list_superalpha_components(base_dir=base_dir, universe_name=universe_name)
        by_factor = {str(row.get("factor")): row for row in components}
        weight_rows = [by_factor.get(f, {"factor": f}) for f in factor_ids]
    else:
        weight_rows = [{"factor": f} for f in factor_ids]
    resolved_expr = parse_combo_expression(combo_expression, weight_rows).expression

    for entry in superalphas_dir.iterdir():
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        meta_path = entry / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(meta.get("universe", "")) != str(universe_name):
            continue
        if int(meta.get("component_count", 0)) != len(factor_ids):
            continue
        # Compare combo expression
        if str(meta.get("combo_expression", "")) != resolved_expr:
            continue
        # Compare component factors (order-sensitive)
        meta_factors = [str(c.get("factor", "")) for c in meta.get("components", [])]
        if meta_factors != factor_ids:
            continue
        # Compare config parameters
        for key, expected in [
            ("component_normalization", config.component_normalization),
            ("final_normalization", config.final_normalization),
            ("component_join", config.component_join),
            ("direction_adjustment", config.direction_adjustment),
            ("weight_normalization", config.weight_normalization),
        ]:
            if str(meta.get(key, "")) != str(expected):
                break
        else:
            # Compare analysis parameters
            if int(meta.get("period", 1)) != int(analysis_config.period):
                continue
            if int(meta.get("layers", 5)) != int(analysis_config.layers):
                continue
            return {
                "status": "cached",
                "superalpha_id": str(meta.get("superalpha_id", entry.name)),
                "meta": meta,
                "summary": dict(meta.get("summary") or {}),
                "artifact_path": str(entry.as_posix()),
                "cache_stage": "pre_signal",
            }
    return None


def _preload_raw_df_for_reproduce(
    base_dir: Path,
    universe_name: str,
    expressions: list[str],
    *,
    duckdb_settings_override: dict[str, Any] | None = None,
) -> pd.DataFrame | None:
    """Load raw_df once from manifest for all factors, avoiding per-factor DuckDB reloads.

    This mirrors closed-loop's source chunk loading approach: load data once,
    share across all expressions in the batch.

    *duckdb_settings_override* takes precedence over manifest settings, ensuring
    SA backtest forces temp_directory to the project disk.
    """
    try:
        from .universe_store import load_universe_input_manifest
        from ..datasource.loader import (
            collect_required_fields_from_expressions,
            load_panel_from_duckdb,
        )

        manifest = load_universe_input_manifest(base_dir=base_dir, universe_name=universe_name)
        payload = manifest.get("payload", manifest) if isinstance(manifest, dict) else {}

        source_backend = str(payload.get("source_backend", "") or "").strip().lower()
        duckdb_path = str(payload.get("duckdb_path", "") or "").strip()
        source_view = str(payload.get("source_view", "") or "").strip()

        if not (duckdb_path and source_view and source_backend.startswith("duckdb")):
            return None

        date_col = str(payload.get("date_col", "date"))
        code_col = str(payload.get("code_col", "code"))
        base_fields = [str(x) for x in payload.get("base_frame_cols", []) if str(x)]
        if not base_fields:
            base_fields = ["pct_chg", "circ_mv"]
        group_fields = [str(x) for x in payload.get("group_fields", []) if str(x)]

        required_fields = collect_required_fields_from_expressions(
            expressions=expressions,
            base_fields=base_fields,
            group_fields=group_fields,
        )

        date_range = payload.get("date_range", {}) if isinstance(payload.get("date_range", {}), dict) else {}
        start_date = str(date_range.get("start", "") or "").strip() or None
        end_date = str(date_range.get("end", "") or "").strip() or None

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

        # Override with SA runtime settings.
        if duckdb_settings_override:
            duckdb_settings.update(duckdb_settings_override)

        run_filters = payload.get("run_filters", {})
        if not isinstance(run_filters, dict):
            run_filters = {}

        raw_df = load_panel_from_duckdb(
            duckdb_path=duckdb_path,
            source_view=source_view,
            required_fields=required_fields,
            start_date=start_date,
            end_date=end_date,
            date_col=date_col,
            code_col=code_col,
            base_fields=base_fields,
            group_fields=group_fields,
            run_filters=run_filters,
            duckdb_settings=duckdb_settings or None,
            sort=False,
        )
        if raw_df is not None and not raw_df.empty:
            mem_mb = float(raw_df.memory_usage(deep=True).sum()) / (1024.0 * 1024.0)
            print(f"[superalpha] preload_raw_df rows={len(raw_df)} cols={len(raw_df.columns)} mem_mb={mem_mb:.1f}")
            return raw_df
    except Exception as exc:
        print(f"[superalpha][warn] preload_raw_df failed, will fall back to per-factor loading: {exc}")
    return None


def build_superalpha_signal(
    *,
    base_dir: str | Path,
    universe_name: str,
    selected_factor_ids: Sequence[str],
    combo_expression: str = "1",
    config: SuperalphaConfig | None = None,
    duckdb_settings_override: dict[str, Any] | None = None,
) -> SuperalphaSignalResult:
    cfg = config or SuperalphaConfig()
    factor_ids = [str(x).strip() for x in selected_factor_ids if str(x).strip()]
    if not factor_ids:
        raise SuperalphaError("selected factors must not be empty")
    if len(set(factor_ids)) != len(factor_ids):
        raise SuperalphaError("selected factors must be unique")
    if len(factor_ids) > int(cfg.max_components):
        raise SuperalphaError(f"selected factors must contain at most {int(cfg.max_components)} components")
    components = list_superalpha_components(base_dir=base_dir, universe_name=universe_name)
    by_factor = {str(row.get("factor")): row for row in components}
    selected: list[dict[str, Any]] = []
    for factor in factor_ids:
        row = by_factor.get(factor)
        if row is None:
            raise SuperalphaError(f"factor is not accepted in this universe: {factor}")
        selected.append(row)
    weights = parse_combo_expression(combo_expression, selected)

    # Preload raw_df once for all reproduce fallbacks (mirrors closed-loop source chunk loading)
    expressions_for_preload = [str(row.get("expression", "") or "") for row in selected]
    raw_df_cache: pd.DataFrame | None = None
    if False and cfg.allow_reproduce_fallback and any(expr for expr in expressions_for_preload):
        raw_df_cache = _preload_raw_df_for_reproduce(
            base_dir=Path(base_dir),
            universe_name=str(universe_name),
            expressions=expressions_for_preload,
            duckdb_settings_override=duckdb_settings_override,
        )

    base_path = Path(base_dir)
    frames: list[pd.DataFrame] = []
    concat_accumulator: pd.DataFrame | None = None
    component_rows: dict[str, int] = {}
    component_date_counts: dict[str, int] = {}
    resolved_components: list[dict[str, Any]] = []
    for row, weight in zip(selected, weights.weights, strict=True):
        factor = str(row.get("factor") or "")

        # Resolve signal with fallback
        raw, signal_meta = _resolve_component_signal(
            factor,
            row,
            base_dir=base_path,
            universe_name=universe_name,
            config=cfg,
            raw_df_cache=raw_df_cache,
            duckdb_settings_override=duckdb_settings_override,
        )
        if raw.empty:
            chain = signal_meta.get("_resolution_chain", [])
            raise SuperalphaError(
                f"unable to resolve signal for factor: {factor}; "
                f"resolution_chain: {' -> '.join(chain)}; "
                f"hint: rerun closed-loop to regenerate data, or check manifest/snapshot availability"
            )

        # Resolve direction
        direction_sign, direction_status, direction_warning = _resolve_direction_sign(
            factor,
            row,
            base_dir=base_path,
            universe_name=universe_name,
        )

        series = _normalise_signal_frame(raw, factor)
        if series.empty:
            raise SuperalphaError(f"signal artifact has no usable signal rows for factor: {factor}")

        # Direction adjustment
        if cfg.direction_adjustment:
            series["value"] = pd.to_numeric(series["value"], errors="coerce") * float(direction_sign)

        # Component normalization
        if cfg.component_normalization == "cs_zscore":
            series["value"] = _cross_sectional_zscore(series["value"], series["date"])
        series = series.dropna(subset=["value"])
        if series.empty:
            raise SuperalphaError(f"signal artifact is all missing/constant after normalization for factor: {factor}")

        series["weighted_value"] = pd.to_numeric(series["value"], errors="coerce") * float(weight)
        component_frame = series[["date", "code", "weighted_value"]]
        component_rows[factor] = len(component_frame)
        component_date_counts[factor] = component_frame["date"].nunique() if "date" in component_frame.columns else 0
        if cfg.component_join == "inner":
            frames.append(component_frame)
        else:
            if concat_accumulator is None:
                concat_accumulator = component_frame.copy()
            else:
                concat_accumulator = pd.concat([concat_accumulator, component_frame], ignore_index=True)
                concat_accumulator = concat_accumulator.groupby(["date", "code"], as_index=False)["weighted_value"].sum(
                    min_count=1
                )

        # Track component metadata
        comp_meta = dict(row)
        comp_meta.update(signal_meta)
        comp_meta["direction_sign"] = direction_sign
        comp_meta["direction_status"] = direction_status
        comp_meta["direction_warning"] = direction_warning
        comp_meta["_direction_info"] = {
            "sign": direction_sign,
            "status": direction_status,
            "warning": direction_warning,
        }
        resolved_components.append(comp_meta)

    # Release preloaded raw_df to reduce peak memory
    if raw_df_cache is not None:
        del raw_df_cache
        gc.collect()

    # Join frames — collect per-component diagnostics
    if not component_rows:
        for comp, frame in zip(resolved_components, frames, strict=True):
            factor_name = str(comp.get("factor", ""))
            component_rows[factor_name] = len(frame)
            component_date_counts[factor_name] = frame["date"].nunique() if "date" in frame.columns else 0

    if cfg.component_join == "inner" and len(frames) > 1:
        # Inner join: keep only date/code present in all components
        merged = frames[0].rename(columns={"weighted_value": "w_0"})
        for i, frame in enumerate(frames[1:], start=1):
            merged = merged.merge(
                frame.rename(columns={"weighted_value": f"w_{i}"}),
                on=["date", "code"],
                how="inner",
            )
        if merged.empty:
            raise SuperalphaError("inner join resulted in empty signal: no overlapping date/code across components")
        # Coverage tracking
        union_unique = len(pd.concat([f[["date", "code"]] for f in frames]).drop_duplicates())
        coverage_before = union_unique
        coverage_after = len(merged)
        coverage_ratio = coverage_after / union_unique if union_unique > 0 else 0.0
        # Sum weighted values
        w_cols = [c for c in merged.columns if c.startswith("w_")]
        merged[SUPERALPHA_FACTOR] = merged[w_cols].sum(axis=1, min_count=1)
        out = merged[["date", "code", SUPERALPHA_FACTOR]].dropna(subset=[SUPERALPHA_FACTOR])
    else:
        # Concat mode (default for single component or concat join)
        stacked = concat_accumulator if concat_accumulator is not None else pd.concat(frames, ignore_index=True)
        out = (
            stacked.groupby(["date", "code"], as_index=False)["weighted_value"]
            .sum(min_count=1)
            .rename(columns={"weighted_value": SUPERALPHA_FACTOR})
        )
        out = out.dropna(subset=[SUPERALPHA_FACTOR])
        coverage_before = len(out)
        coverage_after = len(out)
        coverage_ratio = 1.0

    # Final normalization
    if cfg.final_normalization == "cs_zscore":
        out[SUPERALPHA_FACTOR] = _cross_sectional_zscore(out[SUPERALPHA_FACTOR], out["date"])
        out = out.dropna(subset=[SUPERALPHA_FACTOR])

    out = out.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)
    return SuperalphaSignalResult(
        signal=out,
        components=resolved_components,
        weights=weights,
        coverage_before_join=coverage_before,
        coverage_after_join=coverage_after,
        extra_meta={
            "component_rows": component_rows,
            "component_date_counts": component_date_counts,
            "post_join_rows": coverage_after,
            "post_join_date_count": int(out["date"].nunique()) if "date" in out.columns and not out.empty else 0,
            "coverage_ratio": coverage_ratio,
            "join_method": cfg.component_join,
        },
    )


def run_superalpha_backtest(
    *,
    base_dir: str | Path,
    universe_name: str,
    selected_factor_ids: Sequence[str],
    combo_expression: str = "1",
    name: str = "",
    rerun: bool = False,
    config: SuperalphaConfig | None = None,
    analysis_config: BatchAnalysisConfig | None = None,
) -> dict[str, Any]:
    cfg = config or SuperalphaConfig()
    base_path = Path(base_dir)

    # Default to "light" analysis mode (needed for pre-signal cache lookup)
    if analysis_config is not None:
        analysis_cfg = analysis_config
    else:
        from .analysis_cycle import AnalysisLevelConfig

        analysis_cfg = BatchAnalysisConfig(analysis_level=AnalysisLevelConfig(mode="light"))

    # --- Pre-signal cache lookup ---
    if not rerun:
        cached = find_existing_superalpha_run(
            base_dir=base_dir,
            universe_name=universe_name,
            selected_factor_ids=selected_factor_ids,
            combo_expression=combo_expression,
            config=cfg,
            analysis_config=analysis_cfg,
        )
        if cached is not None:
            print(f"[superalpha] pre-signal cache hit: {cached.get('superalpha_id')}")
            return cached

    # --- Runtime dirs ---
    runtime_dirs = _superalpha_runtime_dirs(base_path, str(universe_name), cfg)
    for d in runtime_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # --- Disk preflight ---
    _check_system_drive_free_space(cfg.min_system_drive_free_space_gb)
    _check_disk_space(base_path / str(universe_name), cfg.min_free_space_gb)
    _check_disk_space(runtime_dirs["tmp_root"], cfg.min_free_space_gb)

    # --- DuckDB settings override ---
    duckdb_override: dict[str, Any] = {}
    if cfg.duckdb_memory_limit:
        duckdb_override["memory_limit"] = cfg.duckdb_memory_limit
    if cfg.duckdb_max_temp_directory_size:
        duckdb_override["max_temp_directory_size"] = cfg.duckdb_max_temp_directory_size
    duckdb_override["temp_directory"] = str(runtime_dirs["duckdb_tmp"].as_posix())
    if cfg.duckdb_threads:
        duckdb_override["threads"] = cfg.duckdb_threads

    # --- File lock ---
    with _superalpha_run_lock(runtime_dirs["locks"] / "backtest.lock"):
        # Redirect Python TMP/TEMP/TMPDIR to SA python_tmp
        with _sa_tmpdir_context(runtime_dirs["python_tmp"]):
            diagnostics_path = runtime_dirs["tmp_root"] / "last_failed_resource_meta.json"
            snapshots: list[dict[str, Any]] = []
            run_dir: Path | None = None
            run_dir_existed_before = False

            try:
                # Stage: before_signal
                snapshots.append(
                    {
                        "stage": "before_signal",
                        "disk": _disk_snapshot(
                            {
                                "system": Path(os.environ.get("SystemDrive", "C:") if os.name == "nt" else "/"),
                                "project": base_path / str(universe_name),
                                "tmp": runtime_dirs["tmp_root"],
                            }
                        ),
                        "process_memory": _process_memory_snapshot(),
                        "tmp_sizes": {k: _safe_dir_size(v) for k, v in runtime_dirs.items()},
                    }
                )

                signal_result = build_superalpha_signal(
                    base_dir=base_dir,
                    universe_name=universe_name,
                    selected_factor_ids=selected_factor_ids,
                    combo_expression=combo_expression,
                    config=cfg,
                    duckdb_settings_override=duckdb_override,
                )
                print(f"[superalpha] signal shape={signal_result.signal.shape}")
                gc.collect()

                # Stage: after_signal
                snapshots.append(
                    {
                        "stage": "after_signal",
                        "disk": _disk_snapshot({"tmp": runtime_dirs["tmp_root"]}),
                        "process_memory": _process_memory_snapshot(),
                    }
                )

                request_payload = {
                    "schema_version": int(cfg.schema_version),
                    "component_normalization": cfg.component_normalization,
                    "final_normalization": cfg.final_normalization,
                    "component_join": cfg.component_join,
                    "direction_adjustment": cfg.direction_adjustment,
                    "universe": str(universe_name),
                    "factors": [str(x) for x in selected_factor_ids],
                    "combo_expression": signal_result.weights.expression,
                    "weight_normalization": cfg.weight_normalization,
                    "period": int(analysis_cfg.period),
                    "layers": int(analysis_cfg.layers),
                    "component_fingerprints": [
                        comp.get("_signal_fingerprint", "") for comp in signal_result.components
                    ],
                    "direction_signs": [comp.get("direction_sign", 1) for comp in signal_result.components],
                    "direction_statuses": [comp.get("direction_status", "") for comp in signal_result.components],
                    "weight_basis": signal_result.weights.basis,
                    "resolved_weights": list(signal_result.weights.weights),
                }
                superalpha_id = f"superalpha_{_stable_hash(request_payload)[:16]}"
                run_dir = base_path / str(universe_name) / "superalphas" / superalpha_id
                meta_path = run_dir / "meta.json"
                if meta_path.exists() and not rerun:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    return {
                        "status": "cached",
                        "superalpha_id": superalpha_id,
                        "meta": meta,
                        "summary": dict(meta.get("summary") or {}),
                        "artifact_path": str(run_dir.as_posix()),
                        "cache_stage": "post_signal",
                    }

                run_dir_existed_before = run_dir.exists()
                run_dir.mkdir(parents=True, exist_ok=True)
                signal = signal_result.signal.copy()
                signal_save = save_dataframe_artifact(
                    signal,
                    run_dir / "superalpha_values",
                    preferred="parquet",
                    index=False,
                )
                (run_dir / "combo_expression.txt").write_text(signal_result.weights.expression, encoding="utf-8")
                components_df = _components_frame(signal_result.components, signal_result.weights.weights)
                components_df.to_csv(run_dir / "components.csv", index=False)

                base = load_universe_base_frame(base_dir=base_dir, universe_name=universe_name)
                if base.empty:
                    raise SuperalphaError(f"universe base frame is empty or missing: {universe_name}")
                print(f"[superalpha] base_frame shape={base.shape}")
                df_raw = build_factor_research_input(base, signal, date_col="date", code_col="code")
                del base, signal
                gc.collect()
                print(
                    f"[superalpha] df_raw shape={df_raw.shape} mem_mb={float(df_raw.memory_usage(deep=True).sum()) / (1024.0 * 1024.0):.1f}"
                )
                outputs = run_factor_analysis_batch(df_raw=df_raw, factor_cols=[SUPERALPHA_FACTOR], config=analysis_cfg)
                del df_raw
                gc.collect()

                # Stage: after_analysis
                snapshots.append(
                    {
                        "stage": "after_analysis",
                        "process_memory": _process_memory_snapshot(),
                    }
                )

                table_paths, factor_metrics_path = _save_analysis_outputs(
                    run_dir=run_dir,
                    outputs=outputs,
                    period=analysis_cfg.period,
                    layers=analysis_cfg.layers,
                )
                summary = _summary_from_metrics(
                    load_saved_dataframe(factor_metrics_path) if factor_metrics_path else pd.DataFrame()
                )
                created_at = _utc_now()
                cache_summary = (
                    _prune_component_cache(
                        runtime_dirs["root"] / "_component_cache",
                        max_size_gb=cfg.component_cache_max_size_gb,
                        max_files=cfg.component_cache_max_files,
                        ttl_days=cfg.component_cache_ttl_days,
                    )
                    if str(cfg.component_cache_policy).lower() == "bounded"
                    else {
                        "policy": str(cfg.component_cache_policy),
                        "deleted_files": 0,
                        "deleted_bytes": 0,
                    }
                )
                cleanup_summary = _cleanup_superalpha_tmp_dirs(runtime_dirs)
                component_status_counts: dict[str, int] = {}
                for comp in signal_result.components:
                    status = str(comp.get("signal_status", "") or "unknown")
                    component_status_counts[status] = component_status_counts.get(status, 0) + 1
                extra_meta = {
                    "superalpha": True,
                    "schema_version": int(cfg.schema_version),
                    "component_count": len(signal_result.components),
                    "component_normalization": cfg.component_normalization,
                    "final_normalization": cfg.final_normalization,
                    "component_join": cfg.component_join,
                    "direction_adjustment": cfg.direction_adjustment,
                    "weight_normalization": cfg.weight_normalization,
                    "weight_basis": signal_result.weights.basis,
                    "analysis_config": _analysis_config_summary(analysis_cfg),
                    "return_semantics": outputs.get("return_semantics") or {},
                    "phase_config": outputs.get("phase_meta") or {},
                    "direction_info": [comp.get("_direction_info", {}) for comp in signal_result.components],
                    "signal_sources": [comp.get("signal_source", "") for comp in signal_result.components],
                    "reproduce_modes": [comp.get("reproduce_source_mode", "") for comp in signal_result.components],
                    "coverage_before_join": signal_result.coverage_before_join,
                    "coverage_after_join": signal_result.coverage_after_join,
                    "resource_diagnostics_path": str((run_dir / "resource_meta.json").as_posix()),
                    "cleanup_summary": cleanup_summary,
                    "cache_summary": cache_summary,
                    "component_status_counts": component_status_counts,
                    "component_resolution_summary": [
                        {
                            "factor": str(comp.get("factor", "")),
                            "signal_status": str(comp.get("signal_status", "")),
                            "reproduce_source_mode": str(comp.get("reproduce_source_mode", "")),
                            "cache_path": str(comp.get("cache_path", "")),
                            "resolution_chain": comp.get("_resolution_chain", []),
                        }
                        for comp in signal_result.components
                    ],
                    **signal_result.extra_meta,
                }
                meta = {
                    "schema_version": int(cfg.schema_version),
                    "superalpha_id": superalpha_id,
                    "analysis_run_id": superalpha_id,
                    "name": str(name or superalpha_id),
                    "universe": str(universe_name),
                    "created_at_utc": created_at,
                    "combo_expression": signal_result.weights.expression,
                    "combo_expression_hash": _stable_hash({"combo_expression": signal_result.weights.expression}),
                    "component_count": len(signal_result.components),
                    "components": _json_records(components_df),
                    "component_normalization": cfg.component_normalization,
                    "final_normalization": cfg.final_normalization,
                    "component_join": cfg.component_join,
                    "direction_adjustment": cfg.direction_adjustment,
                    "weight_normalization": cfg.weight_normalization,
                    "weight_basis": signal_result.weights.basis,
                    "superalpha_values_path": signal_save["path"],
                    "status": "ok",
                    "summary": summary,
                    "period": int(analysis_cfg.period),
                    "layers": int(analysis_cfg.layers),
                    "is_timeseries": bool(analysis_cfg.is_timeseries),
                    "analysis_dir": str(run_dir.as_posix()),
                    "factor_metrics_path": str(factor_metrics_path.as_posix()) if factor_metrics_path else "",
                    "table_paths": {key: str(path.as_posix()) for key, path in table_paths.items()},
                    "alpha_names": [SUPERALPHA_FACTOR],
                    "extra_meta": extra_meta,
                }
                meta_path.write_text(
                    json.dumps(
                        to_serializable(meta),
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                (run_dir / "analysis_meta.json").write_text(
                    json.dumps(
                        to_serializable(meta),
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                # Write resource diagnostics on success
                _write_resource_diagnostics(
                    run_dir / "resource_meta.json",
                    universe=str(universe_name),
                    selected_count=len(selected_factor_ids),
                    cfg=cfg,
                    runtime_dirs=runtime_dirs,
                    snapshots=snapshots,
                    stage="done",
                )

                return {
                    "status": "ok",
                    "superalpha_id": superalpha_id,
                    "meta": meta,
                    "summary": summary,
                    "artifact_path": str(run_dir.as_posix()),
                    "resource_diagnostics_path": str((run_dir / "resource_meta.json").as_posix()),
                    "cleanup_summary": cleanup_summary,
                    "cache_summary": cache_summary,
                    "component_resolution_summary": extra_meta["component_resolution_summary"],
                    "component_status_counts": component_status_counts,
                }
            except Exception as exc:
                # Clean up newly created run_dir on failure
                try:
                    if run_dir is not None and not run_dir_existed_before and run_dir.exists():
                        shutil.rmtree(run_dir, ignore_errors=True)
                except Exception:
                    pass
                # Write failure diagnostics
                _write_resource_diagnostics(
                    diagnostics_path,
                    universe=str(universe_name),
                    selected_count=len(selected_factor_ids),
                    cfg=cfg,
                    runtime_dirs=runtime_dirs,
                    snapshots=snapshots,
                    stage="failed",
                    error=str(exc),
                )
                _cleanup_superalpha_tmp_dirs(runtime_dirs)
                raise


def _save_analysis_outputs(
    *,
    run_dir: Path,
    outputs: dict[str, Any],
    period: int,
    layers: int,
) -> tuple[dict[str, Path], Path | None]:
    table_paths: dict[str, Path] = {}
    factor_metrics = outputs.get("factor_metrics_df", pd.DataFrame())
    dashboard_metrics = build_dashboard_factor_metrics(
        factor_metrics,
        expression_registry_df=pd.DataFrame({"alpha_name": [SUPERALPHA_FACTOR], "expression": ["superalpha_combo"]}),
        period=int(period),
        layers=int(layers),
    )
    frame_map: dict[str, pd.DataFrame] = {
        "factor_metrics": factor_metrics,
        "dashboard_factor_metrics": dashboard_metrics,
        "factor_effectiveness_table": outputs.get("factor_effectiveness_table", pd.DataFrame()),
        "ic_yearly_df": outputs.get("ic_yearly_df", pd.DataFrame()),
        "ic_monthly_df": outputs.get("ic_monthly_df", pd.DataFrame()),
        "period_comparison_df": outputs.get("period_comparison_df", pd.DataFrame()),
        "double_sort_matrix_returns_df": outputs.get("double_sort_matrix_returns_df", pd.DataFrame()),
        "double_sort_spread_returns_df": outputs.get("double_sort_spread_returns_df", pd.DataFrame()),
        "double_sort_summary_df": outputs.get("double_sort_summary_df", pd.DataFrame()),
        "sample_split_metrics_df": outputs.get("sample_split_metrics_df", pd.DataFrame()),
        "phase_metrics_df": outputs.get("phase_metrics_df", pd.DataFrame()),
        "ic_df": outputs.get("ic_df", pd.DataFrame()),
        "portfolio_pnl_df": outputs.get("portfolio_pnl_df", pd.DataFrame()),
        "benchmark_pnl_df": outputs.get("benchmark_pnl_df", pd.DataFrame()),
        "analysis_distribution_histogram": outputs.get("analysis_distribution_histogram_df", pd.DataFrame()),
        "analysis_ic_decay": outputs.get("analysis_ic_decay_df", pd.DataFrame()),
        "analysis_factor_coverage_by_date": outputs.get("analysis_factor_coverage_by_date_df", pd.DataFrame()),
        "direction_policy_df": outputs.get("direction_policy_df", pd.DataFrame()),
        "phase_local_direction_df": outputs.get("phase_local_direction_df", pd.DataFrame()),
    }
    factor_metrics_path: Path | None = None
    for key, frame in frame_map.items():
        if not isinstance(frame, pd.DataFrame):
            continue
        preferred = (
            "parquet"
            if key
            in {
                "portfolio_pnl_df",
                "benchmark_pnl_df",
                "analysis_factor_coverage_by_date",
            }
            else "csv"
        )
        saved = save_dataframe_artifact(frame, run_dir / key, preferred=preferred, index=False)
        path = Path(saved["path"])
        if key == "factor_metrics":
            factor_metrics_path = path
        else:
            table_paths[key] = path
    return table_paths, factor_metrics_path


def _summary_from_metrics(metrics: pd.DataFrame) -> dict[str, Any]:
    if metrics.empty:
        return {}
    row = metrics.iloc[0].to_dict()
    keys = [
        "score_total",
        "score_total_net",
        "score_total_gross",
        "feedback_score",
        "feedback_score_net",
        "feedback_score_gross",
        "long_only_sharpe_ratio",
        "long_short_sharpe_ratio",
        "ic_mean",
        "ir",
        "effectiveness_tier",
    ]
    return {key: _json_value(row.get(key)) for key in keys if key in row}


def _components_frame(components: Sequence[dict[str, Any]], weights: Sequence[float]) -> pd.DataFrame:
    rows = []
    for row, weight in zip(components, weights, strict=True):
        out = dict(row)
        out["weight"] = float(weight)
        rows.append(out)
    return pd.DataFrame(rows)


def _analysis_config_summary(config: BatchAnalysisConfig) -> dict[str, Any]:
    return {
        "period": int(config.period),
        "layers": int(config.layers),
        "is_timeseries": bool(config.is_timeseries),
        "return_col": str(config.return_col),
        "signal_delay": int(config.signal_delay),
        "feedback_phase": str(config.feedback_phase),
        "apply_tradability_constraints": bool(config.apply_tradability_constraints),
        "transaction_cost_config": to_serializable(config.transaction_cost_config),
    }


def _normalise_signal_frame(frame: pd.DataFrame, factor: str) -> pd.DataFrame:
    work = frame.copy()
    if factor not in work.columns:
        value_cols = [c for c in work.columns if c not in {"date", "trade_date", "code", "znz_code"}]
        if len(value_cols) == 1:
            work = work.rename(columns={value_cols[0]: factor})
    if factor not in work.columns:
        return pd.DataFrame(columns=["date", "code", "value"])
    date_col = "date" if "date" in work.columns else "trade_date" if "trade_date" in work.columns else ""
    code_col = "code" if "code" in work.columns else "znz_code" if "znz_code" in work.columns else ""
    if not date_col or not code_col:
        return pd.DataFrame(columns=["date", "code", "value"])
    out = work[[date_col, code_col, factor]].copy()
    out.columns = ["date", "code", "value"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["code"] = out["code"].astype(str)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["date", "code", "value"])


def _cross_sectional_zscore(values: pd.Series, dates: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    grouped = numeric.groupby(dates)
    mean = grouped.transform("mean")
    std = grouped.transform(lambda s: float(s.std(ddof=0)) if len(s.dropna()) > 1 else np.nan)
    z = (numeric - mean) / std.replace(0.0, np.nan)
    return z.astype("float32")


def _parse_fixed_weights(expr: str) -> list[float]:
    text = expr.strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except Exception as exc:
            raise SuperalphaError("invalid fixed weight vector") from exc
        if not isinstance(parsed, list):
            raise SuperalphaError("fixed weight vector must be a list")
        values = parsed
    else:
        values = [part.strip() for part in text.split(",")]
    out: list[float] = []
    for value in values:
        try:
            parsed = float(value)
        except Exception as exc:
            raise SuperalphaError("fixed weights must be numeric") from exc
        if not np.isfinite(parsed):
            raise SuperalphaError("fixed weights must be finite")
        out.append(parsed)
    return out


def _looks_like_fixed_weights(expr: str) -> bool:
    text = expr.strip()
    if text.startswith("[") and text.endswith("]"):
        return True
    return "," in text and bool(re.fullmatch(r"[\s,\-+0-9.eE]+", text))


def _signal_artifact_available(path_value: Any) -> bool:
    text = str(path_value or "").strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return False
    return Path(text).exists()


def _signal_artifact_reason(path_value: Any) -> str:
    text = str(path_value or "").strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return "missing_signal_artifact"
    if not Path(text).exists():
        return "signal_artifact_not_found"
    return ""


def _resolve_component_signal(
    factor: str,
    row: dict[str, Any],
    *,
    base_dir: Path,
    universe_name: str,
    config: SuperalphaConfig,
    raw_df_cache: pd.DataFrame | None = None,
    duckdb_settings_override: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """多级 fallback 信号解析。

    Fallback 顺序:
    1. compact signal_artifact_path
    2. raw alpha parquet (alphas/{factor}.parquet)
    3. superalphas/_component_cache/{factor}.parquet
    4. reproduce_alpha_by_name (if allow_reproduce_fallback)
    5. DuckDB fallback (load_universe_alpha_values)
    """
    base = Path(base_dir)
    universe = str(universe_name)
    resolution_chain: list[str] = []

    signal_path = _clean_optional_text(row.get("signal_artifact_path"))

    # Level 1: compact signal
    if signal_path and signal_path.lower() not in {"nan", "none", "<na>"}:
        p = Path(signal_path)
        if p.exists():
            if not _dataframe_artifact_readable(p):
                meta = _read_error_status(p)
                meta.update(
                    {
                        "_signal_fingerprint": "",
                        "_resolution_chain": resolution_chain + ["compact_read_error"],
                    }
                )
                return pd.DataFrame(), meta
            try:
                raw = load_saved_dataframe(p)
                if not raw.empty:
                    resolution_chain.append("compact_ok")
                    meta = {
                        "signal_status": "compact",
                        "signal_available": True,
                        "can_reproduce": False,
                        "can_backtest": True,
                        "signal_status_reason": "",
                        "signal_source": str(p.as_posix()),
                        "reproduce_source_mode": "compact",
                        "strict_reproducibility": True,
                        "reproduce_warning": "",
                        "cache_path": "",
                        "_signal_fingerprint": _stable_hash(
                            {
                                "path": str(p.as_posix()),
                                "mtime": p.stat().st_mtime,
                                "size": p.stat().st_size,
                            }
                        ),
                        "_resolution_chain": resolution_chain,
                    }
                    return raw, meta
                resolution_chain.append("compact_empty")
            except Exception:
                resolution_chain.append("compact_read_error")
        else:
            resolution_chain.append("compact_missing")
    else:
        resolution_chain.append("compact_missing")

    # Level 2: raw alpha parquet
    raw_alpha_path = base / universe / "alphas" / f"{factor}.parquet"
    if raw_alpha_path.exists():
        if not _dataframe_artifact_readable(raw_alpha_path):
            meta = _read_error_status(raw_alpha_path)
            meta.update(
                {
                    "_signal_fingerprint": "",
                    "_resolution_chain": resolution_chain + ["raw_read_error"],
                }
            )
            return pd.DataFrame(), meta
        try:
            raw = load_saved_dataframe(raw_alpha_path)
            if not raw.empty:
                resolution_chain.append("raw_ok")
                meta = {
                    "signal_status": "raw",
                    "signal_available": True,
                    "can_reproduce": False,
                    "can_backtest": True,
                    "signal_status_reason": "",
                    "signal_source": str(raw_alpha_path.as_posix()),
                    "reproduce_source_mode": "raw_alpha",
                    "strict_reproducibility": True,
                    "reproduce_warning": "",
                    "cache_path": "",
                    "_signal_fingerprint": _stable_hash(
                        {
                            "path": str(raw_alpha_path.as_posix()),
                            "mtime": raw_alpha_path.stat().st_mtime,
                            "size": raw_alpha_path.stat().st_size,
                        }
                    ),
                    "_resolution_chain": resolution_chain,
                }
                return raw, meta
            resolution_chain.append("raw_empty")
        except Exception:
            resolution_chain.append("raw_read_error")
    else:
        resolution_chain.append("raw_missing")

    # Level 3: component cache
    cache_path = next(
        (p for p in _component_cache_candidates(base, universe, factor, row) if p.exists()),
        None,
    )
    if cache_path is not None:
        if not _dataframe_artifact_readable(cache_path):
            resolution_chain.append("cache_read_error")
        else:
            try:
                raw = load_saved_dataframe(cache_path)
                if not raw.empty:
                    resolution_chain.append("cache_ok")
                    meta = {
                        "signal_status": "cached",
                        "signal_available": True,
                        "can_reproduce": False,
                        "can_backtest": True,
                        "signal_status_reason": "",
                        "signal_source": str(cache_path.as_posix()),
                        "reproduce_source_mode": "cache",
                        "strict_reproducibility": True,
                        "reproduce_warning": "",
                        "cache_path": str(cache_path.as_posix()),
                        "_signal_fingerprint": _stable_hash(
                            {
                                "path": str(cache_path.as_posix()),
                                "mtime": cache_path.stat().st_mtime,
                                "size": cache_path.stat().st_size,
                            }
                        ),
                        "_resolution_chain": resolution_chain,
                    }
                    return raw, meta
                resolution_chain.append("cache_empty")
            except Exception:
                resolution_chain.append("cache_read_error")
    elif False:
        try:
            raw = load_saved_dataframe(cache_path)
            if not raw.empty:
                resolution_chain.append("cache_ok")
                meta = {
                    "signal_status": "cached",
                    "signal_available": True,
                    "can_reproduce": False,
                    "can_backtest": True,
                    "signal_status_reason": "",
                    "signal_source": str(cache_path.as_posix()),
                    "reproduce_source_mode": "cache",
                    "strict_reproducibility": True,
                    "reproduce_warning": "",
                    "cache_path": str(cache_path.as_posix()),
                    "_signal_fingerprint": _stable_hash(
                        {
                            "path": str(cache_path.as_posix()),
                            "mtime": cache_path.stat().st_mtime,
                            "size": cache_path.stat().st_size,
                        }
                    ),
                    "_resolution_chain": resolution_chain,
                }
                return raw, meta
            resolution_chain.append("cache_empty")
        except Exception:
            resolution_chain.append("cache_read_error")
    else:
        resolution_chain.append("cache_missing")

    # Level 4: reproduce fallback
    if config.allow_reproduce_fallback:
        try:
            from .reproduce import reproduce_alpha_by_name

            result = reproduce_alpha_by_name(
                alpha_name=factor,
                base_dir=base_dir,
                universe_name=universe_name,
                raw_df=raw_df_cache,
                duckdb_settings_override=duckdb_settings_override,
                compare_with_saved=False,
                mark_lifecycle=False,
            )
            output_df = result.get("output_df")
            if output_df is not None and not output_df.empty:
                resolution_chain.append("reproduce_ok")
                # Cache if configured
                cache_write_path = ""
                if config.cache_reproduced_components:
                    cache_dir = base / universe / "superalphas" / "_component_cache"
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    saved = save_dataframe_artifact(
                        output_df,
                        cache_dir / _component_cache_key(factor, row),
                        preferred="parquet",
                        index=False,
                    )
                    cache_write_path = str(saved["path"])

                meta = {
                    "signal_status": "reproduced",
                    "signal_available": True,
                    "can_reproduce": False,
                    "can_backtest": True,
                    "signal_status_reason": "",
                    "signal_source": result.get("saved", {}).get("path", "") if result.get("saved") else "",
                    "reproduce_source_mode": str(result.get("reproduce_source_mode", "unknown")),
                    "strict_reproducibility": bool(result.get("strict_reproducibility", False)),
                    "reproduce_warning": str(result.get("reproduce_warning", "")),
                    "cache_path": cache_write_path,
                    "_signal_fingerprint": _stable_hash(
                        {
                            "factor": factor,
                            "mode": "reproduce",
                            "cache_key": _component_cache_key(factor, row),
                        }
                    ),
                    "_resolution_chain": resolution_chain,
                }
                return output_df, meta
            resolution_chain.append("reproduce_empty_output")
        except Exception as exc:
            resolution_chain.append(f"reproduce_failed: {type(exc).__name__}")
    else:
        resolution_chain.append("reproduce_disabled")

    # All failed
    chain_summary = " -> ".join(resolution_chain)
    empty_meta = {
        "signal_status": "unavailable",
        "signal_available": False,
        "can_reproduce": False,
        "can_backtest": False,
        "signal_status_reason": f"resolve failed: {chain_summary}",
        "signal_source": "",
        "reproduce_source_mode": "",
        "strict_reproducibility": False,
        "reproduce_warning": "",
        "cache_path": "",
        "_signal_fingerprint": "",
        "_resolution_chain": resolution_chain,
    }
    return pd.DataFrame(), empty_meta


_DIRECTION_LABEL_MAP: dict[str, int] = {
    "top": 1,
    "long": 1,
    "positive": 1,
    "+1": 1,
    "1": 1,
    "bottom": -1,
    "short": -1,
    "negative": -1,
    "-1": -1,
}


def _parse_direction_value(val: Any) -> int | None:
    """Parse a direction value from various formats."""
    if val is None:
        return None
    s = str(val).strip().lower()
    if not s:
        return None
    # Try numeric first
    try:
        num = int(float(s))
        if num in (1, -1):
            return num
    except (ValueError, TypeError):
        pass
    # Try label map
    return _DIRECTION_LABEL_MAP.get(s)


def _resolve_direction_sign(
    factor: str,
    row: dict[str, Any],
    *,
    base_dir: Path,
    universe_name: str,
) -> tuple[int, str, str]:
    """解析 direction_sign。

    优先级:
    1. registry 中的 direction_sign
    2. 源 run metrics: direction_sign > direction_policy > best_layer_direction_train_locked
    3. 默认 +1

    Returns:
        (direction_sign, direction_status, direction_warning)
    """
    # Level 1: registry
    parsed = _parse_direction_value(row.get("direction_sign"))
    if parsed is not None:
        return parsed, "registry", ""

    # Level 2: source run metrics
    source_run_id = str(row.get("analysis_run_id") or "").strip()
    if source_run_id:
        base = Path(base_dir)
        universe = str(universe_name)
        for pattern in [
            f"{universe}/analysis/period_*/analysis_{source_run_id}/factor_metrics.csv",
            f"{universe}/analysis/period_*/analysis_{source_run_id}/factor_metrics.parquet",
        ]:
            for path in sorted(base.glob(pattern)):
                try:
                    metrics = load_saved_dataframe(path)
                    if not metrics.empty and "factor" in metrics.columns:
                        factor_row = metrics[metrics["factor"].astype(str) == factor]
                        if not factor_row.empty:
                            row_data = factor_row.iloc[-1].to_dict()
                            # Priority: direction_sign > direction_policy > best_layer_direction
                            for col, status in [
                                ("direction_sign", "source_factor_metrics"),
                                ("direction_policy", "source_direction_policy"),
                                (
                                    "best_layer_direction_train_locked",
                                    "parsed_best_layer_direction",
                                ),
                            ]:
                                parsed = _parse_direction_value(row_data.get(col))
                                if parsed is not None:
                                    return parsed, status, ""
                except Exception:
                    pass

    # Level 3: default
    return 1, "missing_default_positive", "direction_sign missing; defaulted to +1"


def _normalize_weights(values: Sequence[float], *, clamp_negative: bool = False) -> list[float]:
    """Normalize weights.

    Args:
        clamp_negative: If True (for metadata weights), clamp negatives to 0.
                        If False (for fixed weights), allow negatives and normalize by sum(abs).
    """
    weights = [float(x) if np.isfinite(float(x)) else 0.0 for x in values]
    if clamp_negative:
        weights = [max(0.0, x) for x in weights]
    denom = float(sum(abs(x) for x in weights))
    if denom <= 0:
        raise SuperalphaError("weights are all zero after parsing")
    return [x / denom for x in weights]


def _metadata_weight_value(row: dict[str, Any], key: str) -> float:
    aliases = {
        "feedback_score": ["feedback_score", "score"],
        "long_only_sharpe": [
            "candidate_long_only_sharpe",
            "long_only_sharpe",
            "long_only_sharpe_ratio",
        ],
        "long_short_sharpe": [
            "candidate_long_short_sharpe",
            "long_short_sharpe",
            "long_short_sharpe_ratio",
        ],
        "score": ["score"],
    }
    for col in aliases.get(key, [key]):
        value = row.get(col)
        try:
            parsed = float(value)
        except Exception:
            continue
        if np.isfinite(parsed):
            return parsed
    return 0.0


def _stable_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(
        to_serializable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_json_record(row) for row in frame.to_dict(orient="records")]


def _json_record(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _json_value(value) for key, value in row.items()}


def _json_value(value: Any) -> Any:
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value
