from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alpha_mining.atomic_io import atomic_write_json

from .closed_loop_params import SAFE_SOURCE_CHUNK_HARD_LIMIT_MB


JOB_ROOT_NAME = "_dashboard_jobs"
_PROCESSES: dict[str, subprocess.Popen] = {}
_LOCK = threading.Lock()


class ClosedLoopJobConflict(RuntimeError):
    def __init__(self, message: str, running_job: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.running_job = running_job or {}


def create_closed_loop_job(
    *,
    store_root: str | Path,
    body: dict[str, Any],
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(store_root)
    repo = Path(project_root) if project_root is not None else Path.cwd()
    params = _normalise_params(dict(body.get("params") or body or {}), store_root=root)
    _validate_params(params, repo=repo)
    universe = str(params.get("universe") or "cn_all")
    with _LOCK:
        running = _running_job_for_universe(root, universe)
        if running:
            raise ClosedLoopJobConflict(f"closed-loop job already running for universe {universe}", running)
        job_id = f"cl_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        job_dir = _job_dir(root, job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        command = _build_command(params, repo=repo)
        (job_dir / "request.json").write_text(
            json.dumps({"params": params}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (job_dir / "command.json").write_text(
            json.dumps({"command": command}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        stdout = (job_dir / "stdout.log").open("ab")
        stderr = (job_dir / "stderr.log").open("ab")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(
            command,
            cwd=str(repo),
            stdout=stdout,
            stderr=stderr,
            creationflags=creationflags,
        )
        _PROCESSES[job_id] = proc
        job = {
            "job_id": job_id,
            "status": "running",
            "pid": int(getattr(proc, "pid", 0) or 0),
            "universe": universe,
            "created_at_utc": _utc_now(),
            "started_at_utc": _utc_now(),
            "ended_at_utc": "",
            "exit_code": None,
            "params": params,
            "command": command,
            "job_dir": str(job_dir.as_posix()),
            "stdout_path": str((job_dir / "stdout.log").as_posix()),
            "stderr_path": str((job_dir / "stderr.log").as_posix()),
            "result_summary": {},
        }
        job["lock_owner"] = _lock_owner(root, universe)
        _write_job(job_dir, job)
        thread = threading.Thread(
            target=_monitor_job,
            args=(root, job_id, stdout, stderr),
            daemon=True,
            name=f"closed-loop-job-{job_id}",
        )
        thread.start()
        return _job_response(root, job)


def list_closed_loop_jobs(*, store_root: str | Path, limit: int = 50) -> dict[str, Any]:
    root = Path(store_root)
    jobs = [_refresh_job(root, path.parent.name) for path in _jobs_root(root).glob("*/job.json")]
    jobs = [job for job in jobs if job]
    jobs.sort(key=lambda row: str(row.get("created_at_utc", "")), reverse=True)
    return {
        "status": "ok",
        "total": len(jobs),
        "jobs": [_compact_job(row) for row in jobs[: max(1, int(limit))]],
    }


def get_closed_loop_job(*, store_root: str | Path, job_id: str) -> dict[str, Any]:
    root = Path(store_root)
    job = _refresh_job(root, job_id)
    if not job:
        raise KeyError(job_id)
    return _job_response(root, job)


def cancel_closed_loop_job(*, store_root: str | Path, job_id: str) -> dict[str, Any]:
    root = Path(store_root)
    job = _refresh_job(root, job_id)
    if not job:
        raise KeyError(job_id)
    with _LOCK:
        proc = _PROCESSES.get(job_id)
        if proc is not None and proc.poll() is None:
            proc.terminate()
        job["status"] = "cancelled"
        job["ended_at_utc"] = _utc_now()
        job["exit_code"] = -15
        job["failure_category"] = "cancelled"
        _write_job(_job_dir(root, job_id), job)
    return _job_response(root, job)


def _normalise_params(params: dict[str, Any], *, store_root: Path) -> dict[str, Any]:
    out = dict(params)
    out.setdefault("source_backend", "duckdb")
    out.setdefault("datasource_config", "configs/datasource.local.yaml")
    out.setdefault("duckdb_path", "data/duckdb/market.duckdb")
    out.setdefault("source_view", "v_project_panel_cn_a")
    out.setdefault("base_dir", str(store_root.as_posix()))
    out.setdefault("request_new", 5)
    out.setdefault("batch_size", 5)
    out.setdefault("max_eval", 80)
    out.setdefault("iterations", 1)
    out.setdefault("source_chunk_loading", True)
    out.setdefault("source_chunk_mem_warn_mb", 2560)
    out.setdefault("source_chunk_mem_hard_limit_mb", SAFE_SOURCE_CHUNK_HARD_LIMIT_MB)
    out.setdefault("candidate_artifact_retention_enabled", True)
    out.setdefault("run_health_retention_enabled", True)
    return out


def _validate_params(params: dict[str, Any], *, repo: Path) -> None:
    request_new = _int(params.get("request_new"), 5)
    batch_size = _int(params.get("batch_size"), 5)
    max_eval = _int(params.get("max_eval"), 80)
    iterations = _int(params.get("iterations"), 1)
    layer_max_candidates = _int(params.get("layer_max_candidates"), 0)
    hard_limit = float(params.get("source_chunk_mem_hard_limit_mb") or 0)
    if batch_size > request_new:
        raise ValueError("batch_size must be less than or equal to request_new")
    if max_eval < 1 or max_eval > 5000:
        raise ValueError("max_eval must be between 1 and 5000")
    if iterations < 1 or iterations > 20:
        raise ValueError("iterations must be between 1 and 20")
    if layer_max_candidates > 20000:
        raise ValueError("layer_max_candidates must be <= 20000")
    if hard_limit <= 0:
        raise ValueError("source_chunk_mem_hard_limit_mb must be non-zero for dashboard jobs")
    temp_dir = str(params.get("duckdb_temp_directory") or "").strip()
    if temp_dir:
        temp_path = Path(temp_dir)
        if temp_path.is_absolute():
            try:
                temp_path.resolve().relative_to(repo.resolve())
            except ValueError as exc:
                raise ValueError("duckdb_temp_directory must be inside the project workspace or left empty") from exc


def _build_command(params: dict[str, Any], *, repo: Path) -> list[str]:
    python_path = repo / ".venv" / "Scripts" / "python.exe"
    executable = str(python_path) if python_path.exists() else sys.executable
    command = [executable, "scripts\\run_closed_loop.py"]
    for key, value in params.items():
        if key in {
            "benchmark_enabled",
            "transaction_cost_enabled",
            "source_chunk_loading",
            "candidate_artifact_retention_enabled",
            "analysis_artifact_retention_enabled",
            "run_health_retention_enabled",
        }:
            command.extend(_bool_flag(key, bool(value)))
        elif isinstance(value, bool):
            if value:
                command.append(f"--{key.replace('_', '-')}")
        elif value is not None and str(value).strip() != "":
            command.extend([f"--{key.replace('_', '-')}", str(value)])
    return command


def _bool_flag(key: str, value: bool) -> list[str]:
    mapping = {
        "benchmark_enabled": ("--benchmark-enabled", "--no-benchmark"),
        "transaction_cost_enabled": (
            "--transaction-cost-enabled",
            "--no-transaction-cost",
        ),
        "source_chunk_loading": ("--source-chunk-loading", "--no-source-chunk-loading"),
        "candidate_artifact_retention_enabled": (
            "--candidate-artifact-retention-enabled",
            "--no-candidate-artifact-retention",
        ),
        "analysis_artifact_retention_enabled": (
            "--analysis-artifact-retention-enabled",
            "--no-analysis-artifact-retention",
        ),
        "run_health_retention_enabled": (
            "--run-health-retention-enabled",
            "--no-run-health-retention",
        ),
    }
    true_flag, false_flag = mapping[key]
    return [true_flag if value else false_flag]


def _running_job_for_universe(root: Path, universe: str) -> dict[str, Any] | None:
    for path in _jobs_root(root).glob("*/job.json"):
        job = _refresh_job(root, path.parent.name)
        if job and str(job.get("universe")) == universe and str(job.get("status")) == "running":
            return _compact_job(job)
    return None


def _monitor_job(root: Path, job_id: str, stdout: Any, stderr: Any) -> None:
    try:
        proc = _PROCESSES.get(job_id)
        if proc is None:
            return
        while proc.poll() is None:
            threading.Event().wait(1.0)
        job = _load_job(_job_dir(root, job_id))
        if str(job.get("status")) == "cancelled":
            return
        code = int(proc.returncode or 0)
        job["status"] = "succeeded" if code == 0 else "failed"
        job["exit_code"] = code
        job["ended_at_utc"] = _utc_now()
        job["result_summary"] = _result_summary(root, str(job.get("universe") or ""))
        _write_job(_job_dir(root, job_id), job)
    finally:
        try:
            stdout.close()
            stderr.close()
        except Exception:
            pass


def _refresh_job(root: Path, job_id: str) -> dict[str, Any] | None:
    job_dir = _job_dir(root, job_id)
    if not (job_dir / "job.json").exists():
        return None
    job = _load_job(job_dir)
    if str(job.get("status")) == "running":
        proc = _PROCESSES.get(job_id)
        if proc is not None and proc.poll() is not None:
            code = int(proc.returncode or 0)
            job["status"] = "succeeded" if code == 0 else "failed"
            job["exit_code"] = code
            job["ended_at_utc"] = job.get("ended_at_utc") or _utc_now()
            job["result_summary"] = _result_summary(root, str(job.get("universe") or ""))
            _write_job(job_dir, job)
        elif proc is None:
            pid = _int(job.get("pid"), 0)
            if pid > 0 and _pid_exists(pid):
                job["external_process"] = True
            else:
                job["status"] = "interrupted"
                job["ended_at_utc"] = job.get("ended_at_utc") or _utc_now()
                job["exit_code"] = job.get("exit_code")
                job["failure_category"] = "interrupted"
                _write_job(job_dir, job)
    job["lock_owner"] = _lock_owner(root, str(job.get("universe") or ""))
    return job


def _job_response(root: Path, job: dict[str, Any]) -> dict[str, Any]:
    out = dict(job)
    stdout_tail = _tail(Path(str(job.get("stdout_path") or "")))
    stderr_tail = _tail(Path(str(job.get("stderr_path") or "")))
    out["stdout_tail"] = stdout_tail
    out["stderr_tail"] = stderr_tail
    out["stdout_bytes"] = _size(Path(str(job.get("stdout_path") or "")))
    out["stderr_bytes"] = _size(Path(str(job.get("stderr_path") or "")))
    out.update(_classify_failure(out, stdout_tail=stdout_tail, stderr_tail=stderr_tail))
    out.update(_status_explanation(out))
    lock_owner = out.get("lock_owner")
    if isinstance(lock_owner, dict) and lock_owner:
        age = _lock_age_seconds(lock_owner)
        out["lock_age_seconds"] = age
        out["lock_stale_hint"] = _lock_stale_hint(age)
    else:
        out["lock_age_seconds"] = None
        out["lock_stale_hint"] = ""
    return out


def _compact_job(job: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "job_id",
        "status",
        "pid",
        "universe",
        "created_at_utc",
        "started_at_utc",
        "ended_at_utc",
        "exit_code",
        "result_summary",
        "external_process",
        "failure_category",
        "failure_title",
        "failure_hint",
        "lock_owner",
        "status_label",
        "status_hint",
        "lock_age_seconds",
        "lock_stale_hint",
    ]
    compact = {key: job.get(key) for key in keys if key in job}
    if "lock_owner" not in compact:
        compact["lock_owner"] = None
    compact.update(_status_explanation(compact))
    if isinstance(compact.get("lock_owner"), dict) and compact["lock_owner"]:
        compact["lock_age_seconds"] = _lock_age_seconds(compact["lock_owner"])
        compact["lock_stale_hint"] = _lock_stale_hint(compact["lock_age_seconds"])
    return compact


def _result_summary(root: Path, universe: str) -> dict[str, Any]:
    feedback = root / universe / "feedback" / "expression_scoreboard.csv"
    if not feedback.exists():
        return {}
    try:
        frame = pd_read_csv(feedback)
        return {"scoreboard_rows": int(len(frame))}
    except Exception:
        return {}


def pd_read_csv(path: Path):
    import pandas as pd

    return pd.read_csv(path)


def _tail(path: Path, max_bytes: int = 8192) -> str:
    try:
        if not path.exists():
            return ""
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes), os.SEEK_SET)
            return handle.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _size(path: Path) -> int:
    try:
        return int(path.stat().st_size) if path.exists() else 0
    except Exception:
        return 0


def _jobs_root(root: Path) -> Path:
    return root / JOB_ROOT_NAME / "closed_loop"


def _job_dir(root: Path, job_id: str) -> Path:
    return _jobs_root(root) / str(job_id)


def _load_job(job_dir: Path) -> dict[str, Any]:
    return json.loads((job_dir / "job.json").read_text(encoding="utf-8"))


def _write_job(job_dir: Path, job: dict[str, Any]) -> None:
    atomic_write_json(job_dir / "job.json", job)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _pid_exists(pid: int) -> bool:
    pid = _int(pid, 0)
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _lock_owner(root: Path, universe: str) -> dict[str, Any] | None:
    path = root / str(universe) / ".closed_loop.lock" / "owner.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    allowed = [
        "pid",
        "hostname",
        "started_at_utc",
        "heartbeat_at_utc",
        "universe",
        "config_hash",
    ]
    return {key: payload.get(key) for key in allowed if key in payload}


def _lock_age_seconds(owner: dict[str, Any]) -> float | None:
    value = str(owner.get("heartbeat_at_utc") or owner.get("started_at_utc") or "")
    try:
        ts = datetime.fromisoformat(value).timestamp()
    except Exception:
        return None
    return max(0.0, datetime.now(timezone.utc).timestamp() - ts)


def _lock_stale_hint(age: float | None) -> str:
    if age is None:
        return "Lock owner heartbeat is unavailable."
    if age > 3600:
        return "Lock heartbeat is older than one hour; the closed-loop timeout may clear it on the next run."
    return "Lock heartbeat is recent; wait for the running job or cancel it if it was launched from this dashboard."


def _status_explanation(job: dict[str, Any]) -> dict[str, str]:
    status = str(job.get("status") or "")
    external = bool(job.get("external_process"))
    labels = {
        "running": "Running outside dashboard" if external else "Running",
        "succeeded": "Succeeded",
        "failed": "Failed",
        "cancelled": "Cancelled",
        "interrupted": "Interrupted",
        "queued": "Queued",
    }
    hints = {
        "running": "The child process is still alive but this dashboard process did not launch it."
        if external
        else "The closed-loop child process is running and logs are being tailed.",
        "succeeded": "The closed-loop process exited successfully.",
        "failed": "The process exited with an error; inspect the diagnosis and stderr tail.",
        "cancelled": "The job was terminated from the dashboard.",
        "interrupted": "The dashboard recovered a running job but its process is no longer alive.",
        "queued": "The job is waiting to start.",
    }
    return {
        "status_label": labels.get(status, status.title() if status else "-"),
        "status_hint": hints.get(status, "Inspect the job details and logs for the current state."),
    }


def _classify_failure(job: dict[str, Any], *, stdout_tail: str, stderr_tail: str) -> dict[str, Any]:
    status = str(job.get("status") or "")
    text = f"{stdout_tail}\n{stderr_tail}".lower()
    category = str(job.get("failure_category") or "").strip()
    if not category:
        if "hard limit" in text or "memory protection" in text or "source_chunk_hard_limit" in text:
            category = "memory_protection"
        elif ("duckdb" in text and ("temp" in text or "temporary" in text)) or "temporary directory" in text:
            category = "duckdb_temp"
        elif "empty frame" in text or "no data" in text or "0 rows" in text:
            category = "data_empty"
        elif "closed_loop lock exists" in text or "lock exists" in text:
            category = "lock_conflict"
        elif "config" in text or "argument" in text or "invalid" in text:
            category = "config_error"
        elif "candidate" in text or "generate" in text or "parse error" in text or "eval error" in text:
            category = "candidate_generation"
        elif "analysis" in text or "artifact missing" in text or "portfolio_pnl_df" in text:
            category = "analysis_error"
        elif status == "cancelled":
            category = "cancelled"
        elif status == "interrupted":
            category = "interrupted"
        elif status == "failed":
            category = "unknown"
    details = {
        "memory_protection": (
            "Memory protection triggered",
            "The source chunk memory hard limit was hit. Lower max_eval/request_new/batch_size or raise source_chunk_mem_hard_limit_mb.",
        ),
        "duckdb_temp": (
            "DuckDB temporary storage issue",
            "Check duckdb_temp_directory and duckdb_max_temp_directory_size, or reduce the run size.",
        ),
        "data_empty": (
            "No usable data",
            "Check the selected date range, source view, required fields, and datasource filters.",
        ),
        "lock_conflict": (
            "Closed-loop lock conflict",
            "Another closed-loop run owns the universe lock. Inspect the lock owner and wait for timeout if it is stale.",
        ),
        "config_error": (
            "Configuration error",
            "Review the submitted parameters and datasource config paths.",
        ),
        "candidate_generation": (
            "Candidate generation failed",
            "Reduce generation scope or inspect candidate-related stderr for invalid expressions.",
        ),
        "analysis_error": (
            "Analysis failed",
            "Inspect analysis artifacts and stderr; the mining step may have completed before analysis failed.",
        ),
        "cancelled": ("Job cancelled", "The job was terminated from the dashboard."),
        "interrupted": (
            "Job interrupted",
            "The dashboard recovered a running job whose process is no longer alive.",
        ),
        "unknown": (
            "Job failed",
            "Inspect stderr and stdout for the underlying error.",
        ),
    }
    if not category:
        return {"failure_category": "", "failure_title": "", "failure_hint": ""}
    title, hint = details.get(category, details["unknown"])
    return {"failure_category": category, "failure_title": title, "failure_hint": hint}
