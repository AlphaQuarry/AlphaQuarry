from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_mining.datasource import load_datasource_settings, load_panel_from_duckdb
from alpha_mining.mining import fragment_registry_path, load_fragment_registry
from alpha_mining.workflow.universe_store import (
    get_universe_paths,
    load_universe_expression_registry,
)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    ds = load_datasource_settings(str(args.datasource_config or "") or None)
    duckdb_path = str(args.duckdb_path or ds.paths.duckdb_path).strip()
    source_view = str(args.source_view or ds.source_view).strip() or "v_project_panel_cn_a"
    if not duckdb_path:
        raise ValueError("duckdb path is required")

    start_date, end_date = _resolve_date_range(
        duckdb_path=duckdb_path,
        source_view=source_view,
        date_col=str(args.date_col),
        start_date=str(args.start_date or "").strip(),
        end_date=str(args.end_date or "").strip(),
        lookback_days=int(args.lookback_days),
    )

    run_tag = str(args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S"))
    base_universe = f"{args.universe_prefix}_baseline_{run_tag}"
    mutation_universe = f"{args.universe_prefix}_mutation_{run_tag}"
    warmup_iterations = _non_negative_int(args.warmup_iterations, 0)
    warmup_request_new = _non_negative_int(args.warmup_request_new, 0)

    if warmup_iterations > 0:
        warmup_request = warmup_request_new or _positive_int(args.request_new, 10)
        baseline_warmup_cmd = _build_variant_command(
            args=args,
            datasource_config=str(args.datasource_config),
            duckdb_path=duckdb_path,
            source_view=source_view,
            start_date=start_date,
            end_date=end_date,
            universe_name=base_universe,
            enable_feedback_mutation=False,
            request_new_override=warmup_request,
            iterations=warmup_iterations,
        )
        mutation_warmup_cmd = _build_variant_command(
            args=args,
            datasource_config=str(args.datasource_config),
            duckdb_path=duckdb_path,
            source_view=source_view,
            start_date=start_date,
            end_date=end_date,
            universe_name=mutation_universe,
            enable_feedback_mutation=True,
            request_new_override=warmup_request,
            iterations=warmup_iterations,
        )
        _run_variant_command(baseline_warmup_cmd, "baseline_warmup")
        _run_variant_command(mutation_warmup_cmd, "mutation_warmup")

    baseline_cmd = _build_variant_command(
        args=args,
        datasource_config=str(args.datasource_config),
        duckdb_path=duckdb_path,
        source_view=source_view,
        start_date=start_date,
        end_date=end_date,
        universe_name=base_universe,
        enable_feedback_mutation=False,
        request_new_override=None,
        iterations=1,
    )
    mutation_cmd = _build_variant_command(
        args=args,
        datasource_config=str(args.datasource_config),
        duckdb_path=duckdb_path,
        source_view=source_view,
        start_date=start_date,
        end_date=end_date,
        universe_name=mutation_universe,
        enable_feedback_mutation=True,
        request_new_override=None,
        iterations=1,
    )
    _run_variant_command(baseline_cmd, "baseline")
    _run_variant_command(mutation_cmd, "mutation")

    baseline_row = _collect_variant_metrics(
        universe_name=base_universe,
        base_dir=str(args.base_dir),
        variant="baseline",
        top_n=int(args.top_n),
    )
    mutation_row = _collect_variant_metrics(
        universe_name=mutation_universe,
        base_dir=str(args.base_dir),
        variant="mutation",
        top_n=int(args.top_n),
    )
    baseline_row["warmup_iterations"] = int(warmup_iterations)
    mutation_row["warmup_iterations"] = int(warmup_iterations)
    baseline_row["measured_iterations"] = 1
    mutation_row["measured_iterations"] = 1
    summary_df = pd.DataFrame([baseline_row, mutation_row])

    out_root = Path(str(args.artifacts_dir)).resolve()
    run_out_dir = out_root / str(args.artifacts_run_subdir) / run_tag
    run_out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_out_dir / "feedback_mutation_ab_summary.csv"
    report_path = run_out_dir / "feedback_mutation_ab_report.md"
    summary_df["run_tag"] = run_tag
    summary_df.to_csv(summary_path, index=False)
    report_path.write_text(
        _build_markdown_report(
            summary_df=summary_df,
            top_n=int(args.top_n),
            start_date=start_date,
            end_date=end_date,
            source_view=source_view,
        ),
        encoding="utf-8",
    )
    if not bool(args.no_legacy_output):
        legacy_summary_path = out_root / "feedback_mutation_ab_summary.csv"
        legacy_report_path = out_root / "feedback_mutation_ab_report.md"
        summary_df.to_csv(legacy_summary_path, index=False)
        legacy_report_path.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[ab] legacy_summary={legacy_summary_path.as_posix()}")
        print(f"[ab] legacy_report={legacy_report_path.as_posix()}")
    print(f"[ab] summary={summary_path.as_posix()}")
    print(f"[ab] report={report_path.as_posix()}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare feedback mutation ON/OFF with the same closed-loop setup.")
    parser.add_argument("--datasource-config", default="configs/datasource.local.yaml")
    parser.add_argument("--duckdb-path", default="")
    parser.add_argument("--source-view", default="v_project_panel_cn_a")
    parser.add_argument("--duckdb-memory-limit", default="", help="DuckDB memory_limit, e.g. 6GB")
    parser.add_argument(
        "--duckdb-threads",
        type=int,
        default=0,
        help="DuckDB threads, 0 means DuckDB default",
    )
    parser.add_argument(
        "--duckdb-temp-directory",
        default="",
        help="DuckDB temp_directory for spill files",
    )
    parser.add_argument(
        "--duckdb-temp-isolate-run",
        action="store_true",
        help="Forward run-scoped temp directory isolation to run_closed_loop.",
    )
    parser.add_argument(
        "--no-duckdb-temp-isolate-run",
        action="store_true",
        help="Disable run-scoped temp directory isolation for run_closed_loop.",
    )
    parser.add_argument(
        "--duckdb-max-temp-directory-size",
        default="",
        help="DuckDB max_temp_directory_size, e.g. 100GB",
    )
    parser.add_argument("--duckdb-temp-cleanup-warn-gb", type=float, default=10.0)
    parser.add_argument("--cleanup-duckdb-temp", action="store_true")
    parser.add_argument("--no-cleanup-duckdb-temp", action="store_true")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--code-col", default="code")
    parser.add_argument("--base-dir", default="data/alpha_universe_store")
    parser.add_argument("--artifacts-dir", default="artifacts/dev")
    parser.add_argument("--artifacts-run-subdir", default="feedback_mutation_ab_runs")
    parser.add_argument("--no-legacy-output", action="store_true")
    parser.add_argument("--universe-prefix", default="cn_all_ab")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--group-fields", default="industry,sector")
    parser.add_argument("--vector-fields", default="")
    parser.add_argument("--include-fields", default="")
    parser.add_argument("--exclude-fields", default="")
    parser.add_argument(
        "--search-mode",
        default="operator_only",
        choices=["template_only", "deep_hybrid", "operator_only", "layered_v2"],
    )
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--request-new", type=int, default=10)
    parser.add_argument("--warmup-iterations", type=int, default=0)
    parser.add_argument("--warmup-request-new", type=int, default=0)
    parser.add_argument("--max-eval", type=int, default=120)
    parser.add_argument("--analysis-layers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--mutation-budget-ratio", type=float, default=0.15)
    parser.add_argument("--mutation-max-children-per-parent", type=int, default=3)
    parser.add_argument("--mutation-min-selected-count", type=int, default=0)
    parser.add_argument("--mutation-min-selected-ratio", type=float, default=0.0)
    parser.add_argument("--fragment-cooldown-batches", type=int, default=3)
    parser.add_argument("--fragment-max-age-batches", type=int, default=50)
    parser.set_defaults(duckdb_temp_isolate_run=True, no_duckdb_temp_isolate_run=False)
    return parser


def _build_variant_command(
    args: argparse.Namespace,
    datasource_config: str,
    duckdb_path: str,
    source_view: str,
    start_date: str,
    end_date: str,
    universe_name: str,
    enable_feedback_mutation: bool,
    request_new_override: int | None,
    iterations: int = 1,
) -> list[str]:
    duckdb_runtime = _build_duckdb_runtime_settings(args=args, duckdb_path=duckdb_path)
    request_new = int(request_new_override) if request_new_override is not None else int(args.request_new)
    cmd = [
        str(Path(sys.executable).as_posix()),
        str((ROOT / "scripts" / "run_closed_loop.py").as_posix()),
        "--datasource-config",
        str(datasource_config),
        "--duckdb-path",
        str(duckdb_path),
        "--source-view",
        str(source_view),
        "--start-date",
        str(start_date),
        "--end-date",
        str(end_date),
        "--source-backend",
        "duckdb",
        "--source-chunk-loading",
        "--no-snapshot-input",
        "--universe",
        str(universe_name),
        "--base-dir",
        str(args.base_dir),
        "--search-mode",
        str(args.search_mode),
        "--batch-size",
        str(int(args.batch_size)),
        "--request-new",
        str(max(1, int(request_new))),
        "--max-eval",
        str(int(args.max_eval)),
        "--iterations",
        str(max(1, int(iterations))),
        "--date-col",
        str(args.date_col),
        "--code-col",
        str(args.code_col),
        "--group-fields",
        str(args.group_fields),
        "--vector-fields",
        str(args.vector_fields),
        "--include-fields",
        str(args.include_fields),
        "--exclude-fields",
        str(args.exclude_fields),
        "--analysis-layers",
        str(int(args.analysis_layers)),
        "--feedback-min-explore-ratio",
        "0.30",
        "--mutation-budget-ratio",
        str(_normalized_mutation_budget_ratio(args.mutation_budget_ratio)),
        "--mutation-max-children-per-parent",
        str(_positive_int(args.mutation_max_children_per_parent, 3)),
        "--mutation-min-selected-count",
        str(_non_negative_int(args.mutation_min_selected_count, 0)),
        "--mutation-min-selected-ratio",
        str(_normalized_unit_ratio(args.mutation_min_selected_ratio, 0.0)),
        "--fragment-max-age-batches",
        str(_positive_int(args.fragment_max_age_batches, 50)),
        "--fragment-cooldown-batches",
        str(_positive_int(args.fragment_cooldown_batches, 3)),
        "--seed",
        str(_positive_int(args.seed, 42)),
        "--duckdb-memory-limit",
        str(duckdb_runtime.get("memory_limit", "") or ""),
        "--duckdb-threads",
        str(_non_negative_int(duckdb_runtime.get("threads", 0), 0)),
        "--duckdb-temp-directory",
        str(duckdb_runtime.get("temp_directory", "") or ""),
        "--duckdb-max-temp-directory-size",
        str(duckdb_runtime.get("max_temp_directory_size", "") or ""),
        "--duckdb-temp-cleanup-warn-gb",
        str(max(0.0, float(getattr(args, "duckdb_temp_cleanup_warn_gb", 10.0) or 0.0))),
    ]
    if bool(args.duckdb_temp_isolate_run) and (not bool(args.no_duckdb_temp_isolate_run)):
        cmd.append("--duckdb-temp-isolate-run")
        cmd.extend(["--duckdb-temp-run-id", str(universe_name)])
    if bool(args.no_duckdb_temp_isolate_run):
        cmd.append("--no-duckdb-temp-isolate-run")
    if bool(args.cleanup_duckdb_temp):
        cmd.append("--cleanup-duckdb-temp")
    if bool(args.no_cleanup_duckdb_temp):
        cmd.append("--no-cleanup-duckdb-temp")
    if bool(enable_feedback_mutation):
        cmd.append("--enable-feedback-mutation")
    return cmd


def _run_variant_command(cmd: list[str], variant: str) -> None:
    print(f"[ab] run variant={variant} cmd={' '.join(cmd)}")
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout.strip())
    if proc.stderr:
        print(proc.stderr.strip())
    if int(proc.returncode) != 0:
        raise RuntimeError(
            f"A/B variant '{variant}' failed with code={proc.returncode}. "
            f"stdout_tail={_tail_text(proc.stdout)} stderr_tail={_tail_text(proc.stderr)}"
        )


def _resolve_date_range(
    duckdb_path: str,
    source_view: str,
    date_col: str,
    start_date: str,
    end_date: str,
    lookback_days: int = 365,
) -> tuple[str, str]:
    if start_date and end_date:
        return start_date, end_date

    duckdb_runtime = _build_duckdb_runtime_settings_from_params(
        duckdb_path=duckdb_path,
        memory_limit="",
        threads=0,
        temp_directory="",
        max_temp_directory_size="",
    )
    min_date = pd.NaT
    max_date = pd.NaT
    try:
        probe = load_panel_from_duckdb(
            duckdb_path=duckdb_path,
            source_view=source_view,
            required_fields=[date_col],
            start_date=None,
            end_date=None,
            date_col=date_col,
            code_col="code",
            base_fields=(),
            group_fields=(),
            run_filters={},
            duckdb_settings=duckdb_runtime,
        )
        if date_col in probe.columns:
            dates = pd.to_datetime(probe[date_col], errors="coerce")
            min_date = dates.min()
            max_date = dates.max()
    except Exception:
        pass

    fallback_end = pd.Timestamp.today().normalize()
    resolved_end_ts = (
        pd.to_datetime(end_date, errors="coerce") if end_date else (max_date if pd.notna(max_date) else fallback_end)
    )
    resolved_end = str(resolved_end_ts.strftime("%Y-%m-%d"))
    if start_date:
        resolved_start = start_date
    else:
        tentative = resolved_end_ts - timedelta(days=max(1, int(lookback_days)))
        if pd.notna(min_date) and tentative < min_date:
            tentative = min_date
        resolved_start = tentative.strftime("%Y-%m-%d")
    return str(resolved_start), str(resolved_end)


def _collect_variant_metrics(universe_name: str, base_dir: str, variant: str, top_n: int) -> dict[str, Any]:
    paths = get_universe_paths(base_dir=base_dir, universe_name=universe_name)
    candidate_path = _latest_candidate_csv(paths["root"] / "catalog" / "candidates")
    candidate_df = _read_dataframe(candidate_path)
    scoreboard_path = paths["feedback_scoreboard_csv"]
    scoreboard_df = _read_dataframe(scoreboard_path)
    feedback_hints = _read_json(paths["feedback_dir"] / "feedback_hints.json")
    fragment_df = _load_fragment_registry_from_feedback_dir(paths["feedback_dir"])

    expr_registry = load_universe_expression_registry(base_dir=base_dir, universe_name=universe_name)
    alpha_names = expr_registry["alpha_name"].astype(str).tolist() if ("alpha_name" in expr_registry.columns) else []
    selected_source_dist: dict[str, int] = {}
    if not expr_registry.empty and "source" in expr_registry.columns:
        selected_source_dist = {
            str(k): int(v) for k, v in expr_registry["source"].fillna("").astype(str).value_counts().items()
        }
    candidate_source_dist: dict[str, int] = {}
    if not candidate_df.empty and "source" in candidate_df.columns:
        candidate_source_dist = {
            str(k): int(v) for k, v in candidate_df["source"].fillna("").astype(str).value_counts().items()
        }

    fragment_updates = _extract_fragment_update_stats(feedback_hints)
    fragment_counts = _extract_fragment_registry_counts(fragment_df)
    canonical_stats = _extract_canonical_duplicate_stats(candidate_df)

    row: dict[str, Any] = {
        "variant": variant,
        "universe_name": universe_name,
        "status": "ok",
        "candidate_count": int(len(candidate_df)),
        "candidate_passed": int(
            (candidate_df.get("prefilter_status", pd.Series(dtype=str)).astype(str) == "pass").sum()
        )
        if not candidate_df.empty
        else 0,
        "candidate_rejected": int(
            (candidate_df.get("prefilter_status", pd.Series(dtype=str)).astype(str) != "pass").sum()
        )
        if not candidate_df.empty
        else 0,
        "sample_reject_count": int(
            (candidate_df.get("sample_status", pd.Series(dtype=str)).astype(str) == "reject").sum()
        )
        if not candidate_df.empty and "sample_status" in candidate_df.columns
        else 0,
        "mutation_candidate_count": int(
            (candidate_df.get("source", pd.Series(dtype=str)).astype(str) == "feedback_mutation_v2").sum()
        )
        if not candidate_df.empty
        else 0,
        "candidate_source_dist_json": json.dumps(candidate_source_dist, ensure_ascii=False, sort_keys=True),
        "selected_count": int(len(alpha_names)),
        "selected_source_dist_json": json.dumps(selected_source_dist, ensure_ascii=False, sort_keys=True),
        "scoreboard_rows": int(len(scoreboard_df)),
        "score_col": "",
        "topn_score_mean": 0.0,
        "topn_score_median": 0.0,
        "topn_positive_ratio": 0.0,
        "topn_turnover_mean": 0.0,
    }
    if row["candidate_count"] > 0:
        row["mutation_ratio"] = float(row["mutation_candidate_count"]) / float(row["candidate_count"])
    else:
        row["mutation_ratio"] = 0.0
    score_stats = _scoreboard_topn_stats(scoreboard_df, top_n=top_n)
    row.update(score_stats)
    row.update(fragment_updates)
    row.update(fragment_counts)
    row.update(canonical_stats)
    return row


def _latest_candidate_csv(candidates_dir: Path) -> Path | None:
    if not candidates_dir.exists():
        return None
    files = sorted(
        candidates_dir.glob("*_candidates.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def _read_dataframe(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    target = Path(path)
    if not target.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(target)
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_fragment_registry_from_feedback_dir(feedback_dir: Path) -> pd.DataFrame:
    reg_path = fragment_registry_path(feedback_dir)
    try:
        return load_fragment_registry(reg_path)
    except Exception:
        return pd.DataFrame()


def _extract_fragment_update_stats(feedback_hints: dict[str, Any]) -> dict[str, Any]:
    updates = feedback_hints.get("fragment_feedback_updates", {})
    if not isinstance(updates, dict):
        updates = {}
    return {
        "fragment_feedback_update_positive": int(_to_float(updates.get("positive_updates", 0))),
        "fragment_feedback_update_negative": int(_to_float(updates.get("negative_updates", 0))),
        "fragment_feedback_update_rejected": int(_to_float(updates.get("rejected_updates", 0))),
    }


def _extract_fragment_registry_counts(fragment_df: pd.DataFrame) -> dict[str, Any]:
    if fragment_df is None or fragment_df.empty:
        return {
            "fragment_registry_total": 0,
            "fragment_registry_active": 0,
            "fragment_registry_cooldown": 0,
            "fragment_registry_retired": 0,
            "fragment_registry_cooldown_ratio": 0.0,
        }
    status = fragment_df.get("status", pd.Series(dtype=str)).fillna("").astype(str).str.lower()
    total = int(len(fragment_df))
    active = int((status == "active").sum())
    cooldown = int((status == "cooldown").sum())
    retired = int((status == "retired").sum())
    ratio = float(cooldown) / float(total) if total > 0 else 0.0
    return {
        "fragment_registry_total": total,
        "fragment_registry_active": active,
        "fragment_registry_cooldown": cooldown,
        "fragment_registry_retired": retired,
        "fragment_registry_cooldown_ratio": ratio,
    }


def _extract_canonical_duplicate_stats(candidate_df: pd.DataFrame) -> dict[str, Any]:
    if candidate_df is None or candidate_df.empty or "canonical_hash" not in candidate_df.columns:
        return {
            "canonical_hash_total": 0,
            "canonical_hash_unique": 0,
            "canonical_hash_duplicate_ratio": 0.0,
            "mutation_canonical_hash_total": 0,
            "mutation_canonical_hash_unique": 0,
            "mutation_canonical_hash_duplicate_ratio": 0.0,
        }
    canon = candidate_df["canonical_hash"].fillna("").astype(str).str.strip()
    canon = canon[canon != ""]
    total = int(len(canon))
    unique = int(canon.nunique()) if total else 0
    dup_ratio = float(total - unique) / float(total) if total else 0.0

    mutation_total = 0
    mutation_unique = 0
    if "source" in candidate_df.columns:
        mutation_mask = candidate_df["source"].fillna("").astype(str) == "feedback_mutation_v2"
        mut = candidate_df.loc[mutation_mask, "canonical_hash"].fillna("").astype(str).str.strip()
        mut = mut[mut != ""]
        mutation_total = int(len(mut))
        mutation_unique = int(mut.nunique()) if mutation_total else 0
    mutation_dup_ratio = float(mutation_total - mutation_unique) / float(mutation_total) if mutation_total else 0.0
    return {
        "canonical_hash_total": total,
        "canonical_hash_unique": unique,
        "canonical_hash_duplicate_ratio": dup_ratio,
        "mutation_canonical_hash_total": mutation_total,
        "mutation_canonical_hash_unique": mutation_unique,
        "mutation_canonical_hash_duplicate_ratio": mutation_dup_ratio,
    }


def _scoreboard_topn_stats(scoreboard_df: pd.DataFrame, top_n: int) -> dict[str, Any]:
    if scoreboard_df is None or scoreboard_df.empty:
        return {
            "score_col": "",
            "topn_score_mean": 0.0,
            "topn_score_median": 0.0,
            "topn_positive_ratio": 0.0,
            "topn_turnover_mean": 0.0,
        }
    score_col = (
        "score_total"
        if "score_total" in scoreboard_df.columns
        else ("scoreboard_score" if "scoreboard_score" in scoreboard_df.columns else "")
    )
    if not score_col:
        return {
            "score_col": "",
            "topn_score_mean": 0.0,
            "topn_score_median": 0.0,
            "topn_positive_ratio": 0.0,
            "topn_turnover_mean": 0.0,
        }
    work = scoreboard_df.copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    top = work.sort_values(score_col, ascending=False).head(max(1, int(top_n)))
    scores = top[score_col].dropna()
    turnover = pd.to_numeric(top.get("turnover_long_only_mean", pd.Series(dtype=float)), errors="coerce")
    positive_ratio = float((scores > 0).sum()) / float(len(scores)) if len(scores) else 0.0
    return {
        "score_col": score_col,
        "topn_score_mean": float(scores.mean()) if len(scores) else 0.0,
        "topn_score_median": float(scores.median()) if len(scores) else 0.0,
        "topn_positive_ratio": positive_ratio,
        "topn_turnover_mean": float(turnover.mean()) if len(turnover.dropna()) else 0.0,
    }


def _build_markdown_report(
    summary_df: pd.DataFrame,
    top_n: int,
    start_date: str,
    end_date: str,
    source_view: str,
) -> str:
    if summary_df is None or summary_df.empty:
        return "# Feedback Mutation A/B Report\n\nNo results."
    rows = summary_df.to_dict(orient="records")
    baseline = next((x for x in rows if str(x.get("variant", "")) == "baseline"), {})
    mutation = next((x for x in rows if str(x.get("variant", "")) == "mutation"), {})
    keys = [
        "candidate_count",
        "candidate_passed",
        "candidate_rejected",
        "sample_reject_count",
        "mutation_candidate_count",
        "mutation_ratio",
        "selected_count",
        "scoreboard_rows",
        "topn_score_mean",
        "topn_score_median",
        "topn_positive_ratio",
        "topn_turnover_mean",
        "canonical_hash_duplicate_ratio",
        "mutation_canonical_hash_duplicate_ratio",
        "fragment_feedback_update_positive",
        "fragment_feedback_update_negative",
        "fragment_feedback_update_rejected",
        "fragment_registry_total",
        "fragment_registry_active",
        "fragment_registry_cooldown",
        "fragment_registry_cooldown_ratio",
    ]
    lines = [
        "# Feedback Mutation A/B Report",
        "",
        f"- source_view: `{source_view}`",
        f"- date_range: `{start_date}` to `{end_date}`",
        f"- topN for scoreboard stats: `{int(top_n)}`",
        f"- warmup_iterations: `{int(_to_float(baseline.get('warmup_iterations', 0)))}`",
        "",
        "## Summary Table",
        "",
        "| variant | status | candidate_count | passed | rejected | mutation_count | mutation_ratio | topn_score_mean | topn_positive_ratio | canon_dup_ratio | fragment_cooldown_ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('variant', '')} | {row.get('status', '')} | {int(row.get('candidate_count', 0))} | "
            f"{int(row.get('candidate_passed', 0))} | {int(row.get('candidate_rejected', 0))} | "
            f"{int(row.get('mutation_candidate_count', 0))} | {float(row.get('mutation_ratio', 0.0)):.4f} | "
            f"{float(row.get('topn_score_mean', 0.0)):.6f} | {float(row.get('topn_positive_ratio', 0.0)):.4f} | "
            f"{float(row.get('canonical_hash_duplicate_ratio', 0.0)):.4f} | {float(row.get('fragment_registry_cooldown_ratio', 0.0)):.4f} |"
        )
    lines.extend(["", "## Delta (mutation - baseline)", ""])
    for key in keys:
        b = _to_float(baseline.get(key, 0.0))
        m = _to_float(mutation.get(key, 0.0))
        lines.append(f"- `{key}`: `{m - b:.6f}` (mutation `{m:.6f}` vs baseline `{b:.6f}`)")
    lines.extend(
        [
            "",
            "## Source Distribution",
            "",
            f"- candidate baseline: `{baseline.get('candidate_source_dist_json', '{}')}`",
            f"- candidate mutation: `{mutation.get('candidate_source_dist_json', '{}')}`",
            f"- baseline: `{baseline.get('selected_source_dist_json', '{}')}`",
            f"- mutation: `{mutation.get('selected_source_dist_json', '{}')}`",
            "",
            "## Fragment Feedback",
            "",
            f"- baseline updates: `+{int(_to_float(baseline.get('fragment_feedback_update_positive', 0)))} / -{int(_to_float(baseline.get('fragment_feedback_update_negative', 0)))}` rejected=`{int(_to_float(baseline.get('fragment_feedback_update_rejected', 0)))}`",
            f"- mutation updates: `+{int(_to_float(mutation.get('fragment_feedback_update_positive', 0)))} / -{int(_to_float(mutation.get('fragment_feedback_update_negative', 0)))}` rejected=`{int(_to_float(mutation.get('fragment_feedback_update_rejected', 0)))}`",
            f"- baseline registry: total=`{int(_to_float(baseline.get('fragment_registry_total', 0)))}` active=`{int(_to_float(baseline.get('fragment_registry_active', 0)))}` cooldown=`{int(_to_float(baseline.get('fragment_registry_cooldown', 0)))}`",
            f"- mutation registry: total=`{int(_to_float(mutation.get('fragment_registry_total', 0)))}` active=`{int(_to_float(mutation.get('fragment_registry_active', 0)))}` cooldown=`{int(_to_float(mutation.get('fragment_registry_cooldown', 0)))}`",
            "",
        ]
    )
    return "\n".join(lines)


def _normalized_mutation_budget_ratio(raw: float) -> float:
    try:
        value = float(raw)
    except Exception:
        return 0.15
    return max(0.0, min(1.0, value))


def _normalized_unit_ratio(raw: float, default: float = 0.0) -> float:
    try:
        value = float(raw)
    except Exception:
        return float(default)
    return max(0.0, min(1.0, value))


def _positive_int(raw: Any, default: int) -> int:
    try:
        value = int(raw)
    except Exception:
        return int(default)
    return max(1, value)


def _non_negative_int(raw: Any, default: int = 0) -> int:
    try:
        value = int(raw)
    except Exception:
        return int(default)
    return max(0, value)


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _tail_text(text: str, max_lines: int = 20) -> str:
    if not text:
        return ""
    lines = [x for x in str(text).splitlines() if x.strip()]
    if not lines:
        return ""
    return " | ".join(lines[-max(1, int(max_lines)) :])


def _build_duckdb_runtime_settings(args: argparse.Namespace, duckdb_path: str) -> dict[str, Any]:
    return _build_duckdb_runtime_settings_from_params(
        duckdb_path=duckdb_path,
        memory_limit=str(args.duckdb_memory_limit or "").strip(),
        threads=_non_negative_int(args.duckdb_threads, 0),
        temp_directory=str(args.duckdb_temp_directory or "").strip(),
        max_temp_directory_size=str(args.duckdb_max_temp_directory_size or "").strip(),
    )


def _build_duckdb_runtime_settings_from_params(
    duckdb_path: str,
    memory_limit: str,
    threads: int,
    temp_directory: str,
    max_temp_directory_size: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if str(memory_limit or "").strip():
        out["memory_limit"] = str(memory_limit).strip()
    try:
        tv = int(threads)
    except Exception:
        tv = 0
    if tv > 0:
        out["threads"] = tv

    temp_text = str(temp_directory or "").strip()
    if temp_text:
        temp_path = Path(temp_text)
        if not temp_path.is_absolute():
            temp_path = Path.cwd() / temp_path
    else:
        temp_path = Path(f"{duckdb_path}.tmp")
    out["temp_directory"] = str(temp_path.as_posix())

    if str(max_temp_directory_size or "").strip():
        out["max_temp_directory_size"] = str(max_temp_directory_size).strip()
    return out


if __name__ == "__main__":
    main()
