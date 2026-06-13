from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CORE_TABLE_KEYS = {
    "dashboard_factor_metrics": "dashboard_metrics_bytes",
    "phase_metrics_df": "phase_metrics_bytes",
    "ic_df": "ic_bytes",
    "portfolio_pnl_df": "portfolio_pnl_bytes",
    "benchmark_pnl_df": "benchmark_pnl_bytes",
}
DYNAMIC_TABLE_KEYS = {
    "analysis_distribution_histogram": "distribution_histogram_bytes",
    "analysis_ic_decay": "ic_decay_bytes",
}
PNG_MANIFEST_KEY = "visualization_manifest"


def audit_analysis_artifacts(
    store_root: str | Path = "data/alpha_universe_store",
    *,
    universe: str = "",
    run_id: str = "",
) -> dict[str, Any]:
    root = Path(store_root)
    runs = []
    for meta_path in sorted(root.glob("*/analysis/period_*/analysis_*/analysis_meta.json")):
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rel_parts = meta_path.relative_to(root).parts
        run_universe = str(rel_parts[0]) if rel_parts else ""
        current_run_id = str(payload.get("analysis_run_id") or meta_path.parent.name)
        if universe and run_universe != str(universe):
            continue
        if run_id and current_run_id != str(run_id):
            continue
        runs.append(_audit_one_run(root=root, meta_path=meta_path, universe=run_universe, payload=payload))

    total_bytes = int(sum(int(row.get("total_bytes", 0) or 0) for row in runs))
    return {
        "store_root": str(root.as_posix()),
        "run_count": int(len(runs)),
        "total_bytes": total_bytes,
        "total_mb": _mb(total_bytes),
        "runs": runs,
    }


def _audit_one_run(root: Path, meta_path: Path, universe: str, payload: dict[str, Any]) -> dict[str, Any]:
    analysis_dir = _resolve_path(payload.get("analysis_dir") or meta_path.parent, root=root, meta_path=meta_path)
    table_paths = dict(payload.get("table_paths") or {})
    extra_meta = dict(payload.get("extra_meta") or {})
    row: dict[str, Any] = {
        "universe": str(universe),
        "run_id": str(payload.get("analysis_run_id") or meta_path.parent.name),
        "period": _to_int(payload.get("period")),
        "layers": _to_int(payload.get("layers")),
        "created_at_utc": str(payload.get("created_at_utc", "") or ""),
        "analysis_dir": str(analysis_dir.as_posix()),
        "factor_count": int(len(payload.get("alpha_names") or [])),
        "png_enabled": bool(extra_meta.get("include_visualization_png", False)),
    }
    for output_col in (
        set(CORE_TABLE_KEYS.values())
        | set(DYNAMIC_TABLE_KEYS.values())
        | {"factor_metrics_bytes", "visualization_manifest_bytes"}
    ):
        row[output_col] = 0

    factor_metrics_path = _resolve_path(payload.get("factor_metrics_path", ""), root=root, meta_path=meta_path)
    row["factor_metrics_bytes"] = _file_size(factor_metrics_path)
    for key, output_col in CORE_TABLE_KEYS.items():
        row[output_col] = _file_size(_resolve_path(table_paths.get(key, ""), root=root, meta_path=meta_path))
    for key, output_col in DYNAMIC_TABLE_KEYS.items():
        row[output_col] = _file_size(_resolve_path(table_paths.get(key, ""), root=root, meta_path=meta_path))
    row["visualization_manifest_bytes"] = _file_size(
        _resolve_path(table_paths.get(PNG_MANIFEST_KEY, ""), root=root, meta_path=meta_path)
    )
    row["visualization_png_bytes"] = _tree_size(analysis_dir / "visualizations")
    row["visualization_png_present"] = bool(
        row["visualization_png_bytes"] > 0 or row["visualization_manifest_bytes"] > 0
    )
    row["dynamic_analysis_bytes"] = int(sum(int(row[col]) for col in DYNAMIC_TABLE_KEYS.values()))
    row["core_dashboard_bytes"] = int(
        row["factor_metrics_bytes"] + sum(int(row[col]) for col in CORE_TABLE_KEYS.values())
    )
    row["total_bytes"] = _tree_size(analysis_dir)
    row["total_mb"] = _mb(row["total_bytes"])
    row["dynamic_analysis_mb"] = _mb(row["dynamic_analysis_bytes"])
    row["visualization_png_mb"] = _mb(row["visualization_png_bytes"])
    return row


def _resolve_path(value: Any, *, root: Path, meta_path: Path) -> Path:
    text = str(value or "").strip()
    if not text:
        return Path("")
    path = Path(text)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    root_candidate = root / path
    if root_candidate.exists():
        return root_candidate
    parent_candidate = meta_path.parent / path
    if parent_candidate.exists():
        return parent_candidate
    return cwd_candidate


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size) if path.is_file() else 0
    except Exception:
        return 0


def _tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return _file_size(path)
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += _file_size(item)
    return int(total)


def _mb(value: Any) -> float:
    try:
        return round(float(value) / (1024.0 * 1024.0), 4)
    except Exception:
        return 0.0


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def print_text_report(report: dict[str, Any]) -> None:
    rows = list(report.get("runs") or [])
    print(f"store_root={report.get('store_root')} runs={report.get('run_count')} total_mb={report.get('total_mb')}")
    if not rows:
        return
    columns = [
        ("universe", 18),
        ("run_id", 34),
        ("factors", 7),
        ("total_mb", 9),
        ("dynamic_analysis_mb", 12),
        ("visualization_png_mb", 10),
        ("png_files", 9),
        ("png_cfg", 7),
    ]
    header = " ".join(name.ljust(width) for name, width in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        values = {
            "universe": str(row.get("universe", "")),
            "run_id": str(row.get("run_id", "")),
            "factors": str(row.get("factor_count", 0)),
            "total_mb": f"{float(row.get('total_mb', 0.0)):.4f}",
            "dynamic_analysis_mb": f"{float(row.get('dynamic_analysis_mb', 0.0)):.4f}",
            "visualization_png_mb": f"{float(row.get('visualization_png_mb', 0.0)):.4f}",
            "png_files": "yes" if row.get("visualization_png_present") else "no",
            "png_cfg": "on" if row.get("png_enabled") else "off",
        }
        print(" ".join(values[name][:width].ljust(width) for name, width in columns))


def write_csv(report: dict[str, Any], path: str | Path) -> None:
    rows = list(report.get("runs") or [])
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit per-run analysis artifact disk usage.")
    parser.add_argument("--store-root", default="data/alpha_universe_store")
    parser.add_argument("--universe", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    parser.add_argument("--csv-out", default="", help="Optional path to write per-run CSV report")
    args = parser.parse_args()

    report = audit_analysis_artifacts(
        store_root=args.store_root,
        universe=str(args.universe or ""),
        run_id=str(args.run_id or ""),
    )
    if args.csv_out:
        write_csv(report, args.csv_out)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_text_report(report)


if __name__ == "__main__":
    main()
