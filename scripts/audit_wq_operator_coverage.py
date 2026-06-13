from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_mining.mining.operator_signatures import (
    build_default_operator_signature_registry,
)
from alpha_mining.registry import build_default_registry


WQ_JSON_FILES: tuple[tuple[str, str], ...] = (
    ("arithmetic", "arithmetic_operators.json"),
    ("cross_sectional", "cross_sectional_operators.json"),
    ("group", "group_operators.json"),
    ("logical", "logical_operators.json"),
    ("special", "special_operators.json"),
    ("time_series", "time_series_operators.json"),
    ("transformational", "transformational_operators.json"),
    ("vector", "vector_operators.json"),
)


PARTIAL_SEMANTIC_NOTES: dict[str, str] = {
    "trade_when": "Stateless if_else style; no hold-until-exit state semantics.",
    "ts_regression": "Simplified rolling regression; full lag/rettype variants are not covered.",
    "bucket": "Supports simple range-based bucketing only; richer WQ bucket syntax is not covered.",
    "add": "Binary form only; WQ supports variadic inputs and filter behavior.",
    "subtract": "Binary form only; WQ filter behavior is not covered.",
    "multiply": "Binary form only; WQ filter behavior is not covered.",
    "max": "Binary form only; WQ supports variadic inputs.",
    "min": "Binary form only; WQ supports variadic inputs.",
    "days_from_last_change": "Kept as compatibility 0-based behavior in this phase.",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit WQ JSON operator coverage against local registry/signatures.")
    parser.add_argument(
        "--wq-dir",
        default=r"D:\project_git",
        help="Directory containing *operators.json files.",
    )
    parser.add_argument("--output", default="artifacts/dev/wq_operator_coverage_audit.csv")
    parser.add_argument("--baseline-csv", default="wq_operator_coverage.csv")
    parser.add_argument("--plan-file", default="project_quant_optimization_plan_v3.md")
    return parser.parse_args()


def _load_wq_rows(wq_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, filename in WQ_JSON_FILES:
        path = wq_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing WQ JSON file: {path}")
        items = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(items, list):
            raise ValueError(f"Invalid JSON structure (expected list): {path}")
        for item in items:
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            rows.append(
                {
                    "category": category,
                    "name": name,
                    "description": str(item.get("description", "")).strip(),
                }
            )
    return rows


def _contains_operator_token(text: str, operator_name: str) -> bool:
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(str(operator_name).lower())}(?![A-Za-z0-9_])"
    return re.search(pattern, text) is not None


def _priority_hint(name: str, implemented_exact: bool, planned_in_v3: bool) -> str:
    if implemented_exact:
        return "n/a"
    if planned_in_v3:
        if name in {
            "and",
            "or",
            "not",
            "quantile",
            "truncate",
            "group_backfill",
            "group_median",
            "group_scale",
            "left_tail",
            "right_tail",
            "ts_av_diff",
            "ts_backfill",
            "kth_element",
            "ts_count_nans",
            "ts_covariance",
            "ts_median",
            "ts_scale",
        }:
            return "P1"
        if name in {
            "hump",
            "hump_decay",
            "last_diff_value",
            "inst_tvr",
            "ts_delta_limit",
        }:
            return "P2"
        if name in {
            "generate_stats",
            "combo_a",
            "ts_target_tvr_decay",
            "ts_target_tvr_delta_limit",
            "ts_target_tvr_hump",
        }:
            return "P3"
        if name in {"in", "universe_size", "self_corr"}:
            return "P4"
        return "planned"
    return "unplanned_or_low"


def main() -> None:
    args = _parse_args()
    wq_dir = Path(args.wq_dir)
    output = Path(args.output)
    baseline_csv = Path(args.baseline_csv)
    plan_file = Path(args.plan_file)

    wq_rows = _load_wq_rows(wq_dir=wq_dir)
    registry = build_default_registry()
    reg_names = set(registry.list_names())
    sig_names = set(build_default_operator_signature_registry().names())
    plan_text = plan_file.read_text(encoding="utf-8").lower() if plan_file.exists() else ""

    rows: list[dict[str, Any]] = []
    for row in wq_rows:
        name = str(row["name"])
        implemented_exact = name in reg_names
        signed_exact = name in sig_names
        semantic_status = (
            "partial_semantic"
            if implemented_exact and name in PARTIAL_SEMANTIC_NOTES
            else "implemented_exact"
            if implemented_exact
            else "missing_exact"
        )
        planned_in_v3 = _contains_operator_token(plan_text, name) if plan_text else False
        rows.append(
            {
                "category": row["category"],
                "name": name,
                "implemented_exact": bool(implemented_exact),
                "signed_exact": bool(signed_exact),
                "implemented_local": name if implemented_exact else "",
                "semantic_status": semantic_status,
                "semantic_note": PARTIAL_SEMANTIC_NOTES.get(name, ""),
                "planned_in_v3": bool(planned_in_v3),
                "priority_hint": _priority_hint(name, implemented_exact, planned_in_v3),
            }
        )

    out = pd.DataFrame(rows).sort_values(["category", "name"]).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    baseline_diff_count = -1
    baseline_only: list[str] = []
    audit_only: list[str] = []
    changed_rows = 0
    if baseline_csv.exists():
        baseline = pd.read_csv(baseline_csv)
        merged = pd.merge(
            baseline[["category", "name", "implemented_exact", "implemented_local"]],
            out[["category", "name", "implemented_exact", "implemented_local"]],
            on=["category", "name"],
            how="outer",
            suffixes=("_baseline", "_audit"),
            indicator=True,
        )
        baseline_only = merged[merged["_merge"] == "left_only"]["name"].astype(str).tolist()
        audit_only = merged[merged["_merge"] == "right_only"]["name"].astype(str).tolist()
        comp = merged[merged["_merge"] == "both"].copy()
        if not comp.empty:
            comp["implemented_exact_baseline"] = comp["implemented_exact_baseline"].astype(str).str.lower()
            comp["implemented_exact_audit"] = comp["implemented_exact_audit"].astype(str).str.lower()
            changed = comp[
                (comp["implemented_exact_baseline"] != comp["implemented_exact_audit"])
                | (comp["implemented_local_baseline"].fillna("") != comp["implemented_local_audit"].fillna(""))
            ]
            changed_rows = int(len(changed))
        baseline_diff_count = int(len(baseline_only) + len(audit_only) + changed_rows)

    partial_ops = sorted(
        [name for name in out.loc[out["semantic_status"] == "partial_semantic", "name"].astype(str).tolist()]
    )
    missing_ops = sorted(
        [name for name in out.loc[out["semantic_status"] == "missing_exact", "name"].astype(str).tolist()]
    )
    implemented_ops = sorted(
        [name for name in out.loc[out["semantic_status"] != "missing_exact", "name"].astype(str).tolist()]
    )

    extra_local_ops = sorted([name for name in reg_names if name not in set(out["name"].astype(str).tolist())])

    print(f"[wq-audit] wq_total={len(out)}")
    print(f"[wq-audit] local_registered={len(reg_names)} signature_total={len(sig_names)}")
    print(
        f"[wq-audit] implemented_exact={sum(out['semantic_status'] != 'missing_exact')} partial_semantic={sum(out['semantic_status'] == 'partial_semantic')} missing_exact={sum(out['semantic_status'] == 'missing_exact')}"
    )
    print(f"[wq-audit] implemented_ops={implemented_ops}")
    print(f"[wq-audit] partial_semantic_ops={partial_ops}")
    print(f"[wq-audit] missing_exact_ops={missing_ops}")
    print(f"[wq-audit] local_extra_ops={extra_local_ops}")
    if baseline_diff_count >= 0:
        print(f"[wq-audit] baseline_diff_count={baseline_diff_count}")
        if baseline_only:
            print(f"[wq-audit] baseline_only={sorted(baseline_only)}")
        if audit_only:
            print(f"[wq-audit] audit_only={sorted(audit_only)}")
        if changed_rows:
            print(f"[wq-audit] changed_rows={changed_rows}")
    print(f"[wq-audit] output={output.as_posix()}")


if __name__ == "__main__":
    main()
