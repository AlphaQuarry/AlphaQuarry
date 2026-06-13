from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..atomic_io import atomic_write_dataframe_csv
from ..ast_nodes import (
    BinaryOpNode,
    ExpressionNode,
    FieldNode,
    FunctionCallNode,
    LiteralNode,
    UnaryOpNode,
)
from ..hashing import expression_hash
from ..parser import ExpressionParseError, parse_expression
from .expression_canonicalizer import canonicalize_expression
from .field_universe import is_leakage_field


_DEFAULT_REGISTRY_COLUMNS = [
    "fragment_hash",
    "fragment_expression",
    "fragment_type",
    "operators",
    "fields",
    "windows",
    "groups",
    "output_type",
    "depth",
    "complexity",
    "source_alpha_name",
    "source_alpha_hash",
    "source_expression",
    "source_batch_id",
    "source_score",
    "score_basis",
    "oos_score",
    "positive_count",
    "negative_count",
    "rejected_count",
    "mean_child_score",
    "best_child_score",
    "last_seen_batch",
    "cooldown_until",
    "status",
]


@dataclass(frozen=True)
class FragmentRegistryConfig:
    max_fragment_depth: int = 6
    max_fragment_complexity: int = 16
    cooldown_batches: int = 3
    max_age_batches: int = 50
    top_k: int = 256


def fragment_registry_path(feedback_dir: str | Path) -> Path:
    return Path(feedback_dir) / "fragment_registry.parquet"


def load_fragment_registry(path: str | Path) -> pd.DataFrame:
    stem = Path(path)
    if stem.suffix:
        stem = stem.with_suffix("")
    for ext in [".parquet", ".pkl", ".csv"]:
        candidate = stem.with_suffix(ext)
        if not candidate.exists():
            continue
        try:
            if ext == ".parquet":
                df = pd.read_parquet(candidate)
            elif ext == ".pkl":
                df = pd.read_pickle(candidate)
            else:
                df = pd.read_csv(candidate)
            return _ensure_columns(df)
        except Exception:
            continue
    return pd.DataFrame(columns=list(_DEFAULT_REGISTRY_COLUMNS))


def save_fragment_registry(df: pd.DataFrame, path: str | Path) -> str:
    stem = Path(path)
    if stem.suffix:
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    work = _ensure_columns(df)
    try:
        out = stem.with_suffix(".parquet")
        work.to_parquet(out, index=False)
        return str(out.as_posix())
    except Exception:
        pass
    try:
        out = stem.with_suffix(".pkl")
        work.to_pickle(out)
        return str(out.as_posix())
    except Exception:
        out = stem.with_suffix(".csv")
        atomic_write_dataframe_csv(out, work, index=False, backup=True)
        return str(out.as_posix())


def extract_fragments_from_expression(
    expression: str,
    source_alpha_name: str = "",
    source_batch_id: str = "",
    source_score: float = 0.0,
    score_basis: str = "",
    oos_score: float | None = None,
    max_depth: int = 6,
    max_complexity: int = 16,
) -> list[dict[str, Any]]:
    text = str(expression or "").strip()
    if not text:
        return []
    try:
        root = parse_expression(text)
    except ExpressionParseError:
        return []

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_hash = expression_hash(text)
    for node in _iter_nodes(root):
        depth = _node_depth(node)
        if depth > max(0, int(max_depth)):
            continue
        fields = sorted(_collect_fields(node))
        if not fields:
            continue
        if any(is_leakage_field(f) for f in fields):
            continue
        complexity = len(_collect_operators(node)) + len(fields) + depth
        if complexity > max(1, int(max_complexity)):
            continue
        raw_expr = _node_to_expression(node)
        canon = canonicalize_expression(raw_expr)
        if not canon.passed or not canon.canonical_expression or not canon.canonical_hash:
            continue
        if canon.canonical_hash in seen:
            continue
        seen.add(canon.canonical_hash)
        group_fields = [f for f in fields if _looks_like_group_field(f)]
        item = {
            "fragment_hash": canon.canonical_hash,
            "fragment_expression": canon.canonical_expression,
            "fragment_type": _fragment_type(node),
            "operators": ",".join(sorted(set(_collect_operators(node)))),
            "fields": ",".join(fields),
            "windows": ",".join(str(x) for x in sorted(set(_collect_windows(node)))),
            "groups": ",".join(group_fields),
            "output_type": _output_type(node),
            "depth": int(depth),
            "complexity": int(complexity),
            "source_alpha_name": str(source_alpha_name or ""),
            "source_alpha_hash": str(source_hash),
            "source_expression": text,
            "source_batch_id": str(source_batch_id or ""),
            "source_score": float(source_score) if math.isfinite(float(source_score)) else 0.0,
            "score_basis": str(score_basis or ""),
            "oos_score": float(oos_score)
            if oos_score is not None and math.isfinite(float(oos_score))
            else float("nan"),
            "positive_count": 1 if float(source_score) > 0.0 else 0,
            "negative_count": 1 if float(source_score) <= 0.0 else 0,
            "rejected_count": 0,
            "mean_child_score": float("nan"),
            "best_child_score": float("nan"),
            "last_seen_batch": 0,
            "cooldown_until": 0,
            "status": "active",
        }
        out.append(item)
    return out


def refresh_fragment_registry(
    scoreboard_df: pd.DataFrame | None,
    registry_path: str | Path,
    config: FragmentRegistryConfig | None = None,
    current_batch: int | None = None,
) -> tuple[pd.DataFrame, int, str]:
    cfg = config or FragmentRegistryConfig()
    existing = load_fragment_registry(registry_path)
    batch_no = int(current_batch) if current_batch is not None else _next_batch_index(existing)
    working = _ensure_columns(existing).copy()
    if working.empty:
        working = pd.DataFrame(columns=list(_DEFAULT_REGISTRY_COLUMNS))

    if scoreboard_df is not None and not scoreboard_df.empty and "expression" in scoreboard_df.columns:
        score_col = _select_feedback_score_column(scoreboard_df)
        score_basis = _score_basis(score_col)
        for row in scoreboard_df.to_dict(orient="records"):
            expression = str(row.get("expression", "") or "").strip()
            if not expression:
                continue
            score = _to_float(row.get(score_col, 0.0))
            source_alpha = str(row.get("factor", "") or row.get("alpha_name", "") or "")
            source_batch = str(row.get("batch_id", "") or "")
            fragments = extract_fragments_from_expression(
                expression=expression,
                source_alpha_name=source_alpha,
                source_batch_id=source_batch,
                source_score=score,
                score_basis=score_basis,
                oos_score=row.get("oos_score", None),
                max_depth=int(cfg.max_fragment_depth),
                max_complexity=int(cfg.max_fragment_complexity),
            )
            for item in fragments:
                item["last_seen_batch"] = batch_no
                _upsert_fragment(
                    working,
                    item,
                    batch_no=batch_no,
                    cooldown_batches=int(cfg.cooldown_batches),
                )

    if not working.empty:
        working["last_seen_batch"] = pd.to_numeric(working["last_seen_batch"], errors="coerce").fillna(0).astype(int)
        working["cooldown_until"] = pd.to_numeric(working["cooldown_until"], errors="coerce").fillna(0).astype(int)
        too_old = working["last_seen_batch"] < (int(batch_no) - int(cfg.max_age_batches))
        working.loc[too_old, "status"] = "retired"
        on_cooldown = working["cooldown_until"] > int(batch_no)
        retired = working["status"].astype(str) == "retired"
        working.loc[~retired & on_cooldown, "status"] = "cooldown"
        working.loc[~retired & (~on_cooldown), "status"] = "active"
        working = working.sort_values(
            [
                "status",
                "positive_count",
                "negative_count",
                "source_score",
                "last_seen_batch",
            ],
            ascending=[True, False, True, False, False],
        ).reset_index(drop=True)

    saved_path = save_fragment_registry(working, registry_path)
    return working, batch_no, saved_path


def select_active_fragments(
    registry_df: pd.DataFrame | None,
    current_batch: int,
    max_age_batches: int = 50,
    limit: int = 256,
) -> pd.DataFrame:
    if registry_df is None or registry_df.empty:
        return pd.DataFrame(columns=list(_DEFAULT_REGISTRY_COLUMNS))
    work = _ensure_columns(registry_df).copy()
    work["last_seen_batch"] = pd.to_numeric(work["last_seen_batch"], errors="coerce").fillna(0).astype(int)
    work["cooldown_until"] = pd.to_numeric(work["cooldown_until"], errors="coerce").fillna(0).astype(int)
    work["positive_count"] = pd.to_numeric(work["positive_count"], errors="coerce").fillna(0).astype(float)
    work["negative_count"] = pd.to_numeric(work["negative_count"], errors="coerce").fillna(0).astype(float)
    work["source_score"] = pd.to_numeric(work["source_score"], errors="coerce").fillna(0.0).astype(float)
    active = work.copy()
    active = active[active["status"].astype(str) != "retired"]
    active = active[active["cooldown_until"] <= int(current_batch)]
    active = active[active["last_seen_batch"] >= int(current_batch) - int(max_age_batches)]
    if active.empty:
        return active
    active["fragment_priority"] = (
        1.5 * active["positive_count"] - 1.0 * active["negative_count"] + 0.5 * active["source_score"]
    )
    active = active.sort_values(["fragment_priority", "last_seen_batch"], ascending=[False, False]).reset_index(
        drop=True
    )
    return active.head(max(1, int(limit))).drop(columns=["fragment_priority"], errors="ignore")


def _select_feedback_score_column(df: pd.DataFrame) -> str:
    for col in [
        "feedback_score_net",
        "score_total_net",
        "train_score_total_net",
        "feedback_score",
        "score_total_gross",
        "train_score_total",
        "train_score",
        "score_total",
        "scoreboard_score",
    ]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().any():
                return col
    return "score_total" if "score_total" in df.columns else "scoreboard_score"


def _score_basis(column: str) -> str:
    text = str(column or "").lower()
    if not text:
        return "none"
    if text.endswith("_net") or "_net_" in text:
        return "net"
    if text.endswith("_gross") or "_gross_" in text:
        return "gross"
    return "fallback"


def apply_candidate_feedback_to_registry(
    registry_df: pd.DataFrame | None,
    candidate_df: pd.DataFrame | None,
    current_batch: int,
    cooldown_batches: int = 3,
    evaluated_expressions: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Apply candidate-stage feedback to fragment registry in-place semantics.

    Rules:
    - source=feedback_mutation_v2 and (prefilter_status!=pass or sample_status=reject):
      negative_count++, rejected_count++, cooldown_until pushed.
    - source=feedback_mutation_v2 and expression in evaluated_expressions with pass status:
      positive_count++.
    """
    working = _ensure_columns(registry_df)
    summary = {"positive_updates": 0, "negative_updates": 0, "rejected_updates": 0}
    if candidate_df is None or candidate_df.empty:
        return working, summary

    eval_set = {str(x).strip() for x in (evaluated_expressions or set()) if str(x).strip()}
    batch_no = int(current_batch)
    cd = candidate_df.copy()
    if "source" not in cd.columns or "fragment_hash" not in cd.columns:
        return working, summary
    cd = cd[cd["source"].astype(str) == "feedback_mutation_v2"].copy()
    if cd.empty:
        return working, summary

    working["fragment_hash"] = working["fragment_hash"].astype(str)
    idx_map = {
        str(row["fragment_hash"]): idx for idx, row in working.iterrows() if str(row.get("fragment_hash", "")).strip()
    }
    cooldown_batches = max(1, int(cooldown_batches))

    for row in cd.to_dict(orient="records"):
        fragment_hash = str(row.get("fragment_hash", "") or "").strip()
        if not fragment_hash:
            continue
        if fragment_hash not in idx_map:
            # Keep robustness: if registry row is missing, create a minimal placeholder.
            working.loc[len(working)] = {
                "fragment_hash": fragment_hash,
                "fragment_expression": "",
                "fragment_type": "",
                "operators": "",
                "fields": "",
                "windows": "",
                "groups": "",
                "output_type": "scalar",
                "depth": 0,
                "complexity": 0,
                "source_alpha_name": "",
                "source_alpha_hash": str(row.get("parent_hash", "") or ""),
                "source_expression": str(row.get("parent_expression", "") or ""),
                "source_batch_id": "",
                "source_score": 0.0,
                "oos_score": float("nan"),
                "positive_count": 0,
                "negative_count": 0,
                "rejected_count": 0,
                "mean_child_score": float("nan"),
                "best_child_score": float("nan"),
                "last_seen_batch": batch_no,
                "cooldown_until": 0,
                "status": "active",
            }
            idx_map[fragment_hash] = int(len(working) - 1)

        idx = idx_map[fragment_hash]
        prefilter_status = str(row.get("prefilter_status", "") or "").strip()
        sample_status = str(row.get("sample_status", "") or "").strip()
        expression = str(row.get("expression", "") or "").strip()

        passed_prefilter = prefilter_status == "pass"
        sample_reject = sample_status == "reject"
        should_positive = passed_prefilter and (sample_status in {"", "pass", "skipped"}) and (expression in eval_set)
        should_negative = (not passed_prefilter) or sample_reject

        if should_positive:
            working.at[idx, "positive_count"] = int(_to_float(working.at[idx, "positive_count"])) + 1
            summary["positive_updates"] += 1
            working.at[idx, "last_seen_batch"] = batch_no

        if should_negative:
            working.at[idx, "negative_count"] = int(_to_float(working.at[idx, "negative_count"])) + 1
            working.at[idx, "rejected_count"] = int(_to_float(working.at[idx, "rejected_count"])) + 1
            working.at[idx, "cooldown_until"] = max(
                int(_to_float(working.at[idx, "cooldown_until"])),
                batch_no + cooldown_batches,
            )
            summary["negative_updates"] += 1
            summary["rejected_updates"] += 1
            working.at[idx, "last_seen_batch"] = batch_no

        cooldown_until = int(_to_float(working.at[idx, "cooldown_until"]))
        status = "cooldown" if cooldown_until > batch_no else "active"
        if str(working.at[idx, "status"]) == "retired":
            status = "retired"
        working.at[idx, "status"] = status

    return working.reset_index(drop=True), summary


def _upsert_fragment(df: pd.DataFrame, item: dict[str, Any], batch_no: int, cooldown_batches: int) -> None:
    h = str(item.get("fragment_hash", "")).strip()
    if not h:
        return
    payload = dict(item)
    if int(payload.get("negative_count", 0) or 0) > 0 and int(payload.get("positive_count", 0) or 0) <= 0:
        payload["cooldown_until"] = max(
            int(payload.get("cooldown_until", 0) or 0),
            int(batch_no) + int(cooldown_batches),
        )
        payload["status"] = "cooldown"
    payload["last_seen_batch"] = int(batch_no)
    if "fragment_hash" not in df.columns or df.empty:
        df.loc[len(df)] = payload
        return
    mask = df["fragment_hash"].astype(str) == h
    if not mask.any():
        df.loc[len(df)] = payload
        return
    idx = df.index[mask][0]
    df.at[idx, "fragment_expression"] = str(item.get("fragment_expression", df.at[idx, "fragment_expression"]))
    df.at[idx, "fragment_type"] = str(item.get("fragment_type", df.at[idx, "fragment_type"]))
    df.at[idx, "operators"] = str(item.get("operators", df.at[idx, "operators"]))
    df.at[idx, "fields"] = str(item.get("fields", df.at[idx, "fields"]))
    df.at[idx, "windows"] = str(item.get("windows", df.at[idx, "windows"]))
    df.at[idx, "groups"] = str(item.get("groups", df.at[idx, "groups"]))
    df.at[idx, "output_type"] = str(item.get("output_type", df.at[idx, "output_type"]))
    df.at[idx, "depth"] = max(int(_to_float(df.at[idx, "depth"])), int(item.get("depth", 0) or 0))
    df.at[idx, "complexity"] = max(int(_to_float(df.at[idx, "complexity"])), int(item.get("complexity", 0) or 0))
    df.at[idx, "source_alpha_name"] = str(item.get("source_alpha_name", df.at[idx, "source_alpha_name"]))
    df.at[idx, "source_alpha_hash"] = str(item.get("source_alpha_hash", df.at[idx, "source_alpha_hash"]))
    df.at[idx, "source_expression"] = str(item.get("source_expression", df.at[idx, "source_expression"]))
    df.at[idx, "source_batch_id"] = str(item.get("source_batch_id", df.at[idx, "source_batch_id"]))
    prev_score = _to_float(df.at[idx, "source_score"])
    cur_score = _to_float(item.get("source_score", 0.0))
    if abs(cur_score) >= abs(prev_score):
        df.at[idx, "source_score"] = cur_score
    if math.isfinite(_to_float(item.get("oos_score", float("nan")))):
        df.at[idx, "oos_score"] = _to_float(item.get("oos_score"))
    pos = int(_to_float(df.at[idx, "positive_count"])) + int(item.get("positive_count", 0) or 0)
    neg = int(_to_float(df.at[idx, "negative_count"])) + int(item.get("negative_count", 0) or 0)
    rej = int(_to_float(df.at[idx, "rejected_count"])) + int(item.get("rejected_count", 0) or 0)
    df.at[idx, "positive_count"] = pos
    df.at[idx, "negative_count"] = neg
    df.at[idx, "rejected_count"] = rej
    df.at[idx, "last_seen_batch"] = int(batch_no)
    if int(item.get("negative_count", 0) or 0) > 0 and int(item.get("positive_count", 0) or 0) <= 0:
        df.at[idx, "cooldown_until"] = max(
            int(_to_float(df.at[idx, "cooldown_until"])),
            int(batch_no) + int(cooldown_batches),
        )
    status = "cooldown" if int(_to_float(df.at[idx, "cooldown_until"])) > int(batch_no) else "active"
    if str(df.at[idx, "status"]) == "retired":
        status = "retired"
    df.at[idx, "status"] = status


def _iter_nodes(node: ExpressionNode) -> list[ExpressionNode]:
    out = [node]
    if isinstance(node, UnaryOpNode):
        out.extend(_iter_nodes(node.operand))
    elif isinstance(node, BinaryOpNode):
        out.extend(_iter_nodes(node.left))
        out.extend(_iter_nodes(node.right))
    elif isinstance(node, FunctionCallNode):
        for arg in node.args:
            out.extend(_iter_nodes(arg))
        for arg in node.named_args.values():
            out.extend(_iter_nodes(arg))
    return out


def _node_to_expression(node: ExpressionNode) -> str:
    if isinstance(node, LiteralNode):
        return repr(node.value)
    if isinstance(node, FieldNode):
        return str(node.name)
    if isinstance(node, UnaryOpNode):
        return f"{node.op}{_node_to_expression(node.operand)}"
    if isinstance(node, BinaryOpNode):
        return f"{_node_to_expression(node.left)} {node.op} {_node_to_expression(node.right)}"
    if isinstance(node, FunctionCallNode):
        args = [_node_to_expression(arg) for arg in node.args]
        args.extend([f"{k}={_node_to_expression(v)}" for k, v in sorted(node.named_args.items())])
        return f"{node.name}({', '.join(args)})"
    return ""


def _node_depth(node: ExpressionNode) -> int:
    if isinstance(node, (LiteralNode, FieldNode)):
        return 0
    if isinstance(node, UnaryOpNode):
        return 1 + _node_depth(node.operand)
    if isinstance(node, BinaryOpNode):
        return 1 + max(_node_depth(node.left), _node_depth(node.right))
    if isinstance(node, FunctionCallNode):
        child_depths = [_node_depth(arg) for arg in node.args] + [_node_depth(arg) for arg in node.named_args.values()]
        return 1 + (max(child_depths) if child_depths else 0)
    return 0


def _collect_fields(node: ExpressionNode) -> list[str]:
    if isinstance(node, FieldNode):
        return [str(node.name)]
    if isinstance(node, UnaryOpNode):
        return _collect_fields(node.operand)
    if isinstance(node, BinaryOpNode):
        return _collect_fields(node.left) + _collect_fields(node.right)
    if isinstance(node, FunctionCallNode):
        out: list[str] = []
        for arg in node.args:
            out.extend(_collect_fields(arg))
        for arg in node.named_args.values():
            out.extend(_collect_fields(arg))
        return out
    return []


def _collect_operators(node: ExpressionNode) -> list[str]:
    if isinstance(node, UnaryOpNode):
        return [str(node.op)] + _collect_operators(node.operand)
    if isinstance(node, BinaryOpNode):
        return [str(node.op)] + _collect_operators(node.left) + _collect_operators(node.right)
    if isinstance(node, FunctionCallNode):
        out = [str(node.name)]
        for arg in node.args:
            out.extend(_collect_operators(arg))
        for arg in node.named_args.values():
            out.extend(_collect_operators(arg))
        return out
    return []


def _collect_windows(node: ExpressionNode) -> list[int]:
    windows: list[int] = []
    if isinstance(node, FunctionCallNode):
        if node.name.startswith("ts_"):
            for arg in node.args:
                if isinstance(arg, LiteralNode):
                    try:
                        val = int(arg.value)
                    except Exception:
                        continue
                    if val > 0:
                        windows.append(val)
        for arg in node.args:
            windows.extend(_collect_windows(arg))
        for arg in node.named_args.values():
            windows.extend(_collect_windows(arg))
    elif isinstance(node, UnaryOpNode):
        windows.extend(_collect_windows(node.operand))
    elif isinstance(node, BinaryOpNode):
        windows.extend(_collect_windows(node.left))
        windows.extend(_collect_windows(node.right))
    return windows


def _fragment_type(node: ExpressionNode) -> str:
    if isinstance(node, FieldNode):
        return "field"
    if isinstance(node, UnaryOpNode):
        return "unary"
    if isinstance(node, BinaryOpNode):
        return "binary"
    if isinstance(node, FunctionCallNode):
        name = str(node.name)
        if name.startswith("ts_"):
            return "ts_wrapper"
        if name.startswith("group_"):
            return "group_wrapper"
        if name in {
            "greater",
            "less",
            "greater_equal",
            "less_equal",
            "equal",
            "not_equal",
            "is_nan",
            "is_not_nan",
        }:
            return "condition"
        if name in {"trade_when", "trade_when_hold", "if_else"}:
            return "gate"
    return "function"


def _output_type(node: ExpressionNode) -> str:
    if isinstance(node, FunctionCallNode):
        if node.name in {
            "greater",
            "less",
            "greater_equal",
            "less_equal",
            "equal",
            "not_equal",
            "is_nan",
            "is_not_nan",
        }:
            return "bool"
        if node.name in {"bucket", "densify", "group_cartesian_product"}:
            return "group"
    return "scalar"


def _looks_like_group_field(name: str) -> bool:
    low = str(name or "").lower()
    return low in {"industry", "sector", "subindustry"} or "industry" in low or "sector" in low


def _next_batch_index(df: pd.DataFrame) -> int:
    if df is None or df.empty or "last_seen_batch" not in df.columns:
        return 1
    values = pd.to_numeric(df["last_seen_batch"], errors="coerce").fillna(0).astype(int)
    return int(values.max()) + 1 if len(values) else 1


def _to_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if math.isfinite(out) else 0.0


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(columns=list(_DEFAULT_REGISTRY_COLUMNS)) if df is None else df.copy()
    for col in _DEFAULT_REGISTRY_COLUMNS:
        if col not in out.columns:
            if col in {
                "positive_count",
                "negative_count",
                "rejected_count",
                "last_seen_batch",
                "cooldown_until",
                "depth",
                "complexity",
            }:
                out[col] = 0
            elif col in {
                "source_score",
                "oos_score",
                "mean_child_score",
                "best_child_score",
            }:
                out[col] = 0.0
            else:
                out[col] = ""
    return out[list(_DEFAULT_REGISTRY_COLUMNS)]
