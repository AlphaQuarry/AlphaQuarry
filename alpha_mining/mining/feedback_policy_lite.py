from __future__ import annotations

import json
from typing import Any

import pandas as pd


POLICY_KEYS = (
    "gate_family_weights",
    "negative_gate_family_weights",
    "bucket_family_weights",
    "negative_bucket_family_weights",
    "recipe_weights",
    "negative_recipe_weights",
    "role_pair_type_weights",
    "negative_role_pair_type_weights",
    "operator_tier_weights",
    "negative_operator_tier_weights",
)


def build_feedback_policy_hints(scoreboard_df: pd.DataFrame | None) -> dict[str, Any]:
    hints: dict[str, Any] = {key: {} for key in POLICY_KEYS}
    hints["score_column"] = ""
    hints["score_basis"] = "none"
    if scoreboard_df is None or scoreboard_df.empty or "metadata_json" not in scoreboard_df.columns:
        return hints
    score_col = _select_score_column(scoreboard_df)
    hints["score_column"] = score_col
    hints["score_basis"] = _score_basis(score_col)
    for _, row in scoreboard_df.iterrows():
        metadata = _parse_metadata(row.get("metadata_json", "{}"))
        if not metadata:
            continue
        score = _to_float(row.get(score_col, 0.0)) if score_col else 0.0
        positive = score > 0
        magnitude = min(1.0, abs(score))
        if magnitude <= 0:
            continue
        _add_weight(
            hints,
            "recipe_weights" if positive else "negative_recipe_weights",
            metadata.get("recipe_family"),
            magnitude,
        )
        _add_weight(
            hints,
            "recipe_weights" if positive else "negative_recipe_weights",
            metadata.get("recipe_id"),
            magnitude * 0.5,
        )
        _add_weight(
            hints,
            "gate_family_weights" if positive else "negative_gate_family_weights",
            metadata.get("gate_family"),
            magnitude,
        )
        _add_weight(
            hints,
            "bucket_family_weights" if positive else "negative_bucket_family_weights",
            metadata.get("bucket_family"),
            magnitude,
        )
        _add_weight(
            hints,
            "role_pair_type_weights" if positive else "negative_role_pair_type_weights",
            metadata.get("role_pair_type"),
            magnitude,
        )
        _add_weight(
            hints,
            "operator_tier_weights" if positive else "negative_operator_tier_weights",
            metadata.get("operator_tier"),
            magnitude,
        )
    return hints


def merge_feedback_policy_hints(base: dict[str, Any] | None, policy: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base or {})
    for key, value in dict(policy or {}).items():
        if not isinstance(value, dict):
            continue
        target = dict(merged.get(key, {}) or {}) if isinstance(merged.get(key, {}), dict) else {}
        for name, weight in value.items():
            if not str(name or "").strip():
                continue
            target[str(name)] = float(target.get(str(name), 0.0)) + _to_float(weight)
        merged[key] = target
    return merged


def _parse_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(str(value or "{}"))
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _select_score_column(df: pd.DataFrame) -> str:
    # 优先使用 effectiveness_score（0-100 分制，语义最清晰）
    for column in (
        "effectiveness_score",
        "feedback_score_net",
        "score_total_net",
        "train_score_total_net",
        "feedback_score",
        "score_total_gross",
        "train_score_total",
        "train_score",
        "score_total",
        "scoreboard_score",
    ):
        if column in df.columns:
            return column
    return ""


def _score_basis(column: str) -> str:
    text = str(column or "").lower()
    if not text:
        return "none"
    if text.endswith("_net") or "_net_" in text:
        return "net"
    if text.endswith("_gross") or "_gross_" in text:
        return "gross"
    return "fallback"


def _add_weight(hints: dict[str, dict[str, float]], key: str, name: Any, value: float) -> None:
    text = str(name or "").strip()
    if not text:
        return
    hints.setdefault(key, {})
    hints[key][text] = float(hints[key].get(text, 0.0)) + float(value)


def _to_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out
