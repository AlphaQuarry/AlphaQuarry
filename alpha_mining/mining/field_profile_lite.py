from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..panel_store import PanelStore
from .field_semantics import infer_field_semantic
from .field_universe import FieldUniverse


@dataclass(frozen=True)
class FieldProfile:
    field: str
    role: str
    field_profile_status: str
    coverage_score: float
    finite_score: float
    semantic_priority_score: float
    feedback_score: float
    negative_feedback_score: float
    field_profile_score: float
    recommended_windows: tuple[int, ...]


def build_field_profiles(
    field_universe: FieldUniverse,
    *,
    panel_store: PanelStore | None,
    feedback_hints: dict[str, Any] | None = None,
    min_coverage: float = 0.0,
    min_finite_rate: float = 0.0,
    top_fields_per_family: int = 50,
) -> dict[str, FieldProfile]:
    hints = feedback_hints or {}
    field_weights = hints.get("field_weights", {}) if isinstance(hints, dict) else {}
    negative_weights = hints.get("negative_field_weights", {}) if isinstance(hints, dict) else {}
    categories = {spec.name: tuple(spec.categories) for spec in field_universe.specs}
    out: dict[str, FieldProfile] = {}
    for field in field_universe.scalar_fields:
        semantic = infer_field_semantic(field, categories.get(field, ()))
        coverage, finite = _coverage_and_finite(panel_store, field)
        feedback = _normalized_weight(field_weights, field)
        negative = _normalized_weight(negative_weights, field)
        priority = _semantic_priority(semantic.role)
        profile_score = 0.30 * coverage + 0.25 * finite + 0.25 * priority + 0.15 * feedback - 0.10 * negative
        status_reasons: list[str] = []
        if coverage < max(0.0, min(1.0, float(min_coverage))):
            profile_score *= 0.5
            status_reasons.append("low_coverage")
        if finite < max(0.0, min(1.0, float(min_finite_rate))):
            profile_score *= 0.5
            status_reasons.append("low_finite")
        out[field] = FieldProfile(
            field=field,
            role=semantic.role,
            field_profile_status=",".join(status_reasons) if status_reasons else "pass",
            coverage_score=float(coverage),
            finite_score=float(finite),
            semantic_priority_score=float(priority),
            feedback_score=float(feedback),
            negative_feedback_score=float(negative),
            field_profile_score=float(profile_score),
            recommended_windows=recommended_windows_for_role(semantic.role),
        )
    return _cap_by_family(out, max_per_family=max(1, int(top_fields_per_family)))


def recommended_windows_for_role(role: str) -> tuple[int, ...]:
    value = str(role or "").strip().lower()
    if value in {"price", "liquidity", "moneyflow", "technical"}:
        return (5, 10, 22, 66)
    if value in {"valuation", "finance", "analyst", "size"}:
        return (22, 66, 132)
    if value == "chip":
        return (10, 22, 66)
    return (5, 10, 22, 66, 132)


def aggregate_field_profile_score(fields: list[str] | tuple[str, ...], profiles: dict[str, FieldProfile]) -> float:
    values = [profiles[field].field_profile_score for field in fields if field in profiles]
    if not values:
        return 0.0
    return float(np.mean(values))


def aggregate_recommended_windows(
    fields: list[str] | tuple[str, ...], profiles: dict[str, FieldProfile]
) -> tuple[int, ...]:
    values: set[int] = set()
    for field in fields:
        profile = profiles.get(field)
        if profile is not None:
            values.update(int(w) for w in profile.recommended_windows)
    return tuple(sorted(values))


def _coverage_and_finite(panel_store: PanelStore | None, field: str) -> tuple[float, float]:
    if panel_store is None or not panel_store.has_field(field):
        return 0.5, 0.5
    try:
        panel = panel_store.get_scalar(field)
    except Exception:
        return 0.5, 0.5
    total = int(panel.size)
    if total <= 0:
        return 0.0, 0.0
    numeric = panel.apply(lambda col: np.asarray(col, dtype="float64"), axis=0, result_type="broadcast")
    values = numeric.to_numpy(dtype="float64", copy=False)
    non_nan = ~np.isnan(values)
    finite = np.isfinite(values)
    coverage = float(non_nan.sum()) / float(total)
    finite_score = float(finite.sum()) / float(total)
    return coverage, finite_score


def _semantic_priority(role: str) -> float:
    return {
        "moneyflow": 1.00,
        "liquidity": 0.92,
        "price": 0.86,
        "size": 0.80,
        "valuation": 0.74,
        "chip": 0.70,
        "technical": 0.68,
        "finance": 0.62,
        "analyst": 0.58,
    }.get(str(role or "").lower(), 0.40)


def _normalized_weight(weights: Any, field: str) -> float:
    if not isinstance(weights, dict):
        return 0.0
    try:
        value = float(weights.get(field, 0.0))
    except Exception:
        value = 0.0
    return max(0.0, min(1.0, value))


def _cap_by_family(profiles: dict[str, FieldProfile], *, max_per_family: int) -> dict[str, FieldProfile]:
    if max_per_family <= 0:
        return profiles
    counts: dict[str, int] = {}
    kept: dict[str, FieldProfile] = {}
    for field, profile in sorted(
        profiles.items(),
        key=lambda kv: (kv[1].field_profile_score, kv[0]),
        reverse=True,
    ):
        current = counts.get(profile.role, 0)
        if current >= max_per_family:
            kept[field] = FieldProfile(
                field=profile.field,
                role=profile.role,
                field_profile_status=profile.field_profile_status,
                coverage_score=profile.coverage_score,
                finite_score=profile.finite_score,
                semantic_priority_score=profile.semantic_priority_score,
                feedback_score=profile.feedback_score,
                negative_feedback_score=profile.negative_feedback_score,
                field_profile_score=profile.field_profile_score * 0.5,
                recommended_windows=profile.recommended_windows,
            )
            continue
        kept[field] = profile
        counts[profile.role] = current + 1
    return kept
