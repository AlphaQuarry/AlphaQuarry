from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .feedback_sampler import _extract_substructures

# 候选评分权重默认值
_DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "feedback": 0.20,
    "factor_family_feedback": 0.08,
    "fragment": 0.08,
    "parent": 0.06,
    "mutation_type": 0.05,
    "novelty": 0.18,
    "family_balance": 0.08,
    "factor_family_balance": 0.06,
    "layer_balance": 0.10,
    "field_diversity": 0.13,
    "operator_diversity": 0.08,
    "field_profile": 0.06,
    "recipe": 0.05,
    "role_pair": 0.04,
    "bucket_quality": 0.04,
    "gate_quality": 0.04,
    "sample_quality": 0.06,
    "cost": -0.05,
}


@dataclass(frozen=True)
class CandidateRankerConfig:
    min_explore_ratio: float = 0.30
    complexity_weight: float = 0.10
    use_factor_family_quota: bool = True
    family_max_selected_ratio: float = 0.45
    family_min_explore_ratio: float = 0.25
    use_layer_quota: bool = False
    layer_selection_min_ratio: dict[str, float] | None = None
    layer_selection_max_ratio: dict[str, float] | None = None
    structure_selection_min_ratio: dict[str, float] | None = None
    score_weights: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_SCORE_WEIGHTS))
    normalize_weights: bool = False
    # 自适应探索率
    adaptive_exploration: bool = False
    exploration_window: int = 10
    exploration_base_ratio: float = 0.30
    exploration_max_ratio: float = 0.60
    exploration_boost_threshold: int = 3


def compute_adaptive_explore_ratio(
    recent_results: list[dict[str, Any]],
    config: CandidateRankerConfig,
) -> float:
    """基于最近 N 轮的新因子发现率，计算自适应探索率。"""
    if not config.adaptive_exploration:
        return config.min_explore_ratio

    window = config.exploration_window
    recent = recent_results[-window:] if len(recent_results) > window else recent_results
    if not recent:
        return config.exploration_base_ratio

    rounds_without_new = sum(1 for r in recent if r.get("status") != "ok" or len(r.get("alpha_names", [])) == 0)

    if rounds_without_new >= config.exploration_boost_threshold:
        boost = min(
            config.exploration_max_ratio,
            config.exploration_base_ratio + 0.10 * (rounds_without_new - config.exploration_boost_threshold + 1),
        )
        return boost

    return config.exploration_base_ratio


class CandidateRanker:
    def __init__(self, config: CandidateRankerConfig | None = None) -> None:
        self.config = config or CandidateRankerConfig()

    def rank(
        self,
        candidate_df: pd.DataFrame,
        feedback_hints: dict[str, Any] | None = None,
        max_eval: int = 80,
    ) -> pd.DataFrame:
        if candidate_df is None or candidate_df.empty:
            return pd.DataFrame() if candidate_df is None else candidate_df.copy()
        work = candidate_df.copy()
        if "prefilter_status" in work.columns:
            work = work[work["prefilter_status"].astype(str) == "pass"].copy()
        if "sample_status" in work.columns:
            work = work[work["sample_status"].astype(str).isin(["", "pass", "skipped", "skipped_budget"])].copy()
        if work.empty:
            return work

        hints = feedback_hints or {}
        work["complexity_score"] = work.apply(_complexity_score, axis=1)
        work["feedback_score"] = work.apply(lambda row: _feedback_score(row, hints), axis=1)
        work["factor_family_feedback_score"] = work.apply(lambda row: _factor_family_feedback_score(row, hints), axis=1)
        work["fragment_score"] = work.apply(lambda row: _entity_feedback_score(row, hints, "fragment_hash"), axis=1)
        work["parent_score"] = work.apply(lambda row: _entity_feedback_score(row, hints, "parent_hash"), axis=1)
        work["mutation_type_score"] = work.apply(
            lambda row: _entity_feedback_score(row, hints, "mutation_type"), axis=1
        )
        work["novelty_score"] = work.apply(_novelty_score, axis=1)
        work["family_balance_score"] = _balance_score(work, "family")
        work["factor_family_balance_score"] = _balance_score(work, "factor_family")
        work["layer_balance_score"] = _balance_score(work, "layer")
        work["field_diversity_score"] = _diversity_score(work, "fields")
        work["operator_diversity_score"] = _diversity_score(work, "operators")
        for column in (
            "field_profile_score",
            "recipe_score",
            "role_pair_score",
            "bucket_quality_score",
            "gate_quality_score",
            "sample_quality_score",
            "cost_score",
        ):
            if column not in work.columns:
                work[column] = 0.0
            work[column] = work[column].map(_to_float).astype(float)

        # 使用可配置的权重计算候选分数
        w = dict(self.config.score_weights)
        if self.config.normalize_weights:
            total = sum(abs(v) for v in w.values())
            if total > 0:
                w = {k: v / total for k, v in w.items()}

        score_cols = [
            "feedback_score",
            "factor_family_feedback_score",
            "fragment_score",
            "parent_score",
            "mutation_type_score",
            "novelty_score",
            "family_balance_score",
            "factor_family_balance_score",
            "layer_balance_score",
            "field_diversity_score",
            "operator_diversity_score",
            "field_profile_score",
            "recipe_score",
            "role_pair_score",
            "bucket_quality_score",
            "gate_quality_score",
            "sample_quality_score",
        ]
        candidate_score = sum(
            float(w.get(col.replace("_score", ""), 0.0)) * work[col] for col in score_cols if col in work.columns
        )
        candidate_score -= abs(float(w.get("cost", 0.05))) * work.get("cost_score", 0.0)
        candidate_score -= float(self.config.complexity_weight) * work.get("complexity_score", 0.0)
        work["candidate_score"] = candidate_score

        max_n = max(1, int(max_eval))
        explore_n = int(np.ceil(max_n * max(0.0, min(1.0, float(self.config.min_explore_ratio)))))
        explore_n = min(explore_n, len(work))
        exploit_n = max(0, min(max_n - explore_n, len(work)))
        scored = work.sort_values(["candidate_score", "feedback_score"], ascending=[False, False])
        use_family_quota = bool(self.config.use_factor_family_quota) and _has_nonempty_column(work, "factor_family")
        if use_family_quota:
            exploit = _take_with_family_quota(scored, exploit_n, self.config.family_max_selected_ratio).copy()
        else:
            exploit = _take_diverse(scored, exploit_n, preferred_column="layer").copy()
        exploit["selection_bucket"] = "exploit"
        remaining = work.drop(index=exploit.index, errors="ignore")
        if use_family_quota:
            explore_floor = max(
                0,
                int(np.ceil(max_n * max(0.0, min(1.0, float(self.config.family_min_explore_ratio))))),
            )
            explore_source = (
                remaining.sort_index()
                if len(exploit) >= explore_floor
                else remaining.sort_values(["candidate_score", "feedback_score"], ascending=[False, False])
            )
            explore = _take_with_family_quota(
                explore_source,
                max_n - len(exploit),
                self.config.family_max_selected_ratio,
                existing=exploit,
            ).copy()
        else:
            explore = _take_diverse(remaining.sort_index(), max_n - len(exploit), preferred_column="layer").copy()
        explore["selection_bucket"] = "explore"
        selected = pd.concat([exploit, explore], axis=0).head(max_n)
        if bool(self.config.use_layer_quota):
            selected = _apply_layer_quota(
                selected=selected,
                pool=scored,
                max_n=max_n,
                min_ratios=dict(self.config.layer_selection_min_ratio or {}),
                max_ratios=dict(self.config.layer_selection_max_ratio or {}),
            )
        if self.config.structure_selection_min_ratio:
            selected = _apply_structure_quota(
                selected=selected,
                pool=scored,
                max_n=max_n,
                ratios=dict(self.config.structure_selection_min_ratio or {}),
            )
        return selected.reset_index(drop=True)


def _csv_items(value: Any) -> list[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def _complexity_score(row: pd.Series) -> float:
    return float(
        _to_float(row.get("operator_count", 0))
        + 0.25 * _to_float(row.get("field_count", 0))
        + 0.50 * _to_float(row.get("depth", 0))
    )


def _feedback_score(row: pd.Series, hints: dict[str, Any]) -> float:
    fields = _csv_items(row.get("fields", ""))
    operators = _csv_items(row.get("operators", ""))
    families = _csv_items(row.get("family", ""))
    layers = _csv_items(row.get("layer", ""))
    windows = _csv_items(row.get("windows", ""))
    groups = _csv_items(row.get("groups", ""))
    score = 0.0
    score += _weight_sum(fields, hints.get("field_weights", {}))
    score += _weight_sum(operators, hints.get("operator_weights", {}))
    score += _weight_sum(families, hints.get("family_weights", {}))
    score += _weight_sum(layers, hints.get("layer_weights", {}))
    score += _weight_sum(windows, hints.get("window_weights", {}))
    score += _weight_sum(groups, hints.get("group_weights", {}))
    score -= _weight_sum(fields, hints.get("negative_field_weights", {}))
    score -= _weight_sum(operators, hints.get("negative_operator_weights", {}))
    score -= _weight_sum(families, hints.get("negative_family_weights", {}))
    score -= _weight_sum(layers, hints.get("negative_layer_weights", {}))
    score -= _weight_sum(windows, hints.get("negative_window_weights", {}))
    score -= _weight_sum(groups, hints.get("negative_group_weights", {}))
    # 子结构级别反馈
    expr = str(row.get("expression", "") or "").strip()
    if expr and hints.get("substructure_weights"):
        substructures = _extract_substructures(expr)
        score += _weight_sum(substructures, hints.get("substructure_weights", {}))
    return float(score)


def _factor_family_feedback_score(row: pd.Series, hints: dict[str, Any]) -> float:
    families = _csv_items(row.get("factor_family", ""))
    score = _weight_sum(families, hints.get("factor_family_weights", {}))
    score -= _weight_sum(families, hints.get("negative_factor_family_weights", {}))
    return float(score)


def _entity_feedback_score(row: pd.Series, hints: dict[str, Any], column: str) -> float:
    key = str(row.get(column, "") or "").strip()
    if not key:
        return 0.0
    if column == "fragment_hash":
        pos = (
            float(hints.get("fragment_weights", {}).get(key, 0.0))
            if isinstance(hints.get("fragment_weights", {}), dict)
            else 0.0
        )
        neg = (
            float(hints.get("negative_fragment_weights", {}).get(key, 0.0))
            if isinstance(hints.get("negative_fragment_weights", {}), dict)
            else 0.0
        )
        return pos - neg
    if column == "parent_hash":
        pos = (
            float(hints.get("parent_weights", {}).get(key, 0.0))
            if isinstance(hints.get("parent_weights", {}), dict)
            else 0.0
        )
        neg = (
            float(hints.get("negative_parent_weights", {}).get(key, 0.0))
            if isinstance(hints.get("negative_parent_weights", {}), dict)
            else 0.0
        )
        return pos - neg
    if column == "mutation_type":
        pos = (
            float(hints.get("mutation_type_weights", {}).get(key, 0.0))
            if isinstance(hints.get("mutation_type_weights", {}), dict)
            else 0.0
        )
        neg = (
            float(hints.get("negative_mutation_type_weights", {}).get(key, 0.0))
            if isinstance(hints.get("negative_mutation_type_weights", {}), dict)
            else 0.0
        )
        return pos - neg
    return 0.0


def _novelty_score(row: pd.Series) -> float:
    return 1.0 / (1.0 + _complexity_score(row))


def _balance_score(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(1.0, index=df.index)
    counts = df[column].fillna("").astype(str).value_counts().to_dict()
    return df[column].fillna("").astype(str).map(lambda x: 1.0 / max(1.0, float(counts.get(x, 1)))).astype(float)


def _take_diverse(df: pd.DataFrame, n: int, preferred_column: str = "layer") -> pd.DataFrame:
    limit = max(0, int(n))
    if limit <= 0 or df.empty:
        return df.head(0)
    column = preferred_column if preferred_column in df.columns else "layer"
    if column not in df.columns:
        return df.head(limit)
    layer_values = df[column].fillna("").astype(str)
    nonempty_layers = [x for x in layer_values.tolist() if x]
    if len(set(nonempty_layers)) <= 1:
        return df.head(limit)

    layer_order: list[str] = []
    for value in layer_values.tolist():
        if value and value not in layer_order:
            layer_order.append(value)
    if not layer_order:
        return df.head(limit)

    remaining = df.copy()
    selected: list[Any] = []
    while len(selected) < limit and not remaining.empty:
        progressed = False
        remaining_layers = remaining[column].fillna("").astype(str)
        for layer in layer_order:
            subset = remaining[remaining_layers == layer]
            if subset.empty:
                continue
            idx = subset.index[0]
            selected.append(idx)
            remaining = remaining.drop(index=idx)
            progressed = True
            if len(selected) >= limit:
                break
            remaining_layers = remaining[column].fillna("").astype(str)
        if not progressed:
            selected.extend(remaining.head(limit - len(selected)).index.tolist())
            break
    return df.loc[selected]


def _take_with_family_quota(
    df: pd.DataFrame,
    n: int,
    family_max_selected_ratio: float,
    existing: pd.DataFrame | None = None,
) -> pd.DataFrame:
    limit = max(0, int(n))
    if limit <= 0 or df.empty:
        return df.head(0)
    if "factor_family" not in df.columns:
        return df.head(limit)
    final_limit = limit + (0 if existing is None else len(existing))
    max_per_family = max(
        1,
        int(np.ceil(max(0.0, min(1.0, float(family_max_selected_ratio))) * max(1, final_limit))),
    )
    selected: list[Any] = []
    counts: dict[str, int] = {}
    if existing is not None and not existing.empty and "factor_family" in existing.columns:
        counts.update(existing["factor_family"].fillna("").astype(str).value_counts().to_dict())
    deferred: list[Any] = []
    for idx, row in df.iterrows():
        family = str(row.get("factor_family", "") or "")
        current = int(counts.get(family, 0))
        if family and current >= max_per_family:
            deferred.append(idx)
            continue
        selected.append(idx)
        counts[family] = current + 1
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        for idx in deferred:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= limit:
                break
    return df.loc[selected]


def _apply_layer_quota(
    selected: pd.DataFrame,
    pool: pd.DataFrame,
    max_n: int,
    min_ratios: dict[str, float],
    max_ratios: dict[str, float] | None = None,
) -> pd.DataFrame:
    if selected.empty or pool.empty or "layer" not in selected.columns or "layer" not in pool.columns:
        return selected.head(max_n)
    out = selected.copy()
    selected_ids = set(out.index)
    for layer, ratio in (min_ratios or {}).items():
        layer_text = str(layer or "").strip()
        if not layer_text:
            continue
        try:
            min_count = int(np.ceil(max(0.0, min(1.0, float(ratio))) * max(1, int(max_n))))
        except Exception:
            min_count = 0
        if min_count <= 0:
            continue
        current = int((out["layer"].fillna("").astype(str) == layer_text).sum())
        if current >= min_count:
            continue
        candidates = pool[(pool["layer"].fillna("").astype(str) == layer_text) & (~pool.index.isin(selected_ids))]
        for idx, row in candidates.iterrows():
            if current >= min_count:
                break
            replaceable = out[out["layer"].fillna("").astype(str) != layer_text]
            if replaceable.empty:
                break
            replace_idx = replaceable.index[-1]
            bucket = str(out.loc[replace_idx].get("selection_bucket", "") or "explore")
            out = out.drop(index=replace_idx)
            new_row = row.copy()
            new_row["selection_bucket"] = bucket
            out = pd.concat([out, new_row.to_frame().T], axis=0)
            selected_ids.discard(replace_idx)
            selected_ids.add(idx)
            current += 1
    out = _apply_layer_max_caps(out=out, pool=pool, max_n=max_n, max_ratios=dict(max_ratios or {}))
    return out.head(max_n)


def _apply_layer_max_caps(
    out: pd.DataFrame, pool: pd.DataFrame, max_n: int, max_ratios: dict[str, float]
) -> pd.DataFrame:
    if out.empty or not max_ratios or "layer" not in out.columns or "layer" not in pool.columns:
        return out.head(max_n)
    selected_ids = set(out.index)
    for layer, ratio in max_ratios.items():
        layer_text = str(layer or "").strip()
        if not layer_text:
            continue
        try:
            max_count = int(np.floor(max(0.0, min(1.0, float(ratio))) * max(1, int(max_n))))
        except Exception:
            continue
        current_ids = out[out["layer"].fillna("").astype(str) == layer_text].index.tolist()
        if len(current_ids) <= max_count:
            continue
        drop_ids = current_ids[max_count:]
        out = out.drop(index=drop_ids)
        selected_ids.difference_update(drop_ids)
        replacement_pool = pool[(~pool.index.isin(selected_ids)) & (pool["layer"].fillna("").astype(str) != layer_text)]
        for idx, row in replacement_pool.iterrows():
            if len(out) >= max_n:
                break
            new_row = row.copy()
            new_row["selection_bucket"] = str(new_row.get("selection_bucket", "") or "explore")
            out = pd.concat([out, new_row.to_frame().T], axis=0)
            selected_ids.add(idx)
    return out.head(max_n)


def _apply_structure_quota(
    selected: pd.DataFrame, pool: pd.DataFrame, max_n: int, ratios: dict[str, float]
) -> pd.DataFrame:
    if selected.empty or pool.empty or not ratios:
        return selected.head(max_n)
    out = selected.copy()
    selected_ids = set(out.index)
    for structure, ratio in ratios.items():
        key = str(structure or "").strip().lower()
        if not key:
            continue
        try:
            min_count = int(np.ceil(max(0.0, min(1.0, float(ratio))) * max(1, int(max_n))))
        except Exception:
            min_count = 0
        if min_count <= 0:
            continue
        current = int(out.apply(lambda row: _row_has_structure(row, key), axis=1).sum())
        if current >= min_count:
            continue
        candidates = pool[
            (~pool.index.isin(selected_ids)) & pool.apply(lambda row: _row_has_structure(row, key), axis=1)
        ]
        for idx, row in candidates.iterrows():
            if current >= min_count:
                break
            replaceable = out[~out.apply(lambda item: _row_has_structure(item, key), axis=1)]
            if replaceable.empty:
                break
            replace_idx = replaceable.index[-1]
            bucket = str(out.loc[replace_idx].get("selection_bucket", "") or "explore")
            out = out.drop(index=replace_idx)
            new_row = row.copy()
            new_row["selection_bucket"] = bucket
            out = pd.concat([out, new_row.to_frame().T], axis=0)
            selected_ids.discard(replace_idx)
            selected_ids.add(idx)
            current += 1
    return out.head(max_n)


def _row_has_structure(row: pd.Series, key: str) -> bool:
    metadata = _metadata(row.get("metadata_json", "{}"))
    if key == "bucket":
        return bool(str(metadata.get("bucket_expression", "") or metadata.get("bucket_family", "") or "").strip())
    if key == "gate":
        return bool(str(metadata.get("gate_family", "") or metadata.get("gate_expression", "") or "").strip())
    if key == "recipe":
        return bool(str(metadata.get("recipe_family", "") or metadata.get("recipe_id", "") or "").strip())
    if key in {"role_pair", "role_pair_type"}:
        return bool(str(metadata.get("role_pair_type", "") or "").strip())
    return False


def _metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        import json

        parsed = json.loads(str(value or "{}"))
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _has_nonempty_column(df: pd.DataFrame, column: str) -> bool:
    if column not in df.columns:
        return False
    return df[column].fillna("").astype(str).str.strip().ne("").any()


def _diversity_score(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(1.0, index=df.index)
    counts: dict[str, int] = {}
    for value in df[column].fillna("").astype(str):
        for item in _csv_items(value):
            counts[item] = counts.get(item, 0) + 1

    def score(value: Any) -> float:
        items = _csv_items(value)
        if not items:
            return 1.0
        return float(np.mean([1.0 / max(1.0, float(counts.get(item, 1))) for item in items]))

    return df[column].fillna("").astype(str).map(score).astype(float)


def _weight_sum(items: list[str], weights: Any) -> float:
    if not isinstance(weights, dict):
        return 0.0
    return float(sum(float(weights.get(item, 0.0)) for item in items))


def _to_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if np.isfinite(out) else 0.0
