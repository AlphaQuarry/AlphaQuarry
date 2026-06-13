from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..hashing import expression_hash, normalize_expression, structural_hash
from ..parser import parse_expression
from ..panel_store import PanelStore
from ..atomic_io import atomic_write_dataframe_csv, atomic_write_json
from .bucket_quality_lite import BucketQualityConfig, evaluate_bucket_quality
from .candidate import CandidateRecord
from .candidate_prefilter import CandidatePrefilter
from .candidate_ranker import (
    CandidateRanker,
    CandidateRankerConfig,
    _DEFAULT_SCORE_WEIGHTS,
)
from .explore import (
    DeepExploreConfig,
    build_operator_search_space,
    build_signature_aware_search_space,
)
from .expression_layers import LayeredBuilderConfig, LayeredExpressionBuilder
from .factor_family import infer_factor_family_mix, primary_factor_family_from_mix
from .feedback_policy_lite import (
    build_feedback_policy_hints,
    merge_feedback_policy_hints,
)
from .field_profile_lite import build_field_profiles
from .field_semantics import infer_field_semantic
from .field_universe import FieldUniverse, build_field_universe
from .field_preprocessing import FieldExpressionFactory
from .fragment_mutation import (
    MutationConfig,
    generate_crossover_candidates,
    generate_mutation_candidates,
)
from .pair_generator import build_pair_expression_space
from .sample_evaluator import SampleEvaluator, SampleEvaluatorConfig
from .search import build_search_space
from .template_loader import load_templates


def plan_candidates(
    panel_store: PanelStore,
    config: Any,
    existing_hashes: set[str] | None = None,
    batch_id: str = "",
    feedback_hints: dict[str, Any] | None = None,
    iteration: int = 0,
    sample_panel_store_loader: Callable[[pd.DataFrame], PanelStore | None] | None = None,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    field_universe = build_field_universe(
        panel_store=panel_store,
        explicit_include_fields=getattr(config, "include_fields", ()),
        explicit_exclude_fields=getattr(config, "exclude_fields", ()),
        group_fields=getattr(config, "group_fields", ()),
        vector_fields=getattr(config, "vector_fields", ()),
        search_field_universe=getattr(config, "search_field_universe", ()),
    )
    field_universe = _filter_field_universe_by_factor_family(
        field_universe,
        include_families=getattr(config, "include_factor_families", ()),
        exclude_families=getattr(config, "exclude_factor_families", ()),
    )
    fields = set(field_universe.scalar_fields + field_universe.vector_fields)
    groups = set(field_universe.group_fields)
    deep_cfg = getattr(config, "deep_explore_config", DeepExploreConfig())
    top_field_preprocess_cfg = getattr(config, "field_preprocessing_config", None)
    if top_field_preprocess_cfg is not None:
        deep_cfg = replace(deep_cfg, field_preprocess_config=top_field_preprocess_cfg)
    field_expr_map = FieldExpressionFactory(getattr(deep_cfg, "field_preprocess_config", None)).expression_map(
        field_universe.scalar_fields
    )
    mode = str(getattr(config, "search_mode", "operator_only")).strip().lower()
    effective_feedback_hints = dict(feedback_hints or {})
    feedback_policy_meta = {
        "enabled": bool(getattr(config, "feedback_policy_lite_enabled", True)),
        "history_rows": 0,
    }
    if mode == "layered_v2" and bool(getattr(config, "feedback_policy_lite_enabled", True)):
        scoreboard_path = str(effective_feedback_hints.get("scoreboard_path", "") or "").strip()
        scoreboard_df = pd.DataFrame()
        if scoreboard_path:
            try:
                path = Path(scoreboard_path)
                if path.exists():
                    scoreboard_df = pd.read_csv(path)
            except Exception:
                scoreboard_df = pd.DataFrame()
        feedback_policy_meta["history_rows"] = int(len(scoreboard_df))
        policy_hints = build_feedback_policy_hints(scoreboard_df)
        feedback_policy_meta["score_column"] = str(policy_hints.get("score_column", "") or "")
        feedback_policy_meta["score_basis"] = str(policy_hints.get("score_basis", "none") or "none")
        effective_feedback_hints = merge_feedback_policy_hints(effective_feedback_hints, policy_hints)
        effective_feedback_hints["feedback_score_column"] = feedback_policy_meta["score_column"]
        effective_feedback_hints["feedback_score_basis"] = feedback_policy_meta["score_basis"]
    if mode == "layered_v2" and bool(getattr(config, "field_profile_lite_enabled", True)):
        effective_feedback_hints["field_profiles"] = build_field_profiles(
            field_universe,
            panel_store=panel_store,
            feedback_hints=effective_feedback_hints,
            min_coverage=float(getattr(config, "field_profile_lite_min_coverage", 0.20)),
            min_finite_rate=float(getattr(config, "field_profile_lite_min_finite_rate", 0.80)),
            top_fields_per_family=int(getattr(config, "field_profile_lite_top_fields_per_family", 50)),
        )
    generated: list[dict[str, Any]] = []

    def add_generated(source: str, expr: str, metadata: dict[str, Any] | None = None) -> None:
        text = str(expr or "").strip()
        if not text:
            return
        generated.append(
            {
                "source": str(source or ""),
                "expression": text,
                "metadata": dict(metadata or {}),
            }
        )

    if mode in {"template_only", "deep_hybrid"}:
        templates = load_templates(
            include_families=set(config.template_include_families) if config.template_include_families else None
        )
        template_items = build_search_space(
            templates=templates,
            pools=dict(config.template_pool_override or {}),
            include_families=set(config.template_include_families) if config.template_include_families else None,
            available_fields=set(field_universe.scalar_fields),
            available_groups=groups,
            available_field_specs=list(field_universe.specs),
            skip_templates_with_missing_group=config.mining_config.skip_templates_with_missing_group,
            field_preprocess_config=getattr(deep_cfg, "field_preprocess_config", None),
        )
        for source, expr in template_items:
            add_generated(source, expr)

    if mode in {"operator_only", "deep_hybrid"}:
        if bool(getattr(config, "use_signature_generator", True)):
            signature_items = build_signature_aware_search_space(
                available_fields=fields,
                available_groups=groups,
                config=deep_cfg,
                field_specs=list(field_universe.specs),
                excluded_fields=set(field_universe.excluded_fields),
            )
            for source, expr in signature_items:
                add_generated(source, expr)
        else:
            operator_items = build_operator_search_space(
                available_fields=set(field_universe.scalar_fields),
                available_groups=groups,
                config=deep_cfg,
                excluded_fields=set(field_universe.excluded_fields),
            )
            for source, expr in operator_items:
                add_generated(source, expr)
        pair_items = build_pair_expression_space(
            scalar_fields=field_universe.scalar_fields,
            group_fields=field_universe.group_fields,
            windows=list(deep_cfg.windows),
            max_pairs=min(200, max(0, int(deep_cfg.max_binary_pairs))),
            random_seed=int(deep_cfg.random_seed),
            field_expression_map=field_expr_map,
        )
        for source, expr in pair_items:
            add_generated(source, expr)

    if mode == "layered_v2":
        layer_cfg = LayeredBuilderConfig(
            max_order=int(getattr(config, "layer_max_order", 4)),
            max_candidates=int(
                getattr(
                    config,
                    "layer_max_candidates",
                    max(400, int(getattr(config, "max_eval_expressions", 80)) * 4),
                )
            ),
            layer_budgets=dict(getattr(config, "layer_budgets", {}) or {}),
            windows=tuple(int(w) for w in getattr(deep_cfg, "windows", (5, 10, 22, 66, 132))),
            include_gates=bool(getattr(config, "layer_include_gates", True)),
            enable_stateful_phase2_ops=bool(getattr(config, "enable_stateful_phase2_ops", False)),
            random_seed=int(getattr(deep_cfg, "random_seed", 42)),
            field_rotation_focus_count=int(getattr(config, "field_rotation_focus_count", 0)),
            field_rotation_iteration=int(iteration or 0),
            budget_rotation_mode=str(getattr(config, "budget_rotation_mode", "none")),
            field_preprocess_config=getattr(deep_cfg, "field_preprocess_config", None) or None,
            layer_gate_families=tuple(
                str(x)
                for x in getattr(
                    config,
                    "layer_gate_families",
                    (
                        "liquidity_activity",
                        "moneyflow_pressure",
                        "price_trend",
                        "industry_activity",
                    ),
                )
                or (
                    "liquidity_activity",
                    "moneyflow_pressure",
                    "price_trend",
                    "industry_activity",
                )
            ),
            layer_gate_max_total=int(getattr(config, "layer_gate_max_total", 24)),
            layer_gate_max_per_family=int(getattr(config, "layer_gate_max_per_family", 6)),
            layer_gate_seed_max=int(getattr(config, "layer_gate_seed_max", 18)),
            layer_gate_templates=tuple(
                str(x) for x in getattr(config, "layer_gate_templates", ("if_else_zero",)) or ("if_else_zero",)
            ),
            layer_enable_event_gates=bool(getattr(config, "layer_enable_event_gates", False)),
            layer_enable_bucket_groups=bool(getattr(config, "layer_enable_bucket_groups", True)),
            layer_bucket_max_groups=int(getattr(config, "layer_bucket_max_groups", 12)),
            layer_bucket_max_composite_groups=int(getattr(config, "layer_bucket_max_composite_groups", 6)),
            layer_bucket_ranges=tuple(
                str(x) for x in getattr(config, "layer_bucket_ranges", ("0,1,0.2",)) or ("0,1,0.2",)
            ),
            layer_bucket_field_families=tuple(
                str(x)
                for x in getattr(
                    config,
                    "layer_bucket_field_families",
                    ("size", "liquidity", "valuation", "chip", "technical"),
                )
            ),
            layer_bucket_use_composite_industry=bool(getattr(config, "layer_bucket_use_composite_industry", True)),
            layer_bucket_l1_max_total=int(getattr(config, "layer_bucket_l1_max_total", 24)),
            layer_bucket_l2_max_total=int(getattr(config, "layer_bucket_l2_max_total", 20)),
            max_field_count_for_bucket_l2=int(getattr(getattr(config, "mining_config", None), "max_field_count", 4)),
            layer_enable_recipe_lite=bool(getattr(config, "layer_enable_recipe_lite", True)),
            layer_recipe_max_total=int(getattr(config, "layer_recipe_max_total", 80)),
            layer_recipe_max_per_family=int(getattr(config, "layer_recipe_max_per_family", 16)),
            layer_role_pair_max_total=int(getattr(config, "layer_role_pair_max_total", 80)),
            layer_cross_family_pair_ratio=float(getattr(config, "layer_cross_family_pair_ratio", 0.15)),
            field_profile_lite_enabled=bool(getattr(config, "field_profile_lite_enabled", True)),
            field_profile_lite_min_coverage=float(getattr(config, "field_profile_lite_min_coverage", 0.20)),
            field_profile_lite_min_finite_rate=float(getattr(config, "field_profile_lite_min_finite_rate", 0.80)),
            field_profile_lite_top_fields_per_family=int(
                getattr(config, "field_profile_lite_top_fields_per_family", 50)
            ),
            feedback_policy_lite_enabled=bool(getattr(config, "feedback_policy_lite_enabled", True)),
            layer_operator_tier=str(getattr(config, "layer_operator_tier", "stable")),
            layer_operator_expansion_max_total=int(getattr(config, "layer_operator_expansion_max_total", 100)),
        )
        layered_builder = LayeredExpressionBuilder()
        for item in layered_builder.build(
            field_universe=field_universe,
            feedback_hints=effective_feedback_hints,
            config=layer_cfg,
            existing_hashes=existing_hashes,
        ):
            add_generated(
                item.source,
                item.expression,
                {
                    "layer": item.layer,
                    "layer_order": int(item.layer_order),
                    "layer_family": item.layer_family,
                    "parent_expression": item.parent_expression,
                    "parent_hash": item.parent_hash,
                    "builder_source": item.builder_source,
                    "windows": item.windows,
                    **dict(item.metadata or {}),
                },
            )

    if bool(getattr(config, "enable_feedback_mutation", False)):
        hints = effective_feedback_hints
        active_fragments = hints.get("active_fragments", [])
        fragment_df = pd.DataFrame(active_fragments) if isinstance(active_fragments, list) else pd.DataFrame()
        if not fragment_df.empty:
            ratio = float(getattr(config, "mutation_budget_ratio", 0.15) or 0.15)
            ratio = max(0.0, min(1.0, ratio))
            base_budget = max(1, len(generated))
            mutation_budget = int(base_budget * ratio)
            if mutation_budget <= 0 and ratio > 0:
                mutation_budget = 1
            role_map = _field_role_map(field_universe)
            mutation_candidates = generate_mutation_candidates(
                fragments_df=fragment_df,
                field_roles=role_map,
                group_fields=sorted(groups),
                existing_hashes=existing_hashes or set(),
                config=MutationConfig(
                    windows=tuple(int(w) for w in getattr(deep_cfg, "windows", (5, 10, 22, 66, 132))),
                    max_mutations=max(0, int(mutation_budget)),
                    max_children_per_parent=max(1, int(getattr(config, "mutation_max_children_per_parent", 3))),
                    enable_stateful=bool(getattr(config, "enable_stateful_phase2_ops", False)),
                    stateful_ratio_cap=max(
                        0.0,
                        min(
                            1.0,
                            float(getattr(config, "mutation_stateful_ratio_cap", 0.10) or 0.10),
                        ),
                    ),
                    random_seed=int(getattr(deep_cfg, "random_seed", 42)) + int(iteration or 0),
                    enable_operator_swap=bool(getattr(config, "mutation_enable_operator_swap", True)),
                    rejected_patterns=tuple(getattr(config, "mutation_rejected_patterns", ()) or ()),
                ),
            )
            for item in mutation_candidates:
                add_generated(
                    "feedback_mutation_v2",
                    str(item.get("expression", "")),
                    {
                        "layer": "M1",
                        "layer_order": 5,
                        "layer_family": "feedback_mutation",
                        "parent_expression": str(item.get("parent_expression", "")),
                        "parent_hash": str(item.get("parent_hash", "")),
                        "builder_source": "feedback_mutation_v2",
                        "mutation_type": str(item.get("mutation_type", "")),
                        "fragment_hash": str(item.get("fragment_hash", "")),
                        "feedback_source": str(item.get("feedback_source", "feedback_mutation_v2")),
                        "windows": item.get("windows", ""),
                    },
                )

        # 交叉变异：从两个高分因子中提取片段组合
        if bool(getattr(config, "mutation_enable_crossover", True)):
            top_exprs = [str(r.get("expression", "")) for r in (active_fragments or [])[:20]]
            top_exprs = [e for e in top_exprs if e]
            if len(top_exprs) >= 2:
                crossover_budget = max(1, mutation_budget // 3)
                crossover_candidates = generate_crossover_candidates(
                    top_expressions=top_exprs,
                    existing_hashes=existing_hashes or set(),
                    max_crossovers=crossover_budget,
                    random_seed=int(getattr(deep_cfg, "random_seed", 42)) + int(iteration or 0),
                )
                for item in crossover_candidates:
                    add_generated(
                        "crossover_mutation",
                        str(item.get("expression", "")),
                        {
                            "layer": "M2",
                            "layer_order": 6,
                            "layer_family": "crossover_mutation",
                            "parent_expression": str(item.get("parent_expression", "")),
                            "parent_hash": str(item.get("parent_hash", "")),
                            "builder_source": "crossover_mutation",
                            "mutation_type": "crossover",
                            "fragment_hash": "",
                            "feedback_source": "crossover_mutation",
                            "windows": "",
                        },
                    )

    prefilter = CandidatePrefilter(
        field_kinds=field_universe.kind_map(),
        max_operator_count=int(config.mining_config.max_operator_count),
        max_field_count=int(config.mining_config.max_field_count),
        max_depth=max(
            2,
            int(deep_cfg.max_depth) + 2,
            int(getattr(config, "layer_max_order", 0)) + 2,
        ),
        existing_hashes=existing_hashes or set(),
        reject_naked_division=True,
        preprocess_operator_exemptions={"ts_backfill", "winsorize"},
    )
    rows: list[dict[str, Any]] = []
    seen_batch_hashes: set[str] = set()
    for idx, item in enumerate(generated, start=1):
        source = str(item.get("source", ""))
        expr = str(item.get("expression", ""))
        item_meta = dict(item.get("metadata", {}) or {})
        result = prefilter.check(expr)
        if result.passed and (
            result.expression_hash in seen_batch_hashes or result.canonical_hash in seen_batch_hashes
        ):
            result = CandidatePrefilterResultShim(result, "dedup", "duplicate_in_batch")
        if result.passed:
            seen_batch_hashes.add(result.expression_hash)
            if result.canonical_hash:
                seen_batch_hashes.add(result.canonical_hash)
        family, pair_key = _family_from_source(source)
        family = str(item_meta.get("layer_family") or family)
        factor_family, factor_family_mix_json = _factor_family_for_fields(field_universe, result.fields)
        primary_factor_family = primary_factor_family_from_mix(factor_family, factor_family_mix_json)
        executable_expr = str(result.canonical_expression or expr).strip() if result.passed else expr
        executable_hash = (
            expression_hash(executable_expr) if result.passed and executable_expr else result.expression_hash
        )
        normalized_expression = (
            normalize_expression(executable_expr) if result.passed and executable_expr else result.normalized_expression
        )
        struct_hash = ""
        try:
            struct_hash = structural_hash(parse_expression(executable_expr))
        except Exception:
            struct_hash = ""
        record = CandidateRecord(
            candidate_id=f"{batch_id or 'candidate'}_{idx:05d}",
            expression=executable_expr,
            normalized_expression=normalized_expression,
            expression_hash=executable_hash,
            structural_hash=struct_hash,
            canonical_expression=result.canonical_expression,
            canonical_hash=result.canonical_hash,
            source=source,
            generation_mode=mode,
            family=family,
            original_expression=expr,
            simplified_expression=result.canonical_expression,
            factor_family=factor_family,
            factor_family_mix_json=factor_family_mix_json,
            primary_factor_family=primary_factor_family,
            lint_passed=bool(result.passed),
            lint_reject_reason=str(result.reject_reason or ""),
            lint_status=str(getattr(result, "lint_status", "") or ("passed" if result.passed else "rejected")),
            lint_warning_reason=str(getattr(result, "lint_warning_reason", "") or ""),
            layer=str(item_meta.get("layer", "")),
            layer_family=str(item_meta.get("layer_family", "")),
            parent_expression=str(item_meta.get("parent_expression", "")),
            parent_hash=str(item_meta.get("parent_hash", "")),
            mutation_type=str(item_meta.get("mutation_type", "")),
            fragment_hash=str(item_meta.get("fragment_hash", "")),
            feedback_source=str(item_meta.get("feedback_source", "")),
            builder_source=str(item_meta.get("builder_source", "")),
            layer_order=int(item_meta.get("layer_order", 0) or 0),
            template_id=source if not source.startswith("pair_") and ":" not in source else "",
            fields=",".join(result.fields),
            field_roles=json.dumps(
                {f: field_universe.kind_map().get(f, "") for f in result.fields},
                ensure_ascii=False,
                sort_keys=True,
            ),
            groups=",".join([f for f in result.fields if field_universe.kind_map().get(f) == "group"]),
            operators=",".join(result.operators),
            operator_count=int(result.operator_count),
            field_count=int(result.field_count),
            depth=int(result.depth),
            windows=_format_windows(item_meta.get("windows", tuple(deep_cfg.windows))),
            pair_key=pair_key,
            random_seed=int(deep_cfg.random_seed),
            generation_iteration=0,
            field_profile_score=float(item_meta.get("field_profile_score", 0.0) or 0.0),
            recipe_score=_metadata_policy_score(
                item_meta,
                effective_feedback_hints,
                "recipe_family",
                "recipe_weights",
                "negative_recipe_weights",
            ),
            role_pair_score=_metadata_policy_score(
                item_meta,
                effective_feedback_hints,
                "role_pair_type",
                "role_pair_type_weights",
                "negative_role_pair_type_weights",
            ),
            bucket_quality_score=_metadata_policy_score(
                item_meta,
                effective_feedback_hints,
                "bucket_family",
                "bucket_family_weights",
                "negative_bucket_family_weights",
            ),
            gate_quality_score=_metadata_policy_score(
                item_meta,
                effective_feedback_hints,
                "gate_family",
                "gate_family_weights",
                "negative_gate_family_weights",
            ),
            sample_quality_score=0.0,
            bucket_sample_quality_score=0.0,
            cost_score=float(max(0, int(result.operator_count) - 4)) / 10.0,
            prefilter_status="pass" if result.passed else "reject",
            reject_stage=result.reject_stage,
            reject_reason=result.reject_reason,
            metadata_json=json.dumps(
                {
                    "field_source": getattr(config, "search_field_source", ""),
                    "include_fields": list(getattr(config, "include_fields", ())),
                    "builder_source": str(item_meta.get("builder_source", "")),
                    "feedback_source": str(item_meta.get("feedback_source", "")),
                    "mutation_type": str(item_meta.get("mutation_type", "")),
                    "fragment_hash": str(item_meta.get("fragment_hash", "")),
                    "gate_family": str(item_meta.get("gate_family", "")),
                    "gate_expression": str(item_meta.get("gate_expression", "")),
                    "bucket_family": str(item_meta.get("bucket_family", "")),
                    "bucket_expression": str(item_meta.get("bucket_expression", "")),
                    "bucket_source_field": str(item_meta.get("bucket_source_field", "")),
                    "bucket_source_family": str(item_meta.get("bucket_source_family", "")),
                    "bucket_range": str(item_meta.get("bucket_range", "")),
                    "base_group": str(item_meta.get("base_group", "")),
                    "group_complexity": int(item_meta.get("group_complexity", 0) or 0),
                    "recipe_id": str(item_meta.get("recipe_id", "")),
                    "recipe_family": str(item_meta.get("recipe_family", "")),
                    "role_pair_type": str(item_meta.get("role_pair_type", "")),
                    "field_profile_score": float(item_meta.get("field_profile_score", 0.0) or 0.0),
                    "field_profile_status": _field_profile_status_for_fields(
                        result.fields,
                        effective_feedback_hints.get("field_profiles", {})
                        if isinstance(effective_feedback_hints, dict)
                        else {},
                    ),
                    "profile_recommended_windows": list(item_meta.get("profile_recommended_windows", []) or []),
                    "operator_tier": str(item_meta.get("operator_tier", "")),
                    "generated_group_type": str(item_meta.get("generated_group_type", "")),
                    "context_fields": list(item_meta.get("context_fields", []) or []),
                },
                ensure_ascii=False,
            ),
        )
        rows.append(record.to_dict())

    candidate_df = pd.DataFrame(rows)
    if candidate_df.empty:
        candidate_df = pd.DataFrame(columns=list(CandidateRecord("", "", "", "", "", "", "", "").to_dict().keys()))
    sample_df = pd.DataFrame()
    sample_prefilter_meta: dict[str, Any] = {}
    if bool(getattr(config, "enable_sample_prefilter", False)) and not candidate_df.empty:
        sample_panel_store = panel_store
        sample_prefilter_meta["panel_source"] = "bootstrap"
        if sample_panel_store_loader is not None:
            try:
                loaded_store = sample_panel_store_loader(candidate_df)
            except Exception as exc:
                loaded_store = None
                sample_prefilter_meta["loader_error"] = f"{type(exc).__name__}: {exc}"
            if loaded_store is not None:
                sample_panel_store = loaded_store
                sample_prefilter_meta["panel_source"] = "candidate_loader"
        candidate_df, sample_df = _apply_sample_prefilter(
            candidate_df=candidate_df, panel_store=sample_panel_store, config=config
        )

    max_eval = max(1, int(getattr(config, "max_eval_expressions", 80)))
    if bool(getattr(config, "enable_candidate_ranking", True)):
        mutation_min_count = _non_negative_int(getattr(config, "mutation_min_selected_count", 0), 0)
        mutation_min_ratio = _normalized_ratio(getattr(config, "mutation_min_selected_ratio", 0.0), 0.0)
        min_explore = float(getattr(config, "feedback_min_explore_ratio", 0.30))
        # 自适应探索率：覆盖配置值
        if "adaptive_explore_ratio" in effective_feedback_hints:
            min_explore = float(effective_feedback_hints["adaptive_explore_ratio"])
        if bool(getattr(config, "enable_feedback_mutation", False)):
            min_explore = max(0.30, min(1.0, min_explore))
        rank_max_eval = int(max_eval)
        if bool(getattr(config, "enable_feedback_mutation", False)) and (
            mutation_min_count > 0 or mutation_min_ratio > 0.0
        ):
            rank_max_eval = max(int(max_eval), _eligible_pass_pool_size(candidate_df))
        # 解析 score_weights: 以默认权重为基础，用 JSON 覆盖
        score_w = dict(_DEFAULT_SCORE_WEIGHTS)
        _sw_raw = str(getattr(config, "score_weights_json", "") or "").strip()
        if _sw_raw:
            try:
                _sw = json.loads(_sw_raw)
                if isinstance(_sw, dict):
                    score_w.update({str(k): float(v) for k, v in _sw.items()})
            except Exception:
                pass
        ranked_df = CandidateRanker(
            CandidateRankerConfig(
                min_explore_ratio=float(min_explore),
                use_factor_family_quota=bool(getattr(config, "enable_family_quota", True)),
                family_max_selected_ratio=float(getattr(config, "family_max_selected_ratio", 0.45)),
                family_min_explore_ratio=float(getattr(config, "family_min_explore_ratio", 0.25)),
                use_layer_quota=(mode == "layered_v2"),
                layer_selection_min_ratio=dict(getattr(config, "layer_selection_min_ratio", {}) or {}),
                layer_selection_max_ratio=dict(getattr(config, "layer_selection_max_ratio", {}) or {}),
                structure_selection_min_ratio=dict(getattr(config, "structure_selection_min_ratio", {}) or {}),
                score_weights=score_w,
                complexity_weight=float(getattr(config, "complexity_weight", 0.10)),
            )
        ).rank(
            candidate_df,
            feedback_hints=effective_feedback_hints,
            max_eval=rank_max_eval,
        )
        candidate_df = _merge_ranked_columns(candidate_df, ranked_df)
        if bool(getattr(config, "enable_feedback_mutation", False)):
            passed_df = _reserve_feedback_mutation_candidates(
                ranked_df=ranked_df,
                max_eval=max_eval,
                min_count=mutation_min_count,
                min_ratio=mutation_min_ratio,
            )
        else:
            passed_df = ranked_df.reset_index(drop=True)
    else:
        passed_df = (
            candidate_df[candidate_df["prefilter_status"] == "pass"].copy().head(max_eval).reset_index(drop=True)
        )

    rejected_df = candidate_df[candidate_df["prefilter_status"] != "pass"].copy()
    expressions = passed_df["expression"].astype(str).tolist()
    meta = {
        "field_source": getattr(config, "search_field_source", "panel_store"),
        "scalar_field_count": len(field_universe.scalar_fields),
        "group_field_count": len(field_universe.group_fields),
        "vector_field_count": len(field_universe.vector_fields),
        "excluded_field_count": len(field_universe.excluded_fields),
        "candidate_count": int(len(candidate_df)),
        "passed_candidate_count": int(len(passed_df)),
        "rejected_candidate_count": int(len(rejected_df)),
        "candidate_df": candidate_df,
        "rejected_df": rejected_df,
        "sample_df": sample_df,
        "sample_prefilter_meta": sample_prefilter_meta,
        "layered_generation_diagnostics": _layered_generation_diagnostics(
            generated=generated,
            field_profile_count=len(effective_feedback_hints.get("field_profiles", {}) or {})
            if isinstance(effective_feedback_hints.get("field_profiles", {}), dict)
            else 0,
            feedback_policy_meta=feedback_policy_meta,
            sample_df=sample_df,
            builder_dedup_count=getattr(layered_builder, "dedup_count", 0) if mode == "layered_v2" else 0,
            existing_hashes_count=len(existing_hashes or set()),
        ),
        "field_diagnostics": _field_diagnostics(field_universe),
    }
    return expressions, candidate_df, rejected_df, meta


def save_candidate_artifacts(
    candidate_df: pd.DataFrame,
    rejected_df: pd.DataFrame,
    root: Path,
    batch_id: str,
    sample_df: pd.DataFrame | None = None,
    generation_diagnostics: dict[str, Any] | None = None,
) -> dict[str, str]:
    out_dir = root / "catalog" / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    candidates_path = out_dir / f"{batch_id}_candidates.csv"
    rejected_path = out_dir / f"{batch_id}_rejected.csv"
    atomic_write_dataframe_csv(candidates_path, candidate_df, index=False)
    atomic_write_dataframe_csv(rejected_path, rejected_df, index=False)
    paths["candidates_path"] = str(candidates_path.as_posix())
    paths["rejected_path"] = str(rejected_path.as_posix())
    if sample_df is not None and not sample_df.empty:
        sample_path = out_dir / f"{batch_id}_sample_prefilter.csv"
        atomic_write_dataframe_csv(sample_path, sample_df, index=False)
        paths["sample_prefilter_path"] = str(sample_path.as_posix())
    if generation_diagnostics:
        diagnostics_path = out_dir / f"{batch_id}_generation_diagnostics.json"
        atomic_write_json(diagnostics_path, generation_diagnostics)
        paths["generation_diagnostics_path"] = str(diagnostics_path.as_posix())
    return paths


def prune_candidate_artifacts(root: Path, *, max_batches: int = 200, retention_days: int = 30) -> dict[str, Any]:
    out_dir = Path(root) / "catalog" / "candidates"
    summary: dict[str, Any] = {
        "enabled": True,
        "scanned_files": 0,
        "deleted_files": 0,
        "deleted_bytes": 0,
        "failed_paths": [],
        "retained_batches": 0,
    }
    if not out_dir.exists():
        return summary
    files = [p for p in out_dir.iterdir() if p.is_file() and p.name.startswith("batch_")]
    summary["scanned_files"] = int(len(files))
    by_batch: dict[str, list[Path]] = {}
    for path in files:
        parts = path.name.split("_")
        batch = "_".join(parts[:2]) if len(parts) >= 2 else path.stem
        by_batch.setdefault(batch, []).append(path)
    now = time.time()
    cutoff_age = max(0, int(retention_days)) * 86400
    ordered_batches = sorted(
        by_batch.items(),
        key=lambda item: max((p.stat().st_mtime for p in item[1] if p.exists()), default=0.0),
        reverse=True,
    )
    keep_batches: set[str] = set()
    for idx, (batch, paths) in enumerate(ordered_batches):
        latest_mtime = max((p.stat().st_mtime for p in paths if p.exists()), default=0.0)
        keep_by_count = idx < max(0, int(max_batches))
        keep_by_age = cutoff_age > 0 and (now - latest_mtime) <= cutoff_age
        if keep_by_count or keep_by_age:
            keep_batches.add(batch)
    summary["retained_batches"] = int(len(keep_batches))
    for batch, paths in by_batch.items():
        if batch in keep_batches:
            continue
        for path in paths:
            try:
                size = int(path.stat().st_size) if path.exists() else 0
                path.unlink(missing_ok=True)
                summary["deleted_files"] = int(summary["deleted_files"]) + 1
                summary["deleted_bytes"] = int(summary["deleted_bytes"]) + size
            except Exception:
                summary.setdefault("failed_paths", []).append(str(path.as_posix()))
    return summary


def _format_windows(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return ",".join([x.strip() for x in value.split(",") if x.strip()])
    try:
        return ",".join(str(int(x)) for x in value if int(x) > 0)
    except Exception:
        text = str(value or "").strip()
    return text


def _metadata_policy_score(
    item_meta: dict[str, Any],
    hints: dict[str, Any],
    meta_key: str,
    positive_key: str,
    negative_key: str,
) -> float:
    name = str(item_meta.get(meta_key, "") or "").strip()
    if not name:
        return 0.0
    positive = hints.get(positive_key, {}) if isinstance(hints, dict) else {}
    negative = hints.get(negative_key, {}) if isinstance(hints, dict) else {}
    pos = float(positive.get(name, 0.0)) if isinstance(positive, dict) else 0.0
    neg = float(negative.get(name, 0.0)) if isinstance(negative, dict) else 0.0
    return float(pos - neg)


def _field_profile_status_for_fields(fields: tuple[str, ...], profiles: Any) -> str:
    if not isinstance(profiles, dict):
        return ""
    statuses = []
    for field in fields:
        profile = profiles.get(field)
        status = str(getattr(profile, "field_profile_status", "") or "")
        if status and status != "pass":
            statuses.append(status)
    if not statuses:
        return "pass" if fields else ""
    return ",".join(sorted(dict.fromkeys(",".join(statuses).split(","))))


def _layered_generation_diagnostics(
    *,
    generated: list[dict[str, Any]],
    field_profile_count: int,
    feedback_policy_meta: dict[str, Any],
    sample_df: pd.DataFrame,
    builder_dedup_count: int = 0,
    existing_hashes_count: int = 0,
) -> dict[str, Any]:
    metadata_items = [dict(row.get("metadata", {}) or {}) for row in generated]
    bucket_counts = {
        "evaluated": 0,
        "skipped": 0,
        "error": 0,
        "low_quality": 0,
        "pass": 0,
    }
    sample_counts: dict[str, int] = {}
    if sample_df is not None and not sample_df.empty and "bucket_sample_status" in sample_df.columns:
        for status, count in sample_df["bucket_sample_status"].fillna("").astype(str).value_counts().to_dict().items():
            if not status:
                continue
            if status in bucket_counts:
                bucket_counts[status] += int(count)
            if status in {"pass", "low_quality"}:
                bucket_counts["evaluated"] += int(count)
            elif status in {"skipped", "error"}:
                bucket_counts[status] += 0
    if sample_df is not None and not sample_df.empty and "sample_status" in sample_df.columns:
        sample_counts = {
            str(status): int(count)
            for status, count in sample_df["sample_status"].fillna("").astype(str).value_counts().to_dict().items()
            if str(status)
        }
    layer_counts: dict[str, int] = {}
    for item in metadata_items:
        layer = str(item.get("layer", "") or "")
        if layer:
            layer_counts[layer] = int(layer_counts.get(layer, 0)) + 1
    return {
        "layer_candidate_counts": layer_counts,
        "recipe_candidate_count": int(sum(1 for meta in metadata_items if str(meta.get("recipe_family", "") or ""))),
        "role_pair_candidate_count": int(
            sum(1 for meta in metadata_items if str(meta.get("role_pair_type", "") or ""))
        ),
        "field_profile_count": int(field_profile_count),
        "feedback_policy_enabled": bool(feedback_policy_meta.get("enabled", False)),
        "feedback_policy_history_rows": int(feedback_policy_meta.get("history_rows", 0) or 0),
        "feedback_score_column": str(feedback_policy_meta.get("score_column", "") or ""),
        "feedback_score_basis": str(feedback_policy_meta.get("score_basis", "none") or "none"),
        "sample_prefilter": sample_counts,
        "bucket_quality": bucket_counts,
        "existing_hashes_count": int(existing_hashes_count),
        "builder_dedup_count": int(builder_dedup_count),
    }


def _field_diagnostics(field_universe: FieldUniverse) -> dict[str, Any]:
    scalar = set(field_universe.scalar_fields)
    groups = set(field_universe.group_fields)

    def has_any(tokens: tuple[str, ...]) -> bool:
        return any(any(token in name.lower() for token in tokens) for name in scalar)

    available = {
        "size": has_any(("circ_mv", "total_mv", "float_mv", "cap", "mv")),
        "liquidity": has_any(("amount", "volume", "turnover", "volume_ratio")),
        "moneyflow": has_any(("moneyflow",)),
        "valuation": has_any(("pe", "pb", "ps", "pcf", "valuation")),
        "chip": has_any(("chip", "holder", "concentration")),
        "technical": has_any(("volatility", "amplitude", "atr", "rsi", "macd")),
        "industry": bool(groups.intersection({"industry", "sector", "subindustry"})),
        "ctx_market": any(name.lower().startswith("ctx_") for name in scalar),
    }
    suggested_fields = {
        "size": ["circ_mv", "total_mv", "float_mv"],
        "liquidity": ["amount", "volume", "turnover_rate", "volume_ratio"],
        "moneyflow": [
            "moneyflow_net_mf_amount",
            "moneyflow_buy_lg_amount",
            "moneyflow_sell_lg_amount",
            "moneyflow_buy_elg_amount",
            "moneyflow_sell_elg_amount",
        ],
        "valuation": ["pe", "pb", "ps", "pcf"],
        "chip": ["chip", "holder_count", "holder_concentration"],
        "technical": ["volatility", "amplitude", "atr", "rsi", "macd"],
        "context": ["ctx_mkt_return_*", "ctx_mkt_close_*", "ctx_mkt_amount_*"],
    }
    missing = [key for key, present in available.items() if not present]
    return {
        "available": available,
        "missing": missing,
        "suggested_fields": suggested_fields,
    }


def _family_from_source(source: str) -> tuple[str, str]:
    text = str(source)
    if text.startswith("layered_v2:"):
        parts = text.split(":")
        return (parts[2] if len(parts) >= 3 and parts[2] else "layered"), ""
    if text.startswith("pair_") or text.startswith("pair:"):
        return "pair", text.split(":", 1)[1] if ":" in text else ""
    if text.startswith("feedback_mutation_v2"):
        return "feedback_mutation", ""
    if text.startswith("op_"):
        return "operator", ""
    if text == "op_signature":
        return "operator", ""
    return "template", ""


def _field_role_map(field_universe: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for spec in getattr(field_universe, "specs", ()):
        name = str(getattr(spec, "name", "") or "")
        kind = str(getattr(spec, "field_kind", "") or "")
        if not name or kind != "scalar":
            continue
        categories = tuple(getattr(spec, "categories", ()) or ())
        out[name] = infer_field_semantic(name, categories).role
    return out


def _filter_field_universe_by_factor_family(
    field_universe: FieldUniverse,
    include_families: Any = (),
    exclude_families: Any = (),
) -> FieldUniverse:
    include = {str(x).strip() for x in (include_families or ()) if str(x).strip()}
    exclude = {str(x).strip() for x in (exclude_families or ()) if str(x).strip()}
    if not include and not exclude:
        return field_universe
    kept = []
    removed = set(field_universe.excluded_fields)
    for spec in field_universe.specs:
        if spec.field_kind in {"group", "mask"}:
            kept.append(spec)
            continue
        family = str(getattr(spec, "factor_family", "") or "")
        if include and family not in include:
            removed.add(spec.name)
            continue
        if exclude and family in exclude:
            removed.add(spec.name)
            continue
        kept.append(spec)
    return FieldUniverse(specs=tuple(kept), excluded_fields=tuple(sorted(removed)))


def _factor_family_for_fields(field_universe: FieldUniverse, fields: tuple[str, ...]) -> tuple[str, str]:
    spec_map = {str(spec.name): spec for spec in field_universe.specs}
    category_map: dict[str, str] = {}
    family_values: dict[str, str] = {}
    for field in fields:
        spec = spec_map.get(str(field))
        if spec is None:
            continue
        cats = tuple(getattr(spec, "categories", ()) or ())
        category_map[str(field)] = str(cats[0]) if cats else ""
        family_values[str(field)] = str(getattr(spec, "factor_family", "") or "")
    if family_values and all(family_values.get(str(f), "") for f in fields if str(f) in family_values):
        import json
        from collections import Counter

        counts = Counter(family_values[str(f)] for f in fields if str(f) in family_values and family_values[str(f)])
        if counts:
            ordered = {key: int(counts[key]) for key in sorted(counts)}
            total = float(sum(ordered.values()))
            primary = sorted(ordered.items(), key=lambda item: (-int(item[1]), item[0]))[0][0]
            payload = {
                "counts": ordered,
                "ratios": {key: float(value) / total if total > 0 else 0.0 for key, value in ordered.items()},
                "primary_factor_family": primary,
            }
            return ",".join(ordered.keys()), json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return infer_factor_family_mix(fields, category_map=category_map)


class CandidatePrefilterResultShim:
    def __init__(self, base: Any, stage: str, reason: str) -> None:
        self.expression = base.expression
        self.passed = False
        self.reject_stage = stage
        self.reject_reason = reason
        self.normalized_expression = base.normalized_expression
        self.expression_hash = base.expression_hash
        self.canonical_expression = base.canonical_expression
        self.canonical_hash = base.canonical_hash
        self.lint_status = "rejected"
        self.lint_warning_reason = reason
        self.fields = base.fields
        self.operators = base.operators
        self.operator_count = base.operator_count
        self.field_count = base.field_count
        self.depth = base.depth


def _apply_sample_prefilter(
    candidate_df: pd.DataFrame, panel_store: PanelStore, config: Any
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluator = SampleEvaluator(
        SampleEvaluatorConfig(
            enabled=True,
            min_coverage=float(getattr(config, "sample_prefilter_min_coverage", 0.30)),
            max_inf_ratio=float(getattr(config, "sample_prefilter_max_inf_ratio", 0.01)),
        )
    )
    out = candidate_df.copy()
    rows: list[dict[str, Any]] = []
    max_evaluations = int(getattr(config, "sample_prefilter_max_evaluations", 0) or 0)
    stratified = bool(getattr(config, "sample_prefilter_stratified", True))
    selected_for_sample = _sample_prefilter_selected_indices(
        out, max_evaluations=max_evaluations, stratified=stratified
    )
    bucket_quality_enabled = bool(getattr(config, "bucket_quality_lite_enabled", True))
    bucket_max_evaluations = int(getattr(config, "bucket_quality_max_evaluations", 80) or 0)
    bucket_evaluated_count = 0
    bucket_cache: dict[str, dict[str, Any]] = {}
    bucket_cfg = BucketQualityConfig(
        min_coverage=float(getattr(config, "bucket_quality_min_coverage", 0.50)),
        min_median_group_size=int(getattr(config, "bucket_quality_min_median_group_size", 5)),
        min_group_count=int(getattr(config, "bucket_quality_min_group_count", 3)),
        max_nan_group_ratio=float(getattr(config, "bucket_quality_max_nan_group_ratio", 0.30)),
    )
    evaluated_count = 0
    for idx, row in out.iterrows():
        sample_metrics = _empty_sample_metrics()
        bucket_metrics = _empty_bucket_metrics()
        if str(row.get("prefilter_status", "")) != "pass":
            out.loc[idx, "sample_status"] = "skipped"
            _write_metrics(out, idx, sample_metrics)
            _write_metrics(out, idx, bucket_metrics)
            continue
        if str(row.get("prefilter_status", "")) == "pass" and idx not in selected_for_sample:
            status = "skipped_budget"
            reason = "sample_prefilter_budget_not_selected"
            out.loc[idx, "sample_status"] = status
            out.loc[idx, "sample_reject_reason"] = reason
            _write_metrics(out, idx, sample_metrics)
            _write_metrics(out, idx, bucket_metrics)
            rows.append(
                {
                    "candidate_id": row.get("candidate_id", ""),
                    "expression": str(row.get("expression", "")),
                    "sample_status": status,
                    "sample_reject_reason": reason,
                    "coverage": sample_metrics["sample_coverage"],
                    "inf_ratio": sample_metrics["sample_inf_ratio"],
                    "extreme_ratio": sample_metrics["sample_extreme_ratio"],
                    "unique_count": sample_metrics["sample_unique_count"],
                    "error": "",
                    **sample_metrics,
                    **bucket_metrics,
                }
            )
            continue
        result = evaluator.evaluate(str(row.get("expression", "")), panel_store)
        evaluated_count += 1
        status = str(result.status or ("pass" if result.passed else "reject"))
        sample_metrics = {
            "sample_coverage": float(result.coverage),
            "sample_inf_ratio": float(result.inf_ratio),
            "sample_extreme_ratio": float(result.extreme_ratio),
            "sample_unique_count": int(result.unique_count),
            "sample_quality_score": _sample_quality_score(result),
        }
        if bucket_quality_enabled:
            bucket_metrics, bucket_evaluated_count = _evaluate_bucket_quality_for_row(
                row=row,
                panel_store=panel_store,
                cfg=bucket_cfg,
                max_evaluations=bucket_max_evaluations,
                evaluated_count=bucket_evaluated_count,
                cache=bucket_cache,
            )
        out.loc[idx, "sample_status"] = status
        out.loc[idx, "sample_reject_reason"] = result.reject_reason
        _write_metrics(out, idx, sample_metrics)
        _write_metrics(out, idx, bucket_metrics)
        if float(bucket_metrics.get("bucket_sample_quality_score", 0.0) or 0.0) > 0.0:
            out.loc[idx, "bucket_quality_score"] = _to_float(row.get("bucket_quality_score", 0.0)) + float(
                bucket_metrics.get("bucket_sample_quality_score", 0.0) or 0.0
            )
        rows.append(
            {
                "candidate_id": row.get("candidate_id", ""),
                "expression": result.expression,
                "sample_status": status,
                "sample_reject_reason": result.reject_reason,
                "coverage": result.coverage,
                "inf_ratio": result.inf_ratio,
                "extreme_ratio": result.extreme_ratio,
                "unique_count": result.unique_count,
                "error": result.error,
                **sample_metrics,
                **bucket_metrics,
            }
        )
        if status == "reject" or not result.passed:
            out.loc[idx, "prefilter_status"] = "reject"
            out.loc[idx, "reject_stage"] = "sample"
            out.loc[idx, "reject_reason"] = result.reject_reason
        bucket_reject_reason = _bucket_quality_prefilter_reject_reason(row=row, metrics=bucket_metrics, config=config)
        if bucket_reject_reason:
            out.loc[idx, "prefilter_status"] = "reject"
            out.loc[idx, "reject_stage"] = "bucket_quality"
            out.loc[idx, "reject_reason"] = bucket_reject_reason
    return out, pd.DataFrame(rows)


def _sample_prefilter_selected_indices(
    candidate_df: pd.DataFrame, *, max_evaluations: int, stratified: bool
) -> set[Any]:
    if "prefilter_status" in candidate_df.columns:
        passed = candidate_df[candidate_df["prefilter_status"].astype(str) == "pass"]
    else:
        passed = candidate_df
    if max_evaluations <= 0 or len(passed) <= max_evaluations:
        return set(passed.index.tolist())
    if not stratified:
        return set(passed.head(max_evaluations).index.tolist())
    selected: list[Any] = []

    def add(idx: Any) -> None:
        if idx not in selected and len(selected) < max_evaluations:
            selected.append(idx)

    for kind in ("bucket", "gate", "recipe", "role_pair"):
        subset = passed[passed.apply(lambda row: _row_metadata_feature(row, kind), axis=1)]
        for idx in subset.index.tolist():
            add(idx)
            break
        if len(selected) >= max_evaluations:
            return set(selected)
    for layer in ("L4", "L3", "L2", "L1", "L0"):
        if "layer" not in passed.columns:
            break
        subset = passed[passed["layer"].fillna("").astype(str) == layer]
        for idx in subset.index.tolist():
            add(idx)
            break
        if len(selected) >= max_evaluations:
            return set(selected)
    for idx in passed.index.tolist():
        add(idx)
        if len(selected) >= max_evaluations:
            break
    return set(selected)


def _evaluate_bucket_quality_for_row(
    *,
    row: pd.Series,
    panel_store: PanelStore,
    cfg: BucketQualityConfig,
    max_evaluations: int,
    evaluated_count: int,
    cache: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], int]:
    metadata = _parse_metadata_json(row.get("metadata_json", "{}"))
    bucket_expression = str(metadata.get("bucket_expression", "") or "").strip()
    if not bucket_expression:
        return _empty_bucket_metrics(), evaluated_count
    if cache is not None and bucket_expression in cache:
        metrics = dict(cache[bucket_expression])
        metrics["bucket_sample_cache_hit"] = True
        return metrics, evaluated_count
    if max_evaluations > 0 and evaluated_count >= max_evaluations:
        metrics = _empty_bucket_metrics()
        metrics["bucket_sample_status"] = "skipped"
        metrics["bucket_sample_reject_reason"] = "bucket_quality_budget_exceeded"
        return metrics, evaluated_count
    result = evaluate_bucket_quality(bucket_expression, panel_store, cfg)
    metrics = {
        "bucket_sample_status": result.status,
        "bucket_sample_quality_status": result.quality_status,
        "bucket_sample_reject_reason": result.reject_reason,
        "bucket_sample_coverage": float(result.coverage),
        "bucket_sample_group_count_median": float(result.group_count_median),
        "bucket_sample_group_size_median": float(result.group_size_median),
        "bucket_sample_group_size_min": float(result.group_size_min),
        "bucket_sample_nan_group_ratio": float(result.nan_group_ratio),
        "bucket_sample_is_composite": bool(result.is_composite),
        "bucket_sample_quality_score": float(result.quality_score),
        "bucket_sample_cache_hit": False,
    }
    if cache is not None:
        cache[bucket_expression] = dict(metrics)
    return metrics, evaluated_count + (1 if result.status in {"pass", "low_quality"} else 0)


def _bucket_quality_prefilter_reject_reason(row: pd.Series, metrics: dict[str, Any], config: Any) -> str:
    status = str(metrics.get("bucket_sample_quality_status", metrics.get("bucket_sample_status", "")) or "")
    if status not in {"low_quality", "error"}:
        return ""
    metadata = _parse_metadata_json(row.get("metadata_json", "{}"))
    is_composite = (
        bool(metrics.get("bucket_sample_is_composite", False))
        or int(_to_float(metadata.get("group_complexity", 0))) > 1
    )
    reject_composite = bool(getattr(config, "bucket_quality_reject_low_quality_composite", True))
    reject_plain = bool(getattr(config, "bucket_quality_reject_low_quality_plain", False))
    if (is_composite and reject_composite) or ((not is_composite) and reject_plain):
        reason = str(metrics.get("bucket_sample_reject_reason", "") or status)
        return f"bucket_quality_{reason}"
    return ""


def _row_metadata_feature(row: pd.Series, kind: str) -> bool:
    metadata = _parse_metadata_json(row.get("metadata_json", "{}"))
    key = str(kind or "").strip().lower()
    if key == "bucket":
        return bool(str(metadata.get("bucket_expression", "") or metadata.get("bucket_family", "") or "").strip())
    if key == "gate":
        return bool(str(metadata.get("gate_family", "") or metadata.get("gate_expression", "") or "").strip())
    if key == "recipe":
        return bool(str(metadata.get("recipe_family", "") or metadata.get("recipe_id", "") or "").strip())
    if key == "role_pair":
        return bool(str(metadata.get("role_pair_type", "") or "").strip())
    return False


def _sample_quality_score(result: Any) -> float:
    status = str(getattr(result, "status", "") or ("pass" if getattr(result, "passed", False) else "reject"))
    if status != "pass" or not bool(getattr(result, "passed", False)):
        return 0.0
    coverage = _clip01(getattr(result, "coverage", 0.0))
    inf_penalty = 1.0 - _clip01(getattr(result, "inf_ratio", 0.0))
    extreme_penalty = 1.0 - _clip01(getattr(result, "extreme_ratio", 0.0))
    unique_score = 1.0 if int(getattr(result, "unique_count", 0) or 0) > 1 else 0.0
    return float(0.45 * coverage + 0.20 * inf_penalty + 0.20 * extreme_penalty + 0.15 * unique_score)


def _empty_sample_metrics() -> dict[str, Any]:
    return {
        "sample_coverage": 0.0,
        "sample_inf_ratio": 0.0,
        "sample_extreme_ratio": 0.0,
        "sample_unique_count": 0,
        "sample_quality_score": 0.0,
    }


def _empty_bucket_metrics() -> dict[str, Any]:
    return {
        "bucket_sample_status": "",
        "bucket_sample_quality_status": "",
        "bucket_sample_reject_reason": "",
        "bucket_sample_coverage": 0.0,
        "bucket_sample_group_count_median": 0.0,
        "bucket_sample_group_size_median": 0.0,
        "bucket_sample_group_size_min": 0.0,
        "bucket_sample_nan_group_ratio": 0.0,
        "bucket_sample_is_composite": False,
        "bucket_sample_quality_score": 0.0,
        "bucket_sample_cache_hit": False,
    }


def _write_metrics(out: pd.DataFrame, idx: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        out.loc[idx, key] = value


def _parse_metadata_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(str(value or "{}"))
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _clip01(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return max(0.0, min(1.0, out))


def _to_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if math.isfinite(out) else 0.0


def _merge_ranked_columns(candidate_df: pd.DataFrame, ranked_df: pd.DataFrame) -> pd.DataFrame:
    out = candidate_df.copy()
    score_cols = [
        "complexity_score",
        "candidate_score",
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
        "selection_bucket",
    ]
    for col in score_cols:
        if col not in out.columns:
            out[col] = 0.0 if col != "selection_bucket" else ""
    if ranked_df is None or ranked_df.empty or "candidate_id" not in ranked_df.columns:
        return out
    ranked_map = ranked_df.set_index("candidate_id")
    for col in score_cols:
        if col not in ranked_map.columns:
            continue
        mask = out["candidate_id"].isin(ranked_map.index)
        values = out.loc[mask, "candidate_id"].map(ranked_map[col])
        if col != "selection_bucket":
            values = pd.to_numeric(values, errors="coerce")
        out.loc[mask, col] = values
    return out


def _reserve_feedback_mutation_candidates(
    ranked_df: pd.DataFrame,
    max_eval: int,
    min_count: int = 0,
    min_ratio: float = 0.0,
) -> pd.DataFrame:
    if ranked_df is None or ranked_df.empty:
        return pd.DataFrame() if ranked_df is None else ranked_df.copy()
    limit = max(1, int(max_eval))
    target_n = max(0, int(min_count), int(math.ceil(float(limit) * float(min_ratio))))
    if target_n <= 0 or "source" not in ranked_df.columns:
        return ranked_df.head(limit).reset_index(drop=True)
    mutation_df = ranked_df[ranked_df["source"].astype(str) == "feedback_mutation_v2"].copy()
    if mutation_df.empty:
        return ranked_df.head(limit).reset_index(drop=True)
    reserve_n = min(int(target_n), int(limit), int(len(mutation_df)))
    keep_mutation = mutation_df.head(reserve_n)
    if "candidate_id" in ranked_df.columns and "candidate_id" in keep_mutation.columns:
        remaining = ranked_df[~ranked_df["candidate_id"].isin(set(keep_mutation["candidate_id"].astype(str)))].copy()
    else:
        remaining = ranked_df.drop(index=keep_mutation.index, errors="ignore").copy()
    head_n = max(0, int(limit) - int(len(keep_mutation)))
    selected = pd.concat([remaining.head(head_n), keep_mutation], axis=0, ignore_index=False)
    if "candidate_id" in selected.columns:
        selected = selected.drop_duplicates(subset=["candidate_id"], keep="first")
    return selected.head(limit).reset_index(drop=True)


def _non_negative_int(raw: Any, default: int = 0) -> int:
    try:
        value = int(raw)
    except Exception:
        return int(default)
    return max(0, value)


def _normalized_ratio(raw: Any, default: float = 0.0) -> float:
    try:
        value = float(raw)
    except Exception:
        return float(default)
    return max(0.0, min(1.0, value))


def _eligible_pass_pool_size(candidate_df: pd.DataFrame) -> int:
    if candidate_df is None or candidate_df.empty:
        return 0
    work = candidate_df.copy()
    if "prefilter_status" in work.columns:
        work = work[work["prefilter_status"].astype(str) == "pass"]
    if "sample_status" in work.columns:
        work = work[work["sample_status"].astype(str).isin(["", "pass", "skipped"])]
    return int(len(work))
