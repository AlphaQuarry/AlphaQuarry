from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .finance_fields import FINANCE_ASOF_FIELD_MAP
from ..history.registry import filter_new_expressions
from ..mining import (
    LayeredBuilderConfig,
    LayeredExpressionBuilder,
    build_operator_search_space,
    build_search_space,
    build_signature_aware_search_space,
    load_templates,
)
from ..mining.factor_family import infer_factor_family
from ..mining.field_universe import (
    FieldUniverse,
    field_catalog_to_specs,
    is_leakage_field,
)
from ..simulation.neutralization import (
    neutralization_group_field,
    normalize_neutralization_mode,
)
from ..validators import expression_stats
from ..workflow.universe_store import (
    canonical_simulation_config_json,
    load_seen_expression_hashes_for_universe,
)


def collect_required_fields_from_expressions(
    expressions: Sequence[str],
    base_fields: Iterable[str] = (),
    group_fields: Iterable[str] = (),
    extra_fields: Iterable[str] = (),
) -> list[str]:
    required: list[str] = []

    def _append(field: str) -> None:
        name = str(field or "").strip()
        if not name:
            return
        if name not in required:
            required.append(name)

    for field in base_fields:
        _append(str(field))
    for field in group_fields:
        _append(str(field))
    for field in extra_fields:
        _append(str(field))

    for expression in expressions:
        stats = expression_stats(str(expression))
        for field in stats.unique_fields:
            _append(str(field))
    return required


def get_duckdb_view_columns(duckdb_path: str, source_view: str) -> list[str]:
    try:
        import duckdb  # type: ignore
    except Exception as exc:
        raise RuntimeError("duckdb is required but not installed") from exc

    schema_name, table_name = _split_view_name(source_view)
    conn = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        if schema_name:
            query = (
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = ? AND table_name = ? ORDER BY ordinal_position"
            )
            rows = conn.execute(query, [schema_name, table_name]).fetchall()
        else:
            query = "SELECT column_name FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position"
            rows = conn.execute(query, [table_name]).fetchall()
        return [str(r[0]) for r in rows]
    finally:
        conn.close()


def load_panel_from_duckdb(
    duckdb_path: str,
    source_view: str,
    expressions: Sequence[str] | None = None,
    required_fields: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    date_col: str = "date",
    code_col: str = "code",
    base_fields: Iterable[str] = ("pct_chg", "circ_mv"),
    group_fields: Iterable[str] = (),
    run_filters: dict[str, Any] | None = None,
    duckdb_settings: dict[str, Any] | None = None,
    sort: bool = True,
) -> pd.DataFrame:
    try:
        import duckdb  # type: ignore
    except Exception as exc:
        raise RuntimeError("duckdb is required but not installed") from exc

    selected_cols = [str(date_col), str(code_col)]
    expr_fields = collect_required_fields_from_expressions(
        expressions=expressions or [],
        base_fields=base_fields,
        group_fields=group_fields,
        extra_fields=required_fields or [],
    )
    for field in expr_fields:
        if field not in selected_cols:
            selected_cols.append(str(field))

    effective_source_view, available_cols = resolve_duckdb_source_view_for_fields(
        duckdb_path=duckdb_path,
        source_view=source_view,
        required_fields=selected_cols,
    )
    available_set = set(available_cols)

    missing = [c for c in selected_cols if c not in available_set]
    if missing:
        raise ValueError(f"Missing required columns in source view '{effective_source_view}': {missing}")

    where_clauses: list[str] = []
    params: list[Any] = []

    if start_date:
        where_clauses.append(f"{_qident(date_col)} >= ?")
        params.append(str(start_date))
    if end_date:
        where_clauses.append(f"{_qident(date_col)} <= ?")
        params.append(str(end_date))

    filters = dict(run_filters or {})
    if bool(filters.get("universe_only", True)) and "universe" in available_set:
        where_clauses.append("COALESCE(universe, 0) = 1")

    include_bj = _to_bool(filters.get("include_bj", True), default=True)
    if not include_bj:
        where_clauses.append(f"{_qident(code_col)} NOT LIKE ?")
        params.append("%.BJ")

    include_codes = filters.get("include_codes", [])
    if isinstance(include_codes, (list, tuple)) and include_codes:
        placeholders = ", ".join(["?" for _ in include_codes])
        where_clauses.append(f"{_qident(code_col)} IN ({placeholders})")
        params.extend([str(x) for x in include_codes])

    code_prefix = str(filters.get("code_prefix", "") or "").strip()
    if code_prefix:
        where_clauses.append(f"{_qident(code_col)} LIKE ?")
        params.append(f"{code_prefix}%")

    finance_fields = [c for c in selected_cols if c in FINANCE_ASOF_FIELD_MAP]
    if str(source_view).strip() == "v_project_panel_cn_a" and finance_fields:
        sql, effective_source_view = _build_dynamic_project_panel_finance_sql(
            duckdb_path=duckdb_path,
            selected_cols=selected_cols,
            finance_fields=finance_fields,
            where_clauses=where_clauses,
            date_col=str(date_col),
            code_col=str(code_col),
            sort=sort,
        )
    else:
        view_name_sql = _qident(effective_source_view)
        select_sql = ", ".join([_qident(c) for c in selected_cols])
        sql = f"SELECT {select_sql} FROM {view_name_sql}"
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        if sort:
            sql += f" ORDER BY {_qident(code_col)}, {_qident(date_col)}"

    conn = duckdb.connect(
        str(duckdb_path),
        read_only=True,
        config=_build_duckdb_connection_config(duckdb_path=duckdb_path, duckdb_settings=duckdb_settings),
    )
    try:
        out = conn.execute(sql, params).fetchdf()
    finally:
        conn.close()

    if str(date_col) in out.columns:
        out[str(date_col)] = pd.to_datetime(out[str(date_col)], errors="coerce")
    out.attrs["duckdb_requested_source_view"] = str(source_view)
    out.attrs["duckdb_effective_source_view"] = str(effective_source_view)
    return out.reset_index(drop=True)


def _build_dynamic_project_panel_finance_sql(
    duckdb_path: str,
    selected_cols: Sequence[str],
    finance_fields: Sequence[str],
    where_clauses: Sequence[str],
    date_col: str,
    code_col: str,
    sort: bool = True,
) -> tuple[str, str]:
    non_finance_cols = [c for c in selected_cols if c not in FINANCE_ASOF_FIELD_MAP]
    base_view, _base_cols = _resolve_project_base_source_for_columns(
        duckdb_path=duckdb_path,
        required_fields=non_finance_cols,
    )
    select_parts: list[str] = []
    for col in selected_cols:
        if col in FINANCE_ASOF_FIELD_MAP:
            source, raw_col = FINANCE_ASOF_FIELD_MAP[col]
            select_parts.append(f"{_qident(source + '_asof')}.{_qident(raw_col)} AS {_qident(col)}")
        else:
            select_parts.append(f"b.{_qident(col)} AS {_qident(col)}")

    join_sql = _finance_lateral_joins(finance_fields)
    sql = f"SELECT {', '.join(select_parts)} FROM {_qident(base_view)} b{join_sql}"
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)
    if sort:
        sql += f" ORDER BY {_qident(code_col)}, {_qident(date_col)}"
    return sql, "dynamic_project_panel_finance_asof"


def _resolve_project_base_source_for_columns(duckdb_path: str, required_fields: Sequence[str]) -> tuple[str, list[str]]:
    required = {str(x).strip() for x in required_fields if str(x).strip()}
    for candidate in (
        "v_project_market_daily_base_hot",
        "project_market_daily_base",
        "v_project_market_daily_base",
    ):
        cols = get_duckdb_view_columns(duckdb_path=duckdb_path, source_view=candidate)
        if cols and required.issubset(set(cols)):
            return candidate, cols
    return "v_project_market_daily_base", get_duckdb_view_columns(
        duckdb_path=duckdb_path,
        source_view="v_project_market_daily_base",
    )


def _finance_lateral_joins(finance_fields: Sequence[str]) -> str:
    by_source: dict[str, list[str]] = {}
    for field in finance_fields:
        source, raw_col = FINANCE_ASOF_FIELD_MAP[str(field)]
        by_source.setdefault(source, [])
        if raw_col not in by_source[source]:
            by_source[source].append(raw_col)

    specs = {
        "income": ("fact_finance_income_q", "fi"),
        "balance": ("fact_finance_balancesheet_q", "fb"),
        "cashflow": ("fact_finance_cashflow_q", "fc"),
        "indicator": ("fact_finance_indicator_q", "ff"),
    }
    chunks: list[str] = []
    for source, (table, alias) in specs.items():
        cols = by_source.get(source)
        if not cols:
            continue
        for required in ("ann_date", "end_date"):
            if required not in cols:
                cols.insert(0 if required == "ann_date" else 1, required)
        select_cols = ", ".join(_qident(col) for col in cols)
        chunks.append(
            f" LEFT JOIN LATERAL ("
            f"SELECT {select_cols} FROM {_qident(table)} {alias} "
            f"WHERE {alias}.code = b.code AND {alias}.ann_date <= b.date "
            f"ORDER BY {alias}.ann_date DESC NULLS LAST, {alias}.end_date DESC NULLS LAST LIMIT 1"
            f") {_qident(source + '_asof')} ON TRUE"
        )
    return "".join(chunks)


def resolve_duckdb_source_view_for_fields(
    duckdb_path: str,
    source_view: str,
    required_fields: Sequence[str],
) -> tuple[str, list[str]]:
    requested = str(source_view or "").strip()
    requested_cols = get_duckdb_view_columns(duckdb_path=duckdb_path, source_view=requested)
    if requested != "v_project_panel_cn_a":
        return requested, requested_cols

    required = {str(x).strip() for x in required_fields if str(x).strip()}
    for candidate in (
        "v_project_market_daily_base_hot",
        "project_market_daily_base",
        "v_project_market_daily_base",
    ):
        candidate_cols = get_duckdb_view_columns(duckdb_path=duckdb_path, source_view=candidate)
        if not candidate_cols:
            continue
        if required.issubset(set(candidate_cols)):
            return candidate, candidate_cols
    return requested, requested_cols


def _stratified_limit_layered_candidates(
    candidates: Sequence[dict[str, Any]],
    max_n: int,
    layer_min_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    limit = max(1, int(max_n))
    rows = [dict(row) for row in candidates if str(row.get("expression", "")).strip()]
    if len(rows) <= limit:
        return rows
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    for layer, count in dict(layer_min_counts or {}).items():
        layer_text = str(layer or "").strip()
        needed = max(0, int(count))
        if not layer_text or needed <= 0:
            continue
        for row in rows:
            if len(selected) >= limit or needed <= 0:
                break
            if str(row.get("layer", "")) != layer_text:
                continue
            key = str(row.get("expression", ""))
            if key in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(key)
            needed -= 1
    for row in rows:
        if len(selected) >= limit:
            break
        key = str(row.get("expression", ""))
        if key in selected_keys:
            continue
        selected.append(row)
        selected_keys.add(key)
    return selected[:limit]


def plan_required_fields_for_closed_loop(
    duckdb_path: str,
    source_view: str,
    closed_loop_config: Any,
    universe_base_dir: str,
    universe_name: str,
) -> dict[str, Any]:
    columns = get_duckdb_view_columns(duckdb_path=duckdb_path, source_view=source_view)
    if not columns:
        return {
            "required_fields": list(closed_loop_config.base_frame_cols),
            "selected_expressions": [],
            "available_columns": [],
        }

    date_col = str(closed_loop_config.date_col)
    code_col = str(closed_loop_config.code_col)
    neutral_group_field = neutralization_group_field(
        getattr(closed_loop_config.mining_config.simulation, "neutralization", "NONE")
    )
    if neutral_group_field and neutral_group_field not in columns:
        mode = normalize_neutralization_mode(
            getattr(closed_loop_config.mining_config.simulation, "neutralization", "NONE")
        )
        raise ValueError(
            f"neutralization={mode} requires group field '{neutral_group_field}' in source view '{source_view}'"
        )
    group_fields = [g for g in closed_loop_config.group_fields if g in columns]
    included = {str(x).strip() for x in getattr(closed_loop_config, "include_fields", ()) if str(x).strip()}
    excluded = {str(x).strip() for x in closed_loop_config.exclude_fields if str(x).strip()}

    base_skip = {date_col, code_col, "date", "trade_date", "code", "znz_code"} | set(group_fields) | excluded
    searchable_from_catalog = get_searchable_fields_from_field_catalog(
        duckdb_path=duckdb_path,
        catalog_view="v_project_field_catalog",
        include_fields=included,
        include_factor_families=getattr(closed_loop_config, "include_factor_families", ()),
        exclude_factor_families=getattr(closed_loop_config, "exclude_factor_families", ()),
    )
    if searchable_from_catalog:
        available_fields = sorted(
            [c for c in columns if c not in base_skip and c in searchable_from_catalog and not is_leakage_field(c)]
        )
        field_source = "field_catalog"
    else:
        available_fields = sorted([c for c in columns if c not in base_skip and not is_leakage_field(c)])
        field_source = "view_columns"
    if included:
        available_fields = [c for c in available_fields if c in included]
    available_groups = set(group_fields)
    available_specs = field_catalog_to_specs(
        list(available_fields) + list(available_groups),
        group_fields=available_groups,
        vector_fields=getattr(closed_loop_config, "vector_fields", ()),
        explicit_exclude_fields=excluded,
    )

    mode = str(closed_loop_config.search_mode or "operator_only").strip().lower()
    candidate_expressions: list[str] = []
    if mode in {"template_only", "deep_hybrid"}:
        templates = load_templates(
            include_families=set(closed_loop_config.template_include_families)
            if closed_loop_config.template_include_families
            else None
        )
        tpl_space = build_search_space(
            templates=templates,
            pools=dict(closed_loop_config.template_pool_override or {}),
            include_families=set(closed_loop_config.template_include_families)
            if closed_loop_config.template_include_families
            else None,
            available_fields=set(available_fields),
            available_groups=available_groups,
            available_field_specs=available_specs,
            skip_templates_with_missing_group=closed_loop_config.mining_config.skip_templates_with_missing_group,
        )
        candidate_expressions.extend([expr for _, expr in tpl_space])

    if mode in {"operator_only", "deep_hybrid"}:
        if bool(closed_loop_config.use_signature_generator):
            op_space = build_signature_aware_search_space(
                available_fields=set(available_fields),
                available_groups=available_groups,
                config=closed_loop_config.deep_explore_config,
                field_specs=available_specs,
                excluded_fields=excluded,
            )
        else:
            op_space = build_operator_search_space(
                available_fields=set(available_fields),
                available_groups=available_groups,
                config=closed_loop_config.deep_explore_config,
                excluded_fields=excluded,
            )
        candidate_expressions.extend([expr for _, expr in op_space])

    if mode == "layered_v2":
        layer_space = LayeredExpressionBuilder().build(
            field_universe=FieldUniverse(specs=tuple(available_specs), excluded_fields=tuple(sorted(excluded))),
            feedback_hints={},
            config=LayeredBuilderConfig(
                max_order=int(getattr(closed_loop_config, "layer_max_order", 4)),
                max_candidates=int(
                    getattr(
                        closed_loop_config,
                        "layer_max_candidates",
                        max(400, int(closed_loop_config.max_eval_expressions) * 4),
                    )
                ),
                layer_budgets=dict(getattr(closed_loop_config, "layer_budgets", {}) or {}),
                windows=tuple(
                    int(w)
                    for w in getattr(
                        closed_loop_config.deep_explore_config,
                        "windows",
                        (5, 10, 22, 66, 132),
                    )
                ),
                include_gates=bool(getattr(closed_loop_config, "layer_include_gates", True)),
                enable_stateful_phase2_ops=bool(getattr(closed_loop_config, "enable_stateful_phase2_ops", False)),
                random_seed=int(getattr(closed_loop_config.deep_explore_config, "random_seed", 42)),
                layer_gate_families=tuple(
                    str(x)
                    for x in getattr(
                        closed_loop_config,
                        "layer_gate_families",
                        (
                            "liquidity_activity",
                            "moneyflow_pressure",
                            "price_trend",
                            "industry_activity",
                        ),
                    )
                ),
                layer_gate_max_total=int(getattr(closed_loop_config, "layer_gate_max_total", 24)),
                layer_gate_max_per_family=int(getattr(closed_loop_config, "layer_gate_max_per_family", 6)),
                layer_gate_seed_max=int(getattr(closed_loop_config, "layer_gate_seed_max", 18)),
                layer_enable_bucket_groups=bool(getattr(closed_loop_config, "layer_enable_bucket_groups", True)),
                layer_bucket_max_groups=int(getattr(closed_loop_config, "layer_bucket_max_groups", 12)),
                layer_bucket_max_composite_groups=int(
                    getattr(closed_loop_config, "layer_bucket_max_composite_groups", 6)
                ),
                layer_bucket_ranges=tuple(
                    str(x) for x in getattr(closed_loop_config, "layer_bucket_ranges", ("0,1,0.2",))
                ),
                layer_bucket_l1_max_total=int(getattr(closed_loop_config, "layer_bucket_l1_max_total", 24)),
                layer_bucket_l2_max_total=int(getattr(closed_loop_config, "layer_bucket_l2_max_total", 20)),
                max_field_count_for_bucket_l2=int(
                    getattr(
                        getattr(closed_loop_config, "mining_config", None),
                        "max_field_count",
                        4,
                    )
                ),
                layer_enable_recipe_lite=bool(getattr(closed_loop_config, "layer_enable_recipe_lite", True)),
                layer_recipe_max_total=int(getattr(closed_loop_config, "layer_recipe_max_total", 80)),
                layer_recipe_max_per_family=int(getattr(closed_loop_config, "layer_recipe_max_per_family", 16)),
                layer_role_pair_max_total=int(getattr(closed_loop_config, "layer_role_pair_max_total", 80)),
                layer_cross_family_pair_ratio=float(getattr(closed_loop_config, "layer_cross_family_pair_ratio", 0.15)),
                field_profile_lite_enabled=bool(getattr(closed_loop_config, "field_profile_lite_enabled", True)),
                field_profile_lite_min_coverage=float(
                    getattr(closed_loop_config, "field_profile_lite_min_coverage", 0.0)
                ),
                field_profile_lite_min_finite_rate=float(
                    getattr(closed_loop_config, "field_profile_lite_min_finite_rate", 0.0)
                ),
                field_profile_lite_top_fields_per_family=int(
                    getattr(
                        closed_loop_config,
                        "field_profile_lite_top_fields_per_family",
                        50,
                    )
                ),
                feedback_policy_lite_enabled=bool(getattr(closed_loop_config, "feedback_policy_lite_enabled", True)),
                layer_operator_tier=str(getattr(closed_loop_config, "layer_operator_tier", "stable")),
                layer_operator_expansion_max_total=int(
                    getattr(closed_loop_config, "layer_operator_expansion_max_total", 100)
                ),
            ),
        )
        max_preview = max(1, int(closed_loop_config.max_eval_expressions))
        preview_rows = [{"expression": item.expression, "layer": item.layer} for item in layer_space]
        limited_rows = _stratified_limit_layered_candidates(
            preview_rows,
            max_n=max_preview,
            layer_min_counts={
                "L3": max(1, max_preview // 8),
                "L4": max(1, max_preview // 8),
            },
        )
        candidate_expressions.extend([str(row.get("expression", "")) for row in limited_rows])

    dedup_candidates = list(dict.fromkeys([str(x) for x in candidate_expressions if str(x)]))
    dedup_candidates = dedup_candidates[: max(1, int(closed_loop_config.max_eval_expressions))]

    simulation_cfg_json = canonical_simulation_config_json(closed_loop_config.mining_config.simulation)
    seen_hashes = load_seen_expression_hashes_for_universe(
        base_dir=universe_base_dir,
        universe_name=universe_name,
        simulation_config_json=simulation_cfg_json,
    )
    fresh_candidates, _ = filter_new_expressions(dedup_candidates, seen_hashes)
    request_n = max(1, int(closed_loop_config.request_new_alphas))
    selected_expressions = fresh_candidates[:request_n]

    simulation_universe = str(getattr(closed_loop_config.mining_config.simulation, "universe", "") or "").strip()
    extra_fields = [simulation_universe] if simulation_universe else []
    if neutral_group_field:
        extra_fields.append(neutral_group_field)
    if bool(getattr(closed_loop_config, "include_double_sort", False)):
        extra_fields.extend(
            [
                str(getattr(closed_loop_config, "double_sort_control_col", "total_mv")),
                "circ_mv",
            ]
        )
    if bool(getattr(closed_loop_config, "apply_tradability_constraints", False)):
        extra_fields.extend(["can_buy", "can_sell"])

    required_fields = collect_required_fields_from_expressions(
        expressions=selected_expressions,
        base_fields=closed_loop_config.base_frame_cols,
        group_fields=group_fields,
        extra_fields=extra_fields,
    )

    if date_col not in required_fields:
        required_fields.append(date_col)
    if code_col not in required_fields:
        required_fields.append(code_col)

    required_fields = [f for f in required_fields if f in set(columns)]
    return {
        "required_fields": required_fields,
        "selected_expressions": selected_expressions,
        "available_columns": columns,
        "field_source": field_source,
    }


def get_searchable_fields_from_field_catalog(
    duckdb_path: str,
    catalog_view: str = "v_project_field_catalog",
    include_fields: Iterable[str] = (),
    include_factor_families: Iterable[str] = (),
    exclude_factor_families: Iterable[str] = (),
) -> set[str]:
    try:
        import duckdb  # type: ignore
    except Exception:
        return set()

    conn = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        try:
            catalog_df = conn.execute(f"SELECT * FROM {_qident(catalog_view)}").fetchdf()
        except Exception:
            return set()
    finally:
        conn.close()

    if catalog_df is None or catalog_df.empty:
        return set()
    if "field_name" not in catalog_df.columns:
        return set()

    included = {str(x).strip() for x in include_fields if str(x).strip()}
    include_families = {str(x).strip() for x in include_factor_families if str(x).strip()}
    exclude_families = {str(x).strip() for x in exclude_factor_families if str(x).strip()}
    out: set[str] = set()
    for _, row in catalog_df.iterrows():
        field_name = str(row.get("field_name", "") or "").strip()
        if not field_name:
            continue
        field_type = str(row.get("field_type", "") or "").strip().upper()
        dtype = str(row.get("dtype", "") or "").strip().upper()
        is_searchable = _to_bool(row.get("is_searchable", False), default=False)
        is_default_enabled = _to_bool(row.get("is_default_enabled", True), default=True)
        explicit_enabled = field_name in included
        category = str(row.get("category", "") or "").strip().lower()
        source_table = str(row.get("source_table", "") or "").strip()
        family = str(row.get("factor_family", "") or "").strip() or infer_factor_family(
            field_name, category=category, source_table=source_table
        )
        if include_families and family not in include_families and not explicit_enabled:
            continue
        if exclude_families and family in exclude_families and not explicit_enabled:
            continue
        if (
            is_searchable
            and (is_default_enabled or explicit_enabled)
            and field_type == "SCALAR"
            and _is_numeric_duckdb_dtype(dtype)
        ):
            out.add(field_name)
    return out


def _split_view_name(source_view: str) -> tuple[str | None, str]:
    parts = [p for p in str(source_view or "").split(".") if p]
    if not parts:
        raise ValueError("source_view is empty")
    if len(parts) == 1:
        return None, parts[0]
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError(f"Unsupported source_view format: {source_view}")


def _qident(name: str) -> str:
    parts = [p for p in str(name).split(".") if p]
    quoted: list[str] = []
    for part in parts:
        quoted.append('"' + str(part).replace('"', '""') + '"')
    return ".".join(quoted)


def _to_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if not text:
        return bool(default)
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _is_numeric_duckdb_dtype(dtype: str) -> bool:
    text = str(dtype or "").strip().upper()
    if not text:
        return False
    numeric_tokens = (
        "TINYINT",
        "SMALLINT",
        "INTEGER",
        "BIGINT",
        "HUGEINT",
        "UTINYINT",
        "USMALLINT",
        "UINTEGER",
        "UBIGINT",
        "FLOAT",
        "DOUBLE",
        "REAL",
        "DECIMAL",
        "NUMERIC",
    )
    return any(token in text for token in numeric_tokens)


_DEFAULT_MAX_TEMP_SIZE = "10GB"


def _build_duckdb_connection_config(
    duckdb_path: str,
    duckdb_settings: dict[str, Any] | None,
) -> dict[str, str]:
    if not isinstance(duckdb_settings, dict) or not duckdb_settings:
        db_dir = Path(duckdb_path).resolve().parent
        temp_dir = db_dir / f"{Path(duckdb_path).name}.tmp"
        return {
            "temp_directory": str(temp_dir.as_posix()),
            "max_temp_directory_size": _DEFAULT_MAX_TEMP_SIZE,
        }

    out: dict[str, str] = {}

    memory_limit = str(duckdb_settings.get("memory_limit", "") or "").strip()
    if memory_limit:
        out["memory_limit"] = memory_limit

    threads_raw = duckdb_settings.get("threads", "")
    threads_text = str(threads_raw or "").strip()
    if threads_text:
        try:
            threads_val = int(threads_text)
        except Exception:
            threads_val = 0
        if threads_val > 0:
            out["threads"] = str(threads_val)

    temp_directory = str(duckdb_settings.get("temp_directory", "") or "").strip()
    if temp_directory:
        try:
            temp_path = Path(temp_directory)
            if not temp_path.is_absolute():
                temp_path = Path(duckdb_path).resolve().parent / temp_path
            temp_path.mkdir(parents=True, exist_ok=True)
            out["temp_directory"] = str(temp_path.as_posix())
        except Exception:
            pass

    max_temp_size = str(duckdb_settings.get("max_temp_directory_size", "") or "").strip()
    if max_temp_size:
        out["max_temp_directory_size"] = max_temp_size
    elif "max_temp_directory_size" not in out:
        out["max_temp_directory_size"] = _DEFAULT_MAX_TEMP_SIZE
    return out
