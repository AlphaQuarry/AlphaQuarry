from __future__ import annotations

from .expand import expand_template
from ..schema import AlphaTemplate
from .field_preprocessing import FieldExpressionFactory, FieldPreprocessConfig


def build_search_space(
    templates: list[AlphaTemplate],
    pools: dict[str, dict[str, list]],
    include_families: set[str] | None = None,
    available_fields: set[str] | None = None,
    available_groups: set[str] | None = None,
    available_field_specs: list | None = None,
    skip_templates_with_missing_group: bool = True,
    field_preprocess_config: FieldPreprocessConfig | None = None,
) -> list[tuple[str, str]]:
    """
    Build expression search space.

    Returns list of (template_id, expression).
    """
    out: list[tuple[str, str]] = []
    for tpl in templates:
        if include_families is not None and tpl.family not in include_families:
            continue

        if available_fields is not None and tpl.required_fields:
            if any(field not in available_fields for field in tpl.required_fields):
                continue

        pool = dict(tpl.placeholders or {})
        pool.update(pools.get(tpl.template_id, {}))
        kind_map = {
            str(getattr(s, "name", "")): str(getattr(s, "field_kind", "")) for s in (available_field_specs or [])
        }
        effective_preprocess_config = field_preprocess_config or FieldPreprocessConfig(enabled=False)
        field_factory = FieldExpressionFactory(effective_preprocess_config)

        # Restrict placeholder field candidates to fields present in the local dataset.
        if available_fields is not None:
            for key, values in list(pool.items()):
                if "field" not in key:
                    continue
                filtered = [v for v in values if str(v) in available_fields]
                if tpl.required_field_types:
                    want_group = "group" in str(key).lower()
                    want_vector = "vector" in str(key).lower()
                    if want_group:
                        filtered = [v for v in filtered if kind_map.get(str(v)) == "group"]
                    elif want_vector:
                        filtered = [v for v in filtered if kind_map.get(str(v)) == "vector"]
                    else:
                        filtered = [v for v in filtered if kind_map.get(str(v), "scalar") == "scalar"]
                want_group = "group" in str(key).lower()
                want_vector = "vector" in str(key).lower()
                if not want_group and not want_vector:
                    filtered = [field_factory.expression_for(str(v), kind="scalar") for v in filtered]
                pool[key] = filtered

        # If template depends on groups, skip when required groups are unavailable.
        if skip_templates_with_missing_group:
            required_groups = set(tpl.required_groups or [])
            if available_groups is not None and required_groups:
                if not required_groups.issubset(available_groups):
                    continue

            if available_groups is not None:
                group_values = pool.get("group")
                if group_values is not None:
                    filtered_groups = [g for g in group_values if str(g) in available_groups]
                    if not filtered_groups:
                        continue
                    pool["group"] = filtered_groups

        for expr in expand_template(tpl, pool):
            out.append((tpl.template_id, expr))
    return out
