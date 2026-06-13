from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


from ..panel_store import PanelStore
from .explore import FieldSpec
from .factor_family import infer_factor_family
from .field_semantics import infer_field_semantic


LEAKAGE_FIELD_EXACT = {
    "date",
    "trade_date",
    "code",
    "znz_code",
    "ticker",
    "symbol",
    "universe",
    "tradable",
    "can_trade",
    "can_buy",
    "can_sell",
    "is_one_price_up_limit",
    "is_one_price_down_limit",
    "is_limit_up_close",
    "is_limit_down_close",
    "is_tradeable",
    "is_st",
    "is_suspended",
    "suspended",
    "pct_chg",
    "ret_1d",
    "future_return",
    "return_col",
    "next_return",
    "target",
    "label",
    "y",
}


LEAKAGE_FIELD_PATTERNS = (
    re.compile(r".*future.*", re.IGNORECASE),
    re.compile(r".*target.*", re.IGNORECASE),
    re.compile(r".*label.*", re.IGNORECASE),
    re.compile(r".*next_return.*", re.IGNORECASE),
    re.compile(r"ret_exec_.*", re.IGNORECASE),
    re.compile(r".*_audit_.*", re.IGNORECASE),
)


@dataclass(frozen=True)
class FieldUniverse:
    specs: tuple[FieldSpec, ...]
    excluded_fields: tuple[str, ...]

    @property
    def scalar_fields(self) -> list[str]:
        return sorted([s.name for s in self.specs if s.field_kind == "scalar"])

    @property
    def vector_fields(self) -> list[str]:
        return sorted([s.name for s in self.specs if s.field_kind == "vector"])

    @property
    def group_fields(self) -> list[str]:
        return sorted([s.name for s in self.specs if s.field_kind == "group"])

    @property
    def mask_fields(self) -> list[str]:
        return sorted([s.name for s in self.specs if s.field_kind == "mask"])

    def kind_map(self) -> dict[str, str]:
        return {s.name: s.field_kind for s in self.specs}


def is_leakage_field(name: str) -> bool:
    value = str(name or "").strip()
    if not value:
        return True
    low = value.lower()
    if low in LEAKAGE_FIELD_EXACT:
        return True
    return any(pattern.match(value) for pattern in LEAKAGE_FIELD_PATTERNS)


def build_field_universe(
    panel_store: PanelStore,
    explicit_include_fields: Iterable[str] = (),
    explicit_exclude_fields: Iterable[str] = (),
    group_fields: Iterable[str] = (),
    vector_fields: Iterable[str] = (),
    search_field_universe: Iterable[str] = (),
    allow_return_history_fields: bool = False,
) -> FieldUniverse:
    explicit_included = {str(x).strip() for x in explicit_include_fields if str(x).strip()}
    explicit_excluded = {str(x).strip() for x in explicit_exclude_fields if str(x).strip()}
    configured_groups = {str(x).strip() for x in group_fields if str(x).strip()}
    configured_vectors = {str(x).strip() for x in vector_fields if str(x).strip()}
    search_fields = {str(x).strip() for x in search_field_universe if str(x).strip()}

    group_like = set(panel_store.available_group_like_fields()) | configured_groups
    vectors = set(panel_store.available_vector_fields()) | configured_vectors
    scalars = set(panel_store.available_scalar_fields())
    if search_fields:
        scalars |= search_fields

    specs: list[FieldSpec] = []
    excluded: set[str] = set()

    for name in sorted(group_like):
        if _should_exclude(name, explicit_excluded, allow_return_history_fields):
            excluded.add(name)
            continue
        specs.append(FieldSpec(name=name, field_kind="group", categories=("group",), factor_family=""))

    for name in sorted(vectors):
        if explicit_included and name not in explicit_included:
            excluded.add(name)
            continue
        if _should_exclude(name, explicit_excluded, allow_return_history_fields):
            excluded.add(name)
            continue
        specs.append(
            FieldSpec(
                name=name,
                field_kind="vector",
                categories=("vector",),
                factor_family=infer_factor_family(name, category="vector"),
            )
        )

    for name in sorted(scalars - group_like - vectors):
        if explicit_included and name not in explicit_included:
            excluded.add(name)
            continue
        if _should_exclude(name, explicit_excluded, allow_return_history_fields):
            excluded.add(name)
            continue
        role = _infer_role(name)
        specs.append(
            FieldSpec(
                name=name,
                field_kind="scalar",
                categories=(role,),
                factor_family=infer_factor_family(name, category=role),
            )
        )

    return FieldUniverse(specs=tuple(specs), excluded_fields=tuple(sorted(excluded | explicit_excluded)))


def field_catalog_to_specs(
    field_names: Iterable[str],
    group_fields: Iterable[str] = (),
    vector_fields: Iterable[str] = (),
    explicit_exclude_fields: Iterable[str] = (),
) -> list[FieldSpec]:
    groups = {str(x).strip() for x in group_fields if str(x).strip()}
    vectors = {str(x).strip() for x in vector_fields if str(x).strip()}
    excluded = {str(x).strip() for x in explicit_exclude_fields if str(x).strip()}
    out: list[FieldSpec] = []
    for name in sorted({str(x).strip() for x in field_names if str(x).strip()}):
        if _should_exclude(name, excluded, allow_return_history_fields=False):
            continue
        if name in groups:
            out.append(
                FieldSpec(
                    name=name,
                    field_kind="group",
                    categories=("group",),
                    factor_family="",
                )
            )
        elif name in vectors:
            out.append(
                FieldSpec(
                    name=name,
                    field_kind="vector",
                    categories=("vector",),
                    factor_family=infer_factor_family(name, category="vector"),
                )
            )
        else:
            role = _infer_role(name)
            out.append(
                FieldSpec(
                    name=name,
                    field_kind="scalar",
                    categories=(role,),
                    factor_family=infer_factor_family(name, category=role),
                )
            )
    return out


def _should_exclude(name: str, explicit_excluded: set[str], allow_return_history_fields: bool) -> bool:
    if name in explicit_excluded:
        return True
    if allow_return_history_fields and name.startswith("ret_") and "exec" not in name:
        return False
    return is_leakage_field(name)


def _infer_role(name: str) -> str:
    return infer_field_semantic(name).role
