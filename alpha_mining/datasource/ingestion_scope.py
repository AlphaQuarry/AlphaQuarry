from __future__ import annotations

from typing import Iterable


FACT_GROUP_TABLES: dict[str, tuple[str, ...]] = {
    "p0": ("daily", "daily_basic", "adj_factor"),
    "p1": ("stk_limit", "suspend_d"),
    "p2": ("income_vip", "balancesheet_vip", "cashflow_vip", "fina_indicator_vip"),
    "p3": ("moneyflow",),
    "p3_legacy": ("moneyflow_ths",),
    "p4": ("cyq_perf", "cyq_chips", "stk_factor_pro"),
    "p4_auction": ("stk_auction_o", "stk_auction_c"),
    "p5": ("report_rc",),
    "index": ("index_daily", "index_weight"),
}

DIM_GROUP_TABLES: dict[str, tuple[str, ...]] = {
    "p0": ("trade_cal", "stock_basic", "index_classify", "index_member_all"),
    "p1": ("namechange",),
    "p3": ("ths_index", "ths_member"),
    "index": ("index_basic",),
}

_FACT_AUTO_DEPS: dict[str, tuple[str, ...]] = {
    "daily": ("adj_factor",),
}

_DIM_AUTO_DEPS: dict[str, tuple[str, ...]] = {
    "index_member_all": ("index_classify",),
}

DEFAULT_FACT_GROUPS: tuple[str, ...] = ("p0", "p1")
DEFAULT_DIM_GROUPS: tuple[str, ...] = ("p0", "p1")

# 分层预设: tier -> (fact_groups, dim_groups, 推荐最低积分)
# 积分要求基于 Tushare 官方文档:
#   2000: P0/P1/P2(非VIP)/P3/index
#   5000: P2(VIP)/P4_auction
#   10000: P4(cyq/stk_factor_pro)/P5(report_rc)
TIER_PRESETS: dict[str, tuple[tuple[str, ...], tuple[str, ...], int]] = {
    "basic": (("p0",), ("p0",), 2000),
    "standard": (("p0", "p1"), ("p0", "p1"), 2000),
    "extended": (("p0", "p1", "p2"), ("p0", "p1"), 5000),
    "full": (
        ("p0", "p1", "p2", "p3", "p3_legacy", "p4", "p4_auction", "p5", "index"),
        ("p0", "p1", "p3", "index"),
        10000,
    ),
}


def known_fact_tables() -> tuple[str, ...]:
    return _ordered_tables(FACT_GROUP_TABLES)


def known_dim_tables() -> tuple[str, ...]:
    return _ordered_tables(DIM_GROUP_TABLES)


def resolve_fact_table_selection(
    groups_raw: str = "",
    include_raw: str = "",
    exclude_raw: str = "",
) -> list[str]:
    return resolve_table_selection(
        group_tables=FACT_GROUP_TABLES,
        default_groups=DEFAULT_FACT_GROUPS,
        groups_raw=groups_raw,
        include_raw=include_raw,
        exclude_raw=exclude_raw,
        auto_deps=_FACT_AUTO_DEPS,
    )


def resolve_dim_table_selection(
    groups_raw: str = "",
    include_raw: str = "",
    exclude_raw: str = "",
) -> list[str]:
    return resolve_table_selection(
        group_tables=DIM_GROUP_TABLES,
        default_groups=DEFAULT_DIM_GROUPS,
        groups_raw=groups_raw,
        include_raw=include_raw,
        exclude_raw=exclude_raw,
        auto_deps=_DIM_AUTO_DEPS,
    )


def resolve_table_selection(
    group_tables: dict[str, tuple[str, ...]],
    default_groups: tuple[str, ...],
    groups_raw: str = "",
    include_raw: str = "",
    exclude_raw: str = "",
    auto_deps: dict[str, tuple[str, ...]] | None = None,
) -> list[str]:
    known_groups = {str(k).strip().lower() for k in group_tables.keys()}
    known_tables = _ordered_tables(group_tables)
    known_table_set = set(known_tables)

    group_tokens = _parse_csv_tokens(groups_raw)
    if not group_tokens:
        group_tokens = [str(x).strip().lower() for x in default_groups]
    if len(group_tokens) == 1 and group_tokens[0] == "none":
        selected: list[str] = []
    else:
        unknown_groups = [g for g in group_tokens if g not in known_groups]
        if unknown_groups:
            raise ValueError(f"Unknown groups: {unknown_groups}. Allowed groups: {sorted(known_groups)}")
        selected = []
        for group in group_tokens:
            for table in group_tables.get(group, ()):
                if table not in selected:
                    selected.append(table)

    include_tables = _parse_csv_tokens(include_raw)
    exclude_tables = _parse_csv_tokens(exclude_raw)
    unknown_include = [t for t in include_tables if t not in known_table_set]
    unknown_exclude = [t for t in exclude_tables if t not in known_table_set]
    if unknown_include:
        raise ValueError(f"Unknown include tables: {unknown_include}. Allowed tables: {list(known_tables)}")
    if unknown_exclude:
        raise ValueError(f"Unknown exclude tables: {unknown_exclude}. Allowed tables: {list(known_tables)}")

    for table in include_tables:
        if table not in selected:
            selected.append(table)
    if exclude_tables:
        selected = [t for t in selected if t not in set(exclude_tables)]

    if auto_deps:
        selected = _append_auto_dependencies(
            selected=selected,
            auto_deps=auto_deps,
            known_tables=known_tables,
        )
    return [t for t in known_tables if t in set(selected)]


def _append_auto_dependencies(
    selected: Iterable[str],
    auto_deps: dict[str, tuple[str, ...]],
    known_tables: tuple[str, ...],
) -> list[str]:
    out: list[str] = []
    out_set: set[str] = set()
    for item in selected:
        text = str(item or "").strip()
        if text and text not in out_set:
            out.append(text)
            out_set.add(text)

    changed = True
    while changed:
        changed = False
        for table in list(out):
            for dep in auto_deps.get(table, ()):
                if dep in out_set:
                    continue
                if dep not in set(known_tables):
                    continue
                out.append(dep)
                out_set.add(dep)
                changed = True
    return out


def _ordered_tables(group_tables: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    out: list[str] = []
    for group in group_tables.values():
        for table in group:
            if table not in out:
                out.append(table)
    return tuple(out)


def _parse_csv_tokens(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    out: list[str] = []
    for item in text.split(","):
        token = str(item or "").strip().lower()
        if token:
            out.append(token)
    return out
