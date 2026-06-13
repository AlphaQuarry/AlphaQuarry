from __future__ import annotations

import json
from collections import Counter
from typing import Iterable

from .field_semantics import infer_field_semantic


FACTOR_FAMILIES: tuple[str, ...] = (
    "price_volume",
    "fundamental",
    "moneyflow",
    "analyst",
)


def infer_factor_family(field_name: str, category: str = "", source_table: str = "") -> str:
    name = str(field_name or "").strip().lower()
    cat = str(category or "").strip().lower()
    source = str(source_table or "").strip().lower()
    text = " ".join([name, cat, source])

    if not name:
        return "price_volume"
    semantic = infer_field_semantic(name, (cat,) if cat else ())
    if semantic.factor_family in {"moneyflow", "analyst", "fundamental"}:
        return semantic.factor_family
    if cat in {"filter", "identity", "metadata", "event", "industry"} and not any(
        token in text for token in ["moneyflow", "report_rc", "analyst", "fin_", "finance"]
    ):
        return "price_volume"
    if "moneyflow" in text or name.startswith(("mf_", "mfl_")):
        return "moneyflow"
    if "report_rc" in text or "analyst" in text or "forecast" in text or "rating" in text:
        return "analyst"
    if name.startswith("fin_") or "finance" in text or cat in {"finance", "fundamental"}:
        return "fundamental"
    if any(token in text for token in ["income", "profit", "cashflow", "asset", "liab", "roe", "roa"]):
        return "fundamental"
    return "price_volume"


def infer_factor_family_mix(
    field_names: Iterable[str],
    category_map: dict[str, str] | None = None,
    source_table_map: dict[str, str] | None = None,
) -> tuple[str, str]:
    categories = dict(category_map or {})
    sources = dict(source_table_map or {})
    counts: Counter[str] = Counter()
    for field in field_names:
        name = str(field or "").strip()
        if not name:
            continue
        family = infer_factor_family(name, category=categories.get(name, ""), source_table=sources.get(name, ""))
        if family:
            counts[family] += 1
    if not counts:
        payload = {
            "counts": {"price_volume": 0},
            "ratios": {"price_volume": 0.0},
            "primary_factor_family": "price_volume",
        }
        return "price_volume", json.dumps(payload, ensure_ascii=False, sort_keys=True)
    ordered = {key: int(counts[key]) for key in sorted(counts)}
    total = float(sum(ordered.values()))
    ratios = {key: float(value) / total if total > 0 else 0.0 for key, value in ordered.items()}
    primary = sorted(ordered.items(), key=lambda item: (-int(item[1]), item[0]))[0][0]
    payload = {
        "counts": ordered,
        "ratios": ratios,
        "primary_factor_family": primary,
    }
    return ",".join(ordered.keys()), json.dumps(payload, ensure_ascii=False, sort_keys=True)


def primary_factor_family_from_mix(factor_family: str, factor_family_mix_json: str = "") -> str:
    try:
        payload = json.loads(str(factor_family_mix_json or "{}"))
    except Exception:
        payload = {}
    primary = str(payload.get("primary_factor_family", "") or "").strip() if isinstance(payload, dict) else ""
    if primary:
        return primary
    families = [x.strip() for x in str(factor_family or "").split(",") if x.strip()]
    return families[0] if families else "price_volume"
