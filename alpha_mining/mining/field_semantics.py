from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSemantic:
    name: str
    tokens: tuple[str, ...]
    role: str
    factor_family: str
    gate_family: str
    bucket_family: str
    is_price: bool = False
    is_liquidity: bool = False
    is_moneyflow: bool = False
    is_size: bool = False
    is_valuation: bool = False
    is_chip: bool = False
    is_technical: bool = False
    is_analyst: bool = False
    is_finance: bool = False


def tokenize_field_name(name: str) -> tuple[str, ...]:
    text = str(name or "").strip()
    if not text:
        return ()
    with_boundaries = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    raw = re.split(r"[_\.\s]+", with_boundaries.lower())
    tokens: list[str] = []
    for part in raw:
        for token in re.findall(r"[a-z]+|\d+", part):
            if token:
                tokens.append(token)
    return tuple(tokens)


def infer_field_semantic(name: str, categories: tuple[str, ...] = ()) -> FieldSemantic:
    field = str(name or "").strip()
    tokens = tokenize_field_name(field)
    token_set = set(tokens)
    full = field.lower()
    cats = {str(x).strip().lower() for x in categories if str(x).strip()}

    is_moneyflow = (
        "moneyflow" in full
        or "net_mf" in full
        or "mfl" in token_set
        or "mf" in token_set
        or bool(token_set.intersection({"buy", "sell", "elg", "lg", "md", "sm"}))
        and ("amount" in token_set or "vol" in token_set)
    )
    is_analyst = "analyst" in full or "report_rc" in full or bool(token_set.intersection({"forecast", "rating"}))
    is_chip = (
        full.startswith("cyq_")
        or "cyq" in token_set
        or bool(token_set.intersection({"chip", "holder", "concentration", "winner", "cost"}))
    )
    is_technical = (
        full.startswith("tech_")
        or "technical" in cats
        or bool(token_set.intersection({"tech", "rsi", "mfi", "macd", "atr", "vr", "boll", "kdj", "cci", "obv"}))
    )
    is_valuation = bool(token_set.intersection({"pe", "pb", "ps", "pcf", "dv"})) or "valuation" in cats
    is_finance = (
        full.startswith("fin_")
        or "finance" in cats
        or "fundamental" in cats
        or bool(
            token_set.intersection(
                {
                    "income",
                    "profit",
                    "cashflow",
                    "roe",
                    "roa",
                    "asset",
                    "debt",
                    "liab",
                    "revenue",
                    "margin",
                }
            )
        )
    )
    is_size = _is_size_field(full, token_set)
    if is_size:
        is_valuation = False
    is_liquidity = not is_moneyflow and (
        "liquidity" in cats
        or bool(token_set.intersection({"amount", "vol", "volume", "turnover"}))
        or full in {"turnover_rate", "turnover_rate_f", "volume_ratio"}
    )
    is_price = not is_moneyflow and (
        "price" in cats or bool(token_set.intersection({"open", "high", "low", "close", "price", "vwap", "limit"}))
    )

    if is_moneyflow:
        role = "moneyflow"
    elif is_analyst:
        role = "analyst"
    elif is_chip:
        role = "chip"
    elif is_technical:
        role = "technical"
    elif is_valuation:
        role = "valuation"
    elif is_finance:
        role = "finance"
    elif is_size:
        role = "size"
    elif is_liquidity:
        role = "liquidity"
    elif is_price:
        role = "price"
    else:
        role = _category_role(cats)

    factor_family = _factor_family_for_role(role)
    gate_family = _gate_family_for_role(role)
    bucket_family = _bucket_family_for_role(role)
    return FieldSemantic(
        name=field,
        tokens=tokens,
        role=role,
        factor_family=factor_family,
        gate_family=gate_family,
        bucket_family=bucket_family,
        is_price=role == "price",
        is_liquidity=role == "liquidity",
        is_moneyflow=role == "moneyflow",
        is_size=role == "size",
        is_valuation=role == "valuation",
        is_chip=role == "chip",
        is_technical=role == "technical",
        is_analyst=role == "analyst",
        is_finance=role == "finance",
    )


def _is_size_field(full: str, tokens: set[str]) -> bool:
    if full in {"circ_mv", "total_mv", "float_mv", "free_float_mv", "market_cap"}:
        return True
    if "mv" in tokens or "cap" in tokens:
        return True
    return "market_cap" in full


def _category_role(categories: set[str]) -> str:
    for role in (
        "moneyflow",
        "analyst",
        "chip",
        "technical",
        "valuation",
        "finance",
        "size",
        "liquidity",
        "price",
    ):
        if role in categories:
            return role
    return "unknown"


def _factor_family_for_role(role: str) -> str:
    if role == "moneyflow":
        return "moneyflow"
    if role == "analyst":
        return "analyst"
    if role == "finance":
        return "fundamental"
    return "price_volume"


def _gate_family_for_role(role: str) -> str:
    return {
        "liquidity": "liquidity_activity",
        "moneyflow": "moneyflow_pressure",
        "price": "price_trend",
    }.get(role, "")


def _bucket_family_for_role(role: str) -> str:
    if role in {"size", "liquidity", "valuation", "chip", "technical"}:
        return role
    return ""
