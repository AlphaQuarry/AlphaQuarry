from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


DEFAULT_EXCLUDE_FIELDS = (
    "date",
    "trade_date",
    "code",
    "znz_code",
    "universe",
    "tradable",
    "can_trade",
    "can_buy",
    "can_sell",
    "is_one_price_up_limit",
    "is_one_price_down_limit",
    "is_limit_up_close",
    "is_limit_down_close",
    "is_st",
    "is_suspended",
    "pct_chg",
    "ret_1d",
    "future_return",
    "target",
    "label",
)


@dataclass(frozen=True)
class FieldPreprocessConfig:
    enabled: bool = True
    mode: str = "expression_wrapper"
    ts_backfill_window: int = 120
    winsorize_std: float = 4.0
    apply_to_kinds: tuple[str, ...] = ("scalar",)
    exclude_roles: tuple[str, ...] = ("group", "mask", "vector", "event")
    exclude_fields: tuple[str, ...] = DEFAULT_EXCLUDE_FIELDS
    exempt_from_complexity: bool = True


@dataclass(frozen=True)
class FieldExpression:
    raw_field: str
    expression: str
    preprocess_expression: str
    preprocess_operators: tuple[str, ...]


class FieldExpressionFactory:
    def __init__(self, config: FieldPreprocessConfig | None = None) -> None:
        self.config = config or FieldPreprocessConfig()

    def wrap_scalar_field(self, field: str) -> FieldExpression:
        raw = str(field or "").strip()
        if not raw:
            return FieldExpression(raw, raw, "", tuple())
        if not self._should_wrap(raw, kind="scalar"):
            return FieldExpression(raw, raw, "", tuple())
        window = max(1, int(self.config.ts_backfill_window))
        std = float(self.config.winsorize_std)
        expr = f"winsorize(ts_backfill({raw}, {window}), {std:.1f})"
        return FieldExpression(
            raw_field=raw,
            expression=expr,
            preprocess_expression=expr,
            preprocess_operators=("ts_backfill", "winsorize"),
        )

    def expression_for(self, field: str, kind: str = "scalar", role: str = "") -> str:
        raw = str(field or "").strip()
        if not self._should_wrap(raw, kind=kind, role=role):
            return raw
        return self.wrap_scalar_field(raw).expression

    def expression_map(self, fields: Iterable[str], kind: str = "scalar") -> dict[str, str]:
        return {str(f): self.expression_for(str(f), kind=kind) for f in fields if str(f)}

    def _should_wrap(self, field: str, kind: str = "scalar", role: str = "") -> bool:
        cfg = self.config
        if not bool(cfg.enabled):
            return False
        if str(cfg.mode or "expression_wrapper") != "expression_wrapper":
            return False
        kind_norm = str(kind or "").strip().lower()
        role_norm = str(role or "").strip().lower()
        if kind_norm not in {str(x).lower() for x in cfg.apply_to_kinds}:
            return False
        if kind_norm in {str(x).lower() for x in cfg.exclude_roles}:
            return False
        if role_norm and role_norm in {str(x).lower() for x in cfg.exclude_roles}:
            return False
        if str(field).strip() in set(cfg.exclude_fields):
            return False
        return True
