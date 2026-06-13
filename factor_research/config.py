from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FactorResearchConfig:
    """Shared configuration for factor research functions."""

    period: int = 1
    layers: int = 5
    quantiles: int = 10
    max_lag: int | None = None
    market_value_column: str = "circ_mv"
    return_col: str = "pct_chg"
    threshold: float = 0.7
