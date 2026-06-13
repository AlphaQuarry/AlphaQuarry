from __future__ import annotations

import pandas as pd

from .decay import apply_decay
from .delay import apply_delay
from .neutralization import (
    apply_neutralization,
    neutralization_group_field,
    normalize_neutralization_mode,
)
from .scaling import apply_portfolio_scale
from .settings import AlphaSimulationConfig
from .truncation import apply_portfolio_truncation, apply_truncation
from .universe import apply_universe


def apply_simulation_settings(
    alpha_panel: pd.DataFrame,
    config: AlphaSimulationConfig,
    group_panel: pd.DataFrame | None = None,
    universe_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = alpha_panel
    out = apply_delay(out, config.delay)
    out = apply_decay(out, config.decay)
    out = apply_universe(out, universe_panel=universe_panel)
    portfolio_mode = str(getattr(config, "portfolio_construction", "signal") or "signal").strip().lower()
    if portfolio_mode == "wq_weight":
        out = apply_neutralization(out, config.neutralization, group_panel=group_panel)
        out = apply_portfolio_scale(
            out,
            scale_value=float(getattr(config, "scale_value", 1.0)),
            longscale=getattr(config, "longscale", None),
            shortscale=getattr(config, "shortscale", None),
        )
        out = apply_portfolio_truncation(
            out,
            cap=config.truncation,
            mode=str(getattr(config, "truncation_mode", "long_short_capped_rescale")),
        )
        out = apply_universe(out, universe_panel=universe_panel)
    else:
        out = apply_neutralization(out, config.neutralization, group_panel=group_panel)
        out = apply_truncation(out, config.truncation)
        out = apply_universe(out, universe_panel=universe_panel)
    return out


__all__ = [
    "AlphaSimulationConfig",
    "apply_decay",
    "apply_delay",
    "apply_neutralization",
    "normalize_neutralization_mode",
    "neutralization_group_field",
    "apply_portfolio_scale",
    "apply_simulation_settings",
    "apply_portfolio_truncation",
    "apply_truncation",
    "apply_universe",
]
