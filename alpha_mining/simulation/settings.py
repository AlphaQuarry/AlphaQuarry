from __future__ import annotations

from dataclasses import dataclass


NeutralizationMode = str


@dataclass(frozen=True)
class AlphaSimulationConfig:
    delay: int = 1
    decay: int = 0
    neutralization: NeutralizationMode = "NONE"
    truncation: float | None = None
    pasteurization: bool = True
    # External universe boolean field name in PanelStore (date x code).
    universe: str | None = None
    portfolio_construction: str = "signal"  # signal / wq_weight
    scale_value: float = 1.0
    longscale: float | None = None
    shortscale: float | None = None
    truncation_mode: str = "clip"  # clip / capped_rescale / long_short_capped_rescale
    rescale_after_truncation: bool = True
