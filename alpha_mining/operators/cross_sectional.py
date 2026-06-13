from __future__ import annotations

from .transform import _normalize, _quantile, _rank, _scale, _truncate, _zscore
from ..registry import OperatorRegistry


def register_operators(registry: OperatorRegistry) -> None:
    registry.register("rank", _rank)
    registry.register("zscore", _zscore)
    registry.register("normalize", _normalize)
    registry.register("scale", _scale)
    registry.register("quantile", _quantile)
    registry.register("cs_quantile", _quantile)
    registry.register("truncate", _truncate)
