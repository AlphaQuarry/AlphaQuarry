from __future__ import annotations

from ..registry import OperatorRegistry


def register_operators(registry: OperatorRegistry) -> None:
    registry.register("event_active", _event_active)
    registry.register("event_decay", _event_decay)


def _event_active(event_signal, alpha):
    """Simple event gate: keep alpha only where event signal is true."""
    return alpha.where(event_signal)


def _event_decay(event_signal, alpha, half_life: int = 5):
    """Simplified event decay placeholder for MVP."""
    gated = alpha.where(event_signal)
    return gated.ewm(halflife=max(int(half_life), 1), min_periods=1, adjust=False).mean()
