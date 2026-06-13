from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


OperatorFn = Callable[..., Any]


@dataclass
class OperatorRegistry:
    """Name -> callable registry for expression engine operators.

    The registry maps operator names (e.g., 'ts_mean', 'cs_rank') to their
    implementation functions. Operators can be registered using the decorator
    pattern or direct registration.

    Example:
        >>> registry = OperatorRegistry()
        >>> @registry.register('my_op')
        ... def my_op(x, window):
        ...     return x.rolling(window).mean()
    """

    _operators: dict[str, OperatorFn] = field(default_factory=dict)

    def register(self, name: str, fn: OperatorFn | None = None):
        """Register an operator function.

        Can be used as a decorator or called directly.

        Args:
            name: Operator name to register.
            fn: Implementation function. If None, returns a decorator.

        Returns:
            The registered function, or a decorator if fn is None.

        Example:
            >>> # As decorator
            >>> @registry.register('my_op')
            ... def my_op(x): return x * 2

            >>> # Direct registration
            >>> registry.register('my_op', lambda x: x * 2)
        """
        if fn is None:

            def decorator(inner: OperatorFn) -> OperatorFn:
                self._operators[name] = inner
                return inner

            return decorator
        self._operators[name] = fn
        return fn

    def get(self, name: str) -> OperatorFn:
        """Get an operator function by name.

        Args:
            name: Operator name to retrieve.

        Returns:
            The operator implementation function.

        Raises:
            KeyError: If the operator is not registered.
        """
        if name not in self._operators:
            raise KeyError(f"Operator '{name}' is not registered")
        return self._operators[name]

    def has(self, name: str) -> bool:
        """Check if an operator is registered.

        Args:
            name: Operator name to check.

        Returns:
            True if the operator exists, False otherwise.
        """
        return name in self._operators

    def list_names(self) -> list[str]:
        """Get sorted list of all registered operator names.

        Returns:
            Sorted list of operator names.
        """
        return sorted(self._operators.keys())


def build_default_registry() -> OperatorRegistry:
    """Build registry with all built-in operator modules loaded."""
    registry = OperatorRegistry()

    from .operators import (
        arithmetic,
        cross_sectional,
        event_ops,
        group_ops,
        logical,
        regression,
        time_series,
        transform,
        vector_ops,
    )

    arithmetic.register_operators(registry)
    transform.register_operators(registry)
    time_series.register_operators(registry)
    cross_sectional.register_operators(registry)
    group_ops.register_operators(registry)
    logical.register_operators(registry)
    regression.register_operators(registry)
    vector_ops.register_operators(registry)
    event_ops.register_operators(registry)
    return registry
