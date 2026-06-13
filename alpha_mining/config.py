from __future__ import annotations

from dataclasses import dataclass, field

from .simulation.settings import AlphaSimulationConfig


ExecutionMode = str


@dataclass(frozen=True)
class AlphaMiningConfig:
    """Top-level runtime configuration for alpha mining pipeline.

    Controls the expression generation process including:
    - Expression complexity limits (max operators, max fields)
    - Template family prioritization
    - Simulation settings for factor evaluation

    Attributes:
        mode: Execution mode ('brain_portable', 'brain_local').
        date_col: Name of the date column in input data.
        code_col: Name of the code column in input data.
        max_operator_count: Maximum operators per expression.
        max_field_count: Maximum distinct fields per expression.
        required_base_fields: Fields that must be present in input data.
        prioritized_template_families: Template families to prioritize.
        skip_templates_with_missing_group: Skip templates if group fields missing.
        simulation: AlphaSimulationConfig for factor simulation settings.

    Example:
        >>> config = AlphaMiningConfig(
        ...     max_operator_count=10,
        ...     max_field_count=4,
        ... )
    """

    mode: ExecutionMode = "brain_portable"
    date_col: str = "date"
    code_col: str = "code"
    max_operator_count: int = 8
    max_field_count: int = 3
    required_base_fields: tuple[str, ...] = ("pct_chg", "circ_mv")
    prioritized_template_families: tuple[str, ...] = (
        "single_ts",
        "single_cross",
        "single_group",
    )
    skip_templates_with_missing_group: bool = True
    simulation: AlphaSimulationConfig = field(default_factory=AlphaSimulationConfig)
