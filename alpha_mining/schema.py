from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AlphaTemplate:
    """Serializable template object for candidate expression expansion."""

    template_id: str
    family: str
    expression: str
    placeholders: dict[str, Any] = field(default_factory=dict)
    required_fields: list[str] = field(default_factory=list)
    required_groups: list[str] = field(default_factory=list)
    required_field_types: list[str] = field(default_factory=list)
    default_settings: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    notes: str | None = None


@dataclass(frozen=True)
class AlphaSpec:
    """Concrete expression candidate after placeholder expansion."""

    alpha_name: str
    expression: str
    template_id: str
    template_family: str
    fields: tuple[str, ...] = ()
    groups: tuple[str, ...] = ()
    settings: dict[str, Any] = field(default_factory=dict)
