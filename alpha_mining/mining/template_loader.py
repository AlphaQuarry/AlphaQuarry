from __future__ import annotations

from pathlib import Path

from ..schema import AlphaTemplate

try:
    import yaml
except Exception as exc:  # pragma: no cover - import guard
    yaml = None
    _yaml_import_error = exc
else:
    _yaml_import_error = None


def load_templates(
    template_dir: str | Path | None = None,
    include_families: set[str] | None = None,
) -> list[AlphaTemplate]:
    """Load AlphaTemplate records from yaml files under template_dir."""
    if yaml is None:  # pragma: no cover - depends on runtime env
        raise ImportError("PyYAML is required to load template yaml files") from _yaml_import_error

    base = Path(template_dir) if template_dir is not None else _default_template_dir()
    if not base.exists():
        raise FileNotFoundError(base)

    templates: list[AlphaTemplate] = []
    for path in sorted(base.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        if not isinstance(raw, list):
            raise ValueError(f"Template yaml must be a list: {path}")
        for record in raw:
            tpl = _record_to_template(record)
            if include_families is not None and tpl.family not in include_families:
                continue
            templates.append(tpl)
    return templates


def _record_to_template(record: dict) -> AlphaTemplate:
    if "template_id" not in record or "family" not in record or "expression" not in record:
        raise ValueError("Template record missing required keys: template_id/family/expression")
    return AlphaTemplate(
        template_id=str(record["template_id"]),
        family=str(record["family"]),
        expression=str(record["expression"]),
        placeholders=dict(record.get("placeholders", {})),
        required_fields=list(record.get("required_fields", [])),
        required_groups=list(record.get("required_groups", [])),
        required_field_types=list(record.get("required_field_types", [])),
        default_settings=dict(record.get("default_settings", {})),
        tags=list(record.get("tags", [])),
        notes=record.get("notes"),
    )


def _default_template_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "templates"
