from __future__ import annotations

from pathlib import Path
import re
import warnings

import pandas as pd


REQUIRED_COLUMNS = ["field_name", "field_type", "category"]
DEFAULT_PROJECT_FIELD_CATALOG_PATH = Path("data/lake/meta/field_catalog.parquet")


def load_field_catalog(path: str | Path) -> pd.DataFrame:
    """Compatibility loader for legacy CSV/Excel field catalogs.

    Note:
    - Mainline datasource flow is now driven by `data/lake/meta/field_catalog.parquet`
      and DuckDB view `v_project_field_catalog`.
    - This function is kept for backward compatibility only.
    """
    warnings.warn(
        "load_field_catalog(path) is a compatibility API. "
        "Prefer datasource field_catalog builder + v_project_field_catalog.",
        RuntimeWarning,
        stacklevel=2,
    )
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(p)
    elif suffix == ".csv":
        df = pd.read_csv(p)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(p)
    else:
        raise ValueError("Supported field catalog formats: parquet/csv/excel")
    validate_field_catalog(df)
    return df


def load_project_field_catalog(
    path: str | Path = DEFAULT_PROJECT_FIELD_CATALOG_PATH,
) -> pd.DataFrame:
    """Load project canonical field catalog parquet."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_parquet(p)
    validate_field_catalog(df)
    return df


def validate_field_catalog(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Field catalog missing required columns: {missing}")


def filter_fields(df: pd.DataFrame, category: str | None = None, field_type: str | None = None) -> list[str]:
    out = df
    if category is not None:
        out = out[out["category"] == category]
    if field_type is not None:
        out = out[out["field_type"] == field_type]
    return out["field_name"].dropna().astype(str).tolist()


def infer_exploded_vector_bases(columns: list[str], separator: str = "__") -> list[str]:
    """Infer vector base names from exploded columns such as analyst_eps__0."""
    pattern = re.compile(rf"^(?P<base>.+){re.escape(separator)}\d+$")
    bases = set()
    for col in columns:
        matched = pattern.match(str(col))
        if matched:
            bases.add(matched.group("base"))
    return sorted(bases)
