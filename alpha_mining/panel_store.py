from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from numbers import Number
from typing import Any

import pandas as pd


class _LRUPanelCache:
    """LRU cache for pivot panels with configurable max size. max_size=0 means unlimited."""

    def __init__(self, max_size: int = 0) -> None:
        self._max_size = max(0, int(max_size))
        self._data: OrderedDict[str, pd.DataFrame] = OrderedDict()

    def get(self, key: str) -> pd.DataFrame | None:
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def put(self, key: str, value: pd.DataFrame) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if self._max_size > 0:
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()


@dataclass(frozen=True)
class VectorPanel:
    """Exploded vector field backed by multiple date x code component panels."""

    name: str
    components: tuple[pd.DataFrame, ...]
    component_names: tuple[str, ...]

    @property
    def index(self) -> pd.Index:
        if not self.components:
            raise ValueError(f"VectorPanel '{self.name}' has no components")
        return self.components[0].index

    @property
    def columns(self) -> pd.Index:
        if not self.components:
            raise ValueError(f"VectorPanel '{self.name}' has no components")
        return self.components[0].columns


@dataclass
class PanelStore:
    """Container for date x code panels used by expression engine.

    PanelStore manages the pivot/unpivot lifecycle of market data fields.
    It supports three field types:
    - Scalar: numeric fields pivoted to date x code panels (e.g., close, volume)
    - Group: categorical fields for group operations (e.g., industry, sector)
    - Vector: multi-component fields (e.g., bid_ask_spread__0, bid_ask_spread__1)

    The store uses LRU caching to avoid repeated pivot operations and
    provides lazy loading from a raw long-format DataFrame.

    Attributes:
        max_panel_cache_size: Maximum panels to cache per type (0 = unlimited).
        meta: Metadata including date_col, code_col, vector_separator.
        raw_frame: Raw long-format DataFrame for lazy pivot operations.
        scalar_field_names: Names of available scalar fields.
        group_field_names: Names of available group fields.
        vector_component_map: Mapping of vector base names to component columns.

    Example:
        >>> store = PanelStore.from_long_frame(
        ...     df,
        ...     date_col='trade_date',
        ...     code_col='ts_code',
        ...     group_fields=['industry'],
        ... )
        >>> close_panel = store.get_scalar('close')
        >>> industry_panel = store.get_group('industry')
    """

    max_panel_cache_size: int = 0  # 0 = unlimited (backward compatible)
    meta: dict[str, Any] = field(default_factory=dict)

    raw_frame: pd.DataFrame | None = None
    scalar_field_names: tuple[str, ...] = ()
    group_field_names: tuple[str, ...] = ()
    vector_component_map: dict[str, tuple[str, ...]] = field(default_factory=dict)
    scalar_panels: _LRUPanelCache = field(init=False, repr=False)
    vector_panels: _LRUPanelCache = field(init=False, repr=False)
    group_panels: _LRUPanelCache = field(init=False, repr=False)
    _scalar_field_set: set[str] = field(default_factory=set, init=False, repr=False)
    _group_field_set: set[str] = field(default_factory=set, init=False, repr=False)
    _numeric_like_scalar_field_set: set[str] = field(default_factory=set, init=False, repr=False)
    _canonical_index: pd.Index | None = field(default=None, init=False, repr=False)
    _canonical_columns: pd.Index | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.scalar_panels = _LRUPanelCache(self.max_panel_cache_size)
        self.vector_panels = _LRUPanelCache(self.max_panel_cache_size)
        self.group_panels = _LRUPanelCache(self.max_panel_cache_size)
        self._scalar_field_set = set(self.scalar_field_names)
        self._group_field_set = set(self.group_field_names)
        numeric_like = self.meta.get("numeric_like_scalar_fields", ())
        self._numeric_like_scalar_field_set = set(str(x) for x in numeric_like)

    @classmethod
    def from_long_frame(
        cls,
        df: pd.DataFrame,
        date_col: str = "date",
        code_col: str = "code",
        scalar_fields: list[str] | None = None,
        group_fields: list[str] | None = None,
        vector_fields: list[str] | None = None,
        vector_separator: str = "__",
        max_panel_cache_size: int = 0,
    ) -> "PanelStore":
        """Create a PanelStore from a long-format DataFrame.

        Automatically detects scalar, group, and vector fields based on
        column types and naming conventions.

        Args:
            df: Long-format DataFrame with date and code columns.
            date_col: Name of the date column. Defaults to 'date'.
            code_col: Name of the code column. Defaults to 'code'.
            scalar_fields: Explicit list of scalar fields. If None, auto-detected.
            group_fields: List of categorical group fields (e.g., industry).
            vector_fields: List of vector base names. If None, auto-detected.
            vector_separator: Separator for vector component columns. Defaults to '__'.
            max_panel_cache_size: Maximum panels to cache per type (0 = unlimited).

        Returns:
            Initialized PanelStore instance.

        Raises:
            ValueError: If date_col or code_col not in DataFrame.

        Example:
            >>> df = pd.DataFrame({
            ...     'date': ['2024-01-01', '2024-01-01'],
            ...     'code': ['000001.SZ', '000002.SZ'],
            ...     'close': [10.5, 20.3],
            ...     'industry': ['银行', '房地产'],
            ... })
            >>> store = PanelStore.from_long_frame(df, group_fields=['industry'])
        """
        if date_col not in df.columns or code_col not in df.columns:
            raise ValueError(f"Input dataframe must contain '{date_col}' and '{code_col}'")

        ignore_cols = {date_col, code_col}
        all_fields = [c for c in df.columns if c not in ignore_cols]
        group_fields = [c for c in (group_fields or []) if c in df.columns]

        exploded_map = _detect_exploded_vector_components(
            all_fields=all_fields,
            vector_separator=vector_separator,
            selected_vector_fields=set(vector_fields or []),
        )

        exploded_component_fields = {comp_col for comp_list in exploded_map.values() for _, comp_col in comp_list}

        if scalar_fields is None:
            scalar_fields = [c for c in all_fields if c not in set(group_fields) and c not in exploded_component_fields]
        else:
            scalar_fields = [c for c in scalar_fields if c in df.columns and c not in exploded_component_fields]

        vector_component_map: dict[str, tuple[str, ...]] = {}
        for base_name, components in exploded_map.items():
            sorted_components = tuple(col for _, col in sorted(components, key=lambda t: t[0]))
            vector_component_map[base_name] = sorted_components

        selected_cols = _dedupe_preserve_order(
            [date_col, code_col]
            + list(scalar_fields)
            + list(group_fields)
            + [c for cols in vector_component_map.values() for c in cols]
        )

        working = df.loc[:, selected_cols].copy(deep=False)
        if not pd.api.types.is_datetime64_any_dtype(working[date_col]):
            working[date_col] = pd.to_datetime(working[date_col])
        has_duplicate_keys = bool(working.duplicated([date_col, code_col]).any())
        numeric_like_scalar_fields = tuple(
            f for f in scalar_fields if f in working.columns and _is_numeric_like_series(working[f])
        )

        return cls(
            max_panel_cache_size=max_panel_cache_size,
            raw_frame=working,
            scalar_field_names=tuple(scalar_fields),
            group_field_names=tuple(group_fields),
            vector_component_map=vector_component_map,
            meta={
                "date_col": date_col,
                "code_col": code_col,
                "vector_separator": vector_separator,
                "has_duplicate_keys": has_duplicate_keys,
                "numeric_like_scalar_fields": numeric_like_scalar_fields,
            },
        )

    def has_field(self, field: str) -> bool:
        """Check if a scalar field exists in the store.

        Args:
            field: Name of the scalar field to check.

        Returns:
            True if the field exists, False otherwise.
        """
        return field in self._scalar_field_set

    def has_vector(self, field: str) -> bool:
        """Check if a vector field exists in the store.

        Args:
            field: Base name of the vector field to check.

        Returns:
            True if the vector field exists, False otherwise.
        """
        return field in self.vector_component_map

    def has_group(self, group_field: str) -> bool:
        """Check if a group field exists in the store.

        Args:
            group_field: Name of the group field to check.

        Returns:
            True if the group field exists, False otherwise.
        """
        return group_field in self._group_field_set

    def get_scalar(self, field: str) -> pd.DataFrame:
        """Get a scalar field as a pivoted date x code panel.

        Results are cached using LRU policy to avoid repeated pivot operations.

        Args:
            field: Name of the scalar field.

        Returns:
            DataFrame with date as index and code as columns.

        Raises:
            KeyError: If the field does not exist in the store.

        Example:
            >>> close = store.get_scalar('close')
            >>> print(close.shape)  # (n_dates, n_codes)
        """
        if field not in self._scalar_field_set:
            raise KeyError(f"Scalar field '{field}' not found in PanelStore")
        cached = self.scalar_panels.get(field)
        if cached is not None:
            return cached
        panel = self._pivot_field(field)
        if field in self._numeric_like_scalar_field_set:
            panel = _coerce_panel_to_numeric(panel)
        self.scalar_panels.put(field, panel)
        return panel

    def get_vector(self, field: str) -> VectorPanel:
        """Get a vector field as a VectorPanel with multiple component panels.

        Args:
            field: Base name of the vector field.

        Returns:
            VectorPanel containing component panels.

        Raises:
            KeyError: If the vector field does not exist in the store.

        Example:
            >>> spread = store.get_vector('bid_ask_spread')
            >>> print(spread.component_names)  # ('bid_ask_spread__0', 'bid_ask_spread__1')
        """
        if field not in self.vector_component_map:
            raise KeyError(f"Vector field '{field}' not found in PanelStore")
        cached = self.vector_panels.get(field)
        if cached is not None:
            return cached
        component_names = self.vector_component_map[field]
        component_panels = tuple(_coerce_panel_to_numeric(self._pivot_field(col)) for col in component_names)
        vp = VectorPanel(
            name=field,
            components=component_panels,
            component_names=component_names,
        )
        self.vector_panels.put(field, vp)
        return vp

    def get_group(self, group_field: str) -> pd.DataFrame:
        """Get a group field as a pivoted date x code panel.

        Group fields are categorical (e.g., industry, sector) used for
        group-based operations like group_rank, group_mean, etc.

        Args:
            group_field: Name of the group field.

        Returns:
            DataFrame with date as index and code as columns.

        Raises:
            KeyError: If the group field does not exist in the store.

        Example:
            >>> industry = store.get_group('industry')
            >>> print(industry.head())
        """
        if group_field not in self._group_field_set:
            raise KeyError(f"Group field '{group_field}' not found in PanelStore")
        cached = self.group_panels.get(group_field)
        if cached is not None:
            return cached
        panel = self._pivot_field(group_field)
        self.group_panels.put(group_field, panel)
        return panel

    def get_group_like(self, group_field: str) -> pd.DataFrame:
        """Get a group-like field, falling back to scalar if not a group.

        This method is useful when a field could be either a group or scalar
        (e.g., 'sector' might be stored as either type).

        Args:
            group_field: Name of the group-like field.

        Returns:
            DataFrame with date as index and code as columns.

        Raises:
            KeyError: If the field is neither a group nor scalar.
        """
        if group_field in self._group_field_set:
            return self.get_group(group_field)
        if group_field in self._scalar_field_set:
            return self.get_scalar(group_field)
        raise KeyError(f"Group-like field '{group_field}' not found in PanelStore")

    def get_field(self, field: str) -> pd.DataFrame | VectorPanel:
        """Get any field by name, auto-detecting its type.

        Checks scalar, vector, and group fields in order.

        Args:
            field: Name of the field to retrieve.

        Returns:
            DataFrame for scalar/group fields, VectorPanel for vector fields.

        Raises:
            KeyError: If the field does not exist in any type.

        Example:
            >>> result = store.get_field('close')  # Returns DataFrame
            >>> result = store.get_field('bid_ask_spread')  # Returns VectorPanel
        """
        if field in self._scalar_field_set:
            return self.get_scalar(field)
        if field in self.vector_component_map:
            return self.get_vector(field)
        if field in self._group_field_set:
            return self.get_group(field)
        raise KeyError(f"Field '{field}' not found in PanelStore")

    def clear_cache(self) -> None:
        """Evict all cached pivot panels."""
        self.scalar_panels.clear()
        self.vector_panels.clear()
        self.group_panels.clear()

    def cache_stats(self) -> dict[str, int]:
        """Return current cache sizes for diagnostics."""
        return {
            "scalar": len(self.scalar_panels),
            "vector": len(self.vector_panels),
            "group": len(self.group_panels),
            "max_size": self.max_panel_cache_size,
        }

    def available_scalar_fields(self) -> list[str]:
        """Get list of available scalar field names.

        Returns:
            Sorted list of scalar field names.
        """
        return sorted(self.scalar_field_names)

    def available_vector_fields(self) -> list[str]:
        """Get list of available vector field base names.

        Returns:
            Sorted list of vector field base names.
        """
        return sorted(self.vector_component_map.keys())

    def available_group_fields(self) -> list[str]:
        """Get list of available group field names.

        Returns:
            Sorted list of group field names.
        """
        return sorted(self.group_field_names)

    def available_group_like_fields(self) -> list[str]:
        """Get list of available group-like field names.

        Includes both explicit group fields and non-numeric scalar fields
        that can be used as groups.

        Returns:
            Sorted list of group-like field names.
        """
        names = set(self._group_field_set)
        if self.raw_frame is not None:
            for field in self.scalar_field_names:
                if field in names:
                    continue
                if field in self.raw_frame.columns and not _is_numeric_like_series(self.raw_frame[field]):
                    names.add(field)
        return sorted(names)

    def _pivot_field(self, field: str) -> pd.DataFrame:
        if self.raw_frame is None:
            raise RuntimeError("PanelStore raw_frame is unavailable for lazy pivot")

        date_col = str(self.meta.get("date_col", "date"))
        code_col = str(self.meta.get("code_col", "code"))
        if bool(self.meta.get("has_duplicate_keys", False)):
            panel = self.raw_frame.pivot_table(
                index=date_col,
                columns=code_col,
                values=field,
                aggfunc="last",
                sort=False,
            )
        else:
            panel = self.raw_frame.pivot(index=date_col, columns=code_col, values=field)
        if self._canonical_index is None or self._canonical_columns is None:
            panel = panel.sort_index().sort_index(axis=1)
            self._canonical_index = panel.index
            self._canonical_columns = panel.columns
        else:
            panel = panel.reindex(index=self._canonical_index, columns=self._canonical_columns)
        return panel


def _detect_exploded_vector_components(
    all_fields: list[str],
    vector_separator: str,
    selected_vector_fields: set[str],
) -> dict[str, list[tuple[int, str]]]:
    pattern = re.compile(rf"^(?P<base>.+){re.escape(vector_separator)}(?P<idx>\d+)$")
    mapping: dict[str, list[tuple[int, str]]] = {}
    for col in all_fields:
        matched = pattern.match(col)
        if not matched:
            continue
        base = matched.group("base")
        if selected_vector_fields and base not in selected_vector_fields:
            continue
        idx = int(matched.group("idx"))
        mapping.setdefault(base, []).append((idx, col))
    return mapping


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _is_numeric_like_series(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    sample = series.dropna().head(128)
    if sample.empty:
        return False
    return bool(sample.map(lambda x: isinstance(x, Number)).all())


def _coerce_panel_to_numeric(panel: pd.DataFrame) -> pd.DataFrame:
    non_numeric_cols = [c for c in panel.columns if not pd.api.types.is_numeric_dtype(panel[c])]
    if not non_numeric_cols:
        return panel
    out = panel.copy()
    for c in non_numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out
