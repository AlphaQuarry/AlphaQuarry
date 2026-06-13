from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import AlphaMiningConfig
from ..engine import ExpressionEngine
from ..panel_store import PanelStore, VectorPanel
from ..schema import AlphaTemplate
from ..simulation import apply_simulation_settings
from ..simulation.neutralization import (
    neutralization_group_field,
    normalize_neutralization_mode,
)
from .dedup import deduplicate_expressions
from .prefilter import prefilter_candidates
from .search import build_search_space


@dataclass
class AlphaMiningPipeline:
    """MVP orchestrator for expression execution over one panel store."""

    config: AlphaMiningConfig
    engine: ExpressionEngine

    @classmethod
    def from_panel_store(
        cls, panel_store: PanelStore, config: AlphaMiningConfig | None = None
    ) -> "AlphaMiningPipeline":
        cfg = config or AlphaMiningConfig()
        return cls(config=cfg, engine=ExpressionEngine(panel_store=panel_store))

    def run_expressions(
        self,
        expressions: list[str],
        output_dtype: str | None = None,
        drop_all_nan_rows: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
        deduped = deduplicate_expressions(expressions)
        passed, failed = prefilter_candidates(
            deduped,
            max_operator_count=self.config.max_operator_count,
            max_field_count=self.config.max_field_count,
        )
        alpha_wide, _, _ = self._execute_expressions(
            passed,
            collect_profile=False,
            output_dtype=output_dtype,
            drop_all_nan_rows=drop_all_nan_rows,
        )
        return alpha_wide, failed

    def run_prepared_expressions(
        self,
        expressions: list[str],
        output_dtype: str | None = None,
        drop_all_nan_rows: bool = False,
    ) -> pd.DataFrame:
        """
        Execute expressions directly without dedup/prefilter.

        Use this when candidates were already deduplicated and validated externally.
        """
        alpha_wide, _, _ = self._execute_expressions(
            expressions,
            collect_profile=False,
            output_dtype=output_dtype,
            drop_all_nan_rows=drop_all_nan_rows,
        )
        return alpha_wide

    def run_prepared_expressions_with_profile(
        self,
        expressions: list[str],
        output_dtype: str | None = None,
        drop_all_nan_rows: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute prepared expressions and return timing diagnostics.

        Returns:
        - alpha_wide: date/code + alpha columns
        - expr_timing_df: per-expression timing breakdown
        - operator_timing_df: aggregated operator timing from ExpressionEngine
        """
        alpha_wide, expr_timing_df, operator_timing_df = self._execute_expressions(
            expressions,
            collect_profile=True,
            output_dtype=output_dtype,
            drop_all_nan_rows=drop_all_nan_rows,
        )
        return alpha_wide, expr_timing_df, operator_timing_df

    def run_templates(
        self,
        templates: list[AlphaTemplate],
        pools: dict[str, dict[str, list]] | None = None,
        output_dtype: str | None = None,
        drop_all_nan_rows: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
        pool_map = pools or {}
        include_families = (
            set(self.config.prioritized_template_families) if self.config.prioritized_template_families else None
        )
        expressions = [
            expr
            for _, expr in build_search_space(
                templates=templates,
                pools=pool_map,
                include_families=include_families,
                available_fields=set(self.engine.panel_store.available_scalar_fields())
                | set(self.engine.panel_store.available_vector_fields()),
                available_groups=set(self.engine.panel_store.available_group_like_fields()),
                skip_templates_with_missing_group=self.config.skip_templates_with_missing_group,
            )
        ]
        return self.run_expressions(
            expressions,
            output_dtype=output_dtype,
            drop_all_nan_rows=drop_all_nan_rows,
        )

    def _resolve_neutralization_group_panel(self) -> pd.DataFrame | None:
        mode = normalize_neutralization_mode(self.config.simulation.neutralization)
        group_name = neutralization_group_field(mode)
        if group_name is None:
            return None
        try:
            return self.engine.panel_store.get_group_like(group_name)
        except KeyError as exc:
            raise ValueError(
                f"neutralization={mode} requires group field '{group_name}' in PanelStore/source data"
            ) from exc

    def _resolve_universe_panel(self) -> pd.DataFrame | None:
        universe_field = self.config.simulation.universe
        if not universe_field:
            return None
        try:
            panel = self.engine.panel_store.get_field(universe_field)
        except KeyError:
            return None
        if isinstance(panel, VectorPanel):
            return None
        return panel

    def _execute_expressions(
        self,
        expressions: list[str],
        collect_profile: bool,
        output_dtype: str | None,
        drop_all_nan_rows: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.engine.clear_cache()

        neutralization_group_panel = self._resolve_neutralization_group_panel()
        universe_panel = self._resolve_universe_panel()

        if collect_profile:
            self.engine.enable_operator_profiling(reset=True)
        else:
            self.engine.disable_operator_profiling()

        np_dtype = _resolve_output_dtype(output_dtype)
        base_index: pd.MultiIndex | pd.Index | None = None
        alpha_names: list[str] = []
        out_matrix: np.ndarray | None = None
        expr_rows: list[dict[str, float | str]] = []

        for idx, expr in enumerate(expressions, start=1):
            alpha_name = f"alpha_{idx:04d}"

            t0 = time.perf_counter()
            raw_panel = self.engine.eval(expr, use_cache=False)
            t1 = time.perf_counter()
            adjusted = apply_simulation_settings(
                raw_panel,
                config=self.config.simulation,
                group_panel=neutralization_group_panel,
                universe_panel=universe_panel,
            )
            t2 = time.perf_counter()
            stacked = _stack_panel_compat(adjusted)
            if base_index is None:
                base_index = stacked.index
                out_matrix = np.full((len(base_index), len(expressions)), np.nan, dtype=np_dtype)
            elif not stacked.index.equals(base_index):
                stacked = stacked.reindex(base_index)

            stacked_num = pd.to_numeric(stacked, errors="coerce")
            assert out_matrix is not None  # for type checkers
            out_matrix[:, idx - 1] = np.asarray(stacked_num, dtype=np_dtype)
            alpha_names.append(alpha_name)
            t3 = time.perf_counter()

            if collect_profile:
                expr_rows.append(
                    {
                        "alpha_name": alpha_name,
                        "expression": expr,
                        "eval_sec": t1 - t0,
                        "simulation_sec": t2 - t1,
                        "stack_sec": t3 - t2,
                        "total_sec": t3 - t0,
                    }
                )
            del raw_panel, adjusted, stacked, stacked_num

        if out_matrix is None or base_index is None:
            alpha_wide = pd.DataFrame(columns=["date", "code"])
        else:
            if drop_all_nan_rows:
                row_mask = np.isfinite(out_matrix).any(axis=1)
                out_matrix = out_matrix[row_mask]
                base_index = base_index[row_mask]
            out = pd.DataFrame(out_matrix, index=base_index, columns=alpha_names)
            out.index.names = ["date", "code"]
            alpha_wide = out.reset_index()

        if not collect_profile:
            return (
                alpha_wide,
                pd.DataFrame(
                    columns=[
                        "alpha_name",
                        "expression",
                        "eval_sec",
                        "simulation_sec",
                        "stack_sec",
                        "total_sec",
                    ]
                ),
                pd.DataFrame(columns=["operator", "count", "total_sec", "avg_sec"]),
            )

        expr_timing_df = pd.DataFrame(expr_rows)
        operator_timing_df = self.engine.get_operator_profile()
        return alpha_wide, expr_timing_df, operator_timing_df


def _stack_alpha_panels(alpha_panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not alpha_panels:
        return pd.DataFrame(columns=["date", "code"])

    stacked_series: list[pd.Series] = []
    for alpha_name, panel in alpha_panels.items():
        stacked = _stack_panel_compat(panel)
        stacked.name = alpha_name
        stacked_series.append(stacked)

    out = pd.concat(stacked_series, axis=1)
    out.index.names = ["date", "code"]
    return out.reset_index()


def _stack_panel_compat(panel: pd.DataFrame) -> pd.Series:
    try:
        return panel.stack(dropna=False)
    except ValueError as exc:
        if "dropna must be unspecified" in str(exc):
            return panel.stack(future_stack=True)
        raise
    except TypeError:
        return panel.stack()


def _resolve_output_dtype(output_dtype: str | None) -> np.dtype:
    if output_dtype is None:
        return np.dtype("float64")
    try:
        resolved = np.dtype(output_dtype)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Invalid output_dtype: {output_dtype}") from exc
    if resolved.kind not in {"f"}:
        raise ValueError(f"output_dtype must be floating dtype, got {resolved}")
    return resolved
