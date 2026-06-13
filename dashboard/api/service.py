from __future__ import annotations

import json
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import pandas as pd
import numpy as np
from factor_research import SampleSplitConfig, build_phase_windows
from factor_research.utils import calculate_risk_metrics
from alpha_mining.workflow.factor_library import (
    FactorLibraryConfig,
    check_factor_library_candidate as workflow_check_factor_library_candidate,
    submit_factor_library_candidate as workflow_submit_factor_library_candidate,
)
from alpha_mining.workflow.superalpha import (
    SUPERALPHA_FACTOR,
    SuperalphaConfig,
    SuperalphaError,
    list_superalpha_components as workflow_list_superalpha_components,
    run_superalpha_backtest as workflow_run_superalpha_backtest,
)
from alpha_mining.workflow.universe_store import load_universe_alpha_values
from alpha_mining.live.artifacts import live_paths, read_json
from alpha_mining.live.config import load_live_config
from alpha_mining.live.registry import activate_superalpha as live_activate_superalpha
from alpha_mining.live.registry import list_live_superalphas as live_list_superalphas
from alpha_mining.live.registry import (
    update_superalpha_status as live_update_superalpha_status,
)
from .data_health import (
    FIELD_CATALOG_STALE_AFTER_DAYS,
    LOW_COVERAGE_THRESHOLD,
    base_frame_summary,
    coverage_counts,
    data_health_families,
    low_coverage_count,
    quality_artifact_summary,
    run_health_summary,
)
from .run_compare import compare_artifact_status, compare_metric_rows, top_overlap


DASHBOARD_METRICS_KEY = "dashboard_factor_metrics"
PORTFOLIO_PNL_KEY = "portfolio_pnl_df"
BENCHMARK_PNL_KEY = "benchmark_pnl_df"
VISUALIZATION_MANIFEST_KEY = "visualization_manifest"
PHASE_METRICS_KEY = "phase_metrics_df"
IC_DF_KEY = "ic_df"
ANALYSIS_DISTRIBUTION_HISTOGRAM_KEY = "analysis_distribution_histogram"
ANALYSIS_IC_DECAY_KEY = "analysis_ic_decay"
ANALYSIS_FACTOR_COVERAGE_BY_DATE_KEY = "analysis_factor_coverage_by_date"
SCOREBOARD_RUN_ID = "__scoreboard__"
SCOREBOARD_RUN_LABEL = "All analyzed factors / Scoreboard"
FEEDBACK_SCORE_CANDIDATES = [
    "feedback_score",
    "feedback_score_net",
    "train_score_total",
    "train_score_total_net",
    "train_score",
    "score_total",
    "score_total_net",
    "feedback_score_gross",
    "score_total_gross",
    "scoreboard_score",
]


@dataclass(frozen=True)
class AnalysisRun:
    universe: str
    run_id: str
    period: int
    layers: int
    created_at_utc: str
    analysis_dir: Path
    meta_path: Path
    table_paths: dict[str, Path]
    factor_metrics_path: Path | None
    alpha_names: tuple[str, ...]
    extra_meta: dict[str, Any]

    @property
    def dashboard_metrics_path(self) -> Path | None:
        return self.table_paths.get(DASHBOARD_METRICS_KEY)

    @property
    def metrics_path(self) -> Path | None:
        dashboard_path = self.dashboard_metrics_path
        if dashboard_path is not None and dashboard_path.exists():
            return dashboard_path
        if self.factor_metrics_path is not None and self.factor_metrics_path.exists():
            return self.factor_metrics_path
        return dashboard_path or self.factor_metrics_path

    @property
    def pnl_path(self) -> Path | None:
        return self.table_paths.get(PORTFOLIO_PNL_KEY)

    @property
    def benchmark_pnl_path(self) -> Path | None:
        return self.table_paths.get(BENCHMARK_PNL_KEY)

    @property
    def visualization_manifest_path(self) -> Path | None:
        return self.table_paths.get(VISUALIZATION_MANIFEST_KEY)

    @property
    def phase_metrics_path(self) -> Path | None:
        return self.table_paths.get(PHASE_METRICS_KEY)

    @property
    def ic_path(self) -> Path | None:
        return self.table_paths.get(IC_DF_KEY)

    @property
    def analysis_distribution_histogram_path(self) -> Path | None:
        return self.table_paths.get(ANALYSIS_DISTRIBUTION_HISTOGRAM_KEY)

    @property
    def analysis_ic_decay_path(self) -> Path | None:
        return self.table_paths.get(ANALYSIS_IC_DECAY_KEY)

    @property
    def analysis_factor_coverage_by_date_path(self) -> Path | None:
        return self.table_paths.get(ANALYSIS_FACTOR_COVERAGE_BY_DATE_KEY)


class DashboardStore:
    def __init__(self, store_root: str | Path = "data/alpha_universe_store") -> None:
        self.store_root = Path(store_root)
        self.scan_cache_ttl_seconds = float(os.environ.get("FACTOR_DASHBOARD_SCAN_CACHE_TTL_SECONDS", "5"))
        self.table_cache_max_items = int(os.environ.get("FACTOR_DASHBOARD_TABLE_CACHE_MAX_ITEMS", "24"))
        self.factor_cache_max_items = int(os.environ.get("FACTOR_DASHBOARD_FACTOR_CACHE_MAX_ITEMS", "48"))
        self._scan_cache_expires_at = 0.0
        self._scan_cache: list[AnalysisRun] | None = None
        self._table_cache: OrderedDict[tuple[Any, ...], pd.DataFrame] = OrderedDict()
        self._factor_frame_cache: OrderedDict[tuple[Any, ...], pd.DataFrame] = OrderedDict()

    def clear_cache(self) -> None:
        self._scan_cache_expires_at = 0.0
        self._scan_cache = None
        self._table_cache.clear()
        self._factor_frame_cache.clear()

    def health(self) -> dict[str, Any]:
        universes = self.list_universes()
        return {
            "status": "ok",
            "store_root": str(self.store_root.as_posix()),
            "universe_count": len(universes),
            "run_count": int(sum(int(row.get("run_count", 0)) for row in universes)),
        }

    def overview(self, *, stale_after_days: int = 7) -> dict[str, Any]:
        universes = self.list_universes()
        runs = self._scan_runs()
        catalog, catalog_meta = self._load_field_catalog()
        max_available_end = _max_text(catalog.get("available_end")) if not catalog.empty else None
        warnings: list[dict[str, Any]] = []
        if catalog_meta.get("status") != "ok":
            warnings.append(
                {
                    "code": "field_catalog_missing",
                    "severity": "warning",
                    "message": "Field catalog is not available.",
                }
            )
        elif (
            max_available_end
            and _days_since(max_available_end) is not None
            and _days_since(max_available_end) > int(stale_after_days)
        ):
            warnings.append(
                {
                    "code": "field_catalog_stale",
                    "severity": "warning",
                    "message": f"Field catalog max available date is {max_available_end}.",
                    "days_since": _days_since(max_available_end),
                }
            )
        live_status_by_universe: dict[str, Any] = {}
        for row in universes:
            name = str(row.get("name") or "")
            live = self.get_live_status(universe=name)
            live_status_by_universe[name] = {
                "status": live.get("status"),
                "active_total": live.get("active_total", 0),
                "message": live.get("message"),
            }
            if live.get("status") == "missing":
                warnings.append(
                    {
                        "code": "live_missing",
                        "severity": "info",
                        "universe": name,
                        "message": "Live latest artifact is not available.",
                    }
                )
            if int(row.get("run_count", 0) or 0) <= 0:
                warnings.append(
                    {
                        "code": "no_analysis_runs",
                        "severity": "warning",
                        "universe": name,
                        "message": "No analysis runs are available.",
                    }
                )
        return {
            "status": "ok",
            "store_root": str(self.store_root.as_posix()),
            "universes": universes,
            "universe_count": len(universes),
            "run_count": int(sum(int(row.get("run_count", 0) or 0) for row in universes)),
            "latest_analysis_at_utc": max((r.created_at_utc for r in runs), default=""),
            "field_catalog_status": catalog_meta.get("status"),
            "field_catalog_max_available_end": max_available_end,
            "field_catalog_row_count": int(catalog_meta.get("row_count", 0) or 0),
            "live_status_by_universe": live_status_by_universe,
            "freshness_warnings": warnings,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    def list_data_families(self) -> dict[str, Any]:
        catalog, meta = self._load_field_catalog()
        if catalog.empty:
            return {**meta, "families": []}
        work = catalog.copy()
        work["factor_family"] = work.get("factor_family", "other").fillna("").astype(str).str.strip()
        work.loc[work["factor_family"] == "", "factor_family"] = "other"
        if "coverage_rate" in work.columns:
            work["coverage_rate"] = pd.to_numeric(work["coverage_rate"], errors="coerce")
        rows = []
        for family, group in work.groupby("factor_family", sort=True):
            source_tables = sorted(
                {str(x) for x in group.get("source_table", pd.Series(dtype=str)).dropna().astype(str) if str(x)}
            )
            rows.append(
                {
                    "family": str(family),
                    "field_count": int(len(group)),
                    "searchable_count": int(
                        _bool_series(group.get("is_searchable", pd.Series(False, index=group.index))).sum()
                    ),
                    "enabled_count": int(
                        _bool_series(
                            group.get(
                                "is_default_enabled",
                                pd.Series(False, index=group.index),
                            )
                        ).sum()
                    ),
                    "avg_coverage_rate": _nullable_float(group["coverage_rate"].mean())
                    if "coverage_rate" in group.columns
                    else None,
                    "min_available_start": _min_text(group.get("available_start")),
                    "max_available_end": _max_text(group.get("available_end")),
                    "source_tables": source_tables,
                }
            )
        return {**meta, "families": rows}

    def list_data_fields(
        self,
        *,
        family: str = "",
        q: str = "",
        searchable_only: bool = False,
        limit: int = 200,
        offset: int = 0,
    ) -> dict[str, Any]:
        catalog, meta = self._load_field_catalog()
        if catalog.empty:
            return {**meta, "total": 0, "fields": []}
        work = catalog.copy()
        work["factor_family"] = work.get("factor_family", "other").fillna("").astype(str).str.strip()
        work.loc[work["factor_family"] == "", "factor_family"] = "other"
        family_text = str(family or "").strip()
        if family_text:
            work = work[work["factor_family"].astype(str) == family_text].copy()
        query = str(q or "").strip().lower()
        if query:
            search_cols = [
                col
                for col in [
                    "field_name",
                    "description",
                    "source_table",
                    "category",
                    "factor_family",
                ]
                if col in work.columns
            ]
            haystack = pd.Series("", index=work.index)
            for col in search_cols:
                haystack = haystack + " " + work[col].fillna("").astype(str).str.lower()
            work = work[haystack.str.contains(query, regex=False)].copy()
        if searchable_only and "is_searchable" in work.columns:
            work = work[_bool_series(work["is_searchable"])].copy()

        if "field_name" in work.columns:
            work = work.sort_values(["factor_family", "field_name"], kind="mergesort")
        total = int(len(work))
        start = max(0, int(offset))
        size = max(1, min(int(limit), 5000))
        page = work.iloc[start : start + size].copy()
        return {**meta, "total": total, "fields": _records(page)}

    def list_factor_library(self, *, universe: str = "") -> dict[str, Any]:
        universe_name = str(universe or "").strip()
        roots: list[Path] = []
        if universe_name:
            roots.append(self.store_root / universe_name / "library" / "factor_library_registry.csv")
        else:
            for item in sorted(self.store_root.iterdir()) if self.store_root.exists() else []:
                candidate = item / "library" / "factor_library_registry.csv"
                if candidate.exists():
                    roots.append(candidate)
        frames: list[pd.DataFrame] = []
        for path in roots:
            frame = _read_table(path)
            if frame.empty:
                continue
            frame = frame.copy()
            frame["universe"] = path.parents[1].name if len(path.parents) > 1 else universe_name
            frames.append(frame)
        if not frames:
            return {
                "status": "missing",
                "total": 0,
                "factors": [],
                "message": "factor library registry is not available.",
            }
        work = pd.concat(frames, ignore_index=True)
        if "score" in work.columns:
            work["score"] = pd.to_numeric(work["score"], errors="coerce")
        work = work.sort_values(
            ["status", "score", "factor"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        cfg = FactorLibraryConfig(enabled=True)
        if not work.empty:
            marked = [_mark_legacy_library_row(row, cfg) for row in _records(work)]
            return {"status": "ok", "total": int(len(marked)), "factors": marked}
        return {"status": "ok", "total": int(len(work)), "factors": _records(work)}

    def get_factor_library_status(self, *, universe: str, run_id: str, factor: str) -> dict[str, Any]:
        _ = self._resolve_detail_run(universe=universe, run_id=run_id, factor=factor)
        registry_path = self.store_root / str(universe) / "library" / "factor_library_registry.csv"
        registry = _read_table(registry_path)
        if registry.empty or "factor" not in registry.columns:
            return {
                "status": "ok",
                "factor": str(factor),
                "library_status": "none",
                "registry_row": None,
                "can_check": True,
                "can_submit": False,
            }
        rows = registry[registry["factor"].astype(str) == str(factor)].copy()
        if rows.empty:
            return {
                "status": "ok",
                "factor": str(factor),
                "library_status": "none",
                "registry_row": None,
                "can_check": True,
                "can_submit": False,
            }
        if "submitted_at_utc" in rows.columns:
            rows["_submitted_sort"] = pd.to_datetime(rows["submitted_at_utc"], errors="coerce")
            rows = rows.sort_values("_submitted_sort", ascending=False, na_position="last")
        row = rows.iloc[0].drop(labels=[c for c in ["_submitted_sort"] if c in rows.columns]).to_dict()
        cfg = FactorLibraryConfig(enabled=True)
        row = _mark_legacy_library_row(row, cfg)
        return {
            "status": "ok",
            "factor": str(factor),
            "library_status": str(row.get("library_status_effective") or row.get("status") or "none"),
            "registry_row": row,
            "can_check": True,
            "can_submit": False,
        }

    def check_factor_library_candidate(self, *, universe: str, run_id: str, factor: str) -> dict[str, Any]:
        run, metrics, ic_df, pnl_df, signal_df = self._library_inputs(universe=universe, run_id=run_id, factor=factor)
        cfg = FactorLibraryConfig(enabled=True)
        payload = workflow_check_factor_library_candidate(
            base_dir=self.store_root,
            universe_name=universe,
            run_id=run.run_id,
            factor=factor,
            factor_metrics_df=metrics,
            ic_df=ic_df,
            portfolio_pnl_df=pnl_df,
            signal_df=signal_df,
            config=cfg,
        )
        payload["thresholds"] = _factor_library_thresholds(cfg)
        return payload

    def submit_factor_library_candidate(
        self,
        *,
        universe: str,
        run_id: str,
        factor: str,
        submitted_by: str = "dashboard",
    ) -> dict[str, Any]:
        run, metrics, ic_df, pnl_df, signal_df = self._library_inputs(universe=universe, run_id=run_id, factor=factor)
        cfg = FactorLibraryConfig(enabled=True)
        result = workflow_submit_factor_library_candidate(
            base_dir=self.store_root,
            universe_name=universe,
            run_id=run.run_id,
            factor=factor,
            factor_metrics_df=metrics,
            ic_df=ic_df,
            portfolio_pnl_df=pnl_df,
            signal_df=signal_df,
            config=cfg,
            submitted_by=submitted_by,
        )
        if isinstance(result.get("check"), dict):
            result["check"]["thresholds"] = _factor_library_thresholds(cfg)
        if result.get("submitted"):
            self.clear_cache()
        return result

    def list_superalpha_components(self, *, universe: str) -> dict[str, Any]:
        rows = workflow_list_superalpha_components(base_dir=self.store_root, universe_name=str(universe))
        return {
            "status": "ok",
            "universe": str(universe),
            "total": int(len(rows)),
            "components": rows,
        }

    def list_superalpha_runs(self, *, universe: str) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        root = self.store_root / str(universe) / "superalphas"
        for meta_path in sorted(root.glob("*/meta.json")) if root.exists() else []:
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            rows.append(self._superalpha_meta_to_dict(payload, meta_path=meta_path))
        rows = sorted(rows, key=lambda row: str(row.get("created_at_utc", "")), reverse=True)
        return {
            "status": "ok",
            "universe": str(universe),
            "total": int(len(rows)),
            "runs": rows,
        }

    def run_superalpha_backtest(self, body: dict[str, Any]) -> dict[str, Any]:
        universe = str(body.get("universe") or "").strip()
        factor_ids = body.get("factor_ids") or body.get("factors") or []
        if not universe:
            raise SuperalphaError("universe is required")
        if not isinstance(factor_ids, list) or not [str(x).strip() for x in factor_ids if str(x).strip()]:
            raise SuperalphaError("factor_ids must contain at least one factor")

        component_join = str(body.get("component_join") or "concat").strip().lower()
        if component_join not in {"concat", "inner"}:
            raise SuperalphaError("component_join must be one of: concat, inner")
        try:
            max_components = int(body.get("max_components") or 20)
        except Exception as exc:
            raise SuperalphaError("max_components must be an integer between 1 and 50") from exc
        if max_components < 1 or max_components > 50:
            raise SuperalphaError("max_components must be between 1 and 50")

        duckdb_threads = str(body.get("duckdb_threads") or "").strip()
        if duckdb_threads:
            try:
                parsed_threads = int(duckdb_threads)
            except Exception as exc:
                raise SuperalphaError("duckdb_threads must be a positive integer") from exc
            if parsed_threads <= 0:
                raise SuperalphaError("duckdb_threads must be a positive integer")

        duckdb_memory_limit = str(body.get("duckdb_memory_limit") or "2GB").strip()
        duckdb_max_temp = str(body.get("duckdb_max_temp_directory_size") or "50GB").strip()
        for label, value in {
            "duckdb_memory_limit": duckdb_memory_limit,
            "duckdb_max_temp_directory_size": duckdb_max_temp,
        }.items():
            if not re.match(r"^[1-9][0-9]*(MB|GB|TB)$", value, flags=re.IGNORECASE):
                raise SuperalphaError(f"{label} must look like 512MB, 2GB, or 1TB")

        # Parse safety parameters for SA runtime.
        cfg = SuperalphaConfig(
            component_join=component_join,
            allow_reproduce_fallback=bool(body.get("allow_reproduce_fallback", True)),
            max_components=max_components,
            duckdb_memory_limit=duckdb_memory_limit,
            duckdb_max_temp_directory_size=duckdb_max_temp,
            duckdb_threads=duckdb_threads,
        )

        result = workflow_run_superalpha_backtest(
            base_dir=self.store_root,
            universe_name=universe,
            selected_factor_ids=[str(x).strip() for x in factor_ids if str(x).strip()],
            combo_expression=str(body.get("combo_expression") or "1"),
            name=str(body.get("name") or ""),
            rerun=bool(body.get("rerun", False)),
            config=cfg,
        )
        self.clear_cache()
        return result

    def get_superalpha(self, *, superalpha_id: str) -> dict[str, Any]:
        run, payload = self._require_superalpha_run(superalpha_id)
        return {
            "status": "ok",
            "superalpha_id": run.run_id,
            "universe": run.universe,
            "meta": payload,
            "summary": dict(payload.get("summary") or {}),
        }

    def rename_superalpha(self, *, superalpha_id: str, name: str) -> dict[str, Any]:
        run, payload = self._require_superalpha_run(superalpha_id)
        display_name = self._validate_superalpha_display_name(name)
        touched = 0
        for meta_path in [
            run.analysis_dir / "meta.json",
            run.analysis_dir / "analysis_meta.json",
        ]:
            if not meta_path.exists() or not meta_path.is_file():
                continue
            try:
                current = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                current = dict(payload)
            current["name"] = display_name
            meta_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
            touched += 1
        if touched <= 0:
            payload["name"] = display_name
        self.clear_cache()
        updated = dict(payload)
        updated["name"] = display_name
        return {
            "status": "ok",
            "superalpha_id": run.run_id,
            "run": self._superalpha_meta_to_dict(updated, meta_path=run.analysis_dir / "meta.json"),
        }

    def get_superalpha_detail(self, *, superalpha_id: str, include_test: bool = False) -> dict[str, Any]:
        run, payload = self._require_superalpha_run(superalpha_id)
        factor = SUPERALPHA_FACTOR
        factor_row = self._factor_metric_row(run.metrics_path, factor)
        if not factor_row:
            factor_row = {
                "factor": factor,
                "period": run.period,
                "layers": run.layers,
                "expression": payload.get("combo_expression", ""),
            }
        pnl = self._get_factor_pnl_for_run(run, factor, include_test=include_test)
        analysis_data = self._get_factor_analysis_data_for_run(run, factor, include_test=include_test)
        return {
            "status": "ok",
            "superalpha_id": run.run_id,
            "universe": run.universe,
            "meta": payload,
            "run": self._run_to_dict(run),
            "factor": factor_row,
            "pnl": pnl,
            "analysis_data": analysis_data,
            "metrics": factor_row,
        }

    def get_live_status(self, *, universe: str) -> dict[str, Any]:
        paths = live_paths(self.store_root, str(universe))
        latest = read_json(paths.live_root / "latest.json", None)
        if not isinstance(latest, dict):
            active = live_list_superalphas(
                base_dir=self.store_root,
                universe=str(universe),
                include_paused=True,
                include_retired=True,
            )
            return {
                "status": "missing",
                "universe": str(universe),
                "superalphas": [],
                "active_total": len(active),
                "message": "live latest is not available.",
            }
        out = dict(latest)
        out.setdefault("universe", str(universe))
        out.setdefault("superalphas", [])
        return out

    def list_live_superalphas(
        self,
        *,
        universe: str,
        include_paused: bool = True,
        include_retired: bool = False,
    ) -> dict[str, Any]:
        rows = live_list_superalphas(
            base_dir=self.store_root,
            universe=str(universe),
            include_paused=include_paused,
            include_retired=include_retired,
        )
        return {
            "status": "ok",
            "universe": str(universe),
            "total": len(rows),
            "superalphas": rows,
        }

    def activate_live_superalpha(self, body: dict[str, Any]) -> dict[str, Any]:
        universe = str(body.get("universe") or "").strip()
        superalpha_id = str(body.get("superalpha_id") or "").strip()
        if not universe or not superalpha_id:
            raise ValueError("universe and superalpha_id are required")
        result = live_activate_superalpha(
            base_dir=self.store_root,
            universe=universe,
            superalpha_id=superalpha_id,
            activated_by=str(body.get("activated_by") or "dashboard"),
            max_active=int(
                load_live_config(overrides={"store_root": self.store_root, "universe": universe}).superalpha.max_active
            ),
        )
        self.clear_cache()
        return {"status": "ok", **result}

    def update_live_superalpha_status(self, *, universe: str, superalpha_id: str, status: str) -> dict[str, Any]:
        return live_update_superalpha_status(
            base_dir=self.store_root,
            universe=universe,
            superalpha_id=superalpha_id,
            status=status,
        )

    def list_live_runs(self, *, universe: str, superalpha_id: str = "") -> dict[str, Any]:
        paths = live_paths(self.store_root, str(universe))
        roots = (
            [paths.jobs_dir(superalpha_id)]
            if superalpha_id
            else sorted((paths.live_root / "jobs").glob("*"))
            if (paths.live_root / "jobs").exists()
            else []
        )
        rows: list[dict[str, Any]] = []
        for root in roots:
            for path in sorted(root.glob("*.json"), reverse=True):
                if path.name == "latest.json":
                    continue
                payload = read_json(path, None)
                if isinstance(payload, dict):
                    rows.append(payload)
        return {
            "status": "ok",
            "universe": str(universe),
            "total": len(rows),
            "runs": rows[:200],
        }

    def get_live_data_status(self, *, universe: str) -> dict[str, Any]:
        paths = live_paths(self.store_root, str(universe))
        payload = read_json(paths.data_status_dir / "latest.json", None)
        return payload if isinstance(payload, dict) else {"status": "missing", "universe": str(universe), "fields": []}

    def get_live_holdings(self, *, universe: str, superalpha_id: str, limit: int = 200) -> dict[str, Any]:
        paths = live_paths(self.store_root, str(universe))
        latest = read_json(paths.holdings_dir(superalpha_id) / "latest.json", None)
        if not isinstance(latest, dict):
            return {
                "status": "missing",
                "universe": str(universe),
                "superalpha_id": str(superalpha_id),
                "rows": [],
            }
        artifact = self._resolve_artifact_path(str(latest.get("artifact_path") or latest.get("holdings_path") or ""))
        frame = _read_table(artifact) if artifact.exists() else pd.DataFrame()
        rows = _records(frame.head(max(1, min(int(limit), 5000)))) if not frame.empty else []
        return {
            "status": "ok" if rows else "empty",
            "universe": str(universe),
            "superalpha_id": str(superalpha_id),
            "latest": latest,
            "rows": rows,
        }

    def get_live_orders(self, *, universe: str, superalpha_id: str, limit: int = 500) -> dict[str, Any]:
        paths = live_paths(self.store_root, str(universe))
        latest = read_json(paths.live_root / "orders" / str(superalpha_id) / "latest.json", None)
        if not isinstance(latest, dict):
            return {
                "status": "missing",
                "universe": str(universe),
                "superalpha_id": str(superalpha_id),
                "latest": None,
                "account": {},
                "summary": {},
                "rows": [],
            }
        artifact = self._resolve_artifact_path(
            str(latest.get("orders_csv_path") or latest.get("orders_path") or latest.get("artifact_path") or "")
        )
        frame = _read_table(artifact) if artifact.exists() else pd.DataFrame()
        rows = _records(frame.head(max(1, min(int(limit), 5000)))) if not frame.empty else []
        return {
            "status": "ok" if rows else "empty",
            "universe": str(universe),
            "superalpha_id": str(superalpha_id),
            "latest": latest,
            "account": latest.get("account") or {},
            "summary": latest.get("summary") or {},
            "rows": rows,
        }

    def run_live_preflight(self, body: dict[str, Any]) -> dict[str, Any]:
        # Long production runs belong to scripts/run_live_superalpha.py. The API only exposes a controlled dry-run acknowledgement.
        universe = str(body.get("universe") or "").strip()
        superalpha_id = str(body.get("superalpha_id") or "").strip()
        if not universe or not superalpha_id:
            raise ValueError("universe and superalpha_id are required")
        if not bool(body.get("dry_run", True)):
            raise ValueError(
                "dashboard live run only supports dry_run=true; use scripts/run_live_superalpha.py for production"
            )
        return {
            "status": "accepted",
            "mode": "dry_run",
            "universe": universe,
            "superalpha_id": superalpha_id,
        }

    def _library_inputs(
        self, *, universe: str, run_id: str, factor: str
    ) -> tuple[AnalysisRun, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        run = self._resolve_detail_run(universe=universe, run_id=run_id, factor=factor)
        metrics = self._read_table_cached(run.metrics_path)
        ic_df = self._read_table_cached(run.ic_path)
        pnl_df = self._factor_pnl_frame(run, factor, include_benchmark=False)
        signal_df = pd.DataFrame()
        try:
            signal_df = load_universe_alpha_values(alpha_name=factor, base_dir=self.store_root, universe_name=universe)
        except Exception:
            signal_df = pd.DataFrame()
        return run, metrics, ic_df, pnl_df, signal_df

    def _load_field_catalog(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        duckdb_path = Path(os.environ.get("FACTOR_DASHBOARD_DUCKDB_PATH", "data/duckdb/market.duckdb"))
        candidates = [self.store_root / "data" / "lake" / "meta" / "field_catalog.parquet"]
        if self._uses_default_store_root():
            candidates.append(Path.cwd() / "data" / "lake" / "meta" / "field_catalog.parquet")
        source = ""
        catalog = pd.DataFrame()
        for path in candidates:
            if path.exists():
                catalog = _read_table(path)
                source = str(path.as_posix())
                break
        if catalog.empty:
            meta = {
                "status": "missing",
                "message": "field catalog is not available.",
                "source": source,
                "duckdb_path": str(duckdb_path.as_posix()),
                "row_count": 0,
            }
            return catalog, meta

        coverage_candidates = [self.store_root / "artifacts" / "data_quality" / "field_coverage.csv"]
        if self._uses_default_store_root():
            coverage_candidates.append(Path.cwd() / "artifacts" / "data_quality" / "field_coverage.csv")
        for coverage_path in coverage_candidates:
            if not coverage_path.exists():
                continue
            coverage = _read_table(coverage_path)
            if coverage.empty or "field_name" not in coverage.columns:
                continue
            missing_cols = [
                col
                for col in [
                    "coverage_scope",
                    "coverage_start_date",
                    "coverage_end_date",
                    "coverage_row_count",
                    "non_null_count",
                    "coverage_rate",
                    "missing_rate",
                    "finite_count",
                    "finite_rate",
                    "coverage_status",
                    "coverage_updated_at_utc",
                ]
                if col in coverage.columns and col not in catalog.columns
            ]
            if missing_cols and "field_name" in catalog.columns:
                catalog = pd.merge(
                    catalog,
                    coverage[["field_name", *missing_cols]],
                    on="field_name",
                    how="left",
                )
            break

        catalog = _normalise_field_catalog(catalog)
        meta = {
            "status": "ok",
            "message": None,
            "source": source,
            "duckdb_path": str(duckdb_path.as_posix()),
            "row_count": int(len(catalog)),
            "metadata_note": "field_role, available_at, preprocessing_policy and leakage_safe are rule-inferred catalog hints, not full data-lineage validation.",
        }
        return catalog, meta

    def _uses_default_store_root(self) -> bool:
        default_root = Path("data/alpha_universe_store")
        try:
            return self.store_root.resolve() == (Path.cwd() / default_root).resolve()
        except Exception:
            return self.store_root.as_posix().replace("\\", "/").rstrip("/") == default_root.as_posix()

    def list_universes(self) -> list[dict[str, Any]]:
        grouped: dict[str, list[AnalysisRun]] = {}
        for run in self._scan_runs():
            grouped.setdefault(run.universe, []).append(run)
        if self.store_root.exists():
            for universe_dir in self.store_root.iterdir():
                if universe_dir.is_dir() and self._scoreboard_path(universe_dir.name).exists():
                    grouped.setdefault(universe_dir.name, [])
        rows = []
        for universe, runs in sorted(grouped.items()):
            run_count = len(runs) + (1 if self._scoreboard_path(universe).exists() else 0)
            rows.append(
                {
                    "name": universe,
                    "run_count": run_count,
                    "latest_created_at_utc": max((r.created_at_utc for r in runs), default=""),
                }
            )
        return rows

    def list_runs(self, universe: str) -> list[dict[str, Any]]:
        runs = [run for run in self._scan_runs() if run.universe == str(universe)]
        rows = [self._run_to_dict(run) for run in runs]
        rows = sorted(rows, key=lambda row: str(row.get("created_at_utc", "")), reverse=True)
        scoreboard = self._scoreboard_run_to_dict(str(universe), rows)
        if scoreboard is not None:
            return [scoreboard, *rows]
        return rows

    def compare_runs(self, *, universe: str, left_run_id: str, right_run_id: str, top_n: int = 50) -> dict[str, Any]:
        if str(left_run_id) == SCOREBOARD_RUN_ID or str(right_run_id) == SCOREBOARD_RUN_ID:
            raise ValueError("Run compare only supports real analysis runs, not the scoreboard rollup.")
        left_run = self._require_run(universe=universe, run_id=left_run_id)
        right_run = self._require_run(universe=universe, run_id=right_run_id)
        left = self._compare_metrics_frame(left_run)
        right = self._compare_metrics_frame(right_run)
        left_status = compare_artifact_status(left_run, left)
        right_status = compare_artifact_status(right_run, right)
        warnings: list[str] = []
        for label, frame, status in [
            ("left", left, left_status),
            ("right", right, right_status),
        ]:
            display_label = label.title()
            if status == "missing_metrics":
                warnings.append(f"{display_label} run metrics artifact is missing; comparison metrics are unavailable.")
                continue
            if status == "invalid_metrics":
                warnings.append(
                    f"{display_label} run metrics artifact is invalid; treat this run as not comparable until the artifact is regenerated."
                )
                continue
            if status == "partial_metrics":
                warnings.append(
                    f"{display_label} run metrics are partial; missing columns are shown as Missing artifact."
                )
            missing = [
                col
                for col in [
                    "factor",
                    "feedback_score",
                    "ic_mean",
                    "long_short_sharpe_ratio",
                ]
                if col not in frame.columns
            ]
            if missing:
                warnings.append(f"{display_label} run is missing columns: {', '.join(missing)}")
        n = max(1, min(int(top_n), 500))
        return {
            "status": "ok",
            "universe": str(universe),
            "top_n": n,
            "left": self._run_to_dict(left_run),
            "right": self._run_to_dict(right_run),
            "left_artifact_status": left_status,
            "right_artifact_status": right_status,
            "metrics": compare_metric_rows(left, right),
            "overlap": top_overlap(left, right, top_n=n),
            "warnings": warnings,
        }

    def _compare_metrics_frame(self, run: AnalysisRun) -> pd.DataFrame:
        path = run.metrics_path
        if path is None or not path.exists():
            return pd.DataFrame()
        frame = self._read_table_cached(path)
        if frame.empty:
            return pd.DataFrame()
        return self._ensure_feedback_score_column(frame.copy())

    def data_health(self, *, universe: str) -> dict[str, Any]:
        catalog, meta = self._load_field_catalog()
        catalog_work = catalog.copy() if not catalog.empty else pd.DataFrame()
        if not catalog_work.empty and "factor_family" not in catalog_work.columns:
            catalog_work["factor_family"] = "other"
        warnings: list[dict[str, Any]] = []
        max_available_end = _max_text(catalog_work.get("available_end")) if not catalog_work.empty else None
        stale_days = _days_since(max_available_end)
        if meta.get("status") != "ok":
            warnings.append(
                {
                    "code": "field_catalog_missing",
                    "severity": "warning",
                    "message": "Field catalog is not available.",
                }
            )
        elif stale_days is not None and stale_days > FIELD_CATALOG_STALE_AFTER_DAYS:
            warnings.append(
                {
                    "code": "field_catalog_stale",
                    "severity": "warning",
                    "message": f"Field catalog max available date is {max_available_end}.",
                    "days_since": stale_days,
                }
            )
        coverage = coverage_counts(catalog_work)
        if coverage["coverage_status"] == "missing" and not catalog_work.empty:
            warnings.append(
                {
                    "code": "coverage_not_refreshed",
                    "severity": "info",
                    "message": "Field coverage metrics are not refreshed; run the existing field coverage or panel quality workflow when needed.",
                }
            )
        elif coverage["coverage_status"] == "partial":
            warnings.append(
                {
                    "code": "coverage_partial",
                    "severity": "info",
                    "message": "Some field coverage metrics are missing; coverage summaries are partial.",
                }
            )
        catalog_summary = {
            "status": meta.get("status"),
            "row_count": int(meta.get("row_count", 0) or 0),
            "searchable_count": int(
                _bool_series(catalog_work.get("is_searchable", pd.Series(False, index=catalog_work.index))).sum()
            )
            if not catalog_work.empty
            else 0,
            "avg_coverage_rate": _nullable_float(
                pd.to_numeric(
                    catalog_work.get("coverage_rate", pd.Series(dtype=float)),
                    errors="coerce",
                ).mean()
            )
            if "coverage_rate" in catalog_work.columns
            else None,
            "low_coverage_count": low_coverage_count(catalog_work),
            **coverage,
            "max_available_end": max_available_end,
            "stale_days": stale_days,
            "source": meta.get("source", ""),
        }
        base_summary = base_frame_summary(self.store_root / str(universe) / "base" / "base_frame.parquet")
        if not base_summary["exists"]:
            warnings.append(
                {
                    "code": "base_frame_missing",
                    "severity": "warning",
                    "message": "Universe base frame is not available.",
                }
            )
        run_health = run_health_summary(self.store_root / str(universe) / "feedback" / "run_health.jsonl")
        if not run_health.get("exists"):
            warnings.append(
                {
                    "code": "run_health_missing",
                    "severity": "info",
                    "message": "No closed-loop run health artifact is available for this universe.",
                }
            )
        quality = quality_artifact_summary(Path("artifacts") / "data_quality" / "panel_quality.json")
        if quality.get("exists") and quality.get("overall_status") in {"warn", "fail"}:
            warnings.append(
                {
                    "code": "quality_artifact_warning",
                    "severity": str(quality.get("overall_status")),
                    "message": "Latest panel quality artifact reports warnings or failures.",
                }
            )
        return {
            "status": "ok",
            "universe": str(universe),
            "catalog": catalog_summary,
            "families": data_health_families(catalog_work),
            "universe_base": base_summary,
            "closed_loop_health": run_health,
            "quality_artifact": quality,
            "warnings": warnings,
            "thresholds": {
                "field_catalog_stale_after_days": FIELD_CATALOG_STALE_AFTER_DAYS,
                "low_coverage_threshold": LOW_COVERAGE_THRESHOLD,
            },
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    def get_factors(
        self,
        universe: str,
        run_id: str,
        q: str = "",
        sort_by: str = "feedback_score",
        sort_dir: str = "desc",
        effective_only: bool = False,
        limit: int = 500,
        offset: int = 0,
    ) -> dict[str, Any]:
        if str(run_id) == SCOREBOARD_RUN_ID:
            run = None
            df = self._scoreboard_for_universe(universe)
        else:
            run = self._require_run(universe=universe, run_id=run_id)
            metrics_path = run.metrics_path
            if metrics_path is None or not metrics_path.exists():
                return {"total": 0, "factors": [], "status": "missing_metrics"}
            df = self._read_table_cached(metrics_path)
        if "factor" not in df.columns:
            return {"total": 0, "factors": [], "status": "invalid_metrics"}

        work = self._ensure_feedback_score_column(df)
        if "period" not in work.columns:
            work["period"] = int(run.period if run is not None else 0)
        if "layers" not in work.columns:
            work["layers"] = int(run.layers if run is not None else 0)
        if "expression" not in work.columns:
            work["expression"] = ""
        query = str(q or "").strip().lower()
        if query:
            haystack = work.get("factor", pd.Series("", index=work.index)).astype(str).str.lower()
            if "expression" in work.columns:
                haystack = haystack + " " + work["expression"].astype(str).str.lower()
            work = work[haystack.str.contains(query, regex=False)].copy()

        if effective_only:
            work = self._filter_effective(work)

        sort_col = str(sort_by or "").strip()
        if sort_col == "feedback_score" and "feedback_score" not in work.columns:
            work = self._ensure_feedback_score_column(work)
        if sort_col in work.columns:
            ascending = str(sort_dir or "desc").lower() == "asc"
            sort_values = pd.to_numeric(work[sort_col], errors="coerce")
            if sort_values.notna().any():
                work = work.assign(_sort_value=sort_values).sort_values(
                    ["_sort_value", "factor"],
                    ascending=[ascending, True],
                    na_position="last",
                    kind="mergesort",
                )
                work = work.drop(columns=["_sort_value"])
            else:
                work = work.sort_values(sort_col, ascending=ascending, kind="mergesort")
        elif "factor" in work.columns:
            work = work.sort_values("factor", kind="mergesort")

        total = int(len(work))
        start = max(0, int(offset))
        size = max(1, min(int(limit), 5000))
        page = work.iloc[start : start + size].copy()
        return {"total": total, "factors": _records(page), "status": "ok"}

    def get_factor_pnl(self, universe: str, run_id: str, factor: str, include_test: bool = False) -> dict[str, Any]:
        run = self._resolve_detail_run(universe=universe, run_id=run_id, factor=factor)
        return self._get_factor_pnl_for_run(run, factor, include_test=include_test)

    def _get_factor_pnl_for_run(self, run: AnalysisRun, factor: str, include_test: bool = False) -> dict[str, Any]:
        phase_config = self._phase_config_for_run(run)
        phase_metrics = self._phase_metrics_for_factor(run, factor)
        pnl_path = run.pnl_path
        if pnl_path is None or not pnl_path.exists():
            portfolio_metrics = self._factor_portfolio_metrics(run, factor, phase_config=phase_config)
            return {
                "status": "missing",
                "factor": str(factor),
                "rows": [],
                "message": "portfolio_pnl_df artifact is not available for this analysis run.",
                "phase_config": phase_config,
                "phase_metrics": phase_metrics,
                "portfolio_metrics": portfolio_metrics,
                "benchmark_status": dict(run.extra_meta.get("benchmark_status") or {}),
            }

        work = self._factor_pnl_frame(run, factor, include_benchmark=True)
        portfolio_metrics = self._factor_portfolio_metrics_from_frame(run, factor, work, phase_config=phase_config)
        if not work.empty and "factor" not in work.columns:
            return {
                "status": "invalid",
                "factor": str(factor),
                "rows": [],
                "phase_config": phase_config,
                "phase_metrics": phase_metrics,
                "portfolio_metrics": portfolio_metrics,
                "benchmark_status": dict(run.extra_meta.get("benchmark_status") or {}),
            }
        if work.empty:
            return {
                "status": "empty",
                "factor": str(factor),
                "rows": [],
                "phase_config": phase_config,
                "phase_metrics": phase_metrics,
                "portfolio_metrics": portfolio_metrics,
                "benchmark_status": dict(run.extra_meta.get("benchmark_status") or {}),
            }
        if "trade_date" in work.columns:
            work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
            work = work.sort_values(["portfolio", "trade_date"], kind="mergesort")
            if phase_config:
                work["phase"] = _assign_phase_from_config(work["trade_date"], phase_config)
                if not include_test:
                    work = work[work["phase"].astype(str) != "test"].copy()
            work["trade_date"] = work["trade_date"].dt.strftime("%Y-%m-%d")
        filtered_phase_metrics = _filter_phase_metrics(phase_metrics, include_test=include_test)
        return {
            "status": "ok",
            "factor": str(factor),
            "rows": _records(work),
            "phase_config": phase_config,
            "phase_metrics": filtered_phase_metrics,
            "portfolio_metrics": portfolio_metrics,
            "benchmark_status": dict(run.extra_meta.get("benchmark_status") or {}),
        }

    def get_factor_analysis_data(
        self, universe: str, run_id: str, factor: str, include_test: bool = False
    ) -> dict[str, Any]:
        run = self._resolve_detail_run(universe=universe, run_id=run_id, factor=factor)
        return self._get_factor_analysis_data_for_run(run, factor, include_test=include_test)

    def _get_factor_analysis_data_for_run(
        self, run: AnalysisRun, factor: str, include_test: bool = False
    ) -> dict[str, Any]:
        phase_config = self._phase_config_for_run(run)
        phase_metrics = _filter_phase_metrics(self._phase_metrics_for_factor(run, factor), include_test=include_test)
        ic_series = self._factor_ic_series(run, factor, phase_config=phase_config, include_test=include_test)
        yearly_ic = self._yearly_ic_summary(ic_series)
        monthly_ic = self._monthly_ic_summary(ic_series)
        coverage_series = self._factor_coverage_series(
            run,
            factor,
            phase_config=phase_config,
            include_test=include_test,
        )
        distribution = self._factor_distribution_histogram(run, factor, include_test=include_test)
        ic_distribution = self._ic_distribution_histogram(ic_series, factor=factor)
        ic_decay = self._factor_ic_decay(run, factor, include_test=include_test)
        pnl_frame = self._factor_pnl_frame(run, factor, include_benchmark=False)
        layer_terminal = self._factor_layer_terminal_returns_from_frame(
            run,
            factor,
            pnl_frame,
            phase_config=phase_config,
            include_test=include_test,
        )
        has_data = any(
            [
                ic_series,
                yearly_ic,
                monthly_ic,
                coverage_series,
                distribution,
                ic_distribution,
                ic_decay,
                layer_terminal,
            ]
        )
        return {
            "status": "ok" if has_data else "missing",
            "factor": str(factor),
            "phase_config": _filter_phase_config(phase_config, include_test=include_test),
            "phase_metrics": phase_metrics,
            "ic_series": ic_series,
            "yearly_ic": yearly_ic,
            "monthly_ic": monthly_ic,
            "coverage_series": coverage_series,
            "distribution": distribution,
            "ic_distribution": ic_distribution,
            "ic_decay": ic_decay,
            "layer_terminal_return": layer_terminal,
            "message": None if has_data else "No dynamic analysis data artifacts are available for this factor.",
        }

    def get_factor_visualizations(self, universe: str, run_id: str, factor: str) -> dict[str, Any]:
        run = self._resolve_detail_run(universe=universe, run_id=run_id, factor=factor)
        manifest_path = run.visualization_manifest_path
        if manifest_path is None or not manifest_path.exists():
            return {
                "status": "missing",
                "factor": str(factor),
                "images": [],
                "message": "visualization_manifest artifact is not available for this analysis run.",
            }
        manifest = self._load_visualization_manifest(manifest_path)
        required = {"plot_id", "scope", "factor", "category", "title", "relative_path"}
        if not required.issubset(set(manifest.columns)):
            return {
                "status": "invalid",
                "factor": str(factor),
                "images": [],
                "message": "visualization_manifest is invalid.",
            }
        work = manifest[
            (manifest["scope"].astype(str) == "factor") & (manifest["factor"].astype(str) == str(factor))
        ].copy()
        if work.empty:
            return {
                "status": "empty",
                "factor": str(factor),
                "images": [],
                "message": None,
            }
        if "sort_order" in work.columns:
            work["_sort_order"] = pd.to_numeric(work["sort_order"], errors="coerce")
            work = work.sort_values(["_sort_order", "plot_id"], na_position="last", kind="mergesort")
        images: list[dict[str, Any]] = []
        for _, row in work.iterrows():
            image_path = self._resolve_manifest_image_path(run=run, row=row.to_dict())
            if image_path is None or not image_path.exists():
                continue
            plot_id = str(row.get("plot_id", "") or "")
            params = urlencode({"universe": str(universe), "run_id": str(run_id)})
            images.append(
                {
                    "plot_id": plot_id,
                    "category": str(row.get("category", "") or ""),
                    "title": str(row.get("title", "") or plot_id),
                    "url": f"/api/factors/{quote(str(factor), safe='')}/visualizations/{quote(plot_id, safe='')}/image?{params}",
                    "width": _nullable_int(row.get("width")),
                    "height": _nullable_int(row.get("height")),
                    "sort_order": _nullable_int(row.get("sort_order")),
                }
            )
        return {
            "status": "ok" if images else "empty",
            "factor": str(factor),
            "images": images,
            "message": None,
        }

    def get_visualization_image(self, universe: str, run_id: str, factor: str, plot_id: str) -> Path:
        run = self._resolve_detail_run(universe=universe, run_id=run_id, factor=factor)
        manifest_path = run.visualization_manifest_path
        if manifest_path is None or not manifest_path.exists():
            raise FileNotFoundError("visualization_manifest is missing")
        manifest = self._load_visualization_manifest(manifest_path)
        if not {"plot_id", "scope", "factor", "relative_path"}.issubset(set(manifest.columns)):
            raise FileNotFoundError("visualization_manifest is invalid")
        work = manifest[
            (manifest["scope"].astype(str) == "factor")
            & (manifest["factor"].astype(str) == str(factor))
            & (manifest["plot_id"].astype(str) == str(plot_id))
        ].copy()
        if work.empty:
            raise FileNotFoundError(plot_id)
        image_path = self._resolve_manifest_image_path(run=run, row=work.iloc[0].to_dict())
        if image_path is None or not image_path.exists() or not image_path.is_file():
            raise FileNotFoundError(plot_id)
        return image_path

    def _scan_runs(self) -> list[AnalysisRun]:
        now = time.monotonic()
        if self._scan_cache is not None and now < self._scan_cache_expires_at:
            return list(self._scan_cache)
        runs = self._scan_runs_uncached()
        self._scan_cache = list(runs)
        self._scan_cache_expires_at = now + max(0.0, float(self.scan_cache_ttl_seconds))
        return runs

    def _scan_runs_uncached(self) -> list[AnalysisRun]:
        if not self.store_root.exists():
            return []
        meta_paths = sorted(self.store_root.glob("*/analysis/period_*/analysis_*/analysis_meta.json"))
        runs: list[AnalysisRun] = []
        for meta_path in meta_paths:
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
                universe = meta_path.relative_to(self.store_root).parts[0]
                runs.append(self._parse_run_meta(universe=universe, meta_path=meta_path, payload=payload))
            except Exception:
                continue
        return runs

    def _scoreboard_path(self, universe: str) -> Path:
        return self.store_root / str(universe) / "feedback" / "expression_scoreboard.csv"

    def _analysis_registry_path(self, universe: str) -> Path:
        return self.store_root / str(universe) / "analysis" / "analysis_registry.csv"

    def _scoreboard_run_to_dict(self, universe: str, real_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        path = self._scoreboard_path(universe)
        if not path.exists():
            return None
        frame = self._scoreboard_for_universe(universe)
        if frame.empty:
            return None
        created = max((str(row.get("created_at_utc", "")) for row in real_rows), default="")
        return {
            "universe": str(universe),
            "run_id": SCOREBOARD_RUN_ID,
            "label": SCOREBOARD_RUN_LABEL,
            "is_scoreboard": True,
            "period": 0,
            "layers": 0,
            "created_at_utc": created,
            "analysis_dir": "",
            "factor_count": int(len(frame)),
            "has_dashboard_metrics": True,
            "has_factor_metrics": True,
            "has_portfolio_pnl": bool(any(row.get("has_portfolio_pnl") for row in real_rows)),
            "has_benchmark_pnl": bool(any(row.get("has_benchmark_pnl") for row in real_rows)),
            "has_visualizations": False,
            "has_phase_metrics": bool(any(row.get("has_phase_metrics") for row in real_rows)),
            "has_ic_rows": bool(any(row.get("has_ic_rows") for row in real_rows)),
            "has_analysis_data": bool(any(row.get("has_analysis_data") for row in real_rows)),
            "available_phases": [],
            "phase_config": None,
            "benchmark_config": {},
            "benchmark_status": {},
        }

    def _scoreboard_for_universe(self, universe: str) -> pd.DataFrame:
        path = self._scoreboard_path(universe)
        frame = self._read_table_cached(path)
        if frame.empty:
            return pd.DataFrame()
        work = frame.copy()
        if "factor" not in work.columns and "alpha_name" in work.columns:
            work["factor"] = work["alpha_name"].astype(str)
        if "alpha_name" not in work.columns and "factor" in work.columns:
            work["alpha_name"] = work["factor"].astype(str)
        missing_run_id = "analysis_run_id" not in work.columns
        if not missing_run_id:
            run_id_text = work["analysis_run_id"].astype(str).str.strip()
            missing_run_id = bool(work["analysis_run_id"].isna().any() or run_id_text.isin({"", "nan", "None"}).any())
        if missing_run_id:
            registry = self._read_table_cached(self._analysis_registry_path(universe))
            if not registry.empty and {"alpha_name", "analysis_run_id"}.issubset(registry.columns):
                reg = registry[
                    ["alpha_name", "analysis_run_id", "period", "layers"]
                    if {"period", "layers"}.issubset(registry.columns)
                    else ["alpha_name", "analysis_run_id"]
                ].copy()
                reg = reg.drop_duplicates(subset=["alpha_name"], keep="last")
                work = work.merge(reg, on="alpha_name", how="left", suffixes=("", "_registry"))
                if "analysis_run_id_registry" in work.columns:
                    if "analysis_run_id" not in work.columns:
                        work["analysis_run_id"] = work["analysis_run_id_registry"]
                    else:
                        has_value = work["analysis_run_id"].notna() & ~work["analysis_run_id"].astype(
                            str
                        ).str.strip().isin({"", "nan", "None"})
                        work["analysis_run_id"] = work["analysis_run_id"].where(
                            has_value, work["analysis_run_id_registry"]
                        )
                    work = work.drop(columns=["analysis_run_id_registry"])
                for col in ["period", "layers"]:
                    reg_col = f"{col}_registry"
                    if reg_col in work.columns:
                        if col not in work.columns:
                            work[col] = work[reg_col]
                        else:
                            work[col] = work[col].where(work[col].notna(), work[reg_col])
                        work = work.drop(columns=[reg_col])
        return self._ensure_feedback_score_column(work)

    def _resolve_detail_run(self, universe: str, run_id: str, factor: str) -> AnalysisRun:
        if str(run_id) != SCOREBOARD_RUN_ID:
            return self._require_run(universe=universe, run_id=run_id)
        scoreboard = self._scoreboard_for_universe(universe)
        if scoreboard.empty or "factor" not in scoreboard.columns:
            raise KeyError(f"Scoreboard is empty: universe={universe!r}")
        rows = scoreboard[scoreboard["factor"].astype(str) == str(factor)]
        if rows.empty or "analysis_run_id" not in rows.columns:
            raise KeyError(f"Scoreboard factor cannot be resolved to an analysis run: factor={factor!r}")
        actual = str(rows.iloc[0].get("analysis_run_id") or "").strip()
        if not actual or actual.lower() == "nan":
            raise KeyError(f"Scoreboard factor is missing analysis_run_id: factor={factor!r}")
        return self._require_run(universe=universe, run_id=actual)

    def _parse_run_meta(self, universe: str, meta_path: Path, payload: dict[str, Any]) -> AnalysisRun:
        table_paths = {
            str(name): self._resolve_artifact_path(path)
            for name, path in dict(payload.get("table_paths") or {}).items()
            if str(path or "").strip()
        }
        analysis_dir = self._resolve_artifact_path(payload.get("analysis_dir") or meta_path.parent)
        factor_metrics_raw = str(payload.get("factor_metrics_path", "") or "").strip()
        factor_metrics_path = self._resolve_artifact_path(factor_metrics_raw) if factor_metrics_raw else None
        alpha_names = tuple(str(x) for x in payload.get("alpha_names", []) if str(x))
        return AnalysisRun(
            universe=str(universe),
            run_id=str(payload.get("analysis_run_id") or meta_path.parent.name),
            period=_to_int(payload.get("period"), default=0),
            layers=_to_int(payload.get("layers"), default=0),
            created_at_utc=str(payload.get("created_at_utc", "") or ""),
            analysis_dir=analysis_dir,
            meta_path=meta_path,
            table_paths=table_paths,
            factor_metrics_path=factor_metrics_path,
            alpha_names=alpha_names,
            extra_meta=dict(payload.get("extra_meta") or {}),
        )

    def _require_superalpha_run(self, superalpha_id: str) -> tuple[AnalysisRun, dict[str, Any]]:
        sid = str(superalpha_id or "").strip()
        if not sid:
            raise KeyError("superalpha_id is required")
        patterns = [
            f"*/superalphas/{sid}/analysis_meta.json",
            f"*/superalphas/{sid}/meta.json",
        ]
        for pattern in patterns:
            for meta_path in sorted(self.store_root.glob(pattern)) if self.store_root.exists() else []:
                try:
                    payload = json.loads(meta_path.read_text(encoding="utf-8"))
                    universe = str(payload.get("universe") or meta_path.relative_to(self.store_root).parts[0])
                    run = self._parse_run_meta(universe=universe, meta_path=meta_path, payload=payload)
                    return run, payload
                except Exception:
                    continue
        raise KeyError(f"Unknown superalpha run: superalpha_id={sid!r}")

    def _superalpha_meta_to_dict(self, payload: dict[str, Any], *, meta_path: Path) -> dict[str, Any]:
        sid = str(payload.get("superalpha_id") or payload.get("analysis_run_id") or meta_path.parent.name)
        name = str(payload.get("name") or "").strip()
        extra_meta = dict(payload.get("extra_meta") or {})
        resource_summary = self._superalpha_resource_summary(payload, meta_path=meta_path)
        components = payload.get("components") or []
        if not isinstance(components, list):
            components = []
        return {
            "superalpha_id": sid,
            "run_id": sid,
            "name": name or sid,
            "display_name": name or f"SA {sid.replace('superalpha_', '')[:8]}",
            "universe": str(payload.get("universe") or ""),
            "created_at_utc": str(payload.get("created_at_utc") or ""),
            "combo_expression": str(payload.get("combo_expression") or ""),
            "component_count": _to_int(payload.get("component_count"), default=0),
            "component_join": str(payload.get("component_join") or extra_meta.get("component_join") or ""),
            "components": components,
            "status": str(payload.get("status") or "ok"),
            "summary": dict(payload.get("summary") or {}),
            "resource_summary": resource_summary,
            "cache_summary": dict(payload.get("cache_summary") or extra_meta.get("cache_summary") or {}),
            "cleanup_summary": dict(payload.get("cleanup_summary") or extra_meta.get("cleanup_summary") or {}),
            "artifact_path": str(meta_path.parent.as_posix()),
        }

    def _superalpha_resource_summary(self, payload: dict[str, Any], *, meta_path: Path) -> dict[str, Any]:
        existing = payload.get("resource_summary")
        if isinstance(existing, dict) and existing:
            return dict(existing)
        extra_meta = payload.get("extra_meta")
        if isinstance(extra_meta, dict):
            extra_summary = extra_meta.get("resource_summary") or extra_meta.get("resource_meta")
            if isinstance(extra_summary, dict) and extra_summary:
                return dict(extra_summary)

        diagnostics_path = ""
        if isinstance(extra_meta, dict):
            diagnostics_path = str(extra_meta.get("resource_diagnostics_path") or "").strip()
        candidates = []
        if diagnostics_path:
            candidates.append(self._resolve_artifact_path(diagnostics_path))
        candidates.append(meta_path.parent / "resource_meta.json")
        for path in candidates:
            if not path.exists() or not path.is_file():
                continue
            try:
                resource_meta = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(resource_meta, dict):
                continue
            duckdb_settings = (
                resource_meta.get("duckdb_settings") if isinstance(resource_meta.get("duckdb_settings"), dict) else {}
            )
            runtime_dirs = (
                resource_meta.get("runtime_dirs") if isinstance(resource_meta.get("runtime_dirs"), dict) else {}
            )
            return {
                "duckdb_temp_directory": str(
                    duckdb_settings.get("temp_directory") or runtime_dirs.get("duckdb_tmp") or ""
                ),
                "duckdb_memory_limit": str(duckdb_settings.get("memory_limit") or ""),
                "duckdb_max_temp_directory_size": str(duckdb_settings.get("max_temp_directory_size") or ""),
                "stage": str(resource_meta.get("stage") or ""),
            }
        return {}

    def _validate_superalpha_display_name(self, value: Any) -> str:
        name = str(value or "").strip()
        if not name:
            raise ValueError("name must not be empty")
        if len(name) > 80:
            raise ValueError("name must be at most 80 characters")
        if any(ord(ch) < 32 for ch in name):
            raise ValueError("name must not contain control characters")
        return name

    def _resolve_artifact_path(self, value: Any) -> Path:
        path = Path(str(value))
        if path.is_absolute():
            return path
        return Path.cwd() / path

    def _require_run(self, universe: str, run_id: str) -> AnalysisRun:
        for run in self._scan_runs():
            if run.universe == str(universe) and run.run_id == str(run_id):
                return run
        raise KeyError(f"Unknown analysis run: universe={universe!r}, run_id={run_id!r}")

    def _run_to_dict(self, run: AnalysisRun) -> dict[str, Any]:
        dashboard_path = run.dashboard_metrics_path
        metrics_path = run.metrics_path
        pnl_path = run.pnl_path
        benchmark_path = run.benchmark_pnl_path
        visualization_path = run.visualization_manifest_path
        phase_metrics_path = run.phase_metrics_path
        ic_path = run.ic_path
        hist_path = run.analysis_distribution_histogram_path
        decay_path = run.analysis_ic_decay_path
        coverage_path = run.analysis_factor_coverage_by_date_path
        factor_count = len(run.alpha_names)
        if factor_count <= 0 and metrics_path is not None and metrics_path.exists():
            try:
                factor_count = int(len(pd.read_csv(metrics_path, usecols=["factor"])))
            except Exception:
                pass
        phase_config = self._phase_config_for_run(run)
        return {
            "universe": run.universe,
            "run_id": run.run_id,
            "period": run.period,
            "layers": run.layers,
            "created_at_utc": run.created_at_utc,
            "analysis_dir": str(run.analysis_dir.as_posix()),
            "factor_count": int(factor_count),
            "has_dashboard_metrics": bool(dashboard_path is not None and dashboard_path.exists()),
            "has_factor_metrics": bool(run.factor_metrics_path is not None and run.factor_metrics_path.exists()),
            "has_portfolio_pnl": bool(pnl_path is not None and pnl_path.exists()),
            "has_benchmark_pnl": bool(benchmark_path is not None and benchmark_path.exists()),
            "has_visualizations": bool(visualization_path is not None and visualization_path.exists()),
            "has_phase_metrics": bool(phase_metrics_path is not None and phase_metrics_path.exists()),
            "has_ic_rows": bool(ic_path is not None and ic_path.exists()),
            "has_analysis_data": bool(
                (hist_path is not None and hist_path.exists())
                or (decay_path is not None and decay_path.exists())
                or (coverage_path is not None and coverage_path.exists())
                or (ic_path is not None and ic_path.exists())
                or (pnl_path is not None and pnl_path.exists())
            ),
            "phase_config": phase_config,
            "available_phases": list((phase_config or {}).get("available_phases", [])),
            "benchmark_config": dict(run.extra_meta.get("benchmark_config") or {}),
            "benchmark_status": dict(run.extra_meta.get("benchmark_status") or {}),
        }

    def _phase_config_for_run(self, run: AnalysisRun) -> dict[str, Any] | None:
        configured = dict(run.extra_meta.get("phase_config") or {})
        if configured.get("windows"):
            configured.setdefault(
                "available_phases",
                [str(row.get("key")) for row in configured.get("windows", []) if row.get("key")],
            )
            configured.setdefault("feedback_phase", "train")
            configured.setdefault("test_default_visible", False)
            return configured
        if not (
            run.phase_metrics_path is not None
            and run.phase_metrics_path.exists()
            or run.ic_path is not None
            and run.ic_path.exists()
        ):
            return None
        max_date = self._max_run_date(run)
        if max_date is None:
            return None
        windows = [
            window.to_dict()
            for window in build_phase_windows(SampleSplitConfig(), max_date=max_date, include_test=True)
        ]
        return {
            "windows": windows,
            "available_phases": [str(row.get("key")) for row in windows],
            "feedback_phase": "train",
            "test_default_visible": False,
        }

    def _max_run_date(self, run: AnalysisRun) -> str | None:
        candidates: list[pd.Timestamp] = []
        for path in [run.pnl_path, run.ic_path]:
            if path is None or not path.exists():
                continue
            try:
                frame = (
                    pd.read_parquet(path)
                    if path.suffix.lower() == ".parquet"
                    else pd.read_csv(path, usecols=["trade_date"])
                )
            except Exception:
                continue
            if "trade_date" not in frame.columns:
                continue
            dates = pd.to_datetime(frame["trade_date"], errors="coerce").dropna()
            if not dates.empty:
                candidates.append(pd.Timestamp(dates.max()))
        if not candidates:
            return None
        return max(candidates).strftime("%Y-%m-%d")

    def _phase_metrics_for_factor(self, run: AnalysisRun, factor: str) -> dict[str, Any] | None:
        metrics: dict[str, Any] = {}
        source = self._factor_metric_row(run.phase_metrics_path, factor)
        full = self._factor_metric_row(run.metrics_path, factor)
        if full:
            metrics["full"] = {
                key: full.get(key)
                for key in [
                    "ic_mean",
                    "ir",
                    "positive_ic_ratio",
                    "long_short_total_return",
                    "long_short_annualized_return",
                    "long_short_volatility",
                    "long_short_sharpe_ratio",
                    "long_short_max_drawdown",
                    "long_short_fitness_ratio",
                    "long_short_excess_annualized_return_vs_benchmark",
                    "long_only_total_return",
                    "long_only_annualized_return",
                    "long_only_volatility",
                    "long_only_sharpe_ratio",
                    "long_only_max_drawdown",
                    "long_only_fitness_ratio",
                    "long_only_excess_annualized_return_vs_benchmark",
                    "benchmark_annualized_return",
                    "best_minus_benchmark_annualized_return",
                    "turnover_long_short_mean",
                    "margin_long_short",
                    "margin_long_short_bp",
                    "turnover_long_only_mean",
                    "margin_long_only",
                    "margin_long_only_bp",
                    "score_total",
                ]
                if key in full
            }
        if not source:
            source = full
        if not source:
            return metrics or None
        feedback_phase = str(source.get("feedback_phase") or "train")
        for phase in ["train", "val", "test"]:
            phase_values = {}
            for suffix in [
                "obs",
                "ic_mean",
                "ic_std",
                "ir",
                "positive_ic_ratio",
                "long_short_total_return",
                "long_short_total_return_gross",
                "long_short_total_return_net",
                "long_short_annualized_return",
                "long_short_annualized_return_gross",
                "long_short_annualized_return_net",
                "long_short_volatility",
                "long_short_volatility_gross",
                "long_short_volatility_net",
                "long_short_sharpe_ratio",
                "long_short_sharpe_ratio_gross",
                "long_short_sharpe_ratio_net",
                "long_short_max_drawdown",
                "long_short_max_drawdown_gross",
                "long_short_max_drawdown_net",
                "long_short_fitness_ratio",
                "long_short_fitness_ratio_gross",
                "long_short_fitness_ratio_net",
                "long_short_excess_annualized_return_vs_benchmark",
                "long_short_excess_annualized_return_vs_benchmark_gross",
                "long_short_excess_annualized_return_vs_benchmark_net",
                "long_only_total_return",
                "long_only_total_return_gross",
                "long_only_total_return_net",
                "long_only_annualized_return",
                "long_only_annualized_return_gross",
                "long_only_annualized_return_net",
                "long_only_volatility",
                "long_only_volatility_gross",
                "long_only_volatility_net",
                "long_only_sharpe_ratio",
                "long_only_sharpe_ratio_gross",
                "long_only_sharpe_ratio_net",
                "long_only_max_drawdown",
                "long_only_max_drawdown_gross",
                "long_only_max_drawdown_net",
                "long_only_fitness_ratio",
                "long_only_fitness_ratio_gross",
                "long_only_fitness_ratio_net",
                "long_only_excess_annualized_return_vs_benchmark",
                "long_only_excess_annualized_return_vs_benchmark_gross",
                "long_only_excess_annualized_return_vs_benchmark_net",
                "benchmark_annualized_return",
                "best_minus_benchmark_annualized_return",
                "best_minus_benchmark_annualized_return_gross",
                "best_minus_benchmark_annualized_return_net",
                "turnover_long_short_mean",
                "turnover_long_short_mean_gross",
                "turnover_long_short_mean_net",
                "margin_long_short",
                "margin_long_short_gross",
                "margin_long_short_net",
                "margin_long_short_bp",
                "margin_long_short_bp_gross",
                "margin_long_short_bp_net",
                "turnover_long_only_mean",
                "turnover_long_only_mean_gross",
                "turnover_long_only_mean_net",
                "margin_long_only",
                "margin_long_only_gross",
                "margin_long_only_net",
                "margin_long_only_bp",
                "margin_long_only_bp_gross",
                "margin_long_only_bp_net",
                "score_total",
                "score_total_gross",
                "score_total_net",
                "score_total_basis",
                "feedback_score",
                "feedback_score_gross",
                "feedback_score_net",
                "feedback_score_basis",
            ]:
                key = f"{phase}_{suffix}"
                if key in source:
                    phase_values[suffix] = source.get(key)
            if phase_values:
                _fill_phase_display_metrics_from_basis(phase_values)
                if phase == feedback_phase:
                    if "feedback_score" not in phase_values:
                        phase_values["feedback_score"] = source.get("feedback_score", phase_values.get("score_total"))
                    if "feedback_score_basis" not in phase_values and "feedback_score_basis" in source:
                        phase_values["feedback_score_basis"] = source.get("feedback_score_basis")
                else:
                    phase_values.pop("feedback_score", None)
                    phase_values.pop("feedback_score_gross", None)
                    phase_values.pop("feedback_score_net", None)
                    phase_values.pop("feedback_score_basis", None)
                metrics[phase] = phase_values
        if "feedback_phase" in source:
            metrics["feedback_phase"] = source.get("feedback_phase")
        if "feedback_score" in source:
            metrics["feedback_score"] = source.get("feedback_score")
        return metrics or None

    def _file_signature(self, path: Path | None) -> tuple[str, int, int] | None:
        if path is None or not path.exists():
            return None
        try:
            stat = path.stat()
            return (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
        except Exception:
            return None

    @staticmethod
    def _cache_get(cache: OrderedDict[tuple[Any, ...], pd.DataFrame], key: tuple[Any, ...]) -> pd.DataFrame | None:
        value = cache.get(key)
        if value is None:
            return None
        cache.move_to_end(key)
        return value.copy()

    @staticmethod
    def _cache_put(
        cache: OrderedDict[tuple[Any, ...], pd.DataFrame],
        key: tuple[Any, ...],
        frame: pd.DataFrame,
        max_items: int,
    ) -> pd.DataFrame:
        cache[key] = frame.copy()
        cache.move_to_end(key)
        while len(cache) > max(1, int(max_items)):
            cache.popitem(last=False)
        return frame

    def _read_table_cached(self, path: Path | None, columns: list[str] | None = None) -> pd.DataFrame:
        signature = self._file_signature(path)
        if signature is None:
            return pd.DataFrame()
        cols_key = tuple(columns or ())
        key = ("table", *signature, cols_key)
        cached = self._cache_get(self._table_cache, key)
        if cached is not None:
            return cached
        try:
            if path is not None and path.suffix.lower() == ".parquet":
                frame = pd.read_parquet(path, columns=columns or None)
            else:
                frame = pd.read_csv(path, usecols=columns if columns else None)
        except Exception:
            frame = pd.DataFrame()
        return self._cache_put(self._table_cache, key, frame, self.table_cache_max_items)

    def _read_factor_table(
        self,
        path: Path | None,
        factor: str,
        *,
        columns: list[str] | None = None,
        factor_col: str = "factor",
    ) -> pd.DataFrame:
        signature = self._file_signature(path)
        if signature is None:
            return pd.DataFrame()
        cols = list(columns or [])
        if cols and factor_col not in cols:
            cols.append(factor_col)
        key = ("factor", *signature, str(factor), tuple(cols), str(factor_col))
        cached = self._cache_get(self._factor_frame_cache, key)
        if cached is not None:
            return cached
        try:
            if path is not None and path.suffix.lower() == ".parquet":
                frame = pd.read_parquet(
                    path,
                    columns=cols or None,
                    filters=[(factor_col, "==", str(factor))],
                )
            else:
                frame = pd.read_csv(path, usecols=cols if cols else None)
                if factor_col in frame.columns:
                    frame = frame[frame[factor_col].astype(str) == str(factor)].copy()
        except Exception:
            frame = _read_table(path)
            if not frame.empty and factor_col in frame.columns:
                frame = frame[frame[factor_col].astype(str) == str(factor)].copy()
            if columns:
                keep = [col for col in columns if col in frame.columns]
                frame = frame[keep].copy()
        return self._cache_put(self._factor_frame_cache, key, frame, self.factor_cache_max_items)

    def _factor_pnl_frame(self, run: AnalysisRun, factor: str, *, include_benchmark: bool = True) -> pd.DataFrame:
        columns = [
            "factor",
            "trade_date",
            "portfolio",
            "return",
            "cum_return",
            "return_gross",
            "cum_return_gross",
            "transaction_cost",
            "return_net",
            "cum_return_net",
            "has_net_pnl",
            "cost_model",
            "holding_count",
            "turnover",
            "buy_turnover",
            "sell_turnover",
            "blocked_buy_ratio",
            "blocked_sell_ratio",
            "tradability_return_drag",
        ]
        frame = self._read_factor_table(run.pnl_path, factor, columns=columns)
        frame = self._normalize_pnl_schema(frame)
        if include_benchmark:
            frame = self._append_benchmark_rows_for_factor(run, frame, factor)
            frame = self._normalize_pnl_schema(frame)
        return frame

    def _normalize_pnl_schema(self, frame: pd.DataFrame) -> pd.DataFrame:
        work = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        had_net_schema = {"return_net", "cum_return_net", "has_net_pnl"}.issubset(set(work.columns))
        if "return_gross" not in work.columns:
            work["return_gross"] = work["return"] if "return" in work.columns else pd.Series(dtype=float)
        if "cum_return_gross" not in work.columns:
            work["cum_return_gross"] = work["cum_return"] if "cum_return" in work.columns else pd.Series(dtype=float)
        if "has_net_pnl" not in work.columns:
            work["has_net_pnl"] = False
        if not had_net_schema and not work.empty:
            import logging

            logging.getLogger("dashboard.api").warning(
                "[dashboard] legacy portfolio artifact lacks net pnl schema; fallback to gross only"
            )
        work["has_net_pnl"] = work["has_net_pnl"].fillna(False).astype(bool)
        for col in [
            "transaction_cost",
            "return_net",
            "cum_return_net",
            "buy_turnover",
            "sell_turnover",
        ]:
            if col not in work.columns:
                work[col] = pd.NA
        if "cost_model" not in work.columns:
            work["cost_model"] = None
        if "return" not in work.columns:
            work["return"] = work["return_gross"]
        if "cum_return" not in work.columns:
            work["cum_return"] = work["cum_return_gross"]
        return work

    def _ensure_feedback_score_column(self, frame: pd.DataFrame) -> pd.DataFrame:
        work = frame.copy()
        if work.empty:
            if "feedback_score" not in work.columns:
                work["feedback_score"] = pd.Series(dtype=float)
            return work
        feedback = pd.Series(pd.NA, index=work.index, dtype="Float64")
        if "feedback_score" in work.columns:
            feedback = pd.to_numeric(work["feedback_score"], errors="coerce").astype("Float64")
        for col in FEEDBACK_SCORE_CANDIDATES:
            if col not in work.columns or col == "feedback_score":
                continue
            candidate = pd.to_numeric(work[col], errors="coerce").astype("Float64")
            feedback = feedback.fillna(candidate)
        work["feedback_score"] = feedback
        return work

    def _factor_portfolio_metrics(
        self,
        run: AnalysisRun,
        factor: str,
        *,
        phase_config: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        frame = self._factor_pnl_frame(run, factor, include_benchmark=True)
        return self._factor_portfolio_metrics_from_frame(run, factor, frame, phase_config=phase_config)

    def _factor_portfolio_metrics_from_frame(
        self,
        run: AnalysisRun,
        factor: str,
        frame: pd.DataFrame,
        *,
        phase_config: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        frame = self._normalize_pnl_schema(frame)
        required = {"factor", "portfolio", "trade_date", "return"}
        if frame.empty or not required.issubset(set(frame.columns)):
            return None
        work = frame[frame["factor"].astype(str) == str(factor)].copy()
        if work.empty:
            return None
        work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
        work = work.dropna(subset=["trade_date"])
        feedback_phase = _feedback_phase_from_config(phase_config)
        if phase_config:
            work["phase"] = _assign_phase_from_config(work["trade_date"], phase_config)
            scoped = work[work["phase"].astype(str) == feedback_phase].copy()
        else:
            scoped = work.copy()
        if scoped.empty:
            return {
                "scope_phase": feedback_phase,
                "rows": [],
                "benchmark_available": False,
                "message": "No portfolio PnL rows are available for the feedback phase.",
            }

        period = max(1, int(run.period or 1))
        benchmark_annual = _portfolio_annual_return(scoped, "benchmark", period=period)
        rows: list[dict[str, Any]] = []

        def build_rows(return_col: str, net_only: bool = False) -> list[dict[str, Any]]:
            result: list[dict[str, Any]] = []
            scoped_work = scoped.copy()
            if net_only:
                scoped_work = scoped_work[
                    scoped_work.get("has_net_pnl", pd.Series(False, index=scoped_work.index)).fillna(False).astype(bool)
                    & pd.to_numeric(
                        scoped_work.get(return_col, pd.Series(dtype=float)),
                        errors="coerce",
                    ).notna()
                ].copy()
            for portfolio, part in scoped_work.groupby("portfolio", sort=False):
                name = str(portfolio)
                returns = pd.to_numeric(part.get(return_col, pd.Series(dtype=float)), errors="coerce").dropna()
                if returns.empty:
                    continue
                risk = calculate_risk_metrics(returns, period=period)
                annual = _nullable_float(risk.get("annualized_return"))
                turnover = pd.to_numeric(part.get("turnover", pd.Series(dtype=float)), errors="coerce")
                turnover_mean = float(turnover.mean(skipna=True)) if turnover.notna().any() else None
                excess = None
                if name != "benchmark" and annual is not None and benchmark_annual is not None:
                    excess = annual - benchmark_annual
                result.append(
                    {
                        "portfolio": name,
                        "label": _portfolio_label(name),
                        "total_return": _nullable_float(risk.get("total_return")),
                        "annualized_return": annual,
                        "excess_annualized_return": _nullable_float(excess),
                        "annualized_volatility": _nullable_float(risk.get("volatility")),
                        "max_drawdown": _nullable_float(risk.get("max_drawdown")),
                        "turnover": _nullable_float(turnover_mean),
                        "sharpe": _nullable_float(risk.get("sharpe_ratio")),
                        "fitness": _nullable_float(risk.get("fitness_ratio")),
                        "obs": int(len(returns)),
                    }
                )
            return sorted(
                result,
                key=lambda row: _portfolio_sort_key(str(row.get("portfolio", ""))),
            )

        rows = build_rows("return")
        rows_net = build_rows("return_net", net_only=True)
        return {
            "scope_phase": feedback_phase,
            "rows": rows,
            "rows_net": rows_net,
            "net_available": bool(rows_net),
            "benchmark_available": any(str(row.get("portfolio")) == "benchmark" for row in rows),
            "message": None,
        }

    def _append_benchmark_rows_for_factor(self, run: AnalysisRun, frame: pd.DataFrame, factor: str) -> pd.DataFrame:
        work = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        if not work.empty and "portfolio" in work.columns and (work["portfolio"].astype(str) == "benchmark").any():
            return work
        benchmark = self._benchmark_pnl_for_run(run)
        if benchmark.empty:
            return work
        benchmark = benchmark.copy()
        benchmark.insert(0, "factor", str(factor))
        if work.empty:
            return benchmark
        return pd.concat([work, benchmark], ignore_index=True, sort=False)

    def _benchmark_pnl_for_run(self, run: AnalysisRun) -> pd.DataFrame:
        frame = self._read_table_cached(run.benchmark_pnl_path)
        if not frame.empty and {"trade_date", "return"}.issubset(frame.columns):
            work = frame.copy()
            work["portfolio"] = "benchmark"
            for col in [
                "cum_return",
                "holding_count",
                "turnover",
                "blocked_buy_ratio",
                "blocked_sell_ratio",
                "tradability_return_drag",
            ]:
                if col not in work.columns:
                    work[col] = None
            return work[
                [
                    "trade_date",
                    "portfolio",
                    "return",
                    "cum_return",
                    "holding_count",
                    "turnover",
                    "blocked_buy_ratio",
                    "blocked_sell_ratio",
                    "tradability_return_drag",
                ]
            ].copy()

        # Backward compatibility for runs that stored benchmark rows inside portfolio_pnl_df.
        pnl_path = run.pnl_path
        if pnl_path is None or not pnl_path.exists():
            return pd.DataFrame()
        try:
            pnl = pd.read_parquet(pnl_path)
        except Exception:
            return pd.DataFrame()
        if pnl.empty or "portfolio" not in pnl.columns:
            return pd.DataFrame()
        legacy = pnl[pnl["portfolio"].astype(str) == "benchmark"].copy()
        if legacy.empty:
            return pd.DataFrame()
        cols = [col for col in legacy.columns if col != "factor"]
        return legacy[cols].drop_duplicates(subset=["trade_date", "portfolio"], keep="first").reset_index(drop=True)

    def _factor_metric_row(self, path: Path | None, factor: str) -> dict[str, Any]:
        frame = self._read_factor_table(path, factor)
        if frame.empty or "factor" not in frame.columns:
            return {}
        rows = frame[frame["factor"].astype(str) == str(factor)]
        if rows.empty:
            return {}
        clean = rows.iloc[0].where(pd.notna(rows.iloc[0]), None)
        return clean.to_dict()

    def _factor_ic_series(
        self,
        run: AnalysisRun,
        factor: str,
        *,
        phase_config: dict[str, Any] | None,
        include_test: bool,
    ) -> list[dict[str, Any]]:
        col = f"{factor}_ic"
        path = run.ic_path
        if path is None or not path.exists():
            return []
        try:
            if path.suffix.lower() == ".parquet":
                frame = pd.read_parquet(path, columns=["trade_date", col])
            else:
                frame = pd.read_csv(path, usecols=lambda name: name in {"trade_date", col})
        except Exception:
            return []
        if frame.empty or "trade_date" not in frame.columns or col not in frame.columns:
            return []
        work = frame[["trade_date", col]].copy()
        work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
        work["ic"] = pd.to_numeric(work[col], errors="coerce")
        work = work.dropna(subset=["trade_date", "ic"]).sort_values("trade_date", kind="mergesort")
        if phase_config:
            work["phase"] = _assign_phase_from_config(work["trade_date"], phase_config)
            if not include_test:
                work = work[work["phase"].astype(str) != "test"].copy()
            work = work[work["phase"].astype(str).str.len() > 0].copy()
        else:
            work["phase"] = "full"
        if work.empty:
            return []
        work["cumulative_ic"] = work.groupby("phase", sort=False)["ic"].cumsum()
        work["trade_date"] = work["trade_date"].dt.strftime("%Y-%m-%d")
        return _records(work[["trade_date", "phase", "ic", "cumulative_ic"]])

    @staticmethod
    def _yearly_ic_summary(ic_series: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not ic_series:
            return []
        frame = pd.DataFrame(ic_series)
        if frame.empty or "trade_date" not in frame.columns or "ic" not in frame.columns:
            return []
        frame["year"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.year
        frame["ic"] = pd.to_numeric(frame["ic"], errors="coerce")
        if "phase" not in frame.columns:
            frame["phase"] = "full"
        frame["phase"] = frame["phase"].fillna("full").astype(str)
        grouped = (
            frame.dropna(subset=["year", "ic"])
            .groupby(["phase", "year"], sort=False)["ic"]
            .agg(ic_mean="mean", obs="count")
            .reset_index()
        )
        if grouped.empty:
            return []
        grouped["year"] = grouped["year"].astype(int).astype(str)
        return _records(grouped[["phase", "year", "ic_mean", "obs"]])

    @staticmethod
    def _monthly_ic_summary(ic_series: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not ic_series:
            return []
        frame = pd.DataFrame(ic_series)
        if frame.empty or "trade_date" not in frame.columns or "ic" not in frame.columns:
            return []
        frame["month"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.to_period("M").astype(str)
        frame["ic"] = pd.to_numeric(frame["ic"], errors="coerce")
        if "phase" not in frame.columns:
            frame["phase"] = "full"
        frame["phase"] = frame["phase"].fillna("full").astype(str)
        grouped = (
            frame.dropna(subset=["month", "ic"])
            .groupby(["phase", "month"], sort=False)["ic"]
            .agg(ic_mean="mean", obs="count")
            .reset_index()
        )
        if grouped.empty:
            return []
        return _records(grouped[["phase", "month", "ic_mean", "obs"]])

    def _factor_coverage_series(
        self,
        run: AnalysisRun,
        factor: str,
        *,
        phase_config: dict[str, Any] | None,
        include_test: bool,
    ) -> list[dict[str, Any]]:
        columns = [
            "factor",
            "trade_date",
            "coverage_rate",
            "non_missing_obs",
            "total_obs",
        ]
        frame = self._read_factor_table(run.analysis_factor_coverage_by_date_path, factor, columns=columns)
        if frame.empty or "trade_date" not in frame.columns:
            return []
        work = frame.copy()
        work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
        work = work.dropna(subset=["trade_date"]).sort_values("trade_date", kind="mergesort")
        if phase_config:
            work["phase"] = _assign_phase_from_config(work["trade_date"], phase_config)
            if not include_test:
                work = work[work["phase"].astype(str) != "test"].copy()
            work = work[work["phase"].astype(str).str.len() > 0].copy()
        else:
            work["phase"] = "full"
        work["coverage_rate"] = pd.to_numeric(work.get("coverage_rate"), errors="coerce")
        if "non_missing_obs" in work.columns:
            work["non_missing_obs"] = pd.to_numeric(work["non_missing_obs"], errors="coerce")
        if "total_obs" in work.columns:
            work["total_obs"] = pd.to_numeric(work["total_obs"], errors="coerce")
        work["trade_date"] = work["trade_date"].dt.strftime("%Y-%m-%d")
        keep = [
            c
            for c in [
                "trade_date",
                "phase",
                "coverage_rate",
                "non_missing_obs",
                "total_obs",
            ]
            if c in work.columns
        ]
        return _records(work[keep])

    @staticmethod
    def _ic_distribution_histogram(
        ic_series: list[dict[str, Any]],
        *,
        factor: str,
        bins: int = 30,
    ) -> list[dict[str, Any]]:
        if not ic_series:
            return []
        frame = pd.DataFrame(ic_series)
        if frame.empty or "ic" not in frame.columns:
            return []
        frame["ic"] = pd.to_numeric(frame["ic"], errors="coerce")
        if "phase" not in frame.columns:
            frame["phase"] = "full"
        frame["phase"] = frame["phase"].fillna("full").astype(str)
        frame = frame.dropna(subset=["ic"])
        if frame.empty:
            return []
        finite = frame["ic"].dropna()
        low = float(finite.min())
        high = float(finite.max())
        if low == high:
            width = max(abs(low) * 0.01, 0.01)
            edges = [low - width, low + width]
        else:
            edges = [float(x) for x in pd.interval_range(start=low, end=high, periods=max(1, int(bins))).left]
            edges.append(high)
            edges = sorted(set(edges))
            if len(edges) < 2:
                edges = [low, high]
        rows: list[dict[str, Any]] = []
        for phase, group in frame.groupby("phase", sort=False):
            values = pd.to_numeric(group["ic"], errors="coerce").dropna()
            if values.empty:
                continue
            cut = pd.cut(values, bins=edges, include_lowest=True, duplicates="drop")
            counts = cut.value_counts(sort=False)
            total = int(counts.sum())
            for idx, (interval, count) in enumerate(counts.items()):
                if pd.isna(interval):
                    continue
                left = float(interval.left)
                right = float(interval.right)
                rows.append(
                    {
                        "factor": str(factor),
                        "phase": str(phase),
                        "bin_index": int(idx),
                        "bin_left": left,
                        "bin_right": right,
                        "bin_mid": float((left + right) / 2.0),
                        "count": int(count),
                        "total_count": int(total),
                    }
                )
        return rows

    def _factor_distribution_histogram(
        self, run: AnalysisRun, factor: str, *, include_test: bool
    ) -> list[dict[str, Any]]:
        frame = self._read_table_cached(run.analysis_distribution_histogram_path)
        if frame.empty or "factor" not in frame.columns or "phase" not in frame.columns:
            return []
        work = frame[frame["factor"].astype(str) == str(factor)].copy()
        if not include_test:
            work = work[work["phase"].astype(str) != "test"].copy()
        if work.empty:
            return []
        if "bin_index" in work.columns:
            work["_bin_index"] = pd.to_numeric(work["bin_index"], errors="coerce")
            work = work.sort_values(["phase", "_bin_index"], kind="mergesort").drop(columns=["_bin_index"])
        return _records(work)

    def _factor_ic_decay(self, run: AnalysisRun, factor: str, *, include_test: bool) -> list[dict[str, Any]]:
        frame = self._read_table_cached(run.analysis_ic_decay_path)
        if frame.empty or "factor" not in frame.columns or "phase" not in frame.columns:
            return []
        work = frame[frame["factor"].astype(str) == str(factor)].copy()
        if not include_test:
            work = work[work["phase"].astype(str) != "test"].copy()
        if work.empty:
            return []
        if "lag" in work.columns:
            work["_lag"] = pd.to_numeric(work["lag"], errors="coerce")
            work = work.sort_values(["phase", "_lag"], kind="mergesort").drop(columns=["_lag"])
        return _records(work)

    def _factor_layer_terminal_returns(
        self,
        run: AnalysisRun,
        factor: str,
        *,
        phase_config: dict[str, Any] | None,
        include_test: bool,
    ) -> list[dict[str, Any]]:
        frame = self._factor_pnl_frame(run, factor, include_benchmark=False)
        return self._factor_layer_terminal_returns_from_frame(
            run,
            factor,
            frame,
            phase_config=phase_config,
            include_test=include_test,
        )

    def _factor_layer_terminal_returns_from_frame(
        self,
        run: AnalysisRun,
        factor: str,
        frame: pd.DataFrame,
        *,
        phase_config: dict[str, Any] | None,
        include_test: bool,
    ) -> list[dict[str, Any]]:
        if frame.empty or not {"factor", "trade_date", "portfolio", "return"}.issubset(frame.columns):
            return []
        work = frame[
            (frame["factor"].astype(str) == str(factor)) & frame["portfolio"].astype(str).str.startswith("layer_")
        ].copy()
        if work.empty:
            return []
        work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
        work["return"] = pd.to_numeric(work["return"], errors="coerce")
        work = work.dropna(subset=["trade_date"])
        if phase_config:
            work["phase"] = _assign_phase_from_config(work["trade_date"], phase_config)
            if not include_test:
                work = work[work["phase"].astype(str) != "test"].copy()
            work = work[work["phase"].astype(str).str.len() > 0].copy()
        else:
            work["phase"] = "full"
        work = work.dropna(subset=["return"])
        if work.empty:
            return []
        rows: list[dict[str, Any]] = []
        for (phase, portfolio), group in work.groupby(["phase", "portfolio"], sort=False):
            group = group.sort_values("trade_date", kind="mergesort")
            returns = pd.to_numeric(group["return"], errors="coerce").dropna()
            if returns.empty:
                continue
            layer = str(portfolio).replace("layer_", "", 1)
            rows.append(
                {
                    "phase": phase,
                    "portfolio": str(portfolio),
                    "layer": layer,
                    "terminal_return": float((1.0 + returns.fillna(0.0)).prod() - 1.0),
                    "obs": int(len(returns)),
                }
            )
        if rows:
            by_phase = pd.DataFrame(rows)
            by_phase["_layer_num"] = pd.to_numeric(by_phase["layer"], errors="coerce")
            by_phase["terminal_return"] = pd.to_numeric(by_phase["terminal_return"], errors="coerce")
            rank_corr_by_phase: dict[str, float | None] = {}
            for phase, group in by_phase.dropna(subset=["_layer_num", "terminal_return"]).groupby("phase", sort=False):
                if len(group) < 2:
                    rank_corr_by_phase[str(phase)] = None
                    continue
                corr = group["_layer_num"].corr(group["terminal_return"], method="spearman")
                rank_corr_by_phase[str(phase)] = None if pd.isna(corr) else float(corr)
            for row in rows:
                row["rank_corr"] = rank_corr_by_phase.get(str(row.get("phase")), None)
        return rows

    def _load_visualization_manifest(self, path: Path) -> pd.DataFrame:
        return self._read_table_cached(path)

    @staticmethod
    def _resolve_manifest_image_path(run: AnalysisRun, row: dict[str, Any]) -> Path | None:
        relative_path = str(row.get("relative_path", "") or "").strip()
        if not relative_path:
            return None
        candidate = Path(relative_path)
        if candidate.is_absolute():
            return None
        root = run.analysis_dir.resolve()
        resolved = (run.analysis_dir / candidate).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            return None
        return resolved

    @staticmethod
    def _filter_effective(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if "effectiveness_tier" in df.columns:
            tier = df["effectiveness_tier"].astype(str).str.upper().str.strip()
            mask = tier.isin({"S", "A", "B"})
            if mask.any():
                return df[mask].copy()
        if "score_total" in df.columns:
            score = pd.to_numeric(df["score_total"], errors="coerce")
            return df[score >= 60.0].copy()
        return df


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    clean = df.copy()
    for col in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[col]):
            clean[col] = pd.to_datetime(clean[col], errors="coerce").dt.strftime("%Y-%m-%d")
    clean = clean.where(pd.notna(clean), None)
    return json.loads(clean.to_json(orient="records", force_ascii=False))


def _normalise_field_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "field_name",
        "factor_family",
        "category",
        "source_table",
        "field_type",
        "dtype",
        "unit",
        "available_start",
        "available_end",
        "is_default_enabled",
        "is_searchable",
        "description",
        "field_role",
        "available_at",
        "preprocessing_policy",
        "leakage_safe",
        "coverage_scope",
        "coverage_start_date",
        "coverage_end_date",
        "coverage_row_count",
        "non_null_count",
        "coverage_rate",
        "missing_rate",
        "finite_count",
        "finite_rate",
        "coverage_status",
        "coverage_updated_at_utc",
    ]
    out = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    if "field_name" not in out.columns and "name" in out.columns:
        out["field_name"] = out["name"]
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    for col in [
        "field_name",
        "factor_family",
        "category",
        "source_table",
        "field_type",
        "dtype",
        "unit",
        "description",
        "field_role",
        "available_at",
        "preprocessing_policy",
    ]:
        out[col] = out[col].fillna("").astype(str)
    out["factor_family"] = out["factor_family"].str.strip()
    out.loc[out["factor_family"] == "", "factor_family"] = "other"
    for col in ["is_default_enabled", "is_searchable", "leakage_safe"]:
        out[col] = _bool_series(out[col])
    for col in ["coverage_rate", "missing_rate", "finite_rate"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["coverage_row_count", "non_null_count", "finite_count"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[columns]


def _bool_series(values: pd.Series) -> pd.Series:
    if values is None:
        return pd.Series(dtype=bool)
    return values.fillna(False).map(lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"})


def _min_text(values: pd.Series | None) -> str | None:
    if values is None:
        return None
    clean = values.dropna().astype(str)
    clean = clean[clean.str.len() > 0]
    return str(clean.min()) if not clean.empty else None


def _max_text(values: pd.Series | None) -> str | None:
    if values is None:
        return None
    clean = values.dropna().astype(str)
    clean = clean[clean.str.len() > 0]
    return str(clean.max()) if not clean.empty else None


def _days_since(value: str | None) -> int | None:
    if not value:
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    if getattr(ts, "tzinfo", None) is None:
        ts = ts.tz_localize(timezone.utc)
    today = pd.Timestamp(datetime.now(timezone.utc).date(), tz=timezone.utc)
    return int((today - ts.normalize()).days)


def _factor_library_thresholds(cfg: FactorLibraryConfig) -> dict[str, Any]:
    return {
        "min_score": float(cfg.min_score),
        "staging_min_score": float(cfg.staging_min_score),
        "max_signal_corr": float(cfg.max_signal_corr),
        "max_ic_corr": float(cfg.max_ic_corr),
        "max_pnl_corr": float(cfg.max_pnl_corr),
        "staging_max_corr": float(cfg.staging_max_corr),
    }


def _mark_legacy_library_row(row: dict[str, Any], cfg: FactorLibraryConfig) -> dict[str, Any]:
    out = dict(row)
    status = str(out.get("status") or "").lower()
    score = pd.to_numeric(pd.Series([out.get("score")]), errors="coerce").iloc[0]
    if status == "accepted" and pd.notna(score) and float(score) < float(cfg.min_score):
        out["library_status_effective"] = "legacy_accepted"
        out["legacy_status_warning"] = (
            f"accepted under an older or manual rule with score {float(score):.1f}; "
            f"current accepted min_score is {float(cfg.min_score):.1f}"
        )
    else:
        out["library_status_effective"] = status or "none"
        out["legacy_status_warning"] = ""
    return out


def _read_table(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _nullable_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def _assign_phase_from_config(dates: pd.Series, phase_config: dict[str, Any]) -> pd.Series:
    out = pd.Series("", index=dates.index, dtype=object)
    clean_dates = pd.to_datetime(dates, errors="coerce")
    for window in phase_config.get("windows", []) or []:
        key = str(window.get("key", "") or "")
        start = str(window.get("start", "") or "")
        end = str(window.get("end", "") or "")
        if not key or not start:
            continue
        mask = clean_dates >= pd.Timestamp(start)
        if end:
            mask &= clean_dates <= pd.Timestamp(end)
        out.loc[mask] = key
    return out


def _filter_phase_config(phase_config: dict[str, Any] | None, include_test: bool) -> dict[str, Any] | None:
    if not phase_config:
        return phase_config
    out = dict(phase_config)
    windows = list(out.get("windows", []) or [])
    if not include_test:
        windows = [window for window in windows if str(window.get("key", "")) != "test"]
    out["windows"] = windows
    out["available_phases"] = [str(window.get("key")) for window in windows if window.get("key")]
    return out


def _filter_phase_metrics(phase_metrics: dict[str, Any] | None, include_test: bool) -> dict[str, Any] | None:
    if not phase_metrics or include_test:
        return phase_metrics
    out = dict(phase_metrics)
    out.pop("test", None)
    return out


def _fill_phase_display_metrics_from_basis(values: dict[str, Any]) -> None:
    basis = str(values.get("score_total_basis") or values.get("score_basis") or "").lower()
    prefer_net = basis == "net"
    for key in [
        "long_short_total_return",
        "long_short_annualized_return",
        "long_short_volatility",
        "long_short_sharpe_ratio",
        "long_short_max_drawdown",
        "long_short_fitness_ratio",
        "long_short_excess_annualized_return_vs_benchmark",
        "long_only_total_return",
        "long_only_annualized_return",
        "long_only_volatility",
        "long_only_sharpe_ratio",
        "long_only_max_drawdown",
        "long_only_fitness_ratio",
        "long_only_excess_annualized_return_vs_benchmark",
        "best_minus_benchmark_annualized_return",
        "turnover_long_short_mean",
        "margin_long_short",
        "margin_long_short_bp",
        "turnover_long_only_mean",
        "margin_long_only",
        "margin_long_only_bp",
    ]:
        if _is_finite_number(values.get(key)):
            continue
        net_value = values.get(f"{key}_net")
        gross_value = values.get(f"{key}_gross")
        preferred = net_value if prefer_net else gross_value
        fallback = gross_value if prefer_net else net_value
        if _is_finite_number(preferred):
            values[key] = preferred
        elif _is_finite_number(fallback):
            values[key] = fallback


def _is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _feedback_phase_from_config(phase_config: dict[str, Any] | None) -> str:
    if phase_config:
        value = str(phase_config.get("feedback_phase") or "").strip()
        if value:
            return value
    return "train"


def _portfolio_annual_return(frame: pd.DataFrame, portfolio: str, *, period: int) -> float | None:
    part = frame[frame["portfolio"].astype(str) == str(portfolio)]
    returns = pd.to_numeric(part.get("return", pd.Series(dtype=float)), errors="coerce").dropna()
    if returns.empty:
        return None
    return _nullable_float(calculate_risk_metrics(returns, period=period).get("annualized_return"))


def _nullable_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        numeric = float(value)
        if not pd.notna(numeric):
            return None
        return numeric
    except Exception:
        return None


def _portfolio_label(portfolio: str) -> str:
    text = str(portfolio)
    if text.startswith("layer_"):
        suffix = text.split("_", 1)[1]
        return f"Layer {suffix}"
    labels = {
        "long_short": "Long-short",
        "long_only": "Long-only",
        "long_10": "Long-10",
        "benchmark": "Benchmark",
    }
    return labels.get(text, text.replace("_", " ").title())


def _portfolio_sort_key(portfolio: str) -> tuple[int, int, str]:
    text = str(portfolio)
    if text.startswith("layer_"):
        try:
            return (0, int(text.split("_", 1)[1]), text)
        except Exception:
            return (0, 999, text)
    order = {"long_short": 1, "long_only": 2, "long_10": 3, "benchmark": 4}
    return (order.get(text, 9), 0, text)
