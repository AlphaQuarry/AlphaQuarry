from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SOURCE_VIEW = "v_project_panel_cn_a"
DEFAULT_DUCKDB_PATH = "data/duckdb/market.duckdb"
DEFAULT_LAKE_ROOT = "data/lake"
DEFAULT_FIELD_CATALOG_ENABLED_CATEGORIES: tuple[str, ...] = (
    "price",
    "return",
    "liquidity",
    "valuation",
    "industry",
    "event",
)
DEFAULT_FIELD_CATALOG_NON_SEARCHABLE_FIELDS: tuple[str, ...] = (
    "date",
    "code",
    "universe",
    "tradable",
    "is_st",
    "is_suspended",
    "days_since_listed",
)


@dataclass(frozen=True)
class TushareSettings:
    token: str = ""
    http_url: str = ""
    max_retries: int = 3
    retry_sleep_seconds: float = 1.5
    request_pause_seconds: float = 0.0


@dataclass(frozen=True)
class LakePathSettings:
    lake_root: str = DEFAULT_LAKE_ROOT
    vendor_raw_subdir: str = "vendor_raw/tushare"
    curated_subdir: str = "curated"
    snapshots_subdir: str = "snapshots"
    meta_subdir: str = "meta"
    duckdb_path: str = DEFAULT_DUCKDB_PATH

    @property
    def lake_root_path(self) -> Path:
        return Path(self.lake_root)

    @property
    def vendor_raw_path(self) -> Path:
        return self.lake_root_path / self.vendor_raw_subdir

    @property
    def curated_path(self) -> Path:
        return self.lake_root_path / self.curated_subdir

    @property
    def snapshots_path(self) -> Path:
        return self.lake_root_path / self.snapshots_subdir

    @property
    def meta_path(self) -> Path:
        return self.lake_root_path / self.meta_subdir

    @property
    def duckdb_path_obj(self) -> Path:
        return Path(self.duckdb_path)


@dataclass(frozen=True)
class DataSourceSettings:
    source_backend: str = "duckdb"
    source_view: str = DEFAULT_SOURCE_VIEW
    adjust_mode: str = "qfq"
    moneyflow_source: str = "moneyflow"
    field_catalog_version: str = "v1"
    run_filters: dict[str, Any] = field(default_factory=lambda: {"universe_only": True, "include_bj": True})
    universe_min_days_since_listed: int = 60
    universe_exclude_st: bool = True
    include_bj: bool = True
    tradable_require_close: bool = True
    tradable_require_positive_volume: bool = True
    tradable_require_positive_amount: bool = True
    field_catalog_enabled_categories: tuple[str, ...] = DEFAULT_FIELD_CATALOG_ENABLED_CATEGORIES
    field_catalog_non_searchable_fields: tuple[str, ...] = DEFAULT_FIELD_CATALOG_NON_SEARCHABLE_FIELDS
    update_lookback_trade_days: int = 5
    paths: LakePathSettings = field(default_factory=LakePathSettings)
    tushare: TushareSettings = field(default_factory=TushareSettings)


def load_datasource_settings(
    config_path: str | Path | None = None,
) -> DataSourceSettings:
    payload = _load_yaml_payload(config_path)
    payload = _deep_merge(payload, _load_env_overrides())

    run_filters = payload.get("run_filters", {})
    if isinstance(run_filters, str):
        try:
            run_filters = json.loads(run_filters)
        except Exception:
            run_filters = {"universe_only": str(run_filters).strip().lower() in {"1", "true", "yes"}}
    if not isinstance(run_filters, dict):
        run_filters = {"universe_only": True}
    run_filters = {str(k): v for k, v in run_filters.items()}

    paths_cfg = payload.get("paths", {})
    if not isinstance(paths_cfg, dict):
        paths_cfg = {}
    tushare_cfg = payload.get("tushare", {})
    if not isinstance(tushare_cfg, dict):
        tushare_cfg = {}
    universe_cfg = payload.get("universe", {})
    if not isinstance(universe_cfg, dict):
        universe_cfg = {}
    tradability_cfg = payload.get("tradability", {})
    if not isinstance(tradability_cfg, dict):
        tradability_cfg = {}
    field_catalog_cfg = payload.get("field_catalog", {})
    if not isinstance(field_catalog_cfg, dict):
        field_catalog_cfg = {}
    update_cfg = payload.get("update", {})
    if not isinstance(update_cfg, dict):
        update_cfg = {}

    adjust_mode = str(payload.get("adjust_mode", "qfq") or "qfq").strip().lower()
    if adjust_mode not in {"qfq", "hfq"}:
        raise ValueError(f"Unsupported adjust_mode: {adjust_mode}. Expected one of ['qfq', 'hfq']")

    source_backend = str(payload.get("source_backend", "duckdb") or "duckdb").strip().lower()
    if source_backend not in {"duckdb", "file"}:
        raise ValueError(f"Unsupported source_backend: {source_backend}")
    moneyflow_source = str(payload.get("moneyflow_source", "moneyflow") or "moneyflow").strip().lower()
    if moneyflow_source not in {"moneyflow", "moneyflow_ths"}:
        raise ValueError("Unsupported moneyflow_source. Expected 'moneyflow' or 'moneyflow_ths'")

    include_bj = _to_bool(universe_cfg.get("include_bj", True), default=True)
    universe_min_days_since_listed = max(0, int(universe_cfg.get("min_days_since_listed", 60)))
    universe_exclude_st = _to_bool(universe_cfg.get("exclude_st", True), default=True)
    tradable_require_close = _to_bool(tradability_cfg.get("require_close", True), default=True)
    tradable_require_positive_volume = _to_bool(
        tradability_cfg.get("require_positive_volume", True),
        default=True,
    )
    tradable_require_positive_amount = _to_bool(
        tradability_cfg.get("require_positive_amount", True),
        default=True,
    )
    field_catalog_enabled_categories = tuple(
        _normalize_str_list(
            field_catalog_cfg.get(
                "default_enabled_categories",
                list(DEFAULT_FIELD_CATALOG_ENABLED_CATEGORIES),
            )
        )
        or list(DEFAULT_FIELD_CATALOG_ENABLED_CATEGORIES)
    )
    field_catalog_non_searchable_fields = tuple(
        _normalize_str_list(
            field_catalog_cfg.get(
                "non_searchable_fields",
                list(DEFAULT_FIELD_CATALOG_NON_SEARCHABLE_FIELDS),
            )
        )
        or list(DEFAULT_FIELD_CATALOG_NON_SEARCHABLE_FIELDS)
    )
    update_lookback_trade_days = max(0, int(update_cfg.get("lookback_trade_days", 5)))

    run_filters.setdefault("universe_only", True)
    run_filters.setdefault("include_bj", include_bj)
    run_filters["universe_only"] = _to_bool(run_filters.get("universe_only", True), default=True)
    run_filters["include_bj"] = _to_bool(run_filters.get("include_bj", include_bj), default=include_bj)

    paths = LakePathSettings(
        lake_root=str(paths_cfg.get("lake_root", DEFAULT_LAKE_ROOT)),
        vendor_raw_subdir=str(paths_cfg.get("vendor_raw_subdir", "vendor_raw/tushare")),
        curated_subdir=str(paths_cfg.get("curated_subdir", "curated")),
        snapshots_subdir=str(paths_cfg.get("snapshots_subdir", "snapshots")),
        meta_subdir=str(paths_cfg.get("meta_subdir", "meta")),
        duckdb_path=str(paths_cfg.get("duckdb_path", DEFAULT_DUCKDB_PATH)),
    )
    tushare = TushareSettings(
        token=str(tushare_cfg.get("token", "") or "").strip(),
        http_url=str(tushare_cfg.get("http_url", "") or "").strip(),
        max_retries=max(1, int(tushare_cfg.get("max_retries", 3))),
        retry_sleep_seconds=max(0.0, float(tushare_cfg.get("retry_sleep_seconds", 1.5))),
        request_pause_seconds=max(0.0, float(tushare_cfg.get("request_pause_seconds", 0.0))),
    )

    return DataSourceSettings(
        source_backend=source_backend,
        source_view=str(payload.get("source_view", DEFAULT_SOURCE_VIEW) or DEFAULT_SOURCE_VIEW).strip(),
        adjust_mode=adjust_mode,
        moneyflow_source=moneyflow_source,
        field_catalog_version=str(payload.get("field_catalog_version", "v1") or "v1").strip(),
        run_filters={str(k): v for k, v in run_filters.items()},
        universe_min_days_since_listed=universe_min_days_since_listed,
        universe_exclude_st=universe_exclude_st,
        include_bj=include_bj,
        tradable_require_close=tradable_require_close,
        tradable_require_positive_volume=tradable_require_positive_volume,
        tradable_require_positive_amount=tradable_require_positive_amount,
        field_catalog_enabled_categories=field_catalog_enabled_categories,
        field_catalog_non_searchable_fields=field_catalog_non_searchable_fields,
        update_lookback_trade_days=update_lookback_trade_days,
        paths=paths,
        tushare=tushare,
    )


def _load_yaml_payload(config_path: str | Path | None) -> dict[str, Any]:
    candidates: list[Path] = []
    if config_path is not None and str(config_path).strip():
        candidates.append(Path(config_path))
    else:
        candidates.extend(
            [
                Path("configs/datasource.local.yaml"),
                Path("configs/datasource.yaml"),
            ]
        )

    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        loaded = yaml.safe_load(text)
        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Datasource config must be a mapping: {path}")
        return loaded
    return {}


def _load_env_overrides() -> dict[str, Any]:
    env = os.environ
    run_filters = _read_run_filters_from_env(env)
    out: dict[str, Any] = {
        "paths": {
            "lake_root": env.get("PROJECT_LAKE_ROOT", ""),
            "duckdb_path": env.get("PROJECT_DUCKDB_PATH", ""),
        },
        "source_view": env.get("PROJECT_SOURCE_VIEW", ""),
        "source_backend": env.get("PROJECT_SOURCE_BACKEND", ""),
        "adjust_mode": env.get("PROJECT_ADJUST_MODE", ""),
        "field_catalog_version": env.get("PROJECT_FIELD_CATALOG_VERSION", ""),
        "run_filters": run_filters,
        "universe": {
            "include_bj": _env_bool(env, "PROJECT_INCLUDE_BJ", None),
            "exclude_st": _env_bool(env, "PROJECT_UNIVERSE_EXCLUDE_ST", None),
            "min_days_since_listed": _env_int(env, "PROJECT_UNIVERSE_MIN_DAYS_SINCE_LISTED"),
        },
        "update": {
            "lookback_trade_days": _env_int(env, "PROJECT_UPDATE_LOOKBACK_TRADE_DAYS"),
        },
        "field_catalog": {
            "default_enabled_categories": _env_csv_list(env, "PROJECT_FIELD_CATALOG_ENABLED_CATEGORIES"),
            "non_searchable_fields": _env_csv_list(env, "PROJECT_FIELD_CATALOG_NON_SEARCHABLE_FIELDS"),
        },
        "tushare": {
            "token": env.get("TUSHARE_TOKEN", ""),
            "http_url": env.get("TUSHARE_HTTP_URL", ""),
            "max_retries": _env_int(env, "TUSHARE_MAX_RETRIES"),
            "retry_sleep_seconds": _env_float(env, "TUSHARE_RETRY_SLEEP_SECONDS"),
            "request_pause_seconds": _env_float(env, "TUSHARE_REQUEST_PAUSE_SECONDS"),
        },
    }
    return _drop_empty_values(out)


def _read_run_filters_from_env(env: dict[str, str]) -> dict[str, Any]:
    raw = str(env.get("PROJECT_RUN_FILTERS_JSON", "") or "").strip()
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _env_int(env: dict[str, str], key: str) -> int | None:
    raw = str(env.get(key, "") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _env_float(env: dict[str, str], key: str) -> float | None:
    raw = str(env.get(key, "") or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _env_bool(env: dict[str, str], key: str, default: bool | None) -> bool | None:
    raw = str(env.get(key, "") or "").strip()
    if not raw:
        return default
    return _to_bool(raw, default=bool(default) if default is not None else False)


def _env_csv_list(env: dict[str, str], key: str) -> list[str]:
    raw = str(env.get(key, "") or "").strip()
    if not raw:
        return []
    return _normalize_str_list(raw)


def _normalize_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [x.strip() for x in value.split(",")]
        return [x for x in parts if x]
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out
    return []


def _to_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if not text:
        return bool(default)
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _drop_empty_values(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            cleaned = _drop_empty_values(item)
            if cleaned is None:
                continue
            if isinstance(cleaned, str) and cleaned == "":
                continue
            if isinstance(cleaned, dict) and not cleaned:
                continue
            if isinstance(cleaned, list) and not cleaned:
                continue
            out[str(key)] = cleaned
        return out
    if isinstance(value, list):
        out_list = [_drop_empty_values(x) for x in value]
        return [x for x in out_list if x is not None and x != ""]
    return value


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
