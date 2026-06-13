from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LiveDataConfig:
    date_col: str = "date"
    code_col: str = "code"
    min_field_non_null_rate: float = 0.60
    strict_available_at: bool = False
    catalog_missing_policy: str = "warn"
    market_value_missing_policy: str = "block"


@dataclass
class LiveSuperalphaConfig:
    max_active: int = 5
    live_window_trade_days: int = 320
    lookback_buffer_days: int = 5
    max_expression_lookback_days: int = 720
    allow_reproduce_fallback: bool = True


@dataclass
class LivePortfolioConfig:
    target_count: int = 100
    min_target_count: int = 50
    min_target_fill_ratio: float = 0.70
    max_single_name_weight: float = 0.02
    cash_buffer_ratio: float = 0.02


@dataclass
class LiveTradabilityConfig:
    enabled: bool = True
    strict_required_fields: bool = True
    critical_fields: tuple[str, ...] = (
        "close",
        "can_buy",
        "can_sell",
        "is_st",
        "is_suspended",
        "up_limit",
        "down_limit",
        "is_limit_up_close",
        "is_limit_down_close",
    )
    market_value_any_of: tuple[str, ...] = ("circ_mv", "total_mv", "float_mv")
    include_bj: bool = False
    exclude_st: bool = True
    exclude_suspended: bool = True


@dataclass
class LiveRuntimeConfig:
    lock_timeout_seconds: int = 60
    stale_lock_seconds: int = 43200
    duckdb_memory_limit: str = "2GB"
    duckdb_threads: int = 4
    duckdb_temp_directory: str = "data/duckdb/tmp"
    duckdb_max_temp_directory_size: str = "50GB"


@dataclass
class LiveRetentionConfig:
    keep_daily_artifacts_days: int = 180
    keep_failed_jobs_days: int = 90
    keep_latest_always: bool = True
    max_runs_per_day_per_sa: int = 5


@dataclass
class LiveAccountConfig:
    default_account_id: str | None = None
    account_snapshot_path: str | None = None
    position_path: str | None = None
    max_position_staleness_trade_days: int = 1
    stale_position_policy: str = "block"


@dataclass
class LiveOrdersConfig:
    enabled: bool = True
    required: bool = False
    require_single_superalpha: bool = True
    board_lot_size: int = 100
    min_order_value: float = 5000.0
    allow_fractional_shares: bool = False
    allow_odd_lot_sell: bool = False
    preserve_unsellable_positions: bool = True
    target_gross_exposure: float = 0.98


@dataclass
class LiveFeeConfig:
    buy_fee_bps: float = 2.5
    sell_fee_bps: float = 2.5
    stamp_tax_bps: float = 5.0
    min_commission: float = 5.0


@dataclass
class LiveParityConfig:
    enabled: bool = True
    strict: bool = False
    min_rank_corr: float = 0.98
    min_top_overlap: float = 0.80
    min_bottom_overlap: float = 0.80
    max_missing_ratio_delta: float = 0.02


@dataclass
class LiveConfig:
    universe: str = "cn_all"
    store_root: str | Path = Path("data/alpha_universe_store")
    duckdb_path: str | Path = Path("data/duckdb/market.duckdb")
    source_view: str = "v_project_panel_cn_a"
    schema_version: int = 1
    data: LiveDataConfig = field(default_factory=LiveDataConfig)
    superalpha: LiveSuperalphaConfig = field(default_factory=LiveSuperalphaConfig)
    portfolio: LivePortfolioConfig = field(default_factory=LivePortfolioConfig)
    tradability: LiveTradabilityConfig = field(default_factory=LiveTradabilityConfig)
    runtime: LiveRuntimeConfig = field(default_factory=LiveRuntimeConfig)
    retention: LiveRetentionConfig = field(default_factory=LiveRetentionConfig)
    account: LiveAccountConfig = field(default_factory=LiveAccountConfig)
    orders: LiveOrdersConfig = field(default_factory=LiveOrdersConfig)
    fee: LiveFeeConfig = field(default_factory=LiveFeeConfig)
    parity: LiveParityConfig = field(default_factory=LiveParityConfig)

    def __post_init__(self) -> None:
        self.store_root = Path(self.store_root)
        self.duckdb_path = Path(self.duckdb_path)


def load_live_config(path: str | Path | None = None, *, overrides: dict[str, Any] | None = None) -> LiveConfig:
    payload: dict[str, Any] = {}
    if path:
        p = Path(path)
        if p.exists():
            payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if overrides:
        payload = _deep_merge(payload, overrides)
    return _config_from_payload(payload)


def _config_from_payload(payload: dict[str, Any]) -> LiveConfig:
    cfg = LiveConfig(
        universe=str(payload.get("universe") or "cn_all"),
        store_root=payload.get("store_root") or Path("data/alpha_universe_store"),
        duckdb_path=(payload.get("data") or {}).get("duckdb_path")
        or payload.get("duckdb_path")
        or Path("data/duckdb/market.duckdb"),
        source_view=(payload.get("data") or {}).get("source_view")
        or payload.get("source_view")
        or "v_project_panel_cn_a",
        schema_version=int(payload.get("schema_version") or 1),
    )
    for section_name in (
        "data",
        "superalpha",
        "portfolio",
        "tradability",
        "runtime",
        "retention",
        "account",
        "orders",
        "fee",
        "parity",
    ):
        section = payload.get(section_name)
        target = getattr(cfg, section_name)
        if isinstance(section, dict):
            for key, value in section.items():
                if hasattr(target, key):
                    if key in {"critical_fields", "market_value_any_of"} and isinstance(value, list):
                        value = tuple(str(x) for x in value)
                    setattr(target, key, value)
    cfg.__post_init__()
    return cfg


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
