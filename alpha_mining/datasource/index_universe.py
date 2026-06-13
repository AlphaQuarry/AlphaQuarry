from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


SUPPORTED_INDEX_UNIVERSES: tuple[str, ...] = (
    "hs300",
    "csi500",
    "csi1000",
    "csi2000",
    "csi_all_share",
    "cnindex2000",
    "sme_composite",
)


@dataclass(frozen=True)
class IndexUniverseSpec:
    universe_name: str
    display_name: str
    candidate_codes: tuple[str, ...]
    candidate_symbols: tuple[str, ...]
    candidate_names: tuple[str, ...]
    required: bool = False
    enabled: bool = True


def normalize_index_code(raw: object) -> str:
    text = str(raw or "").strip().upper()
    if not text:
        return ""
    if text.endswith(".XSHG"):
        return f"{text[:-5]}.SH"
    if text.endswith(".XSHE"):
        return f"{text[:-5]}.SZ"
    return text


def load_index_universe_config(path: str | Path) -> dict[str, IndexUniverseSpec]:
    cfg_path = Path(path)
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    raw_specs = payload.get("index_universes", {})
    if not isinstance(raw_specs, dict):
        raise ValueError("index_universes config must contain a mapping")

    specs: dict[str, IndexUniverseSpec] = {}
    for raw_name, raw_spec in raw_specs.items():
        name = str(raw_name or "").strip().lower()
        if name not in SUPPORTED_INDEX_UNIVERSES:
            raise ValueError(f"Unsupported index universe in config: {raw_name}")
        if not isinstance(raw_spec, dict):
            raise ValueError(f"index universe '{name}' must be a mapping")
        specs[name] = IndexUniverseSpec(
            universe_name=name,
            display_name=str(raw_spec.get("display_name") or name),
            candidate_codes=tuple(
                code for code in (normalize_index_code(x) for x in raw_spec.get("candidate_codes", []) or []) if code
            ),
            candidate_symbols=tuple(_unique_non_empty(raw_spec.get("candidate_symbols", []) or [])),
            candidate_names=tuple(_unique_non_empty(raw_spec.get("candidate_names", []) or [])),
            required=bool(raw_spec.get("required", False)),
            enabled=bool(raw_spec.get("enabled", True)),
        )

    missing = [name for name in SUPPORTED_INDEX_UNIVERSES if name not in specs]
    if missing:
        raise ValueError(f"index_universes config missing supported universes: {','.join(missing)}")
    return {name: specs[name] for name in SUPPORTED_INDEX_UNIVERSES}


def resolve_index_universes(
    specs: dict[str, IndexUniverseSpec],
    index_basic_df: pd.DataFrame,
    universe_names: Iterable[str] | None = None,
    missing_policy: str = "warn",
    snapshot_date: str | None = None,
) -> pd.DataFrame:
    names = [str(x or "").strip().lower() for x in (universe_names or specs.keys()) if str(x or "").strip()]
    unknown = [name for name in names if name not in specs]
    if unknown:
        raise ValueError(f"Unsupported index universe names: {','.join(unknown)}")

    policy = str(missing_policy or "warn").strip().lower()
    if policy not in {"warn", "fail"}:
        raise ValueError("missing_policy must be 'warn' or 'fail'")

    basic = _prepare_index_basic(index_basic_df)
    snapshot_text = str(snapshot_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    resolved_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, object]] = []
    for name in names:
        spec = specs[name]
        matched = _match_index_basic(spec, basic)
        status = (
            "active" if matched is not None and bool(spec.enabled) else ("disabled" if not spec.enabled else "missing")
        )
        if status == "missing":
            message = f"index universe '{name}' not found in index_basic; skipping"
            if bool(spec.required) and policy == "fail":
                raise RuntimeError(message)
            if policy == "warn":
                warnings.warn(message, UserWarning, stacklevel=2)
        row = {
            "universe_name": name,
            "display_name": spec.display_name,
            "resolved_index_code": str(matched.get("code", "")) if matched is not None else "",
            "index_daily_code": str(matched.get("code", "")) if matched is not None else "",
            "index_weight_code": str(matched.get("code", "")) if matched is not None else "",
            "resolved_name": str(matched.get("name", "")) if matched is not None else "",
            "market": str(matched.get("market", "")) if matched is not None else "",
            "publisher": str(matched.get("publisher", "")) if matched is not None else "",
            "category": str(matched.get("category", "")) if matched is not None else "",
            "required": bool(spec.required),
            "enabled": bool(spec.enabled),
            "status": status,
            "candidate_codes_json": json.dumps(list(spec.candidate_codes), ensure_ascii=False),
            "candidate_symbols_json": json.dumps(list(spec.candidate_symbols), ensure_ascii=False),
            "candidate_names_json": json.dumps(list(spec.candidate_names), ensure_ascii=False),
            "resolved_at": resolved_at,
            "snapshot_date": snapshot_text,
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=_DIM_COLUMNS)


_DIM_COLUMNS = [
    "universe_name",
    "display_name",
    "resolved_index_code",
    "index_daily_code",
    "index_weight_code",
    "resolved_name",
    "market",
    "publisher",
    "category",
    "required",
    "enabled",
    "status",
    "candidate_codes_json",
    "candidate_symbols_json",
    "candidate_names_json",
    "resolved_at",
    "snapshot_date",
]


def _unique_non_empty(values: Iterable[object]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _prepare_index_basic(index_basic_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["code", "name", "market", "publisher", "category"]
    if index_basic_df is None or index_basic_df.empty:
        return pd.DataFrame(columns=[*cols, "symbol"])
    work = index_basic_df.copy()
    if "ts_code" in work.columns and "code" not in work.columns:
        work = work.rename(columns={"ts_code": "code"})
    if "code" not in work.columns:
        return pd.DataFrame(columns=[*cols, "symbol"])
    for col in cols:
        if col not in work.columns:
            work[col] = ""
        work[col] = work[col].fillna("").astype(str)
    work["code"] = work["code"].map(normalize_index_code)
    if "symbol" not in work.columns:
        work["symbol"] = work["code"].str.split(".", n=1).str[0]
    else:
        work["symbol"] = work["symbol"].fillna("").astype(str)
    return work[[*cols, "symbol"]].drop_duplicates(subset=["code"], keep="last").reset_index(drop=True)


def _match_index_basic(spec: IndexUniverseSpec, basic: pd.DataFrame) -> dict[str, object] | None:
    if not bool(spec.enabled) or basic.empty:
        return None
    for code in spec.candidate_codes:
        matched = basic[basic["code"] == normalize_index_code(code)]
        if not matched.empty:
            return dict(matched.iloc[0])
    for symbol in spec.candidate_symbols:
        matched = basic[basic["symbol"] == str(symbol)]
        if not matched.empty:
            return dict(matched.iloc[0])
    for name in spec.candidate_names:
        matched = basic[basic["name"] == str(name)]
        if not matched.empty:
            return dict(matched.iloc[0])
    return None
