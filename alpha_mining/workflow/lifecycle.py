from __future__ import annotations

import pandas as pd

from .universe_store import (
    append_alpha_lifecycle_records,
    load_alpha_lifecycle_registry,
    update_alpha_lifecycle_status,
)


LIFECYCLE_REGISTERED = "REGISTERED"
LIFECYCLE_MATERIALIZED = "MATERIALIZED"
LIFECYCLE_ANALYZED = "ANALYZED"
LIFECYCLE_PURGED = "PURGED"
LIFECYCLE_FAILED = "FAILED"
LIFECYCLE_REPRODUCED = "REPRODUCED"


def register_alpha_batch(
    expression_df: pd.DataFrame,
    base_dir: str,
    universe_name: str,
    simulation_config_json: str = "",
    input_manifest_id: str = "",
) -> pd.DataFrame:
    if expression_df is None or expression_df.empty:
        return pd.DataFrame()
    required = {"alpha_name", "expression"}
    if not required.issubset(expression_df.columns):
        raise ValueError("expression_df must include alpha_name/expression")

    payload = expression_df.copy()
    payload["alpha_name"] = payload["alpha_name"].astype(str)
    payload["expression"] = payload["expression"].astype(str)
    if "expression_hash" not in payload.columns:
        payload["expression_hash"] = ""
    if "source" not in payload.columns:
        payload["source"] = "unknown"
    payload["status"] = LIFECYCLE_REGISTERED
    payload["simulation_config_json"] = str(simulation_config_json or "")
    payload["input_manifest_id"] = str(input_manifest_id or "")
    return append_alpha_lifecycle_records(payload, base_dir=base_dir, universe_name=universe_name)


def mark_materialized(
    alpha_names: list[str],
    alpha_value_path: str,
    base_dir: str,
    universe_name: str,
) -> pd.DataFrame:
    return update_alpha_lifecycle_status(
        alpha_names=alpha_names,
        status=LIFECYCLE_MATERIALIZED,
        alpha_value_path=alpha_value_path,
        base_dir=base_dir,
        universe_name=universe_name,
    )


def mark_analyzed(
    alpha_names: list[str],
    analysis_run_id: str,
    base_dir: str,
    universe_name: str,
) -> pd.DataFrame:
    return update_alpha_lifecycle_status(
        alpha_names=alpha_names,
        status=LIFECYCLE_ANALYZED,
        analysis_run_id=analysis_run_id,
        base_dir=base_dir,
        universe_name=universe_name,
    )


def mark_purged(
    alpha_names: list[str],
    base_dir: str,
    universe_name: str,
) -> pd.DataFrame:
    return update_alpha_lifecycle_status(
        alpha_names=alpha_names,
        status=LIFECYCLE_PURGED,
        base_dir=base_dir,
        universe_name=universe_name,
    )


def mark_failed(
    alpha_names: list[str],
    error_message: str,
    base_dir: str,
    universe_name: str,
    status: str = LIFECYCLE_FAILED,
    failure_kind: str = "",
    last_error_stage: str = "",
) -> pd.DataFrame:
    return update_alpha_lifecycle_status(
        alpha_names=alpha_names,
        status=status,
        error_message=error_message,
        failure_kind=failure_kind,
        last_error_stage=last_error_stage,
        base_dir=base_dir,
        universe_name=universe_name,
    )


def mark_reproduced(
    alpha_names: list[str],
    base_dir: str,
    universe_name: str,
) -> pd.DataFrame:
    return update_alpha_lifecycle_status(
        alpha_names=alpha_names,
        status=LIFECYCLE_REPRODUCED,
        base_dir=base_dir,
        universe_name=universe_name,
    )


def load_lifecycle_registry(
    base_dir: str,
    universe_name: str,
) -> pd.DataFrame:
    return load_alpha_lifecycle_registry(base_dir=base_dir, universe_name=universe_name)
