from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from factor_research.utils import calculate_risk_metrics

from ..atomic_io import atomic_write_dataframe_csv, read_csv_with_backup
from .artifacts import load_saved_dataframe, save_dataframe_artifact
from .universe_store import get_universe_paths, load_universe_alpha_values

LOGGER = logging.getLogger("alpha_mining.factor_library")


@dataclass(frozen=True)
class FactorLibraryConfig:
    """因子入库配置。

    入库条件（全部满足）：
    1. effectiveness_score >= min_score
    2. 与已入库因子的信号相关性 < max_signal_corr
    3. 与已入库因子的 IC 相关性 < max_ic_corr
    4. 与已入库因子的 PnL 相关性 < max_pnl_corr

    阈值说明（经验值）：
    - min_score=60.0: effectiveness_score 的 60 分意味着因子在 4 个维度
      （预测力、纯多头表现、稳定性、可交易性）的加权得分达到 60%。
      历史数据表明，score >= 60 的因子在样本外的表现显著优于随机。
    - max_*_corr=0.80: 相关性阈值 0.80 对应约 64% 的方差解释度。
      两个相关性 > 0.80 的因子提供的信息高度重叠，同时入库会降低
      组合的分散化效果。此阈值参考了 QLib 和 vibe-trading 的实践。
    - staging_min_score=50.0: 暂存区的最低分数，用于观察边缘因子。
    - staging_max_corr=0.95: 暂存区的相关性阈值，比正式入库更宽松。
    - sharpe_override_threshold=0.15: 当候选因子的 Sharpe 比最接近的
      已入库因子高出此阈值时，即使相关性超标也可入库（择优替换）。
    """

    enabled: bool = False
    min_score: float = 60.0
    staging_min_score: float = 50.0
    max_signal_corr: float = 0.80
    max_ic_corr: float = 0.80
    max_pnl_corr: float = 0.80
    staging_max_corr: float = 0.95
    transaction_cost_enabled: bool = True
    sharpe_override_enabled: bool = True
    sharpe_override_threshold: float = 0.15
    high_corr_threshold: float = 0.80
    manual_submit_enabled: bool = True
    compare_return_basis: str = "effective"
    max_compare_factors: int = 5000


REGISTRY_COLUMNS = [
    "schema_version",
    "universe",
    "factor",
    "expression",
    "expression_hash",
    "analysis_run_id",
    "status",
    "score",
    "score_basis",
    "acceptance_mode",
    "submitted_by",
    "submitted_at_utc",
    "checked_at_utc",
    "library_status_reason",
    "rejection_reason",
    "reject_reasons",
    "signal_corr",
    "ic_corr",
    "long_only_corr",
    "long_short_corr",
    "max_signal_corr",
    "max_ic_corr",
    "max_pnl_corr",
    "max_any_corr",
    "nearest_factor_id",
    "nearest_expression_hash",
    "high_corr_peer_count",
    "high_corr_peer_factors",
    "candidate_long_only_sharpe",
    "candidate_long_short_sharpe",
    "max_peer_long_only_sharpe",
    "max_peer_long_short_sharpe",
    "long_only_sharpe_delta",
    "long_short_sharpe_delta",
    "override_threshold",
    "override_portfolio",
    "override_reason",
    "signal_artifact_path",
    "ic_artifact_path",
    "pnl_artifact_path",
    "direction_policy",
    "direction_sign",
    "best_layer_direction_train_locked",
    "source_score_total_net",
    "source_feedback_score_net",
    "source_score_total_gross",
    "source_feedback_score_gross",
]


def check_factor_library_candidate(
    *,
    base_dir: str | Path,
    universe_name: str,
    run_id: str,
    factor: str,
    factor_metrics_df: pd.DataFrame,
    ic_df: pd.DataFrame | None = None,
    portfolio_pnl_df: pd.DataFrame | None = None,
    signal_df: pd.DataFrame | None = None,
    config: FactorLibraryConfig | None = None,
) -> dict[str, Any]:
    """Check if a factor qualifies for factor library admission.

    Evaluates a factor against multiple quality gates:
    - Effectiveness score threshold
    - Signal correlation with existing library factors
    - IC correlation with existing library factors
    - PnL correlation with existing library factors
    - Sharpe ratio override mechanism

    Args:
        base_dir: Base directory for universe storage.
        universe_name: Target universe name.
        run_id: Analysis run identifier.
        factor: Factor name to check.
        factor_metrics_df: DataFrame with factor performance metrics.
        ic_df: DataFrame with IC time series (optional).
        portfolio_pnl_df: DataFrame with portfolio PnL (optional).
        signal_df: DataFrame with factor signal values (optional).
        config: FactorLibraryConfig with admission thresholds.

    Returns:
        Dictionary with keys:
        - 'status': 'accepted', 'staging', or 'rejected'
        - 'decision': 'pass', 'staging', or 'reject'
        - 'can_submit': Whether factor can be submitted
        - 'score': Effectiveness score
        - 'correlations': Correlation summary
        - 'override': Sharpe override info (if applicable)

    Example:
        >>> result = check_factor_library_candidate(
        ...     base_dir='data/alpha_universe_store',
        ...     universe_name='cn_all',
        ...     run_id='run_20240101',
        ...     factor='momentum_20d',
        ...     factor_metrics_df=metrics_df,
        ... )
        >>> print(result['status'])
    """
    cfg = config or FactorLibraryConfig(enabled=True)
    if not bool(cfg.enabled):
        return _check_payload(
            factor=factor,
            run_id=run_id,
            universe_name=universe_name,
            status="disabled",
            decision="reject",
            can_submit=False,
            reason="factor_library_disabled",
        )
    candidate = _candidate_row(factor_metrics_df, factor)
    if candidate is None:
        return _check_payload(
            factor=factor,
            run_id=run_id,
            universe_name=universe_name,
            status="missing",
            decision="reject",
            can_submit=False,
            reason="factor_metrics_missing",
        )

    registry = load_factor_library_registry(base_dir=base_dir, universe_name=universe_name)
    corr = compute_single_factor_correlations_against_library(
        base_dir=base_dir,
        universe_name=universe_name,
        factor=factor,
        candidate_signal=_factor_signal_frame(signal_df, factor),
        candidate_ic=_factor_ic_frame(ic_df, factor),
        candidate_pnl=_factor_pnl_frame(portfolio_pnl_df, factor),
        accepted_registry=registry,
        config=cfg,
    )
    corr_summary = dict(corr.get("summary") or _empty_corr_payload(has_existing=False))
    has_existing_for_corr = bool(corr_summary.pop("_has_existing_factors", False))
    score = _score_value(candidate, transaction_cost_enabled=cfg.transaction_cost_enabled)
    reject_reasons = _library_reject_reasons(score=score, corr=corr_summary, cfg=cfg)
    status = _library_status(score=score, corr=corr_summary, cfg=cfg)
    missing_reason = _missing_corr_reason(corr_summary) if has_existing_for_corr else ""
    if missing_reason and status == "accepted":
        status = "staging"
    override = _sharpe_override_payload(
        score=score,
        corr_summary=corr_summary,
        peer_rows=list(corr.get("peers") or []),
        candidate_long_only_sharpe=_float_or_nan(corr.get("candidate_long_only_sharpe")),
        candidate_long_short_sharpe=_float_or_nan(corr.get("candidate_long_short_sharpe")),
        cfg=cfg,
    )
    acceptance_mode = "standard"
    if reject_reasons and bool(override.get("triggered")):
        status = "accepted"
        acceptance_mode = "sharpe_override"
        reject_reasons = []
    elif status != "accepted":
        acceptance_mode = ""

    reason = ";".join([x for x in [";".join(reject_reasons), missing_reason] if x])
    if status == "accepted":
        decision = "pass_with_override" if acceptance_mode == "sharpe_override" else "pass"
    elif status == "staging":
        decision = "staging"
    else:
        decision = "reject"

    row = _registry_row_from_check(
        universe_name=universe_name,
        run_id=run_id,
        factor=factor,
        candidate=candidate,
        status=status,
        decision=decision,
        score=score,
        corr=corr_summary,
        reject_reasons=reject_reasons,
        reason=reason,
        acceptance_mode=acceptance_mode,
        override=override,
        submitted_by="",
        artifact_paths={},
    )
    return {
        "status": "ok",
        "factor": str(factor),
        "universe": str(universe_name),
        "analysis_run_id": str(run_id or ""),
        "decision": decision,
        "can_submit": bool(status == "accepted" and cfg.manual_submit_enabled),
        "score": _json_float(score),
        "score_basis": _score_basis(candidate, transaction_cost_enabled=cfg.transaction_cost_enabled),
        "acceptance_mode": acceptance_mode or None,
        "reason": reason,
        "corr": {
            k: _json_float(v) if isinstance(v, (float, int, np.floating, np.integer)) else v
            for k, v in corr_summary.items()
            if k != "_has_existing_factors"
        },
        "override": override,
        "high_corr_peers": list(corr.get("high_corr_peers") or []),
        "peer_details": list(corr.get("peers") or []),
        "row": row,
    }


def submit_factor_library_candidate(
    *,
    base_dir: str | Path,
    universe_name: str,
    run_id: str,
    factor: str,
    factor_metrics_df: pd.DataFrame,
    ic_df: pd.DataFrame | None = None,
    portfolio_pnl_df: pd.DataFrame | None = None,
    signal_df: pd.DataFrame | None = None,
    config: FactorLibraryConfig | None = None,
    submitted_by: str = "dashboard",
) -> dict[str, Any]:
    cfg = config or FactorLibraryConfig(enabled=True)
    check = check_factor_library_candidate(
        base_dir=base_dir,
        universe_name=universe_name,
        run_id=run_id,
        factor=factor,
        factor_metrics_df=factor_metrics_df,
        ic_df=ic_df,
        portfolio_pnl_df=portfolio_pnl_df,
        signal_df=signal_df,
        config=cfg,
    )
    if not bool(check.get("can_submit")):
        return {
            "status": "blocked",
            "submitted": False,
            "factor": str(factor),
            "library_status": str(check.get("row", {}).get("status") or "rejected"),
            "decision": str(check.get("decision") or "reject"),
            "reason": str(check.get("reason") or ""),
            "check": check,
        }

    paths = _library_paths(base_dir=base_dir, universe_name=universe_name)
    paths["root"].mkdir(parents=True, exist_ok=True)

    # Attempt reproduce fallback if signal_df is missing
    reproduced_signal_df = None
    reproduce_info: dict[str, Any] = {}
    if signal_df is None or signal_df.empty:
        try:
            from .reproduce import reproduce_alpha_by_name

            result = reproduce_alpha_by_name(
                alpha_name=factor,
                base_dir=base_dir,
                universe_name=universe_name,
                compare_with_saved=False,
                mark_lifecycle=False,
            )
            output_df = result.get("output_df")
            if output_df is not None and not output_df.empty:
                reproduced_signal_df = output_df
                reproduce_info = {
                    "signal_source": "reproduced",
                    "strict_reproducibility": bool(result.get("strict_reproducibility", False)),
                    "reproduce_source_mode": str(result.get("reproduce_source_mode", "unknown")),
                    "reproduce_warning": str(result.get("reproduce_warning", "")),
                }
                LOGGER.info("submit_factor_library_candidate: reproduced signal for %s", factor)
        except Exception as exc:
            LOGGER.debug(
                "submit_factor_library_candidate: reproduce failed for %s: %s",
                factor,
                exc,
            )

    # Use reproduced signal if available
    effective_signal_df = signal_df
    if reproduced_signal_df is not None and not reproduced_signal_df.empty:
        effective_signal_df = reproduced_signal_df

    artifact_paths = _save_factor_library_series(
        paths=paths,
        factor=factor,
        signal_df=effective_signal_df,
        ic_df=ic_df,
        portfolio_pnl_df=portfolio_pnl_df,
    )
    row = dict(check["row"])
    row.update(artifact_paths)
    row.update(reproduce_info)
    if not str(row.get("signal_artifact_path") or "").strip():
        row["status"] = "staging"
        row["library_status_reason"] = _join_reason(row.get("library_status_reason"), "missing_signal_artifact")
        row["rejection_reason"] = _join_reason(row.get("rejection_reason"), "missing_signal_artifact")
        row["submitted_by"] = str(submitted_by or "dashboard")
        row["submitted_at_utc"] = _utc_now()
        row["schema_version"] = 2
        registry = load_factor_library_registry(base_dir=base_dir, universe_name=universe_name)
        additions = pd.DataFrame([row])
        merged = pd.concat([registry, additions], ignore_index=True) if not registry.empty else additions
        merged = _normalize_registry(merged, universe_name=universe_name)
        merged = merged.drop_duplicates(subset=["factor", "analysis_run_id"], keep="last")
        atomic_write_dataframe_csv(paths["registry"], merged, index=False, backup=True)
        return {
            "status": "blocked",
            "submitted": False,
            "factor": str(factor),
            "library_status": "staging",
            "acceptance_mode": row.get("acceptance_mode") or "standard",
            "registry_path": str(paths["registry"].as_posix()),
            "row": row,
            "check": check,
            "reason": "missing_signal_artifact",
        }
    row["submitted_by"] = str(submitted_by or "dashboard")
    row["submitted_at_utc"] = _utc_now()
    row["status"] = "accepted"
    row["schema_version"] = 2
    registry = load_factor_library_registry(base_dir=base_dir, universe_name=universe_name)
    additions = pd.DataFrame([row])
    merged = pd.concat([registry, additions], ignore_index=True) if not registry.empty else additions
    merged = _normalize_registry(merged, universe_name=universe_name)
    merged = merged.drop_duplicates(subset=["factor", "analysis_run_id"], keep="last")
    atomic_write_dataframe_csv(paths["registry"], merged, index=False, backup=True)
    return {
        "status": "ok",
        "submitted": True,
        "factor": str(factor),
        "library_status": "accepted",
        "acceptance_mode": row.get("acceptance_mode") or "standard",
        "registry_path": str(paths["registry"].as_posix()),
        "row": row,
        "check": check,
    }


def submit_factor_library_candidates(
    *,
    base_dir: str | Path,
    universe_name: str,
    run_id: str,
    factor_metrics_df: pd.DataFrame,
    ic_df: pd.DataFrame | None = None,
    portfolio_pnl_df: pd.DataFrame | None = None,
    signal_df: pd.DataFrame | None = None,
    config: FactorLibraryConfig | None = None,
) -> dict[str, Any]:
    cfg = config or FactorLibraryConfig()
    if not bool(cfg.enabled):
        return {
            "enabled": False,
            "candidate_count": 0,
            "accepted_count": 0,
            "registry_path": "",
            "accepted_factors": [],
            "rows": [],
        }
    paths = _library_paths(base_dir=base_dir, universe_name=universe_name)
    paths["root"].mkdir(parents=True, exist_ok=True)
    registry = load_factor_library_registry(base_dir=base_dir, universe_name=universe_name)
    correlation_registry = registry.copy()
    candidates = _base_score_candidates(
        factor_metrics_df,
        min_score=cfg.staging_min_score,
        transaction_cost_enabled=cfg.transaction_cost_enabled,
    )
    rows: list[dict[str, Any]] = []
    accepted: list[str] = []
    for _, candidate in candidates.iterrows():
        factor = str(candidate.get("factor") or candidate.get("alpha_name") or "")
        if not factor:
            continue
        corr = _max_existing_correlations(
            factor=factor,
            registry=correlation_registry,
            signal_df=signal_df,
            ic_df=ic_df,
            portfolio_pnl_df=portfolio_pnl_df,
        )
        has_existing_for_corr = bool(corr.pop("_has_existing_factors", False))
        score = _score_value(candidate, transaction_cost_enabled=cfg.transaction_cost_enabled)
        reject_reasons = _library_reject_reasons(score=score, corr=corr, cfg=cfg)
        status = _library_status(score=score, corr=corr, cfg=cfg)
        status_reason = ";".join(reject_reasons)
        missing_reason = _missing_corr_reason(corr) if has_existing_for_corr else ""
        if missing_reason:
            if status == "accepted":
                status = "staging"
            if status != "accepted":
                status_reason = ";".join([x for x in [status_reason, missing_reason] if x])
            LOGGER.warning(
                "[factor_library] missing correlation series for factor=%s reason=%s run_id=%s",
                factor,
                missing_reason,
                str(run_id or ""),
            )
        row = {
            "factor": factor,
            "analysis_run_id": str(run_id or ""),
            "status": status,
            "score": score,
            "score_basis": _score_basis(candidate, transaction_cost_enabled=cfg.transaction_cost_enabled),
            "reject_reasons": ";".join(reject_reasons),
            "rejection_reason": ";".join(reject_reasons),
            "library_status_reason": status_reason,
            **corr,
        }
        if status == "accepted":
            accepted.append(factor)
            row.update(
                _save_factor_library_series(
                    paths=paths,
                    factor=factor,
                    signal_df=signal_df,
                    ic_df=ic_df,
                    portfolio_pnl_df=portfolio_pnl_df,
                )
            )
            correlation_registry = pd.concat(
                [correlation_registry, pd.DataFrame([{**row, "status": "accepted"}])],
                ignore_index=True,
            )
        rows.append(row)
    additions = pd.DataFrame(rows)
    if not additions.empty:
        merged = pd.concat([registry, additions], ignore_index=True)
        merged = merged.drop_duplicates(subset=["factor", "analysis_run_id"], keep="last")
        atomic_write_dataframe_csv(paths["registry"], merged, index=False, backup=True)
    return {
        "enabled": bool(cfg.enabled),
        "candidate_count": int(len(candidates)),
        "accepted_count": int(len(accepted)),
        "registry_path": str(paths["registry"].as_posix()),
        "accepted_factors": accepted,
        "rows": rows,
    }


def load_factor_library_registry(*, base_dir: str | Path, universe_name: str) -> pd.DataFrame:
    path = _library_paths(base_dir=base_dir, universe_name=universe_name)["registry"]
    if not path.exists():
        return pd.DataFrame(columns=REGISTRY_COLUMNS)
    try:
        return _normalize_registry(read_csv_with_backup(path), universe_name=universe_name)
    except Exception:
        return pd.DataFrame(columns=REGISTRY_COLUMNS)


def compute_single_factor_correlations_against_library(
    *,
    base_dir: str | Path,
    universe_name: str,
    factor: str,
    candidate_signal: pd.DataFrame | None,
    candidate_ic: pd.DataFrame | None,
    candidate_pnl: pd.DataFrame | None,
    accepted_registry: pd.DataFrame,
    config: FactorLibraryConfig,
) -> dict[str, Any]:
    registry = _normalize_registry(accepted_registry, universe_name=universe_name)
    accepted = registry[registry.get("status", pd.Series(dtype=str)).fillna("").astype(str) == "accepted"].copy()
    if accepted.empty:
        return {
            "summary": _empty_corr_payload(has_existing=False),
            "peers": [],
            "high_corr_peers": [],
        }
    accepted = _limit_compare_factors(accepted, config.max_compare_factors)
    cand_signal = _signal_series(candidate_signal, factor)
    if cand_signal.empty:
        cand_signal = _signal_series(_load_alpha_signal(base_dir, universe_name, factor), factor)
    cand_ic = _ic_series(candidate_ic, factor)
    cand_pnl = _pnl_frame_for_series(candidate_pnl)
    peers: list[dict[str, Any]] = []
    for _, peer in accepted.iterrows():
        peer_factor = str(peer.get("factor") or "")
        if not peer_factor or peer_factor == str(factor):
            continue
        peer_signal = _signal_series(_load_artifact_or_empty(peer.get("signal_artifact_path")), peer_factor)
        if peer_signal.empty:
            peer_signal = _signal_series(_load_alpha_signal(base_dir, universe_name, peer_factor), peer_factor)
        peer_ic = _ic_series(_load_artifact_or_empty(peer.get("ic_artifact_path")), peer_factor)
        peer_pnl = _pnl_frame_for_series(_load_artifact_or_empty(peer.get("pnl_artifact_path")))

        signal_corr = _aligned_corr(cand_signal, peer_signal, keys=["date", "code"])
        ic_corr = _aligned_corr(cand_ic, peer_ic, keys=["trade_date"])
        long_only_corr = _pnl_corr(cand_pnl, peer_pnl, portfolio="long_only", basis=config.compare_return_basis)
        long_short_corr = _pnl_corr(cand_pnl, peer_pnl, portfolio="long_short", basis="gross")
        pnl_values = [x for x in [long_only_corr, long_short_corr] if np.isfinite(x)]
        max_pnl_corr = max(pnl_values) if pnl_values else np.nan
        any_values = [x for x in [signal_corr, ic_corr, max_pnl_corr] if np.isfinite(x)]
        max_any_corr = max(any_values) if any_values else np.nan
        peer_row = {
            "peer_factor": peer_factor,
            "signal_corr": _json_float(signal_corr),
            "ic_corr": _json_float(ic_corr),
            "long_only_corr": _json_float(long_only_corr),
            "long_short_corr": _json_float(long_short_corr),
            "max_pnl_corr": _json_float(max_pnl_corr),
            "max_any_corr": _json_float(max_any_corr),
            "peer_long_only_sharpe": _json_float(
                _portfolio_sharpe_from_pnl_frame(peer_pnl, "long_only", basis=config.compare_return_basis)
            ),
            "peer_long_short_sharpe": _json_float(
                _portfolio_sharpe_from_pnl_frame(peer_pnl, "long_short", basis="gross")
            ),
        }
        peers.append(peer_row)

    summary = _summary_from_peer_rows(peers)
    summary["_has_existing_factors"] = True
    high_corr = [
        row
        for row in peers
        if any(
            np.isfinite(_float_or_nan(row.get(key)))
            and _float_or_nan(row.get(key)) >= float(config.high_corr_threshold)
            for key in ["signal_corr", "ic_corr", "max_pnl_corr"]
        )
    ]
    peers = sorted(peers, key=lambda row: _float_or_nan(row.get("max_any_corr")), reverse=True)
    return {
        "summary": summary,
        "peers": peers[:25],
        "high_corr_peers": high_corr[:25],
        "candidate_long_only_sharpe": _json_float(
            _portfolio_sharpe_from_pnl_frame(cand_pnl, "long_only", basis=config.compare_return_basis)
        ),
        "candidate_long_short_sharpe": _json_float(
            _portfolio_sharpe_from_pnl_frame(cand_pnl, "long_short", basis="gross")
        ),
    }


def _candidate_row(frame: pd.DataFrame | None, factor: str) -> pd.Series | None:
    if frame is None or frame.empty:
        return None
    work = frame.copy()
    if "factor" not in work.columns and "alpha_name" in work.columns:
        work["factor"] = work["alpha_name"]
    if "factor" not in work.columns:
        return None
    rows = work[work["factor"].astype(str) == str(factor)]
    if rows.empty:
        return None
    return rows.iloc[0]


def _join_reason(existing: Any, reason: str) -> str:
    parts = [
        str(x).strip()
        for x in [existing, reason]
        if str(x or "").strip() and str(x).strip().lower() not in {"nan", "none", "<na>"}
    ]
    deduped: list[str] = []
    for part in parts:
        if part not in deduped:
            deduped.append(part)
    return ";".join(deduped)


def _normalize_registry(frame: pd.DataFrame | None, *, universe_name: str) -> pd.DataFrame:
    work = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    for col in REGISTRY_COLUMNS:
        if col not in work.columns:
            work[col] = pd.NA
    if not work.empty:
        work["universe"] = work["universe"].fillna(str(universe_name))
        work.loc[work["universe"].astype(str).str.strip() == "", "universe"] = str(universe_name)
        if "status" in work.columns:
            work["status"] = work["status"].fillna("").astype(str)
    return work


def _limit_compare_factors(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    max_items = max(1, int(limit or 5000))
    if len(frame) <= max_items:
        return frame
    work = frame.copy()
    if "submitted_at_utc" in work.columns:
        work["_submitted_sort"] = pd.to_datetime(work["submitted_at_utc"], errors="coerce")
    else:
        work["_submitted_sort"] = pd.NaT
    work["_score_sort"] = pd.to_numeric(work.get("score", pd.Series(np.nan, index=work.index)), errors="coerce")
    return work.sort_values(["_submitted_sort", "_score_sort"], ascending=[False, False], na_position="last").head(
        max_items
    )


def _load_artifact_or_empty(path_value: Any) -> pd.DataFrame:
    text = str(path_value or "").strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return pd.DataFrame()
    try:
        return load_saved_dataframe(Path(text))
    except Exception:
        return pd.DataFrame()


def _load_alpha_signal(base_dir: str | Path, universe_name: str, factor: str) -> pd.DataFrame:
    try:
        return load_universe_alpha_values(alpha_name=factor, base_dir=base_dir, universe_name=universe_name)
    except Exception:
        return pd.DataFrame()


def _signal_series(frame: pd.DataFrame | None, factor: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["date", "code", "value"])
    work = frame.copy()
    if factor not in work.columns:
        value_cols = [c for c in work.columns if c not in {"date", "trade_date", "code", "znz_code"}]
        if len(value_cols) == 1:
            work = work.rename(columns={value_cols[0]: factor})
    if factor not in work.columns:
        return pd.DataFrame(columns=["date", "code", "value"])
    date_col = "date" if "date" in work.columns else "trade_date" if "trade_date" in work.columns else ""
    code_col = "code" if "code" in work.columns else "znz_code" if "znz_code" in work.columns else ""
    if not date_col or not code_col:
        return pd.DataFrame(columns=["date", "code", "value"])
    out = work[[date_col, code_col, factor]].copy()
    out.columns = ["date", "code", "value"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["code"] = out["code"].astype(str)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["date", "code", "value"])


def _ic_series(frame: pd.DataFrame | None, factor: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["trade_date", "value"])
    col = f"{factor}_ic"
    work = frame.copy()
    if col not in work.columns:
        value_cols = [c for c in work.columns if c not in {"trade_date", "date"}]
        if len(value_cols) == 1:
            work = work.rename(columns={value_cols[0]: col})
    if col not in work.columns:
        return pd.DataFrame(columns=["trade_date", "value"])
    date_col = "trade_date" if "trade_date" in work.columns else "date" if "date" in work.columns else ""
    if not date_col:
        return pd.DataFrame(columns=["trade_date", "value"])
    out = work[[date_col, col]].copy()
    out.columns = ["trade_date", "value"]
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["trade_date", "value"])


def _pnl_frame_for_series(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty or "trade_date" not in frame.columns:
        return pd.DataFrame()
    work = frame.copy()
    if "portfolio" not in work.columns:
        return pd.DataFrame()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    for col in ["return", "return_gross", "return_net"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    if "has_net_pnl" in work.columns:
        work["has_net_pnl"] = work["has_net_pnl"].fillna(False).astype(bool)
    else:
        work["has_net_pnl"] = False
    return work.dropna(subset=["trade_date"])


def _aligned_corr(left: pd.DataFrame, right: pd.DataFrame, *, keys: list[str]) -> float:
    if (
        left.empty
        or right.empty
        or not set(keys + ["value"]).issubset(left.columns)
        or not set(keys + ["value"]).issubset(right.columns)
    ):
        return np.nan
    merged = left.merge(right, on=keys, how="inner", suffixes=("_left", "_right"))
    if len(merged) < 2:
        return np.nan
    corr = pd.to_numeric(merged["value_left"], errors="coerce").corr(
        pd.to_numeric(merged["value_right"], errors="coerce"),
        method="spearman",
    )
    return abs(float(corr)) if pd.notna(corr) else np.nan


def _effective_return(frame: pd.DataFrame, *, basis: str) -> pd.Series:
    basis_text = str(basis or "effective").lower()
    if basis_text == "net" and "return_net" in frame.columns:
        return pd.to_numeric(frame["return_net"], errors="coerce")
    if basis_text == "gross":
        return pd.to_numeric(
            frame["return_gross"] if "return_gross" in frame.columns else frame.get("return"),
            errors="coerce",
        )
    if "return_net" in frame.columns and "has_net_pnl" in frame.columns:
        out = pd.to_numeric(
            frame["return_gross"] if "return_gross" in frame.columns else frame.get("return"),
            errors="coerce",
        )
        mask = frame["has_net_pnl"].fillna(False).astype(bool)
        out = out.copy()
        out.loc[mask] = pd.to_numeric(frame.loc[mask, "return_net"], errors="coerce")
        return out
    return pd.to_numeric(
        frame["return_gross"] if "return_gross" in frame.columns else frame.get("return"),
        errors="coerce",
    )


def _pnl_corr(left: pd.DataFrame, right: pd.DataFrame, *, portfolio: str, basis: str) -> float:
    if left.empty or right.empty:
        return np.nan
    lpart = left[left["portfolio"].astype(str) == str(portfolio)].copy()
    rpart = right[right["portfolio"].astype(str) == str(portfolio)].copy()
    if lpart.empty or rpart.empty:
        return np.nan
    lpart["value"] = _effective_return(lpart, basis=basis)
    rpart["value"] = _effective_return(rpart, basis=basis)
    return _aligned_corr(
        lpart[["trade_date", "value"]],
        rpart[["trade_date", "value"]],
        keys=["trade_date"],
    )


def _portfolio_sharpe_from_pnl_frame(
    frame: pd.DataFrame, portfolio: str, period: int = 1, basis: str = "effective"
) -> float:
    if frame is None or frame.empty or "portfolio" not in frame.columns:
        return np.nan
    part = frame[frame["portfolio"].astype(str) == str(portfolio)].copy()
    if part.empty:
        return np.nan
    returns = _effective_return(part, basis=basis)
    try:
        return float(calculate_risk_metrics(returns, period=period).get("sharpe_ratio", np.nan))
    except Exception:
        return np.nan


def _summary_from_peer_rows(peers: list[dict[str, Any]]) -> dict[str, Any]:
    if not peers:
        return _empty_corr_payload(has_existing=True)

    def best(key: str) -> tuple[float, str]:
        value = np.nan
        factor = ""
        for row in peers:
            current = _float_or_nan(row.get(key))
            if np.isfinite(current) and (not np.isfinite(value) or current > value):
                value = current
                factor = str(row.get("peer_factor") or "")
        return value, factor

    signal_corr, signal_peer = best("signal_corr")
    ic_corr, ic_peer = best("ic_corr")
    long_only_corr, lo_peer = best("long_only_corr")
    long_short_corr, ls_peer = best("long_short_corr")
    max_pnl_corr, pnl_peer = best("max_pnl_corr")
    max_any_corr, nearest = best("max_any_corr")
    if not nearest:
        nearest = _nearest_factor(
            [
                (signal_corr, signal_peer),
                (ic_corr, ic_peer),
                (long_only_corr, lo_peer),
                (long_short_corr, ls_peer),
                (max_pnl_corr, pnl_peer),
            ]
        )
    return {
        "signal_corr": signal_corr,
        "ic_corr": ic_corr,
        "long_only_corr": long_only_corr,
        "long_short_corr": long_short_corr,
        "max_signal_corr": signal_corr,
        "max_ic_corr": ic_corr,
        "max_pnl_corr": max_pnl_corr,
        "max_any_corr": max_any_corr,
        "nearest_factor_id": nearest,
        "nearest_expression_hash": "",
        "_has_existing_factors": True,
    }


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out if np.isfinite(out) else np.nan


def _json_float(value: Any) -> float | None:
    numeric = _float_or_nan(value)
    return float(numeric) if np.isfinite(numeric) else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_payload(
    *,
    factor: str,
    run_id: str,
    universe_name: str,
    status: str,
    decision: str,
    can_submit: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "factor": str(factor),
        "universe": str(universe_name),
        "analysis_run_id": str(run_id or ""),
        "decision": decision,
        "can_submit": bool(can_submit),
        "score": None,
        "score_basis": None,
        "acceptance_mode": None,
        "reason": reason,
        "corr": {},
        "override": {},
        "high_corr_peers": [],
        "peer_details": [],
        "row": {
            "factor": str(factor),
            "analysis_run_id": str(run_id or ""),
            "status": "rejected",
            "library_status_reason": reason,
            "rejection_reason": reason,
        },
    }


def _sharpe_override_payload(
    *,
    score: float,
    corr_summary: dict[str, Any],
    peer_rows: list[dict[str, Any]],
    candidate_long_only_sharpe: float,
    candidate_long_short_sharpe: float,
    cfg: FactorLibraryConfig,
) -> dict[str, Any]:
    high_peers = [
        row
        for row in peer_rows
        if any(
            np.isfinite(_float_or_nan(row.get(key))) and _float_or_nan(row.get(key)) >= float(cfg.high_corr_threshold)
            for key in ["signal_corr", "ic_corr", "max_pnl_corr"]
        )
    ]
    max_peer_lo = max(
        [
            _float_or_nan(row.get("peer_long_only_sharpe"))
            for row in high_peers
            if np.isfinite(_float_or_nan(row.get("peer_long_only_sharpe")))
        ]
        or [np.nan]
    )
    max_peer_ls = max(
        [
            _float_or_nan(row.get("peer_long_short_sharpe"))
            for row in high_peers
            if np.isfinite(_float_or_nan(row.get("peer_long_short_sharpe")))
        ]
        or [np.nan]
    )
    lo_delta = (
        candidate_long_only_sharpe - max_peer_lo
        if np.isfinite(candidate_long_only_sharpe) and np.isfinite(max_peer_lo)
        else np.nan
    )
    ls_delta = (
        candidate_long_short_sharpe - max_peer_ls
        if np.isfinite(candidate_long_short_sharpe) and np.isfinite(max_peer_ls)
        else np.nan
    )
    portfolios: list[str] = []
    if np.isfinite(lo_delta) and lo_delta >= float(cfg.sharpe_override_threshold):
        portfolios.append("long_only")
    if np.isfinite(ls_delta) and ls_delta >= float(cfg.sharpe_override_threshold):
        portfolios.append("long_short")
    triggered = bool(
        cfg.sharpe_override_enabled
        and high_peers
        and portfolios
        and np.isfinite(score)
        and score >= float(cfg.min_score)
    )
    portfolio = "both" if len(portfolios) == 2 else portfolios[0] if portfolios else None
    return {
        "enabled": bool(cfg.sharpe_override_enabled),
        "triggered": triggered,
        "threshold": float(cfg.sharpe_override_threshold),
        "portfolio": portfolio,
        "override_portfolio": portfolio,
        "override_reason": "high_corr_but_sharpe_improved" if triggered else "",
        "candidate_long_only_sharpe": _json_float(candidate_long_only_sharpe),
        "candidate_long_short_sharpe": _json_float(candidate_long_short_sharpe),
        "max_peer_long_only_sharpe": _json_float(max_peer_lo),
        "max_peer_long_short_sharpe": _json_float(max_peer_ls),
        "long_only_sharpe_delta": _json_float(lo_delta),
        "long_short_sharpe_delta": _json_float(ls_delta),
        "high_corr_peer_count": int(len(high_peers)),
        "high_corr_peer_factors": ",".join(
            str(row.get("peer_factor") or "") for row in high_peers if row.get("peer_factor")
        ),
    }


def _registry_row_from_check(
    *,
    universe_name: str,
    run_id: str,
    factor: str,
    candidate: pd.Series,
    status: str,
    decision: str,
    score: float,
    corr: dict[str, Any],
    reject_reasons: list[str],
    reason: str,
    acceptance_mode: str,
    override: dict[str, Any],
    submitted_by: str,
    artifact_paths: dict[str, str],
) -> dict[str, Any]:
    row = {
        "schema_version": 2,
        "universe": str(universe_name),
        "factor": str(factor),
        "expression": str(candidate.get("expression", "") or ""),
        "expression_hash": str(candidate.get("expression_hash", "") or ""),
        "analysis_run_id": str(run_id or ""),
        "status": str(status),
        "score": _json_float(score),
        "score_basis": _score_basis(candidate, transaction_cost_enabled=True),
        "acceptance_mode": acceptance_mode or "",
        "submitted_by": submitted_by,
        "submitted_at_utc": "",
        "checked_at_utc": _utc_now(),
        "library_status_reason": reason,
        "rejection_reason": ";".join(reject_reasons),
        "reject_reasons": ";".join(reject_reasons),
        "signal_corr": _json_float(corr.get("signal_corr")),
        "ic_corr": _json_float(corr.get("ic_corr")),
        "long_only_corr": _json_float(corr.get("long_only_corr")),
        "long_short_corr": _json_float(corr.get("long_short_corr")),
        "max_signal_corr": _json_float(corr.get("max_signal_corr")),
        "max_ic_corr": _json_float(corr.get("max_ic_corr")),
        "max_pnl_corr": _json_float(corr.get("max_pnl_corr")),
        "max_any_corr": _json_float(corr.get("max_any_corr")),
        "nearest_factor_id": str(corr.get("nearest_factor_id") or ""),
        "nearest_expression_hash": str(corr.get("nearest_expression_hash") or ""),
        "high_corr_peer_count": int(override.get("high_corr_peer_count") or 0),
        "high_corr_peer_factors": str(override.get("high_corr_peer_factors") or ""),
        "candidate_long_only_sharpe": _json_float(override.get("candidate_long_only_sharpe")),
        "candidate_long_short_sharpe": _json_float(override.get("candidate_long_short_sharpe")),
        "max_peer_long_only_sharpe": _json_float(override.get("max_peer_long_only_sharpe")),
        "max_peer_long_short_sharpe": _json_float(override.get("max_peer_long_short_sharpe")),
        "long_only_sharpe_delta": _json_float(override.get("long_only_sharpe_delta")),
        "long_short_sharpe_delta": _json_float(override.get("long_short_sharpe_delta")),
        "override_threshold": _json_float(override.get("threshold")),
        "override_portfolio": str(override.get("override_portfolio") or ""),
        "override_reason": str(override.get("override_reason") or ""),
        "direction_policy": str(candidate.get("direction_policy", "") or ""),
        "direction_sign": _json_float(candidate.get("direction_sign")),
        "best_layer_direction_train_locked": str(candidate.get("best_layer_direction_train_locked", "") or ""),
        "source_score_total_net": _json_float(candidate.get("score_total_net")),
        "source_feedback_score_net": _json_float(candidate.get("feedback_score_net")),
        "source_score_total_gross": _json_float(candidate.get("score_total_gross")),
        "source_feedback_score_gross": _json_float(candidate.get("feedback_score_gross")),
    }
    row.update(artifact_paths)
    return row


def _base_score_candidates(
    frame: pd.DataFrame | None,
    *,
    min_score: float,
    transaction_cost_enabled: bool,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    if "factor" not in work.columns and "alpha_name" in work.columns:
        work["factor"] = work["alpha_name"]
    if "factor" not in work.columns:
        return pd.DataFrame()
    score = work.apply(
        lambda row: _score_value(row, transaction_cost_enabled=transaction_cost_enabled),
        axis=1,
    )
    score = pd.to_numeric(score, errors="coerce")
    return work[score >= float(min_score)].copy()


def _max_existing_correlations(
    *,
    factor: str,
    registry: pd.DataFrame,
    signal_df: pd.DataFrame | None,
    ic_df: pd.DataFrame | None,
    portfolio_pnl_df: pd.DataFrame | None,
) -> dict[str, float]:
    if registry is None or registry.empty:
        return _empty_corr_payload(has_existing=False)
    existing = registry[registry.get("status", pd.Series(dtype=str)).astype(str) == "accepted"].copy()
    if existing.empty:
        return _empty_corr_payload(has_existing=False)
    existing_factors = existing["factor"].astype(str).tolist()
    signal_corr, signal_nearest = _max_corr_from_frame(signal_df, factor, existing_factors)
    ic_corr, ic_nearest = _max_corr_from_wide_ic(ic_df, factor, existing_factors)
    long_only_corr, long_only_nearest = _max_corr_from_pnl(
        portfolio_pnl_df, factor, existing_factors, portfolio="long_only"
    )
    long_short_corr, long_short_nearest = _max_corr_from_pnl(
        portfolio_pnl_df, factor, existing_factors, portfolio="long_short"
    )
    pnl_values = [x for x in [long_only_corr, long_short_corr] if np.isfinite(x)]
    max_pnl_corr = max(pnl_values) if pnl_values else np.nan
    nearest_factor = _nearest_factor(
        [
            (signal_corr, signal_nearest),
            (ic_corr, ic_nearest),
            (long_only_corr, long_only_nearest),
            (long_short_corr, long_short_nearest),
        ]
    )
    nearest_hash = ""
    if nearest_factor and "expression_hash" in existing.columns:
        match = existing[existing["factor"].astype(str) == nearest_factor]
        if not match.empty:
            nearest_hash = str(match.iloc[0].get("expression_hash", "") or "")
    return {
        "signal_corr": signal_corr,
        "ic_corr": ic_corr,
        "long_only_corr": long_only_corr,
        "long_short_corr": long_short_corr,
        "max_signal_corr": signal_corr,
        "max_ic_corr": ic_corr,
        "max_pnl_corr": max_pnl_corr,
        "nearest_factor_id": nearest_factor,
        "nearest_expression_hash": nearest_hash,
        "_has_existing_factors": True,
    }


def _save_factor_library_series(
    *,
    paths: dict[str, Path],
    factor: str,
    signal_df: pd.DataFrame | None,
    ic_df: pd.DataFrame | None,
    portfolio_pnl_df: pd.DataFrame | None,
) -> dict[str, str]:
    out: dict[str, str] = {}
    series_dir = paths["series"]
    series_dir.mkdir(parents=True, exist_ok=True)
    for kind, frame in [
        ("signal", _factor_signal_frame(signal_df, factor)),
        ("ic", _factor_ic_frame(ic_df, factor)),
        ("pnl", _factor_pnl_frame(portfolio_pnl_df, factor)),
    ]:
        if frame.empty:
            continue
        saved = save_dataframe_artifact(frame, series_dir / f"{factor}_{kind}", index=False, preferred="parquet")
        out[f"{kind}_artifact_path"] = saved["path"]
    return out


def _factor_signal_frame(frame: pd.DataFrame | None, factor: str) -> pd.DataFrame:
    if frame is None or frame.empty or factor not in frame.columns:
        return pd.DataFrame()
    cols = [c for c in ["trade_date", "date", "znz_code", "code", factor] if c in frame.columns]
    out = frame[cols].copy()
    out[factor] = pd.to_numeric(out[factor], errors="coerce").astype("float32")
    return out


def _factor_ic_frame(frame: pd.DataFrame | None, factor: str) -> pd.DataFrame:
    col = f"{factor}_ic"
    if frame is None or frame.empty or col not in frame.columns:
        return pd.DataFrame()
    cols = [c for c in ["trade_date", col] if c in frame.columns]
    out = frame[cols].copy()
    out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out


def _factor_pnl_frame(frame: pd.DataFrame | None, factor: str) -> pd.DataFrame:
    if frame is None or frame.empty or "factor" not in frame.columns or "portfolio" not in frame.columns:
        return pd.DataFrame()
    work = frame[
        (frame["factor"].astype(str) == factor)
        & (frame.get("portfolio", "").astype(str).isin(["long_only", "long_short"]))
    ].copy()
    cols = [
        c
        for c in [
            "trade_date",
            "portfolio",
            "return",
            "return_gross",
            "return_net",
            "has_net_pnl",
        ]
        if c in work.columns
    ]
    return work[cols].copy() if cols else pd.DataFrame()


def _max_corr_from_frame(frame: pd.DataFrame | None, factor: str, existing_factors: list[str]) -> tuple[float, str]:
    if frame is None or frame.empty or factor not in frame.columns:
        return np.nan, ""
    best_value = np.nan
    best_factor = ""
    target = pd.to_numeric(frame[factor], errors="coerce")
    for other in existing_factors:
        if other not in frame.columns:
            continue
        corr = target.corr(pd.to_numeric(frame[other], errors="coerce"), method="spearman")
        if pd.notna(corr):
            value = abs(float(corr))
            if not np.isfinite(best_value) or value > best_value:
                best_value = value
                best_factor = str(other)
    return best_value, best_factor


def _max_corr_from_wide_ic(frame: pd.DataFrame | None, factor: str, existing_factors: list[str]) -> tuple[float, str]:
    if frame is None or frame.empty:
        return np.nan, ""
    value, nearest = _max_corr_from_frame(frame, f"{factor}_ic", [f"{name}_ic" for name in existing_factors])
    return value, str(nearest).removesuffix("_ic") if nearest else ""


def _max_corr_from_pnl(
    frame: pd.DataFrame | None,
    factor: str,
    existing_factors: list[str],
    *,
    portfolio: str,
) -> tuple[float, str]:
    if (
        frame is None
        or frame.empty
        or "factor" not in frame.columns
        or "trade_date" not in frame.columns
        or "portfolio" not in frame.columns
    ):
        return np.nan, ""
    part = frame[frame.get("portfolio", "").astype(str) == str(portfolio)].copy()
    if part.empty:
        return np.nan, ""
    if portfolio == "long_only" and "return_net" in part.columns and "has_net_pnl" in part.columns:
        part = part.copy()
        net_mask = part["has_net_pnl"].astype(bool)
        part["_library_return"] = pd.to_numeric(part.get("return", np.nan), errors="coerce")
        part.loc[net_mask, "_library_return"] = pd.to_numeric(part.loc[net_mask, "return_net"], errors="coerce")
        return_col = "_library_return"
    else:
        return_col = "return_gross" if "return_gross" in part.columns else "return"
    if return_col not in part.columns:
        return np.nan, ""
    wide = part.pivot_table(index="trade_date", columns="factor", values=return_col, aggfunc="mean")
    if factor not in wide.columns:
        return np.nan, ""
    return _max_corr_from_frame(wide, factor, existing_factors)


def _score_value(row: pd.Series, *, transaction_cost_enabled: bool = True) -> float:
    if transaction_cost_enabled:
        for key in ["feedback_score_net", "score_total_net"]:
            value = row.get(key, np.nan)
            try:
                out = float(value)
            except Exception:
                out = np.nan
            if np.isfinite(out):
                return out
    value = row.get("feedback_score", row.get("score_total", np.nan))
    try:
        return float(value)
    except Exception:
        return np.nan


def _score_basis(row: pd.Series, *, transaction_cost_enabled: bool) -> str:
    if transaction_cost_enabled:
        for key in ["feedback_score_net", "score_total_net"]:
            try:
                if np.isfinite(float(row.get(key, np.nan))):
                    return "net"
            except Exception:
                continue
    return str(row.get("score_total_basis", "") or "effective")


def _library_reject_reasons(*, score: float, corr: dict[str, float], cfg: FactorLibraryConfig) -> list[str]:
    reasons: list[str] = []
    if not np.isfinite(score) or score < float(cfg.min_score):
        reasons.append("score_below_min")
    for key, reason, threshold in [
        ("signal_corr", "signal_corr", cfg.max_signal_corr),
        ("ic_corr", "ic_corr", cfg.max_ic_corr),
        ("max_pnl_corr", "pnl_corr", cfg.max_pnl_corr),
    ]:
        value = corr.get(key, np.nan)
        if np.isfinite(value) and float(value) >= float(threshold):
            reasons.append(reason)
    return reasons


def _library_status(*, score: float, corr: dict[str, float], cfg: FactorLibraryConfig) -> str:
    hard_reasons = _library_reject_reasons(score=score, corr=corr, cfg=cfg)
    if not hard_reasons:
        return "accepted"
    if np.isfinite(score) and score >= float(cfg.staging_min_score):
        numeric_corrs = []
        for value in corr.values():
            try:
                numeric = float(value)
            except Exception:
                continue
            if np.isfinite(numeric):
                numeric_corrs.append(numeric)
        max_corr = max(numeric_corrs or [0.0])
        if max_corr < float(cfg.staging_max_corr):
            return "staging"
    return "rejected"


def _empty_corr_payload(*, has_existing: bool) -> dict[str, Any]:
    return {
        "signal_corr": np.nan,
        "ic_corr": np.nan,
        "long_only_corr": np.nan,
        "long_short_corr": np.nan,
        "max_signal_corr": np.nan,
        "max_ic_corr": np.nan,
        "max_pnl_corr": np.nan,
        "max_any_corr": np.nan,
        "nearest_factor_id": "",
        "nearest_expression_hash": "",
        "_has_existing_factors": bool(has_existing),
    }


def _nearest_factor(values: list[tuple[float, str]]) -> str:
    best_value = np.nan
    best_factor = ""
    for value, factor in values:
        if not factor or not np.isfinite(value):
            continue
        if not np.isfinite(best_value) or float(value) > best_value:
            best_value = float(value)
            best_factor = str(factor)
    return best_factor


def _missing_corr_reason(corr: dict[str, Any]) -> str:
    missing = []
    for key, label in [
        ("signal_corr", "missing_signal_corr"),
        ("ic_corr", "missing_ic_corr"),
        ("long_only_corr", "missing_long_only_corr"),
        ("long_short_corr", "missing_long_short_corr"),
    ]:
        try:
            value = float(corr.get(key, np.nan))
        except Exception:
            value = np.nan
        if not np.isfinite(value):
            missing.append(label)
    return ";".join(missing)


def _library_paths(*, base_dir: str | Path, universe_name: str) -> dict[str, Path]:
    root = get_universe_paths(base_dir=base_dir, universe_name=universe_name)["root"] / "library"
    return {
        "root": root,
        "registry": root / "factor_library_registry.csv",
        "series": root / "series",
    }


# ---------------------------------------------------------------------------
# Factor Health Score
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorHealthConfig:
    """因子健康度配置。"""

    lookback_months: int = 6
    min_ic_threshold: float = 0.02
    decay_warning_threshold: float = 0.5


@dataclass
class FactorHealthReport:
    """因子健康度报告。"""

    factor: str
    health_score: float
    recent_ic_mean: float
    historical_ic_mean: float
    ic_trend: str
    status: str
    recommendation: str


def compute_factor_health(
    factor: str,
    ic_series: pd.Series,
    config: FactorHealthConfig | None = None,
) -> FactorHealthReport:
    """计算单个因子的健康度。

    健康度 = (近期 IC 均值 / 历史 IC 均值) * 100，裁剪到 [0, 100]。
    """
    cfg = config or FactorHealthConfig()

    if ic_series.empty:
        return FactorHealthReport(
            factor=factor,
            health_score=0.0,
            recent_ic_mean=0.0,
            historical_ic_mean=0.0,
            ic_trend="unknown",
            status="critical",
            recommendation="review",
        )

    ic_sorted = ic_series.sort_index()
    n = len(ic_sorted)
    lookback = min(n, cfg.lookback_months * 21)
    recent = ic_sorted.tail(lookback)
    historical = ic_sorted.head(max(1, n - lookback))

    recent_ic = float(recent.mean()) if not recent.empty else 0.0
    historical_ic = float(historical.mean()) if not historical.empty else recent_ic

    if abs(historical_ic) < 1e-6:
        ratio = 1.0 if recent_ic > 0 else 0.0
    else:
        ratio = recent_ic / historical_ic

    if ratio >= 1.0:
        trend = "improving"
    elif ratio >= cfg.decay_warning_threshold:
        trend = "stable"
    else:
        trend = "declining"

    health_score = max(0.0, min(100.0, ratio * 100.0))

    if health_score >= 70:
        status, recommendation = "healthy", "keep"
    elif health_score >= 40:
        status, recommendation = "warning", "review"
    else:
        status, recommendation = "critical", "consider_remove"

    return FactorHealthReport(
        factor=factor,
        health_score=health_score,
        recent_ic_mean=recent_ic,
        historical_ic_mean=historical_ic,
        ic_trend=trend,
        status=status,
        recommendation=recommendation,
    )


def compute_scoreboard_health(
    scoreboard_df: pd.DataFrame,
    config: FactorHealthConfig | None = None,
) -> pd.DataFrame:
    """为 scoreboard 中每个因子计算健康度，返回含 health_score/ic_trend/health_status 列的 DataFrame。

    简化版：用 |ic_mean| / ic_std 作为信号强度（ICIR 的绝对值形式）。
    """
    if scoreboard_df.empty:
        return scoreboard_df

    work = scoreboard_df.copy()
    health_scores: list[float] = []
    ic_trends: list[str] = []
    health_statuses: list[str] = []

    for _, row in work.iterrows():
        ic_mean = _safe_float(row.get("ic_mean", 0.0))
        ic_std = _safe_float(row.get("ic_std", 0.0))

        # 信号强度 = |IC均值| / IC标准差（类似 ICIR）
        if ic_std > 1e-6:
            signal_strength = abs(ic_mean) / ic_std
        else:
            signal_strength = abs(ic_mean) * 10.0  # 无波动时给高分

        health_score = max(0.0, min(100.0, signal_strength * 100.0))

        # 趋势判断：用 ic_mean 的符号和大小简化判断
        if ic_mean > 0.02:
            ic_trend = "improving"
        elif ic_mean > 0.0:
            ic_trend = "stable"
        else:
            ic_trend = "declining"

        if health_score >= 70:
            health_status = "healthy"
        elif health_score >= 40:
            health_status = "warning"
        else:
            health_status = "critical"

        health_scores.append(health_score)
        ic_trends.append(ic_trend)
        health_statuses.append(health_status)

    work["health_score"] = health_scores
    work["ic_trend"] = ic_trends
    work["health_status"] = health_statuses
    return work


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if np.isfinite(out) else 0.0
