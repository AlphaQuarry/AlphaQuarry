from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from alpha_mining.live.artifacts import live_paths, write_latest_index
from alpha_mining.live.config import LiveConfig
from alpha_mining.live.jobs import write_sa_job
from alpha_mining.live.retention import apply_retention


def test_failed_job_does_not_overwrite_successful_holdings_latest(
    tmp_path: Path,
) -> None:
    cfg = LiveConfig(universe="u1", store_root=tmp_path)
    paths = live_paths(cfg.store_root, cfg.universe)
    holdings_dir = paths.holdings_dir("superalpha_demo")
    holdings_dir.mkdir(parents=True, exist_ok=True)
    holdings_path = holdings_dir / "2026-05-22.parquet"
    pd.DataFrame({"code": ["000001.SZ"], "target_weight": [0.1]}).to_parquet(holdings_path, index=False)
    (holdings_dir / "latest.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "artifact_path": str(holdings_path.as_posix()),
                "execute_date": "2026-05-25",
            }
        ),
        encoding="utf-8",
    )

    write_sa_job(
        config=cfg,
        superalpha_id="superalpha_demo",
        job={"job_id": "job_fail", "status": "failed", "error": "boom"},
        update_success_latest=False,
    )
    write_latest_index(
        config=cfg,
        sa_statuses=[
            {
                "superalpha_id": "superalpha_demo",
                "status": "failed_today",
                "stale": True,
            }
        ],
    )

    latest = json.loads((holdings_dir / "latest.json").read_text(encoding="utf-8"))
    global_latest = json.loads((paths.live_root / "latest.json").read_text(encoding="utf-8"))
    assert latest["artifact_path"] == str(holdings_path.as_posix())
    assert global_latest["superalphas"][0]["stale"] is True


def test_retention_keeps_latest_referenced_artifact(tmp_path: Path) -> None:
    cfg = LiveConfig(universe="u1", store_root=tmp_path)
    cfg.retention.keep_daily_artifacts_days = 0
    paths = live_paths(cfg.store_root, cfg.universe)
    holdings_dir = paths.holdings_dir("superalpha_demo")
    holdings_dir.mkdir(parents=True, exist_ok=True)
    keep = holdings_dir / "2026-05-22.parquet"
    old = holdings_dir / "2026-05-21.parquet"
    pd.DataFrame({"code": ["A"]}).to_parquet(keep, index=False)
    pd.DataFrame({"code": ["B"]}).to_parquet(old, index=False)
    (holdings_dir / "latest.json").write_text(json.dumps({"artifact_path": str(keep.as_posix())}), encoding="utf-8")

    summary = apply_retention(config=cfg, now_date="2026-05-25")

    assert keep.exists()
    assert summary["deleted_files"] >= 1
    assert not old.exists()


def test_retention_keeps_latest_referenced_orders_csv_and_json(tmp_path: Path) -> None:
    cfg = LiveConfig(universe="u1", store_root=tmp_path)
    cfg.retention.keep_daily_artifacts_days = 0
    paths = live_paths(cfg.store_root, cfg.universe)
    orders_dir = paths.live_root / "orders" / "superalpha_demo"
    orders_dir.mkdir(parents=True, exist_ok=True)
    keep_csv = orders_dir / "2026-05-22.csv"
    keep_json = orders_dir / "2026-05-22.summary.json"
    old_csv = orders_dir / "2026-05-21.csv"
    keep_csv.write_text("code,order_shares\n000001.SZ,100\n", encoding="utf-8")
    keep_json.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    old_csv.write_text("code,order_shares\n000002.SZ,100\n", encoding="utf-8")
    (orders_dir / "latest.json").write_text(
        json.dumps(
            {
                "orders_csv_path": str(keep_csv.as_posix()),
                "summary_path": str(keep_json.as_posix()),
            }
        ),
        encoding="utf-8",
    )

    summary = apply_retention(config=cfg, now_date="2026-05-25")

    assert keep_csv.exists()
    assert keep_json.exists()
    assert not old_csv.exists()
    assert summary["kept_latest_refs"] >= 2


def test_retention_deletes_old_job_json_using_payload_timestamp(tmp_path: Path) -> None:
    cfg = LiveConfig(universe="u1", store_root=tmp_path)
    cfg.retention.keep_daily_artifacts_days = 180
    cfg.retention.keep_failed_jobs_days = 3
    paths = live_paths(cfg.store_root, cfg.universe)
    jobs_dir = paths.jobs_dir("superalpha_demo")
    jobs_dir.mkdir(parents=True, exist_ok=True)
    old_failed = jobs_dir / "job_failed_old.json"
    recent_failed = jobs_dir / "job_failed_recent.json"
    latest = jobs_dir / "latest.json"
    old_failed.write_text(
        json.dumps({"status": "failed", "updated_at_utc": "2026-05-01T00:00:00+00:00"}),
        encoding="utf-8",
    )
    recent_failed.write_text(
        json.dumps({"status": "failed", "updated_at_utc": "2026-05-24T00:00:00+00:00"}),
        encoding="utf-8",
    )
    latest.write_text(
        json.dumps({"status": "failed", "artifact_path": str(recent_failed.as_posix())}),
        encoding="utf-8",
    )

    summary = apply_retention(config=cfg, now_date="2026-05-27")

    assert not old_failed.exists()
    assert recent_failed.exists()
    assert summary["scanned"] >= 2
    assert summary["deleted_files"] >= 1
    assert summary["skipped_latest"] >= 1
