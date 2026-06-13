from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from dashboard.api.app import create_app


def _write_superalpha(root: Path) -> None:
    run_dir = root / "u1" / "superalphas" / "superalpha_demo"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "superalpha_id": "superalpha_demo",
                "universe": "u1",
                "name": "Demo SA",
                "combo_expression": "1",
                "component_count": 1,
                "components": [{"factor": "alpha_a", "expression": "close", "weight": 1.0}],
                "component_normalization": "none",
                "final_normalization": "none",
                "direction_adjustment": True,
                "period": 1,
                "layers": 10,
                "summary": {"score_total": 60},
            }
        ),
        encoding="utf-8",
    )


def test_live_api_empty_and_activation_are_stable(tmp_path: Path) -> None:
    _write_superalpha(tmp_path)
    client = TestClient(create_app(store_root=tmp_path))

    empty = client.get("/api/live/status", params={"universe": "u1"})
    assert empty.status_code == 200
    assert empty.json()["status"] in {"missing", "ok"}
    assert "datasource.local.yaml" not in json.dumps(empty.json(), ensure_ascii=False)

    activate = client.post("/api/live/superalphas/active", json={"universe": "u1", "superalpha_id": "superalpha_demo"})
    assert activate.status_code == 200
    assert activate.json()["record"]["snapshot"]["component_factor_ids"] == ["alpha_a"]
    assert "datasource.local.yaml" not in json.dumps(activate.json(), ensure_ascii=False)

    active = client.get("/api/live/superalphas/active", params={"universe": "u1"})
    assert active.status_code == 200
    assert active.json()["total"] == 1
    assert "datasource.local.yaml" not in json.dumps(active.json(), ensure_ascii=False)


def test_live_api_returns_holdings_and_failed_job_state(tmp_path: Path) -> None:
    live = tmp_path / "u1" / "live"
    holdings_dir = live / "holdings" / "superalpha_demo"
    jobs_dir = live / "jobs" / "superalpha_demo"
    holdings_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    holdings_path = holdings_dir / "2026-05-25.parquet"
    pd.DataFrame({"code": ["000001.SZ"], "target_weight": [0.1], "block_reason": [""]}).to_parquet(holdings_path, index=False)
    (holdings_dir / "latest.json").write_text(json.dumps({"artifact_path": str(holdings_path.as_posix())}), encoding="utf-8")
    (jobs_dir / "latest.json").write_text(json.dumps({"status": "failed", "error": "boom"}), encoding="utf-8")
    (live / "latest.json").write_text(
        json.dumps({"status": "ok", "superalphas": [{"superalpha_id": "superalpha_demo", "status": "failed_today", "stale": True}]}),
        encoding="utf-8",
    )
    client = TestClient(create_app(store_root=tmp_path))

    status = client.get("/api/live/status", params={"universe": "u1"}).json()
    holdings = client.get("/api/live/holdings", params={"universe": "u1", "superalpha_id": "superalpha_demo"}).json()

    assert status["superalphas"][0]["stale"] is True
    assert holdings["status"] == "ok"
    assert holdings["rows"][0]["code"] == "000001.SZ"


def test_live_api_returns_orders_summary_and_rows(tmp_path: Path) -> None:
    live = tmp_path / "u1" / "live"
    orders_dir = live / "orders" / "superalpha_demo"
    orders_dir.mkdir(parents=True, exist_ok=True)
    orders_path = orders_dir / "2026-05-26.csv"
    pd.DataFrame(
        {
            "code": ["000001.SZ"],
            "side": ["BUY"],
            "order_shares": [100],
            "order_value": [1000.0],
        }
    ).to_csv(orders_path, index=False)
    (orders_dir / "latest.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "orders_csv_path": str(orders_path.as_posix()),
                "account": {"account_total_value": 100000.0, "cash": 20000.0, "position_date": "2026-05-25"},
                "summary": {"estimated_turnover": 1000.0, "estimated_fee": 5.0, "orders_reviewable": True},
            }
        ),
        encoding="utf-8",
    )
    client = TestClient(create_app(store_root=tmp_path))

    orders = client.get("/api/live/orders", params={"universe": "u1", "superalpha_id": "superalpha_demo"}).json()

    assert orders["status"] == "ok"
    assert orders["summary"]["orders_reviewable"] is True
    assert orders["account"]["cash"] == 20000.0
    assert orders["rows"][0]["side"] == "BUY"
