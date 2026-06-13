from __future__ import annotations

import json
from pathlib import Path

from alpha_mining.live.registry import activate_superalpha, list_live_superalphas


def _write_superalpha_meta(root: Path, universe: str, superalpha_id: str = "superalpha_demo") -> Path:
    run_dir = root / universe / "superalphas" / superalpha_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": 2,
        "superalpha_id": superalpha_id,
        "universe": universe,
        "combo_expression": "1",
        "component_count": 1,
        "component_normalization": "cs_zscore",
        "final_normalization": "cs_zscore",
        "direction_adjustment": True,
        "period": 1,
        "layers": 10,
        "summary": {"score_total": 61.5},
        "components": [
            {
                "factor": "alpha_a",
                "expression": "ts_mean(close, 5)",
                "weight": 1.0,
                "direction_sign": -1,
                "direction_status": "registry",
            }
        ],
    }
    meta_path = run_dir / "meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return meta_path


def test_activate_superalpha_writes_frozen_snapshot_and_is_idempotent(
    tmp_path: Path,
) -> None:
    meta_path = _write_superalpha_meta(tmp_path, "u1")

    first = activate_superalpha(
        base_dir=tmp_path,
        universe="u1",
        superalpha_id="superalpha_demo",
        activated_by="test",
    )
    second = activate_superalpha(
        base_dir=tmp_path,
        universe="u1",
        superalpha_id="superalpha_demo",
        activated_by="test",
    )

    assert first["status"] == "active"
    assert second["active_count"] == 1
    rows = list_live_superalphas(base_dir=tmp_path, universe="u1", include_paused=True, include_retired=True)
    assert len(rows) == 1
    row = rows[0]
    snap = row["snapshot"]
    assert snap["superalpha_id"] == "superalpha_demo"
    assert snap["source_meta_path"].endswith("superalpha_demo/meta.json")
    assert snap["source_meta_mtime"] == meta_path.stat().st_mtime
    assert snap["source_meta_hash"]
    assert snap["combo_expression"] == "1"
    assert snap["component_factor_ids"] == ["alpha_a"]
    assert snap["component_expressions"] == ["ts_mean(close, 5)"]
    assert snap["component_weights"] == [1.0]
    assert snap["direction_signs"] == [-1.0]
    assert snap["direction_sources"] == ["registry"]
    assert snap["summary_metrics"]["score_total"] == 61.5


def test_snapshot_survives_source_meta_changes(tmp_path: Path) -> None:
    meta_path = _write_superalpha_meta(tmp_path, "u1")
    activate_superalpha(base_dir=tmp_path, universe="u1", superalpha_id="superalpha_demo")
    meta_path.unlink()

    rows = list_live_superalphas(base_dir=tmp_path, universe="u1")

    assert rows[0]["snapshot"]["component_expressions"] == ["ts_mean(close, 5)"]
    assert rows[0]["source_meta_exists"] is False


def test_activate_superalpha_enforces_max_active_and_ignores_paused(
    tmp_path: Path,
) -> None:
    _write_superalpha_meta(tmp_path, "u1", "sa1")
    _write_superalpha_meta(tmp_path, "u1", "sa2")
    _write_superalpha_meta(tmp_path, "u1", "sa3")

    activate_superalpha(base_dir=tmp_path, universe="u1", superalpha_id="sa1", max_active=2)
    activate_superalpha(base_dir=tmp_path, universe="u1", superalpha_id="sa2", max_active=2)
    try:
        activate_superalpha(base_dir=tmp_path, universe="u1", superalpha_id="sa3", max_active=2)
        raised = False
    except ValueError as exc:
        raised = "max_active" in str(exc)
    assert raised is True

    from alpha_mining.live.registry import update_superalpha_status

    update_superalpha_status(base_dir=tmp_path, universe="u1", superalpha_id="sa2", status="paused")
    result = activate_superalpha(base_dir=tmp_path, universe="u1", superalpha_id="sa3", max_active=2)

    assert result["status"] == "active"
