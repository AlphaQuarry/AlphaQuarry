from __future__ import annotations

import pandas as pd

from alpha_mining.workflow.closed_loop import (
    ClosedLoopConfig,
    _save_or_update_input_manifest,
)


def test_closed_loop_manifest_records_new_schema_entries(tmp_path) -> None:  # type: ignore[no-untyped-def]
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-04-21")],
            "code": ["000001.SZ"],
            "close": [1.0],
            "pct_chg": [0.0],
            "circ_mv": [1.0],
        }
    )
    cfg = ClosedLoopConfig(
        universe_base_dir=str(tmp_path),
        universe_name="u1",
        moneyflow_source="moneyflow",
    )

    manifest = _save_or_update_input_manifest(df, cfg)

    assert manifest["field_preprocessing_config"]["enabled"] is True
    assert manifest["moneyflow_source"] == "moneyflow"
    assert "simulation_config" in manifest
    assert "operator_registry" in manifest
    assert "signature_registry" in manifest
