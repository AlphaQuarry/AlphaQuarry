from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from alpha_mining.workflow.visualization_artifacts import (
    MANIFEST_COLUMNS,
    attach_visualization_manifest_to_analysis_meta,
    save_factor_visualization_artifacts,
)

pytest.importorskip("matplotlib", reason="visualization tests require matplotlib")


def test_save_factor_visualization_artifacts_writes_manifest_and_pngs(
    tmp_path: Path,
) -> None:
    analysis_dir = tmp_path / "analysis_alpha00001_l10_ts1"
    analysis_dir.mkdir()
    trade_dates = pd.date_range("2025-01-01", periods=6, freq="D")

    manifest = save_factor_visualization_artifacts(
        analysis_dir=analysis_dir,
        factor_cols=["alpha00001"],
        df_step2=pd.DataFrame({"alpha00001": [0.1, 0.2, None, 0.4, 0.5, 0.6]}),
        ic_df=pd.DataFrame(
            {
                "trade_date": trade_dates,
                "alpha00001_ic": [0.01, 0.02, -0.01, 0.03, 0.02, 0.01],
            }
        ),
        summary_df=pd.DataFrame({"factor": ["alpha00001"], "ic_mean": [0.013], "ir": [0.5]}),
        lag_analysis_results=[
            {
                "factor": "alpha00001",
                "lag_ic_values": [0.03, 0.02, 0.01],
                "half_life": 1,
            }
        ],
        layer_results={
            "alpha00001": pd.DataFrame(
                {
                    "trade_date": list(trade_dates) * 2,
                    "layer": [1] * 6 + [10] * 6,
                    "return": [0.001, 0.002, -0.001, 0.003, 0.001, 0.002] * 2,
                }
            )
        },
    )

    manifest_path = analysis_dir / "visualization_manifest.csv"
    assert manifest_path.exists()
    assert manifest.columns.tolist() == MANIFEST_COLUMNS
    assert set(manifest["category"]) == {"distribution", "ic", "layer"}
    assert set(manifest["plot_id"]) == {
        "alpha00001__distribution",
        "alpha00001__ic_overview",
        "alpha00001__ic_decay",
        "alpha00001__yearly_ic",
        "alpha00001__layer_terminal",
    }
    assert all(not Path(path).is_absolute() for path in manifest["relative_path"])
    for relative_path in manifest["relative_path"]:
        image_path = analysis_dir / str(relative_path)
        assert image_path.exists()
        assert image_path.suffix == ".png"
        assert image_path.stat().st_size > 0


def test_save_factor_visualization_artifacts_skips_missing_inputs(
    tmp_path: Path,
) -> None:
    manifest = save_factor_visualization_artifacts(
        analysis_dir=tmp_path,
        factor_cols=["alpha00001"],
        df_step2=pd.DataFrame({"alpha00001": [1.0, 2.0, 3.0]}),
        ic_df=pd.DataFrame(),
        summary_df=pd.DataFrame(),
        lag_analysis_results=None,
        layer_results={},
    )

    assert manifest["plot_id"].tolist() == ["alpha00001__distribution"]
    assert (tmp_path / "visualization_manifest.csv").exists()


def test_attach_visualization_manifest_to_analysis_meta(tmp_path: Path) -> None:
    analysis_dir = tmp_path / "analysis_alpha00001_l10_ts1"
    analysis_dir.mkdir()
    meta_path = analysis_dir / "analysis_meta.json"
    manifest_path = analysis_dir / "visualization_manifest.csv"
    manifest_path.write_text(",".join(MANIFEST_COLUMNS) + "\n", encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "analysis_run_id": "run1",
                "table_paths": {"portfolio_pnl_df": "pnl.parquet"},
            }
        ),
        encoding="utf-8",
    )

    updated = attach_visualization_manifest_to_analysis_meta(meta_path, manifest_path)

    assert updated["table_paths"]["portfolio_pnl_df"] == "pnl.parquet"
    assert updated["table_paths"]["visualization_manifest"] == manifest_path.as_posix()
    stored = json.loads(meta_path.read_text(encoding="utf-8"))
    assert stored["table_paths"]["visualization_manifest"] == manifest_path.as_posix()
